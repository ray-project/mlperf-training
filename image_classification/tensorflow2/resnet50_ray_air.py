import tensorflow as tf
import numpy as np

import ray
from ray.air import session, Checkpoint
from ray.train.tensorflow import prepare_dataset_shard, TensorflowTrainer
from ray.air.config import ScalingConfig

DATASET_SIZE = 100
IMAGE_DIMS = (1, 224, 224, 3)
TRAIN_DATASET_SIZE_RATIO = 0.8


def build_model():
    return tf.keras.applications.resnet50.ResNet50(
        weights=None,
        #input_tensor=None,
        #input_shape=None,
        #pooling=None,
        #classes=1000,
    )


def train_loop_for_worker(config):
    dataset_shard = session.get_dataset_shard("train")
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = build_model()
        #model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
        model.compile(
            optimizer="Adam", loss="mean_squared_error", metrics=["mse"])

    def to_tf_dataset(dataset, batch_size):
        def to_tensor_iterator():
            for batch in dataset.iter_tf_batches(
                batch_size=batch_size, dtypes=tf.float32
            ):
                yield batch["image"][0], batch["label"]

        output_signature = (
            tf.TensorSpec(shape=IMAGE_DIMS, dtype=tf.uint8),
            tf.TensorSpec(shape=(None), dtype=tf.int32),
        )
        tf_dataset = tf.data.Dataset.from_generator(
            to_tensor_iterator, output_signature=output_signature
        )
        return prepare_dataset_shard(tf_dataset)

    for epoch in range(config["num_epochs"]):
        tf_dataset = to_tf_dataset(dataset=dataset_shard, batch_size=1)
        model.fit(tf_dataset)
        # You can also use ray.air.callbacks.keras.Callback
        # for reporting and checkpointing instead of reporting manually.
        session.report(
            {},
            checkpoint=Checkpoint.from_dict(
                dict(epoch=epoch, model=model.get_weights())
            ),
        )

def build_dataset():
    ds = ray.data.from_items(
            [{
                "image": np.empty(IMAGE_DIMS, dtype=np.uint8),
                "label": 1,
                } for _ in range(int(DATASET_SIZE * TRAIN_DATASET_SIZE_RATIO))]
            )
    return ds

if __name__ == "__main__":
    train_dataset = build_dataset()
    trainer = TensorflowTrainer(train_loop_for_worker,
        scaling_config=ScalingConfig(num_workers=1),
        datasets={"train": train_dataset},
        train_loop_config={"num_epochs": 2})
    result = trainer.fit()
