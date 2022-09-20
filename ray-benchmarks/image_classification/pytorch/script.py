#!/usr/bin/env python

import torchvision
from torchvision import transforms
import torch

#model = torchvision.models.resnet50()
#print(model)

# Amog says to create a dataset with the map function, then pass that dataset to the dataloader.
# This is the UDF
# https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/python/dataset.py#L205-L218
# Will want the thing to run in a Ray cluster.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
])

train_dataset = torchvision.datasets.ImageFolder(
    root='/workspace/dev/data/balajis-tiny-imagenet/train',
    transform=transform
)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

model = torchvision.models.resnet50(weights=None)

for data, label in dataloader:
    data, label = data.to('cuda'), label.to('cuda')
    print(f'{data.nelement() * data.element_size() / (1 << 20):.02f} MB')

    break


