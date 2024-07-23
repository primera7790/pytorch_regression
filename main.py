import os

from array import array

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image

from src.data_creating import get_data
from src.DatasetRegClass import DatasetReg


if not os.path.isdir('dataset'):
    os.mkdir('dataset')
    get_data()

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=(0.5, ), std=(0.5, ))
    ]
)

dataset = DatasetReg('/dataset', transform=transform)
# dataset.show_img(1278)

train_set, val_set, test_set = random_split(dataset, [0.7, 0.1, 0.2])

print(f'Размер train_set: {len(train_set)}\n'
      f'Размер val_set:   {len(val_set)}\n'
      f'Размер test_set:  {len(test_set)}')

train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)