import os
import json

from array import array
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image


class DatasetReg(Dataset):
    def __init__(self, path, transform=None):
        root_path = Path(__file__).parent.parent
        self.path = os.path.join(root_path, path[1:])
        self.transform = transform

        self.list_name_file = os.listdir(self.path)
        if 'coords.json' in self.list_name_file:
            self.list_name_file.remove('coords.json')

        self.len_dataset = len(self.list_name_file)

        with open(os.path.join(self.path, 'coords.json'), 'r') as f:
            self.dict_coords = json.load(f)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index: int):
        name_file = self.list_name_file[index]
        path_img = os.path.join(self.path, name_file)

        img = Image.open(path_img)
        coords = torch.tensor(self.dict_coords[name_file], dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        return img, coords

    def show_img(self, index: int):
        img, coords = self[index]
        print(f'Координаты центра: x={coords[1]}\n'
              f'                   y={coords[0]}')
        plt.scatter(coords[1], coords[0], marker='o', color='red')
        plt.imshow(img, cmap='gray')
        plt.show()