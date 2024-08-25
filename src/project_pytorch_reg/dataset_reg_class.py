import os
import json
import torch
from torch.utils.data import Dataset

from PIL import Image
import matplotlib.pyplot as plt


class DatasetReg(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.names_list = os.listdir(path)
        if 'coords.json' in self.names_list:
            self.names_list.remove('coords.json')

            with open(os.path.join(path, 'coords.json'), 'r') as file:
                self.coords_dict = json.load(file)

        self.length = len(self.names_list)

    def __len__(self):
        return self.length

    def get_raw_data(self, index):
        file_name = self.names_list[index]

        img_path = os.path.join(self.path, file_name)
        img = Image.open(img_path)

        coords = self.coords_dict[file_name]

        return img, coords

    def __getitem__(self, index: int):
        img, coords = self.get_raw_data(index)

        if self.transform:
            img = self.transform(img)
            coords = torch.tensor(coords, dtype=torch.float32)

        return img, coords

    def show_image(self, index: int):
        img, coords = self.get_raw_data(index)

        plt.title(f'Image {index}')
        plt.scatter(coords[1], coords[0], marker='o', color='red')
        plt.imshow(img, cmap='gray')
        plt.show()
