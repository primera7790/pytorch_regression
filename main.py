import os
import yaml

import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import v2

from src.data_creating import data_creating
from src.dataset_reg_class import DatasetReg


config_file_name = 'params_all.yaml'
config_path = os.path.join('config', config_file_name)
config = yaml.safe_load(open(config_path))

if not os.path.isdir('dataset'):
    os.mkdir('dataset')
    data_creating(config)

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=(0.5, ), std=(0.5, ))
])

dataset = DatasetReg(config['data_path'], transform=transform)
train_set, val_set, test_set = random_split(dataset, config['random_split'].values())

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)