import os
import yaml

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import v2

from src.data_creating import data_creating
from src.dataset_reg_class import DatasetReg
from src.model_class import ModelReg


config_file_name = 'params_all.yaml'
config_path = os.path.join('config', config_file_name)
config = yaml.safe_load(open(config_path))

if not os.path.isdir('dataset'):
    os.mkdir('dataset')
    data_creating(data_path=config['data_path'], **config['data_creating'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

model = ModelReg(**config['model_class']).to(device)

# model.test(16, device)

loss_model = nn.MSELoss()
learning_rate = config['train']['learning_rate']
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss_list = []
train_accuracy_list = []
val_loss_list = []
val_accuracy_list = []

EPOCHS = config['train']['epoch_num']

for epoch in range(EPOCHS):
    model.train()

    loss_per_batch = []
    true_answers = 0
    mean_train_loss = 0

    train_loop = tqdm(train_loader, leave=False)
    for x, targets in train_loop:
        input_size = config['model_class']['input_size']
        x = x.reshape(-1, input_size).to(device)
        targets = targets.to(device)

        pred = model(x)
        loss = loss_model(pred, targets)

        opt.zero_grad()
        loss.backward()

        opt.step()

        loss_per_batch.append(loss.item())
        mean_train_loss = sum(loss_per_batch) / len(loss_per_batch)

        train_loop.set_description(f'Epoch [{epoch}/{EPOCHS}], train_loss={mean_train_loss:.4f}')

        true_answers += (torch.round(pred) == targets).all(dim=1).sum().item()

    accuracy_train_current_epoch = true_answers / len(train_set)

    train_loss_list.append(mean_train_loss)
    train_accuracy_list.append(accuracy_train_current_epoch)

    model.eval()
    with torch.no_grad():

        loss_per_batch = []
        true_answers = 0

        for x, targets in val_loader:
            input_size = config['model_class']['input_size']
            x = x.reshape(-1, input_size).to(device)
            targets = targets.to(device)

            pred = model(x)
            loss = loss_model(pred, targets)

            loss_per_batch.append(loss.item())
            true_answers += (torch.round(pred) == targets).all(dim=1).sum().item()

        mean_val_loss = sum(loss_per_batch) / len(loss_per_batch)
        accuracy_val_current_epoch = true_answers / len(val_set)

        val_loss_list.append(mean_val_loss)
        val_accuracy_list.append(accuracy_val_current_epoch)

    print(f'Epoch [{epoch + 1}/{EPOCHS}], \
    train_loss={mean_train_loss}, \
    train_accuracy={accuracy_train_current_epoch:.4f}, \
    val_loss={mean_val_loss}, \
    val_accuracy={accuracy_val_current_epoch:.4f}')

