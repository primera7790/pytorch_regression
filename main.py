import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import v2

import os
import yaml
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.data_creating import data_creating
from src.dataset_reg_class import DatasetReg
from src.model_class import ModelReg


def preparing_data(config):
    if not os.path.isdir('dataset'):
        os.mkdir('dataset')
        data_creating(data_path=config['data_path'], **config['data_creating'])

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=(0.5,), std=(0.5,))
    ])

    dataset = DatasetReg(config['data_path'], transform=transform)
    train_set, val_set, test_set = random_split(dataset, config['random_split'].values())

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


def visualisation(train_loss, val_loss, train_acc, val_acc):
    plt.style.use('dark_background')

    plt.subplot(1, 2, 1)
    plt.plot(train_loss[1:])
    plt.plot(val_loss[1:])
    plt.legend(['loss_train', 'loss_val'])

    plt.subplot(1, 2, 2)
    plt.plot(train_acc[1:])
    plt.plot(val_acc[1:])
    plt.legend(['acc_train', 'acc_val'])

    plt.show()


def train(config, device):
    load_to_continue = config['save_load']['load_mode']['load_to_continue'] \
        if config['save_load']['load_mode']['load_to_continue'] != 'None' else None
    load_best = config['save_load']['load_mode']['load_best'] \
        if config['save_load']['load_mode']['load_best'] != 'None' else None

    if load_to_continue is None and load_best is None:
        train_loader, val_loader, test_loader = preparing_data(config)
    else:
        train_loader, val_loader, test_loader = None, None, None

    model = ModelReg(**config['model_class']).to(device)
    # model.test(16, device)

    loss_model = nn.MSELoss()
    learning_rate = config['train']['learning_rate']
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **config['train']['lr_scheduler'])

    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    lr_list = []
    best_loss = None
    epochs_without_improve = 0

    first_epoch = 0
    EPOCHS = config['train']['epoch_num']

    if load_to_continue or load_best:
        file_to_load_name = config['save_load']['current_params_file_name'] if load_to_continue \
            else config['save_load']['best_params_file_name']
        saved_params_dir_path = config['saved_params_path']

        if not os.path.exists(os.path.join(saved_params_dir_path, file_to_load_name)):
            print('No save parameters found.')
            exit()

        param_dicts = torch.load(os.path.join(saved_params_dir_path, file_to_load_name), map_location=device)

        train_loader = param_dicts['loaders']['train']
        val_loader = param_dicts['loaders']['val']
        test_loader = param_dicts['loaders']['test']

        model.load_state_dict(param_dicts['state_dicts']['model'])
        opt.load_state_dict(param_dicts['state_dicts']['opt'])
        lr_scheduler.load_state_dict(param_dicts['state_dicts']['lr_scheduler'])

        first_epoch = param_dicts['current_hyperparameters']['epoch'] + 1
        EPOCHS = param_dicts['current_hyperparameters']['EPOCHS'] if first_epoch != param_dicts['current_hyperparameters']['EPOCHS'] \
            else config['train']['epoch_num']

        train_loss_list = param_dicts['loss_lists']['train']
        train_accuracy_list = param_dicts['loss_lists']['val']
        val_loss_list = param_dicts['accuracy_lists']['train']
        val_accuracy_list = param_dicts['accuracy_lists']['val']
        lr_list = param_dicts['lr_list']
        best_loss = param_dicts['best_loss']

    for epoch in range(first_epoch, EPOCHS):
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

            train_loop.set_description(f'Epoch [{epoch + 1}/{EPOCHS}], train_loss={mean_train_loss:.4f}')

            true_answers += (torch.round(pred) == targets).all(dim=1).sum().item()

        train_set_len = int(config['data_creating']['img_num'] * config['random_split']['train'])
        accuracy_train_current_epoch = true_answers / train_set_len

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

            val_set_len = int(config['data_creating']['img_num'] * config['random_split']['val'])
            accuracy_val_current_epoch = true_answers / val_set_len

            val_loss_list.append(mean_val_loss)
            val_accuracy_list.append(accuracy_val_current_epoch)

        lr_scheduler.step(mean_val_loss)
        lr = lr_scheduler.get_last_lr()[0]
        lr_list.append(lr)

        model_params_dict = {
            'loaders': {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            },
            'state_dicts': {
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            },
            'current_hyperparameters': {
                'epoch': epoch,
                'EPOCHS': EPOCHS,
            },
            'loss_lists': {
                'train': train_loss_list,
                'val': val_loss_list
            },
            'accuracy_lists': {
                'train': train_accuracy_list,
                'val': val_accuracy_list
            },
            'lr_list' : lr_list,
            'best_loss': best_loss
        }

        params_to_save_path = config['saved_params_path']
        if not os.path.isdir(params_to_save_path):
            os.mkdir(params_to_save_path)

        current_params_file_name = config['save_load']['current_params_file_name']
        torch.save(model_params_dict, os.path.join(params_to_save_path, current_params_file_name))

        threshold = config['save_load']['threshold']
        best_params_file_name = config['save_load']['best_params_file_name']
        if best_loss is None or mean_val_loss < best_loss - best_loss * threshold:
            print(f'Best loss updated: {best_loss} --> {mean_val_loss}')
            best_loss = mean_val_loss
            epochs_without_improve = 0
            torch.save(model_params_dict, os.path.join(params_to_save_path, best_params_file_name))

        print(f'Epoch [{epoch + 1}/{EPOCHS}], \
        train_loss={mean_train_loss:.4f}, \
        train_accuracy={accuracy_train_current_epoch:.4f}, \
        val_loss={mean_val_loss:.4f}, \
        val_accuracy={accuracy_val_current_epoch:.4f}, \
        lr={lr:.8f}'.rstrip('0'))

        if epochs_without_improve == config['train']['epochs_on_plateau_allow']:
            break

        epochs_without_improve += 1

    visualisation(train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list)


def predict(config, device):
    model = ModelReg(**config['model_class']).to(device)

    loss_model = nn.MSELoss()

    file_to_load_name = config['save_load']['best_params_file_name']
    saved_params_dir_path = config['saved_params_path']

    if not os.path.exists(os.path.join(saved_params_dir_path, file_to_load_name)):
        print('No save parameters found.')
        exit()

    param_dicts = torch.load(os.path.join(saved_params_dir_path, file_to_load_name), map_location=device)

    test_loader = param_dicts['loaders']['test']

    model.load_state_dict(param_dicts['state_dicts']['model'])

    model.eval()
    with torch.no_grad():

        loss_per_batch = []
        true_answers = 0

        test_loop = tqdm(test_loader, leave=False)
        for x, targets in test_loop:
            input_size = config['model_class']['input_size']
            x = x.reshape(-1, input_size).to(device)
            targets = targets.to(device)

            pred = model(x)
            loss = loss_model(pred, targets)

            loss_per_batch.append(loss.item())
            true_answers += (torch.round(pred) == targets).all(dim=1).sum().item()

        mean_test_loss = sum(loss_per_batch) / len(loss_per_batch)

        test_set_len = int(config['data_creating']['img_num'] * config['random_split']['test'])
        accuracy_test_current_epoch = true_answers / test_set_len

    print(f'Loss: {mean_test_loss:.4f} \nAccuracy: {accuracy_test_current_epoch:.4f}')


if __name__ == '__main__':
    config_file_name = 'params_all.yaml'
    config_path = os.path.join('config', config_file_name)
    config = yaml.safe_load(open(config_path))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    if config['train_or_predict'] == 'train':
        train(config, device)
    elif config['train_or_predict'] == 'predict':
        predict(config, device)

