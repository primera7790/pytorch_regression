from fastapi import FastAPI

import os
import yaml
import torch
import random

from main import train, predict
from src.dataset_reg_class import DatasetReg
from src.data_creating import data_creating

app = FastAPI()

config_file_name = 'params_all.yaml'
config_path = os.path.join('config', config_file_name)
config = yaml.safe_load(open(config_path))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')


@app.get('/data_creating/')
def creating_data():
    data_creating(data_path=config['data_path'], **config['data_creating'])
    img_num = config['data_creating']['img_num']
    train_num = int(img_num * config['random_split']['train'])
    val_num = int(img_num * config['random_split']['val'])
    test_num = int(img_num * config['random_split']['test'])
    return (
        f'Creating data: Done \
        Number of images: {img_num} \
        Train: {train_num}\
        Validation: {val_num}\
        Test: {test_num}'
    )

@app.get('/train/')
def model_train():
    return train(config, device)


@app.get('/predict/')
def model_predict():
    return predict(config, device)


@app.get('/parameters/')
def get_params():
    return config


@app.post('/set_params/{parameter_name}')
def set_params(parameter_name: str, new_value):

    try:
        new_value = int(new_value) if float(new_value) % 1 == 0 else float(new_value)
    except:
        new_value = str(new_value)


    with open(config_path, 'w') as file:
        parameter_name_list = parameter_name.split('__')

        if len(parameter_name_list) == 1:
            config[parameter_name_list[0]] = new_value
        elif len(parameter_name_list) == 2:
            config[parameter_name_list[0]][parameter_name_list[1]] = new_value
        elif len(parameter_name_list) == 3:
            config[parameter_name_list[0]][parameter_name_list[1]][parameter_name_list[2]] = new_value

        yaml.dump(config, file)
    return config

@app.get('/images/')
def get_random_image():
    dataset = DatasetReg(config['data_path'])
    image_idx = random.randint(0,len(dataset) - 1)
    dataset.show_image(image_idx)
    return f'Image index: {image_idx}'
