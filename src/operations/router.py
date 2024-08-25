import os
import shutil
import random

import torch
import yaml
from fastapi import APIRouter

from project_pytorch_reg.project_main import train, predict, preparing_data
from project_pytorch_reg.dataset_reg_class import DatasetReg


router = APIRouter(
    prefix='/operations',
    tags=['Operations']
)


config_file_name = 'params_all.yaml'
config_path = os.path.join('project_pytorch_reg', 'config', config_file_name)
dataset_dir_path = os.path.join('project_pytorch_reg', 'dataset')

config = yaml.safe_load(open(config_path))

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@router.get('/data_creating')
def creating_data():
    if os.path.isdir(dataset_dir_path):
        shutil.rmtree(dataset_dir_path)
    preparing_data(config)

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


@router.get('/train')
def model_train():
    return train(config, device)


@router.get('/predict')
def model_predict():
    return predict(config, device)


@router.get('/image')
def get_random_image():
    dataset = DatasetReg(dataset_dir_path)
    image_idx = random.randint(0,len(dataset) - 1)
    dataset.show_image(image_idx)
    return f'Image index: {image_idx}'
