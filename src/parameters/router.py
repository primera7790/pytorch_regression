import os

import torch
import yaml
from fastapi import APIRouter


router = APIRouter(
    prefix='/parameters',
    tags=['Parameters']
)


config_file_name = 'params_all.yaml'
config_path = os.path.join('project_pytorch_reg', 'config', config_file_name)
dataset_dir_path = os.path.join('project_pytorch_reg', 'dataset')

config = yaml.safe_load(open(config_path))

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@router.get('/get_params')
def get_params():
    return config


@router.post('/set_params/{parameter_name}')
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