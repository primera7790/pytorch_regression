import os
import json
import yaml

import numpy as np

from PIL import Image


def data_creating():
    config_file_name = 'params_all.yaml'
    config_path = os.path.join('config', config_file_name)
    config = yaml.safe_load(open(config_path))['data_creating']

    img_num, img_size, square_size, border_size = config.values()

    img = np.random.randint(0, 50, [img_num, img_size, img_size], dtype=np.uint8)
    square = np.random.randint(100, 200, [img_num, square_size, square_size], dtype=np.uint8)

    coords_dict = {}

    for i in range(img.shape[0]):

        half_square_size = square_size // 2
        axis_coords = np.random.randint(border_size + half_square_size,
                                        img_size - border_size - half_square_size, 2)

        axis_0_start_point = axis_coords[0] - half_square_size
        axis_0_end_point = axis_coords[0] + (square_size - half_square_size)

        axis_1_start_point = axis_coords[1] - half_square_size
        axis_1_end_point = axis_coords[1] + (square_size - half_square_size)

        img[i, axis_0_start_point:axis_0_end_point, axis_1_start_point:axis_1_end_point] = square[i]

        img_name = f'image_{i}.jpeg'
        img_path = os.path.join('dataset', img_name)

        image = Image.fromarray(img[i])
        image.save(img_path)

        coords_dict[img_name] = axis_coords.tolist()

    coords_json_name = 'coords.json'
    coords_json_path = os.path.join('dataset', coords_json_name)
    with open(coords_json_path, 'w') as file:
        json.dump(coords_dict, file, indent=2)

    return
