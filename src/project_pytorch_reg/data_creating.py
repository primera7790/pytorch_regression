import os
import json
import numpy as np


from PIL import Image
from tqdm import tqdm


def data_creating(data_path, img_num, img_size, square_size, border_size):

    img = np.random.randint(0, 50, [img_num, img_size, img_size], dtype=np.uint8)
    square = np.random.randint(100, 200, [img_num, square_size, square_size], dtype=np.uint8)

    coords_dict = {}
    data_creating_loop = tqdm(range(img_num))
    for i in data_creating_loop:
        half_square_size = square_size // 2
        axis_coords = np.random.randint(border_size + half_square_size,
                                        img_size - border_size - half_square_size, 2)

        axis_0_start_point = axis_coords[0] - half_square_size
        axis_0_end_point = axis_coords[0] + (square_size - half_square_size)

        axis_1_start_point = axis_coords[1] - half_square_size
        axis_1_end_point = axis_coords[1] + (square_size - half_square_size)

        img[i, axis_0_start_point:axis_0_end_point, axis_1_start_point:axis_1_end_point] = square[i]

        img_name = f'image_{i}.jpeg'
        img_path = os.path.join(data_path, img_name)

        image = Image.fromarray(img[i])
        image.save(img_path)

        coords_dict[img_name] = axis_coords.tolist()

    coords_json_name = 'coords.json'
    coords_json_path = os.path.join(data_path, coords_json_name)
    with open(coords_json_path, 'w') as file:
        json.dump(coords_dict, file, indent=2)

    return
