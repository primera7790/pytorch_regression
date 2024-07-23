import os
import sys
import json
import struct

from os import path

import numpy as np

from PIL import Image


def get_data():
    img = np.random.randint(0, 50, [100000, 64, 64], dtype=np.uint8)
    square = np.random.randint(100, 200, [100000, 15, 15], dtype=np.uint8)

    coords = np.empty([100000, 2])

    data = {}

    for i in range(img.shape[0]):

        x = np.random.randint(20, 44)
        y = np.random.randint(20, 44)

        img[i, (y - 7):(y + 8), (x - 7):(x + 8)] = square[i]

        coords[i] = [y, x]

        name_img = f'img_{i}.jpeg'
        path_img = os.path.join('dataset/', name_img)

        image = Image.fromarray(img[i])
        image.save(path_img)

        data[name_img] = [y, x]

    with open('dataset/coords.json', 'w') as f:
        json.dump(data, f, indent=2)

    # os.rmdir('dataset/.ipynb_checkpoints')
    return
