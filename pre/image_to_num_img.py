"""
split image to number images
"""

import random
import time

import numpy as np
import os

from PIL import Image

from common.convolution import conv_2d_default
from pre.img_to_num_img import img_to_num

src_prefix = "img/full/no_timestamp/"
dst_prefix = "img/number/1/"
jpg = ".jpg"

# convert image to number image

for file in os.listdir(src_prefix):
    print(f"processing {file}...")
    image = Image.open(src_prefix + file)
    image = conv_2d_default(np.array(image))
    # print_img(image)
    result = img_to_num(image)
    if not result:
        continue
    result_len = len(result)

    if "+" in file and result_len != 4:
        # op stick to number
        # print("more than one op")
        continue
    r = random.Random()
    for index, i in enumerate(result):
        if result_len == 4 and index == 2:
            # skip char `+`
            continue
        img = np.ones((25, 20), dtype=np.uint8)
        img *= 255
        h = i.shape[0]
        w = i.shape[1]
        if h > 25:
            h = 25
        if w > 20:
            w = 20
        img[:h, :w] = i[:h, :w]
        img = Image.fromarray(img, "L")
        rand_str = f"_{int(time.time())}_{r.randint(0, 1000)}"
        if result_len == 3 and index == 2:
            number_char = file[3]
        else:
            number_char = file[index]
        img_path = dst_prefix + number_char + rand_str + jpg
        img.save(img_path)
