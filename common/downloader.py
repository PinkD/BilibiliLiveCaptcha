"""
download auto marked picture
NOTE: train number model first
"""

import base64
import io
import json
import time

import numpy as np
import urllib.request

from PIL import Image

from common import cookie
from common.convolution import conv_2d_default
from train.number.predict import parse_op

url = "https://api.live.bilibili.com/lottery/v1/SilverBox/getCaptcha"

opener = urllib.request.build_opener()
opener.addheaders.clear()
opener.addheaders.append((
    "Cookie",
    cookie
))
# opener.addheaders.append((
#     "Cookie",
#     "CURRENT_FNVAL=16"
# ))
opener.addheaders.append((
    "User-Agent",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36"
))

while True:
    data = opener.open(url).read().decode()
    data = json.loads(data)
    if data["code"] == 0:
        # with open("tmp.jpg", "wb") as f:
        #     f.write(base64.b64decode(data["data"]["img"][23:]))
        # image = Image.open("tmp.jpg")
        image = Image.open(io.BytesIO(base64.b64decode(data["data"]["img"][23:])))
        image = image.convert('L', (0.2989, 0.5870, 0.1140, 0))
        img = conv_2d_default(np.array(image))
        op = parse_op(img)
        if op:
            image.save(f"img/test/{op}_{int(time.time())}.jpg")

            print(f"file `{op}_{int(time.time())}.jpg` saved")
        else:
            print("predict fail")
    else:
        print(data)
    time.sleep(0.3)

# manual input
# while True:
#     data = opener.open(url).read().decode()
#     data = json.loads(data)
#     if data["code"] == 0:
#         with open("tmp.jpg", "wb") as f:
#             f.write(base64.b64decode(data["data"]["img"][23:]))
#         image = Image.open("tmp.jpg")
#         image = image.convert('L', (0.2989, 0.5870, 0.1140, 0))
#         asciify.print_img(conv_2d_default(np.array(image)))
#         op = input("input operation:")
#         image.save(f"img/{op}.jpg")
#         print(f"file `{op}.jpg` saved")
#     else:
#         print(data)
