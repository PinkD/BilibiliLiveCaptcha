"""
predict with pre model
NOTE: need to train pre model first
"""

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from pre.img_to_num_img import img_to_num

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=3, kernel_size=3, activation=tf.nn.relu),
    tf.keras.layers.AvgPool2D(),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.relu),
    tf.keras.layers.AvgPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(84, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])
model.load_weights("train/save/num")


def parse_op(image):
    result = img_to_num(image)
    if not result:
        return None
    result_len = len(result)
    name = ""
    for index, i in enumerate(result):
        if result_len == 4 and index == 2:
            name += "+"
            continue
        img = np.ones((25, 20), dtype=np.uint8)
        img *= 255
        h = i.shape[0]
        w = i.shape[1]
        if h > 25 or w > 20:
            return None
        img[:h, :w] = i
        if result_len == 3 and index == 2:
            name += "-"
        name += predict(img)
    return name


def predict(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)

    X = np.array([img], dtype=np.float32)
    X /= 255
    result = model.predict(X)
    m = 0
    m_index = 0
    for index, j in enumerate(result[0]):
        if m < j:
            m = j
            m_index = index
    return str(m_index)
