"""
train a pre model
"""


from random import shuffle

import numpy as np
import os

import tensorflow as tf
from PIL import Image
from tensorflow.python.framework.errors_impl import NotFoundError


def gen_data(prefix_path):
    X = []
    Y = []
    files = os.listdir(prefix_path)
    shuffle(files)
    for file in files:
        img = Image.open(prefix_path + file)
        x = np.array(img)
        x = x.reshape(x.shape[0], x.shape[1], 1)
        X.append(x)
        Y.append(int(file.split("_")[0]))
    X = np.array(X, dtype=np.float32)
    X /= 255
    Y = np.array(Y, dtype=np.uint8)
    return X, Y


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

model.compile(
    optimizer=tf.train.GradientDescentOptimizer(0.05),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

X, Y = gen_data("img/number/1/")
try:
    model.load_weights("train/save/num")
    print("load successfully")
except NotFoundError:
    print("load fail, training...")
    model.fit(X, Y, epochs=32, use_multiprocessing=True)
    model.save_weights("train/save/num")

X, Y = gen_data("img/number/2/")
loss, acc = model.evaluate(X, Y)
print(f"loss: {loss:2f}, acc: {acc:2f}")

X, Y = gen_data("img/number/3/")
y = model.predict(X)

# for i_y, i in enumerate(y):
#     m = 0
#     m_index = 0
#     for index, j in enumerate(i):
#         if m < j:
#             m = j
#             m_index = index
#     print(f"{m_index}: {m:.2f}, answer is {Y[i_y]}")
