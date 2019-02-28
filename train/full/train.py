"""
train a model
"""

import numpy as np
import os

import tensorflow as tf
from PIL import Image
from tensorflow.python.framework.errors_impl import NotFoundError


def gen_data(prefix_path):
    print(f"loading all jpg file in {prefix_path}")
    X = []
    Y = []
    files = os.listdir(prefix_path)
    for file in files:
        img = Image.open(prefix_path + file)
        x = np.array(img)
        x = x.reshape(x.shape[0], x.shape[1], 1)
        X.append(x)
        op = file.split("_")[0].replace(".jpg", "")

        if op[2] == "-":
            y = 0
        else:  # == "+"
            y = 1000
        y += int(op.replace("-", "").replace("+", ""))
        Y.append(y)
    X = np.array(X, dtype=np.float32)
    X /= 255
    Y = np.array(Y, dtype=np.uint8)
    return X, Y


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=3, activation=tf.nn.relu),
    tf.keras.layers.AvgPool2D(),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation=tf.nn.relu),
    tf.keras.layers.AvgPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(180, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(120, activation=tf.nn.relu),
    tf.keras.layers.Dense(1755, activation=tf.nn.softmax),
])

# about output layer, don't know if it's right
# first bit: 0 -> `-`, 1 -> `+`
# second bit: cannot be 0
# third bit: no limit
# forth bit: cannot be 0

"""
count = 0
for i in range(2000):
    if i / 100 % 10 == 0:
        print(i)
        continue
    if i % 10 == 0:
        print(i)
        continue
    if i / 10 - i % 10 <= 0:
        print(i)
        continue
    count += 1

print(count)
"""

model.compile(
    optimizer=tf.train.GradientDescentOptimizer(0.05),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

valX, valY = gen_data("img/full/no_timestamp/1/")

try:
    model.load_weights("train/save/full")
    print("load successfully")
except NotFoundError:
    print("load fail, training...")
    X, Y = gen_data("img/full/labeled/")
    tmpX, tmpY = gen_data("img/full/unchecked/")
    X = np.concatenate((X, tmpX))
    Y = np.concatenate((Y, tmpY))
    # tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='train/log', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(
        X, Y,
        epochs=24,
        use_multiprocessing=True,
        shuffle=True,
        # callbacks=[tbCallBack],
        validation_data=(valX, valY))
    model.save_weights("train/save/full")

loss, acc = model.evaluate(valX, valY)
print(f"loss: {loss:2f}, acc: {acc:2f}")
