import pandas as pd
import cv2
import tensorflow as tf
import numpy as np
from sklearn import model_selection
import keras
import os

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20

IMAGE_PATH = r"C:\Users\User\Documents\DIPLOM\qa\qa010.jpg"
NUM_CLASSES = 2
PATH_TO_MODEL = r""

NUM_CLASSES_LIST = [13, 5, 6, 4, 8, 2, 2]
PATH_TO_MODEL_LIST = ['E:\\models\\dates_model.h5', 'E:\\models\\material_model_1.h5', 'E:\\models\\material_model_2.h5', 
                      'E:\\models\\material_model_3.h5', 'E:\\models\\technique_model.h5', 'E:\\models\\stamps_model.h5', 
                      'E:\\models\\casing_model.h5']

def padding(path):
    # read image
    img = cv2.imread(os.path.join(os.path.curdir, path))
    old_h, old_w, channels = img.shape

    # create new image of desired size and color (white) for padding
    new_w = max(old_h, old_w)
    new_h = max(old_h, old_w)
    color = (255,255,255)
    result = np.full((new_h,new_w, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_w - old_w) // 2
    y_center = (new_h - old_h) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_h, 
        x_center:x_center+old_w] = img

    return result

image = padding(IMAGE_PATH)
image = cv2.resize(np.array(image), (IMAGE_SIZE, IMAGE_SIZE))
image = image.astype(float)/255


def init_model(num_classes, path_to_model):
    model = keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu" ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu" ),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.load_weights(path_to_model)

    return model

# Вариант 1: прогон по конкретному классу 
# model = init_model(NUM_CLASSES, PATH_TO_MODEL)
# prediction = model.predict(image.reshape((1, 256, 256, 3)))[0]
# print(prediction)
# print(list(prediction).index(max(prediction)))

# Вариант 2: прогон по всем классам сразу (нужно указать в глобальных переменных количество классов и пути к моделям соответственно)
results = []
for num, path in list(zip(NUM_CLASSES_LIST, PATH_TO_MODEL_LIST)):
    model = init_model(num, path)
    prediction = model.predict(image.reshape((1, 256, 256, 3)))[0]
    results.append(list(prediction).index(max(prediction)))
    # print(list(prediction).index(max(prediction)))

print(results)