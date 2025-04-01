import pandas as pd
import cv2
import tensorflow as tf
import numpy as np
import keras
import os
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# GLOBAL VARIABLES & DATASET LOAD
IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 30

df = pd.read_csv("markdown_3.csv",
    names=["filename", "dates", "material_1", "material_2",
           "material_3", "technique", "stamps", "casing"], dtype={'casing': bool})
df = df.replace(np.nan, None)

filename = df['filename'].to_list() # make list of all file paths to images

# CHANGING IMAGE SIZE TO SQUARE
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

# PREPROCESSING TRAINING DATA
pic_matrix = [] # will contain numerical representation of images

for i in filename:
    image = padding(i)
    image = cv2.resize(np.array(image), (IMAGE_SIZE, IMAGE_SIZE))
    pic_matrix.append(image.astype(float)/255)

material = df['material_2'].to_list() # make list of all materials

label = [] # will contain numerical representaion of materials
l = list(set(material)) # словарь

for i in material:
    label.append(l.index(i)) # each materials will equal its index in dict

print(l)
print(set(label))

# x = int(round(len(label)*0.95))
# ds_train_x, ds_test_x = tf.split(pic_matrix, num_or_size_splits=[x, len(label)-x])
# ds_train_y, ds_test_y = tf.split(label, num_or_size_splits=[x, len(label)-x])

ds_train_x, ds_test_x = tf.split(pic_matrix, num_or_size_splits=2)
ds_train_y, ds_test_y = tf.split(label, num_or_size_splits=2)

ds_train = tf.data.Dataset.from_tensor_slices((ds_train_x, ds_train_y))
ds_test = tf.data.Dataset.from_tensor_slices((ds_test_x, ds_test_y))

ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True) 
ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=True) 

class_num = len(l) # number of classes (in this case, number of different dates)

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
    tf.keras.layers.Dense(class_num, activation='softmax'), #num of classes len(l)
])

model.compile(
    optimizer = Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()
history = model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test)
model.save("E:\\models\\material_model_2.h5")

# VISUALISATION OF RESULTS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()