import os
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_name = 'cifar10'
dir = 'random_images'
batch_size = 128
tensorboard_callback = TensorBoard(log_dir=f'logs/{dataset_name}')

print("Approximately model creating will take ~10-15 minutes")
n_epochs = int(input("Input how many epochs model will have: "))

_, info = tfds.load(dataset_name, split=['train', 'test'], with_info=True)
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

train_data = train_data / 255
test_data = test_data / 255

train_labels, test_labels = train_labels.flatten(), test_labels.flatten()

class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
input_shape = info.features['image'].shape

model = Sequential([
    Conv2D(32, 3, padding='same', input_shape=input_shape, activation='relu'),
    BatchNormalization(),
    Conv2D(32, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, 3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax'),
])

val_loss_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
val_accuracy_stop = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=15)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=n_epochs,
                 validation_data=(test_data, test_labels),
                 callbacks=[tensorboard_callback, val_loss_stop, val_accuracy_stop, lr_schedule],
                 batch_size=batch_size,
                 verbose=1,
                 steps_per_epoch=len(train_data) // batch_size,
                 validation_steps=len(test_data) // batch_size)

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

train_generator = datagen.flow(train_data, train_labels, batch_size=batch_size)
test_generator = datagen.flow(test_data, test_labels, batch_size=batch_size)

history2 = model.fit(train_generator, epochs=n_epochs,
                 validation_data=test_generator,
                 callbacks=[tensorboard_callback, val_loss_stop, val_accuracy_stop, lr_schedule],
                 batch_size=batch_size,
                 verbose=1,
                 steps_per_epoch=len(train_data) // batch_size,
                 validation_steps=len(test_data) // batch_size)

loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Loss: {loss}, Accuracy: {accuracy}")

model.save("image_cnn_model.h5", include_optimizer=True)


