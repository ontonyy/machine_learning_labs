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

# Define hyperparameters
batch_size = 128

n_epochs = int(input("Input how many epochs model will have: "))

(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

train_data = train_data / 255
test_data = test_data / 255

train_labels, test_labels = train_labels.flatten(), test_labels.flatten()


def create_model():
    return Sequential([
        Conv2D(32, 3, padding='same', input_shape=(32, 32, 3), activation='relu'),
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
        Dense(10, activation='softmax'),
    ])


def create_callbacks():
    tensorboard_callback = TensorBoard(log_dir=f'logs/cifar10')
    val_loss_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    val_accuracy_stop = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=15)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    return [tensorboard_callback, val_loss_stop, val_accuracy_stop, lr_schedule]


def compile_model():
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])


def train_model_simple():
    model.fit(train_data, train_labels, epochs=n_epochs,
              validation_data=(test_data, test_labels),
              callbacks=create_callbacks(),
              batch_size=batch_size,
              verbose=1,
              steps_per_epoch=len(train_data) // batch_size,
              validation_steps=len(test_data) // batch_size)


def train_model_with_augmentation():
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

    model.fit(train_generator, epochs=n_epochs,
              validation_data=test_generator,
              callbacks=create_callbacks(),
              batch_size=batch_size,
              verbose=1,
              steps_per_epoch=len(train_data) // batch_size,
              validation_steps=len(test_data) // batch_size)


def evaluate():
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f"Loss: {loss}, Accuracy: {accuracy}")


def save_model():
    model.save("image_cnn_model.h5", include_optimizer=True)


if __name__ == '__main__':
    print("Approximately model creating will take ~10-15 minutes")
    model = create_model()
    compile_model()
    train_model_simple()
    train_model_with_augmentation()
    evaluate()
    save_model()
