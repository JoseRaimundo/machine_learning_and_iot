import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import shutil

# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Librariesauxiliares
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

def run_tensorflow(x_train, x_test, y_train, y_test, EPOCHS, BATCH_SIZE):
    # plt.figure()
    # plt.imshow(x_train[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()


    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=10)

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

    print('\nTest accuracy:', test_acc)