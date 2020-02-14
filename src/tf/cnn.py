import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import shutil
# import matplotlib.pyplot as plt

# Carrega os dados e organizam na pasta
def split_list(data_dir, train_dir, test_dir):
    data = os.listdir(data_dir)
    print(len(data))
    try:
        os.makedirs(train_dir)
        os.makedirs(test_dir)
    except OSError:
        print ("Creation of the directory %s failed" % train_dir)
        print ("Creation of the directory %s failed" % test_dir)
    else:
        print ("Successfully created the directory %s " % train_dir)
        print ("Successfully created the directory %s " % test_dir)

    for i in range(len(data)):
        src = os.path.join(data_dir, data[i])
        if(i < int(len(data)/3)):
            shutil.move(src, test_dir)
        else:
            shutil.move(src, train_dir)

# Carregando dados
PATH_DATA = '../../databases/cnn/flowers-recognition'
data_dir = os.path.join(PATH_DATA, 'flowers')
train_dir = os.path.join(PATH_DATA, 'train')
test_dir = os.path.join(PATH_DATA, 'test')
total_train = 0
total_test  = 0

if  not os.path.exists(train_dir):
    for i in os.listdir(data_dir):
        class_data_dir = os.path.join(data_dir, i)
        class_train_dir = os.path.join(train_dir, i)
        class_test_dir = os.path.join(test_dir, i)
        print(class_train_dir)
        split_list(class_data_dir, class_train_dir, class_test_dir)
    # data = os.listdir(train_dir)

# Computa o tamanho dos vetores
for i in os.listdir(train_dir):
    class_data_dir = os.path.join(train_dir, i)
    total_train += len(os.listdir(class_data_dir))

for i in os.listdir(test_dir):
    class_data_dir = os.path.join(test_dir, i)
    total_test += len(os.listdir(class_data_dir))


batch_size = 100
epochs = 100
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Organiza os conjutos de treino e test (usando a palavra teste para referi-se à validação)
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=train_dir,
                                                            shuffle=True,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            class_mode="categorical")


val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode="binary")

# Criando rede neural
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(6, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
# Treinando modelo
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_test // batch_size
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
print("val_acc")
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
# print("teste")