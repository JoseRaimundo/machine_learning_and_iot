from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import requests
import re
import seaborn
import matplotlib.pyplot as plt
# from tensorflow import feature_column
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers

# Carrega os dados
dataframe = pd.read_csv("../../dataset/classification/data/HomeC.csv")
dataframe = dataframe.drop(axis = 1, labels = ['icon','summary','cloudCover',  'time'])

# dataframe = pd.read_csv("../../dataset/test.csv")
# dataframe = dataframe.drop(axis = 1, labels = ['thal'])



def normalize_calss(dataframe = dataframe, target = 'target', limiar = 1):
    class_target = dataframe[target]
    temp_list = []
    for i in class_target.values.tolist():
        if (i > limiar):
            temp_list.append(1)
        else:
            temp_list.append(0)
        
    dataframe = dataframe.drop(axis = 1, labels = [target])
    dataframe['target'] = temp_list
    return dataframe
    

dataframe = normalize_calss(target='use [kW]')
print(dataframe.head())

# Faz o split dos dados
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)


# Converte de dataframe para o tipo de entrada do tensorflow
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
    return ds

batch_size = 5 
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Carrega os dados de forma organizada
feature_columns = []
for i in range(len(dataframe.columns)-1):
    feature_columns.append(feature_column.numeric_column(dataframe.columns[i]))

# Construindo a topologia da rede
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Treino
model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)