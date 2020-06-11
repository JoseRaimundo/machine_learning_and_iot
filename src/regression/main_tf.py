import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import common

# Acessando os dados

def run_tensorflow(x_train, x_test, y_train, y_test, EPOCHS, batch_size):
  def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


  model = build_model()

  early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

  history = model.fit(x_train, y_train, epochs=EPOCHS,
                      validation_split = 0.2, verbose=1, callbacks=[early_stop])


  preds = model.predict(x_test).flatten()
  for i in range(10):
      print({y_test[i], " " , preds[i]})

  loss, mae, mse = model.evaluate(x_test, y_test, verbose=11)
  print("|------------- RESULTADO --------------- |")
  print("|- MAE : ", mae)
  print("|- MSE : ", mse)
  print("|------------- RESULTADO --------------- |")

