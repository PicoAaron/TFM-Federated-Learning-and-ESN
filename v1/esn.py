import keras
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
from numpy import array

import matplotlib.pyplot as plt

def esn_layer(units, connectivity, leaky, spectral_radius):
  layer = tfa.layers.ESN(
      units = units,
      connectivity = connectivity,
      leaky = leaky,
      spectral_radius = spectral_radius,
      use_norm2 = False,
      use_bias = True,
      activation = 'tanh',
      kernel_initializer = 'glorot_uniform',
      recurrent_initializer = 'glorot_uniform',
      bias_initializer = 'zeros',
      return_sequences=False,
      go_backwards=False,
      unroll=False
  )
  return layer


def test(model, test_x, test_y):
  test_loss, test_acc = model.evaluate(test_x, test_y)
  return test_acc, test_loss


# Modelo con ESN: entrada + reservoir + salida
def ESN():
    input_shape = (15, 1)

    inputs = keras.Input(shape=input_shape)
    reservoir1 = esn_layer(10, 0.05, 1, 0.9)(inputs)
    outputs = keras.layers.Dense(1)(reservoir1)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer=Adam(learning_rate=0.05)
    model.compile(optimizer=optimizer, loss='mse', metrics='mse')

    return model
