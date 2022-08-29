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


def replace(n, num):
  if n != 0:
    return num
  else:
    return 0


def process(n):
  return n/3


def threshold(n, m):
  if n > m/2:
    return n
  else:
    return 0


def structure(weights_list, one_or_zero_first=False, one_or_zero_end=False):

  if one_or_zero_first:
    map = np.vectorize(replace)
    structures_list = map(weights_list, 1)
    #print(structures_list)
  else:
    structures_list = weights_list

  addition = structures_list[0]
  for i in range(1, len(structures_list)):
    addition = np.add(addition, structures_list[i])
  #print(addition)

  #return addition
  map = np.vectorize(threshold)
  structure_result = map(addition, np.max(addition))

  if one_or_zero_end:
    map = np.vectorize(replace)
    structure_result = map(structure_result, 1)
  
  return structure_result


def structure_best(weights_list, performance_list, one_or_zero_first=False, one_or_zero_end=False):

  if one_or_zero_first:
    sorted_weights = [i for _,i in sorted(zip(performance_list, weights_list), reverse=True)]

    structures_list = []

    for i in range(len(sorted_weights)):
      map = np.vectorize(replace)
      w = map(sorted_weights[i], i+1)
      structures_list.append(w)
  
  else:
    structures_list = weights_list

  addition = structures_list[0]
  for i in range(1, len(structures_list)):
    addition = np.add(addition, structures_list[i])
  #print(addition)

  
  
  if one_or_zero_end:
    map = np.vectorize(threshold)
    structure_result = map(addition, np.max(addition))

    map = np.vectorize(replace)
    structure_result = map(structure_result, 1)
  else:
    structure_result = addition
  
  return structure_result


def test(model, test_x, test_y):
  test_loss, test_acc = model.evaluate(test_x, test_y, steps=100)
  return test_acc, test_loss


# Modelo con ESN: entrada + reservoir + salida
def ESN():
    input_shape = (30, 1)

    inputs = keras.Input(shape=input_shape)
    reservoir = esn_layer(100, 0.25, 1, 0.9)(inputs)
    outputs = keras.layers.Dense(1)(reservoir)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer=Adam(learning_rate=0.05)
    model.compile(optimizer=optimizer, loss='mae', metrics='accuracy')

    return model
