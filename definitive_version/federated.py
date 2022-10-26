import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import pandas as pd

import random
import json
import os
import sys

#from prediction import prediction

import warnings
warnings.filterwarnings('ignore')


# Parameters
experiments=1
num_epochs = 15
num_rounds = 500

train_steps = 500
date = '2018-11-01T00:00+10:00'


# ESN Hiperparameters 
neurons=100
connectivity=0.1
leaky=1
spectral_radius=0.9
steps=24
lr=0.005


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


def test(model, test_x, test_y, steps_test=100):
  test_loss, test_acc, mse, mae = model.evaluate(test_x, test_y, steps=steps_test)
  #test_loss, test_acc = model.evaluate(test_x, test_y)
  return test_acc, test_loss


# Modelo ESN: entrada + reservoir + salida
def ESN(neurons=100, connectivity=0.1, leaky=1, spectral_radius=0.9, steps=30, lr=0.05):
    input_shape = (steps, 1)

    inputs = keras.Input(shape=input_shape)
    reservoir = esn_layer(neurons, connectivity, leaky, spectral_radius)(inputs)
    outputs = keras.layers.Dense(1)(reservoir)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer=Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy', 'mse', 'mae'])

    return model


def calculate_eps(A):
    laplacian = np.diag(np.sum(A, axis = 1)) - A
    eps = 1 / np.max(np.diag(laplacian))
    return eps


def prepare_global_test(data_network, aemo):
    first = True
    for node in data_network:
        _, _, aux_x_test, aux_y_test  = wind_data( aemo, node, date )
        if first:
            global_test_x = aux_x_test
            global_test_y = aux_y_test
            first = False
        else:
            global_test_x = np.concatenate([global_test_x, aux_x_test])
            global_test_y = np.concatenate([global_test_y, aux_y_test])

    global_test = {'x': global_test_x, 'y': global_test_y}

    return global_test


def aux_prepare_network(A, data_network, neighbors, aemo):
    eps = calculate_eps(A)

    network = {}

    model = {}
    train_data = {}
    test_data = {}
    saved_history = {}
    consenso_history = {}

    for node in data_network:
        x_train, y_train, x_test, y_test  = wind_data( aemo, node, date )

        train_data.update( {node: {'x': x_train, 'y': y_train} } )
        test_data.update( {node: {'x': x_test, 'y': y_test} } )
        model.update( {node: ESN(neurons, connectivity, leaky, spectral_radius, steps, lr)} )
        saved_history.update( {node:{'loss': [], 'val_loss': [], 'consenso': []}} )

    global_test = prepare_global_test(data_network, aemo)

    network.update({'model': model, 'train_data': train_data, 'test_data': test_data, 'neighbors': neighbors, 'saved_history': saved_history, 'eps': eps, 'global_test': global_test})
    
    return network


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return X, y


def aux_wind_data(df, name):
  seq_aux = df[name]
  
  seq = []
  for elem in seq_aux:
    if np.isnan(elem):
      seq.append(0.0)
    else: seq.append(elem)
	
  seq_x, seq_y = split_sequence(seq, steps)

  x = []
  y = []
  #zero = [ 0 for i in range(steps)]
  for i in range(len(seq_x)):
    #if not np.array_equal(seq_x[i], zero):
    x.append(seq_x[i])
    y.append(seq_y[i])
  
  x = np.array(x)
  y = np.array(y)

  x = x.reshape((x.shape[0], x.shape[1], 1))

  return x, y


def wind_data(df, name, date):
  #df[name] = df[name]/np.linalg.norm(df[name])

  df_train = df.loc[df['timestamp'] < date]
  df_test = df.loc[df['timestamp'] >= date]

  df_train.to_csv('train.csv')
  df_test.to_csv('test.csv')

  x_train, y_train = aux_wind_data(df_train, name)
  x_test, y_test = aux_wind_data(df_test, name)

  return x_train, y_train, x_test, y_test


def prepare_network(data="data/aemo_2018_mean_hour.csv"):
    
    aemo = pd.read_csv(data, sep=',', header=0)

    with open('data/data_network.json') as file:
        data_network = json.load(file)

    agents = []

    with open('data/A.json') as file:
        A = json.load(file)

    with open('data/neighbors.json') as file:
        neighbors = json.load(file)

    network = aux_prepare_network(A, data_network, neighbors, aemo)

    return network


def write_weights(name, mode, title, weights):
  path=f'logs/weights'
  file=f'{path}/{name}.txt'

  try:
    os.mkdir('logs')
  except:
    pass
  
  try:
    os.mkdir(path)
  except:
    pass

  with open(file, mode) as f:
    f.write(f'{title}:\n')
    f.write('------------------------------------------------------------------------\n\n')
    for layer in range(len(weights)):
        f.write(f'Layer {layer}:\n')
        f.write(f'{weights[layer]}\n\n')
    f.write('\n\n')


def write_evaluation(name, mode, text):
  path=f'logs/evaluation'
  file=f'{path}/{name}.txt'

  try:
    os.mkdir('logs')
  except:
    pass
  
  try:
    os.mkdir(path)
  except:
    pass
  with open(file, mode) as f:
    f.write(f'{text}\n')


def write_data(name, train_type,num_experiment, data):
  path = f'results/experiment_results/{train_type}/experiment_{num_experiment}'
  file=f'{path}/{name}.json'
  
  try:
    os.mkdir('results')
  except:
    pass

  try:
    os.mkdir('results/experiment_results')
  except:
    pass

  try:
    os.mkdir(f'results/experiment_results/{train_type}')
  except:
    pass

  try:
    os.mkdir(path)
  except:
    pass

  with open(file, 'w') as file:
    json.dump(data, file, indent=4)


def train(model, train_data, test_data, saved_history, epoch, train_steps):
    for node in model:

            history = model[node].fit(
                train_data[node]['x'],
                train_data[node]['y'],
                steps_per_epoch=train_steps,
                epochs=1,
                verbose=1
            )

            #test_loss, test_acc = model[node].evaluate(test_data[node]['x'], test_data[node]['y'])

            loss = saved_history[node]['loss']
            loss.append(history.history['loss'][-1])

            mse = saved_history[node]['mse']
            #mse.append(history.history['mse'][-1])

            mae = saved_history[node]['mae']
            #mae.append(history.history['mae'][-1])

            accuracy = saved_history[node]['accuracy']
            #accuracy.append(history.history['accuracy'][-1])

            #val_loss = saved_history[node]['val_loss']
            #val_loss.append(test_loss)

            #saved_history.update( {node: {'loss': loss, 'val_loss': val_loss, 'consenso': saved_history[node]['consenso']}} )
            saved_history.update( {node: {'loss': loss, 'consenso': saved_history[node]['consenso'], 'mse': mse, 'mae': mae, 'accuracy': accuracy}} )
            
            write_weights(f'{node}_weights', 'a', f'Entrenamiento {epoch}', model[node].get_weights())
            #write_evaluation(f'{node}_evaluation', 'a', f'Entrenamiento {epoch}: {test_loss}\n')


def consenso(node, model, neighbors, eps, model_aux, process_structure=False, one_or_zero_first=False, one_or_zero_end=False):

    wi = model[node].get_weights()
    
    w_neighbors =  []

    for n in neighbors[node]:
        w_neighbors.append( model_aux[n].get_weights() )

    layers_nc = []
    for layer in range(0, len(wi)):
        if layer not in layers_nc:
            addition = 0
            for wj in w_neighbors:
                addition += (wj[layer] - wi[layer])

            wi[layer] = wi[layer] + (eps * addition)

    return wi


def rondas_consenso(model, neighbors, eps, test, saved_history, num_rounds, epoch, train_steps=20, log=False ):

    for round in range(1, num_rounds+1):
        model_aux = model.copy()
        for node in model:
            model[node].set_weights( consenso(node, model, neighbors, eps, model_aux) )
            
    for node in model:
        #weights = model[node].get_weights()
        #weights[0] = structure(weights[0])
        #model[node].set_weights( weights )

        write_weights(f'{node}_weights', 'a', f'Consenso {epoch}', model[node].get_weights())

        if log:
            test_loss, test_acc, test_mse, test_mae = model[node].evaluate(test[node]['x'], test[node]['y'], steps=100)

            loss_consenso = saved_history[node]['consenso']
            loss_consenso.append(test_loss)

            mse = saved_history[node]['mse']
            mse.append(test_mse)

            mae = saved_history[node]['mae']
            mae.append(test_mae)

            accuracy = saved_history[node]['accuracy']
            accuracy.append(test_acc)


            saved_history.update( {node: {'loss': saved_history[node]['loss'], 'consenso': loss_consenso, 'mse': mse, 'mae': mae, 'accuracy': accuracy}} )

            write_evaluation(f'{node}_evaluation', 'a', f'Consenso {epoch}: {test_loss}')
            
            data = {'Federated': network['saved_history'][node]}
            write_data(f'{node}', 'federated',num_experiment, data)


def join_results(data):
    # Create the unified results (means of the diferents nodes)
    #----------------------------------------------------------
    results = {}

    for category_name in data[0]:
        category = {}

        for subcategory_name in data[0][category_name]:
            
            subcategory = []

            for i in range(len(data[0][category_name][subcategory_name])):
                step = []   
                for d in data:
                    #print(i)
                    #print(d)
                    step.append(d[category_name][subcategory_name][i])
                subcategory.append(np.mean(step))
            
            category.update( {subcategory_name: subcategory} )

        results.update( {category_name: category} )
    
    return results


def average_results(path=f'./results/processed_results/', train_type='federated', title_1='', title_2=''):
    #path = f'./results/processed_results/'

    path = path + train_type

    files = os.listdir(path)

    # Read the results of the experiment
    #----------------------------------------------------------
    data = []

    for file_name in files:
        if file_name != 'results_average.json':
            try:
                with open(f'{path}/{file_name}') as file:
                    #data = json.load(file)
                    data.append(json.load(file))
            except:
                pass
    

    results = join_results(data)
    
    with open(f'{path}/results_average.json', 'w') as file:
        json.dump(results, file, indent=4)


def process(num_experiment, train_type):
  
    path = f'./results/experiment_results/{train_type}/experiment_{num_experiment}'
    path_processed = f'./results/processed_results/{train_type}'

    try:
        os.mkdir('results/processed_results')
    except:
      pass
    try:
        os.mkdir(path_processed)
    except:
      pass

    files = os.listdir(path)

    # Read the results of the experiment
    #----------------------------------------------------------
    data = []

    for file_name in files:
        
        with open(f'{path}/{file_name}') as file:
            #data = json.load(file)
            data.append(json.load(file))

    #print(data)
    #----------------------------------------------------------

    # Create the unified results (means of the diferents nodes)
    #----------------------------------------------------------
    
    results = join_results(data)

    with open(f'{path_processed}/results_{num_experiment}.json', 'w') as file:
        json.dump(results, file, indent=4)

    #with open(f'{path}/results_{num_experiment}.json', 'w') as file:
        #json.dump(results, file, indent=4)
  

if __name__ == "__main__":
  

    for num_experiment in range(1, experiments+1):

      network = prepare_network(data="data/aemo_2018_mean_hour.csv")
      
      for node in network['model']:
          write_weights(f'{node}_weights', 'w', '', '')
          write_evaluation(f'{node}_evaluation', 'w', f'Inicio\n----------------\n')

      for epoch in range(num_epochs):

          train(network['model'], network['train_data'], network['test_data'], network['saved_history'], epoch, train_steps)

          rondas_consenso(network['model'], network['neighbors'], network['eps'], network['test_data'], network['saved_history'], num_rounds, epoch, train_steps, log=True)

      for node in network['model']:
          #test_loss, test_acc = network['model'][node].evaluate(network['global_test']['x'], network['global_test']['y'], steps=train_steps)
          #write_evaluation(f'{node}_evaluation', 'a', f'Consenso {epoch}: {test_loss}')

          data = {'Federated': network['saved_history'][node]}
          write_data(f'{node}', 'federated',num_experiment, data)

      

      try:
          os.mkdir('model')
      except:
          pass
      try:
          os.mkdir(f'model/federated')
      except:
          pass
      try:
          os.mkdir(f'model/federated/experiment_{num_experiment}')
      except:
          pass

      model = network['model']['ARWF1']
      model.save(f'model/federated/experiment_{num_experiment}/federated.h5')

      parameters = {'train_type': 'federated',
                      'neurons': neurons, 
                      'train_steps': train_steps, 
                      'epochs': num_epochs,
                      'learning_rate': lr,
                      'steps': steps, 
                      'rounds_consensus': num_rounds,
                      'separation_date':date}
      with open(f'model/federated/parameters.json', 'w') as file:
          json.dump(parameters, file, indent=4)


      process(num_experiment, 'federated')
    average_results(train_type='federated')
