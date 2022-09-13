import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import consenso
from consenso import Consensus

import communication
from communication import Communication

import train
from train import Train

import esn
from esn import ESN, test, structure, structure_best

import dataset
from dataset import adjacency, adjacency_radius, wind_data, sequence_many

import write
from write import write_data, write_evaluation, write_weights

import process_results
from process_results import process

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import random
import json
import sys



def prepare_network():
    
    aemo = pd.read_csv("data/aemo_2018.csv", sep=',', header=0)

    with open('data/data_network.json') as file:
        data_network = json.load(file)

    agents = []

    A, neighbors = adjacency(data_network, 4)
    eps = calculate_eps(A)

    print(neighbors)
    '''A = np.array([[0, 1, 1, 0],
                  [1, 0, 1, 0],
                  [1, 1, 0, 1],
                  [0, 0, 1, 0]])

    
    neighbors = {
                        'ARWF1': ['BALDHWF1', 'BLUFF1'],
                        'BALDHWF1': ['ARWF1', 'BLUFF1'],
                        'BLUFF1': ['ARWF1', 'BALDHWF1', 'BOCORWF1'],
                        'BOCORWF1': ['BLUFF1']
    }'''

    #print(A)

    first = True
    for node in data_network:
        _, _, aux_x_test, aux_y_test  = wind_data( aemo, node, 80 )
        if first:
            global_test_x = aux_x_test
            global_test_y = aux_y_test
            first = False
        else:
            global_test_x = np.concatenate([global_test_x, aux_x_test])
            global_test_y = np.concatenate([global_test_y, aux_y_test])

    global_test = {'x': global_test_x, 'y': global_test_y}

    model = {}
    train_data = {}
    test_data = {}
    saved_history = {}

    for node in data_network:
        x_train, y_train, x_test, y_test  = wind_data( aemo, node, 80 )

        train_data.update( {node: {'x': x_train, 'y': y_train} } )
        test_data.update( {node: {'x': x_test, 'y': y_test} } )
        model.update( {node: ESN()} )
        saved_history.update( {node:{'loss': [], 'val_loss': [], 'loss_consensus_local': [], 'loss_consensus_global': []}} )

    return model, train_data, test_data, neighbors, saved_history, eps, global_test


def prepare_network_saved():
    
    aemo = pd.read_csv("data/aemo_2018.csv", sep=',', header=0)

    with open('data/data_network.json') as file:
        data_network = json.load(file)

    agents = []

    with open('data/A.json') as file:
        A = json.load(file)

    with open('data/neighbors.json') as file:
        neighbors = json.load(file)

    eps = calculate_eps(A)



    #print(A)

    first = True
    for node in data_network:
        _, _, aux_x_test, aux_y_test  = wind_data( aemo, node, 80 )
        if first:
            global_test_x = aux_x_test
            global_test_y = aux_y_test
            first = False
        else:
            global_test_x = np.concatenate([global_test_x, aux_x_test])
            global_test_y = np.concatenate([global_test_y, aux_y_test])

    global_test = {'x': global_test_x, 'y': global_test_y}

    model = {}
    train_data = {}
    test_data = {}
    saved_history = {}

    for node in data_network:
        x_train, y_train, x_test, y_test  = wind_data( aemo, node, 80 )

        train_data.update( {node: {'x': x_train, 'y': y_train} } )
        test_data.update( {node: {'x': x_test, 'y': y_test} } )
        model.update( {node: ESN()} )
        saved_history.update( {node:{'loss': [], 'val_loss': [], 'loss_consensus_local': [], 'loss_consensus_global': []}} )

    return model, train_data, test_data, neighbors, saved_history, eps, global_test



def calculate_eps(A):
    laplacian = np.diag(np.sum(A, axis = 1)) - A
    eps = 1 / np.max(np.diag(laplacian))
    return eps


def train(epoch, steps):
    for node in model:

            history = model[node].fit(
                train_data[node]['x'],
                train_data[node]['y'],
                steps_per_epoch=steps,
                epochs=1,
                verbose=1
            )

            test_loss, test_acc = model[node].evaluate(test_data[node]['x'], test_data[node]['y'], steps=steps)

            loss = saved_history[node]['loss']
            loss.append(history.history['loss'][-1])

            val_loss = saved_history[node]['val_loss']
            val_loss.append(test_loss)

            saved_history.update( {node: {'loss': loss, 'val_loss': val_loss}} )

            write_weights(f'{node}_weights', 'a', f'Entrenamiento {epoch}', model[node].get_weights())
            write_evaluation(f'{node}_evaluation', 'a', f'Entrenamiento {epoch}: {test_loss}\n')


def consenso(node, model_aux, process_structure=False, one_or_zero_first=False, one_or_zero_end=False):

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


def rondas_consenso(num_rounds, epoch, steps=20, log=False ):

    for round in range(1, num_rounds+1):
        model_aux = model.copy()
        for node in model:
            model[node].set_weights( consenso(node, model_aux) )
            
    for node in model:
        #weights = model[node].get_weights()
        #weights[0] = structure(weights[0])
        #model[node].set_weights( weights )

        write_weights(f'{node}_weights', 'a', f'Consenso {epoch}', model[node].get_weights())

        if log:
            test_loss, test_acc = model[node].evaluate(global_test['x'], global_test['y'], steps=steps)
            write_evaluation(f'{node}_evaluation', 'a', f'Consenso {epoch}: {test_loss}')


def threshold(n, m):
    if n > m/2:
        return 1
    else:
        return 0

def repeated_links(model):
    structures = []
    for node in model:
        structures.append(model[node].get_weights()[0])

    definitive_structure = []
    for i in range(len(structures[0])):
        row = []
        for j in range(len(structures[0][i])):
            values = [ x[i][j] for x in structures ]
            v = np.mean(values)
            row.append(v)
        definitive_structure.append(row)

    print(definitive_structure)
    print(len(definitive_structure))
    print(len(definitive_structure[0]))
    #map = np.vectorize(threshold)
    #structure_result = map(definitive_structure, np.max(definitive_structure))
    #print(structure_result)

    #max = np.max(np.max(definitive_structure))
    

    for node in model:
        weights = model[node].get_weights()
        weights[0] = definitive_structure
        #print(weights)
        model[node].set_weights(weights)

    return model
    

    


if __name__ == "__main__":

    experiments=5
    num_epochs = 5
    num_rounds = 50
    steps = 100

    #print(json.dumps(neighbors))

    for num_experiment in range(1, experiments+1):

        model, train_data, test_data, neighbors, saved_history, eps, global_test = prepare_network_saved()

        #model = repeated_links(model)
        
        for node in model:
            write_weights(f'{node}_weights', 'w', '', '')
            write_evaluation(f'{node}_evaluation', 'w', f'Inicio\n----------------\n')

        for epoch in range(num_epochs):

            train(epoch, steps)

            rondas_consenso(num_rounds, epoch, steps=steps, log=False)


        train(num_epochs, steps)

        for node in model:
            test_loss, test_acc = model[node].evaluate(global_test['x'], global_test['y'], steps=steps)
            #write_evaluation(f'{node}_evaluation', 'a', f'Consenso {epoch}: {test_loss}')

            data = {'Federated': saved_history[node], 'Consenso': {'val_loss': [test_loss]}}
            write_data(f'{node}', num_experiment, data)

        process(num_experiment)