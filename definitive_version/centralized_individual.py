import pandas as pd
import json
import os
import numpy as np
from federated import wind_data
from federated import ESN, test
from federated import write_data, write_evaluation, write_weights
from federated import process


# Parameters
total_experiments=1
total_epochs=1
date = '2018-11-01T00:00+10:00'


# ESN Hiperparameters 
neurons=100
connectivity=0.1
leaky=1
spectral_radius=0.9
steps=30
lr=0.05


def machine_learning(node, num_experiment=0):

    x_train, y_train, x_test, y_test  = wind_data( data, node, date )

    model = ESN(neurons, connectivity, leaky, spectral_radius, steps, lr)

    results = {}
    results.update( { node: {'loss': [], 'val_loss': []} } )

    for epoch in range(total_epochs):
        print()
        print(f'Epoch {epoch} for node: {node}')
        history = model.fit(x_train,
                            y_train,
                            #validation_data = (x_test, y_test),
                            steps_per_epoch=100,
                            epochs=1,
                            verbose=1)
    
    
        test_acc, test_loss = test(model, x_test, y_test)

        l = results[node]['loss']
        l.append(history.history['loss'][0])
        vl = results[node]['val_loss']
        vl.append(test_loss)

        results.update( {node: {'loss': l, 'val_loss': vl } } )


    data_to_write = {'ML': results[node]}
    write_data(node, num_experiment, data_to_write)

    
if __name__ == "__main__":
    
    data = pd.read_csv("data/aemo_2018.csv", sep=',', header=0)

    with open('data/data_network.json') as file:
        nodes = json.load(file)

    nodes = {'ARWF1': nodes['ARWF1']}

    for experiment in range(1, total_experiments+1):
        for node in nodes:
            machine_learning(node, num_experiment=experiment)

        
        process(experiment)