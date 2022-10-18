import pandas as pd
import json
import os
import numpy as np
from federated import wind_data
from federated import ESN, test
from federated import write_data
from federated import process


# Parameters
total_experiments=1
total_epochs=5
date = '2018-11-01T00:00+10:00'

# ESN Hiperparameters 
neurons=100
connectivity=0.1
leaky=1
spectral_radius=0.9
steps=30
lr=0.05


def machine_learning(local=False, num_experiment=0):
    data = pd.read_csv("data/aemo_2018.csv", sep=',', header=0)

    with open('data/data_network.json') as file:
        nodes = json.load(file)

    first = True

    for node in nodes:
        aux_x_train, aux_y_train, aux_x_test, aux_y_test  = wind_data( data, node, date )
        if first:
            x_train = aux_x_train
            y_train = aux_y_train
            x_test = aux_x_test
            y_test = aux_y_test
            first = False

        else:
            x_train = np.concatenate([x_train, np.array(aux_x_train)])
            y_train = np.concatenate([y_train, aux_y_train])
            x_test = np.concatenate([x_test, aux_x_test])
            y_test = np.concatenate([y_test, aux_y_test])

    model = ESN(neurons, connectivity, leaky, spectral_radius, steps, lr)

    # Si queremos que el modelo centralizado global haga predicciones sobre granjas concretas 
    if local:

        results = {}
        for node in nodes:
            results.update( { node: {'loss': [], 'val_loss': []} } )

        for epoch in range(total_epochs):
            print(f'Epoch: {epoch}')
            history = model.fit(x_train,
                                y_train,
                                #validation_data = (x_test, y_test),
                                steps_per_epoch=100,
                                epochs=1,
                                verbose=1)
            
            for node in nodes:
                _, _, test_node_x, test_node_y = wind_data( data, node, date )
                test_acc, test_loss = test(model, test_node_x, test_node_y)

                l = results[node]['loss']
                l.append(history.history['loss'][0])
                vl = results[node]['val_loss']
                vl.append(test_loss)

                results.update( {node: {'loss': l, 'val_loss': vl } } )
     
        for node in nodes: 
            data_to_write = {'ML': results[node]}
            write_data(node, num_experiment, data_to_write)

        process(num_experiment)

    # Si queremos que el modelo centralizado global haga predcciones sobre el conjunto 
    # de datos de todas las granjas
    else:

        saved_history = { 'loss': [], 'val_loss': []}

        for epoch in range(total_epochs):
            print(f'Epoch: {epoch}')
            history = model.fit(x_train,
                                y_train,
                                #validation_data = (x_test, y_test),
                                steps_per_epoch=100,
                                epochs=1,
                                verbose=1)


            test_acc, test_loss = test(model, x_test, y_test)

            l = saved_history['loss']
            l.append(history.history['loss'][0])
            vl = saved_history['val_loss']
            vl.append(test_loss)
            saved_history.update( {'loss': l, 'val_loss': vl } )

        try:
            os.mkdir('results')
        except:
            pass

        try:
            os.mkdir('results/processed_results')
        except:
            pass

        data = {'ML': saved_history}
        with open(f'./results/processed_results/results_{num_experiment}.json', 'w') as file:
            json.dump(data, file, indent=4)
    

if __name__ == "__main__":
    
    for experiment in range(1, total_experiments+1):
        machine_learning(local=True, num_experiment=experiment)