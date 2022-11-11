import pandas as pd
import json
import os
import numpy as np
from federated import wind_data
from federated import ESN, test
from federated import write_data
from federated import process, average_results
from prediction import prediction


# Parameters
total_experiments=1
total_epochs= 50
date = '2018-11-01T00:00+10:00'
train_steps=100

# ESN Hiperparameters 
neurons=100
connectivity=0.1
leaky=1
spectral_radius=0.9
steps=24
lr=0.005


def machine_learning(local=False, num_experiment=0):
    data = pd.read_csv("data/aemo_2018_mean_hour.csv", sep=',', header=0)

    with open('data/data_network.json') as file:
        nodes = json.load(file)

    first = True

    for node in nodes:
        print(node)
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
            results.update( { node: {'loss': [], 'val_loss': [], 'mse': [], 'mae': [], 'accuracy': [] } } )

        for epoch in range(total_epochs):
            print(f'Epoch: {epoch}')
            history = model.fit(x_train,
                                y_train,
                                #validation_data = (x_test, y_test),
                                steps_per_epoch=train_steps,
                                epochs=1,
                                verbose=1)
            
            for node in nodes:
                _, _, test_node_x, test_node_y = wind_data( data, node, date )
                #test_acc, test_loss, test_mse, test_mae,  = test(model, test_node_x, test_node_y)
                test_loss, test_acc, test_mse, test_mae = model.evaluate(test_node_x, test_node_y, steps=100)

                l = results[node]['loss']
                l.append(history.history['loss'][0])

                vl = results[node]['val_loss']
                vl.append(test_loss)

                mse = results[node]['mse']
                mse.append(test_mse)

                mae = results[node]['mae']
                mae.append(test_mae)

                accuracy = results[node]['accuracy']
                accuracy.append(test_acc)

                results.update( {node: {'loss': l, 'val_loss': vl, 'mse': mse, 'mae': mae, 'accuracy': accuracy } } )
     
        for node in nodes: 
            data_to_write = {'centralized_global': results[node]}
            write_data(node, 'centralized_global', num_experiment, data_to_write)

        #process(num_experiment)

        return model

    # Si queremos que el modelo centralizado global haga predcciones sobre el conjunto 
    # de datos de todas las granjas
    else:

        saved_history = { 'loss': [], 'val_loss': [], 'mse': [], 'mae': [], 'accuracy': []}

        for epoch in range(total_epochs):
            print(f'Epoch: {epoch}')
            history = model.fit(x_train,
                                y_train,
                                #validation_data = (x_test, y_test),
                                steps_per_epoch=train_steps,
                                epochs=1,
                                verbose=1)


            #test_acc, test_loss = test(model, x_test, y_test, 100)
            test_loss, test_acc, test_mse, test_mae = model.evaluate(test_node_x, test_node_y, steps=100)

            l = saved_history['loss']
            l.append(history.history['loss'][0])

            vl = saved_history['val_loss']
            vl.append(test_loss)

            mse = saved_history[node]['mse']
            mse.append(test_mse)

            mae = saved_history[node]['mae']
            mae.append(test_mae)

            accuracy = saved_history[node]['accuracy']
            accuracy.append(test_acc)

            saved_history.update( {'loss': l, 'val_loss': vl, 'mse': mse, 'mae': mae, 'accuracy': accuracy} )

        data_to_write = {'centralized_global': saved_history}
        write_data('centralized_global', 'centralized_global', num_experiment, data_to_write)

        return model
    

if __name__ == "__main__":
    
    for experiment in range(1, total_experiments+1):
        model = machine_learning(local=True, num_experiment=experiment)


        try:
            os.mkdir('model')
        except:
            pass
        try:
            os.mkdir(f'model/centralized_global')
        except:
            pass
        try:
            os.mkdir(f'model/centralized_global/experiment_{experiment}')
        except:
            pass

        model.save(f'model/centralized_global/experiment_{experiment}/centralized_global.h5')

        parameters = {'train_type': 'centralized_global',
                        'neurons': neurons,
                        'connectivity': connectivity,
                        'leaky': leaky,
                        'spectral_radius': spectral_radius,
                        'train_steps': train_steps, 
                        'epochs': total_epochs,
                        'learning_rate': lr,
                        'steps': steps, 
                        'separation_date':date}
        with open(f'model/centralized_global/parameters.json', 'w') as file:
            json.dump(parameters, file, indent=4)

        process(experiment, 'centralized_global')
    average_results(train_type='centralized_global')
        