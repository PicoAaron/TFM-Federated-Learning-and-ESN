import pandas as pd
import json
import os
import numpy as np
from federated import wind_data
from federated import ESN, test
from federated import write_data
from federated import process, average_results
from prediction import prediction
import matplotlib.pyplot as plt

# Parameters
total_experiments=1
total_epochs=50
date = '2018-11-01T00:00+10:00'
train_steps = 100


# ESN Hiperparameters 
neurons=100
connectivity=0.1
leaky=1
spectral_radius=0.9
steps=24
lr=0.005


def machine_learning(node, num_experiment=0):

    x_train, y_train, x_test, y_test  = wind_data( data, node, date )

    model = ESN(neurons, connectivity, leaky, spectral_radius, steps, lr)

    results = {}
    results.update( { node: {'loss': [], 'val_loss': [], 'mse': [], 'mae': [], 'accuracy': [] } } )

    for epoch in range(total_epochs):
        print()
        print(f'Epoch {epoch} for node: {node}')
        history = model.fit(x_train,
                            y_train,
                            #validation_data = (x_test, y_test),
                            steps_per_epoch=train_steps,
                            epochs=1,
                            verbose=1)
    
        #test_acc, test_loss = test(model, x_test, y_test)
        test_loss, test_acc, test_mse, test_mae = model.evaluate(x_test, y_test, steps=100)

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

    data_to_write = {'Centralized Individual': results[node]}
    write_data(node, 'centralized_individual',num_experiment, data_to_write)

    #prediction(model, x_train, y_train, y_test)
    return model

    
if __name__ == "__main__":
    
    data = pd.read_csv("data/aemo_2018_mean_hour.csv", sep=',', header=0)

    with open('data/data_network.json') as file:
        nodes = json.load(file)

    nodes = {'ARWF1': nodes['ARWF1']}

    try:
        os.mkdir('model')
    except:
        pass
    try:
        os.mkdir(f'model/centralized_global')
    except:
        pass

    for experiment in range(1, total_experiments+1):
        try:
            os.mkdir(f'model/centralized_individual/experiment_{experiment}')
        except:
            pass

        for node in nodes:
            model = machine_learning(node, num_experiment=experiment)

            model.save(f'model/centralized_individual/experiment_{experiment}/centralized_individual_{node}.h5')

            parameters = {'train_type': 'centralized_individual', 'steps': steps, 'separation_date':date}
            with open(f'model/centralized_individual/parameters.json', 'w') as file:
                json.dump(parameters, file, indent=4)
        
        process(experiment, 'centralized_individual')
    average_results(train_type='centralized_individual')

'''
    print(data['ARWF1'])
    data = np.array(data['ARWF1'])
    print(data)

    
    x_average = np.convolve(data, np.ones(7), 'valid') / 7

    print(len(x_average))

    plt.figure(figsize=(15, 6))
    #plt.plot(data)
    #plt.plot(x_average_dia)
    plt.plot(x_average)
    

    plt.savefig(f'results/prediction_average.png')
'''