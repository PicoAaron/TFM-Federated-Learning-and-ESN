import pandas as pd
import json
import numpy as np
from dataset import adjacency, adjacency_radius, wind_data, sequence_many
from esn import ESN, test
from write import write_data, write_evaluation, write_weights
from process_results import process


def machine_learning(local=False, num_experiment=0):
    data = pd.read_csv("data/aemo_2018.csv", sep=',', header=0)
    nodes = ['ARWF1', 'BALDHWF1', 'BLUFF1', 'BOCORWF1']

    first = True

    for node in nodes:
        aux_x_train, aux_y_train, aux_x_test, aux_y_test  = wind_data( data, node, 80 )
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

    model = ESN()

    if local:

        results = {}
        for name in nodes:
            results.update( { name: {'loss': [], 'val_loss': []} } )

        for i in range(10):
            history = model.fit(x_train,
                                y_train,
                                #validation_data = (x_test, y_test),
                                steps_per_epoch=100,
                                epochs=1,
                                verbose=1)
            
            for name in nodes:
                _, _, test_node_x, test_node_y = wind_data( data, node, 80 )
                test_acc, test_loss = test(model, test_node_x, test_node_y)

                l = results[name]['loss']
                l.append(history.history['loss'][0])
                vl = results[name]['val_loss']
                vl.append(test_loss)

                '''
                print(l)
                print(vl)
                print(history.history['loss'][0])
                print(test_loss)
                '''

                results.update( {name: {'loss': l, 'val_loss': vl } } )

        
        for node in nodes: 
            data = {'ML (Local)': results[node]}
            write_data(node, num_experiment, data)


        process(num_experiment)



    else:
        history = model.fit(x_train,
                            y_train,
                            validation_data = (x_test, y_test),
                            steps_per_epoch=100,
                            epochs=6,
                            verbose=1)

        data = {'ML':history.history}
        with open(f'./results/processed_results/results_{num_experiment}.json', 'w') as file:
            json.dump(data, file, indent=4)
    

if __name__ == "__main__":
    
    for num_experiment in range(1, 5+1):
        machine_learning(local=True, num_experiment=num_experiment)