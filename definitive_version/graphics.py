import matplotlib.pyplot as plt
import json

if __name__ == "__main__":

    federated = True
    centralized_global = True
    centralized_individual = True
    

    path_1 = f'./results/processed_results/federated/results_average.json'
    path_2 = f'./results/processed_results/centralized_global/results_average.json'
    path_3 = f'./results/processed_results/centralized_individual/results_average.json'

    if federated:
        with open(path_1) as file:
            data_1 = json.load(file)

        plt.plot(data_1['Federated']['consenso'])

    if centralized_global:
        with open(path_2) as file:
            data_2 = json.load(file)

        plt.plot(data_2['centralized_global']['val_loss'])

    if centralized_individual:
        with open(path_3) as file:
            data_3 = json.load(file)

        plt.plot(data_3['Centralized Individual']['val_loss'])

    
    
    


    plt.title(f'Model loss')
    plt.ylabel('val loss')
    plt.xlabel('epoch')
    plt.legend(['Federated Learning', 'Machine Learning global', 'Machine Learning individual'], loc='upper left')
    plt.savefig(f'results/history.png')