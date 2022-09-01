import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    metodo_1 = 'repeated_links'
    metodo_2 = 'ML_local'

    path_1 = f'./results/FLvsML/{metodo_1}/results_average.json'
    path_2 = f'./results/FLvsML/{metodo_2}/results_average.json'

    with open(path_1) as file:
        data_1 = json.load(file)

    with open(path_2) as file:
        data_2 = json.load(file)

    plt.plot(data_1['Federated']['val_loss'])
    plt.plot([11.4388 , 8.4191 , 7.2449 , 6.5515 , 6.08112 , 5.7477])
    plt.title(f'Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'results/results_{metodo_1}_{metodo_2}.png')