import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    '''metodo_1 = 'repeated_links'
    metodo_2 = 'best_structure'
    metodo_3 = 'best_links'
    metodo_4 = 'ML_average'
    metodo_5 = 'average_structure'
    metodo_6 = 'original_structure'

    path_1 = f'./results/FLvsML/{metodo_1}/results_average.json'
    path_2 = f'./results/FLvsML/{metodo_2}/results_average.json'
    path_3 = f'./results/FLvsML/{metodo_3}/results_average.json'
    path_4 = f'./results/FLvsML/{metodo_4}/results_average.json'
    path_5 = f'./results/FLvsML/{metodo_5}/results_average.json'
    path_6 = f'./results/FLvsML/{metodo_6}/results_average.json'

    with open(path_1) as file:
        data_1 = json.load(file)

    with open(path_2) as file:
        data_2 = json.load(file)

    with open(path_3) as file:
        data_3 = json.load(file)

    with open(path_4) as file:
        data_4 = json.load(file)
    
    with open(path_5) as file:
        data_5 = json.load(file)

    with open(path_6) as file:
        data_6 = json.load(file)

    plt.plot(data_1['Federated']['val_loss'])
    plt.plot(data_2['Federated']['val_loss'])
    plt.plot(data_3['Federated']['val_loss'])
    plt.plot(data_4['No Federated (Local)']['val_loss'])
    plt.plot(data_5['Federated']['val_loss'])
    plt.plot(data_6['Federated']['val_loss'])'''

    plt.plot([11.0398, 5.5378, 4.9438, 4.4759, 4.1753, 3.9754])
    plt.plot([10.0280, 7.1938, 6.0512, 5.4071, 4.9543, 4.6055])

    plt.title(f'Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['Repeated Links', 'Best Structure', 'Best Links', 'Machine Learning','Average Structure', 'Original Structure'], loc='upper right')
    plt.legend(['Federated Learning', 'Machine Learning', 'Best Links', 'Machine Learning','Average Structure', 'Original Structure'], loc='upper left')
    #plt.savefig(f'results/results_esn_methods.png')
    plt.savefig(f'results/FL_ML_global.png')