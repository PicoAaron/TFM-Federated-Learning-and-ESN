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

    

    plt.plot([9.582002873514211 , 7.491558633832369 , 6.201419468019522 , 5.4632321194106455 , 5.038963781852348 , 4.76593960687226 , 4.567047140411302 , 4.405258244860406 , 4.270188110949947 , 4.157831181264391])
    #plt.plot([9.720033264160156, 7.812127304077149, 6.793407917022705, 6.112060546875, 5.615701961517334, 5.225644302368164, 4.914515590667724, 4.66768856048584, 4.468250846862793, 4.301808071136475])

    plt.title(f'Model loss')
    plt.ylabel('val loss')
    plt.xlabel('epoch')
    #plt.legend(['Repeated Links', 'Best Structure', 'Best Links', 'Machine Learning','Average Structure', 'Original Structure'], loc='upper right')
    plt.legend(['Federated Learning', 'Machine Learning'], loc='upper left')
    #plt.savefig(f'results/results_esn_methods.png')
    plt.savefig(f'results/FL_AEMO.png')