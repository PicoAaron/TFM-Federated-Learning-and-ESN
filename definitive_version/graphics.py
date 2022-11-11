import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
from federated import wind_data

def history():
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
    plt.ylabel('val loss (mae)')
    plt.xlabel('epoch')
    plt.legend(['ACo-L', 'Global model', 'Single model'], loc='upper right', fontsize=14)

    axes = plt.gca()
    axes.title.set_size(14)
    axes.xaxis.label.set_size(14)
    axes.yaxis.label.set_size(14)

    plt.savefig(f'results/history.png', bbox_inches='tight',pad_inches = 0.05)


def plot_prediction():

    data = pd.read_csv("data/aemo_2018_mean_hour.csv", sep=',', header=0)
    x_train, y_train, x_test, y_test  = wind_data( data, 'ARWF1', '2018-11-01T00:00+10:00' )

    y_train = y_train[-30*24:]

    with open('results/prediction/prediction_values_federated.json') as file:
        federated = json.load(file)
    
    with open('results/prediction/prediction_values_aggregated.json') as file:
        aggregated = json.load(file)

    pred_federated = federated['prediction']
    pred_aggregated = aggregated['prediction']

    real = np.concatenate((y_train, y_test))
    pred_federated = np.concatenate((y_train, pred_federated))
    pred_aggregated = np.concatenate((y_train, pred_aggregated))

    # Plot day mean data ----------------------------
    range_interval = 24
    real_day_average = np.convolve(real, np.ones(range_interval), 'valid') / range_interval
    pred_federated_day_average = np.convolve(pred_federated, np.ones(range_interval), 'valid') / range_interval
    pred_aggregated_day_average = np.convolve(pred_aggregated, np.ones(range_interval), 'valid') / range_interval
    y_train_day_average = np.convolve(y_train, np.ones(range_interval), 'valid') / range_interval
    #y_train_day_average = y_train_day_average[range_interval_y_train+range_interval:]

    #plt.figure(figsize=(15, 6))
    #plt.xticks(to_show, labels)

    

    labels = range(-30, 60, 5)

    to_show = [x*5*24 -12 for x in range(18)]

    plt.figure(figsize=(20, 6))
    axes = plt.gca()
    axes.title.set_size(14)
    axes.xaxis.label.set_size(14)
    axes.yaxis.label.set_size(14)
    plt.xticks( to_show, labels)
    plt.ylabel('Wind Energy Production (MW)')
    plt.xlabel('Days')
    plt.plot(real_day_average)
    plt.plot(pred_federated_day_average, 'r' )
    plt.plot(pred_aggregated_day_average, 'g--' )
    plt.plot(y_train_day_average, 'tab:blue')
    plt.legend(['Real data', 'ACo-L Prediction', 'GL Prediction'], loc='upper right', fontsize=12)
    plt.savefig(f'results/predicitions.png', bbox_inches='tight',pad_inches = 0.05)

if __name__ == "__main__":
    
    #history()
    plot_prediction()
    