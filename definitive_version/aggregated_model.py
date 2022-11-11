import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, f1_score
from math import sqrt
from federated import wind_data


def calculate_error(name, y_test, pred):
  mse = mean_squared_error(
                y_true = y_test,
                y_pred = pred
            )

  rmse = sqrt (mse)

  mae = mean_absolute_error(
                y_true = y_test,
                y_pred = pred
            )

  mape = mean_absolute_percentage_error(
                y_true = y_test,
                y_pred = pred
            )

  print(f"Error de test (mse): {mse}")
  print(f"Error de test (rmse): {rmse}")
  print(f"Error de test (mae): {mae}")
  print(f"Error de test (mape): {mape}")

  errors = {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}

  return errors
  

def prediction(model, name, x_train, y_train, x_test ,y_test):

  predict_aux = model.predict(x_test)
  pred = [ x[0] for x in predict_aux ]

  errors = calculate_error(name, y_test, pred)

  with open(f'results/prediction_{name}.json', 'w') as file:
      json.dump(errors, file, indent=4)

  real = y_test#np.concatenate((y_train, y_test))
  predict = pred#np.concatenate((y_train, pred))

  #print(real.shape)
  #print(predict.shape)

  


  # Plot original data ----------------------------
  #plt.figure(figsize=(15, 6))
  plt.xticks(to_show, labels)
  plt.ylabel('Wind Energy Production (MW)')
  plt.xlabel('Days')
  plt.plot(real)
  plt.plot(predict, 'r' )
  #plt.plot(y_train)
  
  plt.savefig(f'results/predicition_{name}.png', bbox_inches='tight',pad_inches = 0.05)

  plt.figure()

  
  # Plot day mean data ----------------------------
  range_interval = 24
  real_day_average = np.convolve(real, np.ones(range_interval), 'valid') / range_interval
  predict_day_average = np.convolve(predict, np.ones(range_interval), 'valid') / range_interval
  y_train_day_average = np.convolve(y_train, np.ones(range_interval), 'valid') / range_interval
  #y_train_day_average = y_train_day_average[range_interval_y_train+range_interval:]

  #plt.figure(figsize=(15, 6))
  #plt.xticks(to_show, labels)

  
  axes = plt.gca()
  axes.title.set_size(14)
  axes.xaxis.label.set_size(14)
  axes.yaxis.label.set_size(14)
  plt.xticks( to_show, labels)
  plt.ylabel('Wind Energy Production (MW)')
  plt.xlabel('Days')
  plt.plot(real_day_average)
  plt.plot(predict_day_average, 'r' )
  #plt.plot(y_train_day_average)
  plt.legend(['Real data', 'Predicted data'], loc='upper right', fontsize=12)
  plt.savefig(f'results/predicition_{name}_day.png', bbox_inches='tight',pad_inches = 0.05)

  plt.figure()

  # Plot week mean data ----------------------------
  range_interval = 24*7
  real_week_average = np.convolve(real, np.ones(range_interval), 'valid') / range_interval
  predict_week_average = np.convolve(predict, np.ones(range_interval), 'valid') / range_interval
  y_train_week_average = np.convolve(y_train, np.ones(range_interval), 'valid') / range_interval
  #y_train_week_average = y_train_week_average[range_interval_y_train+range_interval:]

  #plt.figure(figsize=(15, 6))
  plt.xticks(to_show, labels)
  plt.ylabel('Wind Energy Production (MW)')
  plt.xlabel('Days')
  plt.plot(real_week_average)
  plt.plot(predict_week_average, 'r' )
  #plt.plot(y_train_week_average)
  
  plt.savefig(f'results/predicition_{name}_week.png', bbox_inches='tight',pad_inches = 0.05)

  plt.figure()
  

if __name__ == "__main__":

    global_pred = False

    train_type = 'centralized_individual'

    with open(f'model/{train_type}/parameters.json') as file:
        parameters = json.load(file)

    train_type = parameters['train_type']
    steps = parameters['steps']
    date = parameters['separation_date']

    with open('data/data_network.json') as file:
        nodes = json.load(file)
    
    #nodes =  {'ARWF1': [], 'BLUFF1': []}

    model = {}
    for node in nodes:
      node_model = keras.models.load_model(f'model/{train_type}/experiment_1/{train_type}_{node}.h5')
      model.update({node: node_model})


    data = pd.read_csv("data/aemo_2018_mean_hour.csv", sep=',', header=0)

    mse = []
    rmse = []
    mae = []
    mape = []

    if global_pred:

        for node in nodes:
            x_train, y_train, x_test, y_test  = wind_data( data, node, date )
            
            pred = []
            for node in nodes:
                predict_aux = model[node].predict(x_test)
                aux = [ x[0] for x in predict_aux ]
                pred.append(aux)

            #print(pred)
            result = []

            for i in range(len(pred[0])):
                result.append( np.mean( [ x[i] for x in pred ] ) )

            pred = result
            #print(pred)
                
            errors = calculate_error(node, y_test, pred)
            mse.append(errors['mse'])
            rmse.append(errors['rmse'])
            mae.append(errors['mae'])
            mape.append(errors['mape'])

        mse = np.array(mse)
        rmse = np.array(rmse)
        mae = np.array(mae)
        mape = np.array(mape)

        mse = np.mean(mse)
        rmse = np.mean(rmse)
        mae = np.mean(mae)
        mape = np.mean(mape)

        results = {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}

        with open(f'results/prediction/prediction_aggregated.json', 'w') as file:
            json.dump(results, file, indent=4)

    else:

        x_train, y_train, x_test, y_test  = wind_data( data, 'ARWF1', date )
            
        pred = []
        for node in nodes:
            predict_aux = model[node].predict(x_test)
            aux = [ x[0] for x in predict_aux ]
            pred.append(aux)

        #print(pred)
        result = []

        for i in range(len(pred[0])):
            result.append( np.mean( [ x[i] for x in pred ] ) )

        result = [float(x) for x in result]
        #print(result)

        to_write = {'prediction': result}
        with open(f'results/prediction/prediction_values_aggregated.json', 'w') as file:
            json.dump(to_write, file, indent=4)

    

     


