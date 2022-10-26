import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, f1_score
from federated import wind_data


def prediction_aux2(model, x_train, n):
  result = x_train[-1]
  aux = x_train[-1]
  
  for i in range(n):
    #print(aux)
    aux_input = aux.reshape((1,  steps, 1))
    #print(f'Datos: {aux_input}')
    value = model.predict(aux_input, verbose=0)
    #print(f'Valor predicho: {value}')
    result = np.append(result, value[0][0])
    aux = result[-steps:]

  #print(result)
  return result[steps:]


def prediction_aux(model, x_train, n):
  results = []
  aux = x_train[-1]
  
  for i in range(n):
    predict = model.predict(aux)
    np.append([ x[0] for x in predict ][-steps:] )
    results.append(predict[-1][0])

  
  return results


def prediction2(model, name, x_train, y_train, x_test, y_test):

  predict_aux = prediction_aux(model, x_train, len(y_test))

  #print(len(y_test))
  #print(predict_aux.shape)
  #range_y_train = -24*7
  #y_train_aux = y_train[range_y_train:]

  real = np.concatenate((y_train, y_test))
  predict = np.concatenate((y_train, predict_aux))

  #print(real.shape)
  #print(predict.shape)

  # Plot original data ----------------------------
  plt.figure(figsize=(15, 6))
  plt.plot(y_test)
  plt.plot(predict_aux, 'r' )
  #plt.plot(y_train)
  
  plt.savefig(f'results/predicition_{name}.png')

  plt.figure()

  
  # Plot day mean data ----------------------------
  range = 24
  real_day_average = np.convolve(real, np.ones(range), 'valid') / range
  predict_day_average = np.convolve(predict, np.ones(range), 'valid') / range
  y_train_day_average = np.convolve(y_train, np.ones(range), 'valid') / range
  #y_train_day_average = y_train_day_average[range_y_train+range:]

  plt.figure(figsize=(15, 6))
  plt.plot(real_day_average)
  plt.plot(predict_day_average, 'r' )
  plt.plot(y_train_day_average)
  
  plt.savefig(f'results/predicition_{name}_day.png')

  plt.figure()

  # Plot week mean data ----------------------------
  range = 24*7
  real_week_average = np.convolve(real, np.ones(range), 'valid') / range
  predict_week_average = np.convolve(predict, np.ones(range), 'valid') / range
  y_train_week_average = np.convolve(y_train, np.ones(range), 'valid') / range
  #y_train_week_average = y_train_week_average[range_y_train+range:]

  plt.figure(figsize=(15, 6))
  plt.plot(real_week_average)
  plt.plot(predict_week_average, 'r' )
  plt.plot(y_train_week_average)
  
  plt.savefig(f'results/predicition_{name}_week.png')

  plt.figure()


def calculate_error(name, y_test, pred):
  mse = mean_squared_error(
                y_true = y_test,
                y_pred = pred
            )

  mae = mean_absolute_error(
                y_true = y_test,
                y_pred = pred
            )

  mape = mean_absolute_percentage_error(
                y_true = y_test,
                y_pred = pred
            )

  print(f"Error de test (mse): {mse}")
  print(f"Error de test (mae): {mae}")
  print(f"Error de test (mape): {mape}")

  errors = {'mse': mse, 'mae': mae, 'mape': mae}

  with open(f'results/prediction_{name}.json', 'w') as file:
          json.dump(errors, file, indent=4)


def prediction(model, name, x_train, y_train, x_test ,y_test):

  predict_aux = model.predict(x_test)
  pred = [ x[0] for x in predict_aux ]

  calculate_error(name, y_test, pred)

  real = np.concatenate((y_train, y_test))
  predict = np.concatenate((y_train, pred))

  #print(real.shape)
  #print(predict.shape)

  # Plot original data ----------------------------
  plt.figure(figsize=(15, 6))
  plt.plot(real)
  plt.plot(predict, 'r' )
  plt.plot(y_train)
  
  plt.savefig(f'results/predicition_{name}.png')

  plt.figure()

  
  # Plot day mean data ----------------------------
  range = 24
  real_day_average = np.convolve(real, np.ones(range), 'valid') / range
  predict_day_average = np.convolve(predict, np.ones(range), 'valid') / range
  y_train_day_average = np.convolve(y_train, np.ones(range), 'valid') / range
  #y_train_day_average = y_train_day_average[range_y_train+range:]

  plt.figure(figsize=(15, 6))
  plt.plot(real_day_average)
  plt.plot(predict_day_average, 'r' )
  plt.plot(y_train_day_average)
  
  plt.savefig(f'results/predicition_{name}_day.png')

  plt.figure()

  # Plot week mean data ----------------------------
  range = 24*7
  real_week_average = np.convolve(real, np.ones(range), 'valid') / range
  predict_week_average = np.convolve(predict, np.ones(range), 'valid') / range
  y_train_week_average = np.convolve(y_train, np.ones(range), 'valid') / range
  #y_train_week_average = y_train_week_average[range_y_train+range:]

  plt.figure(figsize=(15, 6))
  plt.plot(real_week_average)
  plt.plot(predict_week_average, 'r' )
  plt.plot(y_train_week_average)
  
  plt.savefig(f'results/predicition_{name}_week.png')

  plt.figure()
  


if __name__ == "__main__":
  
    #train_type = 'federated'
    train_type = 'centralized_global'
    #train_type = 'centralized_individual'

    with open(f'model/{train_type}/parameters.json') as file:
        parameters = json.load(file)

    train_type = parameters['train_type']
    steps = parameters['steps']
    date = parameters['separation_date']
    
    if train_type == 'centralized_individual':
      model = keras.models.load_model(f'model/{train_type}/experiment_1/{train_type}_ARWF1.h5')
    else:
      model = keras.models.load_model(f'model/{train_type}/experiment_1/{train_type}.h5')

    data = pd.read_csv("data/aemo_2018_mean_hour.csv", sep=',', header=0)
    x_train, y_train, x_test, y_test  = wind_data( data, 'ARWF1', date )
    #print(y_train)
    #print(y_test)
    
    y_train = y_train[-24*7*4:]
    #y_test = y_test[:24*3]
    '''
    y = np.concatenate((y_train, y_test))
    
    plt.figure(figsize=(15, 6))
    range = 24
    real_day_average = np.convolve(y, np.ones(range), 'valid') / range
    plt.plot(real_day_average)
    plt.savefig(f'results/prueba.png')'''
    

    prediction(model, train_type, x_train, y_train, x_test, y_test)