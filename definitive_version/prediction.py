import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, f1_score
from math import sqrt
from federated import wind_data

#labels = [x for x in range(0,90)]
#to_show = [x*5-5 for x in range(0,30)]

#labels = [x*5 -30 for x in range(0,90)]
#to_show = [x*5 for x in range(0,19)]


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


def prediction_aux3(model, x_train, n):
  result = x_train[-1]
  aux = x_train
  
  for i in range(n):
    value = model.predict(aux, verbose=0)
    #print(f'Valor predicho: {value[-1][0]}')
    result = np.append(result, value[-1][0])
    seq= result[-steps:]
    seq = [ [x] for x in seq]
    #print(seq)
    #print(aux)
    aux = np.concatenate(  (aux, [seq])  )
    #print(seq)


  return result[steps:]


def prediction_aux(model, x_train, n):
  results = []
  aux = x_train[-1]
  
  for i in range(n):
    predict = model.predict(aux)
    aux = np.append(aux, [ x[0] for x in predict ][-steps:] )
    #results.append(predict[-1][0])

  pred = [ x[0] for x in predict ]
  print(pred)
  return predict[steps:]


def prediction2(model, name, x_train, y_train, x_test, y_test):

  predict_aux = prediction_aux2(model, x_train, len(y_test))
  print(predict_aux)

  #print(len(y_test))
  #print(predict_aux.shape)
  #range_interval_y_train = -24*7
  #y_train_aux = y_train[range_interval_y_train:]

  real = np.concatenate((y_train, y_test))
  predict = np.concatenate((y_train, predict_aux))

  #print(real.shape)
  #print(predict.shape)

  # Plot original data ----------------------------
  plt.figure(figsize=(15, 6))
  #plt.xticks(to_show)
  plt.ylabel('Wind Energy Production (MW)')
  plt.xlabel('Day')
  plt.plot(real)
  plt.plot(predict, 'r' )
  plt.plot(y_train)
  
  plt.savefig(f'results/predicition_{name}.png')

  plt.figure()

  
  # Plot day mean data ----------------------------
  range_interval = 24
  real_day_average = np.convolve(real, np.ones(range_interval), 'valid') / range_interval
  predict_day_average = np.convolve(predict, np.ones(range_interval), 'valid') / range_interval
  y_train_day_average = np.convolve(y_train, np.ones(range_interval), 'valid') / range_interval
  #y_train_day_average = y_train_day_average[range_interval_y_train+range_interval:]

  plt.figure(figsize=(15, 6))
  #plt.xticks(to_show, labels)
  plt.ylabel('Wind Energy Production (MW)')
  plt.xlabel('Day')
  plt.plot(real_day_average)
  plt.plot(predict_day_average, 'r' )
  plt.plot(y_train_day_average)
  
  plt.savefig(f'results/predicition_{name}_day.png')

  plt.figure()

  # Plot week mean data ----------------------------
  range_interval = 24*7
  real_week_average = np.convolve(real, np.ones(range_interval), 'valid') / range_interval
  predict_week_average = np.convolve(predict, np.ones(range_interval), 'valid') / range_interval
  y_train_week_average = np.convolve(y_train, np.ones(range_interval), 'valid') / range_interval
  #y_train_week_average = y_train_week_average[range_interval_y_train+range_interval:]

  plt.figure(figsize=(15, 6))
  #plt.xticks(to_show, labels)
  plt.ylabel('Wind Energy Production (MW)')
  plt.xlabel('Day')
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

  result = [float(x) for x in pred]
  to_write = {'prediction': result}
  with open(f'results/prediction/prediction_values_{name}.json', 'w') as file:
    json.dump(to_write, file, indent=4)

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

  plt.figure(figsize=(15, 6))
  plt.xticks(to_show, labels)
  plt.ylabel('Wind Energy Production (MW)')
  plt.xlabel('Days')
  plt.plot(real_week_average)
  plt.plot(predict_week_average, 'r' )
  #plt.plot(y_train_week_average)
  
  plt.savefig(f'results/predicition_{name}_week.png', bbox_inches='tight',pad_inches = 0.05)

  plt.figure()
  

if __name__ == "__main__":

    pred_global = False
  
    train_type = 'federated'
    #train_type = 'centralized_global'
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
    

    if (pred_global):

      with open('data/data_network.json') as file:
        nodes = json.load(file)

      mse = []
      rmse = []
      mae = []
      mape = []

      for node in nodes:
        x_train, y_train, x_test, y_test  = wind_data( data, node, date )

        predict_aux = model.predict(x_test)
        pred = [ x[0] for x in predict_aux ]

        errors = calculate_error(node, y_test, pred)
        mse.append(errors['mse'])
        rmse.append(errors['rmse'])
        mae.append(errors['mae'])
        mape.append(errors['mape'])

      mse = np.array(mse)
      rmse = np.array(rmse)
      mae = np.array(mae)
      mape = np.array(mape)

      mse_st_dev = np.std(mse)
      rmse_st_dev = np.std(rmse)
      mae_st_dev = np.std(mae)
      mape_st_dev = np.std(mape)


      mse = np.mean(mse)
      rmse = np.mean(rmse)
      mae = np.mean(mae)
      mape = np.mean(mape)

      results = {'mse': mse, 'mse_st_dev': mse_st_dev,
                 'rmse': rmse, 'rmse_st_dev': rmse_st_dev,
                 'mae': mae, 'mae_st_dev': mae_st_dev,
                 'mape': mape, 'mape_st_dev': mape_st_dev}

      with open(f'results/prediction/prediction_{train_type}.json', 'w') as file:
        json.dump(results, file, indent=4)

    else:

      x_train, y_train, x_test, y_test  = wind_data( data, 'ARWF1', date )
      
      y_train = y_train[-24*7:]
      #y_test = y_test[:24]

      labels = range(0, 60, 5)
      #to_show = [x-24*3 for x in range(0, 24*2*3)]

      to_show = []
      for i in range(0, len(y_test)+1):
        if i % (24*5) == 0:
          to_show.append(i)
      #y_train = []
      prediction(model, train_type, x_train, y_train, x_test, y_test)
    

     


