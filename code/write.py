import numpy as np

def write_weights(name, mode, title, weights):
  with open(f'logs/weights/{name}.txt', mode) as f:
    f.write(f'{title}:\n')
    f.write('------------------------------------------------------------------------\n\n')
    for layer in range(len(weights)):
        f.write(f'Layer {layer}:\n')
        f.write(f'{weights[layer]}\n\n')
    f.write('\n\n')


def write_evaluation(name, mode, text):
  with open(f'logs/evaluation/{name}.txt', mode) as f:
    f.write(f'{text}\n')

def write(name, mode, text):

  with open(f'logs/data/{name}.txt', mode) as f:
    with np.printoptions(threshold=np.inf):
      f.write(f'{text}\n')