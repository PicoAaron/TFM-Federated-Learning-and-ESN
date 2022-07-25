'''
Prueba para comprobar que el consenso se hace correctamente con los pesos de las capas
de los modelos; así como que los modelos adquieren los resultados como sus nuevos pesos
de manera correcta.

Para ello:

1) Al modelo de cada nodo se le da unos pesos iniciales:
    Nodo 0: todos sus pesos son 0.2
    Nodo 1: todos sus pesos son 0.4
    Nodo 2: todos sus pesos son 0.6
    Nodo 3: todos sus pesos son 0.8

2) Se ha desactivado el entrenamiento, por lo que solo se realiza una ronda de consenso
tras otro, siendo el algoritmo de consenso lo único que varía los pesos de cada nodo


Resultado esperado:
    Con lo descrito, la media de los pesos (de cada valor de las matrices de pesos de las
    diferentes capas) es de 0.5.

    Se considera que está prueba pasa satisfactoriamente si los pesos de los diferentes
    nodos acaban en un valor cercano al 0.5.

    El resultado del consenso no llega a la media exacta por no tener en cuenta la
    sincronización entre los nodos.

'''

import time

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message
from spade.template import Template
from spade import quit_spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import consenso
from consenso import Consensus

import communication
from communication import Communication

import esn
from esn import ESN

import dataset
from dataset import sequence

import write
from write import write_weights, write_evaluation

import pandas as pd


class NodeAgent(Agent):
    
    def __init__(self, jid, password, n, epochs, rounds_consensus, neighbors, A, x, y, initial_weights):
        self.jid = jid
        self.n = n
        self.epoch = -1
        self.max_epochs = epochs
        self.round = 0
        self.rounds_consensus = rounds_consensus
        self.neighbors={}
        for name in neighbors:
            self.neighbors.update( {name: None} )
        self.A = A
        self.training_activated = True
        self.consensus_activated = False
        self.all_weights_prepared = False
        self.weights_shared = False
        self.x = x
        self.y = y
        self.test_x = x
        self.test_y = y
        self.global_test_x = x
        self.global_test_y = y
        self.initial_weights = initial_weights
        self.finished = False
        self.loss = -1
        Agent.__init__(self, jid, password)


    # A partir de una estructura de pesos y un número,
    # devuelve la misma estructura pero con todos los
    # pesos iguales dicho número
    def create_weights(self, weights, n):
        for layer in weights:
            layer[:] = n
        return weights


    class Train(CyclicBehaviour):
        def __init__(self, outer):
            self.agent = outer
            self.batch_size = 1028
            CyclicBehaviour.__init__(self)

        async def run(self):
            if self.agent.training_activated:
                self.agent.epoch += 1
                
                if self.agent.epoch <= self.agent.max_epochs:
                    print(f'{self.agent.jid}: TRAINING epoch {self.agent.epoch}')
                    # No entrenamos en esta prueba, para ver si el consenso lleva los pesos
                    # de las redes a la media de los asignados al inicio, que debería ser 0.5 todos
                    self.agent.training_activated = False
                    self.agent.consensus_activated = True
                    print('ENTRENADO')
                else:
                    print(f'{self.agent.jid}: FINISHED')
                    self.agent.training_activated = False
                    self.agent.finished = True
                

    async def setup(self):
        # Creamos el modelo
        self.model = ESN()

        # Le asignamos unos pesos iniciales (para nuestra prueba del consenso)
        w_for_consensus = self.create_weights( self.model.get_weights(), self.initial_weights)
        self.model.set_weights(w_for_consensus)

        write_weights(f'{self.jid}_weights', 'w', 'Pesos inciales', self.model.get_weights())
        write_evaluation(f'{self.jid}_evaluation', 'w', 'Inicio')

        communication = Communication(self)
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(communication, template)
       
        self.train= self.Train(self)
        self.add_behaviour(self.train)
        
        self.consensus = Consensus(self)
        self.add_behaviour(self.consensus)
        

def prepare_network(n):

    data = pd.read_csv("./aemo_2018.csv", sep=',', header=0)
    '''
    A = np.array([[0, 1, 1, 0],
                  [1, 0, 1, 0],
                  [1, 1, 0, 1],
                  [0, 0, 1, 0]])'''

    A = np.array([[0, 1, 1, 0, 0, 1, 1, 0],
                  [1, 0, 1, 0, 1, 0, 1, 0],
                  [1, 1, 0, 1, 1, 1, 0, 1],
                  [0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 1, 0, 0, 1, 1, 0],
                  [1, 0, 1, 0, 1, 0, 1, 0],
                  [1, 1, 0, 1, 1, 1, 0, 1],
                  [0, 0, 1, 0, 0, 0, 1, 0]])

    agents = []
    initial_weights = [0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.7, 0.9]

    for i in range(n):
        neighbors = []
        for j in range(len(A[i])):
            if A[i][j] == 1:
                neighbors.append( f'node{j}@localhost' )

        data_x, data_y = sequence(data, i+1)

        senderagent = NodeAgent(f'node{i}@localhost', "sender_password", n, 1, 5, neighbors, A, data_x, data_y, initial_weights[i])
        agents.append(senderagent)
        senderagent.start()
        senderagent.web.start(hostname="127.0.0.1", port=f'1000{i}')

    return agents
    

if __name__ == "__main__":

    agents = prepare_network(8)

    while True:
        try:
            time.sleep(1)
            agent_status = [agent.finished for agent in agents]
            if not False in agent_status:
                print('All agents finished')
                break
        except KeyboardInterrupt:
            for agent in agents:
                agent.stop()
            break
    print("Agents finished")