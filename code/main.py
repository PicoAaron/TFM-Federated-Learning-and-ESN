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

import train
from train import Train

import esn
from esn import ESN, test

import dataset
from dataset import adjacency, adjacency_radius, wind_data, sequence_many

import write
from write import write, write_evaluation, write_weights

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import json


class NodeAgent(Agent):
    
    def __init__(self, jid, password, n, epochs, rounds_consensus, neighbors, A, x, y, test_x, test_y, global_test_x, global_test_y):
        self.jid = jid
        self.n = n
        self.epoch = -1
        self.max_epochs = epochs
        self.round = 0
        self.rounds_consensus = rounds_consensus
        self.neighbors={}
        for name in neighbors:
            self.neighbors.update( {name: None} )
        # For evaluate against cons weighted--------
        self.neighbors_cons_w = {}
        for name in neighbors:
            self.neighbors_cons_w.update( {name: None} )
        #-------------------------------------------
        self.A = A
        self.training_activated = True
        self.consensus_activated = False
        self.all_weights_prepared = False
        self.weights_shared = False
        self.x = x
        self.y = y
        self.test_x = test_x
        self.test_y = test_y
        self.global_test_x = global_test_x
        self.global_test_y = global_test_y
        self.finished = False
        self.loss = -1
        self.loss_cons_w = -1
        self.saved_history={'loss': [], 'val_loss': []}
        self.saved_history_cons_w={'loss': [], 'val_loss': []}
        self.saved_history_no_cons={'loss': [], 'val_loss': []}
        Agent.__init__(self, jid, password)
                

    async def setup(self):
        self.model = ESN()
        #self.model_cons_w = ESN()
        self.model_no_cons = ESN()
        
        #self.model_cons_w.set_weights( self.model.get_weights() )
        self.model_no_cons.set_weights( self.model.get_weights() )

        #write(f'{self.jid}_train', 'w', self.x)
        #write(f'{self.jid}_test', 'w', self.test_x)
        write_weights(f'{self.jid}_weights', 'w', 'Pesos inciales', self.model.get_weights())

        #test_acc, test_loss = test(self.model, self.test_x, self.test_y)
        #self.loss = test_loss
        #write(f'{self.jid}_evaluation', 'w', f'Inicio -> LOSS: {test_loss}\n')
        write_evaluation(f'{self.jid}_evaluation', 'w', f'Inicio\n----------------\n')

        communication = Communication(self)
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(communication, template)
       
        self.train= Train(self)
        self.add_behaviour(self.train)
        
        self.consensus = Consensus(self)
        self.add_behaviour(self.consensus)
        

def prepare_network_antig(n):

    data = pd.read_csv("../data/aemo_2018.csv", sep=',', header=0)

    test_x, test_y = sequence_many(data, 15, 30)

    A = np.array([[0, 1, 1, 0],
                  [1, 0, 1, 0],
                  [1, 1, 0, 1],
                  [0, 0, 1, 0]])

    agents = []
    names = ['GULLRWF1', 'GUNNING1', 'HALLWF1', 'HALLWF2', 'HDWF1']
    for i in range(4):
        neighbors = []
        for j in range(len(A[i])):
            if A[i][j] == 1:
                neighbors.append( f'node{j}@localhost' )

        #data_x, data_y = sequence_many(data, (i+1)*3, ((i+1)*3)+3)
        data_x, data_y, d, c = wind_data( data, names[i], 80 )

        print(neighbors)
        senderagent = NodeAgent(f'node{i}@localhost', "sender_password", n, 10, 10, neighbors, A, data_x, data_y, test_x, test_y)
        agents.append(senderagent)
        senderagent.start()
        senderagent.web.start(hostname="127.0.0.1", port=f'1000{i}')

    return agents


def prepare_network(n):
    
    data = pd.read_csv("../data/aemo_2018.csv", sep=',', header=0)

    with open('../data/data_network.json') as file:
        data_network = json.load(file)

    agents = []

    A, neighbors_list, pos = adjacency_radius(data_network, 4)
    '''A = np.array([[0, 1, 1, 0],
                  [1, 0, 1, 0],
                  [1, 1, 0, 1],
                  [0, 0, 1, 0]])
    
    neighbors_dict = {
                        'ARWF1': ['baldhwf1@localhost', 'bluff1@localhost'],
                        'BALDHWF1': ['arwf1@localhost', 'bluff1@localhost'],
                        'BLUFF1': ['arwf1@localhost', 'baldhwf1@localhost', 'bocorwf1@localhost'],
                        'BOCORWF1': ['bluff1@localhost']
    }'''

    #print(A)

    global_test_x, global_test_y = sequence_many(data, 5, 30)

    for node in data_network:
        #neighbors = neighbors_dict[node]
        neighbors = []
        for neighbor in neighbors_list[node]:
            neighbors.append( f'{neighbor.lower()}@localhost' )
        #print(neighbors)
        
        x_train, y_train, x_test, y_test  = wind_data( data, node, 80 )

        

        #print(f'x_train: {len(x_train)}')
        #print(f'x_test: {len(x_test)}')

        #print(f'{node.lower()}: {neighbors}')
        
        senderagent = NodeAgent(f'{node.lower()}@localhost', "sender_password", n, 5, 15, neighbors, A, x_train, y_train, x_test, y_test, global_test_x, global_test_y)
        agents.append(senderagent)
        senderagent.start()
        #senderagent.web.start(hostname="127.0.0.1", port=f'1000{i}')

    return agents



if __name__ == "__main__":

    agents = prepare_network(4)
    print()
    print('Nodes started')
    print()

    while True:
        try:
            time.sleep(1)
            agent_status = [agent.finished for agent in agents]
            if not False in agent_status:
                print('All agents finished')
                for agent in agents:
                    plt.plot(agent.saved_history['loss'])
                    plt.plot(agent.saved_history['val_loss'])
                    plt.title('model loss')
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'test'], loc='upper left')
                    plt.savefig(f'../logs/images/all')
                
                for agent in agents:
                    plt.figure()
                    plt.plot(agent.saved_history['val_loss'])
                    #plt.plot(agent.saved_history_cons_w['val_loss'])
                    plt.plot(agent.saved_history_no_cons['val_loss'])
                    plt.title('model val_loss')
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    #plt.legend(['Consensus', 'Consensus Weighted', 'No Consensus'], loc='upper left')
                    plt.legend(['Consensus', 'No Consensus'], loc='upper left')
                    plt.savefig(f'../logs/images/{agent.jid}')

                break


        except KeyboardInterrupt:
            for agent in agents:
                agent.stop()
            break

    print("Agents finished")