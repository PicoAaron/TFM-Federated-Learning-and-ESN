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
from dataset import sequence, sequence_many, adjacency, wind_data

import write
from write import write, write_weights

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import json


class NodeAgent(Agent):
    
    def __init__(self, jid, password, n, epochs, rounds_consensus, neighbors, A, x, y, test_x, test_y):
        self.jid = jid
        self.n = n
        self.epoch = -1
        self.max_epochs = epochs
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
        self.test_x = test_x
        self.test_y = test_y
        self.finished = False
        self.loss = -1
        Agent.__init__(self, jid, password)
                

    async def setup(self):
        self.model = ESN()
        
        write_weights(f'{self.jid}_weights', 'w', 'Pesos inciales', self.model.get_weights())

        #test_acc, test_loss = test(self.model, self.test_x, self.test_y)
        #self.loss = test_loss
        #write(f'{self.jid}_evaluation', 'w', f'Inicio -> LOSS: {test_loss}\n')
        write(f'{self.jid}_evaluation', 'w', f'Inicio')

        communication = Communication(self)
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(communication, template)
       
        self.train= Train(self)
        self.add_behaviour(self.train)
        
        self.consensus = Consensus(self)
        self.add_behaviour(self.consensus)
        

def prepare_network_antig(n):

    data = pd.read_csv("./aemo_2018.csv", sep=',', header=0)

    test_x, test_y = sequence_many(data, 15, 30)

    A = np.array([[0, 1, 1, 0],
              [1, 0, 1, 0],
              [1, 1, 0, 1],
              [0, 0, 1, 0]])

    agents = []

    for i in range(4):
        neighbors = []
        for j in range(len(A[i])):
            if A[i][j] == 1:
                neighbors.append( f'node{j}@localhost' )

        data_x, data_y = sequence_many(data, (i+1)*3, ((i+1)*3)+3)

        print(neighbors)
        senderagent = NodeAgent(f'node{i}@localhost', "sender_password", n, 10, 5, neighbors, A, data_x, data_y, test_x, test_y)
        agents.append(senderagent)
        senderagent.start()
        senderagent.web.start(hostname="127.0.0.1", port=f'1000{i}')

    return agents


def prepare_network(n):
    
    data = pd.read_csv("./aemo_2018.csv", sep=',', header=0)

    test_x, test_y = sequence_many(data, 15, 30)

    with open('data_network.json') as file:
        data_network = json.load(file)

    A, neighbors_list = adjacency(data_network, 5)

    agents = []

    for node in data_network:

        neighbors = []
        #print(neighbors)
        for neighbor in neighbors_list[node]:
            neighbors.append( f'{neighbor.lower()}@localhost' )

        data_x, data_y = wind_data( data, node )

        #print(f'{node.lower()}: {neighbors}')
        
        senderagent = NodeAgent(f'{node.lower()}@localhost', "sender_password", n, 5, 1, neighbors, A, data_x, data_y, test_x, test_y)
        agents.append(senderagent)
        senderagent.start()
        #senderagent.web.start(hostname="127.0.0.1", port=f'1000{i}')

    print('3')

    return agents

if __name__ == "__main__":

    agents = prepare_network(4)

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