from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message
from spade.template import Template
from spade import quit_spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message
import time

import numpy as np
import pickle

import write
from write import write, write_weights

import esn
from esn import test

class Consensus(CyclicBehaviour):

    def __init__(self, outer):
        self.agent = outer
        self.rounds = 0
        self.eps = self.calculate_eps()
        CyclicBehaviour.__init__(self)

    # Consenso
    # ----------------------------------------------------------------

    def do_consensus_antig(self):
        #print(self.agent.neighbors)
        w = self.agent.model.get_weights() # Weights of Node
        values = []
        for neighbor in self.agent.neighbors:
            n_weights = self.agent.neighbors.get(neighbor)
            '''
            if n_weights is not None:
                w_aux = [ np.array(row) for row in n_weights ]
                values.append(np.array(w_aux))'''
            values.append(n_weights)

        values = np.array(values)
        #print(f'values: {values}')

        addition = 0
        for v in values:
            addition += v - w

        #print(f'addition: {addition}')
        w = w + (self.eps * addition)
        
        self.agent.model.set_weights(w)


    def do_consensus(self):
        #print(self.agent.neighbors)
        w = self.agent.model.get_weights() # Weights of Node

        w_neighbors = []
        for neighbor in self.agent.neighbors:
            n_weights = self.agent.neighbors.get(neighbor)
            w_neighbors.append(n_weights)

        #w_neighbors = np.array(w_neighbors)
        #print(f'values: {values}')


        layers_nc = []
        for layer in range(len(w)):
            if layer not in layers_nc:
                addition = 0
                for v in w_neighbors:
                    addition += (v['weights'][layer]  - w[layer])

                #print(f'addition: {addition}')
                w[layer] = w[layer] + (self.eps * addition)
        
        self.agent.model.set_weights(w)


    def do_consensus_weighted(self):
        #print(self.agent.neighbors)
        w = self.agent.model.get_weights() # Weights of Node

        w_neighbors = []
        for neighbor in self.agent.neighbors:
            n_weights = self.agent.neighbors.get(neighbor)
            w_neighbors.append(n_weights)

        #w_neighbors = np.array(w_neighbors)
        #print(f'values: {values}')

        #best = 1/self.agent.loss
        best = -1
        for l in [n['loss'] for n in w_neighbors]:
            g = 1/l
            if g > best:
                best = g

        layers_nc = []
        for layer in range(len(w)):
            if layer not in layers_nc:
                addition = 0
                for v in w_neighbors:
                    goodness = (1 / v['loss']) / best
                    loss = v['loss']
                    print(f'LOSS neighbor: { loss }, LOSS node: {self.agent.loss}, GOODNESS: {goodness}, BEST: {best}')
                    addition += (v['weights'][layer]  - w[layer])  * goodness

                #print(f'addition: {addition}')
                w[layer] = w[layer] + (self.eps * addition)
        
        self.agent.model.set_weights(w)


    # ---------------------------------------------------------------

    # Valor de epsilon
    # ----------------------------------------------------------------

    def calculate_eps(self):
        laplacian = np.diag(np.sum(self.agent.A, axis = 1)) - self.agent.A
        eps = 1 / np.max(np.diag(laplacian))
        return eps

    # ----------------------------------------------------------------


    async def message_to(self, id, message):
        msg = Message(to=id)                        # Instantiate the message
        msg.set_metadata("performative", "inform")  # Set the "inform" FIPA performative
        msg.body = message                          # Set the message content
        await self.send(msg)


    async def run(self):

        if (self.agent.consensus_activated):

            if self.agent.all_weights_prepared:
                if self.rounds < self.agent.rounds_consensus:
                    self.rounds +=1
                    self.do_consensus()
                    self.agent.weights_shared = False
                    self.agent.all_weights_prepared = False
                    for neighbor in self.agent.neighbors:
                        self.agent.neighbors.update( { neighbor: None } )
                    
                else:
                    self.agent.consensus_activated = False
                    self.agent.training_activated = True
                    self.rounds = 0

                    # log de pesos
                    write_weights(f'{self.agent.jid}_weights', 'a', f'Consenso {self.agent.epoch}', self.agent.model.get_weights())

                    # log de evaluación
                    test_acc, test_loss = test(self.agent.model, self.agent.test_x, self.agent.test_y)
                    write(f'{self.agent.jid}_evaluation', 'a', f'Consenso {self.agent.epoch} -> LOSS: {test_loss}\n')

                    #print(f'Consenso {self.agent.epoch}: {self.agent.jid}: {str(self.agent.model.get_weights())}')
                    print(f'Consenso {self.agent.epoch}: {self.agent.jid}')