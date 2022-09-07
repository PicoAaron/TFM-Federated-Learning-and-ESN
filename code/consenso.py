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
from write import write_evaluation, write_weights

import esn
from esn import test, structure, structure_best

class Consensus(CyclicBehaviour):

    def __init__(self, outer):
        self.agent = outer
        self.rounds = 0
        self.eps = self.calculate_eps()
        CyclicBehaviour.__init__(self)

    # Consenso
    # ----------------------------------------------------------------

    def do_consensus_average_structure(self, process_structure=False, one_or_zero_first=False, one_or_zero_end=False):
        #print(self.agent.neighbors)
        w = self.agent.model.get_weights() # Weights of Node

        w_neighbors = []
        for neighbor in self.agent.neighbors:
            n_weights = self.agent.neighbors.get(neighbor)
            w_neighbors.append(n_weights)

        #w_neighbors = np.array(w_neighbors)
        #print(f'values: {values}')

        for layer in range(0, len(w)):
            addition = 0
            for v in w_neighbors:
                addition += (v['weights'][layer] - w[layer])

            #print(f'addition: {addition}')
            w[layer] = w[layer] + (self.eps * addition)
        
        self.agent.model.set_weights(w)

    
    def do_consensus_first_structure(self, process_structure=False, one_or_zero_first=False, one_or_zero_end=False):
        w = self.agent.model.get_weights() # Weights of Node

        w_neighbors = []
        for neighbor in self.agent.neighbors:
            n_weights = self.agent.neighbors.get(neighbor)
            w_neighbors.append(n_weights)

        layers_nc = [0, 1, 2]
        for layer in range(0, len(w)):
            if layer not in layers_nc:
                addition = 0
                for v in w_neighbors:
                    addition += (v['weights'][layer] - w[layer])

                w[layer] = w[layer] + (self.eps * addition)
        
        self.agent.model.set_weights(w)


    def do_consensus_best_structure(self, process_structure=False, one_or_zero_first=False, one_or_zero_end=False):
        w = self.agent.model.get_weights() # Weights of Node

        w_neighbors = []
        for neighbor in self.agent.neighbors:
            n_weights = self.agent.neighbors.get(neighbor)
            w_neighbors.append(n_weights)

        # We get the best structure among the neighbors
        best_loss = self.agent.loss
        best = self.agent.model.get_weights()
        for neighbor in w_neighbors:
            if neighbor['loss'] < best_loss:
                best_loss = neighbor['loss']
                best = neighbor['weights']

        layers_nc = [0]
        for layer in range(0, len(w)):
            if layer not in layers_nc:
                addition = 0
                for v in w_neighbors:
                    addition += (v['weights'][layer] - w[layer])

                w[layer] = w[layer] + (self.eps * addition)

            else:
                w[layer] = best[layer]
        
        self.agent.model.set_weights(w)


    def do_consensus_repeated_links(self, process_structure=False, one_or_zero_first=False, one_or_zero_end=False):
        w = self.agent.model.get_weights() # Weights of Node

        w_neighbors = []
        for neighbor in self.agent.neighbors:
            n_weights = self.agent.neighbors.get(neighbor)
            w_neighbors.append(n_weights)

        #if process_structure:
        # For the layer 0 (structure)
        w_all = w_neighbors.copy()
        structure_list = [x['weights'][0] for x in w_all]
        structure_list.append(w[0])

        new_structure = structure(structure_list, one_or_zero_first, one_or_zero_end)
        w[0] = new_structure

        layers_nc = []
        for layer in range(1, len(w)):
            if layer not in layers_nc:
                addition = 0
                for v in w_neighbors:
                    addition += (v['weights'][layer] - w[layer])

                w[layer] = w[layer] + (self.eps * addition)
        
        self.agent.model.set_weights(w)


    def do_consensus_best_links(self, process_structure=False, one_or_zero_first=False, one_or_zero_end=False):
        w = self.agent.model.get_weights() # Weights of Node

        w_neighbors = []
        for neighbor in self.agent.neighbors:
            n_weights = self.agent.neighbors.get(neighbor)
            w_neighbors.append(n_weights)

        #if process_structure:
        # For the layer 0 (structure)
        w_all = w_neighbors.copy()
        structure_list = [x['weights'][0] for x in w_all]
        structure_list.append(w[0])
        #print(structure_list[0])

        performance_list = [n['loss'] for n in w_neighbors]

        new_structure = structure_best(structure_list, performance_list,one_or_zero_first, one_or_zero_end)
        w[0] = new_structure

        #print('------------------')
        #print(f'JID: {self.agent.jid}')
        #print(new_structure)
        #print('------------------')

        layers_nc = []
        for layer in range(1, len(w)):
            if layer not in layers_nc:
                addition = 0
                for v in w_neighbors:
                    addition += (v['weights'][layer] - w[layer])

                w[layer] = w[layer] + (self.eps * addition)
        
        self.agent.model.set_weights(w)


    '''
    def do_consensus_antig(self):
        #print(self.agent.neighbors)
        w = self.agent.model.get_weights() # Weights of Node
        values = []
        for neighbor in self.agent.neighbors:
            n_weights = self.agent.neighbors.get(neighbor)['weights']
            
            #if n_weights is not None:
            #    w_aux = [ np.array(row) for row in n_weights ]
            #    values.append(np.array(w_aux))
            values.append(n_weights)

        values = np.array(values)
        #print(f'values: {values}')

        addition = 0
        for v in values:
            addition += v - w

        #print(f'addition: {addition}')
        w = w + (self.eps * addition)
        
        self.agent.model.set_weights(w)


    def do_consensus_2(self):
        #print(self.agent.neighbors)
        w = self.agent.model.get_weights() # Weights of Node

        w_neighbors = []
        for neighbor in self.agent.neighbors:
            n_weights = self.agent.neighbors.get(neighbor)
            w_neighbors.append(n_weights)

        #w_neighbors = np.array(w_neighbors)
        #print(f'values: {values}')

        best_loss = self.agent.loss
        best = self.agent.model.get_weights()
        for neighbor in w_neighbors:
            if neighbor['loss'] < best_loss:
                best_loss = neighbor['loss']
                best = neighbor['weights']

        layers_nc = [0, 1]
        for layer in range(len(w)):
            if layer not in layers_nc:
                addition = 0
                for v in w_neighbors:
                    addition += (v['weights'][layer] - w[layer])

                #print(f'addition: {addition}')
                w[layer] = w[layer] + (self.eps * addition)

            else:
                w[layer] = best[layer]

        
        self.agent.model.set_weights(w)


    def do_consensus_weighted(self):
        #print(self.agent.neighbors)
        w = self.agent.model_cons_w.get_weights() # Weights of Node

        w_neighbors = []
        for neighbor in self.agent.neighbors_cons_w:
            n_weights = self.agent.neighbors_cons_w.get(neighbor)
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
        
        self.agent.model_cons_w.set_weights(w)



    '''

    # ---------------------------------------------------------------

    # Valor de epsilon
    # ----------------------------------------------------------------

    def calculate_eps(self):
        laplacian = np.diag(np.sum(self.agent.A, axis = 1)) - self.agent.A
        eps = 1 / np.max(np.diag(laplacian))
        #print(f'EPS: {eps}')
        return eps
        #return 1/3

    # ----------------------------------------------------------------


    async def message_to(self, id, message):
        msg = Message(to=id)                        # Instantiate the message
        msg.set_metadata("performative", "inform")  # Set the "inform" FIPA performative
        msg.body = message                          # Set the message content
        await self.send(msg)


    async def run(self):

        if (self.agent.consensus_activated):

            if self.agent.all_weights_prepared:
                if self.agent.round < self.agent.rounds_consensus:
                    self.agent.round +=1

                    
                    if self.agent.round == 1:
                        self.do_consensus_average_structure(process_structure=True, one_or_zero_first=True)
                    elif self.agent.round == self.agent.rounds_consensus:
                        self.do_consensus_average_structure(process_structure=True, one_or_zero_end=True)
                    else:
                        self.do_consensus_average_structure()
                    
                    
                    #self.do_consensus_weighted()
                    self.agent.weights_shared = False
                    self.agent.all_weights_prepared = False
                    for neighbor in self.agent.neighbors:
                        self.agent.neighbors.update( { neighbor: None } )
                    
                else:
                    self.agent.consensus_activated = False
                    self.agent.training_activated = True
                    self.agent.round = 0

                
                    # log de pesos
                    write_weights(f'{self.agent.jid}_weights', 'a', f'Consenso {self.agent.epoch}', self.agent.model.get_weights())

                    # log de evaluación
                    test_acc, test_loss = test(self.agent.model, self.agent.test_x, self.agent.test_y)
                    gloabal_test_acc, global_test_loss = test(self.agent.model, self.agent.global_test_x, self.agent.global_test_y)

                    loss = self.agent.saved_history['loss_consensus_local']
                    loss.append(test_loss)
                    self.agent.saved_history.update({'loss_consensus_local': loss})

                    loss_global = self.agent.saved_history['loss_consensus_global']
                    loss_global.append(global_test_loss)
                    self.agent.saved_history.update({'loss_consensus_global': loss_global})

                    write_evaluation(f'{self.agent.jid}_evaluation', 'a', f'Consenso {self.agent.epoch}:')
                    write_evaluation(f'{self.agent.jid}_evaluation', 'a', f'LOCAL  -> LOSS: {test_loss}')
                    write_evaluation(f'{self.agent.jid}_evaluation', 'a', f'GLOBAL -> LOSS: {global_test_loss}\n\n----------------\n')
                    
                    #print(f'Consenso {self.agent.epoch}: {self.agent.jid}: {str(self.agent.model.get_weights())}')
                    print(f'Consenso {self.agent.epoch}: {self.agent.jid}')
            