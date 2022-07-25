from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message
from spade.template import Template
from spade import quit_spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message
import time

import keras
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import numpy as np

import write
from write import write, write_weights

import esn
from esn import test

class Train(CyclicBehaviour):
        def __init__(self, outer):
            self.agent = outer
            self.batch_size = 1028
            CyclicBehaviour.__init__(self)

        async def run(self):
            if self.agent.training_activated:
                self.agent.epoch += 1
                
                if self.agent.epoch <= self.agent.max_epochs:
                    print()
                    print(f'{self.agent.jid}: TRAINING epoch {self.agent.epoch}')

                    self.agent.model.fit(   self.agent.x,
                                            self.agent.y,
                                            #validation_data = (self.agent.test_x, self.agent.test_y),
                                            #batch_size=self.batch_size,
                                            #steps_per_epoch=len(self.agent.x)/self.batch_size,
                                            steps_per_epoch=100,
                                            epochs=1,
                                            verbose=1)
                    print()

                    # log de pesos
                    write_weights(f'{self.agent.jid}_weights', 'a', f'Entrenamiento {self.agent.epoch}', self.agent.model.get_weights())

                    # log de evaluación
                    test_acc, test_loss = test(self.agent.model, self.agent.test_x, self.agent.test_y)
                    write(f'{self.agent.jid}_evaluation', 'a', f'Entrenamiento {self.agent.epoch} -> LOSS: {test_loss}')

                    self.agent.loss = test_loss
                    self.agent.training_activated = False
                    self.agent.consensus_activated = True
                else:
                    print(f'{self.agent.jid}: FINISHED')
                    self.agent.training_activated = False
                    self.agent.finished = True
