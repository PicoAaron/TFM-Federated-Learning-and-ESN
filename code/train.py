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

import matplotlib.pyplot as plt

import write
from write import write_evaluation, write_weights

import esn
from esn import test

class Train(CyclicBehaviour):
        def __init__(self, outer):
            self.agent = outer
            self.batch_size = 1028
            self.steps=100
            CyclicBehaviour.__init__(self)
            

        async def run(self):
            if self.agent.training_activated:
                self.agent.epoch += 1
                
                if self.agent.epoch <= self.agent.max_epochs:
                    print()
                    print(f'{self.agent.jid}: TRAINING epoch {self.agent.epoch}')

                    history = self.agent.model.fit(   self.agent.x,
                                            self.agent.y,
                                            #validation_data = (self.agent.test_x, self.agent.test_y),
                                            #batch_size=self.batch_size,
                                            #steps_per_epoch=len(self.agent.x)/self.batch_size,
                                            steps_per_epoch=self.steps,
                                            epochs=1,
                                            verbose=1)

                    #time.sleep(0)

                    # log de evaluación
                    #test_acc, test_loss = history.history['val_accuracy'][-1], history.history['val_loss'][-1]
                    test_acc, test_loss = test(self.agent.model, self.agent.test_x, self.agent.test_y)
                    #global_test_acc, global_test_loss = test(self.agent.model, self.agent.global_test_x, self.agent.global_test_y)

                    #print( f"history: {self.agent.saved_history['loss']}")
                    loss = self.agent.saved_history['loss']
                    loss.append(history.history['loss'][-1])
                    self.agent.saved_history.update({'loss': loss})

                    val_loss = self.agent.saved_history['val_loss']
                    #val_loss.append(history.history['val_loss'][-1])
                    val_loss.append(test_loss)
                    self.agent.saved_history.update({'val_loss': val_loss})

                    global_test_acc, global_test_loss = test(self.agent.model, self.agent.global_test_x, self.agent.global_test_y)


                    # Model with  NO CONSENSUS ------------------------------------------------------------
                    history_no_cons = self.agent.model_no_cons.fit(   self.agent.x,
                                            self.agent.y,
                                            #validation_data = (self.agent.test_x, self.agent.test_y),
                                            #batch_size=self.batch_size,
                                            #steps_per_epoch=len(self.agent.x)/self.batch_size,
                                            steps_per_epoch=self.steps,
                                            epochs=1,
                                            verbose=1)

                    #time.sleep(0)

                    #test_acc_no_cons, test_loss_no_cons = history_no_cons.history['val_accuracy'][-1], history_no_cons.history['val_loss'][-1]
                    test_acc_no_cons, test_loss_no_cons = test(self.agent.model_no_cons, self.agent.test_x, self.agent.test_y)

                    loss_no_cons = self.agent.saved_history_no_cons['loss']
                    loss_no_cons.append(history_no_cons.history['loss'][-1])
                    self.agent.saved_history_no_cons.update({'loss': loss_no_cons})

                    val_loss_no_cons = self.agent.saved_history_no_cons['val_loss']
                    #val_loss_no_cons.append(history_no_cons.history['val_loss'][-1])
                    val_loss_no_cons.append(test_loss_no_cons)
                    self.agent.saved_history_no_cons.update({'val_loss': val_loss_no_cons})

                    

                    
                    #---------------------------------------------------------------------------------------

                    '''
                    # Model with  CONSENSUS WEIGHTED ------------------------------------------------------------
                    history_cons_w = self.agent.model_cons_w.fit(   self.agent.x,
                                            self.agent.y,
                                            validation_data = (self.agent.test_x, self.agent.test_y),
                                            #batch_size=self.batch_size,
                                            #steps_per_epoch=len(self.agent.x)/self.batch_size,
                                            steps_per_epoch=self.steps,
                                            epochs=1,
                                            verbose=1)

                    loss_cons_w = self.agent.saved_history_cons_w['loss']
                    loss_cons_w.append(history_cons_w.history['loss'][-1])

                    val_loss = self.agent.saved_history_cons_w['val_loss']
                    val_loss.append(history_cons_w.history['val_loss'][-1])

                    self.agent.saved_history_cons_w.update({'loss': loss_cons_w})

                    test_acc_cons_w, test_loss_cons_w = history_cons_w.history['val_accuracy'][-1], history_cons_w.history['val_loss'][-1]
                    #---------------------------------------------------------------------------------------
                    '''


                    #self.agent.saved_history.update({'val_loss': self.agent.saved_history['val_loss'].append(history.history['val_loss'][-1])})

                    # log de pesos
                    write_weights(f'{self.agent.jid}_weights', 'a', f'Entrenamiento {self.agent.epoch}', self.agent.model.get_weights())
                    #write_weights(f'{self.agent.jid}_weights', 'a', f'Entrenamiento {self.agent.epoch} WITH CONSENSUS WEIGHTED', self.agent.model_no_cons.get_weights())
                    write_weights(f'{self.agent.jid}_weights', 'a', f'Entrenamiento {self.agent.epoch} WITH NO CONSENSUS', self.agent.model_no_cons.get_weights())

                    #time.sleep(0)

                    write_evaluation(f'{self.agent.jid}_evaluation', 'a', f'Entrenamiento {self.agent.epoch}: {test_loss}')
                    write_evaluation(f'{self.agent.jid}_evaluation', 'a', f'Loss datos globales: {global_test_loss}\n')

                    #write_evaluation(f'{self.agent.jid}_evaluation', 'a', f'Entrenamiento {self.agent.epoch} WITH CONSENSUS WEIGHTED: {test_loss_cons_w}')
                    write_evaluation(f'{self.agent.jid}_evaluation', 'a', f'Entrenamiento {self.agent.epoch} WITH NO CONSENSUS: {test_loss_no_cons}\n')
                    
                    #write_evaluation(f'{self.agent.jid}_evaluation', 'a', f'LOCAL  -> LOSS: {test_loss}')
                    #write_evaluation(f'{self.agent.jid}_evaluation', 'a', f'GLOBAL -> LOSS: {global_test_loss}\n')

                    #time.sleep(0)

                    

                    self.agent.loss = global_test_loss
                    #self.agent.loss_cons_w  = test_loss_cons_w
                    
                    self.agent.training_activated = False
                    self.agent.consensus_activated = True
                else:
                    print(f'{self.agent.jid}: FINISHED')
                    
                    self.agent.training_activated = False
                    self.agent.finished = True
