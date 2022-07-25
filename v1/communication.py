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

import codecs

class Communication(CyclicBehaviour):

    def __init__(self, outer):
        self.agent = outer
        CyclicBehaviour.__init__(self)


    async def message_to(self, id, message):
        msg = Message(to=id)                        # Instantiate the message
        msg.set_metadata("performative", "inform")  # Set the "inform" FIPA performative
        msg.body = message                          # Set the message content
        await self.send(msg)


    async def receive_weights(self):

        prepared = True
        #print(f'{self.agent.jid}: {self.agent.neighbors}')
        for neighbor in self.agent.neighbors:
            
            if self.agent.neighbors.get(neighbor) is None:
                prepared = False
                await self.message_to(neighbor, 'request')

        self.agent.all_weights_prepared = prepared


    async def run(self):
        msg = await self.receive(timeout=1)

        if msg:
            neighbor = format(msg.sender)
            print(f'{self.agent.jid}: Message received from {neighbor}') #with content: {format(msg.body)}')
            

            if format(msg.body) == 'request':
                #w = [ a.tolist() for a in self.agent.x ]
                message = [self.agent.model.get_weights(), self.agent.loss]
                pickled = codecs.encode(pickle.dumps(message), "base64").decode()
                await self.message_to(neighbor, pickled)
                
            
            else:
                message = pickle.loads(codecs.decode(format(msg.body).encode(), "base64"))
                weights = message[0]
                loss = message[1]
                #weights = pickle.loads( format(msg.body) )
                #weights = []
                #for row in weights_aux:
                    #print(row)
                    #weights.append(np.array(row))
                #weights = np.array(weights)

                #print('Actualizando pesos')
                #print(weights)
                self.agent.neighbors.update( { neighbor: {'weights': weights, 'loss': loss} } )
                #print(f'{self.agent.jid}: {self.agent.neighbors}')
        
        elif self.agent.all_weights_prepared == False:
            await self.receive_weights()