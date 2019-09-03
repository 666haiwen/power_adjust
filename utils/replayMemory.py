import os
import random
import pickle
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory(object):
    """
        Save Transition into Memory to train.
        @param:
        capacity: the capacity of memory
        path: the path to save in order to read next time
        s: bool, true: means save whenever a new case added; false: only save when capacity is full (Again).
    """
    def __init__(self, capacity, path, s=False):
        self.capacity = capacity
        self.memory = []
        self.postion = 0
        self.cnt = 0
        self.path = path
        self.s = s
    
    def reset(self):
        self.memory = []
        self.postion = 0
        self.cnt = 0

    def push(self, *args):
        """ Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.postion] = Transition(*args)
        self.postion = (self.postion + 1) % self.capacity
        if self.postion == self.capacity - 1 or self.s == True:
            self.save()
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def read(self, memory=None, rate=1):
        """ Read the data from disk by pickel"""
        if (not os.path.exists(self.path)) and (memory==None):
            return
        if memory != None:
            self.memory = []
            self.postion = 0
            for v in memory:
                if v.reward > 0:
                    self.push(v.state, v.action, v.next_state, v.reward, v.done)
            self.save()
        else:
            with open(self.path, 'rb') as fp:
                data = pickle.load(fp)
                self.memory = data['memory']
                self.postion = data['position']
                if rate < 1:
                    length = int(len(self.memory) * rate)
                    self.memory = self.memory[:length]
                    self.postion = len(self.memory)
    
    def set_state_and_save(self, stateList):
        """
        Set state from outside and save into disk.
        """
        for state in stateList:
            self.memory.append(Transition(state, None, None, None))
        self.postion = len(stateList)
        self.save()

    def get_state(self):
        return [m.state for m in self.memory]

    def translate_type(self):
        memory = [None for i in range(len(self.memory))]
        for i in range(len(self.memory)):
            memory[i] = Transition(self.memory[i].state[:,:,:18], self.memory[i].action, self.memory[i].next_state[:,:,:18], self.memory[i].reward, self.memory[i].done)
        self.memory = memory
        self.postion = len(memory) % self.capacity
        self.save()

    def save(self):
        with open(self.path, 'wb') as fp:
            pickle.dump({
                'memory': self.memory,
                'position': self.postion
            }, fp)

    def __len__(self):
        return len(self.memory)
