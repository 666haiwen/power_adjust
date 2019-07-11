import os
import random
import pickle
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    """
        Save Transition into Memory to train.
        @param:
        capacity: the capacity of memory
        path: the path to save in order to read next time
        save: bool, true: means save whenever a new case added; false: only save when capacity is full (Again).
    """
    def __init__(self, capacity, path, save=False):
        self.capacity = capacity
        self.memory = []
        self.postion = 0
        self.cnt = 0
        self.path = path
        self.save = save
    
    def push(self, *args):
        """ Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.postion] = Transition(*args)
        self.postion = (self.postion + 1) % self.capacity
        if self.postion == self.capacity - 1 or self.save == True:
            self.__save()
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def read(self):
        """ Read the data from disk by pickel"""
        if not os.path.exists(self.path):
            return
        with open(self.path, 'rb') as fp:
            self.memory = pickle.load(fp)
            self.postion = (self.capacity - 1) if len(self.memory) == self.capacity else len(self.memory)
    
    def set_state_and_save(self, stateList):
        """
        Set state from outside and save into disk.
        """
        for state in stateList:
            self.memory.append(Transition(state, None, None, None))
        self.postion = len(stateList)
        self.__save()

    def get_state(self):
        return [m.state for m in self.memory]

    def __len__(self):
        return len(self.memory)
    
    def __save(self):
        with open(self.path, 'wb') as fp:
            pickle.dump(self.memory, fp)
