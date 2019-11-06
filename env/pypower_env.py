import os
import numpy as np
import random
import pickle
from pypower.api import ppoption, case39
from env.custom_runpf import runpf


class Env(object):
    """
        Env of pypower calculation.
    """
    def __init__(self, dataset='case39', rand=False, thread=None):
        self.path = 'env/data/{}/data.pkl'.format(dataset)
        if not os.path.exists(self.path):
            raise ValueError("There are no dateset named '{}' \
                in the path of 'end/data'".format(dataset))
        with open(self.path, 'rb') as fp:
            self.ppc = case39()
            self.ppopt = ppoption(PF_ALG=1, VERBOSE=0)
            self.dataset = pickle.load(fp)
            
            success = np.array(self.dataset['success'])
            self.data_index = np.where(success == 0)[0]
            self.gen_index = self.ppc['gen'][:, 0]
        
        self.rand = rand
        self.capacity = len(self.data_index)
        self.fix_data_index = random.randint(0, self.capacity - 1)
        self.cnt = 0

        if thread != None:
            assert type(thread) == int
            self.thread = thread
        
        self.action_space = self.ppc['gen'].shape[0] * 2 * 4
        self.state_dim = (self.ppc['bus'].shape[0], 4)
        self.value = [-50, -20, -10, -5, 5, 10, 20, 50]
    
    def reset(self, index=None):
        """
            Reset the state.
            @param:
            index: set the data index from outside.
        """
        if index != None:
            self.cnt = index
        if self.rand:
            self.cnt = (self.cnt + 1) % self.capacity
        self.data = self.dataset['gen'][self.data_index[self.cnt]]
        self.ppc['bus'] = self.dataset['bus'][self.data_index[self.cnt]]
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.state[:, :2] = self.ppc['bus'][:, 2:4]
        for i, v  in enumerate(self.gen_index):
            self.state[int(v - 1)][2] = self.data[i][1]
            self.state[int(v - 1)][3] = self.data[i][2]
        return self.state, self.cnt
    
    def step(self, action):
        """
            Get an action from agent. \n
            Return the state, reward and next state after this action \n
            @returns:\n
            next_state, reward, done
        """
        index = action >> 3
        value = self.value[action % 8]
        pg = self.data[index][1]
        qg = self.data[index][2]
        self.data[index][1] += value
        if qg != 0:
            self.data[index][2] += value * qg / pg
        self.ppc['gen'] = self.data
        for i, v  in enumerate(self.gen_index):
            self.state[int(v - 1)][2] = self.data[i][1]
            self.state[int(v - 1)][3] = self.data[i][2]
        success, normF = runpf(self.ppc, self.ppopt)
        if success == 1:
            return self.state, 10, True
        else:
            return self.state, -np.log(normF), False
    
    def get_action(self, action):
        """
            translate the action into the dict to modify LF.** files.
            @param:
            action: the action value;
            @return:
            stateChange: the dict of which value in state should be modified and how to be
            reback_stateChange: Undo action
            {
                index: the index of state
                value: 0/1 for acLines and value to be added for generators
                node: AC / generator
            }
        """
        return {
            'index': action >> 3,
            'value': self.value[action % 8]
        }