import os
import random
import numpy as np
import json
from TrendData import TrendData


class Env(object):
    """
        Env of power calculation.
        Return the State, Reward by action of agent.
    """
    def __init__(self, dataset='36nodes', runPath='env/run/', target='state-section', rand=False, train=True):
        """
            @params:
            36nodes: the path to the template folder, load initialize settings.
            runPath: the path to the WMLFRTMsg.exe folder
            rand: initialize random or not, default: False
            target: the target of task, 'state-section' means section trend state adjust, 
                    'state-voltage' means voltage trend state adjust
                    'convergence' means trend convergence adjust
            train: load train of test dataset; True of False; default: True
        """
        random.seed(7)
        self.path = 'env/data/{}/'.format(dataset)
        if not os.path.exists(self.path):
            raise ValueError("There are no dataset named '{}' \
                in the path of 'env/data/'".format(dataset))
        else:
            name = 'disconvergence.json' if target == 'convergence' else 'convergence.json'
            with open(self.path + name, 'r') as fp:
                self.dataset = json.load(fp)
        self.rand = rand
        self.train = train
        self.capacity = len(self.dataset['train']) if train else len(self.dataset['test'])
        self.fix_data_index = random.randint(0, self.capacity - 1)
        self.cnt = 0

        self.trendData = TrendData(self.dataset['train'][self.fix_data_index], runPath, target=target)
        self.target = target
        if target != 'state-section' and target != 'state-voltage' and target != 'convergence':
            raise ValueError("the param of target must be 'state-section' \
                or 'state-voltage' or 'convergence'\
                but get '{}' instead".format(target))
        self.action_space = self.trendData.g_len * 2 * 4 if target == 'state-section' else self.trendData.ac_len * 2
        self.state_dim = self.trendData.nodesNum
        self.acs_index_begin = self.trendData.g_len + self.trendData.l_len
        # only change for Pg of generators
        self.value = [-1, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1]

    def reset(self, index=None):
        """
            Reset the state.
            @param:
            index: set the data index from outside.
        """
        if index != None:
            self.cnt = index
        if self.rand:
            if self.train:
                self.trendData.reset(self.dataset['train'][self.cnt])
            else:
                self.trendData.reset(self.dataset['test'][self.cnt])
                
            self.cnt = (self.cnt + 1) % self.capacity
        else:
            self.trendData.reset(self.dataset['train'][self.fix_data_index])

    def step(self, action):
        """
            Get an action from agent.
            Return the state, reward and next state after this action
        """
        stateChange, reback_stateChange = self.get_action(action)
        reward, done = self.trendData.reward(stateChange, reback_stateChange)
        return self.trendData.state, reward, done

    def get_state(self):
        return self.trendData.state

    def get_action(self, action):
        """
            Only modify the mark of ACs.
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
        if self.target == 'state-section':
            return {
                'index': action >> 3,
                'value': self.value[action % 8],
                'node': 'generator'
            }, {
                'index': action >> 3,
                'value': self.value[7 - action % 8],
                'node': 'generator'
            }
        else:
            return {
                'index': action >> 1,
                'value': action % 2,
                'node': 'AC'
            }, {
                'index': action >> 1,
                'value': (action + 1) % 2,
                'node': 'AC'
            }
