import os
import random
import json
import numpy as np
from shutil import copyfile
from env.TrendData import TrendData


class Env(object):
    """
        Env of power calculation.
        Return the State, Reward by action of agent.
    """
    def __init__(self, dataset='36nodes', runPath='env/run/', target='state-section', 
                classifer_model=None, rand=False, thread=None):
        """
            @params:
            36nodes: the path to the template folder, load initialize settings.
            runPath: the path to the WMLFRTMsg.exe folder
            rand: initialize random or not, default: False
            target: the target of task, 'state-section' means section trend state adjust, 
                    'state-voltage' means voltage trend state adjust
                    'convergence' means trend convergence adjust
            classifer_model: model of classifer instead WML.exe, default None;
            train: load train of test dataset; True of False; default: True
            thread: id of distributed trainning env Id, default: None, means not thread env
        """
        self.path = 'env/data/{}/'.format(dataset)
        if not os.path.exists(self.path):
            raise ValueError("There are no dataset named '{}' \
                in the path of 'env/data/'".format(dataset))
        else:
            name = 'disconvergence.json' if target == 'convergence' else 'convergence.json'
            with open(self.path + name, 'r') as fp:
                self.dataset = json.load(fp)
                self.dataset = self.dataset['train'] + self.dataset['test']
        self.rand = rand
        self.classifer_model = classifer_model
        self.capacity = len(self.dataset)
        self.fix_data_index = random.randint(0, self.capacity - 1)
        self.cnt = 0

        if thread != None:
            assert type(thread) == int
            runPath = 'env/thread_run/actor_{}/'.format(thread)
            if not os.path.exists(runPath):
                os.mkdir(runPath)
                sourcePath = 'env/run/'
                files = os.listdir(sourcePath)
                for file_name in files:
                    copyfile(sourcePath + file_name, runPath + file_name)

        self.trendData = TrendData(self.dataset[self.fix_data_index], runPath, target=target)
        self.target = target
        if target != 'state-section' and target != 'state-voltage' and target != 'convergence':
            raise ValueError("the param of target must be 'state-section' \
                or 'state-voltage' or 'convergence'\
                but get '{}' instead".format(target))
        self.action_space = self.trendData.g_len * 2 * 4 if target == 'state-section' else self.trendData.ac_len
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
            self.trendData.reset(self.dataset[self.cnt])
                
            self.cnt = (self.cnt + 1) % self.capacity
        else:
            self.trendData.reset(self.dataset[self.fix_data_index])
        return self.cnt

    def step(self, action):
        """
            Get an action from agent. \n
            Return the state, reward and next state after this action\n
            @returns:\n
            next_state, reward, done
        """
        stateChange, reback_stateChange = self.get_action(action)
        reward, done = self.trendData.reward(stateChange, reback_stateChange, self.classifer_model)
        return self.trendData.state, reward, done

    def score(self):
        """
            Return current value for state-section or state-voltage
        """
        if self.target == 'state-section':
            return self.trendData.pre_value
        elif self.target == 'state-voltage':
            return self.trendData.pre_value[0]
        else:
            return None

    def get_state(self):
        return self.trendData.state

    def get_reverse(self, action):
        """
            R
        """
        if self.target == 'state-section':
            return ((action >> 3) << 3) + 7 - action % 8
        else:
            return ((action >> 1) << 1) + (action + 1) % 2

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
                'index': action,
                'node': 'AC'
            }, {
                'index': action,
                'node': 'AC'
            }
