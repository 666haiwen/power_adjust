import random
import numpy as np
from const import CFG
from TrendData import TrendData


class action(object):
    def __init__(self):
        # self.G_Num = 8
        # self.L_Num = 10
        # self.Features = 2
        self.G_Num = 0
        self.L_Num = 1
        self.Features = 1


class Env(object):
    """
        Env of power calculation.
        Return the State, Reward by action of agent.
    """
    def __init__(self, path='template/', runPath='run/', rand=False):
        self.path = path
        self.runPath = runPath
        self.rand = rand
        # self.action_space = 18 * 2 * 2 # 18 nodes, pg/qg , +/-
        self.trendData = TrendData(self.path, self.runPath)
        self.action_space = self.trendData.g_len * 2 * 2 * 4 # number * features * directions * values
        self.state_dim = self.trendData.nodesNum
        self.value = [-2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2]

    def reset(self):
        """
            Reset the state.
        """
        self.state = self.trendData.set_state_from_files(self.path).copy()

    def random_reset(self, index=None):
        """
            Reset the env randomly.
            @param:
            index: the index of the state to reset, dafault: None(random set)
        """
        if index == None:
            index = random.randint(0, self.state_num - 1)
        self.state = self.stateList[index]
        self.trendData.set_state(self.state)
        return index

    def step(self, action):
        """
            Get an action from agent.
            Return the state, reward and next state after this action
        """
        stateChange = self.get_action(action)
        self.state[0][stateChange['feature']][stateChange['index']] += stateChange['value']
        return self.trendData.reward(stateChange)

    def get_state(self):
        return self.state

    def get_action(self, action):
        """
        value: 8 actions
        features: 2 actions
        index: generators num + loads num
        """
        return {
            'index': action >> 4,
            'feature': action >> 3 & 0x1,
            'value': self.value[action % 8],
            'node': 'loads' if action >= 8 * 16 else 'generators'
        }
    
    def load_data(self, path):
        """
            Load data into stateList.
            @param:
            path: the path to the data file
        """
        self.stateList = np.load(path).astype(np.float32)
        self.state_num = len(self.stateList)

    def get_random_sample(self):
        """
            get state list
        """
        return self.stateList
