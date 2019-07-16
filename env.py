import random
import numpy as np
from const import CFG, FEATURENS_NUM
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
    def __init__(self, path='template/', runPath='run/'):
        self.path = path
        self.runPath = runPath
        # self.action_space = 18 * 2 * 2 # 18 nodes, pg/qg , +/-
        self.action_space = CFG.DATA.GENERATORS * FEATURENS_NUM * 2 * 4 # number * features * directions * values
        self.value = [-2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2]
        self.trendData = TrendData(self.path, self.runPath)

    def reset(self, state=None):
        """
            Reset the state.
        """
        if state == None:
            self.trendData = TrendData(self.path, self.runPath)
            self.state = self.trendData.get_state()
        else:
            self.state = state
            self.trendData.set_state(state)

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
            'index': action >> 3,
            'feature': 0,
            'value': self.value[action % 8],
            'node': 'generators'
        }

    def set_random_sample_and_save(self, num, path=None):
        """
        set random init env and save into disk.
        @param:
            num: the number of random init.
            path: the path to save.
        """
        state = self.trendData.get_state()
        size = state.shape
        self.stateList = np.zeros((num , ) + size, dtype=np.float32)
        for i in range(num):
            for channel in range(size[1]):
                for features in range(size[2]):
                    state[0][channel][features] = round(random.random()*10,2)
            self.stateList[i] = state.copy()
        if path != None:
            np.save(path, self.stateList)

    def load_data(self, path):
        self.stateList = np.load(path).astype(np.float32)
        self.state_num = len(self.stateList)

    def random_reset(self, index=None):
        """
            Reset the env randomly.
            @param:
            index: the index of the state to reset, dafault: None(random set)
        """
        if index == None:
            index = random.randint(0, len(self.stateList) - 1)
        self.state = self.stateList[index]
        self.trendData.set_state(self.state)
        return index

    def get_random_sample(self):
        """
            get state list
        """
        return self.stateList
