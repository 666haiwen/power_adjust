import random
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
        self.action_space = 8 * 2 * 2 * 4 # number * features * directions * values
        self.value = [-2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2]

    def reset(self):
        """
            Reset the state.
        """
        self.trendData = TrendData(self.path, self.runPath)
        self.state = self.trendData.get_state()

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

    # def set_random_sample(self, num):
    #     for i in range(num):

    #         pass

    def __set_random_action(self):
        return {
            'index': random.randint(8, 17),
            'feature': random.randint(0, 1),
            'value': random.randint(5, 20) / 10 * (random.randint(0, 1) * 2 - 1),
            'node': 'loads'
        }
