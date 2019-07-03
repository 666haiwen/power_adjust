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
        self.action_space = 8 * 2 * 2

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
        self.state[0][0][stateChange['index']][stateChange['feature']] += stateChange['value']
        return self.trendData.reward(stateChange)

    def get_state(self):
        return self.state

    def get_action(self, action):
        """
        0: pg -0.1
        1: pg +0.1
        2: qg -0.1
        3: qg +0.1
        """
        return {
            'index': int(action / 4),
            'feature': int((action % 4) / 2),
            'value': -0.1 if action % 2 == 0 else 0.1,
            'node': 'loads' if action >= 8 * 4 else 'generators'
        }
        