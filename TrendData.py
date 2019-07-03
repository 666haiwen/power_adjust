import os
import subprocess
import tempfile
import time
import numpy as np
from const import FEATURENS_NUM, FEATURENS, THERESHOLD

class TrendData(object):
    """
        Load the data of TrendData from disk to memeory.
        And check that the data is reasonable or not.
    """
    def __init__(self, path='template/', runPath='run/', buses=None, generators=None, loads=None):
        self.path = path
        self.runPath = runPath
        self.buses = self.__load_buses() if buses == None else buses
        self.generators, self.g_index = self.__load_generators() if generators == None else generators
        self.loads, self.l_index = self.__load_loads() if loads == None else loads
        self.nodesNum = len(self.g_index) + len(self.l_index)
        self.g_len = len(self.g_index)
        self.l_len = len(self.l_index)
        self.tmp_out = tempfile.SpooledTemporaryFile(10*1000)
    
    def get_state(self):
        """
            Translate data into state type. shape:[nodesNum, featuresNum]
            @return:
            state: darray, shape:[nodesNum, featuresNum]
        """
        state = np.zeros((1, 1, self.nodesNum, FEATURENS_NUM), dtype=np.float32)
        for i in range(self.g_len):
            state[0][0][i][0] = self.__get_generators(i)['Pg']
            state[0][0][i][1] = self.__get_generators(i)['Qg']
            state[0][0][i][2] = self.__get_generators(i)['V0']
            state[0][0][i][4] = self.__get_generators(i)['Type']
        for i in range(self.l_len):
            state[0][0][i + self.g_len][0] = self.__get_loads(i)['Pg']
            state[0][0][i + self.g_len][1] = self.__get_loads(i)['Qg']
            state[0][0][i + self.g_len][2] = self.__get_loads(i)['V0']
            state[0][0][i + self.g_len][3] = 1
            state[0][0][i + self.g_len][4] = self.__get_loads(i)['Type']
        return state

    def set_state(self, state):
        """
            Set data from state outside.
        """
        for i in range(self.g_len):
            self.generators[self.g_index[i]]['Pg'] = state[i][0]
            self.generators[self.g_index[i]]['Qg'] = state[i][1]
            self.generators[self.g_index[i]]['V0'] = state[i][2]
            self.generators[self.g_index[i]]['Type'] = state[i][4]

        for i in range(self.l_len):
            self.loads[self.l_index[i]]['Pg'] = state[i + self.g_len][0]
            self.loads[self.l_index[i]]['Qg'] = state[i + self.g_len][1]
            self.loads[self.l_index[i]]['V0'] = state[i + self.g_len][2]
            self.loads[self.l_index[i]]['Type'] = state[i + self.g_len][4]

    def reward(self, action=None):
        """
            1.Change the trendData by action
            2.Run WMLFRTMsg.exe
            3.Get the result of convergencing or not
            @return:
            reward: 1: Convergence 0: DisConvergence -1: OutOf range.
            done: True or False
        """
        flag = self.__changeData(action) if action != None else True
        if flag:
            self.__output()
            cwd = os.getcwd()
            os.chdir(self.runPath)
            fileno = self.tmp_out.fileno()
            p =subprocess.Popen('WMLFRTMsg.exe', stdout=fileno,stderr=fileno)
            p.wait()
            self.tmp_out.seek(0)
            os.chdir(cwd)
            with open(os.path.join(self.runPath,  'LF.CAL'), 'r', encoding='gbk') as fp:
                firstLine = fp.readline()
                data = firstLine.split(',')
                if int(data[0]) != 1:
                    return 0, False
                return 1, True
        return -1, True

    def __changeData(self, action):
        """
            Change the trendData in memory by action from env.
            @return:
            flag: Out of range by threshold. True: state ok; False: Out of threshold
        """
        feature = FEATURENS[action['feature']]
        if action['node'] == 'loads':
            index = self.l_index[action['index'] - self.g_len]
            self.loads[index][feature] += action['value']
            if self.loads[index][feature] < THERESHOLD[feature][0] or \
               self.loads[index][feature] > THERESHOLD[feature][1]:
               self.loads[index][feature] -= action['value']
               return False

        elif action['node'] == 'generators':
            index = self.l_index[action['index']]
            self.generators[index][feature] += action['value']
            if self.generators[index][feature] < THERESHOLD[feature][0] or \
               self.generators[index][feature] > THERESHOLD[feature][1]:
               self.generators[index][feature] -= action['value']
               return False
        
        return True

    def __get_generators(self, index):
        return self.generators[self.g_index[index]]
    
    def __get_loads(self, index):
        return self.loads[self.l_index[index]]

    def __load_buses(self):
        """
            load buses data into memory.
        """
        buses = [None]
        with open(os.path.join(self.path,'LF.L1'), 'r', encoding='gbk') as fp:
            for line in fp:
                data = line.split(',')[:-1]
                buses.append({
                    'vBase': float(data[1])
                })
        return buses
    

    def __load_generators(self):
        """
            load generators data into memory.
        """
        generators = []
        index = []
        with open(os.path.join(self.path,'LF.L5'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                if (int(data[0]) == 1):
                    index.append(i)
                P = float(data[3])
                Q = float(data[4])
                V = float(data[5])
                PMin = float(data[10])
                PMax = float(data[9])
                QMin = float(data[8])
                QMax = float(data[7])
                generators.append({
                    'mark': int(data[0]),
                    'Type': int(data[2]),
                    'Pg': P,
                    'Qg': Q,
                    'V0': V,
                    'QMax': QMax,
                    'QMin': QMin,
                    'PMax': PMax,
                    'PMin': PMin,
                    'data': data.copy()
                })
        return generators, index
    
    def __load_loads(self):
        """
            load loads data into memory.
        """
        loads = []
        index = []
        with open(os.path.join(self.path,'LF.L6'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                if (int(data[0]) == 1):
                    index.append(i)
                P = float(data[4])
                Q = float(data[5])
                V = float(data[6])
                PMin = float(data[11])
                PMax = float(data[10])
                QMin = float(data[9])
                QMax = float(data[8])
                loads.append({
                    'mark': int(data[0]),
                    'Type': int(data[3]),
                    'Pg': P,
                    'Qg': Q,
                    'V0': V,
                    'QMax': QMax,
                    'QMin': QMin,
                    'PMax': PMax,
                    'PMin': PMin,
                    'data': data.copy()
                })
        return loads, index
    
    def __output(self):
        """
            write the adjust input data to the dst dirs
        """
        def fpWrite(fp, data):
            for s in data:
                fp.write(s + ',')
            fp.write('\n')

        with open(os.path.join(self.runPath, 'LF.L5'), 'w+', encoding='utf-8') as fp:
            for v in self.generators:
                data = v['data']
                data[3] = '{:.3f}'.format(v['Pg'])
                data[4] = '{:.3f}'.format(v['Qg'])
                fpWrite(fp, data)

        with open(os.path.join(self.runPath, 'LF.L6'), 'w+', encoding='utf-8') as fp:
            for v in self.loads:
                data = v['data']
                data[4] = '{:.3f}'.format(v['Pg'])
                data[5] = '{:.3f}'.format(v['Qg'])
                fpWrite(fp, data)