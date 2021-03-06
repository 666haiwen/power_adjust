import os
from shutil import copyfile
import subprocess
import tempfile
import time
import numpy as np
from .const import proximity_section, proximity_voltage
from .const import FEATURENS_NUM, RATE, SECTION_TASK, VOLTAGE_TASK

class TrendData(object):
    """
        Load the data of TrendData from disk to memeory.
        And check that the data is reasonable or not.
    """
    def __init__(self, path='None', runPath='run/', target='state-section', buses=None, generators=None, loads=None, ACs=None):
        """
            @params:
            path: the path to the template folder, load initialize settings.
            runPath: the path to the WMLFRTMsg.exe folder
            target: the target of task, 'state-section' means section trend state adjust, 
                    'state-voltage' means voltage trend state adjust
                    'convergence' means trend convergence adjust
            buses: load buses from outer, default: None
            generators: load generators from outer, default: None
            loads: load loads from outer, default: None
            ACs: load ACs from outer, default: None
        """
        self.path = path
        self.runPath = runPath
        self.target = target
        # just for convince
        self.target_line = [v - 1 for v in SECTION_TASK['index']]
        # copy file to reset run env
        self.tmp_out = tempfile.SpooledTemporaryFile(10*1000)
        self.copy_files()
        self.run()

        # read data
        self.buses, self.bus_name = self.__load_buses() if buses == None else buses
        self.generators, self.g_index = self.__load_generators() if generators == None else generators
        self.loads, self.l_index = self.__load_loads() if loads == None else loads
        self.ACs = self.__load_AC_lines()
        self.ACs_output = self.__load_AC_output_lines()
        
        # set params
        self.ac_len = len(self.ACs) if target != 'state-section' else 26 # 26 just for 36nodes (index > 26) are all Shunt capacitor
        self.g_len = len(self.g_index)
        self.l_len = len(self.l_index)
        self.nodesNum = self.g_len + self.l_len + self.ac_len
        self.state_dim = (FEATURENS_NUM, self.nodesNum)
        # initialize the state
        self.state = np.zeros((1, FEATURENS_NUM, self.nodesNum), dtype=np.float32)
    
    def reset(self, path):
        """
            reset the env.
        """
        self.path = path
        self.copy_files()

        self.generators, self.g_index = self.__load_generators()
        self.loads, self.l_index = self.__load_loads()
        self.ACs = self.__load_AC_lines()
        if self.target != 'convergence':
            self.run()
            self.ACs_output = self.__load_AC_output_lines()
            self.pre_value = self.__calculate_state_section_reward()

        self.get_state()

    def copy_files(self):
        """
            Replace the files in run dir by template dir
        """
        dirs = ['LF.L0', 'LF.L1', 'LF.L2', 'LF.L3', 'LF.L4', 'LF.L5', 'LF.L6', 'LF.L7']
        for fileName in dirs:
            copyfile(self.path + '/' + fileName, self.runPath + '/' + fileName)

    def set_state_from_files(self, path):
        """
            Read generators and loads from disk,and translate them into state.
            @param:
            path: the path to the file read
            @return:
            state: the file in state type; shape=(1, 2, self.nodesNum); generators + loads + aclines(mark)
        """
        self.path = path
        self.generators, self.g_index = self.__load_generators()
        self.loads, self.l_index = self.__load_loads()
        self.ACs = self.__load_AC_lines()
        return self.get_state()
    
    def get_state(self):
        """
            Translate data into state type. shape:[nodesNum, featuresNum]
            @return:
            state: darray, shape:[1, nodesNum, featuresNum]
        """
        # set generators data into state
        for i in range(self.g_len):
            self.state[0][0][i] = self.__get_generators(i)['Pg']
            self.state[0][1][i] = self.__get_generators(i)['Qg']

        # set loads data into state
        for i in range(self.l_len):
            self.state[0][0][i + self.g_len] = self.__get_loads(i)['Pg']
            self.state[0][1][i + self.g_len] = self.__get_loads(i)['Qg']
        
        # set marks of ac_lines
        for i in range(self.ac_len):
            index = i + self.g_len + self.l_len
            self.state[0][0][index] = self.ACs_output[i]['Pg'] if self.target == 'state-section'\
                else self.ACs[i]['mark']
            self.state[0][1][index] = self.ACs_output[i]['Qg'] if self.target == 'state-section'\
                else self.ACs[i]['mark']

        return self.state

    def set_state(self, state):
        """
            Set data from state outside.
        """
        for i in range(self.g_len):
            self.generators[self.g_index[i]]['Pg'] = state[0][0][i]
            self.generators[self.g_index[i]]['Qg'] = state[0][1][i]

        for i in range(self.l_len):
            self.loads[self.l_index[i]]['Pg'] = state[0][0][i + self.g_len]
            self.loads[self.l_index[i]]['Qg'] = state[0][1][i + self.g_len]
        
        # set marks of ac_lines
        if self.target != 'state-section':
            for i in range(self.ac_len):
                self.ACs[i]['mark'] = state[0][0][i + self.g_len + self.l_len]

    def run(self):
        """
            Run the WMLFRTMsg.exe
        """
        cwd = os.getcwd()
        os.chdir(self.runPath)
        fileno = self.tmp_out.fileno()
        p =subprocess.Popen('WMLFRTMsg.exe', stdout=fileno,stderr=fileno)
        p.wait()
        self.tmp_out.seek(0)
        os.chdir(cwd)

    def reward(self, action=None, reback_action=None):
        """
            1.Change the trendData by action
            2.Run WMLFRTMsg.exe
            3.Get the result and reward
            @return:
            reward: return reward;
            done: True or False
        """
        flag = self.__changeData(action) if action != None else True
        if flag:
            self.__output()
            self.run()
            
            convergence = True
            with open(os.path.join(self.runPath,  'LF.CAL'), 'r', encoding='gbk') as fp:
                firstLine = fp.readline()
                data = firstLine.split(',')
                if int(data[0]) != 1:
                    convergence = False
            
            if self.target == 'convergence':
                if convergence == True:
                    return 1, True
                else:
                    return -0.1, False
            else:
                # state from convergence to disconvergence
                if convergence == False:
                    self.__changeData(reback_action)
                    return -1, True
                
                # still convergence
                if self.target == 'state-section':
                    return self.__state_section_reward()
                elif self.target == 'state-voltage':
                    return self.__state_voltage_reward()

        return -1, True

    def __calculate_state_section_reward(self):
        value = 0
        for index in self.target_line:
            value += self.ACs_output[index]['Pg']
        return abs(value)

    def __state_section_reward(self):
        """
            Reward Function for state section problem.
            @param:
            pre_value: sum of target line of the state before action
            @return:
            reward: value
            done: True or False
        """
        self.ACs_output = self.__load_AC_output_lines()
        value = self.__calculate_state_section_reward()
        pre_value = self.pre_value
        self.pre_value = value
        # finish the target of state section adjust
        if value >= RATE[0] * SECTION_TASK['value'] and\
            value <= RATE[1] * SECTION_TASK['value']:
            return 1, True
        
        # not finish the target
        return proximity_section(value) - proximity_section(pre_value), False

    def __state_voltage_reward(self):
        """
            Reward Function for state section problem.
            @param:
            pre_value: sum of target line of the state before action
            @return:
            reward: value
            done: True or False
        """
        pass

    def __changeData(self, action):
        """
            Change the trendData in memory by action from env.
            @param:
            action: the action from agent. Apply the action into files in the disk. type: dict
        """
        if action['node'] == 'AC':
            self.ACs[action['index']]['mark'] = action['value']
            ac_index = action['index'] + self.g_len + self.l_len
            self.state[0][0][ac_index] = action['value']
            self.state[0][1][ac_index] = action['value']            

        elif action['node'] == 'generator':
            index = self.g_index[action['index']]
            if self.generators[index]['Qg'] == 0 or self.generators[index]['Pg'] == 0:
                rate = 0
            else:
                rate = self.generators[index]['Qg'] / self.generators[index]['Pg']
            self.generators[index]['Pg'] += action['value']
            self.generators[index]['Qg'] += action['value'] * rate
            self.state[0][0][action['index']] += action['value']
            self.state[0][1][action['index']] += action['value'] * rate
        
        return True
        
    def __get_generators(self, index):
        return self.generators[self.g_index[index]]
    
    def __get_loads(self, index):
        return self.loads[self.l_index[index]]

    def __load_buses(self):
        """
            load buses data into memory.
            @returns:
            buses: the LF.L1 data contain vBase
            bus_index: the index in LF.L1 for bus_Name
        """
        buses = [None]
        bus_name = []
        with open(os.path.join(self.path, 'LF.L1'), 'r', encoding='gbk') as fp:
            for line in fp:
                data = line.split(',')[:-1]
                buses.append({
                    'vBase': float(data[1])
                })
                bus_name.append(data[-1][1:-1])
        return buses, bus_name
    

    def __load_generators(self):
        """
            load generators data into memory.
        """
        generators = []
        index = []
        with open(os.path.join(self.path, 'LF.L5'), 'r', encoding='gbk') as fp:
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
    
    def __load_AC_lines(self):
        """
            load ac_lines(LF.L2) data into memory.
        """
        ACs = []
        with open(os.path.join(self.path, 'LF.L2'), 'r', encoding='gbk') as fp:
            for line in fp:
                data = line.split(',')[:-1]
                R = float(data[4])
                X = float(data[5])
                ACs.append({
                    'mark': int(data[0]),
                    'R': R,
                    'X': X,
                    'data': data.copy()
                })
        return ACs
    
    def __load_AC_output_lines(self):
        """
            load output data of ac_lines(LF.LP2) into memory.
        """
        ACs_output = []
        with open(os.path.join(self.runPath, 'LF.LP2'), 'r', encoding='gbk') as fp:
            for line in fp:
                data = line.split(',')[:-1]
                ACs_output.append({
                    'I': int(data[1]),
                    'J': int(data[2]),
                    'Pg': float(data[3]),
                    'Qg': float(data[4]),
                    'data': data.copy()
                })
        return ACs_output

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
        
        with open(os.path.join(self.runPath, 'LF.L2'), 'w+', encoding='utf-8') as fp:
            for v in self.ACs:
                data = v['data']
                data[0] = '{}'.format(v['mark'])
                fpWrite(fp, data)

        if self.target == 'state-section':
            with open(os.path.join(self.runPath, 'LF.L5'), 'w+', encoding='utf-8') as fp:
                for v in self.generators:
                    data = v['data']
                    data[3] = '{:.3f}'.format(v['Pg'])
                    data[4] = '{:.3f}'.format(v['Qg'])
                    fpWrite(fp, data)
