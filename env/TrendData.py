import os
from shutil import copyfile
import subprocess
import tempfile
import time
import numpy as np
import torch
from env.const import proximity_section, proximity_voltage
from env.const import FEATURENS_NUM, RATE, SECTION_TASK, VOLTAGE_TASK

class TrendData(object):
    """
        Load the data of TrendData from disk to memeory.
        And check that the data is reasonable or not.
    """
    def __init__(self, path='None', runPath='env/run/', target='state-section', buses=None, generators=None, loads=None, ACs=None):
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

        # read data
        self.buses, self.bus_name = self.__load_buses() if buses == None else buses
        self.generators, self.g_index = self.__load_generators() if generators == None else generators
        self.loads, self.l_index, self.loads_mark_num = self.__load_loads() if loads == None else loads
        self.ACs, self.CRs = self.__load_AC_lines()
        self.ACs_output = self.__load_AC_output_lines()
        
        # set params
        self.ac_len = len(self.ACs)
        self.cr_len = len(self.CRs)
        self.g_len = len(self.generators)
        self.l_len = len(self.loads)
        self.ac_out_len = len(self.ACs_output)

        if 'state' in target:
            self.ac_out_len = SECTION_TASK['ac_out_len']
            self.nodesNum = self.g_len + self.l_len + self.ac_out_len
        else:
            self.nodesNum = self.g_len + self.l_len + self.cr_len
        self.state_dim = (FEATURENS_NUM, self.nodesNum)
        # initialize the state
        self.state = np.zeros((1, FEATURENS_NUM, self.nodesNum), dtype=np.float32)
        self.Pg = 0

    
    def reset(self, path, restate=True, cr_set=False):
        """
            reset the env.
        """
        self.path = path
        self.copy_files()

        self.generators, self.g_index = self.__load_generators()
        self.loads, self.l_index, self.loads_mark_num = self.__load_loads()
        self.ACs, self.CRs = self.__load_AC_lines()

        self.ac_len = len(self.ACs)
        self.cr_len = len(self.CRs)
        self.g_len = len(self.generators)
        self.l_len = len(self.loads)

        if restate:
            self.get_state()
        if self.target == 'state-section':
            self.run()
            self.ACs_output = self.__load_AC_output_lines()
            self.pre_value = self.calculate_state_section_reward()
            self.Pg = sum(self.state[0,0,:self.g_len])
        if self.target == 'state-voltage':
            self.pre_value = [0]
        if cr_set:
            self.__set_crs_by_loads()


    def copy_files(self):
        """
            Replace the files in run dir by template dir
        """
        dirs = os.listdir(self.path)
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
        self.loads, self.l_index, self.loads_mark_num = self.__load_loads()
        self.ACs, self.CRs = self.__load_AC_lines()
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
        
        if self.target == 'state-section':
            # set Pg Qg of ac lines
            for i in range(self.ac_out_len):
                index = i + self.g_len + self.l_len
                self.state[0][0][index] = self.ACs_output[i]['Pg']
                self.state[0][1][index] = self.ACs_output[i]['Qg']
        elif self.target == 'state-voltage':
            # set marks of Shunt Capacitor Reactor
            for i in range(self.CRs):
                index = i + self.g_len + self.l_len
                self.state[0][0][index] = self.CRs[i]['mark']
                self.state[0][1][index] = self.CRs[i]['mark']

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
                self.CRs[i]['mark'] = state[0][0][i + self.g_len + self.l_len]

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
        
        convergence = True
        Cal_path = os.path.join(self.runPath,  'LF.CAL')
        while not os.path.exists(Cal_path):
            time.sleep(0.2)

        with open(Cal_path, 'r', encoding='gbk') as fp:
            firstLine = fp.readline()
            data = firstLine.split(',')
            if int(data[0]) != 1:
                convergence = False
        return convergence
        # data = self.__run_result()
        # return sum(data[:, 0]), sum(data[:, 1])


    def test(self, data, content=['g'], dataset='case36', balance=True, alpha_Pg=1.0, alpha_Qg=1.0):
        """
            Test the result by vae.
            @params:
                data: numpy data to set generators/loads/acs
                content: reload content, type: list
                dataset: 'case36'/'DongBei_Case'
                balance: balance Pg/Qg between generators and loads, default: True
                alpha: the coefficient of balance, default:1.2
        """
        dim = data.shape[1]
        if dataset != 'case36' and dataset != 'DongBei_Case' and dataset != 'case2000':
            raise ValueError("params of env test function must belong to \
                ['case36', 'DongBei_Case'], but input {} instead.".format(dataset))
        if 'cr' in content:
            cr_begin = self.g_len + self.l_len
            for i in range(self.cr_len):
                self.CRs[i]['mark'] = int(data[i % dim][cr_begin + int(i / dim)] + 0.5)

        if 'g' in content:
            if dataset == 'case36':
                for i in range(self.g_len):
                    self.generators[i]['Pg'] = data[0][i + self.l_len]
                    self.generators[i]['Qg'] = data[1][i + self.l_len]
            elif dataset == 'DongBei_Case' or dataset == 'case2000':
                for i in range(self.g_len):
                    self.generators[i]['mark'] = int(data[0][i + self.l_len] + 0.5)
                    self.generators[i]['Pg'] = data[1][i + self.l_len]
                    self.generators[i]['Qg'] = data[2][i + self.l_len]
                    self.generators[i]['Type'] = int(data[3][i + self.l_len] + 0.5)

        if balance:
            loads_pg = sum([x['Pg'] * x['mark'] for x in self.loads])
            loads_qg = sum([x['Qg'] * x['mark'] for x in self.loads])
            generators_pg = sum([x['Pg'] * x['mark'] for x in self.generators])
            generators_qg = sum([x['Qg'] * x['mark'] for x in self.generators])
            mark_generators = sum([x['mark'] for x in self.generators])
            rate_pg = (loads_pg * alpha_Pg - generators_pg) / mark_generators
            rate_qg = (loads_qg * alpha_Qg - generators_qg) / mark_generators
            
            if 'g' in content:
                for i in range(self.g_len):
                    if self.generators[i]['mark'] == 1:
                        self.generators[i]['Pg'] += rate_pg
                        self.generators[i]['Qg'] += rate_qg


        if 'l' in content:
            for i in range(self.l_len):
                if dataset == 'case36':
                    self.loads[i]['Pg'] = data[0][i]
                    self.loads[i]['Qg'] = data[1][i]
                else:
                    self.loads[i]['mark'] =  int(data[0][i] + 0.5)
                    self.loads[i]['Pg'] = data[1][i]
                    self.loads[i]['Qg'] = data[2][i]
        self.__output(content=content + ['cr'])
        return self.run()
        

    def reward(self, action=None, reback_action=None, classifer_model=None):
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
            convergence = self.run()
            
            if self.target == 'convergence':
                if convergence == True:
                    return 10, True
                else:
                    return self.__convergence_reward(classifer_model), False
            else:
                # state from convergence to disconvergence
                if convergence == False:
                    self.state = np.zeros((1, FEATURENS_NUM, self.nodesNum), dtype=np.float32)
                    return -10, True
                
                # still convergence
                if self.target == 'state-section':
                    return self.__state_section_reward()
                elif self.target == 'state-voltage':
                    return self.__state_voltage_reward()
        return -10, True


    def __convergence_reward(self, classifer_model):
        if classifer_model == None:
            return -0.1
        
        data = np.zeros((1, classifer_model.dim), dtype=np.float32)
        # set generators data into state
        for i in range(self.g_len):
            data[0][i * 2] = self.state[0][0][i]
            data[0][i * 2 + 1] = self.state[0][1][i]

        # set loads data into state
        for i in range(self.g_len, self.l_len + self.g_len):
            data[0][i * 2] = self.state[0][0][i]
            data[0][i * 2 + 1] = self.state[0][1][i]
        
        cr_begin = (self.g_len + self.l_len) * 2
        for i in range(self.cr_len):
            data[0][cr_begin + i] = self.CRs[i]['mark']
        
        data = torch.from_numpy(data).float()
        output = classifer_model(data)[0][1]
        return - abs(1 - output)


    def calculate_state_section_reward(self):
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
        # section reward
        self.ACs_output = self.__load_AC_output_lines()
        value = self.calculate_state_section_reward()
        self.pre_value = value
        section_reward = proximity_section(value)

        # Pg reward
        Pg = sum(self.state[0,0,:self.g_len])
        Pg_reward = abs(Pg - self.Pg) * SECTION_TASK['Pg_rate']

        # finish the target of state section adjust
        if value >= RATE[0] * SECTION_TASK['value'] and\
            value <= RATE[1] * SECTION_TASK['value']:
            return 10, True
        
        return section_reward + Pg_reward, False

    def __state_voltage_reward(self):
        """
            Reward Function for state section problem.
            @param:
            pre_value: sum of target line of the state before action
            @return:
            reward: value
            done: True or False
        """
        # section voltage reward
        value = []
        with open(os.path.join(self.runPath, 'LF.LP1'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                if i in VOLTAGE_TASK['index']:
                    value.append(float(data[1]))
                    break
        reward = 0
        flag = True
        self.pre_value = value
        for v in value:
            if not (v >= VOLTAGE_TASK['min'] and v <= VOLTAGE_TASK['max']):
                flag = False
                reward -= abs(VOLTAGE_TASK['value'] - v) * VOLTAGE_TASK['rate']
        if flag:
            return 10, flag

        return reward, flag

    def __changeData(self, action):
        """
            Change the trendData in memory by action from env.
            @param:
            action: the action from agent. Apply the action into files in the disk. type: dict
        """
        if action['node'] == 'CR':
            self.CRs[action['index']]['mark'] = 1 - self.CRs[action['index']]['mark']
            cr_index = action['index'] + self.g_len + self.l_len
            self.state[0][0][cr_index] = 1 - self.state[0][0][cr_index]
            self.state[0][1][cr_index] = 1 - self.state[0][1][cr_index]

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
            if self.generators[index]['Pg'] <= -1:
                return False
        
        return True
        
    def __get_generators(self, index):
        return self.generators[self.g_index[index]]
    
    def __get_loads(self, index):
        return self.loads[self.l_index[index]]

    def __run_result(self):
        """
            After run WMLFRTMsg.exe.Read Pg and Qg of balance_machine. \n
            @returns: \n
            data: array; [[Pg, Qg], [Pg, Qg]].. shape(n,2); means different value of PV machine.
        """
        flag = False
        data = []
        for v in self.tmp_out:
            line = v.decode().strip().split('  ')
            if 'Slack' in line[0]:
                flag = True
                continue
            if flag:
                if not 'BUS' in line[0]:
                    continue
                PV_data = []
                for v in line[1:]:
                    if v != '':
                        print(v)
                        PV_data.append(float(v))
                data.append(PV_data)
        return np.array(data)

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
        Crs = []
        with open(os.path.join(self.path, 'LF.L2'), 'r', encoding='gbk') as fp:
            for line in fp:
                data = line.split(',')[:-1]
                R = float(data[4])
                X = float(data[5])
                I = int(data[1])
                J = int(data[2])
                if I == J:
                    Crs.append({
                        'mark': int(data[0]),
                        'I': I,
                        'J': J,
                        'R': R,
                        'X': X,
                        'data': data.copy()
                    })
                else:
                    ACs.append({
                        'mark': int(data[0]),
                        'I': I,
                        'J': J,
                        'R': R,
                        'X': X,
                        'data': data.copy()
                    })
        return ACs, Crs
    
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
        mark = 0
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
                    'busName': int(data[1]),
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
                mark += loads[-1]['mark']
        return loads, index, mark
    
    def __output(self, content=['g']):
        """
            write the adjust input data to the dst dirs
        """
        def fpWrite(fp, data):
            for s in data:
                fp.write(s + ',')
            fp.write('\n')
        
        if 'cr' in content:
            with open(os.path.join(self.runPath, 'LF.L2'), 'w+', encoding='utf-8') as fp:
                for v in self.ACs:
                    fpWrite(fp, v['data'])
                for v in self.CRs:
                    data = v['data']
                    data[0] = '{}'.format(v['mark'])
                    fpWrite(fp, data)

        if self.target == 'state-section' or 'vae' in self.target:
            if 'g' in content:
                with open(os.path.join(self.runPath, 'LF.L5'), 'w+', encoding='utf-8') as fp:
                    for v in self.generators:
                        data = v['data']
                        data[0] = '{}'.format(v['mark'])
                        data[3] = '{:.3f}'.format(v['Pg'])
                        data[4] = '{:.3f}'.format(v['Qg'])
                        data[2] = '{:.3f}'.format(v['Type'])
                        fpWrite(fp, data)

        if 'vae' in self.target and 'l' in content:
            with open(os.path.join(self.runPath, 'LF.L6'), 'w+', encoding='utf-8') as fp:
                for v in self.loads:
                    data = v['data']
                    data[0] = '{}'.format(v['mark'])
                    data[4] = '{:.3f}'.format(v['Pg'])
                    data[5] = '{:.3f}'.format(v['Qg'])
                    fpWrite(fp, data)
    
    def __set_crs_by_loads(self, rate=0.20):
        """
            set shunt Capacitor Reactors near heavy loads.
            set Capacitor to 1 (whose X < 0)
            set Reactors to 0 (whose X > 0)
            @paramters:
            rate: float, top rate of heavy loads. The shunt capcitor reactors near them should be reset.
        """
        sorted_loads = sorted(self.loads, key=lambda load: load['Pg'], reverse=True)
        high_loads_list = []
        cnt = 0
        limits = int(self.loads_mark_num * rate)
        for load in sorted_loads:
            if load['mark'] == 1 and cnt < limits:
                cnt += 1
                high_loads_list.append(load['busName'])
        
        for i, cr in enumerate(self.CRs):
            if cr['X'] >= 99999:
                continue
            if cr['I'] in high_loads_list:
                print('!!!!')
                if cr['X'] < 0:
                    if self.CRs[i]['mark'] == 0:
                        print('Open capacitance {}!'.format(cr['I']))
                        self.CRs[i]['mark'] = 1
                else:
                    if self.CRs[i]['mark'] == 1:
                        print('Close Reactance {}!'.format(cr['I']))
                        self.CRs[i]['mark'] = 0
