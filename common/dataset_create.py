import os
import json
import sys
import numpy as np
import pickle as pkl
import random
import math
import matplotlib.pyplot as plt
from common.utils import create_map, dijkstra


class powerDatasetLoader(object):

    def __init__(self, path):
        self.path = path
        # set file list of disconvergence examples
        self.dataList = {
            'data': [],
            'label': [],
            'path': []
        }
        random.seed(7)
        convergence = []
        disconvergence = []
        with open(path + 'allresult.txt', 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                label =  line.split(' ')
                file_path = path + label[0]
                label = 1 if len(label) > 2 else 0
                self.dataList['label'].append(label)
                self.dataList['path'].append(file_path)
                if label == 1:
                    convergence.append(file_path)
                else:
                    disconvergence.append(file_path)
        self.shape = (0,)
        self.mark = False
        self.loads_num = 0
        self.generators_num = 0
        self.key_dict = {}

        with open(path + 'convergence.json', 'w') as fp:
            json.dump({"train": convergence, "test": []} ,fp)
        
        with open(path + 'disconvergence.json', 'w') as fp:
            json.dump({"train": disconvergence, "test": []} ,fp)

    
    def set_dataset(self):
        idx_set = set([i for i in range(len(self.dataList['path']))])
        capacity = len(idx_set)
        train_idx_set = set(random.sample(range(0, capacity), int(capacity * 0.8)))
        train_dataset = {
            'data': np.zeros((len(train_idx_set), ) + self.shape, dtype=np.float32),
            'label': np.zeros(len(train_idx_set), dtype=np.int32),
            'path': []
        }
        for i, data_idx in enumerate(train_idx_set):
            print('train data [{}]'.format(i))
            train_dataset['data'][i] = self.one_data(data_idx)
            train_dataset['label'][i] = self.dataList['label'][data_idx]
            train_dataset['path'].append(self.dataList['path'][data_idx])

        test_idx_set = idx_set - train_idx_set
        test_dataset = {
            'data': np.zeros((len(test_idx_set), ) + self.shape, dtype=np.float32),
            'label': np.zeros(len(test_idx_set), dtype=np.int32),
            'path': []
        }
        for i, data_idx in enumerate(test_idx_set):
            print('test data [{}]'.format(i))
            test_dataset['data'][i] = self.one_data(data_idx)
            test_dataset['label'][i] = self.dataList['label'][data_idx]
            test_dataset['path'].append(self.dataList['path'][data_idx])
        with open(self.path + 'train.pkl', 'wb') as fp:
            pkl.dump(train_dataset, fp)
        with open(self.path + 'test.pkl', 'wb') as fp:
            pkl.dump(test_dataset, fp)


    def set_distance_matrix(self):
        edges = []
        prefix = self.dataList['path'][0]
        crs = []
        num = 0
        with open(prefix + '/LF.L2', encoding='utf8') as fp:
            for line in fp:
                line = list(filter(lambda str_: len(str_) != 0, [x.strip() for x in line.split(',')]))
                i = int(line[1])
                j = int(line[2])
                num = max(num, i, j)
                if i != j:
                    # line[4]: 线路正序电阻
                    r = float(line[4])
                    # line[5]: 线路正序电抗
                    x = float(line[5])
                    edges.append({'i': i, 'j': j, 'dis': math.sqrt(r * r + x * x)})
                else:
                    crs.append(i)
        num += 1
        edge_map = create_map(edges, num)
        shorest = np.ones((num, num))
        for i in range(num):
            print(i)
            shorest[i] = dijkstra(edge_map, num, i)
        np.save(self.path + 'dis.npy', shorest)
        

    def one_data(self, data_idx):
        return np.zeros(self.shape, dtype=np.float32)
    

    def set_pg_qg(self, path):
        with open(path, 'rb') as fp:
            dataset = pkl.load(fp)
            all_data = dataset['data']
            label = dataset['label']
        key_dict = self.key_dict
        all_data = all_data.swapaxes(1, 2)
        convergenced_num = sum(label)
        disconvergenced_num = len(label) - convergenced_num

        # GeneratorPgSum
        # GeneratorQgSum
        # GeneratorPgMarkSum
        # GeneratorQgMarkSum
        # LoadsPgSum
        # LoadsQgSum
        # LoadsPgMarkSum
        # LoadsQgMarkSum
        # GeneratorMarkSum
        # LoadsMarkSum
        convergenced = np.zeros((convergenced_num, 10))
        disconvergenced = np.zeros((disconvergenced_num, 10))
        convergenced_cnt = disconvergenced_cnt = 0
        end_index = self.loads_num + self.generators_num
        for i in range(label.shape[0]):
            data = all_data[i]
            if label[i] == 1:
                convergenced[convergenced_cnt][0] = data[key_dict['pg'], self.loads_num:end_index].sum()
                convergenced[convergenced_cnt][1] = data[key_dict['qg'], self.loads_num:end_index].sum()
                convergenced[convergenced_cnt][4] = sum(data[key_dict['pg'], :self.loads_num])
                convergenced[convergenced_cnt][5] = sum(data[key_dict['qg'], :self.loads_num])
                
                if self.mark:
                    convergenced[convergenced_cnt][2] = sum(data[key_dict['pg'], self.loads_num:end_index] * data[key_dict['mark'], self.loads_num:end_index])
                    convergenced[convergenced_cnt][3] = sum(data[key_dict['qg'], self.loads_num:end_index] * data[key_dict['mark'], self.loads_num:end_index])
                    convergenced[convergenced_cnt][6] = sum(data[key_dict['pg'], :self.loads_num] * data[key_dict['mark'], :self.loads_num])
                    convergenced[convergenced_cnt][7] = sum(data[key_dict['qg'], :self.loads_num] * data[key_dict['mark'], :self.loads_num])
                    convergenced[convergenced_cnt][8] = sum(data[key_dict['mark'], self.loads_num:end_index])
                    convergenced[convergenced_cnt][9] = sum(data[key_dict['mark'], :self.loads_num])


                convergenced_cnt += 1
            else:
                disconvergenced[disconvergenced_cnt][0] = sum(data[key_dict['pg'], self.loads_num:end_index])
                disconvergenced[disconvergenced_cnt][1] = sum(data[key_dict['qg'], self.loads_num:end_index])
                disconvergenced[disconvergenced_cnt][4] = sum(data[key_dict['pg'], :self.loads_num])
                disconvergenced[disconvergenced_cnt][5] = sum(data[key_dict['qg'], :self.loads_num])
                if self.mark:
                    disconvergenced[disconvergenced_cnt][2] = sum(data[key_dict['pg'], self.loads_num:end_index] * data[key_dict['mark'], self.loads_num:end_index])
                    disconvergenced[disconvergenced_cnt][3] = sum(data[key_dict['qg'], self.loads_num:end_index] * data[key_dict['mark'], self.loads_num:end_index])
                    disconvergenced[disconvergenced_cnt][6] = sum(data[key_dict['pg'], :self.loads_num] * data[key_dict['mark'], :self.loads_num])
                    disconvergenced[disconvergenced_cnt][7] = sum(data[key_dict['qg'], :self.loads_num] * data[key_dict['mark'], :self.loads_num])
                    disconvergenced[disconvergenced_cnt][8] = sum(data[key_dict['mark'], self.loads_num:end_index])
                    disconvergenced[disconvergenced_cnt][9] = sum(data[key_dict['mark'], :self.loads_num])

                disconvergenced_cnt += 1
        
        np.save(self.path + 'convergenced_features.npy', convergenced)
        np.save(self.path + 'disconvergenced_features.npy', disconvergenced)
    
    def plot_pg_qg(self, plot=False):
        convergenced = np.load(self.path + 'convergenced_features.npy')
        disconvergenced = np.load(self.path + 'disconvergenced_features.npy')

        print('----------Convergenced Samples----------')
        if self.mark:
            print('Average of generator marks: {:.3f}'.format(convergenced[:, 8].mean()))
            print('Average of Loads marks:     {:.3f}\n'.format(convergenced[:, 9].mean()))
        print('Average Generator Pg Sum: {:.3f}'.format(convergenced[:, 0].mean()))
        print('Average Loads Pg Sum    : {:.3f}'.format(convergenced[:, 4].mean()))
        print('Average Pg Rage: {:.3f}'.format(convergenced[:, 0].mean() / convergenced[:, 4].mean()))

        if self.mark:
            print('Average Mark Generator Pg Sum: {:.3f}'.format(convergenced[:, 2].mean()))
            print('Average Mark Loads Pg Sum    : {:.3f}'.format(convergenced[:, 6].mean()))
            print('Average Mark Pg Rage: {:.3f}\n'.format(convergenced[:, 2].mean() / convergenced[:, 6].mean()))

        print('Average Generator Qg Sum: {:.3f}'.format(convergenced[:, 1].mean()))
        print('Average Loads Qg Sum    : {:.3f}'.format(convergenced[:, 5].mean()))
        print('Average Qg Rage: {:.3f}'.format(convergenced[:, 1].mean() / convergenced[:, 5].mean()))
        if self.mark:
            print('Average Mark Generator Qg Sum: {:.3f}'.format(convergenced[:, 3].mean()))
            print('Average Mark Loads Qg Sum    : {:.3f}'.format(convergenced[:, 7].mean()))
            print('Average Mark Qg Rage: {:.3f}'.format(convergenced[:, 3].mean() / convergenced[:, 7].mean()))

        print('\n----------DisConvergenced Samples----------')
        if self.mark:
            print('Average of generator marks: {:.3f}'.format(disconvergenced[:, 8].mean()))
            print('Average of Loads marks:     {:.3f}\n'.format(disconvergenced[:, 9].mean()))
        print('Average Generator Pg Sum: {:.3f}'.format(disconvergenced[:, 0].mean()))
        print('Average Loads Pg Sum    : {:.3f}'.format(disconvergenced[:, 4].mean()))
        print('Average Pg Rage: {:.3f}'.format(disconvergenced[:, 0].mean() / disconvergenced[:, 4].mean()))

        if self.mark:
            print('Average Mark Generator Pg Sum: {:.3f}'.format(disconvergenced[:, 2].mean()))
            print('Average Mark Loads Pg Sum    : {:.3f}'.format(disconvergenced[:, 6].mean()))
            print('Average Mark Pg Rage: {:.3f}\n'.format(disconvergenced[:, 2].mean() / disconvergenced[:, 6].mean()))

        print('Average Generator Qg Sum: {:.3f}'.format(disconvergenced[:, 1].mean()))
        print('Average Loads Qg Sum    : {:.3f}'.format(disconvergenced[:, 5].mean()))
        print('Average Qg Rage: {:.3f}'.format(disconvergenced[:, 1].mean() / disconvergenced[:, 5].mean()))

        if self.mark:
            print('Average Mark Generator Qg Sum: {:.3f}'.format(disconvergenced[:, 3].mean()))
            print('Average Mark Loads Qg Sum    : {:.3f}'.format(disconvergenced[:, 7].mean()))
            print('Average Mark Qg Rage: {:.3f}'.format(disconvergenced[:, 3].mean() / disconvergenced[:, 7].mean()))

        if plot:
            convergenced_num = convergenced.shape[0]
            convergenced_x = [i for i in range(convergenced_num)]
            # plt.plot(convergenced_x, convergenced[:, 0], color='green', label='GeneratorPgSum')
            # plt.plot(convergenced_x, convergenced[:, 4], color='red', label='LoadsPgSum')
            # plt.legend()
            # plt.show()

            # plt.plot(convergenced_x, convergenced[:, 2], color='green', label='MarkGeneratorPgSum')
            # plt.plot(convergenced_x, convergenced[:, 6], color='red', label='MarkLoadsPgSum')
            # plt.legend()
            # plt.show()

            # plt.plot(convergenced_x, convergenced[:, 1], color='green', label='GeneratorQgSum')
            # plt.plot(convergenced_x, convergenced[:, 5], color='red', label='LoadsQgSum')
            # plt.legend()
            # plt.show()

            # plt.plot(convergenced_x, convergenced[:, 3], color='green', label='MarkGeneratorQgSum')
            # plt.plot(convergenced_x, convergenced[:, 7], color='red', label='MarkLoadsQgSum')
            # plt.legend()
            # plt.show()

            # Disconvergenced
            disconvergenced_num = disconvergenced.shape[0]
            disconvergenced_x = [i for i in range(disconvergenced_num)]
            # plt.plot(disconvergenced_x, disconvergenced[:, 0], color='green', label='GeneratorPgSum')
            # plt.plot(disconvergenced_x, disconvergenced[:, 4], color='red', label='LoadsPgSum')
            # plt.legend()
            # plt.show()

            # plt.plot(disconvergenced_x, disconvergenced[:, 2], color='green', label='MarkGeneratorPgSum')
            # plt.plot(disconvergenced_x, disconvergenced[:, 6], color='red', label='MarkLoadsPgSum')
            # plt.legend()
            # plt.show()

            # plt.plot(disconvergenced_x, disconvergenced[:, 1], color='green', label='GeneratorQgSum')
            # plt.plot(disconvergenced_x, disconvergenced[:, 5], color='red', label='LoadsQgSum')
            # plt.legend()
            # plt.show()

            # plt.plot(disconvergenced_x, disconvergenced[:, 3], color='green', label='MarkGeneratorQgSum')
            # plt.plot(disconvergenced_x, disconvergenced[:, 7], color='red', label='MarkLoadsQgSum')
            # plt.legend()
            # plt.show()

            # Marks
            # plt.plot(disconvergenced_x, disconvergenced[:, 8], color='green', label='DisGeneratorMark')
            # plt.plot(disconvergenced_x, disconvergenced[:, 9], color='red', label='DisLoadsMark')
            # plt.legend()
            # plt.show()
            # plt.plot(convergenced_x, convergenced[:, 8], color='green', label='GeneratorMark')
            # plt.plot(convergenced_x, convergenced[:, 9], color='red', label='LoadsMark')
            # plt.legend()
            # plt.show()


class dataLoader_36Nodes(powerDatasetLoader):
    def __init__(self, path='env/data/case36/'):
        """
        loads: (10, 2) Pg, Qg
        generators: (9, 2) Pg, Qg
        Ac: (134,) => (67, 0) Mark Mark
        """
        super(dataLoader_36Nodes, self).__init__(path)
        self.loads_num = 10
        self.generators_num = 9
        self.ac_num = 134
        self.shape = (86, 2)
        self.mark = False
        self.key_dict = {
            'pg': 0,
            'qg': 1
        }
    
    def transfer_data(self, data):
        """
        data shape: (channels, n)
        """
        res = {
            'generators': [],
            'loads': [],
            'acs': []
        }
        for i in range(self.loads_num):
            res['loads'].append({
                'pg': data[0][i],
                'qg': data[1][i],
            })
        for i in range(self.loads_num, self.loads_num + self.generators_num):
            res['generators'].append({
                'pg': data[0][i],
                'qg': data[0][i],
            })
        for i in range(self.loads_num + self.generators_num, self.shape[0]):
            res['acs'] += [data[0][i], data[1][i]]
        return res

    def one_data(self, data_idx):
        path = self.dataList['path'][data_idx]
        result = np.zeros(self.shape, dtype=np.float32)
        with open(os.path.join(path,'LF.L6'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                result[i] = np.array([float(data[4]), float(data[5])])

        with open(os.path.join(path, 'LF.L5'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                result[i + self.loads_num] = \
                    np.array([float(data[3]), float(data[4])])
                
        with open(os.path.join(path, 'LF.L2'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                result[int(i / 2) + self.loads_num + self.generators_num][i % 2] = float(data[0])
        return result
    

class dataLoader_118Nodes(powerDatasetLoader):
    def __init__(self, path='env/data/case118/'):
        """
        loads: (91, 2) Pg, Qg
        generators: (54, 2) Pg, Qg
        """
        super(dataLoader_118Nodes, self).__init__(path)
        self.loads_num = 91
        self.generators_num = 54
        self.ac_num = 0
        self.shape = (145, 2)
        self.mark = False
        self.key_dict = {
            'pg': 0,
            'qg': 1
        }

class dataLoader_2000Nodes(powerDatasetLoader):
    def __init__(self, path='env/data/dongbei_LF-2000/'):
        """
        loads: (816, 4) Marks, Pg, Qg, Vbase
        generators: (531, 4) Marks, Pg, Qg, Vbase
        acs: (2970, 1) Marks
        """
        super(dataLoader_2000Nodes, self).__init__(path)
        self.loads_num = 816
        self.generators_num = 531
        self.shape = (816 + 531, 4)
        self.mark = True
        self.key_dict = {
            'mark': 0,
            'pg': 1,
            'qg': 2,
            'vbase': 3
        }
    
    def transfer_data(self, data):
        """
        data shape: (channels, n)
        """
        res = {
            'generators': [],
            'loads': []
        }
        for i in range(self.loads_num):
            res['loads'].append({
                'mark': data[0][i],
                'pg': data[1][i],
                'qg': data[2][i],
                'vbase': data[3][i]
            })
        for i in range(self.loads_num, self.loads_num + self.generators_num):
            res['generators'].append({
                'mark': data[0][i],
                'pg': data[1][i],
                'qg': data[2][i],
                'vbase': data[3][i]
            })
        return res

    def one_data(self, data_idx):
        result = np.zeros(self.shape, dtype=np.float32)
        path = self.dataList['path'][data_idx]
        with open(os.path.join(path,'LF.L6'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                result[i] = np.array([float(data[0]), float(data[4]), float(data[5]), float(data[6])])

        with open(os.path.join(path, 'LF.L5'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                result[i + self.loads_num] = \
                    np.array([float(data[0]), float(data[3]), float(data[4]), float(data[5])])

        return result

    def test_ac(self):
        base_ac_mark = self.get_ac_data(0)
        for i in range(1, len(self.dataList['path'])):
            test_ac_mark = self.get_ac_data(i)
            diff_ac_mark = base_ac_mark - test_ac_mark
            if diff_ac_mark.min() == 0 and diff_ac_mark.max() == 0:
                continue
            else:
                print('{} diffierent'.format(i))
    

    def get_ac_data(self, data_idx):
        result = np.zeros(2970)
        path = self.dataList['path'][data_idx]
        with open(os.path.join(path,'LF.L2'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                result[i] = int(data[0])
        return result
