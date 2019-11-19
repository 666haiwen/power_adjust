import os
import json
import numpy as np
import pickle as pkl
import random


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
        with open(path + 'allresult.txt', 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                label =  line.split(' ')
                file_path = path + label[0]
                label = 1 if len(label) > 2 else 0
                self.dataList['label'].append(label)
                self.dataList['path'].append(file_path)
        self.shape = (0,)

    
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
    

    def one_data(self, data_idx):
        return np.zeros(self.shape, dtype=np.float32)


class dataLoader_36Nodes(powerDatasetLoader):

    def __init__(self, path='env/data/36nodes_new/'):
        super(dataLoader_36Nodes, self).__init__(path)
        self.shape = (172, )
 
    def one_data(self, data_idx):
        path = self.dataList['path'][data_idx]
        generators = []
        with open(os.path.join(path, 'LF.L5'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                generators.extend([float(data[3]), float(data[4])])
        
        loads = []
        with open(os.path.join(path,'LF.L6'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                loads.extend([float(data[4]), float(data[5])])
        
        ACs = []
        with open(os.path.join(path, 'LF.L2'), 'r', encoding='gbk') as fp:
            for line in fp:
                data = line.split(',')[:-1]
                ACs.append(int(data[0]))
        return np.array(loads + generators + ACs, dtype=np.float32)
    

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


# data = dataLoader_36Nodes('env/data/36nodes_new/')
data = dataLoader_2000Nodes()
data.test_ac()
# data.set_dataset()
