import os
import json
import numpy as np
import pickle as pkl
import random


class dataLoader_36Nodes(object):

    def __init__(self, path):
        self.path = path
        # set file list of disconvergence examples
        self.dataList = {
            'data': [],
            'label': [],
            'path': []
        }
        self.convergence_files = []
        self.disconvergence_files = []
        self.acMarks = []
        with open(path + 'allresult.txt', 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                print(i)
                label =  line.split(' ')
                # data = self.trendData.set_state_from_files(path +e fileName + '/' + subFile + '/')
                # filter the data whoses value too large
                file_path = path + label[0]
                if len(label) > 2:
                    self.convergence_files.append(file_path)
                    label = 1
                else:
                    self.disconvergence_files.append(file_path)
                    label = 0
                self.dataList['data'].append(self.create_one_data(file_path))
                self.dataList['label'].append(label)
                self.dataList['path'].append(file_path)
        random.seed(7)
        # self.train_test_dataset(self.convergence_files, path + 'convergence.json')
        # self.train_test_dataset(self.disconvergence_files, path + 'disconvergence.json')
        self.set_dataset()


    def create_one_data(self, path):
        generators = []
        with open(os.path.join(path, 'LF.L5'), 'r', encoding='gbk') as fp:
            for i, line in enumerate(fp):
                data = line.split(',')[:-1]
                generators.extend([float(data[4]), float(data[5])])
        
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
        return np.array(generators + loads + ACs)
        

    def train_test_dataset(self, fileList, save_path=None):
        num = len(fileList)
        testDataFlag = [False for i in range(num)]
        testNum = int(num * 0.2)
        while sum(testDataFlag) < testNum:
            randInt = random.randint(0, num - 1)
            while testDataFlag[randInt]:
                randInt = random.randint(0, num - 1)
            testDataFlag[randInt] = True

        test_dataset = []
        train_dataset = []
        for i, flag in enumerate(testDataFlag):
            if flag:
                test_dataset.append(fileList[i])
            else:
                train_dataset.append(fileList[i])
        
        res = {
            'train': train_dataset,
            'test': test_dataset
        }
        if save_path != None:
            with open(save_path, 'w', encoding='utf-8') as fp:
                json.dump(res, fp)
        return res
    
    
    def set_dataset(self):
        idx_set = set([i for i in range(len(self.dataList['data']))])
        capacity = len(idx_set)
        train_idx_set = set(random.sample(range(0, capacity), int(capacity * 0.8)))
        shape = self.dataList['data'][0].shape
        train_dataset = {
            'data': np.zeros((len(train_idx_set), shape[0])),
            'label': np.zeros(len(train_idx_set), dtype=np.int32),
            'path': []
        }
        for i, data_idx in enumerate(train_idx_set):
            train_dataset['data'][i] = self.dataList['data'][data_idx]
            train_dataset['label'][i] = self.dataList['label'][data_idx]
            train_dataset['path'].append(self.dataList['path'][data_idx])

        test_idx_set = idx_set - train_idx_set
        test_dataset = {
            'data': np.zeros((len(test_idx_set), shape[0])),
            'label': np.zeros(len(test_idx_set), dtype=np.int32),
            'path': []
        }
        for i, data_idx in enumerate(test_idx_set):
            test_dataset['data'][i] = self.dataList['data'][data_idx]
            test_dataset['label'][i] = self.dataList['label'][data_idx]
            test_dataset['path'].append(self.dataList['path'][data_idx])
        with open(self.path + 'train.pkl', 'wb') as fp:
            pkl.dump(train_dataset, fp)
        with open(self.path + 'test.pkl', 'wb') as fp:
            pkl.dump(test_dataset, fp)


data = dataLoader_36Nodes('env/data/36nodes_new/')
