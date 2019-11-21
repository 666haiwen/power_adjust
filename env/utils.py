import os
import json
import numpy as np
import pickle as pkl
import random

import matplotlib.pyplot as plt

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
    
    def set_pg_qg(self, path):
        with open(path, 'rb') as fp:
            dataset = pkl.load(fp)
            all_data = dataset['data']
            label = dataset['label']
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
        for i in range(label.shape[0]):
            data = all_data[i]
            if label[i] == 1:
                convergenced[convergenced_cnt][0] = data[1, self.loads_num:].sum()
                convergenced[convergenced_cnt][1] = data[2, self.loads_num:].sum()
                convergenced[convergenced_cnt][2] = sum(data[1, self.loads_num:] * data[0, self.loads_num:])
                convergenced[convergenced_cnt][3] = sum(data[2, self.loads_num:] * data[0, self.loads_num:])

                convergenced[convergenced_cnt][4] = sum(data[1, :self.loads_num])
                convergenced[convergenced_cnt][5] = sum(data[2, :self.loads_num])
                convergenced[convergenced_cnt][6] = sum(data[1, :self.loads_num] * data[0, :self.loads_num])
                convergenced[convergenced_cnt][7] = sum(data[2, :self.loads_num] * data[0, :self.loads_num])

                convergenced[convergenced_cnt][8] = sum(data[0, self.loads_num:])
                convergenced[convergenced_cnt][9] = sum(data[0, :self.loads_num])
                convergenced_cnt += 1
            else:
                disconvergenced[disconvergenced_cnt][0] = sum(data[1, self.loads_num:])
                disconvergenced[disconvergenced_cnt][1] = sum(data[2, self.loads_num:])
                disconvergenced[disconvergenced_cnt][2] = sum(data[1, self.loads_num:] * data[0, self.loads_num:])
                disconvergenced[disconvergenced_cnt][3] = sum(data[2, self.loads_num:] * data[0, self.loads_num:])

                disconvergenced[disconvergenced_cnt][4] = sum(data[1, :self.loads_num])
                disconvergenced[disconvergenced_cnt][5] = sum(data[2, :self.loads_num])
                disconvergenced[disconvergenced_cnt][6] = sum(data[1, :self.loads_num] * data[0, :self.loads_num])
                disconvergenced[disconvergenced_cnt][7] = sum(data[2, :self.loads_num] * data[0, :self.loads_num])

                disconvergenced[disconvergenced_cnt][8] = sum(data[0, self.loads_num:])
                disconvergenced[disconvergenced_cnt][9] = sum(data[0, :self.loads_num])
                disconvergenced_cnt += 1
        
        np.save(self.path + 'convergenced_features.npy', convergenced)
        np.save(self.path + 'disconvergenced_features.npy', disconvergenced)
    
    def plot_pg_qg(self):
        convergenced = np.load(self.path + 'convergenced_features.npy')
        disconvergenced = np.load(self.path + 'disconvergenced_features.npy')

        print('----------Convergenced Samples----------')
        print('Average of generator marks: {:.3f}'.format(convergenced[:, 8].mean()))
        print('Average of Loads marks:     {:.3f}\n'.format(convergenced[:, 9].mean()))
        print('Average Generator Pg Sum: {:.3f}'.format(convergenced[:, 0].mean()))
        print('Average Loads Pg Sum    : {:.3f}'.format(convergenced[:, 4].mean()))
        print('Average Pg Rage: {:.3f}'.format(convergenced[:, 0].mean() / convergenced[:, 4].mean()))

        print('Average Mark Generator Pg Sum: {:.3f}'.format(convergenced[:, 2].mean()))
        print('Average Mark Loads Pg Sum    : {:.3f}'.format(convergenced[:, 6].mean()))
        print('Average Mark Pg Rage: {:.3f}\n'.format(convergenced[:, 2].mean() / convergenced[:, 6].mean()))

        print('Average Generator Qg Sum: {:.3f}'.format(convergenced[:, 1].mean()))
        print('Average Loads Qg Sum    : {:.3f}'.format(convergenced[:, 5].mean()))
        print('Average Qg Rage: {:.3f}'.format(convergenced[:, 1].mean() / convergenced[:, 5].mean()))

        print('Average Mark Generator Qg Sum: {:.3f}'.format(convergenced[:, 3].mean()))
        print('Average Mark Loads Qg Sum    : {:.3f}'.format(convergenced[:, 7].mean()))
        print('Average Mark Qg Rage: {:.3f}'.format(convergenced[:, 3].mean() / convergenced[:, 7].mean()))

        print('\n----------DisConvergenced Samples----------')
        print('Average of generator marks: {:.3f}'.format(disconvergenced[:, 8].mean()))
        print('Average of Loads marks:     {:.3f}\n'.format(disconvergenced[:, 9].mean()))
        print('Average Generator Pg Sum: {:.3f}'.format(disconvergenced[:, 0].mean()))
        print('Average Loads Pg Sum    : {:.3f}'.format(disconvergenced[:, 4].mean()))
        print('Average Pg Rage: {:.3f}'.format(disconvergenced[:, 0].mean() / disconvergenced[:, 4].mean()))

        print('Average Mark Generator Pg Sum: {:.3f}'.format(disconvergenced[:, 2].mean()))
        print('Average Mark Loads Pg Sum    : {:.3f}'.format(disconvergenced[:, 6].mean()))
        print('Average Mark Pg Rage: {:.3f}\n'.format(disconvergenced[:, 2].mean() / disconvergenced[:, 6].mean()))

        print('Average Generator Qg Sum: {:.3f}'.format(disconvergenced[:, 1].mean()))
        print('Average Loads Qg Sum    : {:.3f}'.format(disconvergenced[:, 5].mean()))
        print('Average Qg Rage: {:.3f}'.format(disconvergenced[:, 1].mean() / disconvergenced[:, 5].mean()))

        print('Average Mark Generator Qg Sum: {:.3f}'.format(disconvergenced[:, 3].mean()))
        print('Average Mark Loads Qg Sum    : {:.3f}'.format(disconvergenced[:, 7].mean()))
        print('Average Mark Qg Rage: {:.3f}'.format(disconvergenced[:, 3].mean() / disconvergenced[:, 7].mean()))

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
        plt.plot(disconvergenced_x, disconvergenced[:, 8], color='green', label='DisGeneratorMark')
        plt.plot(disconvergenced_x, disconvergenced[:, 9], color='red', label='DisLoadsMark')
        plt.legend()
        plt.show()
        plt.plot(convergenced_x, convergenced[:, 8], color='green', label='GeneratorMark')
        plt.plot(convergenced_x, convergenced[:, 9], color='red', label='LoadsMark')
        plt.legend()
        plt.show()

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
# data.set_pg_qg('env/data/dongbei_LF-2000/train.pkl')
data.plot_pg_qg()

# data.set_dataset()
