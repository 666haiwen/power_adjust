import os
import json
import random


class dataLoader_36Nodes(object):

    def __init__(self, path):
        self.path = path
        # set file list of disconvergence examples
        self.dataList = []
        self.convergence_files = []
        self.disconvergence_files = []
        self.acMarks = []
        with open(path + 'allresult.txt', 'r', encoding='utf-8') as fp:
            for line in fp:
                label =  line.split(' ')
                # data = self.trendData.set_state_from_files(path + fileName + '/' + subFile + '/')
                # filter the data whoses value too large
                if len(label) > 2:
                    self.convergence_files.append(path + label[0])
                else:
                    self.disconvergence_files.append(path + label[0])
        random.seed(7)
        self.train_test_dataset(self.convergence_files, path + 'convergence.json')
        self.train_test_dataset(self.disconvergence_files, path + 'disconvergence.json')

    def train_test_dataset(self, fileList, save_path):
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
        
        with open(save_path, 'w', encoding='utf-8') as fp:
            json.dump({
                'train': train_dataset,
                'test': test_dataset
            }, fp)

        
data = dataLoader_36Nodes('env/data/36nodes/')
# memory = ReplayMemory(100000, CFG.MEMORY + 'data.pkl')
