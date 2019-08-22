import os
import json
import random
import numpy as np
from TrendData import TrendData
from const import CFG


class dataLoader_36Nodes(object):

    def __init__(self, path):
        self.path = path
        # set file list of disconverage examples
        self.dataList = []
        self.trendData = TrendData(path='template/36nodes-from-datasets/')
        with open(path + 'allresult.txt', 'r', encoding='utf-8') as fp:
            for line in fp:
                data =  line.split(' ')
                # converage continue
                if len(data) > 2:
                    continue
                
                # save the disconverage examples
                fileName, subFile = data[0].split('/')
                data = self.trendData.set_state_from_files(path + fileName + '/' + subFile + '/')
                # filter the data whoses value too large
                if np.max(data) < 20:
                    self.dataList.append(data)
        
        # save dataset
        self.save_data(len(self.dataList))
        self.save_data(1000, 'sample')


    def save_data(self, dataNum, name=''):
        # read examples into state type
        random.seed(CFG.SEED)
        testDataFlag = [False for i in range(dataNum)]
        testNum = int(dataNum * CFG.TEST)
        while sum(testDataFlag) < testNum:
            randInt = random.randint(0, dataNum - 1)
            while testDataFlag[randInt]:
                randInt = random.randint(0, dataNum - 1)
            testDataFlag[randInt] = True

        stateShape = (1, 2, self.trendData.nodesNum)
        trainDataset = np.zeros((dataNum - testNum, ) + stateShape, dtype=np.float32)
        testDataset = np.zeros((testNum, ) + stateShape, dtype=np.float32)
        testCnt = 0
        for i in range(dataNum):
            if testDataFlag[i]:
                testDataset[testCnt] = self.dataList[i]
                testCnt += 1
            else:
                trainDataset[i - testCnt] = self.dataList[i]
        
        # save
        np.save(self.path + 'train_{}.npy'.format(name), trainDataset)
        np.save(self.path + 'test_{}.npy'.format(name), testDataset)

data = dataLoader_36Nodes('data/36nodes/')
