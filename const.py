from easydict import EasyDict as edict

# generators + loads
DATA_SET = '6958nodes'
NODES_NUM = 457 + 4950
# actually 2353
# FEATURENS= ['Pg', 'Qg', 'V0', 'Node', 'Type']
# FEATURENS= ['Pg', 'Qg']
FEATURENS= ['Pg']
FEATURENS_NUM = len(FEATURENS)
THERESHOLD = {
    'Pg': [-1, 10],
    'Qg': [0, 10]
}


CFG = edict()
""" ################
    SETTING
"""
""" ################
    DATA
"""
CFG.DATA = edict()
CFG.DATA.GENERATORS = 457
CFG.DATA.LOADS = 4950
CFG.DATA.NODES_NUM = NODES_NUM
CFG.DATA.FEATURES_NUM = len(FEATURENS)
""" #################
    PARAM
"""
CFG.EPOCHS = 100000
CFG.BATCH_SIZE = 512
CFG.GAMMA = 0.999
CFG.EPS_START = 0.9
CFG.EPS_END = 0.05
CFG.EPS_DECAY = 1000
CFG.TARGET_UPDATE = 20
CFG.SEED = 7
CFG.LOG = 'log/{}-1/'.format(DATA_SET)
""" ##############
    MODEL
"""
CFG.LOAD_MODEL = True
CFG.SAVE_EPOCHS = 100
CFG.MODEL_PATH = 'model/{}/singel_init_model_state_1.pth'.format(DATA_SET)
""" ##############
    ENV
"""
CFG.MEMORY = 'memory/{}/'.format(DATA_SET)
CFG.RANDOM_INIT = False
CFG.ENV = 'template/{}/'.format(DATA_SET)
CFG.TEST_DATA = 'memory/{}/test.npy'.format(DATA_SET)
CFG.TRAIN_DATA = 'memory/{}/train.npy'.format(DATA_SET)
