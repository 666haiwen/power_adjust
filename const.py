from easydict import EasyDict as edict

MODEL = 'DQN'
DATA_SET = '36nodes'
# FEATURENS= ['Pg', 'Qg', 'V0', 'Node', 'Type']
FEATURENS= ['Pg', 'Qg']
FEATURENS_NUM = len(FEATURENS)


CFG = edict()
""" ################
    SETTING
"""
CFG.TEST = 0.2
CFG.TRAIN_DATA = 'data/{}/train_sample.npy'.format(DATA_SET)
CFG.TEST_DATA = 'data/{}/test_sample.npy'.format(DATA_SET)
CFG.TRAIN_TEST = 'train'
CFG.TEST_EPOCH = 50
""" #################
    PARAM
"""
CFG.RANDOM_BEGIN = 500
CFG.EPOCHS = 100000
CFG.BATCH_SIZE = 512
CFG.GAMMA = 0.999
CFG.EPS_START = 0.90
CFG.EPS_END = 0.05
CFG.EPS_DECAY = 2000
CFG.EPS_DECAY_RANDOM_BEGIN = 10000
CFG.TARGET_UPDATE = 50
CFG.SEED = 7
CFG.LR = 1e-3
""" ################
    LOG
"""
CFG.LOG = 'log/{}/{}'.format(MODEL, DATA_SET)
CFG.REWARD_LOG = 100

""" ##############
    MODEL
"""
CFG.LOAD_MODEL = False
CFG.SAVE_EPOCHS = 100
CFG.MEMORY_READ = False
CFG.MODEL_PATH = 'model/{}/{}/paper_model_state_0.pth'.format(MODEL, DATA_SET)


""" ##############
    ENV
"""
CFG.MEMORY = 'memory/{}/{}/'.format(MODEL, DATA_SET)
CFG.RANDOM_INIT = True
# CFG.ENV = 'template/{}/'.format(DATA_SET)
CFG.ENV = 'template/36nodes-from-datasets/'
