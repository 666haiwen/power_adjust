from easydict import EasyDict as edict


CFG = edict()

""" #################
    PARAM
"""
CFG.RANDOM_BEGIN = 500
CFG.EPOCHS = 100000
CFG.BATCH_SIZE = 512
CFG.GAMMA = 0.99
CFG.EPS_START = 0.95
CFG.EPS_END = 0.05
CFG.EPS_DECAY = 5000
CFG.EPS_DECAY_RANDOM_BEGIN = 10000
CFG.SEED = 7
CFG.LR = 1e-3

""" ################
    LOG
"""
CFG.LOG = 'log/'
CFG.REWARD_LOG = 100

""" ##############
    MODEL
"""
CFG.LOAD_MODEL = False
CFG.MEMORY_READ = False
CFG.TARGET_UPDATE = 30
CFG.SAVE_EPOCHS = 100
CFG.MODEL_PATH = 'model/36nodes.pth'


""" ##############
    ENV
"""
CFG.MEMORY = 'memory/'
CFG.RANDOM_INIT = True
