from easydict import EasyDict as edict


CFG = edict()

""" #################
    PARAM
"""
CFG.EPOCHS = 100000
CFG.BATCH_SIZE = 512
CFG.GAMMA = 0.99
CFG.EPS_START = 0.95
CFG.EPS_END = 0.05
CFG.EPS_DECAY = 5000
CFG.EPS_BEGIN = 0.4
CFG.EPS_ALPHA = 7
CFG.NUM_PROCESS = 8
CFG.SEED = 7
CFG.LR = 1e-3
CFG.TAU = 0.001
""" ################
    LOG
"""
CFG.LOG = 'log/case118_state_adjust/'
CFG.REWARD_LOG = 100

""" ##############
    MODEL
"""
CFG.LOAD_MODEL = False
CFG.MEMORY_READ = False
CFG.TARGET_UPDATE = 30
CFG.SAVE_EPOCHS = 500
CFG.MODEL_PATH = 'model/case118_state_adjust.pth'
# CFG.MODEL_PATH = 'model/39nodes_convergenced_with_convergenced.pth'

""" ##############
    ENV
"""
CFG.MEMORY = 'memory/'
CFG.RANDOM_INIT = True
