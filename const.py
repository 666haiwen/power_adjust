from easydict import EasyDict as edict

MODEL = 'DQN'
DATA_SET = '118nodes'
# generators + loads
NODES_NUM = 54 + 91
# FEATURENS= ['Pg', 'Qg', 'V0', 'Node', 'Type']
FEATURENS= ['Pg', 'Qg']
FEATURENS_NUM = len(FEATURENS)
THERESHOLD = {
    'Pg': [-1, 10],
    'Qg': [0, 10]
}


CFG = edict()
""" ################
    SETTING
"""

""" #################
    PARAM
"""
CFG.EPOCHS = 50000
CFG.BATCH_SIZE = 512
CFG.GAMMA = 0.999
CFG.EPS_START = 0.9
CFG.EPS_END = 0.05
CFG.EPS_DECAY = 1000
CFG.TARGET_UPDATE = 50
CFG.SEED = 7
CFG.LOG = 'log/{}/{}'.format(MODEL, DATA_SET)


""" ##############
    MODEL
"""
CFG.LOAD_MODEL = True
CFG.SAVE_EPOCHS = 100
CFG.MODEL_PATH = 'model/{}/{}/singel_init_model_state_0.pth'.format(MODEL, DATA_SET)


""" ##############
    ENV
"""
CFG.MEMORY = 'memory/{}/{}/'.format(MODEL, DATA_SET)
CFG.RANDOM_INIT = False
CFG.ENV = 'template/{}/'.format(DATA_SET)
