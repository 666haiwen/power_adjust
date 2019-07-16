from easydict import EasyDict as edict

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
""" ################
    DATA
"""
CFG.DATA = edict()
CFG.DATA.GENERATORS = 54
CFG.DATA.LOADS = 91
CFG.DATA.NODES_NUM = NODES_NUM
CFG.DATA.FEATURES_NUM = len(FEATURENS)
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
CFG.LOG = 'log/118-0/'
""" ##############
    MODEL
"""
CFG.LOAD_MODEL = True
CFG.SAVE_EPOCHS = 100
CFG.MODEL_PATH = 'model/118nodes/singel_init_model_state_0.pth'
""" ##############
    ENV
"""
CFG.MEMORY = 'memory/118nodes/'
CFG.RANDOM_INIT = False
CFG.ENV = 'template/118nodes/'
CFG.TEST_DATA = 'memory/118nodes/test.npy'
CFG.TRAIN_DATA = 'memory/118nodes/train.npy'
