from easydict import EasyDict as edict

NODES_NUM = 18
# FEATURENS= ['Pg', 'Qg', 'V0', 'Node', 'Type']
FEATURENS= ['Pg', 'Qg']
FEATURENS_NUM = len(FEATURENS)
THERESHOLD = {
    'Pg': [0, 10],
    'Qg': [0, 10]
}


CFG = edict()
""" ################
    DATA
"""
CFG.DATA = edict()
CFG.DATA.NODES_NUM = 18
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
CFG.LOG = 'log/7/'
""" ##############
    MODEL
"""
CFG.LOAD_MODEL = True
CFG.SAVE_EPOCHS = 100
CFG.MODEL_PATH = 'model/random_genrators_36nodes_policy_net_model_state_1.pth'
""" ##############
    ENV
"""
CFG.ENV = 'template/36nodes/'
CFG.TEST_DATA = 'memory/36nodes/test.npy'
CFG.TRAIN_DATA = 'memory/36nodes/train.npy'
