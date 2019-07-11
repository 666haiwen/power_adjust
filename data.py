import random
import math
import time
import os
from env import Env
from const import CFG
from replayMemory import ReplayMemory, Transition


def generate_train_data(num):
    env = Env(CFG.ENV)
    env.set_random_sample_and_save(num, 'memory/train.npy')

def generate_test_data(num):
    env = Env(CFG.ENV)
    env.set_random_sample_and_save(num, 'memory/test.npy')


generate_train_data(1000)
generate_test_data(200)
