import random
import math
import os
import numpy as np
from env.Env import Env
from const import CFG


def select_action():
    reward = -10
    action_res = 0
    done_res = False
    next_res = None
    for action in range(n_actions):
        next_state, reward_tuple, done = env.step(action)
        reverse_action = env.get_reverse(action)
        env.step(reverse_action)
        if reward_tuple[0] > reward:
            reward = reward_tuple[0]
            action_res = action
            next_res = next_state
        if done and reward_tuple[0] > 0:
            action_res = action
            done_res = done
            next_res = next_state
            break
    return next_res, action_res, reward, done_res

# env init
env = Env(rand=CFG.RANDOM_INIT)
n_actions = env.action_space
env.reset()
state = env.get_state()
print('value = ', env.score())
while True:
    state, action, reward, done = select_action()
    env.step(action)
    print(env.get_action(action)[0])
    print(reward)
    print('value = ', env.score())
    if done:
        break