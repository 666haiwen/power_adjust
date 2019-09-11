import random
import math
import os
import numpy as np
from env.Env import Env
from utils.replayMemory import ReplayMemory, Transition
from const import CFG
from model import DQN

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1), False

def test():
    print('Begin to Test!\n')
    data_num = env.capacity
    success_cnt = 0
    for epoch in range(data_num):
        index = env.reset()
        print("epoch[{}/{}]  index: [{}]".format(epoch, CFG.EPOCHS, index))
        state = torch.from_numpy(env.get_state()).to(device)
        cnt = 0
        while True and cnt <= 50:
            cnt += 1
            action, rand = select_action(state)
            next_state, reward_tuple, done = env.step(action.item())
            if type(reward_tuple) != tuple:
                reward_tuple = (reward_tuple, reward_tuple)
            reward = torch.tensor([reward_tuple[0]], device=device, dtype=torch.float)
            done = torch.tensor([done], device=device, dtype=torch.float)
            next_state = torch.from_numpy(next_state).to(device)
            
            value = env.score()
            if not rand:
                # writer.add_scalar('action_reward', reward, steps_done)
                print('actions: {};  reward: {};  value: {}'.format(env.get_action(action.item())[0], reward, value))

            # finish
            if done:
                break
            # Move to the next state
            state = next_state
        if done and reward != -1 and reward != -10:
            success_cnt += 1
            writer.add_scalar('Done', 1, epoch)
        else:
            writer.add_scalar('Done', 0, epoch)
        writer.add_scalar('Value', value, epoch)
        print('Epoch = {}   Value: [{:.3f}]'\
            .format(epoch, value))
        # Update the target network, copying all weights and biases in DQN
        if epoch % CFG.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    print('result: [{}/{}] = {:.4f}'.format(success_cnt, data_num, success_cnt / data_num * 100))

# env init
env = Env(rand=CFG.RANDOM_INIT, train=False)
n_actions = env.action_space
state_dim = env.state_dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seed
random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)

# model
policy_net = DQN(state_dim, n_actions).to(device)
target_net = DQN(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if CFG.LOAD_MODEL == True and os.path.exists(CFG.MODEL_PATH):
    checkpoint = torch.load(CFG.MODEL_PATH)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])

# Test
writer = SummaryWriter(CFG.LOG)
test()
writer.close()
