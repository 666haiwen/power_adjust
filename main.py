import random
import math
import time
import os
import numpy as np
from collections import namedtuple
from itertools import count
from env import Env
from replayMemory import ReplayMemory, Transition
from const import CFG
from model import DQN

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tensorboardX import SummaryWriter

env = Env('template/36nodes/')
n_actions = env.action_space
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('log')

steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = CFG.EPS_END + (CFG.EPS_START - CFG.EPS_END) * \
        math.exp(-1. * steps_done / CFG.EPS_DECAY)
    if len(memory) > CFG.BATCH_SIZE:
        steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < CFG.BATCH_SIZE:
        return torch.tensor([0], dtype=torch.float)
    minSample = min(len(successMemory), int(CFG.BATCH_SIZE / 2))
    sucessSample = random.randrange(minSample) if minSample > 0 else 0
    transitions = memory.sample(CFG.BATCH_SIZE - sucessSample) + successMemory.sample(sucessSample)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(CFG.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * CFG.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss


random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
policy_net = DQN(CFG.DATA.NODES_NUM, CFG.DATA.FEATURES_NUM, n_actions).to(device)
target_net = DQN(CFG.DATA.NODES_NUM, CFG.DATA.FEATURES_NUM, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000, 'memory/data.pkl')
memory.read()
successMemory = ReplayMemory(1000, 'memory/success.pkl', True)
successMemory.read()

epoch_end = 0
if CFG.LOAD_MODEL == True and os.path.exists(CFG.MODEL_PATH):
    checkpoint = torch.load(CFG.MODEL_PATH)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])
    epoch_end = checkpoint['epoch']
    rewards = checkpoint['rewards']
    # optimizer.load_state_dict(checkpoint['op#timizer_state_dict'])


first_time = time.time()
rewards = []
losses = []
for epoch in range(epoch_end + 1, CFG.EPOCHS):
    before_time = time.time()
    env.reset()
    state = torch.from_numpy(env.get_state()).to(device)
    train_loss = 0
    cnt = 0
    reward = 0
    while True:
        action = select_action(state)
        print('actions:', env.get_action(action))
        reward, done = env.step(action.item())
        reward = torch.tensor([reward], device=device, dtype=torch.float)

        # Observe a new state
        next_state = torch.from_numpy(env.get_state()).to(device)
        
        # Store the transition into memeory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        loss = optimize_model().item()
        if loss != 0:
            train_loss += loss
            cnt += 1
            convergence = 0
        
        # finish or not
        if done:
            print('One epoch done, reward = {}'.format(reward.item()))
            if reward == 1:
                successMemory.push(state, action, next_state, reward)
            break
    # Update the target network, copying all weights and biases in DQN
    if epoch % CFG.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    cnt = 1 if cnt == 0 else cnt
    time_now = time.time()
    rewards.append(reward)
    losses.append(train_loss / cnt)
    writer.add_scalar('Train/Loss', train_loss / cnt, epoch)
    writer.add_scalar('Train/Reward', reward, epoch)
    print('====> Epoch: {} \tAverage loss : {:.4f}\tTime cost: {:.0f}\tAll time: {:.0f}'.format(
        epoch, losses[-1], time_now - before_time, time_now - first_time))
    if (epoch + 1) % CFG.SAVE_EPOCHS == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rewards': rewards,
            'losses': losses
        }, CFG.MODEL_PATH)
