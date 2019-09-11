import random
import math
import os
import numpy as np
from env.Env import Env
from common.replayMemory import PrioritizedReplayBuffer, Transition
from const import CFG
from model import DQN, Dueling_DQN

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def select_action(state, steps_done, decay):
    sample = random.random()
    eps_threshold = CFG.EPS_END + (CFG.EPS_START - CFG.EPS_END) * \
        math.exp(-1. * steps_done / decay)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), False
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), True


def optimize_model():
    transitions, weights, batch_idxes = memory.sample(CFG.BATCH_SIZE)
    weights = torch.tensor(weights, device=device)
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

    next_state_values = torch.zeros(CFG.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * CFG.GAMMA) + reward_batch

    # Compute Huber loss
    td_errors = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduce=None) * weights
    new_priorities = np.abs(td_errors.cpu().detach().numpy()) + 1e-6
    memory.update_priorities(batch_idxes, new_priorities)
    td_errors = td_errors.mean()

    # Optimize the model
    optimizer.zero_grad()
    td_errors.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return td_errors.item(), expected_state_action_values.mean()


def train(steps_done):
    print('Begin to Train!\n')
    decay = CFG.EPS_DECAY
    for epoch in range(epoch_end + 1, CFG.EPOCHS):
        index = env.reset()
        print("epoch[{}/{}]  index: [{}]".format(epoch, CFG.EPOCHS, index))
        state = torch.from_numpy(env.get_state()).to(device)

        train_loss = cnt = q_cnt = reward_cnt = 0
        while True:
            cnt += 1
            action, rand = select_action(state, steps_done, decay)
            next_state, reward_tuple, done = env.step(action.item())
            if type(reward_tuple) != tuple:
                reward_tuple = (reward_tuple, reward_tuple)
            reward = torch.tensor([reward_tuple[0]], device=device, dtype=torch.float)
            reward_cnt += reward
            done = torch.tensor([done], device=device, dtype=torch.float)
            next_state = torch.from_numpy(next_state).to(device)
            
            value = env.score()
            # Perform one step of the optimization (on the target network)
            if len(memory) >= CFG.BATCH_SIZE:
                loss, q_value = optimize_model()
                q_cnt += q_value
                steps_done += 1
                train_loss += loss
                writer.add_scalars('train_q', {
                    'q': q_value,
                    'loss': loss
                    }, steps_done)

            if not rand:
                # writer.add_scalar('action_reward', reward, steps_done)
                print('actions: {};  reward: {};  value: {:.3f}'.format(env.get_action(action.item())[0], reward.item(), value))

            if done:
                next_state = None
            # Store the transition into memory
            memory.push(state, action, next_state, reward, done)
            # Store the reverse transition into memory
            if not done:
                reverse_action = torch.tensor([[env.get_reverse(action.item())]], device=device, dtype=torch.long)
                reverse_reward = torch.tensor([reward_tuple[1]], device=device, dtype=torch.float)
                memory.push(next_state, reverse_action, state, reverse_reward, done)

            # finish
            if done:
                break
            # Move to the next state
            state = next_state

        print('Epoch = {}   reward = {}  Loss: {:.4f}  Value: [{:.3f}]'\
            .format(epoch, reward.item(), train_loss / cnt, value))
        # writer.add_scalar('value', value, epoch)
        writer.add_scalars('dqn', {
            'score': reward_cnt,
            'q_value': q_cnt / cnt
        }, epoch)
        writer.add_scalar('Loss', train_loss / cnt, epoch)
        writer.add_scalar('Done', done and reward > 0, epoch)
        # Update the target network, copying all weights and biases in DQN
        if epoch % CFG.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (epoch + 1) % CFG.SAVE_EPOCHS == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, CFG.MODEL_PATH)
            memory.save()

# env init
env = Env(rand=CFG.RANDOM_INIT)
n_actions = env.action_space
state_dim = env.state_dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
steps_done = 0

# seed
random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)

# model
policy_net = Dueling_DQN(state_dim, n_actions).to(device)
target_net = Dueling_DQN(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimize
# optimizer = optim.RMSprop(policy_net.parameters(), lr=CFG.LR)
optimizer = optim.Adam(policy_net.parameters(), lr=CFG.LR)
# data memory
memory = PrioritizedReplayBuffer(100000, CFG.MEMORY + 'train_data.pkl')
if CFG.MEMORY_READ:
    memory.read()

epoch_end = 0
steps_done = 0
if CFG.LOAD_MODEL == True and os.path.exists(CFG.MODEL_PATH):
    checkpoint = torch.load(CFG.MODEL_PATH)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])
    epoch_end = checkpoint['epoch']
    steps_done = epoch_end * 20
    # env.reset(epoch_end)
# Train!
writer = SummaryWriter(CFG.LOG)
train(steps_done)
writer.close()
