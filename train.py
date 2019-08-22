import random
import math
import os
from env import Env
from replayMemory import ReplayMemory, Transition
from const import CFG
from model import DQN

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def save_model(path, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def select_action(state, decay):
    global steps_done
    sample = random.random()
    eps_threshold = CFG.EPS_END + (CFG.EPS_START - CFG.EPS_END) * \
        math.exp(-1. * steps_done / decay)
    if len(memory) > CFG.BATCH_SIZE:
        steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), False
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), True


def optimize_model():
    if len(memory) + len(successMemory) < CFG.BATCH_SIZE:
        return torch.tensor([0], dtype=torch.float)
    minSample = min(len(successMemory), int(CFG.BATCH_SIZE / 2)) if len(memory) > CFG.BATCH_SIZE / 2 else CFG.BATCH_SIZE - len(memory)
    transitions = memory.sample(CFG.BATCH_SIZE - minSample) + successMemory.sample(minSample)
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


def train():
    global steps_done
    # load = False
    success_cnt = [0 for i in range(CFG.REWARD_LOG)]
    # decay = CFG.EPS_DECAY
    decay = CFG.EPS_DECAY_RANDOM_BEGIN
    dataset_num = env.state_num
    for epoch in range(epoch_end + 1, CFG.EPOCHS):
        # reset the env
        # if epoch == CFG.RANDOM_BEGIN and CFG.RANDOM_INIT == True:
        #     steps_done = 0
        #     decay = CFG.EPS_DECAY_RANDOM_BEGIN

        if CFG.RANDOM_INIT == True:
            env.random_reset(epoch % dataset_num)
        # else:
        #     env.reset()
        state = torch.from_numpy(env.get_state()).to(device)

        random_cnt = train_loss = cnt = 0
        while True:
            cnt += 1
            action, rand = select_action(state, decay)
            random_cnt += rand
            print('actions: {},  random: {}'.format(env.get_action(action.item()), rand))

            reward, done = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float)

            # Observe a new state
            next_state = torch.from_numpy(env.get_state()).to(device)

            # Perform one step of the optimization (on the target network)
            loss = optimize_model().item()
            if loss != 0:
                train_loss += loss

            # Store the transition into memeory
            if reward == 1:
                successMemory.push(state, action, next_state, reward)
                success_cnt[epoch % CFG.REWARD_LOG] = 1
            else:
                memory.push(state, action, next_state, reward)
                success_cnt[epoch % CFG.REWARD_LOG] = 0
            # Move to the next state
            state = next_state

            # finish or not
            if done:
                print('Epoch = {}, reward = {}'.format(epoch, reward.item()))
                break

        cnt = 1 if cnt == 0 else cnt
        writer.add_scalar('Loss', train_loss / cnt, epoch)
        writer.add_scalar('Reward', reward, epoch)
        writer.add_scalar('DQN_Reward', sum(success_cnt) / CFG.REWARD_LOG, epoch)
        # Update the target network, copying all weights and biases in DQN
        if epoch % CFG.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (epoch + 1) % CFG.SAVE_EPOCHS == 0:
            save_model(CFG.MODEL_PATH, epoch)

        if epoch == CFG.RANDOM_BEGIN:
            save_model(CFG.MODEL_PATH + '.single', epoch)


# env init
env = Env(CFG.ENV, rand=CFG.RANDOM_INIT)
if CFG.RANDOM_INIT == True:
    env.load_data(CFG.TRAIN_DATA)
n_actions = env.action_space
state_dim = env.state_dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
steps_done = 0

# seed
random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)

# model
policy_net = DQN(state_dim, n_actions).to(device)
target_net = DQN(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimize
optimizer = optim.RMSprop(policy_net.parameters(), lr=CFG.LR)

# data memory
memory = ReplayMemory(100000, CFG.MEMORY + 'data.pkl')
successMemory = ReplayMemory(10000, CFG.MEMORY + 'success.pkl', True)
if CFG.MEMORY_READ:
    memory.read()
    successMemory.read()

epoch_end = 0
if CFG.LOAD_MODEL == True and os.path.exists(CFG.MODEL_PATH):
    checkpoint = torch.load(CFG.MODEL_PATH)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])
    epoch_end = checkpoint['epoch']

# Train!
writer = SummaryWriter(CFG.LOG)
train()
writer.close()
