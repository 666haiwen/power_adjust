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


def select_action(state):
    sample = random.random()
    eps_threshold = CFG.EPS_START
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
        return
    transitions = memory.sample(CFG.BATCH_SIZE)
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


def collect_data(index):
    # collenct data
    while len(memory) < CFG.BATCH_SIZE:
        env.random_reset(index)
        state = torch.from_numpy(env.get_state()).to(device)
        while True:
            action = select_action(state)
            reward, done = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float)

            # Observe a new state
            next_state = torch.from_numpy(env.get_state()).to(device)
            memory.push(state, action, next_state, reward)
            if done or len(memory) >= CFG.BATCH_SIZE:
                break


def single_train(index):
    # train
    for epoch in range(CFG.TEST_EPOCH):
        # reset the env
        env.random_reset(index)
        state = torch.from_numpy(env.get_state()).to(device)

        while True:
            action = select_action(state)
            reward, done = env.step(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float)

            # Observe a new state
            next_state = torch.from_numpy(env.get_state()).to(device)

            # Perform one step of the optimization (on the target network)
            optimize_model()
            memory.push(state, action, next_state, reward)
            # Store the transition into memeory
            if reward == 1:
                return True
            # Move to the next state
            state = next_state

            # finish or not
            if done:
                break
        # Update the target network, copying all weights and biases in DQN
        if epoch % CFG.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    return False


def model_load():
    if not os.path.exists(CFG.MODEL_PATH):
        print('Can\'t find model in {}'.format(CFG.MODEL_PATH))
        return
    checkpoint = torch.load(CFG.MODEL_PATH)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])


def test():
    testNum = env.state_num
    success = []
    for index in range(testNum):
        memory.reset()
        model_load()
        if CFG.TEST_EPOCH > 1:
            collect_data(index)
        if single_train(index) == True:
            print('success')
            success.append(index)
        else:
            print('fail')
    print(success)
    print("success: {}/{}  {:.2f}%".format(len(success), testNum, len(success) / testNum))


# env init
env = Env(CFG.ENV, rand=CFG.RANDOM_INIT)
env.load_data(CFG.TEST_DATA)
n_actions = env.action_space
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seed
random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)

# model
policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimize
optimizer = optim.RMSprop(policy_net.parameters())

# data memory
memory = ReplayMemory(1000, CFG.MEMORY + 'data.pkl')
test()
