import random
import math
import os
import time
import numpy as np
from env.Env import Env
from common.replayMemory import PrioritizedReplayBuffer, Transition
from const import CFG
from model import DQN, Dueling_DQN

import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def select_action(state, eps_threshold, n_actions, device, policy_net):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), False
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), True

def Actor(idx, policy_net, device):
  env = Env(rand=CFG.RANDOM_INIT, dataset='36nodes_new', thread=idx)
  random.seed(idx + CFG.SEED)
  n_actions = env.action_space
  eps_threshold = CFG.EPS_BEGIN ** (1 + idx * CFG.EPS_ALPHA / CFG.NUM_PROCESS)
  local_memory = BatchStorage(args.n_steps, args.gamma)

  env.reset(random.randint(0, env.capacity))
  state = torch.from_numpy(env.get_state()).to(device)
  cnt = 0
  while True:  
    cnt += 1

    action, rand = select_action(state, eps_threshold, n_actions, device, policy_net)
    next_state, reward, done = env.step(action.item())
    reward = torch.tensor([reward], device=device, dtype=torch.float)
    done = torch.tensor([done], device=device, dtype=torch.float)
    next_state = torch.from_numpy(next_state).to(device)
    local_memory.push(state, action, next_state, reward, done)

    if done:



def Learn(policy_net, target_net, epoch_end):
  for epoch_end in range(epoch_end + 1, CFG.EPOCHS):
    pass
  pass


if __name__ == '__main__':  
  env = Env(rand=CFG.RANDOM_INIT, dataset='36nodes_new')
  n_actions = env.action_space
  state_dim = env.state_dim

  torch.manual_seed(CFG.SEED)
  # mp.set_start_method('spawn')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # model
  policy_net = Dueling_DQN(state_dim, n_actions).to(device)
  target_net = Dueling_DQN(state_dim, n_actions).to(device)
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()

  # Load model
  epoch_end = 0
  if CFG.LOAD_MODEL == True and os.path.exists(CFG.MODEL_PATH):
    checkpoint = torch.load(CFG.MODEL_PATH)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])
    epoch_end = checkpoint['epoch']
  policy_net.share_memory()
  

  processes = []
  for actor_id in range(CFG.NUM_PROCESS):
    p = mp.Process(target=Actor, args=(actor_id, policy_net, device))
    p.start()
    processes.append(p)
  p = mp.Process(target=Learn, args=(policy_net, target_net, epoch_end))
  for p in processes:
    p.join()
