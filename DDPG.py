import os
import numpy as np
from model import Actor, Critic
from common.replayMemory import PrioritizedReplayBuffer, Transition
from common.noisy import OrnsteinUhlenbeckActionNoise

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DDPG(object):

    def __init__(self, env, args, device):
        self.env = env
        self.args = args
        self.device = device
        state_dim = env.state_dim
        action_dim = env.action_space
        max_action = env.max_action

        # model
        self.actor_net = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor_net.state_dict())

        self.critic_net = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic_net.state_dict())

        # optimize
        self.actor_optimizer = optim.Adam(self.actor_net.parameters())
        self.critic_optimizer = optim.Adam(self.critic_net.parameters())
        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim, dtype=np.float32), sigma=args.SIGMA * np.ones(action_dim, dtype=np.float32))

        # data memory
        self.memory = PrioritizedReplayBuffer(100000, args.MEMORY + 'train_data.pkl')
        if args.MEMORY_READ:
            self.memory.read()

    def soft_update(self):
        self.__soft_update(self.actor_net, self.actor_target)
        self.__soft_update(self.critic_net, self.critic_target)

    def __soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.args.TAU) + param.data * self.args.TAU)

    def optimize_model(self):
        transitions = self.memory.sample(self.args.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # Compute the target Q value
        target_Q = self.critic_target(next_state_batch, self.actor_net(next_state_batch))
        target_Q = reward_batch + ((1 - done_batch) * self.args.GAMMA * target_Q)

        # Compute critic loss
        critic_loss = F.mse_loss(self.critic_net(state_batch, action_batch), target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic_net(state_batch, self.actor_net(state_batch)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss, actor_loss

    def train(self):
        self.env.reset()
        self.ou_noise.reset()
        state = torch.from_numpy(self.env.get_state()).to(self.device).unsqueeze(0)
        
        critic_loss_cnt = actor_loss_cnt = cnt = 0
        while True:
            cnt += 1
            action = self.actor_net(state)
            action = action.cpu().detach().numpy() + self.ou_noise()
            # print('actions:', env.get_action(action.item()))
            next_state, reward, done = self.env.step(action)
            reward = torch.tensor([reward], device=self.device, dtype=torch.float)
            done = torch.tensor([done], device=self.device, dtype=torch.float)
            action = torch.from_numpy(action).to(self.device)
            next_state = torch.from_numpy(next_state).to(self.device).unsqueeze(0)

            # Store the transition into memeory
            self.memory.push(state, action, next_state, reward, done)
            
            # Perform one step of the optimization (on the target network)
            if len(self.memory) >= self.args.BATCH_SIZE:
                critic_loss, actor_loss = self.optimize_model()
                critic_loss_cnt += critic_loss.item()
                actor_loss_cnt += actor_loss.item()
            # Move to the next state
            state = next_state
            # finish or not
            if done:
                print('One epoch done, reward = {}'.format(reward.item()))
                break
        return actor_loss_cnt / cnt, critic_loss_cnt / cnt, reward

    def save_model(self, epoch):
        torch.save({
                'epoch': epoch,
                'actor_model_state_dict': self.actor_net.state_dict(),
                'critic_model_state_dict': self.critic_net.state_dict(),
            }, self.args.MODEL_PATH)

    def load_model(self):
        checkpoint = torch.load(self.args.MODEL_PATH)
        self.actor_net.load_state_dict(checkpoint['actor_model_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_model_state_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_model_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_model_state_dict'])
        return checkpoint['epoch']
