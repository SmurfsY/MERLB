import argparse
from collections import namedtuple
from itertools import count

import os, sys, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return None


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []
        # print(ind)
        for i in ind:
            # print(self.storage[i])
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.max_action = max_action
        self.action_dim = action_dim

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a))  # * self.max_action

        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TD3():
    def __init__(self, config):

        self.device = config.device
        self.config = config
        self.actor = Actor(self.config.state_dim, self.config.action_dim, self.config.max_action).to(self.device)
        self.actor_target = Actor(self.config.state_dim, self.config.action_dim, self.config.max_action).to(self.device)
        self.critic_1 = Critic(self.config.state_dim, self.config.action_dim).to(self.device)
        self.critic_1_target = Critic(self.config.state_dim, self.config.action_dim).to(self.device)
        self.critic_2 = Critic(self.config.state_dim, self.config.action_dim).to(self.device)
        self.critic_2_target = Critic(self.config.state_dim, self.config.action_dim).to(self.device)

        random_seed(self.config.seed)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())


        self.memory = Replay_buffer(self.config.memory_capacity)
        # self.writer = SummaryWriter(self.config.model_saved_path)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def choose_action(self, state):
        # print(state)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        action = self.actor(state)
        action_result = torch.clamp((action + torch.randn_like(action) * 0.1), 0, self.config.max_action)
        return action_result.cpu().data.numpy().flatten()


    def choose_best_action(self, state):
        # print(state)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        action = self.actor(state)
        action = torch.clamp(action, 0, self.config.max_action)

        return action.cpu().data.numpy().flatten()

    def update(self, num_iteration):

        if self.num_training % 500 == 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")
        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(self.config.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            done = torch.FloatTensor(d).to(self.device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, self.config.policy_noise).to(self.device)
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.config.max_action, self.config.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.config.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()


            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates:
            actor_loss = 0
            if i % self.config.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- self.config.tau) * target_param.data) + self.config.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - self.config.tau) * target_param.data) + self.config.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - self.config.tau) * target_param.data) + self.config.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

        return loss_Q1 + loss_Q2, actor_loss

    def save(self):
        torch.save(self.actor.state_dict(), self.config.model_saved_path+'actor.pth')
        torch.save(self.actor_target.state_dict(), self.config.model_saved_path+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), self.config.model_saved_path+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), self.config.model_saved_path+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), self.config.model_saved_path+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), self.config.model_saved_path+'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(self.config.model_saved_path + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(self.config.model_saved_path + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(self.config.model_saved_path + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(self.config.model_saved_path + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(self.config.model_saved_path + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(self.config.model_saved_path + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")