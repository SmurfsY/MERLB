import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class Replay_buffer():
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
    def __init__(self, state_dim, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mu_head = nn.Linear(128, 1)
        self.log_std_head = nn.Linear(128, 1)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC():
    def __init__(self, config):
        super(SAC, self).__init__()

        self.config =config
        self.policy_net = Actor(config.state_dim).to(config.device)
        self.value_net = Critic(config.state_dim).to(config.device)
        self.Target_value_net = Critic(config.state_dim).to(config.device)
        self.Q_net1 = Q(config.state_dim, config.action_dim).to(config.device)
        self.Q_net2 = Q(config.state_dim, config.action_dim).to(config.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate_a)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=config.learning_rate_c)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=config.learning_rate_c)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=config.learning_rate_c)

        self.memory = Replay_buffer(self.config.memory_capacity)
        self.num_transition = 0 # pointer of replay buffer
        self.num_training = 1


        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)


    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.config.device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action.item() # return a scalar, float32


    def evaluate(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(0, 1)

        z = noise.sample()

        action = torch.tanh(batch_mu + batch_sigma*z.to(self.config.device))
        log_prob = dist.log_prob(batch_mu + batch_sigma * z.to(self.config.device)) - torch.log(1 - action.pow(2) + self.config.min_Val)

        return action, log_prob, z, batch_mu, batch_log_sigma

    def update(self, num_iteration):
        if self.num_training % 500 == 0:
            print("Training ... {} times ".format(self.num_training))

        V_loss_total = 0
        Q_loss_total = 0
        pi_loss_tatal = 0
        for _ in range(num_iteration):
            #for index in BatchSampler(SubsetRandomSampler(range(self.config.capacity)), self.config.batch_size, False):
            x, y, u, r, d = self.memory.sample(self.config.batch_size)
            state = torch.FloatTensor(x).to(self.config.device)
            next_state = torch.FloatTensor(y).to(self.config.device)
            action = torch.FloatTensor(u).to(self.config.device)
            reward = torch.FloatTensor(r).to(self.config.device)
            done = torch.FloatTensor(d).to(self.config.device)

            target_value = self.Target_value_net(next_state)
            next_q_value = reward + (1 - done) * self.config.gamma * target_value

            excepted_value = self.value_net(state)
            excepted_Q1 = self.Q_net1(state, action)
            excepted_Q2 = self.Q_net2(state, action)
            sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(state)
            excepted_new_Q = torch.min(self.Q_net1(state, sample_action), self.Q_net2(state, sample_action))
            next_value = excepted_new_Q - log_prob

            # !!!Note that the actions are sampled according to the current policy,
            # instead of replay buffer. (From original paper)
            V_loss = self.value_criterion(excepted_value, next_value.detach()).mean()  # J_V
            V_loss_total += V_loss
            # Dual Q net
            Q1_loss = 0.5 * self.Q1_criterion(excepted_Q1, next_q_value.detach()).mean() # J_Q
            Q2_loss = 0.5 * self.Q2_criterion(excepted_Q2, next_q_value.detach()).mean()
            Q_loss_total += Q1_loss+Q2_loss

            pi_loss = (log_prob - excepted_new_Q).mean() # according to original paper
            pi_loss_tatal += pi_loss
            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.Q1_optimizer.zero_grad()
            Q1_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            Q2_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()

            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph = True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            # update target v net update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - self.config.tau) + param * self.config.tau)

            self.num_training += 1
        return V_loss_total+Q_loss_total, pi_loss_tatal

    def save(self):
        torch.save(self.policy_net.state_dict(), self.config.model_saved_path+'policy_net.pth')
        torch.save(self.value_net.state_dict(), self.config.model_saved_path+'value_net.pth')
        torch.save(self.Q_net1.state_dict(), self.config.model_saved_path+'Q_net1.pth')
        torch.save(self.Q_net2.state_dict(), self.config.model_saved_path+'Q_net2.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, path):
        if path:
            self.path = path
        else:
            self.path = self.config.model_saved_path
        self.policy_net.load_state_dict(torch.load(self.path+'policy_net.pth'))
        self.value_net.load_state_dict(torch.load( self.path+'value_net.pth'))
        self.Q_net1.load_state_dict(torch.load(self.path+'Q_net1.pth'))
        self.Q_net2.load_state_dict(torch.load(self.path+'Q_net2.pth'))
        print("model has been load")

