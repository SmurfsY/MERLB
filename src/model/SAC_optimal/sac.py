import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.model.SAC_optimal.utils import soft_update, hard_update
from src.model.SAC_optimal.model import GaussianPolicy, QNetwork, DeterministicPolicy
from src.model.replay_memory import ReplayMemory

class SAC(object):
    def __init__(self, config):
        self.config = config
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.batch_size = config.batch_size

        self.memory = ReplayMemory(config.memory_capacity, config.seed)

        self.policy_type = config.SAC_policy
        self.target_update_interval = config.SAC_target_delay
        self.automatic_entropy_tuning = config.automatic_entropy_tuning

        self.device = torch.device("cuda" if config.device else "cpu")

        self.critic = QNetwork(config.state_dim, config.action_dim, config.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=config.learning_rate_c)

        self.critic_target = QNetwork(config.state_dim, config.action_dim, config.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(config.action_dim).to(self.device)).item()
                # self.target_entropy = self.alpha
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=config.learning_rate_a)

            self.policy = GaussianPolicy(config.state_dim, config.action_dim, config.hidden_size, action_space=None).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=config.learning_rate_a)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(config.state_dim, config.action_dim, config.hidden_size, action_space=None).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=config.learning_rate_a)

    def choose_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, num_iteration):
        Q1_loss_total = []
        Q2_loss_total = []
        pi_loss_tatal = []
        alpha_loss_total = []
        alpha_all = []

        for i in range(num_iteration):
            # Sample a batch from memory
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.config.batch_size)

            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + (1-mask_batch) * self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            pi, log_pi, _ = self.policy.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone() # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


            if (i+1) % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

            Q1_loss_total.append(qf1_loss.item())
            Q2_loss_total.append(qf2_loss.item())
            pi_loss_tatal.append(policy_loss.item())
            alpha_loss_total.append(alpha_loss.item())
            alpha_all.append(alpha_tlogs.item())


        return np.mean(Q1_loss_total), np.mean(Q2_loss_total), np.mean(pi_loss_tatal), np.mean(alpha_loss_total),\
               alpha_all

    # Save model parameters
    def save(self, path):

        torch.save(self.policy.state_dict(), path + 'actor_net.pth')
        torch.save(self.critic.state_dict(), path + 'critic_net.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    # Load model parameters
    def load(self, path):
        self.policy.load_state_dict(torch.load(path + 'actor_net.pth'))
        self.critic.load_state_dict(torch.load(path + 'critic_net.pth'))
        print("model has been load")

