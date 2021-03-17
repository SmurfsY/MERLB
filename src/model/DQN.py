import os, sys, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from src.model.replay_memory import ReplayMemory

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)


class Net(nn.Module):
    def __init__(self, feature_numbers, action_nums):
        super(Net, self).__init__()

        deep_input_dims = feature_numbers
        self.bn_input = nn.BatchNorm1d(deep_input_dims)
        self.bn_input.weight.data.fill_(1)
        self.bn_input.bias.data.fill_(0)

        neuron_nums = [128, 256, 128]
        self.mlp = nn.Sequential(
            nn.Linear(deep_input_dims, neuron_nums[0]),
            # nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            # nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], neuron_nums[2]),
            # nn.BatchNorm1d(neuron_nums[2]),
            nn.ReLU(),
            nn.Linear(neuron_nums[2], action_nums)
        )

    def forward(self, input):
        actions_value = self.mlp(self.bn_input(input))
        return actions_value


class RewardNet(nn.Module):
    def __init__(self, state_nums, action_nums):
        super(RewardNet, self).__init__()
        self.state_dim = state_nums
        self.action_dim = action_nums

        self.fc1 = nn.Linear(state_nums + action_nums, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state_dim, action_dim):
        x = torch.cat([state_dim, action_dim], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQN:
    def __init__(
            self,config
    ):
        self.action_space = config.action_space
        self.action_nums = config.action_numbers  # 动作的具体数值？[0,0.01,...,budget]
        self.state_nums = config.state_dim
        self.lr = config.learning_rate_a
        self.gamma = config.reward_decay
        self.epsilon_max = config.e_greedy  # epsilon 的最大值
        self.replace_target_iter = config.replace_target_iter  # 更换 target_net 的步数
        self.memory_size = config.memory_capacity  # 记忆上限
        self.batch_size = config.batch_size  # 每次更新时从 memory 里面取多少记忆出来
        self.epsilon = 0.9
        self.device = config.device

        if not os.path.exists('result'):
            os.mkdir('result')

        # hasattr(object, name)
        # 判断一个对象里面是否有name属性或者name方法，返回BOOL值，有name特性返回True， 否则返回False。
        # 需要注意的是name要用括号括起来


        # 记录学习次数（用于判断是否替换target_net参数）
        self.learn_step_counter = 0

        # 将经验池<状态-动作-奖励-下一状态>中的转换组初始化为0
        self.memory = ReplayMemory(self.memory_size, config.seed)
        self.Reward_memory = ReplayMemory(self.memory_size, config.seed)

        # 创建奖励函数网络
        self.Reward_net = RewardNet(self.state_nums, 1).to(self.device)
        # 创建target_net（目标神经网络），eval_net（训练神经网络）
        self.eval_net = Net(self.state_nums, self.action_nums).to(self.device)

        self.target_net = copy.deepcopy(self.eval_net)
        # 优化器
        self.Q_optimizer = torch.optim.RMSprop(self.eval_net.parameters(), momentum=0.95, lr=self.lr)
        self.reward_optimizer = torch.optim.Adam(self.Reward_net.parameters(), lr=self.lr)
        # 损失函数为，均方损失函数
        self.loss_func = nn.MSELoss()

        self.cost_his = []  # 记录所有的cost变化，plot画出

    # 重置epsilon
    def reset_epsilon(self, e_greedy):
        self.epsilon = e_greedy

    # 选择动作
    def choose_action(self, state):
        torch.cuda.empty_cache()
        # 统一 state 的 shape, torch.unsqueeze()这个函数主要是对数据维度进行扩充
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)

        random_probability = max(self.epsilon, 0.5) # 论文的取法
        self.eval_net.eval()
        with torch.no_grad():
            if np.random.uniform() > random_probability:
                # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
                actions_value = self.eval_net.forward(state)
                # torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor),按维度dim 返回最大值
                # torch.max(a,1) 返回每一行中最大值的那个元素，且返回索引（返回最大元素在这一行的行索引）
                action_index = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
                action = self.action_space[action_index]  # 选择q_eval值最大的那个动作
            else:
                index = np.random.randint(0, self.action_nums)
                action = self.action_space[index]  # 随机选择动作
        self.eval_net.train()

        return action

    # 选择最优动作
    def choose_best_action(self, state):
        # 统一 state 的 shape (1, size_of_state)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)

        self.eval_net.eval()
        with torch.no_grad():
            actions_value = self.eval_net.forward(state)
            action_index = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            action = self.action_space[action_index]  # 选择q_eval值最大的那个动作
        return action

    # 定义DQN的学习过程

    def get_reward(self, batch_state, batch_action):
        state = torch.FloatTensor(batch_state).to(self.device).unsqueeze(0)
        action = list([batch_action])
        action = torch.FloatTensor(action).to(self.device).unsqueeze(0)

        return self.Reward_net(state, action).detach().cpu().numpy()[0]

    def learn_Q(self, update_num):

        avg_loss = []
        for i in range(update_num):
            # update the parameters
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter += 1

            # sample batch from memory
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)
            batch_state = torch.FloatTensor(state_batch).to(self.device)
            batch_next_state = torch.FloatTensor(next_state_batch).to(self.device)
            batch_action = torch.LongTensor(action_batch).to(self.device).unsqueeze(1)
            batch_reward = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)


            # q_eval
            # print(batch_state.size())
            q_eval_all_value = self.eval_net(batch_state)
            # print(q_eval_all_value.size())
            q_eval = q_eval_all_value.gather(1, batch_action)
            q_next = self.target_net(batch_next_state)

            q_next_value = q_next.max(1)[0].view(self.batch_size, 1)
            q_target = batch_reward + self.gamma * (1- mask_batch ) * q_next_value
            # print('q_eval:', q_eval)
            # print('q_target:', q_target)
            # print('i:',i)
            loss = self.loss_func(q_eval, q_target)

            self.Q_optimizer.zero_grad()
            loss.backward()
            self.Q_optimizer.step()
            avg_loss.append(loss.item())

        return loss.item()

    def learn_reward(self):
        # sample batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.Reward_memory.sample(batch_size=self.batch_size, out=False)
        batch_state = torch.FloatTensor(state_batch).to(self.device)
        batch_next_state = torch.FloatTensor(next_state_batch).to(self.device)
        batch_action = torch.FloatTensor(action_batch).to(self.device).unsqueeze(1)
        batch_reward = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)


        model_reward = self.Reward_net(batch_state, batch_action)

        # print(model_reward.shape, batch_reward.shape)
        loss = self.loss_func(model_reward, batch_reward)

        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()

        return loss.item()

    def control_epsilon(self, rate):
        # 逐渐增加epsilon，增加行为的利用性
        r_epsilon = 2e-5  # 降低速率
        self.epsilon = max(0.95 * rate, 0.05)

    def save(self, path):

        torch.save(self.eval_net.state_dict(), path + 'q_eval_net.pth')

        print("====================================")
        print("Model has been saved...")
        print("====================================")

    # Load model parameters
    def load(self, path):
        self.eval_net.load_state_dict(torch.load(path + 'q_eval_net.pth'))
        print("model has been load")