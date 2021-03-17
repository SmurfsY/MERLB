import random
import os
import numpy as np
import logging
import time
import torch
# from src.model.DIAYN import DIAYN
# from src.model.SAC import SAC
from src.model.SAC_optimal.sac import SAC
from src.SAC_AUTO_BID.normal_allocate_budget.train_eval_impression_new import model_train
# from src.SAC_AUTO_BID.train_eval_impression import model_train


class Config:
    def __init__(self, model_type):
        self.campaign_id = '3427'
        self.mode = 'train'
        self.model_type = model_type    # TD3/DIAYN/DQN
        self.mode_save_type = 'pctr' # pctr,clk,reward
        # 出价数据
        self.reward_type = '01456'   # action_type:1,2,3
        self.budget_para_int = 16
        self.budget_para = str(self.budget_para_int)   # budget_para
        self.budget_total = 36000000 / self.budget_para_int
        self.fraction_type = 24
        #预先1000个进行分配budget
        self.fraction_cost = 1000
        self.budget_allocate = True

        # 动作存在梯度
        self.trend_action = False

        # 第一高价
        self.first_bid = True

        #动作区间区分
        self.action_gap = False

        self.bid_function = 'refine'  # refine, direct

        self.budget_allocate_num  = 1000
        # set路径选择
        if self.action_gap:
            self.action_gap_path = 'gap_len'
        else:
            self.action_gap_path = 'no_gap_len'

        if self.budget_allocate:
            self.budget_allocate_path = 'PRE'
        else:
            self.budget_allocate_path = 'NORMAL'
        if self.trend_action:
            self.trend_action_path = 'trend'
        else:
            self.trend_action_path = 'no_trend'
        if self.first_bid:
            self.first_bid_path = 'first'
        else:
            self.first_bid_path = 'second'

        # 奖励函数
        self.reward_function = 'refine_reward'  # real_pctr,max_set，newest11-4,9-2,real_pctr_right, refine_reward
        self.set_path = self.action_gap_path+'/'+self.budget_allocate_path+'/'+self.trend_action_path+'/'+\
                        self.first_bid_path+'/'+self.bid_function+'/'+self.reward_function+'/'



        # 地址
        self.Data_path = '../../data/'
        self.data_set = 'ipinyou'


        self.train_day = 6
        self.test_day = 7

        self.data_log_path = self.Data_path + '/' + self.data_set + '/' + self.campaign_id + '/'
        self.train_set = self.data_log_path + '{}_data.csv'.format('new_train')
        self.test_set = self.data_log_path + '{}_data.csv'.format('new_test')
        # self.train_set = self.data_log_path + '{}_data.csv'.format(self.train_day)
        # self.test_set = self.data_log_path + '{}_data.csv'.format(self.test_day)


        self.model_saved_path = '../../model/' + self.data_set + '/' + self.campaign_id + '/' + \
                                self.model_type + '/' + self.budget_para + '/'

        self.best_model_save_path = self.model_saved_path + 'best_{}_model/'.format(self.mode_save_type)


        self.result_path = '../../result/' + self.data_set + '/' + self.campaign_id + '/' + \
                           self.model_type + '/' + self.budget_para + '/'
        self.logging_file = '../../logging/' + self.data_set + '/' + self.campaign_id + '/' + \
                            self.model_type + '/' + self.budget_para + '/' + self.set_path
        self.heuristic_path = '../../result/' + self.data_set + '/' + self.campaign_id + '/' + 'best_bid_result/'
        self.replay_memory_path = self.result_path + 'replay_memory/'
        self.result_train_action_path = self.result_path + self.set_path + 'train_action/'
        self.result_test_action_path = self.result_path + self.set_path + 'test_action/'
        self.reward_pi_loss_q1_loss_q2_loss = self.result_path + self.set_path + 'figures_items/'

        # 模型
        self.learning_rate_a = 3e-4
        self.learning_rate_c = 3e-4
        self.hidden_size = 128
        self.reward_decay = 1.0
        self.tau = 0.0005
        self.gamma = 1.0
        self.noise_clip = 1
        self.policy_noise = 0.2
        self.dropout = 0.2
        self.alpha = 0.2
        # 数据方面
        self.ob_impression = 30000
        self.pctr_update_item = 1   #平均pctr更新频率


        # SAC
        self.SAC_policy = "Gaussian"
        self.automatic_entropy_tuning = True
        self.SAC_target_delay = 1


        self.state_dim = len(self.reward_type)

        self.action_dim = 1
        self.min_Val = torch.tensor(1e-7).float()

        self.max_action = 1.0
        self.memory_capacity = 1000000
        self.num_update = 128    #   训练更新轮数  64*512 in memory
        self.policy_delay = 4


        # self.test_episode = 1
        # self.ob_episode = 1

        # 训练
        self.seed = 250
        self.batch_size = 256
        self.num_train_epochs = 50    # 20     #训练轮数

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_id = 0

        # 输出
        self.test = True                                                                   #输出测试集预测
        self.model_saved = True                                                            #是否保存模型
        self.logging2file = True
        self.save_action = True

        #   DQN
        self.action_space = list(np.arange(0, 300))  # 动作空间
        self.action_numbers = len(self.action_space)  # 动作的数量
        self.e_greedy = 0.9
        self.replace_target_iter = 3


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return None


def main(config):
    model = SAC(config)
    random_seed(config.seed)
    logging.info(model)
    if config.budget_allocate:
        print('pre allocate budget')
    else:
        print('normal allocate')


    model_trained = model_train(model, config)
    if config.model_saved:
        model_trained.save('final_model/')


if __name__ == "__main__":
    config = Config('SAC')
    # print('test')

    if not os.path.exists(config.best_model_save_path): #创建最优模型存储路径
        os.makedirs(config.best_model_save_path)



    if not os.path.exists(config.result_path):  #创建结果目录
        os.makedirs(config.result_path)
    if not os.path.exists(config.replay_memory_path):   #创建replay_memory路径
        os.makedirs(config.replay_memory_path)
    if not os.path.exists(config.result_train_action_path): #创建train中action路径
        os.makedirs((config.result_train_action_path))
    if not os.path.exists(config.result_test_action_path):  #创建test中action路径
        os.makedirs((config.result_test_action_path))
    if not os.path.exists(config.reward_pi_loss_q1_loss_q2_loss): # 创建画图数据路径
        os.makedirs(config.reward_pi_loss_q1_loss_q2_loss)
    if not os.path.exists(config.logging_file): #创建日志路径
        os.makedirs(config.logging_file)




    if config.logging2file == True:  #是否生成日志
        if not os.path.exists(config.logging_file):
            os.makedirs(config.logging_file)
        file = time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
        path = os.path.join(config.logging_file, file)
        print('log has been open')
        logging.basicConfig(filename=path, format='%(levelname)s: %(message)s', level=logging.INFO)
    main(config)
