import os
import logging
import time
import torch


from src.DRLB.DRLB_model import DRLB
from src.DRLB.train_eval import train_eval

class Config:
    def __init__(self, budget_para):
        # 地址
        self.budget_para = budget_para
        self.budget_para_int = int(budget_para)
        self.Data_path = '../../data/'
        self.data_set = 'ipinyou'
        self.campaign_id = '3476'
        self.total_budget = True
        self.data_log_path = self.Data_path + '/' + self.data_set + '/' + self.campaign_id + '/'
        self.train_set = self.data_log_path + '{}_data.csv'.format('new_train')  # new_train
        self.test_set = self.data_log_path + '{}_data.csv'.format('new_test')   # new_test
        self.model_saved_path = '../../model/' + self.data_set + '/' + self.campaign_id + '/DRLB/' + str(self.budget_para) +'/'
        self.result_path = '../../result/' + self.data_set + '/' + self.campaign_id + '/DRLB/' + str(self.budget_para) +'/'
        self.logging_file = '../../logging/' + self.data_set + '/' + self.campaign_id + '/DRLB/' + str(self.budget_para) +'/'

        #模型
        self.action_space = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]  # 动作空间
        self.action_numbers = len(self.action_space)  # 动作的数量
        self.feature_numbers = 7  # 状态的特征数量
        self.learning_rate = 0.001  # 学习率
        self.reward_decay = 1  # 奖励折扣因子,偶发过程为1
        self.e_greedy = 0.9  # 贪心算法ε
        self.replace_target_iter = 100  # 每300步替换一次target_net的参数
        self.memory_size = 100000 # 经验池的大小
        self.batch_size = 3  # 每次更新时从memory里面取多少数据出来，mini-batch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_id = 0
        self.seed = 250
        self.num_train_epochs = 1000

        # 日志
        self.logging2file = True
        self.save_action_lambda = False
def main(config):
    model=DRLB(config)
    best_model = train_eval(config, model)



if __name__ == "__main__":
    budget_paras = [2,4,8,16]
    for budget_para in budget_paras:
        config = Config(budget_para)

        if config.save_action_lambda:
            if not os.path.exists(config.result_path + 'train_record/'):
                os.makedirs(config.result_path + 'train_record/')
            if not os.path.exists(config.result_path + 'test_record/'):
                os.makedirs(config.result_path + 'test_record/')

        if config.logging2file == True:  #是否生成日志
            if not os.path.exists(config.logging_file):
                os.makedirs(config.logging_file)
            file = time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
            path = os.path.join(config.logging_file, file)
            logging.basicConfig(filename=path, format='%(levelname)s: %(message)s', level=logging.INFO)
        main(config)
