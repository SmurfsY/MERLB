import sys
import random
import math
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def get_lin_bid(average_ctr, pctr, lin_para):
    bid_price = int((pctr * lin_para) / average_ctr)
    bid_price = bid_price if bid_price <= 300 else 300
    return bid_price

def store_action(config, best_hb,train_data, budget_para, train_len, all_train_avg_pctr):
    envir = best_hb[best_hb.budget_para.isin([budget_para])]
    base_bid_price = envir.iloc[0].base_bid_price
    if config.win_state:
        average_pctr = envir.iloc[0].average_ctr
        train_len = envir.iloc[0].win_imps
    else:
        average_pctr = all_train_avg_pctr

    test_bid_result, test_result = heuristic_bid_test(train_data, budget_para, base_bid_price, average_pctr, train_len)
    return test_bid_result,test_result

def heuristic_bid_test(test_data, budget_para, base_bid_price, average_pctr, train_len):
    # 分析数据
    real_imps = len(test_data)
    total_budget = []
    for index, day in enumerate(test_data.day.unique()):
        current_day_budget = np.sum(test_data[test_data.day.isin([day])].market_price)
        total_budget.append(current_day_budget)
    print(total_budget)
    budget = np.divide(total_budget, budget_para)

    # 数据统计
    real_clks = 0  # 真实点击
    win_clks = 0  # 赢得点击数
    win_imps = 0  # 赢标数
    win_pctr = 0  # 赢得pctr
    bids = 0
    all_spend = []
    end_time_all = []
    bid_action = []

    for day_index, day in enumerate(test_data.day.unique()):
        # 构造当前天的数据
        current_day_data = test_data[test_data.day.isin([day])]
        clks = list(current_day_data['clk'])
        pctrs = list(current_day_data['pctr'])
        market_prices = list(current_day_data['market_price'])
        time_frac = list(current_day_data['time_fraction'])
        minutes = list(current_day_data['minutes'])
        # 当天数据每天初始化
        spend = 0  # 花费
        early_stop = False

        today_budget = budget[day_index]
        try:
            with tqdm(range(len(current_day_data))) as tdqm_t:
                for impression_index in tdqm_t:

                    bids += 1
                    real_clks += clks[impression_index]
                    bid = get_lin_bid(average_pctr, pctrs[impression_index], base_bid_price)
                    if bid > market_prices[impression_index] and spend + bid <= today_budget:
                        # 赢标奖励
                        win_pctr += pctrs[impression_index]
                        win_imps += 1
                        # 赢标点击特征
                        if clks[impression_index] == 1:
                            win_clks += 1
                        spend += market_prices[impression_index]

                        #修改平均pctr
                        total_pctr = average_pctr * train_len + pctrs[impression_index]
                        train_len = train_len + 1
                        average_pctr = total_pctr / train_len

                        bid_action.append([pctrs[impression_index], bid, market_prices[impression_index],
                                           clks[impression_index], minutes[impression_index]])
                    # 记录第一次预算不够的早停情况
                    if not early_stop:
                        if spend + bid > today_budget:
                            early_stop = time_frac[impression_index]

        except KeyboardInterrupt:
            tdqm_t.close()
            raise
        tdqm_t.close()
        # 当天结束后进行统计
        all_spend.append(spend)

        if not early_stop:
            end_time_all.append('{}F'.format(day))
        else:
            end_time_all.append(str(day) + '_' + str(early_stop))
    cpm = (np.sum(spend) / bids) if bids > 0 else 0

    test_bid_result = pd.DataFrame(data=bid_action, columns=['pctr', 'bid', 'market_price', 'clk', 'minutes'])
    test_result= [average_pctr, budget_para, win_clks, real_clks, bids, win_imps, real_imps, budget, all_spend,
                  cpm, base_bid_price, win_pctr, end_time_all]

    return test_bid_result, test_result

def heuristic_bid_train(train_data, total_budget, budget_para, lin_para):
    # 分析数据
    real_imps = len(train_data)
    average_pctr = np.mean(train_data.pctr)
    budget = np.divide(total_budget, budget_para)

    # 数据统计
    real_clks = 0  # 真实点击
    win_clks = 0  # 赢得点击数
    win_imps = 0  # 赢标数
    win_pctr = 0 # 赢得pctr
    bids = 0
    all_spend = []
    end_time_all = []


    for day_index, day in enumerate(train_data.day.unique()):
        # 构造当前天的数据
        current_day_data = train_data[train_data.day.isin([day])]
        clks = list(current_day_data['clk'])
        pctrs = list(current_day_data['pctr'])
        market_prices = list(current_day_data['market_price'])
        time_frac = list(current_day_data['time_fraction'])


        # 当天数据每天初始化
        spend = 0  # 花费
        early_stop = False

        today_budget = budget[day_index]
        try:
            with tqdm(range(len(current_day_data))) as tdqm_t:
                for impression_index in tdqm_t:

                    bids += 1
                    real_clks += clks[impression_index]
                    bid = get_lin_bid(average_pctr, pctrs[impression_index], lin_para)
                    if bid > market_prices[impression_index] and spend + bid <= today_budget:

                        # 赢标奖励
                        win_pctr += pctrs[impression_index]
                        win_imps += 1
                        # 赢标点击特征
                        if clks[impression_index] == 1:
                            win_clks += 1
                        spend += market_prices[impression_index]

                        #修改平均pctr
                        total_pctr = average_pctr * real_imps + pctrs[impression_index]
                        real_imps = real_imps + 1
                        average_pctr = total_pctr / real_imps

                    if not early_stop:  # 记录第一次预算不够的早停情况
                        if spend + bid > today_budget:
                            early_stop = time_frac[impression_index]
        except KeyboardInterrupt:
            tdqm_t.close()
            raise
        tdqm_t.close()
        # 当天结束后进行统计
        all_spend.append(spend)


        if not early_stop:
            end_time_all.append('{}F'.format(day))
        else:
            end_time_all.append(str(day) + '_' + str(early_stop))
    cpm = (np.sum(all_spend) / bids) if bids > 0 else 0


    return [average_pctr, budget_para, win_clks, real_clks, bids, win_imps, real_imps
        , budget, all_spend, cpm, lin_para, win_pctr,end_time_all]

def train_eval(config):

    print('reverse_type', config.reverse_type)
    total_budget = []

    train_data = pd.read_csv(config.train_data_path)
    train_data.sort_values(by='minutes', inplace=True)
    train_data['day'] = train_data['minutes'].apply(lambda x: int(str(x)[6:8]))

    for index, day in enumerate(train_data.day.unique()):
        current_day_budget = np.sum(train_data[train_data.day.isin([day])].market_price)
        total_budget.append(current_day_budget)
    print(total_budget)

    budget_proportions = [2, 4, 8, 16]
    lin_paras = np.arange(1, 300)
    train_result = []
    for budget_para in budget_proportions:
        for lin_para in lin_paras:
            print([budget_para, lin_para])
            train_result.append(heuristic_bid_train(train_data, total_budget, budget_para, lin_para))

    # 存储训练集中最优数据
    HB_bid_lin_PD = pd.DataFrame(data=train_result,
                                 columns=['average_ctr', 'budget_para', 'clks', 'real_clks', 'bids', 'win_imps',
                                          'real_imps'
                                     , 'budget', 'spend', 'cpm', 'base_bid_price', 'win_pctr','end_time'])
    # HB_bid_lin_PD.to_csv(HB_line_save_path + '{}_lin_bid.csv'.format(train_data_name), index=False)

    best_hb_base_bid_max = HB_bid_lin_PD.groupby(['budget_para']).apply(lambda x: x[x.clks == x.clks.max()])
    print('saving....')
    best_hb_base_bid_max.to_csv(config.HB_line_best_save_path + config.insert_type + 'best_{}_lin_bid.csv'.format(config.train_data_name),
                                index=False)
    # for budget_para in budget_proportions:
    #     train_bid_result, test_result = store_action(best_hb_base_bid_max, train_data, budget_para)
    #     train_bid_result.to_csv(config.HB_line_best_save_path + config.insert_type + '{}_{}_bid_action.csv'.format(
    #         budget_para, config.train_data_name), index=False)
    #

    print('**********************测试开始！！**********************')
    test_data = pd.read_csv(config.test_data_path)
    test_data.sort_values(by='minutes', inplace=True)
    test_results = []
    test_data['day'] = test_data['minutes'].apply(lambda x: int(str(x)[6:8]))

    train_len = len(train_data)
    all_train_avg_pctr = np.mean(train_data['pctr'])
    for budget_para in budget_proportions:
        test_bid_result, test_result = store_action(config, best_hb_base_bid_max, test_data, budget_para, train_len, all_train_avg_pctr)
        test_results.append(test_result)
        pd.DataFrame(data=test_results,
                     columns=['average_ctr', 'budget_para', 'clks', 'real_clks', 'bids', 'win_imps', 'real_imps'
                         , 'budget', 'spend', 'cpm', 'base_bid_price', 'win_pctr','end_time']).to_csv(
            config.HB_line_best_save_path + config.insert_type + 'best_{}_lin_bid.csv'.format(config.test_data_name), index=False)


class Config:
    def __init__(self):
        self.model = 'train'
        self.day = {
            '1458': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '2259': [19, 20, 21, 22, 23, 24, 25],
            '2261': [24, 25, 26, 27, 28],
            '2821': [21, 22, 23, 24, 25],
            '2997': [23, 24, 25, 26, 27],
            '3358': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3386': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3427': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3476': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        }
        self.Data_path = '../../data/'
        self.data_set = 'ipinyou/'
        self.campaign_id = '1458'

        self.train_data_name = 'new_train_data' #new_train_data
        self.test_data_name = 'new_test_data' #new_test_data

        self.result_path = '../../result/' + self.data_set + self.campaign_id + '/'
        self.data_log_path = '../../data/' + self.data_set + self.campaign_id + '/'
        self.heuristic_path = self.result_path + 'HB/'
        self.reverse_type = 'normal'
        self.win_state = False

        if self.win_state:
            win_state = 'win'
        else:
            win_state = 'all'

        if self.reverse_type == 'reverse':
            self.insert_type = self.reverse_type
            train_data_name = self.test_data_name
            test_data_name = self.train_data_name

        else:
            self.insert_type = 'normal'
            train_data_name = self.train_data_name
            test_data_name = self.test_data_name

        # path
        self.train_data_path = self.Data_path + self.data_set + self.campaign_id + '/' + train_data_name + '.csv'
        self.test_data_path = self.Data_path + self.data_set + self.campaign_id + '/' + test_data_name + '.csv'
        self.HB_line_save_path = self.result_path + 'HB/' + 'original/'
        self.HB_line_best_save_path = self.result_path + 'HB/' + '{}_dynamic_hb/'.format(win_state) + self.insert_type + '/'

if __name__ == '__main__':
    config = Config()
    if not os.path.exists(config.HB_line_save_path):
        os.makedirs(config.HB_line_save_path)
    if not os.path.exists(config.HB_line_best_save_path):
        os.makedirs(config.HB_line_best_save_path)
    train_eval(config)