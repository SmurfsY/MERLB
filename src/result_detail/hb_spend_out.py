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

def store_action(best_hb,train_data, budget_para):
    envir = best_hb[best_hb.budget_para.isin([budget_para])]
    base_bid_price = envir.iloc[0].base_bid_price
    average_pctr = envir.iloc[0].average_ctr
    test_bid_result, test_result = heuristic_bid_test(train_data, budget_para, base_bid_price, average_pctr)
    return test_bid_result,test_result

def heuristic_bid_test(test_data, budget_para, base_bid_price, average_pctr):
    # 分析数据
    real_imps = len(test_data)
    total_budget = []
    for index, day in enumerate(test_data.day.unique()):
        current_day_budget = np.sum(test_data[test_data.day.isin([day])].market_price)
        total_budget.append(current_day_budget)
    print(total_budget)
    budget = np.divide(total_budget, budget_para)

    # 数据统计
    day_win_clk = []
    day_win_imp = []
    day_win_pctr = []
    real_clks = 0  # 真实点击
    win_clks = 0  # 赢得点击数
    win_imps = 0  # 赢标数
    win_pctr = 0  # 赢得pctr
    bids = 0
    spend_out_lose_clk = [0,0,0]
    low_bid_lose_clk = [0,0,0]
    day_spend = []
    end_time_all = []
    bid_action = []
    print(test_data.day.unique())
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

        current_day_win_clk = 0
        current_day_win_imps = 0
        current_day_win_pctr = 0
        today_budget = budget[day_index]
        try:
            with tqdm(range(len(current_day_data))) as tdqm_t:
                for impression_index in tdqm_t:

                    bids += 1
                    real_clks += clks[impression_index]
                    bid = get_lin_bid(average_pctr, pctrs[impression_index], base_bid_price)
                    if spend + bid > today_budget:
                        if clks[impression_index] == 1:
                            spend_out_lose_clk[day_index] += 1
                    if spend + bid <= today_budget:
                        if clks[impression_index] == 1:
                            if bid <= market_prices[impression_index]:
                                low_bid_lose_clk[day_index] += 1
                    if bid > market_prices[impression_index] and spend + bid <= today_budget:

                        # 赢标奖励
                        win_pctr += pctrs[impression_index]
                        win_imps += 1
                        current_day_win_pctr += pctrs[impression_index]
                        current_day_win_imps  += 1

                        # 赢标点击特征
                        if clks[impression_index] == 1:

                            win_clks += 1
                            current_day_win_clk +=1

                        # 花费消耗
                        if config.bid_type == 'first':
                            spend += bid
                        elif config.bid_type == 'second':
                            spend += market_prices[impression_index]

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
        day_spend.append(spend)
        day_win_clk.append(current_day_win_clk)
        day_win_imp.append(current_day_win_imps)
        day_win_pctr.append(current_day_win_pctr)
        if not early_stop:
            end_time_all.append('{}F'.format(day))
        else:
            end_time_all.append(str(day) + '_' + str(early_stop))


    cpm = (np.sum(day_spend) / np.sum(day_win_imp)) if np.sum(day_win_imp) > 0 else 0
    cpc = (np.sum(day_spend) / np.sum(day_win_clk)) if np.sum(day_win_clk) > 0 else 0
    cpc_day = [cost / clk for cost,clk in zip(day_spend, day_win_clk)]
    cmp_day = [cost / imp for cost,imp in zip(day_spend, day_win_imp)]
    all_spend = np.sum(day_spend)

    test_bid_result = pd.DataFrame(data=bid_action, columns=['pctr', 'bid', 'market_price', 'clk', 'minutes'])
    test_result= [low_bid_lose_clk, spend_out_lose_clk, np.sum(day_win_clk), average_pctr, budget_para, real_clks, bids, win_imps, real_imps, budget, day_spend,
                  base_bid_price, win_pctr, end_time_all, day_win_clk, day_win_pctr,  cpc_day, cmp_day, cpm, cpc, all_spend]

    return test_bid_result, test_result

def heuristic_bid_train(train_data, total_budget, budget_para, lin_para):
    # 分析数据
    real_imps = len(train_data)
    average_pctr = np.sum(train_data.pctr) / real_imps
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

                        # 花费消耗
                        if config.bid_type == 'first':
                            spend += bid
                        elif config.bid_type == 'second':
                            spend += market_prices[impression_index]
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

    print('**********************测试开始！！**********************')
    budget_proportions = [2, 4, 8, 16]
    best_hb_base_bid_max = pd.read_csv(config.HB_line_best_save_path + config.insert_type + 'best_{}_lin_bid.csv'.format(config.train_data_name))
    test_data = pd.read_csv(config.test_data_path)
    test_data.sort_values(by='minutes', inplace=True)
    test_results = []
    test_data['day'] = test_data['minutes'].apply(lambda x: int(str(x)[6:8]))

    for budget_para in budget_proportions:
        test_bid_result, test_result = store_action(best_hb_base_bid_max, test_data, budget_para)
        test_results.append(test_result)
        # test_bid_result.to_csv(
        #     config.HB_line_best_save_path + config.insert_type + 'bin_lin_result_{}_{}.csv'.format(config.test_data_name, budget_para))

    print('save data')
    pd.DataFrame(data=test_results,
                 columns=['low_bid_lose_clk', 'spend_out_lose_clk','all_win_clk', 'average_pctr', 'budget_para', 'real_clks', 'bids', 'win_imps', 'real_imps', 'budget', 'all_spend',
              'base_bid_price', 'win_pctr', 'end_time_all', 'day_win_clk', 'day_win_pctr',  'cpc_day', 'cmp_day', 'cpm', 'cpc'
                          , 'all_spend']).to_csv(
        config.HB_line_best_save_path + config.insert_type + '_best_{}_lin_day_bid_without_budget.csv'.format(config.test_data_name), index=False)


class Config:
    def __init__(self):
        self.model = 'test'
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
        self.div_data = ['train', 'test']
        self.Data_path = '../../data/'
        self.data_set = 'ipinyou/'
        self.campaign_id = '3476'

        self.train_data_name = 'new_train_data' #new_train_data
        self.test_data_name = 'new_test_data' #new_test_data

        self.result_path = '../../result/' + self.data_set + self.campaign_id + '/'
        self.data_log_path = '../../data/' + self.data_set + self.campaign_id + '/'
        self.heuristic_path = self.result_path + 'HB/'
        self.reverse_type = 'normal'    #reverse,normal

        self.bid_type = 'second' #first,second

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
        self.HB_line_save_path = self.result_path + 'HB/' + 'new_original/'
        self.HB_line_best_save_path = self.result_path + 'HB/' + 'new_original/' + self.insert_type + '/' + self.bid_type + '/'

if __name__ == '__main__':
    config = Config()
    if not os.path.exists(config.HB_line_save_path):
        os.makedirs(config.HB_line_save_path)
    if not os.path.exists(config.HB_line_best_save_path):
        os.makedirs(config.HB_line_best_save_path)
    train_eval(config)