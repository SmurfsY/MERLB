

import pandas as pd

def get_init_lambda(config):
    HB_result = pd.read_csv(
        '../../result/ipinyou/{}/HB/original/best/best_new_train_data_lin_bid.csv'.format(config.campaign_id))
    # HB_result = pd.read_csv('../../result/ipinyou/{}/HB/FAB_BASE_BID/best/best_train_data_lin_bid.csv'.format(config.campaign_id))
    HB_DATA = HB_result[HB_result.budget_para.isin([config.budget_para_int])].reset_index()
    HB_bid = HB_DATA.loc[0,'base_bid_price']
    init_lambda = HB_result['average_ctr'][0] / HB_bid


    return init_lambda

def statistics(remain_budget, origin_t_spent, origin_t_win_imps,
               origin_t_auctions, origin_t_clks, origin_reward_t, origin_profit_t,  auc_t_datas, bid_arrays, t):
    cpc = 30000
    market_price_list = list(auc_t_datas.market_price)
    clk_list = list(auc_t_datas.clk)
    pctr_list = list(auc_t_datas.pctr)
    if remain_budget > 0:
        if remain_budget - origin_t_spent <= 0 :
            temp_t_auctions = 0
            temp_t_spent = 0
            temp_t_win_imps = 0
            temp_reward_t = 0
            temp_t_clks = 0
            temp_profit_t = 0
            temp_budget = remain_budget
            for i in range(len(auc_t_datas)):
                temp_t_auctions += 1
                if temp_budget - temp_t_spent >= 0:
                    if market_price_list[i] <= bid_arrays[i]:
                        temp_t_spent += market_price_list[i]
                        temp_t_win_imps += 1
                        temp_t_clks += clk_list[i]
                        temp_profit_t += (pctr_list[i] * cpc - market_price_list[i])
                        temp_reward_t += pctr_list[i]
                        temp_budget -= market_price_list[i]
                else:
                    continue
            t_auctions = temp_t_auctions
            t_spent = temp_t_spent if temp_t_spent > 0 else 0
            t_win_imps = temp_t_win_imps
            t_clks = temp_t_clks
            reward_t = temp_reward_t
            profit_t = temp_profit_t
        else:
            t_spent, t_win_imps, t_auctions, t_clks, reward_t, profit_t \
                = origin_t_spent, origin_t_win_imps, origin_t_auctions, origin_t_clks, origin_reward_t, origin_profit_t
    else:
        t_auctions = 0
        t_spent = 0
        t_win_imps = 0
        reward_t = 0
        t_clks = 0
        profit_t = 0

    return t_win_imps, t_spent, t_auctions, reward_t, t_clks, profit_t