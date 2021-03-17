import pandas as pd
import numpy as np


def get_init_lambda(config):
    HB_result = pd.read_csv(
        '../../result/ipinyou/{}/HB/new_original/normal/second/normalbest_new_test_data_lin_bid.csv'.format(config.campaign_id))
    # HB_result = pd.read_csv('../../result/ipinyou/{}/HB/FAB_BASE_BID/best/best_train_data_lin_bid.csv'.format(config.campaign_id))
    HB_DATA = HB_result[HB_result.budget_para.isin([config.budget_para_int])].reset_index()
    HB_bid = HB_DATA.loc[0,'base_bid_price']
    # init_lambda = HB_result['average_ctr'][0] / HB_bid


    return HB_bid

# state_0:平均pctr，state_1:pctr，state_2:近1000的花费，state_4:剩余预算
# def get_reward(bid_flag, pctrs, state_0, state_1, state_2, state_4, Min_Threshold_pctr,
#                Max_Threshold_pctr, impression_index, ctr):            #前几个傻子奖励函数要
def get_reward(bid_flag, pctrs, state_1, state_4, Min_Threshold_pctr, Max_Threshold_pctr, impression_index,
               state_0, ctr, bid, HB_base_bid, config, origin_bid, market_prices, action):

    #
    # '''8-20版本奖励函数'''
    # if bid_flag == 0:
    #     reward = 0
    # elif bid_flag == 'win_imp':
    #     if pctrs[impression_index] < Min_Threshold_pctr:
    #         reward = -pctrs[impression_index] * (1-state_4)      # 赢得impression时判断pctr与min，低于进行惩罚，高于奖励，通过剩余预算率调整
    #     elif pctrs[impression_index] < Max_Threshold_pctr:                                      # 高于min时，*剩余预算率，低于时/剩余预算率
    #         reward = -(1-state_4) / 10
    #     else:
    #         reward = pctrs[impression_index] * state_4
    # elif bid_flag == 'win_clk':
    #     # reward = state_1*100
    #     reward = 1
    # elif bid_flag == 'lose_imp':
    #     if pctrs[impression_index] < Min_Threshold_pctr:
    #         reward = pctrs[impression_index] * state_4
    #     elif pctrs[impression_index] < Max_Threshold_pctr:
    #         reward = (1-state_4) * pctrs[impression_index] / 10
    #     else:
    #         reward = -pctrs[impression_index] * state_4
    # elif bid_flag == 'lose_clk':
    #     # reward = -state_1*100
    #     reward = -1

    #
    '''9-2版本奖励函数'''



    '''9-2版本2奖励函数——有点击的imp也区分开来, 修改了早停时的惩罚函数大小'''
    # impression_state = state_1 / state_0
    # if bid_flag == 0:
    #     reward = 0
    # elif bid_flag == 'win_imp':
    #     if pctrs[impression_index] < Min_Threshold_pctr:
    #         reward = -(1 - state_4) / 10  # 赢得impression时判断pctr与min，低于进行惩罚，高于奖励，通过剩余预算率调整
    #     elif pctrs[impression_index] < Max_Threshold_pctr:  # 高于min时，*剩余预算率，低于时/剩余预算率
    #         reward = -(1 - state_4) * pctrs[impression_index]
    #     else:
    #         reward = pctrs[impression_index]
    # elif bid_flag == 'win_clk':
    #     reward = impression_state * pctrs[impression_index] / ctr
    # elif bid_flag == 'lose_imp':
    #     if pctrs[impression_index] < Min_Threshold_pctr:
    #         reward = (1 - state_4) / 10
    #     elif pctrs[impression_index] < Max_Threshold_pctr:
    #         reward = (1 - state_4) * pctrs[impression_index]
    #     else:
    #         reward = -pctrs[impression_index]
    # elif bid_flag == 'lose_clk':
    #     # reward = -state_1*100
    #     reward = -impression_state * pctrs[impression_index] / ctr
    # print('reward:9-2')

    # '''9-10版本  除了win_clk  其余全部负奖励'''
    # if bid_flag == 0:
    #     reward = 0
    # elif bid_flag == 'win_imp':
    #     if pctrs[impression_index] < Min_Threshold_pctr:
    #         reward = -(1 - state_4) / 10    # 赢得impression时判断pctr与min，低于进行惩罚，高于奖励，通过剩余预算率调整
    #     elif pctrs[impression_index] < Max_Threshold_pctr:                                      # 高于min时，*剩余预算率，低于时/剩余预算率
    #         reward = -(1 - state_4) * pctrs[impression_index]
    #     else:
    #         reward = pctrs[impression_index]
    # elif bid_flag == 'win_clk':
    #     # reward = state_1*100
    #     reward = pctrs[impression_index] / ctr
    # elif bid_flag == 'lose_imp':
    #     if pctrs[impression_index] < Min_Threshold_pctr:
    #         reward = (1 - state_4) / 10
    #     elif pctrs[impression_index] < Max_Threshold_pctr:
    #         reward = (1 - state_4) * pctrs[impression_index]
    #     else:
    #         reward = -pctrs[impression_index]
    # elif bid_flag == 'lose_clk':
    #     # reward = -state_1*100
    #     reward = -pctrs[impression_index] / ctr




    if config.reward_function == 'max_set':         # MAX set
        if bid_flag == 0:
            reward = 0
        elif bid_flag == 'spend_out':
            reward = -pctrs[impression_index]
        elif bid_flag == 'win_imp':
            if pctrs[impression_index] < Min_Threshold_pctr:
                reward = -pctrs[impression_index]  # 赢得impression时判断pctr与min，低于进行惩罚，高于奖励，通过剩余预算率调整
            elif pctrs[impression_index] < Max_Threshold_pctr:  # 高于min时，*剩余预算率，低于时/剩余预算率
                reward = 0
            else:
                reward = pctrs[impression_index] * state_4

        elif bid_flag == 'win_clk':
            reward = state_1

        elif bid_flag == 'lose_imp':
            if pctrs[impression_index] < Min_Threshold_pctr:
                reward = pctrs[impression_index] * state_4
            elif pctrs[impression_index] < Max_Threshold_pctr:
                reward = 0
            else:
                reward = -pctrs[impression_index] * state_4

        elif bid_flag == 'lose_clk':
            reward = -state_1
    elif config.reward_function == 'real_pctr':     # 纯pctr作为奖励函数
        if bid_flag == 0:
            reward = 0
        elif bid_flag == 'win_imp':
            reward = pctrs[impression_index]

        elif bid_flag == 'win_clk':
            reward = pctrs[impression_index]

        elif bid_flag == 'lose_imp':
            reward = -pctrs[impression_index]

        elif bid_flag == 'lose_clk':
            reward = -pctrs[impression_index]
    elif config.reward_function == 'newest11-4':        # 新编奖励函数
        if bid_flag == 0:
            reward = 0
        elif bid_flag == 'win_imp':
            if pctrs[impression_index] < Min_Threshold_pctr:
                reward = -pctrs[impression_index]  # 赢得impression时判断pctr与min，低于进行惩罚，高于奖励，通过剩余预算率调整
            elif pctrs[impression_index] < Max_Threshold_pctr:  # 高于min时，*剩余预算率，低于时/剩余预算率
                reward = 0
            else:
                reward = pctrs[impression_index] * state_4

        elif bid_flag == 'win_clk':
            reward = state_1

        elif bid_flag == 'lose_imp':
            if pctrs[impression_index] < Min_Threshold_pctr:
                reward = pctrs[impression_index] * state_4
            elif pctrs[impression_index] < Max_Threshold_pctr:
                reward = 0
            else:
                reward = -pctrs[impression_index] * state_4

        elif bid_flag == 'lose_clk':
            reward = -state_1
    elif config.reward_function == '9-2':   #9-2奖励函数
        if bid_flag == 0:
            reward = 0
        elif bid_flag == 'win_imp':
            if pctrs[impression_index] < Min_Threshold_pctr:
                reward = -(1 - state_4) / 10    # 赢得impression时判断pctr与min，低于进行惩罚，高于奖励，通过剩余预算率调整
            elif pctrs[impression_index] < Max_Threshold_pctr:                                      # 高于min时，*剩余预算率，低于时/剩余预算率
                reward = -(1 - state_4) * pctrs[impression_index]
            else:
                reward = pctrs[impression_index]
        elif bid_flag == 'win_clk':
            # reward = state_1*100
            reward = pctrs[impression_index] / ctr
        elif bid_flag == 'lose_imp':
            if pctrs[impression_index] < Min_Threshold_pctr:
                reward = (1 - state_4) / 10
            elif pctrs[impression_index] < Max_Threshold_pctr:
                reward = (1 - state_4) * pctrs[impression_index]
            else:
                reward = -pctrs[impression_index]
        elif bid_flag == 'lose_clk':
            # reward = -state_1*100
            reward = -pctrs[impression_index] / ctr
    elif config.reward_function == 'real_pctr_right':
        if bid_flag == 0:
            reward = 0
        elif bid_flag == 'spend_out':
            reward = 0
        elif bid_flag == 'win_imp':
            reward = pctrs[impression_index]

        elif bid_flag == 'win_clk':
            reward = pctrs[impression_index]

        elif bid_flag == 'lose_imp':
            reward = 0

        elif bid_flag == 'lose_clk':
            reward = 0
    elif config.reward_function == 'refine_reward':
        # print('reward function: refine reward')
        if bid_flag == 0:
            reward = 0
        elif bid_flag == 'spend_out':
            reward = -pctrs[impression_index]
        # 基于调整的奖励函数
        # print('original bid:', origin_bid)
        # print('market price:', market_prices)
        # print('bid:', bid)
        if origin_bid >= market_prices and bid < market_prices:
            reward = pctrs[impression_index] * action

        elif origin_bid >= market_prices and bid >= market_prices:
            reward = pctrs[impression_index] * state_4 / (abs((bid - origin_bid)) + 1)
            # reward = pctrs[impression_index] * state_4 * (1-abs(action))
        elif origin_bid < market_prices and bid < market_prices:
            reward = pctrs[impression_index] * (action - 1)
        elif origin_bid < market_prices and bid >= market_prices:
            reward = pctrs[impression_index] * action



    # 10-26修改中间的奖励
    # if bid_flag == 0:
    #     reward = 0
    # elif bid_flag == 'win_imp':
    #     if pctrs[impression_index] < Min_Threshold_pctr:
    #         reward = -pctrs[impression_index]  # 赢得impression时判断pctr与min，低于进行惩罚，高于奖励，通过剩余预算率调整
    #     elif pctrs[impression_index] < Max_Threshold_pctr:  # 高于min时，*剩余预算率，低于时/剩余预算率
    #         reward = -(1 - state_4) * pctrs[impression_index]
    #     else:
    #         reward = pctrs[impression_index] * state_4
    #
    # elif bid_flag == 'win_clk':
    #     reward = state_1
    #
    # elif bid_flag == 'lose_imp':
    #     if pctrs[impression_index] < Min_Threshold_pctr:
    #         reward = pctrs[impression_index] * state_4
    #     elif pctrs[impression_index] < Max_Threshold_pctr:
    #         reward = -(1 - state_4) * pctrs[impression_index]
    #     else:
    #         reward = -pctrs[impression_index] * state_4
    #
    # elif bid_flag == 'lose_clk':
    #     reward = -state_1

    # 10-26真正修改关于action的奖励函数
    # if bid_flag == 0:
    #     reward = 0
    # elif bid_flag == 'win_imp':
    #     if pctrs[impression_index] < Min_Threshold_pctr:
    #         reward = -pctrs[impression_index]  # 赢得impression时判断pctr与min，低于进行惩罚，高于奖励，通过剩余预算率调整
    #     elif pctrs[impression_index] < Max_Threshold_pctr:  # 高于min时，*剩余预算率，低于时/剩余预算率
    #         reward = -(1 - state_4) * pctrs[impression_index]
    #     else:
    #         reward = pctrs[impression_index] * state_4
    #
    # elif bid_flag == 'win_clk':
    #     reward = state_1
    #
    # elif bid_flag == 'lose_imp':
    #     if pctrs[impression_index] < Min_Threshold_pctr:
    #         reward = pctrs[impression_index] * state_4
    #     elif pctrs[impression_index] < Max_Threshold_pctr:
    #         reward = -(1 - state_4) * pctrs[impression_index]
    #     else:
    #         reward = -pctrs[impression_index] * state_4
    #
    # elif bid_flag == 'lose_clk':
    #     reward = -state_1

    if config.trend_action:
        hb_bid = pctrs[impression_index] / state_0 * HB_base_bid
        hb_bid = int(np.where(hb_bid >= 300, 300, hb_bid))
        adaptiv_x = abs(hb_bid-bid)
        # print(adaptiv_x)
        reward = reward / adaptiv_x if adaptiv_x != 0 else reward

    return reward
