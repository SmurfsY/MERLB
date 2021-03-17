import os, sys, random
import numpy as np
from src.DRLB.DRLB_get_function.get_time_step_result import statistics

def bid_func(auc_pCTRS, lamda):
    return auc_pCTRS/ lamda

def get_state(data, lambda_t, time_t, B_t, current_budget):
    cpc = 30000
    ROL_t = len(data)
    # print(ROL_t)
    if time_t == 0:
        remain_budget = B_t[0]
    else:
        remain_budget = B_t[time_t - 1]


    if ROL_t == 0:
        t_spend = 0
        t_win_imps = 0
        t_real_clk = 0
        t_win_clk = 0
        t_profit = 0
        t_reward = 0
        immediate_reward = 0

    early_stop = False
    bid = bid_func(data.pctr, lambda_t)
    bid = np.where(bid>=300, 300, bid)
    win_data = data[data.loc[:,'market_price']<=bid]
    t_spend = np.sum(win_data.market_price)
    t_win_imps = len(win_data)
    t_real_clk = np.sum(data.clk)
    t_win_clk = np.sum(win_data.clk)
    t_profit = np.sum(win_data.pctr*cpc-win_data.market_price)
    t_reward = np.sum(win_data.pctr)
    immediate_reward = t_reward


    if t_spend > remain_budget:
        early_stop = True
        t_win_imps, t_spend, ROL_t, t_reward, t_win_clk, t_profit = statistics(remain_budget, t_spend, t_win_imps, ROL_t,
                                                                           t_win_clk, t_reward, t_profit, data, bid, time_t)

    B_t[time_t] = remain_budget - t_spend
    # print('WIN_IMPS', t_win_imps)
    # print('ROL_T', ROL_t)
    BCR_t = (B_t[time_t] - B_t[time_t-1]) / B_t[time_t-1] if B_t[time_t-1]!=0 else 0
    CPM_t = t_spend / t_win_imps if t_win_imps != 0 else 0
    WR_t = t_win_imps / ROL_t if ROL_t>0 else 0

    state = [(time_t+1), B_t[time_t]/current_budget, ROL_t, BCR_t, CPM_t, WR_t, t_reward]


    return state, B_t, early_stop, t_win_clk, t_win_imps, t_reward, ROL_t