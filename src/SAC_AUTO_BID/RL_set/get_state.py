import numpy as np
import time
# 辣鸡奖励函数
# def get_next_state(train_data, impression_index, Min_Threshold_pctr, Max_Threshold_pctr,bid_pctrs,bid_nums,
#                    recent_cost_state, cost_index, recent_cost, config,cost,real_clks, win_clks,
#                    Budget_allocate_by_clk_rate, pctrs, time_frac, current_day, average_pctr, total_bid_pctrs, total_bid_nums):
#     if impression_index + 1 == len(train_data):
#         state_1 = 0
#     else:
#         state_1 = get_state_1_optimized_pctr(pctrs[impression_index+1], Min_Threshold_pctr, Max_Threshold_pctr)
#     state_0 = get_state_0_average_pctr(total_bid_pctrs, total_bid_nums, bid_pctrs, bid_nums)
#     state_2, _, _ = get_state_2_cost_rate(recent_cost_state, cost_index, recent_cost, config, current_day)
#     state_3 = get_state_3_Budget_allocate_right_rate(time_frac[impression_index], real_clks, win_clks, Budget_allocate_by_clk_rate)
#     state_4 = get_state_4_remain_budget(cost, config, current_day)
#     if config.reward_type == '124':
#         state_ = np.array([state_1, state_0, state_2, state_4])
#     else:
#         state_ = np.array([state_1, state_0, state_2, state_3, state_4])
#     return state_
#
# def get_state_0_average_pctr(all_pctr, all_bids, total_pctr, total_bid):
#     state_0 = (np.sum(total_pctr) + np.sum(all_pctr)) / (np.sum(total_bid) + np.sum(all_bids))
#     return state_0
#
#
# def get_state_1_optimized_pctr(pctr, Min_Threshold_pctr, Max_Threshold_pctr):
#     # if pctr >= Min_Threshold_pctr:
#     #     optimized_pctr = pctr
#     #     if Max_Threshold_pctr >= pctr:
#     #         optimized_pctr = optimized_pctr / Max_Threshold_pctr
#     # else:
#     #     optimized_pctr = pctr
#     # state_1 = optimized_pctr
#     if pctr > Max_Threshold_pctr:
#         state_1 = 1
#     else:
#         state_1 = pctr
#
#     return pctr
#     # return state_1
#
#
# def get_state_2_cost_rate(recent_cost_state, cost_index, recent_cost, config, current_day):
#     if cost_index % config.fraction_cost == 0:
#         recent_cost_state_new = recent_cost / config.budget_total[current_day]  # 替代状态 rate
#         state_2 = recent_cost_state_new
#         recent_cost_new = 0
#     else:
#         recent_cost_state_new = recent_cost_state
#         state_2 = recent_cost_state_new
#         recent_cost_new = recent_cost
#     return state_2, recent_cost_state_new, recent_cost_new
#
#
# def get_state_3_Budget_allocate_right_rate(time_fraction, real_clks, win_clks, Budget_allocate_by_clk_rate):
#     if time_fraction == 0:
#         Budget_allocate_right_rate = 1  # stae:3
#     else:
#         real_clk_t = real_clks[time_fraction-1]
#         True_win_clk_rate = win_clks[time_fraction-1] / real_clk_t if real_clk_t > 0 else 0 # 实际买到的clk/真实点击
#         Budget_allocate_right_rate =  True_win_clk_rate * Budget_allocate_by_clk_rate[
#                                          time_fraction - 1]   # state：3 购买正确率
#     state_3 = Budget_allocate_right_rate
#
#     return state_3
#
# def get_state_4_remain_budget(cost, config, current_day):
#     state_4 = 1 - (np.sum(cost) / config.budget_total[current_day])
#     return state_4


# Max奖励函数
def get_next_state(current_day_data, impression_index, Min_Threshold_pctr, Max_Threshold_pctr,
                   recent_cost_state, cost_index, recent_cost, config,cost,real_clks, win_clks,
                   Budget_allocate_by_clk_rate, pctrs, time_frac, pctr_list, today_budget, n,
                   remain_budget, current_allocate_budget, HB_base_bid, original_avg_train_pctr):


    if impression_index + 1 == len(current_day_data):
        state_1 = 0
        state_6 = 0
    else:
        origin_bid = get_refine_bid(original_avg_train_pctr, pctrs[impression_index + 1], HB_base_bid)
        state_1 = get_state_1_optimized_pctr(pctrs[impression_index+1], Min_Threshold_pctr, Max_Threshold_pctr)
        state_6 = origin_bid / 300
    state_0 = get_state_0_average_pctr(pctr_list)
    state_2, _, _ = get_state_2_cost_rate(recent_cost_state, cost_index, recent_cost, config, today_budget)
    state_3 = get_state_3_Budget_allocate_right_rate(time_frac[impression_index], real_clks, win_clks, Budget_allocate_by_clk_rate)
    state_4 = get_state_4_remain_budget(remain_budget, current_allocate_budget)
    state_5 = get_state_5_remain_time(n+1, config)


    if config.reward_type == '012456':
        state_ = np.array([state_0, state_1, state_2, state_4, state_5, state_6])
    elif config.reward_type == '01456':
        state_ = np.array([state_0, state_1, state_4, state_5, state_6])
    elif config.reward_type == 'all':
        state_ = np.array([state_0, state_1, state_2, state_3, state_4, state_5, state_6])

    return state_

def get_state_0_average_pctr(pctr_list):
    state_0 = pctr_list[0]
    return state_0


def get_state_1_optimized_pctr(pctr, Min_Threshold_pctr, Max_Threshold_pctr):
    # if pctr >= Min_Threshold_pctr:
    #     optimized_pctr = pctr
    #     if Max_Threshold_pctr >= pctr:
    #         optimized_pctr = optimized_pctr / Max_Threshold_pctr
    # else:
    #     optimized_pctr = pctr
    # state_1 = optimized_pctr
    if pctr > Max_Threshold_pctr:
        state_1 = 1
    else:
        state_1 = pctr

    return state_1


def get_state_2_cost_rate(recent_cost_state, cost_index, recent_cost, config, today_budget):
    if cost_index % config.fraction_cost == 0:
        recent_cost_state_new = recent_cost / today_budget  # 替代状态 rate
        state_2 = recent_cost_state_new
        recent_cost_new = 0
    else:
        recent_cost_state_new = recent_cost_state
        state_2 = recent_cost_state_new
        recent_cost_new = recent_cost
    return state_2, recent_cost_state_new, recent_cost_new


def get_state_3_Budget_allocate_right_rate(time_fraction, real_clks, win_clks, Budget_allocate_by_clk_rate):
    if time_fraction == 0:
        Budget_allocate_right_rate = 1  # stae:3
    else:
        real_clk_t = real_clks[time_fraction-1]
        True_win_clk_rate = win_clks[time_fraction-1] / real_clk_t if real_clk_t > 0 else 0 # 实际买到的clk/真实点击
        Budget_allocate_right_rate =  True_win_clk_rate * Budget_allocate_by_clk_rate[
                                         time_fraction - 1]   # state：3 购买正确率
    state_3 = Budget_allocate_right_rate

    return state_3

def get_state_4_remain_budget(remain_budget, current_allocate_budget):
    state_4 = remain_budget / current_allocate_budget
    return state_4

def get_state_5_remain_time(n, config):
    remain_time_rate = n / config.budget_allocate_num
    return remain_time_rate

def get_refine_bid(avg_pctr, current_pctr, best_base_bid):
    bid = best_base_bid * current_pctr / avg_pctr
    bid = int(min(bid, 300))
    return bid

