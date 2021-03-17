import numpy as np
import pandas as pd
import logging
from src.DRLB.DRLB_get_function.get_state import get_state
from src.DRLB.DRLB_get_function.get_time_step_result import get_init_lambda

def test_model(config, model, test_data):
    record_time = []
    record_action = [0]
    record_lambda = []
    record_total_ROL = []

    win_clks = []
    win_imps = []
    win_reward = []
    early_stop_time = []


    total_budget = []
    for index, day in enumerate(test_data.day.unique()):
        current_day_budget = np.sum(test_data[test_data.day.isin([day])].market_price)
        total_budget.append(current_day_budget)
    total_budget = np.divide(total_budget, config.budget_para_int)
    print(total_budget)
    for day_index, day in enumerate(test_data.day.unique()):

        B_t = [0 for i in range(96)]
        B_t[0] = total_budget[day_index]
        current_budget = total_budget[day_index]
        current_day_data = test_data[test_data.day.isin([day])]
        # record result
        current_day_win_clks = 0
        current_day_win_imps = 0
        current_day_reward = 0

        early_stop = False

        # lambda
        lambda_t = get_init_lambda(config)
        record_lambda.append(lambda_t)
        optimal_lambda = 0

        # loss
        episode_loss = 0

        done = 0

        for time_t in range(96):
            record_time.append('{}_{}'.format(day, time_t))

            if time_t == 94:
                done = 1
                break
            if time_t == 0:
                current_time_data = current_day_data[current_day_data.fraction.isin([time_t])]
                # 第一个时段因为直接用的初始lambda，所以不会存在以第一个时段为开始的奖励，只是用第一个时段构造第一个状态
                next_state, new_B_t, early_stop, time_win_clk, time_win_imps, last_reward, ROL_t = get_state(
                    current_time_data, lambda_t, time_t,
                    B_t, current_budget)
                record_total_ROL.append(ROL_t)

                current_day_win_clks += time_win_clk
                current_day_win_imps += time_win_imps
                current_day_reward += last_reward
                continue
            else:
                state = next_state
                action = model.choose_action(state)
                # action = model.choose_best_action(state)
                lambda_t = lambda_t * (1 + action)
                record_lambda.append(lambda_t)
                current_time_data = current_day_data[current_day_data.fraction.isin([time_t])]

                # 第一个时段因为直接用的初始lambda，所以不会存在以第一个时段为开始的奖励，只是用第一个时段构造第一个状态
                next_state, new_B_t, early_stop, time_win_clk, time_win_imps, t_real_reward, ROL_t = get_state(
                    current_time_data, lambda_t, time_t,
                    B_t, current_budget)
                record_total_ROL.append(ROL_t)

            record_action.append(action)
            B_t = new_B_t

            net_reward = model.get_reward(state, action)
            max_reward = np.max([t_real_reward, net_reward])
            # print(state, action, max_reward, next_state, done)

            current_day_win_clks += time_win_clk
            current_day_win_imps += time_win_imps
            current_day_reward += max_reward

            if early_stop:
                early_stop_time.append('{}_{}'.format(day, time_t))
                done = 1

                break

        win_clks.append(current_day_win_clks)
        win_imps.append(current_day_win_imps)
        win_reward.append(current_day_reward)


    test_record_data = pd.DataFrame(data=[[record_time, record_total_ROL, record_action, record_lambda]],
                 columns=['time', 'ROL_t', 'action', 'lambda'])
    total_win_clk = np.sum(win_clks)
    total_win_imps = np.sum(win_imps)
    total_reward = np.sum(win_reward)
    # print(total_win_clk)

    test_result_data = pd.DataFrame(data=[[total_win_clk, total_win_imps, total_reward]], columns=['win_clks', 'win_imps', 'win_reward'])
    test_msg = 'Test result: win_clks:{0}, win_imps:{1}, win_reward:{2}'.format(total_win_clk, total_win_imps, total_reward)
    return test_record_data, test_result_data, test_msg

