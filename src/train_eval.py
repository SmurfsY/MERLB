import pandas as pd
import numpy as np
import datetime
import os
import logging
import time

from tqdm import tqdm
# from src.model.TD3 import TD3
# from src.model.DIAYN import DIAYN

logger = logging.getLogger(__name__)


def choose_eCPC(config, original_ctr):
    hb_result = config.heuristic_path + 'Train_best_bid_line_300.txt'
    ecpc_result = pd.read_csv(hb_result)

    for index, item in ecpc_result.iterrows():
        if item['prop'] == config.budget_para_int:
            return ecpc_result.loc[index, 'base_bid_price'] / original_ctr
        else:
            continue


def get_reward(episode_state, current_data,eCPC, t):
    hb_bids_data = current_data[current_data.pctr * eCPC >= current_data.market_price]
    hb_bids_clk = np.sum(hb_bids_data.clk)
    reward = 0
    if len(current_data) > 1:
        if episode_state.loc[t, 'cost'] > np.sum(hb_bids_data.market_price):
            if episode_state.loc[t, 'win_clks'] >= hb_bids_clk:
                reward = 1
            else:
                reward = -5
        else:
            if episode_state.loc[t, 'win_clks'] >= hb_bids_clk:
                reward = 5
            else:
                reward = -2.5

    else:
        reward = 0
    return reward/1000

def model_train(model, config):

    train_data = pd.read_csv(config.train_set)
    test_data = pd.read_csv(config.test_set)

    fraction_type = config.fraction_type    # 分片粒度
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size GPU/CPU = %d", config.batch_size)
    logger.info("  Train device:%s, id:%d", config.device, config.device_id)

    # for _, item in tqdm(train_data.iterrows()):
    #     real_fraction_clk[int(item['time_fraction'])] += item['clk']
    td_error, action_loss = 0, 0
    original_ctr = np.sum(train_data.clk) / len(train_data)
    total_clks = np.sum(train_data.clk)

    eCPC = choose_eCPC(config, original_ctr)

    train_results = []
    test_results = []
    test_actions = []
    best_model = model
    print('*********************STARTING TRAIN**********************')
    for episode in range(config.num_train_epochs):

        logger.info('Epoch [{}/{}]'.format(episode + 1, config.num_train_epochs))
        episode_state = {'win_clks': [0 for i in range(fraction_type)],     # win的点击数
                         'profits': [0 for i in range(fraction_type)],  # 利润
                         'reward': [0 for i in range(fraction_type)],   # 奖励
                         'cost': [0 for i in range(fraction_type)],     # 花费
                         'win_true_value': [0 for i in range(fraction_type)],   # win的真实价值：base_bid * pctr
                         'lose_true_value': [0 for i in range(fraction_type)],  # lose的真实价值
                         'win_imp_with_clk_value': [0 for i in range(fraction_type)],   # win的有点击的impression的真实价值
                         'win_imp_without_clk_cost': [0 for i in range(fraction_type)],  # win的没有点击的impression的花费
                         'lose_imp_with_clk_value': [0 for i in range(fraction_type)],  # lose有点击的impression的价值
                         'clk_no_win_imp': [0 for i in range(fraction_type)],  # 时段的没有win点击数
                         'lose_imp_without_clk_cost': [0 for i in range(fraction_type)],    # 时段中lose的没有点击的impression的市场价格和
                         'no_clk_imp': [0 for i in range(fraction_type)],      # 时段中没有点击的impression数
                         'no_clk_no_win_imp': [0 for i in range(fraction_type)],   # 没有点击没有win的impression数
                         'actions': [0 for i in range(fraction_type)],  # 动作
                         'real_clks': [0 for i in range(fraction_type)],    # 真实点击数
                         'bid_nums': [0 for i in range(fraction_type)],     # 出价次数
                         'win_imps': [0 for i in range(fraction_type)],      # 赢标次数
                         'win_rate': 0.0,
                         't_ctr': [0.0 for i in range(fraction_type)],
                         't_reward': [0.0 for i in range(fraction_type)],
                         }
        episode_state = pd.DataFrame(episode_state)
        start_time = time.time()
        end_slot_time = 23
        done = 0
        for t in range(config.fraction_type):
            current_data = train_data[train_data.time_fraction.isin([t])]


            if t == 0:
                state = np.array([1, 0, 0, 0])  # current_time_slot, budget_left_ratio, cost_t_ratio, ctr_t, win_rate_t
                action = model.choose_action(state)
                init_action = action
                bids = current_data.pctr * eCPC / (1 + init_action)
                bids = np.where(bids >= 300, 300, bids)
            else:
                state = state_
                action = next_action
                bids = current_data.pctr * eCPC / (1 + action)
                bids = np.where(bids >= 300, 300, bids)


            episode_state.loc[t, 'actions'] = action
            # print(action)
            win_impression_data = current_data[bids >= current_data.market_price]    # 当前时段win的impression
            lose_impression_data = current_data[bids < current_data.market_price]    # 当前时段lose的impression

            episode_state.loc[t, 'cost'] += np.sum(win_impression_data.market_price)     # 统计当前花费
            episode_state.loc[t, 'profits'] += np.sum(win_impression_data.pctr * eCPC -
                                                  win_impression_data.market_price)  # 统计利润

            episode_state.loc[t,'win_true_value'] = np.sum(win_impression_data.pctr * eCPC)         # 对win的impression的期望价值进行统计
            episode_state.loc[t, 'lose_true_value']= np.sum(lose_impression_data.pctr * eCPC)  # 对lose的impression的期望价值精心统计

            win_impression_with_clk = win_impression_data[win_impression_data.clk.isin([1])]     # 构造win的有点击pd
            lose_impression_with_clk = lose_impression_data[lose_impression_data.clk.isin([1])]   #构造lose的impression中有点击的pd

            episode_state.loc[t, 'win_imp_with_clk_value'] = np.sum(win_impression_with_clk.pctr * eCPC)             # 对win的impression中有点击的真实价值进行统计
            episode_state.loc[t, 'lose_imp_with_clk_value'] = np.sum(lose_impression_with_clk.pctr * eCPC)           # 对lose的impression中有点击的真实价值进行统计
            episode_state.loc[t, 'clk_no_win_imp'] = len(lose_impression_with_clk)
            episode_state.loc[t, 'win_imp_without_clk_cost'] = np.sum(win_impression_data[win_impression_data.clk.isin([0])].market_price)    #对win的impression中没有点击的花费进行统计
            episode_state.loc[t, 'lose_imp_without_clk_cost'] = np.sum(lose_impression_data[lose_impression_data.clk.isin([0])].market_price)    # 对lose的impression中的没有点击的花费进行统计

            episode_state.loc[t, 'win_clks'] = len(win_impression_with_clk)      # win的点击数
            episode_state.loc[t, 'win_imps'] = len(win_impression_data)                 # win的impression数

            episode_state.loc[t, 'real_clks'] = np.sum(current_data['clk'])  # 真实点击
            episode_state.loc[t, 'no_clk_imp'] = len(current_data) - episode_state.loc[t, 'real_clks']  # 没有点击的的impression数


            episode_state.loc[t, 'bid_nums'] = len(current_data)                        # 出价次数
            episode_state.loc[t, 'no_clk_no_win_imp'] = len(lose_impression_data[lose_impression_data.clk.isin([0])])   # 没有点击没有win的impression

            win_impression_bids = list(bids[bids >= current_data.market_price])
            win_impression_market_price = list(win_impression_data.market_price)


            if np.sum(episode_state.cost) > config.budget_total:
                end_slot_time = t
                episode_state.iloc[t, :] = 0
                for index_iter in range(len(current_data)):
                    if np.sum(episode_state.cost) > config.budget_total - np.sum(current_data.loc[:index_iter, 'market_price']):
                        # continue
                        break
                    current_impression = current_data.iloc[index_iter, :]

                    if t == 0:
                        current_action = init_action
                    else:
                        current_action = next_action
                    bid = current_impression.pctr * eCPC / (1 + current_action)
                    bid = bid if bid <= 300 else 300
                    episode_state.loc[t, 'real_clks'] += current_impression.clk
                    episode_state.loc[t, 'bid_nums'] += 1

                    if current_impression.clk == 1:
                        episode_state.loc[t, 'real_clks'] += 1
                    else:
                        episode_state.loc[t, 'no_clk_imp'] += 1

                    if bid >= current_impression.market_price:
                        if current_impression.clk == 1:
                            episode_state.loc[t, 'win_clks'] += 1
                            episode_state.loc[t, 'win_imp_with_clk_value'] += current_impression.pctr * eCPC
                        else:
                            episode_state.loc[t, 'win_imp_without_clk_cost'] += current_impression.market_price
                        win_impression_bids.append(bid)
                        win_impression_market_price.append(current_impression.market_price)

                        episode_state.loc[t, 'profits'] += current_impression.pctr * eCPC - current_impression.market_price
                        episode_state.loc[t, 'win_true_value'] += current_impression.pctr * eCPC
                        episode_state.loc[t, 'real_clks'] += current_impression.clk
                        episode_state.loc[t, 'bid_nums'] += 1
                        episode_state.loc[t, 'cost'] += current_impression.market_price
                        episode_state.loc[t, 'win_imps'] += 1

                    else:
                        episode_state.loc[t, 'lose_true_value'] += current_impression.pctr * eCPC
                        episode_state.loc[t, 'lose_imp_without_clk_cost'] += current_impression.market_price
                        if current_impression.clk == 1:
                            episode_state.loc[t, 'lose_imp_with_clk_value'] += current_impression.pctr * eCPC
                            episode_state.loc[t, 'clk_no_win_imp'] += 1
                    episode_state.loc[t, 't_ctr'] = episode_state.loc[t, 'win_clks'] / episode_state.loc[t, 'win_imps'] if \
                        episode_state.loc[t, 'win_imps'] > 0 else 0
                    episode_state.loc[t, 'win_rate'] = episode_state.loc[t, 'win_imps'] / episode_state.loc[t, 'bid_nums']
            else:
                episode_state.loc[t, 't_ctr'] = episode_state.loc[t, 'win_clks'] / episode_state.loc[t, 'win_imps'] if \
                episode_state.loc[t, 'win_imps'] > 0 else 0
                episode_state.loc[t, 'win_rate'] = episode_state.loc[t, 'win_imps'] / episode_state.loc[t, 'bid_nums']
            budget_left_ratio =(config.budget_total - np.sum(episode_state.cost)) / config.budget_total
            time_left_ratio = (fraction_type - 1 - t) / fraction_type
            avg_time_spend = budget_left_ratio / time_left_ratio if time_left_ratio > 0 else 0
            cost_t_ratio = episode_state.loc[t, 'cost'] / config.budget_total
            # print(avg_time_spend, cost_t_ratio, episode_state['t_ctr'][t], episode_state['win_rate'][t])
            state_ = np.array([avg_time_spend, cost_t_ratio, episode_state.loc[t, 't_ctr'], episode_state.loc[t, 'win_rate']])    # state
            # print('state_:', state_)
            action_ = model.choose_action(state_)
            next_action = action_
            episode_state.loc[t, 'reward'] = get_reward(episode_state, current_data,eCPC, t)

            if t == fraction_type - 1:
                done = 1

            Replay_buffer_data = list((state, state_, action, episode_state.loc[t, 'reward'], done))
            # print('Replay_buffer_data: ', Replay_buffer_data)
            # print('doneee')
            model.memory.push(Replay_buffer_data)

            if np.sum(episode_state.cost) >= config.budget_total:
                break
        # logger.info('Epoch [{}/{}]'.format(episode + 1, config.num_train_epochs))
        end_time = time.time()
        if (episode > 0) and ((episode + 1) % config.ob_episode == 0):
            current_result = [np.sum(episode_state.t_reward), np.sum(episode_state.profits),
                              np.sum(episode_state.cost), int(np.sum(episode_state.win_clks)),
                              int(np.sum(episode_state.real_clks)), np.sum(episode_state.bid_nums),
                              np.sum(episode_state.win_imps),
                              np.sum(episode_state.cost) / np.sum(episode_state.win_imps) if np.sum(episode_state.win_imps) > 0 else 0,
                              end_slot_time, td_error, action_loss]
            train_results.append(current_result)
            td_error, action_loss = model.update(config.num_update)

            episode_state.actions.to_csv(config.result_path + 'train_result_actions.csv')
            episode_state.win_clks.to_csv(config.result_path + 'train_result_win_clk.csv')
            time_dif = start_time - end_time
            # print(episode)
            # print(train_results)
            # if train_results[3][episode-1] > train_results[3][episode-2]:
            #     improve = '*'
            # else:
            #     improve = ''
            improve = ''

            msg = 'Episode: {0}, win_clk:{1}, td_error: {2:>5.6f}, Val Acc: {3:>6.2%},  Time: {4} {5}'
            logging.info(msg.format(episode, np.sum(episode_state.win_clks), td_error, action_loss, time_dif, improve))

            print('Train_result: episode {}, win_clks={}, reward={}, cost={}, real_clks={}, bids_num={}, '
                  'win_imps={}, cpm={}, end_slot={}, td_error={}, action_loss={}\n'.format(
                    episode+1,
                    np.sum(episode_state.win_clks),
                    np.sum(episode_state.reward),
                    np.sum(episode_state.cost),
                    np.sum(episode_state.real_clks),
                    np.sum(episode_state.bid_nums),
                    np.sum(episode_state.win_imps),
                    np.sum(episode_state.cost) / np.sum(episode_state.win_imps)
                    if np.sum(episode_state.win_imps) > 0 else 0,
                    end_slot_time,
                    td_error, action_loss
                  ))

            test_result, test_action, test_hour_clk = model_test(config, test_data, eCPC, model)
            test_results.append(test_result)
            test_actions.append(test_action)
    test_result_df = pd.DataFrame(data=test_results, columns=['win_clk', 'reward', 'cost','real_clk','bid_nums',
                                                                    'win_imps', 'cpm'])
    test_result_df.to_csv(config.result_path + 'test_result.csv')
    pd.DataFrame(test_actions).to_csv(config.result_path + 'test_action.csv')

    pd.DataFrame(data=train_results, columns=['t_reward', 'profits', 'cost', 'win_clk', 'real_clk', 'win_bid_num',
                                              'win_imps', 'cpm', 'end_slot', 'td_error', 'action_loss'])

    return model

def model_test(config, test_data, eCPC, model):
    test_results = []

    fraction_type = config.fraction_type

    episode_state = {'win_clks': [0 for i in range(fraction_type)],  # win的点击数
                     'profits': [0 for i in range(fraction_type)],  # 利润
                     'reward': [0 for i in range(fraction_type)],  # 奖励
                     'cost': [0 for i in range(fraction_type)],  # 花费
                     'win_true_value': [0 for i in range(fraction_type)],  # win的真实价值：base_bid * pctr
                     'lose_true_value': [0 for i in range(fraction_type)],  # lose的真实价值
                     'win_imp_with_clk_value': [0 for i in range(fraction_type)],  # win的有点击的impression的真实价值
                     'win_imp_without_clk_cost': [0 for i in range(fraction_type)],  # win的没有点击的impression的花费
                     'lose_imp_with_clk_value': [0 for i in range(fraction_type)],  # lose有点击的impression的价值
                     'clk_no_win_imp': [0 for i in range(fraction_type)],  # 时段的没有win点击数
                     'lose_imp_without_clk_cost': [0 for i in range(fraction_type)],  # 时段中lose的没有点击的impression的市场价格和
                     'no_clk_imp': [0 for i in range(fraction_type)],  # 时段中没有点击的impression数
                     'no_clk_no_win_imp': [0 for i in range(fraction_type)],  # 没有点击没有win的impression数
                     'actions': [0 for i in range(fraction_type)],  # 动作
                     'real_clks': [0 for i in range(fraction_type)],  # 真实点击数
                     'bid_nums': [0 for i in range(fraction_type)],  # 出价次数
                     'win_imps': [0 for i in range(fraction_type)],  # 赢标次数
                     'win_rate': [0 for i in range(fraction_type)],
                     't_ctr': [0.0 for i in range(fraction_type)],
                     't_reward': 0.0,
                     'done': 0
                     }
    episode_state = pd.DataFrame(episode_state)

    for t in range(fraction_type):
        current_data = test_data[test_data.time_fraction.isin([t])]

        if t == 0:
            state = np.array([1, 0, 0, 0])  # current_time_slot, budget_left_ratio, cost_t_ratio, ctr_t, win_rate_t
            action = model.choose_action(state)
            init_action = action
            bids = current_data.pctr * eCPC / (1 + init_action)
            bids = np.where(bids >= 300, 300, bids)
        else:
            state = state_
            action = next_action
            bids = current_data.pctr * eCPC / (1 + action)
            bids = np.where(bids >= 300, 300, bids)

        episode_state.loc[t, 'actions'] = action

        win_impression_data = current_data[bids >= current_data.market_price]  # 当前时段win的impression
        lose_impression_data = current_data[bids < current_data.market_price]  # 当前时段lose的impression

        episode_state.loc[t, 'cost'] += np.sum(win_impression_data.market_price)  # 统计当前花费
        episode_state.loc[t, 'profits'] += np.sum(win_impression_data.pctr * eCPC -
                                              win_impression_data.market_price)  # 统计利润

        episode_state.loc[t, 'win_true_value'] = np.sum(win_impression_data.pctr * eCPC)  # 对win的impression的期望价值进行统计
        episode_state.loc[t, 'lose_true_value'] = np.sum(lose_impression_data.pctr * eCPC)  # 对lose的impression的期望价值精心统计

        win_impression_with_clk = win_impression_data[win_impression_data.clk.isin([1])]  # 构造win的有点击pd
        lose_impression_with_clk = lose_impression_data[
            lose_impression_data.clk.isin([1])]  # 构造lose的impression中有点击的pd

        episode_state.loc[t, 'win_imp_with_clk_value'] = np.sum(
            win_impression_with_clk.pctr * eCPC)  # 对win的impression中有点击的真实价值进行统计
        episode_state.loc[t, 'lose_imp_with_clk_value'] = np.sum(
            lose_impression_with_clk.pctr * eCPC)  # 对lose的impression中有点击的真实价值进行统计
        episode_state.loc[t, 'clk_no_win_imp'] = len(lose_impression_with_clk)
        episode_state.loc[t, 'win_imp_without_clk_cost'] = np.sum(
            win_impression_data[win_impression_data.clk.isin([0])]['market_price'])  # 对win的impression中没有点击的花费进行统计
        episode_state.loc[t, 'lose_imp_without_clk_cost'] = np.sum(
            lose_impression_data[lose_impression_data.clk.isin([0])][
                'market_price'])  # 对lose的impression中的没有点击的花费进行统计

        episode_state.loc[t, 'win_clks'] = len(win_impression_with_clk)  # win的点击数
        episode_state.loc[t, 'win_imps'] = len(win_impression_data)  # win的impression数

        episode_state.loc[t, 'real_clks'] = np.sum(current_data.clk)  # 真实点击
        episode_state.loc[t, 'no_clk_imp'] = len(current_data) - episode_state.loc[t, 'real_clks']  # 没有点击的的impression数

        episode_state.loc[t, 'bid_nums'] = len(current_data)  # 出价次数
        episode_state.loc[t, 'no_clk_no_win_imp'] = len(
            lose_impression_data[lose_impression_data.clk.isin([0])])  # 没有点击没有win的impression

        win_impression_bids = list(bids[bids >= current_data.market_price])
        win_impression_market_price = list(win_impression_data.market_price)

        if np.sum(episode_state.cost) > config.budget_total:
            end_slot_time = t
            episode_state.iloc[t, :] = 0
            for index_iter in range(len(current_data)):
                if np.sum(episode_state.cost) > config.budget_total - np.sum(current_data.loc[:index_iter, 'market_price']):
                    # continue
                    break
                current_impression = current_data.iloc[index_iter, :]

                if t == 0:
                    current_action = init_action
                else:
                    current_action = next_action
                bid = current_impression.pctr * eCPC / (1 + current_action)
                bid = bid if bid <= 300 else 300
                episode_state.loc[t, 'real_clks'] += current_impression.clk
                episode_state.loc[t, 'bid_nums'] += 1

                if current_impression.clk == 1:
                    episode_state.loc[t, 'real_clks'] += 1
                else:
                    episode_state.loc[t, 'no_clk_imp'] += 1

                if bid >= current_impression.market_price:
                    if current_impression.clk == 1:
                        episode_state.loc[t, 'win_clks'] += 1
                        episode_state.loc[t, 'win_imp_with_clk_value'] += current_impression.pctr * eCPC
                    else:
                        episode_state.loc[t, 'win_imp_without_clk_cost'] += current_impression.market_price
                    win_impression_bids.append(bid)
                    win_impression_market_price.append(current_impression.market_price)

                    episode_state.loc[t, 'profits'] += current_impression.pctr * eCPC - current_impression.market_price
                    episode_state.loc[t, 'win_true_value'] += current_impression.pctr * eCPC
                    episode_state.loc[t, 'real_clks'] += current_impression.clk
                    episode_state.loc[t, 'bid_nums'] += 1
                    episode_state.loc[t, 'cost'] += current_impression.market_price
                    episode_state.loc[t, 'win_imps'] += 1

                else:
                    episode_state.loc[t, 'lose_true_value'] += current_impression.pctr * eCPC
                    episode_state.loc[t, 'lose_imp_without_clk_cost'] += current_impression.market_price
                    if current_impression.clk == 1:
                        episode_state.loc[t, 'lose_imp_with_clk_value']+= current_impression.pctr * eCPC
                        episode_state.loc[t, 'clk_no_win_imp'] += 1
                episode_state.loc[t, 't_ctr'] = episode_state.loc[t, 'win_clks'] / episode_state.loc[t, 'win_imps'] if \
                episode_state.loc[t, 'win_imps'] > 0 else 0
                episode_state.loc[t, 'win_rate'] = episode_state.loc[t, 'win_imps'] / episode_state.loc[t, 'bid_nums']
        else:
            episode_state.loc[t, 't_ctr'] = episode_state.loc[t, 'win_clks'] / episode_state.loc[t, 'win_imps'] \
                if episode_state.loc[t, 'win_imps'] > 0 else 0
            episode_state.loc[t, 'win_rate'] = episode_state.loc[t, 'win_imps'] / episode_state.loc[t, 'bid_nums']

        budget_left_ratio = (config.budget_total - np.sum(episode_state.cost)) / config.budget_total
        time_left_ratio = (fraction_type - 1 - t) / fraction_type
        avg_time_spend = budget_left_ratio / time_left_ratio if time_left_ratio > 0 else 0
        cost_t_ratio = episode_state.loc[t, 'cost'] / config.budget_total

        state_ = np.array([avg_time_spend, cost_t_ratio, episode_state.loc[t, 't_ctr'], episode_state.loc[t, 'win_rate']])  # state
        # print('state_:', state_)
        action_ = model.choose_action(state_)
        next_action = action_
        episode_state.loc[t, 'reward'] = get_reward(episode_state, current_data, eCPC, t)

        if np.sum(episode_state.cost) > config.budget_total:
            break

    print('*****************************测试结果***************************')
    print('win_clks={}, reward={}, cost={}, real_clks={}, bids_num={}, win_imps={}, cpm={}'.format(
        np.sum(episode_state.win_clks),
        np.sum(episode_state.reward),
        np.sum(episode_state.cost),
        np.sum(episode_state.real_clks),
        np.sum(episode_state.bid_nums),
        np.sum(episode_state.win_imps),
        np.sum(episode_state.cost) / np.sum(episode_state.win_imps)
        if np.sum(episode_state.win_imps) > 0 else 0))
    test_result = [np.sum(episode_state.win_clks),
        np.sum(episode_state.reward),
        np.sum(episode_state.cost),
        np.sum(episode_state.real_clks),
        np.sum(episode_state.bid_nums),
        np.sum(episode_state.win_imps),
        np.sum(episode_state.cost) / np.sum(episode_state.win_imps)
                   if np.sum(episode_state.win_imps) > 0 else 0]

    test_results.append(test_result)
    test_actions = episode_state.actions
    test_hour_clk = episode_state.win_clks
    return test_result, test_actions, test_hour_clk
