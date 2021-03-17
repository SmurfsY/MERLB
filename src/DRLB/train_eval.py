import numpy as np
import pandas as pd
import logging, copy
from src.DRLB.DRLB_get_function.get_state import get_state
from src.DRLB.DRLB_get_function.get_time_step_result import get_init_lambda
from src.DRLB.test_eval import test_model


logger = logging.getLogger(__name__)




def train_eval(config, model):
    train_data = pd.read_csv(config.train_set)
    test_data = pd.read_csv(config.test_set)
    train_data.sort_values(by='minutes', inplace=True)
    test_data.sort_values(by='minutes', inplace=True)
    train_data['fraction'] = train_data['time_fraction'] * 4 + (
                train_data['minutes'].apply(lambda x: int(int(str(x)[10:12]) / 15)))
    test_data['fraction'] = test_data['time_fraction'] * 4 + (
                test_data['minutes'].apply(lambda x: int(int(str(x)[10:12]) / 15)))
    train_data['day'] = train_data['minutes'].apply(lambda x:int(str(x)[6:8]))
    test_data['day'] = test_data['minutes'].apply(lambda x: int(str(x)[6:8]))


    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size GPU/CPU = %d", config.batch_size)
    logger.info("  Train device:%s, id:%d", config.device, config.device_id)
    logger.info("  Ad_id:%s Train day:%s, Test day:%s", config.campaign_id, config.train_set, config.test_set)

    total_budget = []
    for index, day in enumerate(train_data.day.unique()):
        current_day_budget = np.sum(train_data[train_data.day.isin([day])].market_price)
        total_budget.append(current_day_budget)
    # print(total_budget)
    total_budget = np.divide(total_budget, config.budget_para_int)
    logger.info("  Train_Total_budget:%s", total_budget)

    print(total_budget)
    best_train_clk = 0
    best_model = copy.deepcopy(model)

    for episode in range(config.num_train_epochs):
        print('--------第{}轮训练--------\n'.format(episode + 1))

        record_time = []
        record_action = [0]
        record_lambda = []
        record_total_ROL = []

        win_clks = []
        win_imps = []
        win_reward = []
        early_stop_time = []

        # loss
        episode_Q_loss = []
        episode_reward_loss = []

        state_action_pair = []

        for day_index, day in enumerate(train_data.day.unique()):
            B_t = [0 for i in range(96)]
            B_t[0] = total_budget[day_index]
            # print(B_t)
            current_budget = total_budget[day_index]
            current_day_data = train_data[train_data.day.isin([day])]
            # record result
            current_day_win_clks = 0
            current_day_win_imps = 0
            current_day_reward = 0

            early_stop =False

            # lambda
            lambda_t = get_init_lambda(config)
            record_lambda.append(lambda_t)
            optimal_lambda = 0

            model.reset_epsilon()

            done = 0

            for time_t in range(96):

                record_time.append('{}_{}'.format(day, time_t))
                if len(model.Reward_memory) >= config.batch_size:
                    reward_loss = model.learn_reward()
                    episode_reward_loss.append(reward_loss)

                if time_t == 94:
                    done = 1
                    break
                if time_t == 0:
                    current_time_data = current_day_data[current_day_data.fraction.isin([time_t])]
                    # 第一个时段因为直接用的初始lambda，所以不会存在以第一个时段为开始的奖励，只是用第一个时段构造第一个状态
                    next_state, new_B_t, early_stop, time_win_clk, time_win_imps, last_reward, ROL_t = get_state(current_time_data, lambda_t, time_t,
                                                                                    B_t, current_budget)
                    record_total_ROL.append(ROL_t)
                    # last_reward = t_real_reward
                    current_day_win_clks += time_win_clk
                    current_day_win_imps += time_win_imps
                    current_day_reward += last_reward
                    continue
                else:
                    state = next_state
                    action = model.choose_action(state)
                    lambda_t = lambda_t * (1 + action)
                    record_lambda.append(lambda_t)
                    current_time_data = current_day_data[current_day_data.fraction.isin([time_t])]

                    # 第一个时段因为直接用的初始lambda，所以不会存在以第一个时段为开始的奖励，只是用第一个时段构造第一个状态
                    next_state, new_B_t, early_stop, time_win_clk, time_win_imps, t_real_reward, ROL_t = get_state(current_time_data, lambda_t, time_t,
                                                                                    B_t, current_budget)
                    record_total_ROL.append(ROL_t)

                record_action.append(action)
                B_t = new_B_t

                if early_stop:
                    early_stop_time.append('{}_{}'.format(day, time_t))
                    done = 1
                    # print('early_stop:{}'.format(time_t+1))



                net_reward = model.get_reward(state, action)[0]
                state_action_pair.append((state, action, net_reward, next_state, done))
                model.Q_memory.push(state, action, net_reward, next_state, done)


                model.control_epsilon(time_t+1)

                if len(model.Q_memory) >= config.batch_size:
                    print('__')
                    Q_loss = model.learn_Q()
                    episode_Q_loss.append(Q_loss)

                # print(time_win_clk)
                # print(time_win_imps)
                # print(max_reward)

                current_day_win_clks += time_win_clk
                current_day_win_imps += time_win_imps
                current_day_reward += t_real_reward

                if early_stop:
                    break

            win_clks.append(current_day_win_clks)
            win_imps.append(current_day_win_imps)
            win_reward.append(current_day_reward)



        train_record_pd = pd.DataFrame(data=[[record_time, record_total_ROL, record_action, record_lambda]], columns=['time', 'ROL_t','action', 'lambda'])
        total_win_clk = np.sum(win_clks)
        total_win_imps = np.sum(win_imps)
        total_reward = np.sum(win_reward)

        # print(state_action_pair)
        # 算法2
        for (s,a,r,s_,d) in tuple(state_action_pair):
            net_reward = model.get_reward(s, a)[0]
            max_reward = max(net_reward, total_reward)
            # print(s, a, max_reward, s_, d)
            model.Reward_memory.push(s, a, max_reward, s_, d)


        if best_train_clk < total_win_clk:
            print('*******best_model has been save*******')
            best_train_clk = total_win_clk
            best_model = copy.deepcopy(model)

        train_msg = 'Train_result:win_clk:{0}, win_imps: {1}, win_reward: {2:>5.6f}, Q_loss:{3:>5.6f} Reward_loss:{4:>5.6f} ' \
              'total reward:{5} early_stop_time:{6}'.format(total_win_clk, total_win_imps, total_reward, np.mean(episode_Q_loss),
                                                            np.mean(episode_reward_loss), total_reward, early_stop_time)
        logger.info('Epoch [{}/{}]'.format(episode + 1, config.num_train_epochs))

        logging.info(train_msg)
        print(train_msg)

        test_record_pd, test_result_pd, test_msg = test_model(config, model, test_data)
        logging.info(test_msg)
        print(test_msg)

        if config.save_action_lambda:
            train_record_pd.to_csv(config.result_path + 'train_record/{}episode_action_lambda'.format(episode+1))
            test_record_pd.to_csv(config.result_path + 'test_record/{}episode_action_lambda'.format(episode+1))
    best_model_result, best_result, best_test_msg =  test_model(config, best_model, test_data)
    print(best_test_msg)
    logger.info('Best_model in test result:' + best_test_msg)
    return model

