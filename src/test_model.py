from src.SAC_AUTO_BID.main import Config
from src.SAC_AUTO_BID.base_train_test.train_eval_impression import model_test
import pandas as pd
import numpy as np
# from src.model.SAC import SAC
from src.model.SAC_optimal.sac import SAC
import time


if __name__ == "__main__":
    config = Config()
    train_data = pd.read_csv(config.train_set)
    test_data = pd.read_csv(config.test_set)

    model = SAC(config)
    model.load(config.model_saved_path + '/test-271/')
    # model.load('../temp/1.6/1.6-251-265/model/1.6/')
    Min_Threshold_pctr = train_data[train_data.clk.isin([1])].pctr.min()
    Max_Threshold_pctr = train_data[train_data.clk.isin([1])].pctr.max()

    average_pctr = 0.0033552027734940937
    train_data_clk = train_data[train_data.clk.isin([1])]
    fraction_type = config.fraction_type
    Budget_allocate_by_clk_rate = pd.Series([0.0 for i in range(fraction_type)])
    for i in range(fraction_type):
        Budget_allocate_by_clk_rate[i] = np.sum(train_data_clk[train_data_clk.time_fraction == i].clk) / np.sum(train_data_clk.clk)
    cost_per_pctr = np.sum(train_data_clk.market_price) / np.sum(train_data_clk.pctr)

    start_time_episode = time.time()
    total_budget = []
    for index, day in enumerate(train_data.day.unique()):
        current_day_budget = np.sum(train_data[train_data.day.isin([day])].market_price)
        total_budget.append(current_day_budget)

    config.budget_total = np.divide(total_budget, config.budget_para_int)

    test_results, test_action, test_end_slot_time, test_total_reward = model_test(config, train_data, test_data, model,
                                                                                  Min_Threshold_pctr,
                                                                                  Max_Threshold_pctr,
                                                                                  Budget_allocate_by_clk_rate,
                                                                                  cost_per_pctr, average_pctr)
    end_time_episode = time.time()
    time_dif_episode = end_time_episode - start_time_episode
    test_msg = 'test_result: win_clk:{0}, end_slot_time:{1}, episode_time:{2}, total_reward:{3}'.format(
        np.sum(test_results.win_clks), test_end_slot_time, time_dif_episode, test_total_reward)
    print(test_msg)
    print('*************************test_result***********************')
    print(test_msg)