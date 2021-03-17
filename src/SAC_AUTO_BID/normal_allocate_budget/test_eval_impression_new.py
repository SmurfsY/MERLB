import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from src.SAC_AUTO_BID.RL_set.reward_function import get_reward,get_init_lambda
from src.SAC_AUTO_BID.RL_set.get_state import get_state_0_average_pctr, get_state_1_optimized_pctr, \
    get_state_2_cost_rate, get_state_4_remain_budget, get_state_3_Budget_allocate_right_rate, get_state_5_remain_time,get_refine_bid

logger = logging.getLogger(__name__)


def test_model(model, config, CPM):
    train_data = pd.read_csv(config.train_set)
    test_data = pd.read_csv(config.test_set)

    # test!
    fraction_type = config.fraction_type  # 分片粒度


    # 训练集相关数据
    Min_Threshold_pctr = train_data[train_data.clk.isin([1])].pctr.min()
    Max_Threshold_pctr = train_data[train_data.clk.isin([1])].pctr.max()

    # 第一天来自于原始数据集的平均，后天的用更新
    original_avg_train_pctr = np.mean(list(train_data.pctr))
    temp_pctr = np.mean(list(train_data.pctr))
    pctr_list = [temp_pctr, len(train_data)]

    train_data_clk = train_data[train_data.clk.isin([1])]
    ctr = len(train_data_clk) / len(train_data)  #ctr
    # state3需求
    Budget_allocate_by_clk_rate = pd.Series([0.0 for i in range(fraction_type)])
    for i in range(fraction_type):
        Budget_allocate_by_clk_rate[i] = np.sum(train_data_clk[train_data_clk.time_fraction == i].clk) / np.sum(train_data_clk.clk)
    if config.trend_action or config.bid_function == 'refine':
        HB_base_bid = get_init_lambda(config)
    else:
        HB_base_bid = 1

    # 预算设置
    total_budget = []
    for index, day in enumerate(test_data.day.unique()):
        current_day_budget = np.sum(test_data[test_data.day.isin([day])].market_price)
        total_budget.append(current_day_budget)

    total_budget = np.divide(total_budget, config.budget_para_int)
    logger.info("Test_Total_budget:%s", total_budget)

    print('*********************STARTING test**********************')
    print(total_budget)
    end_slot_time = False

    test_action = []
    total_reward = 0
    win_pctr = 0


    # state2相关
    recent_cost_state = 0.0  # state:2
    recent_cost = 0  # 用于累计前面的impression花费          # state2 相关
    cost_index = 0  # 拍得次数

    # 统计每个回合总的收益情况
    total_win_clks = []  # win的点击数
    total_cost = []  # 花费
    total_no_clk_imp = []  # 时段中没有点击的impression数
    total_real_clks = []  # 真实点击数
    total_bid_nums = []  # 出价次数
    total_win_imps = []  # 赢标次数

    end_time_all = []
    for day_index, day in enumerate(test_data.day.unique()):
        # 构造当前天的数据
        current_day_data = test_data[test_data.day.isin([day])]
        clks = list(current_day_data['clk'])
        pctrs = list(current_day_data['pctr'])
        market_prices = list(current_day_data['market_price'])
        time_frac = list(current_day_data['time_fraction'])
        days = list(current_day_data['day'])
        current_data_time = list(
            current_day_data['minutes'].apply(lambda x: int(str(x)[8:10]) * 60 + int(str(x)[10:12])))

        print('CPM:', CPM[0])

        bid_pctrs = [0 for i in range(fraction_type)]  # 总的PCTR
        win_clks = [0 for i in range(fraction_type)]  # win的点击数
        cost = [0 for i in range(fraction_type)]  # 花费
        no_clk_imp = [0 for i in range(fraction_type)]  # 时段中没有点击的impression数
        real_clks = [0 for i in range(fraction_type)]  # 真实点击数
        bid_nums = [0 for i in range(fraction_type)]  # 出价次数
        win_imps = [0 for i in range(fraction_type)]  # 赢标次数



        today_total_budget = total_budget[day_index]

        # 如果均匀分配预算
        if config.budget_allocate:
            N = config.budget_allocate_num
            n = N
            count_num = 0

            per_1000_budget = CPM[0] * config.budget_allocate_num  # 根据动态调整的CPM分配此次预算
            current_allocate_budget = min(per_1000_budget, today_total_budget)  # 如果预算不够则剩下的所有钱全部分配
            remain_budget = current_allocate_budget  # 剩余预算赋值
            today_total_budget -= remain_budget  # 总预算中减去分配出去的预算

            today_budget = current_allocate_budget
        else:
            today_budget = today_total_budget

        current_day = day_index
        early_stop = False

        reset_rate = pctr_list[0] / temp_pctr  # 根据当前平均pctr变化率  修改
        Max_Threshold_pctr = Max_Threshold_pctr * reset_rate  # 根据平均pctr的变化来改变Max
        Min_Threshold_pctr = Min_Threshold_pctr * reset_rate
        temp_pctr = pctr_list[0]

        try:
            with tqdm(range(len(current_day_data))) as tdqm_t:
                for impression_index in tdqm_t:
                    """根据当前每条数据修改所对应的环境参数"""

                    # 在键值对中修改  来获取最后的平均PCTR
                    total_pctr = pctr_list[0] * pctr_list[1] + pctrs[impression_index]
                    pctr_num = pctr_list[1] + 1
                    pctr_list = [total_pctr / pctr_num, pctr_num]

                    if impression_index % config.pctr_update_item == 0:
                        state_0 = get_state_0_average_pctr(pctr_list)
                    state_1 = get_state_1_optimized_pctr(pctrs[impression_index], Min_Threshold_pctr,
                                                         Max_Threshold_pctr)
                    state_2, recent_cost_state, recent_cost = get_state_2_cost_rate(recent_cost_state,
                                                                                    impression_index, recent_cost,
                                                                                    config, today_budget)
                    state_3 = get_state_3_Budget_allocate_right_rate(time_frac[impression_index], real_clks,
                                                                     win_clks,
                                                                     Budget_allocate_by_clk_rate)
                    # state_4 = get_state_4_remain_budget(remain_budget, per_1000_budget)
                    state_4 = get_state_4_remain_budget(remain_budget, per_1000_budget)
                    state_5 = get_state_5_remain_time(n, config)
                    origin_bid = get_refine_bid(original_avg_train_pctr, pctrs[impression_index], HB_base_bid)
                    state_6 = origin_bid / 300

                    if config.reward_type == '012456':
                        state = np.array([state_0, state_1, state_2, state_4, state_5, state_6])
                    elif config.reward_type == '01456':
                        state = np.array([state_0, state_1, state_4, state_5, state_6])
                    elif config.reward_type == 'all':
                        state = np.array([state_0, state_1, state_2, state_3, state_4, state_5, state_6])

                    if config.model_type == 'DQN':
                        action = model.choose_action(state)
                        bid = action
                    elif config.model_type == 'SAC':
                        # action中的映射
                        if config.action_gap:
                            action = model.choose_action(state, evaluate=True)
                            if impression_index == 0:
                                min_action = action
                                max_action = action
                            min_action = min(action, min_action)
                            max_action = max(action, max_action)
                            gap_len = max_action - min_action
                            action = (action - min_action) / gap_len if gap_len != 0 else (action - min_action)
                        else:
                            action = model.choose_action(state)
                            action = action / 2 + 0.5
                            # 正常原始模型出价

                        # refine/direct
                        if config.bid_function == 'direct':
                            # 出价
                            bid = action * 300
                            bid = int(np.where(bid >= 300, 300, bid))
                        elif config.bid_function == 'refine':
                            action = (action - 0.5) * 2
                            origin_bid = get_refine_bid(original_avg_train_pctr, pctrs[impression_index], HB_base_bid)
                            temp_len = min(origin_bid, 300 - origin_bid)
                            bid = origin_bid + temp_len * action
                            bid = bid[0]

                    # 时段特征
                    t = time_frac[impression_index]
                    if clks[impression_index] == 1:
                        real_clks[t] += 1
                    else:
                        no_clk_imp[t] += 1

                    bid_pctrs[t] += pctrs[impression_index]
                    bid_nums[t] += 1

                    # 赢标特征
                    if bid > market_prices[impression_index] and bid <= remain_budget:
                        # 修改CPM
                        CPM[0] = (CPM[0] * CPM[1] + market_prices[impression_index]) / CPM[1]
                        CPM[1] = CPM[1] + 1

                        if config.first_bid:
                            recent_cost += bid  # 状态相关
                            cost[t] += bid
                            remain_budget -= bid
                        else:
                            recent_cost += market_prices[impression_index]  # 状态相关
                            cost[t] += market_prices[impression_index]
                            remain_budget -= market_prices[impression_index]

                        cost_index += 1
                        bid_flag = 'win_imp'
                        win_imps[t] += 1




                        win_pctr += pctrs[impression_index]
                        # 赢标点击特征
                        if clks[impression_index] == 1:
                            if pctrs[impression_index] > Max_Threshold_pctr:
                                Max_Threshold_pctr = pctrs[impression_index]  # 修改最大的pctr阈值
                            if pctrs[impression_index] < Min_Threshold_pctr:
                                Min_Threshold_pctr = pctrs[impression_index]  # 修改最小的pctr阈值
                            bid_flag = 'win_clk'

                            win_clks[t] += 1
                    else:
                        # 预算花光的情况下
                        if config.budget_allocate:
                            if bid > remain_budget:
                                bid_flag = 'spend_out'
                            else:
                                bid_flag = 'lose_imp'
                                if clks[impression_index] == 1:
                                    bid_flag = 'lose_clk'
                        else:
                            if np.sum(cost) + bid > today_budget:
                                bid_flag = 'spend_out'
                            else:
                                bid_flag = 'lose_imp'
                                if clks[impression_index] == 1:
                                    bid_flag = 'lose_clk'


                    # 记录早停情况
                    if not config.budget_allocate:
                        if not early_stop:
                            print('early stop:', t)
                            if np.sum(cost) + bid > today_budget:
                                early_stop = time_frac[impression_index]

                        # (s, a, s_, r, d)
                    reward = get_reward(bid_flag, pctrs, state_1, state_4, Min_Threshold_pctr, Max_Threshold_pctr,
                                        impression_index, state_0, ctr, bid, HB_base_bid, config, origin_bid,
                                        market_prices[impression_index], action[0])
                    total_reward += reward

                    if config.budget_allocate:
                        n -= 1
                        if n == 0:
                            today_total_budget += remain_budget  # 上一次分配剩余的预算加回来
                            n = N  # 重置计数器
                            per_1000_budget = CPM[0] * config.budget_allocate_num  # 根据动态调整的CPM分配此次预算
                            current_allocate_budget = min(per_1000_budget, today_total_budget)  # 如果预算不够则剩下的所有钱全部分配
                            remain_budget = current_allocate_budget  # 剩余预算赋值
                            count_num += 1
                            today_total_budget -= remain_budget  # 总预算中减去分配出去的预算
                            if today_total_budget == 0:  # 如果减去了分配预算 今日预算归零 则早停
                                if not early_stop:
                                    print('early stop:', t)
                                    early_stop = t

                    test_action.append(
                        [state_0, state_1, state_2, state_3, state_4, state_5, state_6, pctrs[impression_index], origin_bid, temp_len,
                         action[0], bid, market_prices[impression_index], clks[impression_index], reward,
                         Min_Threshold_pctr, Max_Threshold_pctr, bid_flag])
        except KeyboardInterrupt:
            tdqm_t.close()
            raise
        tdqm_t.close()

        if not early_stop:
            end_time_all.append('{}F'.format(day))

        # 一天完了之后再训练一次
        total_win_clks.append(np.sum(win_clks))
        total_bid_nums.append(impression_index)
        total_cost.append(np.sum(cost))
        total_no_clk_imp.append((np.sum(no_clk_imp)))  # 时段中没有点击的impression数
        total_real_clks.append(np.sum(real_clks))  # 真实点击数
        total_win_imps.append(np.sum(win_imps))  # 赢标次数

    # 整个数据跑完之后进行统计
    total_bid_nums_all = np.sum(total_bid_nums)
    total_win_clks_all = np.sum(total_win_clks)
    total_cost_all = np.sum(total_cost)
    total_real_clks_all = np.sum(total_real_clks)
    total_win_imps_all = np.sum(total_win_imps)
    total_no_clk_imp_all = np.sum(total_no_clk_imp)

    print('day cost:', total_cost)
    print('test cost:', total_cost_all)

    test_results = pd.DataFrame(data=[[total_win_clks_all, total_cost_all, total_real_clks_all, total_bid_nums_all,
                                       total_win_imps_all, total_no_clk_imp_all, win_pctr, total_reward]],
                                columns=['win_clks', 'cost', 'real_clks', 'bid_nums', 'win_imps', 'no_clk_imp', 'win_pctr', 'win_reward'])
    test_action_pd = pd.DataFrame(data=test_action, columns=['state_0','state_1', 'state_2','state_3', 'state_4',
                                                             'state_5', 'state_6', 'pctr', 'origin_bid', 'temp_len','action', 'bid',
                                                             'market_price', 'clk', 'reward','Min_Threshold_pctr',
                                                             'Max_Threshold_pctr', 'bid_flag'])


    return test_results, test_action_pd, end_time_all



