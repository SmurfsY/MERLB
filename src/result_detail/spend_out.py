import pandas as pd
import numpy as np
from tqdm import tqdm


def get_cpc(result):
    result = result[result.clk.isin([1])]
    spend_out_clk = len(result[result.bid_flag.isin(['spend_out'])])
    print('spend_out_clk', spend_out_clk)

    win_clk = len(result[result.bid_flag.isin(['win_clk'])])

    win_clk_result = result[result.bid_flag.isin(['win_clk'])]

    optimized_bid = list(win_clk_result['bid'].values)
    market_price = list(win_clk_result['market_price'].values)
    optimized_win_clk_result_without = 0
    optimized_lsot_clk_result_without = 0
    for index in range(len(win_clk_result)):
        # optimized_bid_current = float(optimized_bid[index][1:-2])
        optimized_bid_current = float(optimized_bid[index])

        if optimized_bid_current >= market_price[index]:
            optimized_win_clk_result_without += 1
        if optimized_bid_current < market_price[index]:
            optimized_lsot_clk_result_without += 1
    print('in win win clk:', optimized_win_clk_result_without)
    print('in win lost clk:', optimized_lsot_clk_result_without)

    optimized_bid = list(result['bid'].values)
    market_price = list(result['market_price'].values)
    optimized_win_without = 0
    optimized_lost_without = 0
    for index in range(len(result)):
        # optimized_bid_current = float(optimized_bid[index][1:-2])
        optimized_bid_current = float(optimized_bid[index])

        if optimized_bid_current >= market_price[index]:
            optimized_win_without += 1
        if optimized_bid_current < market_price[index]:
            optimized_lost_without += 1

    total_cost = np.sum(result[result.bid_flag.isin(['win_imp', 'win_clk'])].market_price)

    # result = result[result.bid_flag.isin(['win_imp', 'win_clk'])]

    total_clk = len(result[result.bid_flag.isin(['lose_clk', 'win_clk'])])
    print('in budget',total_clk)


    result = result[result.bid_flag.isin(['lose_clk'])]
    optimized_bid = list(result['bid'].values)
    market_price = list(result['market_price'].values)
    equal_lost_clk = 0
    for index in range(len(result)):
        # optimized_bid_current = float(optimized_bid[index][1:-2])
        optimized_bid_current = int(optimized_bid[index])
        if optimized_bid_current == market_price[index]:

            equal_lost_clk += 1

    print('equal lost:', equal_lost_clk)



    spend_out_clk = len(result[result.bid_flag.isin(['spend_out'])])




    print('win clk without budget:', optimized_win_without)
    print('win clk with budget lost clk:', optimized_lost_without)

    print('low price lose clk:', win_clk)
    print('win clk spend out:', spend_out_clk)
    print(total_cost)

def get_equal_clk(result):
    spend_out_lose_clk = [0,0,0]
    low_bid_lose_clk = [0,0,0]
    win_clk = [0,0,0]
    test_data = result
    for day_index, day in enumerate(result.day.unique()):
        current_day_data = test_data[test_data.day.isin([day])]
        clks = list(current_day_data['clk'])
        pctrs = list(current_day_data['pctr'])
        market_prices = list(current_day_data['market_price'])
        bid = list(current_day_data['bid'])
        bid_flag = list(current_day_data['bid_flag'])
        try:
            with tqdm(range(len(current_day_data))) as tdqm_t:
                for impression_index in tdqm_t:
                    if bid_flag[impression_index] == 'spend_out':
                        if clks[impression_index] == 1:
                            spend_out_lose_clk[day_index] += 1
                    if bid_flag[impression_index] == 'lose_clk':
                        if clks[impression_index] == 1:
                            low_bid_lose_clk[day_index] += 1
                    if bid_flag[impression_index] == 'win_clk':
                        win_clk[day_index] += 1
        except KeyboardInterrupt:
            tdqm_t.close()
            raise
        tdqm_t.close()
    print(win_clk)
    print(np.sum(win_clk))
    print('早停导致丢失', spend_out_lose_clk)
    print('出价不够高丢失点击',low_bid_lose_clk)

class Config:
    def __init__(self):
        self.camp = 1458
        self.budget_para = 16
        self.episode = 1

        '''
        2	4	8	16
1458	32	20	15	1
3358	12	3	2	2
3386	2	1	1	1
3427	42	38	2	1
3476	3	39	4	3



        '''

if __name__ == "__main__":
    config = Config()
    result_path = '../../result/ipinyou/{}/SAC/{}/no_gap_len/PRE/no_trend/second/refine/refine_reward/test_action/test_action_{}.csv'.format(
        config.camp, config.budget_para, config.episode
    )

    data_path = '../../data/ipinyou/{}/new_test_data.csv'.format(config.camp)
    print('camp:', config.camp)
    print('budget para:', config.budget_para)
    result = pd.read_csv(result_path)
    test_data = pd.read_csv(data_path)

    print('oringinal data len:', len(result))
    result['day'] = test_data['day']
    print(result['day'].unique())
    # 带不带[]
    # result['bid'] = result['bid'].apply(lambda x:float(x))
    result['bid'] = result['bid'].apply(lambda x: float(x[1:-2]))
    result['clk'] = result['clk'].apply(lambda x: int(x))
    days = np.unique(result['day'])

    print('real clk:', np.sum(result['clk']))   # 真实点击

    # result = result[result.bid_flag.isin(['spend_out'])]   #赢标的


    # SAC_LOST_CLK = get_cpc(result)
    get_equal_clk(result)
