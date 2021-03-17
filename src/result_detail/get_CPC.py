import pandas as pd
import numpy as np

def get_cpc(result, model):
    HB_model_result = result[result['origin_bid'] >= result['market_price']]
    SACHB_model_result = result[result['bid'] >= result['market_price']]
    print('result_len:', len(result))
    if model == 'HB':
        model_result = HB_model_result
        print('HB win clk:', np.sum(model_result['clk']))

    elif model == 'SAC+HB':
        model_result = SACHB_model_result
        print('SAC+HB win clk:', np.sum(model_result['clk']))
    print('HB win imps:', len(HB_model_result))
    print('SAC+HB win imps:', len(SACHB_model_result))

    model_day_cost =[]
    model_day_clk = []
    model_day_imps = []
    model_day_pctr = []

    for day in days:
        current_day_data = model_result[model_result.day.isin([day])]
        model_day_cost.append(np.sum(current_day_data['market_price']))  #赢标的花费
        model_day_clk.append(np.sum(current_day_data['clk']))   #赢得点击
        model_day_pctr.append(np.sum(current_day_data['pctr']))  # 赢得点击
        model_day_imps.append(len(current_day_data))    #赢得展示机会数

    print('cost:', np.sum(model_day_cost), model_day_cost)
    day_CPC = [cost / clk for cost,clk in zip(model_day_cost, model_day_clk)]
    all_CPC = np.sum(model_day_cost) / np.sum(model_day_clk)

    day_CPM = [cost / clk for cost,clk in zip(model_day_cost, model_day_imps)]
    all_CPM = np.sum(model_day_cost) / np.sum(model_day_imps)

    print('clk:', np.sum(model_day_clk), model_day_clk)
    print('pctr:', np.sum(model_day_pctr), model_day_pctr)
    print('day_cpc:', day_CPC)
    print('all_cpc:', all_CPC)
    print('day_cpm:', day_CPM)
    print('all_cpm:', all_CPM)

    return day_CPC, all_CPC, day_CPM, all_CPM

class Config:
    def __init__(self):
        self.camp = 1458
        self.budget_para = 2
        self.episode = 32

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

    result = result[result.bid_flag.isin(['win_clk', 'win_imp'])]   #赢标的


    SAC_all_CPC, SAC_CPC, SAC_all_CPM, SAC_CPM = get_cpc(result, 'HB')
