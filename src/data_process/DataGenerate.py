import numpy as np
import pandas as pd
from tqdm import tqdm
'''
构造输入数据
clk market_price hour pctr,minutes time_fraction
'''


class Config():
    # 路径
    def __init__(self):
        self.day = {
            '1458': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '2259': [19, 20, 21, 22, 23, 24, 25],
            '2261': [24, 25, 26, 27, 28],
            '2821': [21, 22, 23, 24, 25, 26],
            '2997': [23, 24, 25, 26, 27],
            '3358': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3386': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3427': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3476': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        }
        self.div_data = ['train', 'test']
        self.Data_path = '../../data/'
        self.data_set = 'ipinyou'
        self.campaign_id = '3358'
        self.data_log_path = self.Data_path + self.data_set + '/' + self.campaign_id +'/'
        self.pctr_path = self.Data_path + self.data_set + '/new_auc_submission/' + self.campaign_id +'/'

    #数据参数


def main(config):
    # for index, day in tqdm(enumerate(config.div_data)):
    for index, day in tqdm(enumerate(config.day[config.campaign_id])):
        train_data = pd.read_csv(config.data_log_path + '{}_data.csv'.format(day))
        # train_data = pd.read_csv(config.data_log_path + '{}_log.csv'.format(day), sep='\t')
        # train_ctrs = pd.read_csv(config.data_log_path + '{}_submission.csv'.format(day), header=None)
        train_ctrs = pd.read_csv(config.pctr_path + '{}_test_submission.csv'.format(day), header=None)
        # print(train_data)
        train_log = {'clk': train_data['clk'].apply(lambda x: int(x)),
                     # 'market_price':train_data['payprice'].apply(lambda x:int(x)),
                     'market_price': train_data['market_price'].apply(lambda x: int(x)),
                     'hour': train_data['hour'],
                     'pctr': train_ctrs.values[:, 1],
                     'minutes': train_data['minutes'],
                     # 'minutes': train_data['timestamp'].apply(lambda x: int(x)),
                     'time_fraction': train_data['time_fraction'],
                     'day': train_data['minutes'].apply(lambda x: int(str(x)[6:8]))
                     }
        train_data = pd.DataFrame(train_log)
        train_data.sort_values(by='minutes', ascending=True, inplace=True)
        print('Saving......')
        train_data.to_csv(config.data_log_path + '{}_new_auc_data.csv'.format(day), index=False)



if __name__ == '__main__':

    config = Config()
    main(config)

