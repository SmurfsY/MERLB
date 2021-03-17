import gensim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import copy
import os

'''
clk,market_price,hour,pctr,minutes,time_fraction
构造数据的输入
'''


class Data_example():   # 定义输入每一个impression
    def __init__(self, click=None, market_price=None, hour=None, pctr=None, minutes=None, time_fraction=None):
        self.click = click
        self.market_price = market_price
        self.hour = hour
        self.pctr = pctr
        self.minutes = minutes
        self.time_fraction = time_fraction

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class State_example():
    def __init__(self, e_clks=None, e_cost=None, e_profits=None, actions=None, state_=None, real_clks=None, bid_nums=None, imps=None):
        self.e_clks = e_clks
        self.e_cost = e_cost
        self.e_profits = e_profits
        self.actions = actions
        self.state_ = state_
        self.real_clks = real_clks
        self.bid_nums = bid_nums
        self.imps = imps

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class AdvData(Dataset):          #构造
    def __init__(self, data, config):
        self.data = data

    def __getitem__(self, index):
        temp_data = self.data[index]
        # input_text = np.array(temp_data.input_text).astype(np.long)
        click = np.array(temp_data.click, dtype=int)
        market_price = np.array(temp_data.market_price, dtype=int)
        hour = np.array(temp_data.hour).astype(np.long)
        pctr = np.array(temp_data.pctr).astype(np.long)
        minutes = np.array(temp_data.minutes).astype(np.long)
        time_fraction = np.array(temp_data.time_fraction, dtype=int)
        return click, hour, market_price, minutes, pctr, time_fraction

    def __len__(self):
        return len(self.data)


class Reader():
    def __init__(self, config):
        self.train_path = config.train_set
        self.test_path = config.test_set

    def read_data(self):
        train_data, test_data = [], []
        # print(pd.read_csv(self.train_path, index_col=None))

        train_clk = list(pd.read_csv(self.train_path)['clk'])
        train_market_price = list(pd.read_csv(self.train_path)['market_price'])
        train_hour = list(pd.read_csv(self.train_path)['hour'])
        train_pctr = list(pd.read_csv(self.train_path)['pctr'])
        train_minutes = list(pd.read_csv(self.train_path)['minutes'])
        train_time_fraction = list(pd.read_csv(self.train_path)['time_fraction'])

        test_clk = list(pd.read_csv(self.test_path)['clk'])
        test_market_price = list(pd.read_csv(self.test_path)['market_price'])
        test_hour = list(pd.read_csv(self.test_path)['hour'])
        test_pctr = list(pd.read_csv(self.test_path)['pctr'])
        test_minutes = list(pd.read_csv(self.test_path)['minutes'])
        test_time_fraction = list(pd.read_csv(self.test_path)['time_fraction'])

        for clk, market_price, hour, pctr, minutes, time_fraction in zip(train_clk, train_market_price, train_hour,
                                                                          train_pctr, train_minutes, train_time_fraction):
            example = Data_example(click=clk,
                                   market_price=market_price,
                                   hour=hour,
                                   pctr=pctr,
                                   minutes=minutes,
                                   time_fraction=time_fraction)
            train_data.append(example)
        for clk, market_price, hour, pctr, minutes, time_fraction in zip(test_clk, test_market_price, test_hour,
                                                                          test_pctr, test_minutes, test_time_fraction):
            example = Data_example(click=clk,
                                   market_price=market_price,
                                   hour=hour, pctr=pctr,
                                   minutes=minutes,
                                   time_fraction=time_fraction)
            test_data.append(example)
        return train_data, test_data

# class Config:
#     train_set = '../data/ipinyou/1458/train_data.csv'
#     test_set = '../data/ipinyou/1458/test_data.csv'
#     fraction_type = 24
# config = Config()
# reader = Reader(config)
# train_data, test_data = reader.read_data()
#
# real_fraction_clk = []
# for time_slot in range(config.fraction_type):
#     slot_click = 0
#     for item in enumerate(train_data):
#         # print(item)
#         print(item)
#         slot_click += 1 if (item[:, 5] == time_slot) & (item[:, 0] == 1) else 0
#     real_fraction_clk.append(slot_click)
# print(train_data)


def result_save(config, examples, genders=None, ages=None):   #存储测试集预测结果
    user_ids = []
    for example in examples:
        user_ids.append(example.user_id)
    if genders != None and ages!=None:
        df = pd.DataFrame({'user_id': user_ids, 'gender': genders, 'age': ages})
    elif genders != None:
        df = pd.DataFrame({'user_id': user_ids, 'gender': genders})
    elif ages != None:
        df = pd.DataFrame({'user_id': user_ids, 'age': ages})
    else:
        raise ValueError("no result output, please check")
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    path = os.path.join(config.result_path, 'result.csv')
    df.to_csv(path, index=False)
    return None


