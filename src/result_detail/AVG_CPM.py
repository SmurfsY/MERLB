import sys
import numpy as np
import pandas as pd

train_data = pd.read_csv('../../data/ipinyou/3427/new_train_data.csv')
test_data = pd.read_csv('../../data/ipinyou/3427/new_test_data.csv')

print('train cpm:', np.mean(train_data['market_price']))
print('train cost:', np.sum(train_data['market_price']))
print('test cpm:', np.mean(test_data['market_price']))
print('test cost:', np.sum(test_data['market_price']))

print('train pctr:', np.mean(train_data['pctr']))
print('test pctr:', np.mean(test_data['pctr']))

print('total train clk:', np.sum(train_data['clk']))
print('total test clk:', np.sum(test_data['clk']))