import pandas as pd
import numpy as np

temp = 1458
budget_para = 4
episode = 20
result_path = '../../result/ipinyou/{}/SAC/{}/no_gap_len/PRE/no_trend/second/refine/refine_reward/test_action/test_action_{}.csv'.format(
    temp,budget_para,episode
)

result = pd.read_csv(result_path)
result = result[result.clk.isin([1])]

original_bid = list(result['origin_bid'].values)
optimized_bid = list(result['bid'].values)
market_price = list(result['market_price'].values)

optimized_win = 0
optimized_lose = 0
for index in range(len(result)):
    # optimized_bid_current = float(optimized_bid[index][1:-2])
    optimized_bid_current = float(optimized_bid[index])
    # print(type(original_bid[index]), type(market_price[index]), type(optimized_bid_current))
    if original_bid[index] >= market_price[index] and optimized_bid_current < market_price[index]:
        optimized_lose += 1
    elif original_bid[index] < market_price[index] and optimized_bid_current >= market_price[index]:
        optimized_win += 1

print('optimized_win:', optimized_win)
print('optimized_lose:', optimized_lose)