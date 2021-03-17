import pandas as pd
import numpy as np

data_id = '3358'

days = {
            '1458': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '2259': [19, 20, 21, 22, 23, 24, 25],
            '2261': [24, 25, 26, 27, 28],
            '2821': [21, 22, 23, 24, 25, 26],
            '2997': [23, 24, 25, 26, 27],
            '3358': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3386': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3427': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3476': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        }[data_id]
train_days = days[0:-3]
test_days = days[-3:]
dfs = [None] * len(days)

print(days)
print(train_days)
print(test_days)
data_path = '../../data/ipinyou/' + data_id
for idx in range(len(days)):
    day = days[idx]
    dfs[idx] = pd.read_csv(data_path + '/{}_new_auc_data.csv'.format(day))

train_df = pd.concat([dfs[0], dfs[1]])
for index in range(len(train_days)):
    if index == len(train_df):
        break
    if index == 1 or index == 0:
        continue
    train_df = pd.concat([train_df, dfs[index]])


pd.DataFrame(train_df).to_csv(data_path+ '/new_auc_train_data.csv', index=False)

test_df = pd.concat([dfs[-3], dfs[-2]])
test_df = pd.concat([test_df, dfs[-1]])


pd.DataFrame(test_df).to_csv(data_path+ '/new_auc_test_data.csv', index=False)


