import numpy as np
import pandas as pd
import sys
from src.DRLB.DRLB_get_function.get_time_step_result import get_init_lambda
from src.DRLB.DRLB_main import Config
#

train_data = pd.read_csv('../../data/ipinyou/1458/train_data.csv')
print(np.sum(train_data['market_price']))



