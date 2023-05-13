import numpy as np
import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
import pandas as pd

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]

    prices = env.df['Close'].to_numpy()
    prices = prices[start:end]
    signal_features = env.df.loc[:, ['Open', 'Close', 'High', 'Low']].to_numpy()[start:end]
    return prices, signal_features


class MyStocksEnv(StocksEnv):
    _process_data = my_process_data


def createEnv(stock_no, window_size = 12, frame_bounds = (12, 1200)):
    csv_name = 'dataset\stock_data_' + str(stock_no) + '.csv'
    data = pd.read_csv(csv_name)
    read_df = pd.DataFrame(data)
    env = MyStocksEnv(df = read_df, window_size = window_size, frame_bound = frame_bounds)
    return env

createEnv(2330)
