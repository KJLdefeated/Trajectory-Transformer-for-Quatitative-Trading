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

def my_calculate_reward(self, action):
    step_reward = -1

    trade = False
    if ((action == Actions.Buy.value and self._position == Positions.Short) or
        (action == Actions.Sell.value and self._position == Positions.Long)):
        trade = True
    if trade:
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price
        step_reward = 5
        if action == Actions.Sell.value:
            step_reward += price_diff
        #else:
        #    step_reward -= price_diff

    return step_reward


class MyStocksEnv(StocksEnv):
    _process_data = my_process_data
    _calculate_reward = my_calculate_reward


def createEnv(stock_no, window_size = 12, frame_bounds = (12, 1200)):
    csv_name = 'dataset\stock_data_' + str(stock_no) + '.csv'
    data = pd.read_csv(csv_name)
    read_df = pd.DataFrame(data)
    read_df = read_df.loc[::-1].reset_index(drop=True)
    env = MyStocksEnv(df = read_df, window_size = window_size, frame_bound = frame_bounds)
    return env

createEnv(2330)
