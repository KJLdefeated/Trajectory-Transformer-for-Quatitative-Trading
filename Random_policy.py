import os
import torch
import random
import numpy as np

from env.market import Market
from helper.args_parser import model_launcher_parser
from helper.data_logger import generate_algorithm_logger, generate_market_logger



def get_stock_code_and_action(env, a, use_greedy=False, use_prob=False):
    # Reshape a.
    if not use_greedy:
        a = a.reshape((-1,))
        # Calculate action index depends on prob.
        if use_prob:
            # Generate indices.
            a_indices = np.arange(a.shape[0])
            # Get action index.
            action_index = np.random.choice(a_indices, p=a)
        else:
            # Get action index.
            action_index = np.argmax(a)
    else:
        #if use_prob:
        #    # Calculate action index
        #    #if np.random.uniform() < self.epsilon:
        #        action_index = np.floor(a).astype(int)
        #    else:
        #        action_index = np.random.randint(0, self.a_space)
        
            # Calculate action index
        action_index = np.floor(a).astype(int)

    # Get action
    action = action_index % 3
    # Get stock index
    stock_index = np.floor(action_index / 3).astype(np.int)
    # Get stock code.
    stock_code = env.codes[stock_index]

    return stock_code, action, action_index

def main():
    #mode = args.mode
    mode = 'test'
    # codes = args.codes
    codes = ["2303"]
    # codes = ["AU88", "RB88", "CU88", "AL88"]
    # codes = ["T9999"]
    # market = args.market
    market = 'future'
    # episode = args.episode
    episode = 1000
    training_data_ratio = 0.95
    # training_data_ratio = args.training_data_ratio

    model_name = os.path.basename(__file__).split('.')[0]

    env = Market(codes, start_date="2012-01-01", end_date="2018-01-01", **{
        "market": market,
        "mix_index_state": False,
        "logger": generate_market_logger(model_name),
        "training_data_ratio": training_data_ratio,
    })
    state = env.reset()
    while True:
        a = random.randint(0, env.trader.action_space-1)
        s_next, r, status, info = env.forward("2303", a)
        print(s_next, r)
        if status == env.Done:
            break

main()