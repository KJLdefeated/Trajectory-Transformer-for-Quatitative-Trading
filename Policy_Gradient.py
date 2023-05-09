import os
import torch

from env.market import Market
from helper.args_parser import model_launcher_parser
from helper.data_logger import generate_algorithm_logger, generate_market_logger



def main():
    #mode = args.mode
    mode = 'test'
    # codes = args.codes
    codes = ["600036"]
    # codes = ["600036", "601998"]
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
        #"logger": generate_market_logger(model_name),
        "training_data_ratio": training_data_ratio,
    })

main()