import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt
import numpy as np
import pickle

quat_type = "stocks-v0"
env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)
# env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)

action_dim = env.action_space.n

episode = 100
T = 0
episode_data = {}
for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
    episode_data[k] = []
for _ in range(episode):
    observation = env.reset()
    while True:
        action = np.random.rand(action_dim)
        next_observation, reward, done, _ = env.step(np.argmax(action))
        episode_data['observations'].append(observation.reshape(-1).astype('float32'))
        episode_data['next_observations'].append(next_observation.reshape(-1).astype('float32'))
        episode_data['actions'].append(action)
        episode_data['rewards'].append(np.array([reward]).astype('float32'))
        episode_data['terminals'].append(done)
        if done:
            break

for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
    episode_data[k] = np.stack(episode_data[k])
with open('/home/kjlin0508/Course_work/AI_Intro/RL_for_Quatitatitive_Trading/Trajectory_Transformer/trajectory/datasets/Random/{}_r'.format(quat_type) + '.pkl', 'wb') as f:
    pickle.dump(episode_data, f)