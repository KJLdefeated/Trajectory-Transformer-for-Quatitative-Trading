import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
from buildEnv import createEnv
from DDQN import Agent
import torch
from tqdm import tqdm

quat_type = 2330
env = createEnv(2330)
# env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)

testing_agent = Agent(env)
testing_agent.target_net.load_state_dict(torch.load("/home/kjlin0508/Course_work/AI_Intro/RL_for_Quatitatitive_Trading/Tables/DDQN.pt"))

action_dim = env.action_space.n

episode = 100
T = 0
episode_data = {}
for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
    episode_data[k] = []
for _ in tqdm(range(episode)):
    observation = env.reset().reshape(-1)
    while True:
        tempstate = observation.reshape(-1)
        for i in range(12):
            for j in range(4):
                tempstate[i*4+j] = (observation[44+j] - observation[i*4+j])/observation[44+j]
        Q = testing_agent.target_net(torch.FloatTensor(tempstate.reshape(48))).squeeze(0).detach()
        action = int(torch.argmax(Q).numpy())
        next_observation, reward, done, _ = env.step(action)
        observation = next_observation.reshape(48)
        episode_data['observations'].append(tempstate.astype('float32'))
        next_observation = next_observation.reshape(-1)
        tempstate = next_observation
        for i in range(12):
            for j in range(4):
                tempstate[i*4+j] = (next_observation[44+j] - next_observation[i*4+j])/next_observation[44+j]
        episode_data['next_observations'].append(tempstate.astype('float32'))
        episode_data['actions'].append(np.array(Q))
        #print(Q)
        episode_data['rewards'].append(np.array([reward]).astype('float32'))
        episode_data['terminals'].append(done)
        if done:
            break

for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
    episode_data[k] = np.stack(episode_data[k])
with open('/home/kjlin0508/Course_work/AI_Intro/RL_for_Quatitatitive_Trading/Trajectory_Transformer/trajectory/datasets/Medium/stock_{}'.format(quat_type) + '.pkl', 'wb') as f:
    pickle.dump(episode_data, f)