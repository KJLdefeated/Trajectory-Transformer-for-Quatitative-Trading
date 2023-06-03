from DDQN import gen_offline_data
from buildEnv import createEnv
import numpy as np
import pickle

episodes = 1
stock = 2330
env = createEnv(2330)

data = gen_offline_data(episodes, env)

for _ in range(episodes):
    observation = env.reset()
    while True:
        probs = np.random.rand(2)
        action = np.random.choice(2, p=probs/np.sum(probs))
        next_observation, reward, done, _ = env.step(action)
        data['observations'].append(observation.reshape(-1).astype('float32'))
        data['next_observations'].append(next_observation.reshape(-1).astype('float32'))
        data['actions'].append(probs)
        data['rewards'].append(np.array([reward]).astype('float32'))
        data['terminals'].append(done)
        if done:
            break

with open('Trajectory_Transformer/trajectory/datasets/Medium/DDQN_{}_{}'.format(episodes, stock) + '.pkl', 'wb') as f:
    pickle.dump(data, f)
