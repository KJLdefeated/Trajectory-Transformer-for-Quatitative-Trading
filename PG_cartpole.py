import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from torch import Tensor
import numpy as np
import gym
import random
from collections import deque
from torch.distributions import Categorical
import os
from tqdm import tqdm
import buildEnv

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class Agent():
    def __init__(self, env, learning_rate=0.01, GAMMA=0.99, batch_size=5):
        self.env = env
        self.n_actions = 2  # the number of actions
        self.count = 0

        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size

        self.policy_net = PolicyNet()

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)


total_rewards = []

def train(env):
    agent = Agent(env)
    episode = 10000
    rewards = []
    for e in range(episode):
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)
        # env.render(mode='rgb_array')

        state_pool = []
        action_pool = []
        reward_pool = []
        steps = 0
        ep_rew = 0

        while True:
            probs = agent.policy_net(Tensor(state).reshape(4))

            m = Categorical(logits=probs)
            action = m.sample()
            next_state, reward, done, _ = env.step(action.item())
            # env.render(mode='rgb_array')

            ep_rew += reward
            state_pool.append(state)
            action_pool.append((m.log_prob(action), probs))
            reward_pool.append(reward)

            state = next_state
            state = torch.from_numpy(state).float()
            state = Variable(state)

            steps += 1

            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                rewards.append(ep_rew)
                print("{}: rew: {}".format(e, ep_rew))
                break

        if e > 0 and e % agent.batch_size == 0:
            # Discount reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * agent.gamma + reward_pool[i]
                    reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            agent.optimizer.zero_grad()

            for i in range(steps):
                state = state_pool[i]
                action = action_pool[i]
                reward = reward_pool[i]

                loss = -action[0] * reward  # Negtive score function x reward
                loss.backward()

            agent.optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0
            torch.save(agent.policy_net.state_dict(), "./Tables/PG_cartpole.pt")
    total_rewards.append(rewards)



def test(env):
    rewards = []
    #profits = []
    testing_agent = Agent(env)
    testing_agent.policy_net.load_state_dict(torch.load("./Tables/PG_cartpole.pt"))
    for _ in range(30):
        state = env.reset()
        reward = 0
        while True:
            probs = testing_agent.policy_net(Tensor(state).reshape(4))
            m = Categorical(logits=probs)
            action = m.sample()
            next_state, temp, done, _ = env.step(action.item())
            reward = reward + temp
            if done:
                rewards.append(reward)
                #profits.append(env._total_profit)
                #print(env._total_profit)
                break
            state = next_state

    print(f"reward: {np.mean(rewards)}")
    #print(f"profit: {np.mean(profits)}")
    #print(env.max_possible_profit())


if __name__ == "__main__":
    env = gym.make('CartPole-v0') 
    #print(env.shape)  
    os.makedirs("./Tables", exist_ok=True)

    # training section:
    train(env)
        
    # testing section:
    test(env)
    env.close()

    os.makedirs("./Rewards", exist_ok=True)
    np.save("./Rewards/PG_cartpole_rewards.npy", np.array(total_rewards))
