import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.optim.lr_scheduler as Scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from buildEnv import createEnv, MyStocksEnv
from torch.distributions import Categorical
import logging
#from skopt.space import Real, Integer
#from skopt import gp_minimize
logging.basicConfig(filename='train.log', level=logging.DEBUG)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network
        self.linearlayer1 = nn.Linear(num_inputs, hidden_size)
        self.linearlayer2 = nn.Linear(hidden_size, hidden_size)
        self.linearlayer3 = nn.Linear(hidden_size, hidden_size)
        self.outputlayer = nn.Linear(hidden_size, num_outputs)
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        x = self.linearlayer1(inputs)
        x = F.relu(x)
        x = self.linearlayer2(x)
        x = F.relu(x)
        x = self.linearlayer3(x)
        x = F.relu(x)
        action = self.outputlayer(x)
        return action
        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own critic network
        self.linearlayer1 = nn.Linear(num_inputs+num_outputs, hidden_size)
        self.linearlayer2 = nn.Linear(hidden_size, hidden_size)
        self.linearlayer3 = nn.Linear(hidden_size, hidden_size)
        self.outputlayer = nn.Linear(hidden_size, 1)
        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network
        x = self.linearlayer1(torch.cat([inputs, actions], dim=-1))
        x = F.relu(x)
        x = self.linearlayer2(x)
        x = F.relu(x)
        x = self.linearlayer3(x)
        x = F.relu(x)
        v = self.outputlayer(x)
        return v
        ########## END OF YOUR CODE ##########        
        

class DDPG(object):
    def __init__(self, num_inputs, action_space, env, epsilon, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-4, lr_c=1e-3, lr_a_decay=0.995, lr_c_decay=0.995, step_size=100):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)
        self.actor_scedule = Scheduler.StepLR(self.actor_optim, step_size=step_size, gamma=lr_a_decay)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)
        self.critic_scedule = Scheduler.StepLR(self.critic_optim, step_size=step_size, gamma=lr_c_decay)

        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.env = env

        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, epsilon=0.0):
        self.actor.eval()
        probs = self.actor((Variable(state)))
        probs = probs.detach()
        m = Categorical(logits= probs)
        action = m.sample().item()
        if random.random() > epsilon:
            return self.env.action_space.sample()
        return action

    def update_parameters(self, batch):
        state_batch = Variable(torch.cat([b.state for b in batch]))
        action_batch = Variable(torch.cat([b.action for b in batch]))
        reward_batch = Variable(torch.cat([b.reward for b in batch]))
        mask_batch = Variable(torch.cat([b.mask for b in batch]))
        next_state_batch = Variable(torch.cat([b.next_state for b in batch]))
        
        # Calculate policy loss and value loss
        # Update the actor and the critic
        q_v = self.critic(state_batch, action_batch)
        next_action = self.actor_target(next_state_batch)
        next_q = self.critic_target(next_state_batch, next_action)
        q_target = reward_batch.view(-1,1) + self.gamma * next_q * (1-mask_batch.view(-1,1))
        value_loss = F.mse_loss(q_v, q_target)
        #update critic
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()
        #update actor
        ploss = self.critic(state_batch, self.actor(state_batch))
        policy_loss = -ploss.mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/lunarlander/'):
            os.makedirs('preTrained/lunarlander/')

        if actor_path is None:
            actor_path = "preTrained/lunarlander/ddpg_actor_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic_path is None:
            critic_path = "preTrained/lunarlander/ddpg_critic_{}_{}_{}".format(env_name, timestamp, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

def train(env:MyStocksEnv, lr_a_, lr_c_, lr_a_decay_, lr_c_decay_, noise_scale_, batch_size_ , env_name = 'Stock_Market'):   
    # Define a tensorboard writer
    #writer = SummaryWriter("./tb_record_3/DDPG/train-{}-{}".format(lr_a_, lr_c_))

    logging.info('lr_a = {}, lr_c = {} , lr_a_decay={} , lr_c_decay={}, noise_scale = {} , batch_size = {}'.format(
        lr_a_, lr_c_, lr_a_decay_, lr_c_decay_, noise_scale_, batch_size_))
    torch.manual_seed(10)

    num_episodes = 1000
    gamma = 0.995
    tau = 0.002
    lr_a = lr_a_ #1e-4
    lr_c = lr_c_ #1e-3
    lr_a_decay=lr_a_decay_ #0.995
    lr_c_decay=lr_c_decay_ #0.995
    hidden_size = 128
    noise_scale = noise_scale_ #0.3
    replay_size = 100000
    batch_size = batch_size_ #128
    epsilon = 0.03
    updates_per_step = 1
    print_freq = 20
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0

    agent = DDPG(num_inputs = env.reset().reshape(-1).shape[0],
                 action_space = env.action_space.n, 
                 env = env, 
                 epsilon= epsilon,
                 gamma = gamma, 
                 tau = tau, 
                 hidden_size = hidden_size,
                lr_a= lr_a, 
                lr_c= lr_c, 
                lr_a_decay= lr_a_decay, 
                lr_c_decay = lr_c_decay)
    #ounoise = OUNoise(env.action_space)
    memory = ReplayMemory(replay_size)
    
    for i_episode in range(num_episodes):
        
        #ounoise.scale = noise_scale
        #ounoise.reset()
        
        state = torch.Tensor([env.reset().reshape(-1)])

        episode_reward = 0
        val_loss = []
        act_loss = []
        while True:
            
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples 
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic
            total_numsteps+=1
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            next_state = torch.Tensor([next_state.reshape(-1)])
            memory.push(state, action, torch.Tensor([done]), next_state, torch.Tensor([reward]))
            if len(memory) >= batch_size and total_numsteps%updates_per_step == 0:
                batch = memory.sample(batch_size)
                v_loss, a_loss = agent.update_parameters(batch)
                val_loss.append(v_loss)
                act_loss.append(a_loss)
            episode_reward += reward
            state = next_state
            if done:
                break
            ########## END OF YOUR CODE ########## 
        

        rewards.append(episode_reward)
        actor_loss = np.mean(act_loss)
        critic_loss = np.mean(val_loss)
        t = 0
        
        state = torch.Tensor([env.reset().reshape(-1)])
        episode_reward = 0
        while True:
            action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            
            #env.render()
            
            episode_reward += reward

            next_state = torch.Tensor([next_state.reshape(-1)])

            state = next_state
            
            t += 1
            if done:
                print(info)
                break

        rewards.append(episode_reward)
        # update EWMA reward and log the results
        ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
        ewma_reward_history.append(ewma_reward)
        if i_episode % print_freq == 0:
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}, val loss: {:.2f}, act loss: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward, critic_loss, actor_loss))    
            logging.info("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}, val loss: {:.2f}, act loss: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward, critic_loss, actor_loss))

        #Logging
        #writer.add_scalar('Reward', episode_reward, i_episode)
        #writer.add_scalar('EWMA Reward', ewma_reward, i_episode)
        #writer.add_scalar('Critic loss', critic_loss, i_episode)
        #writer.add_scalar('Actor loss', actor_loss, i_episode)

        #if ewma_reward >= 120:
        #    agent.save_model(env_name, '.pth')
        #    logging.info("Running reward is now {} and the total episode is {}.".format(ewma_reward, i_episode))
        #    #break
        #    return (ewma_reward+500)/(i_episode+1) #For tuning
    
    agent.save_model(env_name, '.pth')  
    logging.info("Running reward is now {} and the total episode is {}.".format(ewma_reward, i_episode))
    return (ewma_reward+500)/(i_episode+1) #For tuning

def main():
    """
    Training args
    """
    lr_a = 0.001
    lr_c = 0.001
    lr_a_decay_ = 1.0
    lr_c_decay_ = 1.0
    noise_scale_ = 0.3
    batch_size_ = 64


    env = createEnv(2330)

    train(env, lr_a, lr_c, lr_a_decay_, lr_c_decay_, noise_scale_, batch_size_)
if __name__ == '__main__':
    main()