import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

'''
def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.title('CartPole-v0')
    plt.xlabel('epoch')
    plt.ylabel('rewards')
'''

def DDQN():
    plt.figure(figsize=(10, 5))
    plt.title('Reward1')
    plt.xlabel('epoch')
    plt.ylabel('rewards')
    rewards = np.load("Rewards/DDQN_rewards.npy").reshape(-1)
    '''
    rewards2 = np.load(".\Rewards\DDQN_rewards_iter2_new4000.npy").reshape(150,1)
    rewards = np.concatenate((rewards, rewards2), axis=0)
    np.save(".\Rewards\DDQN_rewards.npy", rewards)
    '''
    rewards_avg = np.mean(rewards)
    plt.plot([i for i in range(len(rewards))], rewards, label='DDQN', color='gray')
    plt.legend(loc="best")
    plt.savefig("./Plots/reward1.png")
    #plt.show()
    plt.close()

def tb_plot():
    reward1 = np.load('Rewards/DDQN_rewards.npy')
    reward2 = np.load('Rewards/DDQN_rewards_oor.npy')
    w1 = SummaryWriter('tb_record_1/Reward_comp/Reward1')
    w2 = SummaryWriter('tb_record_1/Reward_comp/Reward2')
    for i in range(len(reward1)):
        w1.add_scalar('reward', reward1[i], i)
        w2.add_scalar('reward', reward2[i], i)
    

if __name__ == "__main__":
    '''
    Plot the trend of Rewards
    '''   
    #os.makedirs("./Plots", exist_ok=True)

    #DDQN()
    tb_plot()
