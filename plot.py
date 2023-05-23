import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

'''
def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.title('CartPole-v0')
    plt.xlabel('epoch')
    plt.ylabel('rewards')
'''

def DDQN():
    plt.figure(figsize=(10, 5))
    plt.title('DDQN train')
    plt.xlabel('epoch')
    plt.ylabel('rewards')
    rewards = np.load(".\Rewards\DDQN_rewards.npy").reshape(4000,1)
    '''
    rewards2 = np.load(".\Rewards\DDQN_rewards_iter2_new4000.npy").reshape(150,1)
    rewards = np.concatenate((rewards, rewards2), axis=0)
    np.save(".\Rewards\DDQN_rewards.npy", rewards)
    '''
    rewards_avg = np.mean(rewards, axis=1)
    plt.plot([i for i in range(4000)], rewards_avg[:4000],
             label='DDQN', color='gray')
    plt.legend(loc="best")
    plt.savefig("./Plots/DDQN.png")
    plt.show()
    plt.close()










if __name__ == "__main__":
    '''
    Plot the trend of Rewards
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--DDQN", action="store_true")
    args = parser.parse_args()

        
    os.makedirs("./Plots", exist_ok=True)

    if args.DDQN:
        DDQN()
