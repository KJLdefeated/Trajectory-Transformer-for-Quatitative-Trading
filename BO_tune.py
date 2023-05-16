import gym
from PG_test import train
from skopt.space import Real, Integer
from skopt import gp_minimize

search_space = [
    Real(0, 0.01,name='lr'),
    Real(0.7, 1,name='lr_decay'),
]

def objective(params):
    print(params)
    episodes_num = train(lr=params[0], lr_decay=params[1])
    return episodes_num

result = gp_minimize(objective, search_space, n_calls=5, random_state=0)

print("Best hyperparameters: ", result.x)
print("Best objective value: ", result.fun)
print("Hyperparameters tried: ", result.x_iters)
print("Objective values at each step: ", result.func_vals)