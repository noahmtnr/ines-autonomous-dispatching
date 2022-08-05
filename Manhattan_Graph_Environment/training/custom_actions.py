# imports
import sys

import numpy as np

sys.path.insert(0, "")
import os;
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv

# initialize environment
env = GraphEnv()
sum_reward = 0
env.reset()
# take random or custom actions -> mainly for debugging purposes
for i in range(10):
    print("Position: ", env.position)
    action = np.random.randint(0, 91)
    #action = 0
    print("Action: ", action)
    state, reward, done, info = env.step(action)
    sum_reward += reward
    if(done):
        env.reset()
