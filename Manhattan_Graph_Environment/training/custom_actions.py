import sys

import numpy as np
sys.path.insert(0,"")
import os ;
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv, CustomCallbacks


env=GraphEnv()

sum_reward=0
env.reset()

for i in range(20):
    print("Position: ", env.position)
    action = np.random.randint(0,69)
    print("Action: ", action)
    state, reward, done, info = env.step(action)
    sum_reward += reward
    action = action
    print("Action: ", action)
    state, reward, done, info = env.step(action)
    sum_reward += reward
