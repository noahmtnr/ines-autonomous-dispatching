"""
Test Class for Random Agent.
"""

# imports
import numpy as np
import pandas as pd
import json
import os
import shutil
import sys
import gym

import ray


from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
sys.path.insert(0,"")

from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from gym_graphenv.envs.GraphworldManhattan import GraphEnv

env=GraphEnv()

env.availableTrips()

"""
Run the random agent for 30 steps.
:param: Environment Object
:return: Float (sum of the agent's reward)
"""
def run_one_episode (env):
    env.reset()
    sum_reward = 0
    for i in range(30):
        env.available_actions = env.get_available_actions()
        print(env.available_actions)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        sum_reward+=reward
        #env.render()
        if done:
            print("DELIVERY DONE! sum_reward: ",sum_reward, " time: ",env.time,  "deadline time: ", env.deadline,"pickup time: ", env.pickup_time )
            break

        print("sum_reward: ",sum_reward, " time: ",env.time, "deadline time: ", env.deadline, "pickup time: ", env.pickup_time)
    return sum_reward

for i in range(10):
    run_one_episode(env)