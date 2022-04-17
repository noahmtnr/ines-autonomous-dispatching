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

from ManhattanGraph import ManhattanGraph
from gym_graphenv.envs.GraphworldManhattan import GraphEnv

env=GraphEnv()

file_name = "tmp/dqn/graphworld"



trainer.restore(file_name)
env = gym.make("graphworld-v0")
state = env.reset()

sum_reward = 0
n_step = 20
for step in range(n_step):
    action = trainer.compute_action(state)
    state, reward, done, info = env.step(action)
    sum_reward += reward
    #env.render()
    if done == 1:
        print("cumulative reward", sum_reward)
        state = env.reset()
        sum_reward = 0

ray.shutdown()