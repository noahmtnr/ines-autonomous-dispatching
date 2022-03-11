import gym
from gym.utils import seeding
import numpy as np
import random

class Gridworld_v1(gym.Env): # define custom environment as subclass of gym.Env
    # GLOBAL VARIABLES:

    
    def __init__(self):
        #self.action_space = gym.spaces.Discrete(2)
        #self.observation_space = gym.spaces.Discrete(self.RT_MAX + 1)
        pass

    def step(self, action):
        pass

    def render(self): # method for visualization; optional
        pass

    def reset(self):
        self.position = self.start_hub
        pass

    def seed(self, seed=None): # optional
        pass

    def close(self): # optional
        pass