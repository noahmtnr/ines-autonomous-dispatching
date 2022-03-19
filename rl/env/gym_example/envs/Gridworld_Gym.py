import gym
from gym.utils import seeding
import numpy as np
import random

class Gridworld_v0(gym.Env): # define custom environment as subclass of gym.Env
    # GLOBAL VARIABLES:

    LF_MIN = 0
    RT_MAX = 9
    MAX_STEPS = 10
    REWARD_AWAY = -1
    REWARD_GOAL = MAX_STEPS

    metadata = {
    "render.modes": ["human"]
  }
    
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4) # 4 valid actions (1- up, 2-down, 3-left, 4-right)
        self.observation_space = gym.spaces.Box(low=0, high=9,
                                        shape=(10,10), dtype=np.int) #the grid with 10x10 dimensions
        self.final_hub = (4, 3)
        self.hubs = [(random.randrange(0, 9), random.randrange(0, 9)) for i in range(5)] 
        if self.final_hub in self.hubs:
            self.hubs.remove(self.final_hub)

        self.seed()
        self.reset()

    def step(self, action):
        nxtposition=(-1,-1)
        if self.done:
            # should never reach this point
            print("EPISODE DONE!!!")
        elif self.count == self.MAX_STEPS:
            self.done = True
        else:
            #assert self.action_space.contains(action)
            self.count += 1
            
            if action == 'u' or action == 1: #up
                nxtposition = (self.position[0] - 1, self.position[1])
            elif action == 'd' or action == 2: #down
                nxtposition = (self.position[0] + 1, self.position[1])
            elif action == 'l' or action == 3: #left
                nxtposition = (self.position[0], self.position[1] - 1)
            elif action == 'r' or action == 4: #right
                nxtposition = (self.position[0], self.position[1] + 1)
            else:
                pass

            if (nxtposition[0] >= 0) and (nxtposition[0] <= (self.RT_MAX)):
                if (nxtposition[1] >= 0) and (nxtposition[1] <= (self.RT_MAX)):
                    if nxtposition != (1, 1):
                        self.position = nxtposition
            
            
            if self.position == self.final_hub:
                # on goal now
                self.reward = self.REWARD_GOAL
                self.done = 1
            else: 
                # moving away from goal
                self.reward = self.REWARD_AWAY
    

    
        return [self.state, self.reward, self.done, "self.info"]

    def render(self, mode="human"): # method for visualization; optional
        s = "position: {:2d}, {:2d}  reward: {:2d}"
        print(s.format(self.position[0],self.position[1], self.reward))

    def reset(self):
        
        self.position = self.hubs[random.randint(0, 4)]
        self.count = 0
        self.state = self.position # needed ? 
        self.reward = 0
        self.done = False
        self.info = {}
        return self.state
        

    def seed(self, seed=None): # optional
        # self.np_random, seed = seeding.np_random(seed)
        # print(self.np_random)
        # return [seed]
        pass

    def close(self): # optional
        pass
