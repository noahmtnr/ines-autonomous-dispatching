from datetime import datetime, timedelta
import gym
from gym.utils import seeding
import numpy as np
import random
import pandas as pd

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
        random_number = np.random.randint(31536000) # random seconds number in order to generate a random date
        self.time=datetime(2021,1,1,12,0,0)+timedelta(seconds=random_number)
   

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
                        self.time=self.time+timedelta(minutes=1)
            
            
            if self.position == self.final_hub:
                # on goal now
                self.reward = self.REWARD_GOAL
                self.done = 1
            else: 
                # moving away from goal
                self.reward = self.REWARD_AWAY
    

    
        return [self.state, self.reward, self.done, "self.info"]

    def availableActions(self):
        position=str(self.position)
        start_timestamp=str(self.time)
        list=[]
        time_window=5
        end_timestamp = str(start_timestamp + timedelta(minutes=time_window))
        grid=pd.read_csv('rl\env\data_gridworld_timestamps.csv')
        #grid=grid.head(10)
        paths=grid['Path Timestamp']
        for index in range(len(paths)):
            #print(grid['Path Timestamp'][index])
            dict = eval(grid['Path Timestamp'][index])
            for tupel_position in dict:
                #print(tupel_position ,dict[tupel_position])
                position_timestamp=dict[tupel_position]
                if str(tupel_position) == position and start_timestamp <= position_timestamp \
                and end_timestamp >= dict[tupel_position] and str(tupel_position) != grid['Dropoff Coordinates'][index]:
                   list.append([position_timestamp,grid['Dropoff Coordinates'][index]])
                   # TODO slice and retrun dictionary in order to get only the route to the final node from the current node
        return list



    def render(self, mode="human"): # method for visualization; optional
        #s = "position: {:2d}  reward: {:2d} "
        # print(s.format(self.position, self.reward))
        
        # print("Position: "+ str(self.position)+ " Reward: "+ str(self.reward)+ " Time: "+str(self.time))
        pass


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
