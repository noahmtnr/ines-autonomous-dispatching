from datetime import datetime, timedelta
import gym
from gym.utils import seeding
import numpy as np
import random
import pandas as pd
from gym import spaces
import itertools

class Gridworld_v1(gym.Env): # define custom environment as subclass of gym.Env
    # GLOBAL VARIABLES:

    POS_MIN = 0
    POS_MAX = 100
    MAX_STEPS = 200
    REWARD_AWAY = -1
    REWARD_GOAL = MAX_STEPS

    metadata = {
     "render.modes": ["human"]
   }
    
    def __init__(self,env_config = None):
        env_config = env_config or {}
        num_states = 100
        self.action_space = gym.spaces.Discrete(4) # 4 valid actions (1- up, 2-down, 3-left, 4-right)
        self.observation_space = gym.spaces.Discrete(num_states)
        self.final_hub = 43
        #self.hubs = [(random.randrange(0, 9), random.randrange(0, 9)) for i in range(5)] 

        #if self.final_hub in self.hubs:
        #    self.hubs.remove(self.final_hub)

        self.seed()
        self.reset()
        random_number = np.random.randint(31536000) # random seconds number in order to generate a random date
        self.time=datetime(2021,1,1,12,0,0)+timedelta(seconds=random_number)
   
    
    def step(self, action):
        assert self.action_space.contains(action)
        nxtposition=0
        # if done:
        # #     should never reach this point
        #      print("EPISODE DONE!!!")
        # elif self.count == self.MAX_STEPS:
        #         done = True
        # else:
            #     assert self.action_space.contains(action)
        self.count += 1

        if action == 'u' or action == 1: #up
                nxtposition = self.state - 10
        elif action == 'd' or action == 2: #down
                nxtposition = self.state + 1
        elif action == 'l' or action == 3: #left
                nxtposition = self.state-1
        else: #right
                nxtposition = self.state+ 1
       
        done = self.count >= self.MAX_STEPS
            
        if (nxtposition> 0) and (nxtposition <= (self.POS_MAX)):
                    self.state = nxtposition
                    self.time=self.time+timedelta(minutes=1)
        
        else:
                reward = self.REWARD_AWAY
        if self.state == self.final_hub:
            # on goal now
            reward = self.REWARD_GOAL
            done = True
        else: 
            # moving away from goal
            reward = self.REWARD_AWAY
        
        return self.state, reward, done, self.info
      
    def availableActions(self):
        position=str(self.position)
        start_timestamp=str(self.time)
        possible_routes=[]
        time_window=5
        end_timestamp = str(start_timestamp + timedelta(minutes=time_window))
        grid=pd.read_csv('rl\env\data_gridworld_timestamps.csv')
        paths=grid['Path Timestamp']
        
        for index in range(len(paths)):
            route_dict = eval(grid['Path Timestamp'][index])

            for tupel_position in route_dict:
                position_timestamp= route_dict[tupel_position]

                if tupel_position == position and start_timestamp <= position_timestamp \
                and end_timestamp >= route_dict[tupel_position] and tupel_position != grid['Dropoff Coordinates'][index]:
                  
                    route_dict_keys = list(route_dict)
                    current_node_index = route_dict_keys.index(position)
                    current_to_final_route_node=dict(itertools.islice(route_dict.items(),current_node_index,None))
                    #print("sliced",current_to_final_route_node)
                    possible_routes.append(current_to_final_route_node)

        return possible_routes



    def render(self, mode="human"): # method for visualization; optional
        #s = "position: {:2d}  reward: {:2d} "
        # print(s.format(self.position, self.reward))
        
        # print("Position: "+ str(self.position)+ " Reward: "+ str(self.reward)+ " Time: "+str(self.time))
        pass


    def reset(self):
        self.count = 0
        #self.state = random.randint(1, 100)
        self.state = 42
        self.info = {}
        return self.state
        

    def seed(self, seed=None): # optional
        # self.np_random, seed = seeding.np_random(seed)
        # print(self.np_random)
        # return [seed]
        pass

    def close(self): # optional
        pass