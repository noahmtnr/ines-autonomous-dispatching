# imports
import sys
sys.path.insert(0,"")
from ManhattanGraph import ManhattanGraph
from gym_graphenv.envs.GraphworldManhattan import GraphEnv
import numpy as np
import pandas as pd
import json
import os
import shutil
import gym
import pickle
from datetime import datetime, timedelta
import random
import ray
import warnings
warnings.filterwarnings('ignore')
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG


# class for DQN Agent
class DQNAgent:
    def __init__(self):
        sys.path.insert(0,"")
        #Set trainer configuration
        self.trainer_config = DEFAULT_CONFIG.copy()
        self.trainer_config['num_workers'] = 1
        self.trainer_config["train_batch_size"] = 400
        self.trainer_config["gamma"] = 0.95
        self.trainer_config["n_step"] = 10
        self.trainer_config["framework"] = "torch"
        #trainer_config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))

    def run_one_episode (self,env,reward_list,env_config):
        
        # Initialize trainer
        trainer = DQNTrainer(self.trainer_config,env)
        state = env.reset()
        print("reset done")

        # get information
        route=[]
        route_timestamps=[]
        sum_reward = 0
        sum_travel_time = timedelta(seconds=0)
        print(sum_travel_time)
        sum_distance = 0
        results = []
        for i in range(30):
            # get action and result
            action = trainer.compute_action(state)
            state, reward, done, info = env.step(action)
            results.append([state,reward,done,info])
            print("TEST", i)

            # get data from action
            route.append(info.get('hub_index'))
            route_timestamps.append(info.get('timestamp'))

            sum_reward += reward
            sum_travel_time +=timedelta(seconds=info.get('step_travel_time'))
            delivey_time = datetime.strptime(env_config["delivery_timestamp"], '%Y-%m-%d %H:%M:%S')
            time_until_deadline= delivey_time-sum_travel_time
            sum_distance += info.get('distance')/1000
            number_hubs=info.get('count_hubs')
            
            # check if finished
            if done==1:
                print("DELIVERY DONE! sum_reward: ",sum_reward)
                print("DELIVERY DONE! Travel Time: ",sum_travel_time)
                print("DELIVERY DONE! Distance: ",sum_distance)
                print("DELIVERY DONE! Hubs: ",number_hubs)
                print("DELIVERY DONE! Route: ",route)
                print("DELIVERY DONE! until deadline: ",time_until_deadline)
                break

            # print("sum_reward: ",sum_reward)
            # print("sum_reward: ",sum_reward, " time: ",env.time, "deadline time: ", env.deadline, "pickup time: ", env.pickup_time)
        reward_list={"pickup_hub":env_config['pickup_hub_index'],"delivery_hub":env_config['delivery_hub_index'],"reward":sum_reward, "hubs":number_hubs, "route":route, "time":str(sum_travel_time), "dist":sum_distance, "time_until_deadline":time_until_deadline, "timestamps":route_timestamps}

        return reward_list