# imports
import sys

sys.path.insert(0, "")
from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv, CustomCallbacks
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


# class for Rainbow Agent
class RainbowAgent:
    def __init__(self, ):
        sys.path.insert(0, "")
        # Set trainer configuration
        self.trainer_config = DEFAULT_CONFIG.copy()
        self.trainer_config['num_workers'] = 3
        self.trainer_config["train_batch_size"] = 400
        self.trainer_config["gamma"] = 0.99
        self.trainer_config["callbacks"] = CustomCallbacks
        self.trainer_config["n_step"] = 3  # [between 1 and 10]  //was 5 and 7
        self.trainer_config["noisy"] = True
        self.trainer_config["num_atoms"] = 70  # [more than 1] //was 51,20
        self.trainer_config["v_min"] = -210000
        self.trainer_config["v_max"] = 210000  # (set v_min and v_max according to your expected range of returns).

        # here from trainRainbow die config
        # self.trainer_config["train_batch_size"] = 400
        # self.trainer_config["framework"] = "torch"

    def run_one_episode(self, reward_list, env_config):
        # Initialize trainer
        rainbow_trainer = DQNTrainer(self.trainer_config, env=GraphEnv)
        mode = {"env-name": "graphworld-v0",
                "env": GraphEnv,
                "iterations": 10,
                }
        # checkpoint anpassen
        file_name = "tmp/rainbow/graphworld\checkpoint_000010\checkpoint-10"

        # Restore the Trainer
        rainbow_trainer.restore(file_name)
        env = gym.make(mode["env-name"])
        state = env.reset()
        print("reset done")

        # get information
        route = [env_config["pickup_hub_index"]]
        route_timestamps = [datetime.strptime(env_config["pickup_timestamp"], '%Y-%m-%d %H:%M:%S')]
        sum_reward = 0
        sum_travel_time = timedelta(seconds=0)
        print(sum_travel_time)
        sum_distance = 0
        results = []

        for i in range(30):
            action = rainbow_trainer.compute_action(state)
            state, reward, done, info = env.step(action)
            sum_reward += reward
            # env.render()
            if done == True:
                print("cumulative reward", sum_reward)
                state = env.reset()
                sum_reward = 0

            # get data from action
            route.append(action)
            route_timestamps.append(info.get('timestamp'))

            sum_travel_time += timedelta(seconds=info.get('step_travel_time'))
            delivey_time = datetime.strptime(env_config["delivery_timestamp"], '%Y-%m-%d %H:%M:%S')
            time_until_deadline = delivey_time - sum_travel_time
            sum_distance += info.get('distance') / 1000
            number_hubs = info.get('count_hubs')
            # add reward
            sum_reward += reward

            # check if finished
            if done == 1:
                print("DELIVERY DONE! sum_reward: ", sum_reward)
                print("DELIVERY DONE! Route: ", route)
                print("DELIVERY DONE! Travel Time: ", sum_travel_time)
                print("DELIVERY DONE! Distance: ", sum_distance)
                print("DELIVERY DONE! Hubs: ", number_hubs)
                print("DELIVERY DONE! unitl deadline: ", time_until_deadline)
                break

            # print("sum_reward: ",sum_reward)
            # print("sum_reward: ",sum_reward, " time: ",env.time, "deadline time: ", env.deadline, "pickup time: ", env.pickup_time)
        reward_list = {"pickup_hub": env_config['pickup_hub_index'], "delivery_hub": env_config['delivery_hub_index'],
                       "reward": sum_reward, "hubs": number_hubs, "route": route, "time": str(sum_travel_time),
                       "dist": sum_distance, "time_until_deadline": time_until_deadline, "timestamps": route_timestamps}

        return reward_list