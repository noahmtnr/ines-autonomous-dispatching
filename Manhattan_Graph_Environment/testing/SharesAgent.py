import sys
sys.path.insert(0,"")
from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv
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

class SharesAgent:

    def run_one_episode (env,reward_list,env_config):
        env.reset()
        print("reset done")
        counter = 0
        sum_reward = 0
        sum_travel_time = timedelta(seconds=0)
        sum_distance = 0
        route = [env_config["pickup_hub_index"]]
        route_timestamps = [datetime.strptime(env_config["pickup_timestamp"], '%Y-%m-%d %H:%M:%S')]
        done = False
        current_hub = env_config["pickup_hub_index"]
        while not done:
            # visualize current situation
            # env.render()

            # select most useful shared ride, otherwise wait
            best_gain = 0
            best_hub = 0
            for hub in range(env.action_space.n):
                # check distance gained
                if (env.state["remaining_distance"][hub] > 0) and (env.state["remaining_distance"][hub] > best_gain):
                    best_hub = hub
                    best_gain = env.state["remaining_distance"][hub]

            if best_gain <= 0:
                action = current_hub
            else:
                action = best_hub

            print(f"Our destination hub is: {action}")
            state, reward, done, info = env.step(action)

            route.append(action)
            current_hub = action
            route_timestamps.append(info.get('timestamp'))
            sum_travel_time +=timedelta(seconds=info.get('step_travel_time'))
            delivey_time = datetime.strptime(env_config["delivery_timestamp"], '%Y-%m-%d %H:%M:%S')
            time_until_deadline= delivey_time-sum_travel_time
            sum_distance += info.get('distance')/1000
            number_hubs=info.get('count_hubs')
            # add reward
            sum_reward+=reward
            
            if done:
                print("DELIVERY DONE! sum_reward: ",sum_reward)
                print("DELIVERY DONE! Route: ",route)
                print("DELIVERY DONE! Travel Time: ",sum_travel_time)
                print("DELIVERY DONE! Distance: ",sum_distance)
                print("DELIVERY DONE! Hubs: ",number_hubs)
                print("DELIVERY DONE! unitl deadline: ",time_until_deadline)
                # if action!=env_config["delivery_hub_index"]:
                #     raise Exception("DID NOT ARRIVE IN FINAL HUB")
                break

        reward_list={"pickup_hub":env_config['pickup_hub_index'],"delivery_hub":env_config['delivery_hub_index'],"reward":sum_reward, "hubs":number_hubs, "route":route, "time":str(sum_travel_time), "dist":sum_distance, "time_until_deadline":time_until_deadline, "timestamps":route_timestamps}
        return reward_list
