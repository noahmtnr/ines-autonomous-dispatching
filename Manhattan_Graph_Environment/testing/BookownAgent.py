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


class BookownAgent:

    def run_one_episode (env,reward_list,env_config):
        route = [env_config["pickup_hub_index"]]
        route_timestamps=[]
        env.reset()
        print("reset done")
        print("Delivery Hub",env_config["delivery_hub_index"])
        sum_reward = 0
        sum_travel_time = timedelta(seconds=0)
        print(sum_travel_time)
        sum_distance = 0
        done = False
        while not done:
            action = env_config["delivery_hub_index"]
            # action = final hub
            state, reward, done, info = env.step(action)
            route.append(action)
            print("Timestamps",info.get('timestamp') )
            route_timestamps.append(info.get('timestamp'))
            sum_reward += reward
            sum_travel_time +=timedelta(seconds=info.get('step_travel_time'))
            delivey_time = datetime.strptime(env_config["delivery_timestamp"], '%Y-%m-%d %H:%M:%S')
            time_until_deadline= delivey_time-sum_travel_time
            sum_distance += info.get('distance')/1000
            number_hubs=info.get('count_hubs')
            #env.render()
            if done:
                print("DELIVERY DONE! sum_reward: ",sum_reward)
                print("DELIVERY DONE! Route: ",route)
                print("DELIVERY DONE! Travel Time: ",sum_travel_time)
                print("DELIVERY DONE! Distance: ",sum_distance)
                print("DELIVERY DONE! Hubs: ",number_hubs)
                print("DELIVERY DONE! unitl deadline: ",time_until_deadline)
                break

            print("sum_reward: ",sum_reward)
            # print("sum_reward: ",sum_reward, " time: ",env.time, "deadline time: ", env.deadline, "pickup time: ", env.pickup_time)
        reward_list={"pickup_hub":env_config['pickup_hub_index'],"delivery_hub":env_config['delivery_hub_index'],"reward":sum_reward, "hubs":number_hubs, "route":route, "time":str(sum_travel_time), "dist":sum_distance, "time_until_deadline":time_until_deadline, "timestamps":route_timestamps}
        print(reward_list)
        return reward_list

    