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

class CostAgent:

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
        count_shares = 0
        count_bookowns = 0
        count_wait = 0
        steps = 0
        while not done:
            # visualize current situation
            # env.render()

            # look in adjacency matrix for costs from the current position
            array = env.learn_graph.adjacency_matrix('remaining_distance')[env.position].astype(int)
            min = np.amin(array)
            # array = np.where(array==min,sys.maxsize,array)
            
            # get minimal value in array
            #while(action==env.position):
            min_value = np.amin(array)
            print(min_value)
            # if multiple entries have the same value
            all_indices = np.where(array==min)
            print(f"Alle: {all_indices[0]}")
            action = np.random.choice(all_indices[0])
            print(action)
            # select random of all_indices
                #while(action==env.position):
                #    action = np.random.choice(all_indices[0])

            # select action and show it
            #action = env.action_space[dest_hub]
            print(f"Our destination hub is: {action}")
            state, reward, done, info = env.step(action)

            route.append(action)
            route_timestamps.append(info.get('timestamp'))
            sum_travel_time +=timedelta(seconds=info.get('step_travel_time'))
            delivey_time = datetime.strptime(env_config["delivery_timestamp"], '%Y-%m-%d %H:%M:%S')
            time_until_deadline= delivey_time-sum_travel_time
            sum_distance += info.get('distance')/1000
            number_hubs=info.get('count_hubs')
            dist_shares = info.get("dist_covered_shares")
            dist_bookowns = info.get("dist_covered_bookown")
            # add reward
            sum_reward+=reward
            action_choice = info.get("action")

            if action_choice == "Share":
                count_shares += 1
            elif action_choice == "Book":
                count_bookowns += 1
            elif action_choice == "Wait":
                count_wait += 1
            steps += 1
            
            if action == env_config['delivery_hub_index']:
                done = True
                
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

        if count_bookowns == 0:
            ratio = 0
        else:
            ratio = float(count_shares/count_bookowns)
        reward_list={"pickup_hub":env_config['pickup_hub_index'],"delivery_hub":env_config['delivery_hub_index'],"reward":sum_reward, "hubs":number_hubs, "route":route, "time":sum_travel_time, "dist":sum_distance, "time_until_deadline":time_until_deadline, "timestamps":route_timestamps, "count_bookowns": count_bookowns, "steps": steps, "ratio_share_to_own": ratio, "dist_covered_shares": dist_shares, "dist_covered_bookown": dist_bookowns}
        return reward_list
