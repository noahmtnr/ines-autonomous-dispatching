import sys
sys.path.insert(0,"")
from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattanBenchmark import GraphEnv
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
        sum_travel_time = timedelta(seconds=0)
        print(sum_travel_time)
        count_shares = 0
        count_bookowns = 0
        count_wait = 0
        steps = 0
        number_hubs = 0
        while (not done) and (steps<200):
            # visualize current situation
            # env.render()
            old_hub = current_hub
            # select most useful shared ride, otherwise wait
            best_gain = 0
            best_hub = 0
            for hub in range(env.action_space.n):
                # check distance gained and whether it is shared ride
                # print(env.state["distinction"])
                # print(env.state["remaining_distance"])
                print("Distinction: ", env.state["distinction"])
                if ((env.state["remaining_distance"][hub]) > 0) and ((env.state["remaining_distance"][hub]) > best_gain) and ((env.state["distinction"][hub]) == 1):
                    print("Distinction for hub ", hub, " is ", env.state["distinction"][hub])
                    best_hub = hub
                    best_gain = env.state["remaining_distance"][hub]

            if best_gain <= 0:
                action = current_hub # wait
            else:
                action = best_hub # take the shared ride

            print(f"Our destination hub is: {action}")
            state, reward, done, info = env.step(action)

            route.append(action)
            current_hub = action
            route_timestamps.append(info.get('timestamp'))
            sum_travel_time +=timedelta(seconds=info.get('step_travel_time'))
            delivey_time = datetime.strptime(env_config["delivery_timestamp"], '%Y-%m-%d %H:%M:%S')
            time_until_deadline= timedelta(hours=24)-sum_travel_time
            sum_distance += info.get('distance')/1000
            if old_hub!=current_hub:
                number_hubs+=1
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
            
            print("Step: ", steps)
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

        # TODO: for comparison
        print("Ratio Shares to All Steps: ", float(count_shares/steps))

        if steps>=50:
            print("Not Delivered")
        if count_bookowns == 0:
            ratio = 0
        else:
            ratio = float(count_shares/count_bookowns)
        reward_list={"pickup_hub":env_config['pickup_hub_index'],"delivery_hub":env_config['delivery_hub_index'],"reward":sum_reward, "hubs":number_hubs, "route":route, "time":str(sum_travel_time), "dist":sum_distance, "time_until_deadline":time_until_deadline, "timestamps":route_timestamps, "count_bookowns": count_bookowns, "steps": steps, "ratio_share_to_own": ratio,"dist_covered_shares": dist_shares, "dist_covered_bookown": dist_bookowns}
        return reward_list
