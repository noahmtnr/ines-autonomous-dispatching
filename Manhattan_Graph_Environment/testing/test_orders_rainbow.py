# imports
import csv
import sys
sys.path.insert(0, "")
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
from config.definitions import ROOT_DIR
from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv, CustomCallbacks
sys.path.append(os.path.join(ROOT_DIR, "Manhattan_Graph_Environment", "gym_graphenv"))


# class for Rainbow Agent
class TestOrders:
    def __init__(self, ):
        sys.path.insert(0, "")
        # Set trainer configuration
        self.trainer_config = DEFAULT_CONFIG.copy()
        self.trainer_config['num_workers'] = 1
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

    def test_order(self):
        
        # Initialize trainer
        rainbow_trainer = DQNTrainer(self.trainer_config, GraphEnv)
        # checkpoint anpassen
        file_name = os.path.join(ROOT_DIR, 'tmp', 'rainbow', 'graphworld','checkpoint_000025','checkpoint-25')

        # Restore the Trainer
        rainbow_trainer.restore(file_name)
        # env = gym.make('graphworld-v0')
        env = GraphEnv(use_config=True)
        obs = env.reset()
        print(env.position)
        print("reset done")

        # get information
        list_nodes=[]
        list_hubs=[env.position]
        list_actions=["start"]
        # route = [env_config["pickup_hub_index"]]
        # route_timestamps = [datetime.strptime(env_config["pickup_timestamp"], '%Y-%m-%d %H:%M:%S')]
        sum_reward = 0
        # sum_travel_time = timedelta(seconds=0)
        # print(sum_travel_time)
        # sum_distance = 0
        # results = []
        done = False
        while not done:
            action = rainbow_trainer.compute_action(obs)
            print("action test", action)
            state, reward, done, info = env.step(action)
            sum_reward += reward
            
            if(info["route"][-1] != env.final_hub and info["action"] != "Wait"):
                print(list_nodes)
                list_nodes.extend(info["route"][0:-1])
                print(list_nodes)
            list_hubs.append(info["hub_index"])
            list_actions.append(info["action"])
                

            # get data from action
            # route.append(action)
            # route_timestamps.append(info.get('timestamp'))

            # sum_travel_time += timedelta(seconds=info.get('step_travel_time'))
            # delivey_time = datetime.strptime(env_config["delivery_timestamp"], '%Y-%m-%d %H:%M:%S')
            # time_until_deadline = delivey_time - sum_travel_time
            # sum_distance += info.get('distance') / 1000
            # number_hubs = info.get('count_hubs')
            # # add reward
            # sum_reward += reward

            # check if finished
            if done == True:
                print("DELIVERY DONE! sum_reward: ", sum_reward)
                print("DELIVERY DONE! Route: ", list_hubs)
                print("DELIVERY DONE! Actions: ", list_actions)
                print("DELIVERY DONE! Nodes: ", list_nodes)
                break

            # print("sum_reward: ",sum_reward)
            # print("sum_reward: ",sum_reward, " time: ",env.time, "deadline time: ", env.deadline, "pickup time: ", env.pickup_time)
        # reward_list = {"pickup_hub": env_config['pickup_hub_index'], "delivery_hub": env_config['delivery_hub_index'],
        #                "reward": sum_reward, "hubs": number_hubs, "route": route, "time": str(sum_travel_time),
        #                "dist": sum_distance, "time_until_deadline": time_until_deadline, "timestamps": route_timestamps}
        return list_hubs, list_actions, list_nodes


def create_test_order():
    pickup_hub=111
    delivery_hub=67
    pickup_timestamp="2016-01-02 09:05:00"
    delivery_timestamp="2016-01-02 21:05:00"
    env_config = {'pickup_hub_index': pickup_hub,
                      'delivery_hub_index': delivery_hub,
                      'pickup_timestamp':pickup_timestamp,
                      'delivery_timestamp': delivery_timestamp,
                      }
    with open('env_config.pkl', 'wb') as f:
        pickle.dump(env_config, f)

    test = TestOrders()
    list_hubs, list_actions, list_nodes = test.test_order()
    write_in_file_orders(list_hubs,list_actions,list_nodes,pickup_hub,delivery_hub,pickup_timestamp,delivery_timestamp)


def write_in_file_orders(hubs,actions,nodes,pickup_hub,delivery_hub,pickup_timestamp,delivery_timestamp):
    
    filepath = os.path.join(ROOT_DIR, 'data', 'others', 'test_orders_dashboard.csv')
    mycsv = csv.reader(open(filepath))
    
    for row in mycsv:
        id = row[0]
    try:
        idInt = int(id)
    except:
        idInt = 0
    idInt +=1
    row_list = [[idInt,pickup_hub,delivery_hub,pickup_timestamp,delivery_timestamp, hubs, actions, nodes],]

    with open(filepath, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)
   

create_test_order()