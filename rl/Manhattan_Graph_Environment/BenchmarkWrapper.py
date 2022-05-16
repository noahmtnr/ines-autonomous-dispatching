import sys
from unittest import result
sys.path.insert(0,"")
from RandomAgent import RandomAgent
from CostAgent import CostAgent
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

class BenchmarkWrapper:

     def __init__(self,agent_name=None):
        if(agent_name != None):
            self.name=agent_name
        else:
            self.name =sys.argv[2]

     def file_read(self):
        if len(sys.argv) > 1:
            first_arg = sys.argv[1]
            file_path='data/'+first_arg
            orders = pd.read_csv(file_path, nrows=2)
        else:
            orders = pd.read_csv('data/random_orders.csv', nrows=2)
        return orders

     def read_orders(self):
            reward_list=[]
            orders=self.file_read()
            for index, row in orders.iterrows():
                order={"pickup_node":row['pickup_node_id'],"delivery_node":row['delivery_node_id'],"pickup_timestamp":row['pickup_timestamp'] , "delivery_timestamp":row['delivery_timestamp']  }
                reward_list.append(self.proceed_order(order))
            return reward_list
   
     def proceed_order(self, order):
        manhattan_graph = ManhattanGraph(filename='simple', num_hubs=70)
        pick_up_hub_index = ManhattanGraph.get_hub_index_by_node_index(manhattan_graph,order.get('pickup_node'))
        delivery_hub_index = ManhattanGraph.get_hub_index_by_node_index(manhattan_graph,order.get('delivery_node'))
                # print(pick_up_hub_index,delivery_hub_index)
        env_config = {'pickup_hub_index':pick_up_hub_index,
                    'delivery_hub_index':delivery_hub_index,
                    'pickup_timestamp': order.get('pickup_timestamp'),
                    'delivery_timestamp': order.get('delivery_timestamp')
                }
        reward_list=[]
        with open('env_config.pkl', 'wb') as f:
            pickle.dump(env_config, f)
        env=GraphEnv()
        for i in range(1): 
            if self.name == "random":
                print("random")
                reward_list=RandomAgent.run_one_episode(env,reward_list,env_config)
            elif self.name == "cost":
                print("cost")
                reward_list=CostAgent.run_one_episode(env,reward_list,env_config)
        return reward_list


def main():
    # benchmark = BenchmarkWrapper("random")
    # results = benchmark.read_orders()
    # print("Random",results)
    benchmark2 = BenchmarkWrapper("cost")
    results = benchmark2.read_orders()
    print("Cost",results)

main()