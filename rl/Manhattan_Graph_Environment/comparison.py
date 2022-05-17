# imports
import pandas as pd
import numpy as np
from .. import BenchmarkWrapper

# define class
class Comparer:
    def __init__(num_agents,*agents_arg):
        self.num_agents = num_agents
        self.agents_arg = agents_arg
    def set_agents(self):
        self.agents = {}
        # go through all agents
        for i in range(self.num_agents):
            # append each agent to the list, use name as key 
            self.agents.update({self.agents_arg.name,self.agents_arg(i)})

    def establish_compare_onetrip(self):
        self.compare_dict = {}
        i = 1
        # iterate over dictionary 
        for key in self.agents:
            array = self.agents[key].read_orders()

            # get the only order
            for i in len(array):
                current = array[i]
                # get individual aspects out of dictionary
                # reward
                reward = current.get("reward")
                # route
                route = current.get("route")
                # distance
                dist = current.get("dist")
                # remaining time
                time = current.get("time")
                # travelled hubs
                num_hubs = current.get("hubs")

                # for one order
            self.compare_dict.update({key,[reward,route,dist,time,num_hubs]})
        
        compare_onetrip()
    
    def compare_onetrip(self):
        # do ranking on each aspect
        # rank on reward (max)
        reward_sort = sorted(self.compare_dict.items(),key=lambda item: item[2]))
        i = 0
        for key in reward_sort:
            print(f"Rank {i}: {key} with reward of {reward_sort[key]}")
            i += 1
        # rank on dist (min)
        dist_sort = sorted(self.compare_dict.items(),key=lambda item: item[6]))
        i = 0
        for key in dist_sort:
            print(f"Rank {i}: {key} with travelled distance of {dist_sort[key]}")
            i += 1
        # rank on time (min)
        time_sort = sorted(self.compare_dict.items(),key=lambda item: item[5]))
        i = 0
        for key in time_sort:
            print(f"Rank {i}: {key} with travel time of {time_sort[key]}")
            i += 1
        # rank on ratio of travelled and remaining time
        # rank on hubs (min)
        hub_sort = sorted(self.compare_dict.items(),key=lambda item: item[3]))
        i = 0
        for key in hub_sort:
            print(f"Rank {i}: {key} with {hub_sort[key]} hubs")
            i += 1
        
        # display the route each agent took
        # TODO
    
    def compare_multipletrips(self):
        self.compare_dict = {}
        i = 1

        # iterate over dictionary 
        for key in self.agents:
            array = self.agents[key].read_orders()

            # get all orders
            reward_array = []
            # route_array = []
            dist_array = []
            time_array = []
            hubs_array = []

            print(f"Compare the agents for {len(array)} trips:")
            for i in len(array):
                current = array[i]
                # get individual aspects out of dictionary
                # reward
                reward = current.get("reward")
                reward_array.append(reward)
                # route
                # route = current.get("route")
                # route_array.append(route)
                # distance
                dist = current.get("dist")
                dist_array.append(dist)
                # remaining time
                time = current.get("time")
                time_array.append(time)
                # travelled hubs
                num_hubs = current.get("hubs")
                hubs_array.append(num_hubs)

            reward_mean = Average(reward_array)
            dist_mean = Average(dist_array)
            time_mean = Average(time_array)
            num_hubs_mean = Average(hubs_array)
            self.compare_dict.update({key,[reward_mean,dist_mean,time_mean,num_hubs_mean]})
        
        compare_onetrip()
    
    def Average(lst):
        return sum(lst) / len(lst)

    def main():
        set_agents(self.num_agents,self.agents_arg)


# test comparer
w1 = BenchmarkWrapper("random")
w2 = BenchmarkWrapper("cost")
c = Comparer(2,w1,w2)
c.establish_compare_onetrip()
c.compare_multiple_trips()

