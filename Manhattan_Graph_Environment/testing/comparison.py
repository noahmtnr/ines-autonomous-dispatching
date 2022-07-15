# imports
import sys
import pandas as pd
import numpy as np
from BenchmarkWrapper import BenchmarkWrapper
import operator
sys.path.insert(0,"")

# define class
class Comparer:
    def __init__(self,num_agents,*agents_arg):
        self.num_agents = num_agents
        self.agent_names = []
        self.agents_arg = []

        print(num_agents)
        print(agents_arg)
        i = 0
        while i < (self.num_agents*2):
            if i < self.num_agents:
                self.agent_names.append(agents_arg[i])
            else:
                self.agents_arg.append(agents_arg[i])
            i += 1
        print("Names", self.agent_names)
        print("ARgs",self.agents_arg)
        Comparer.set_agents(self)

    def set_agents(self):
        self.agents = {}
        # go through all agents
        for i in range(self.num_agents):
            # append each agent to the list, use name as key 
            print("Ausgaben Start")
            print(self.agent_names[i])
            print(self.agents_arg[i])
            self.agents[self.agent_names[i]] = self.agents_arg[i]
        # print("Agentlist",self.agents)

    def establish_compare_onetrip(self):
        self.compare_dict = {}
        i = 1
        # iterate over dictionary 
        counter = 0
        for key in self.agents.keys():
            print("Ausgabe Zweite")
            print(key, " : ", self.agents[key])
            array = self.agents[key].read_orders()
            # get the only order
            for current in array:
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
            self.compare_dict[key]= [reward,route,dist,time,num_hubs]

            print(self.compare_dict)

            counter += 1
        
        Comparer.compare_onetrip(self)
    
    def compare_onetrip(self):
        # do ranking on each aspect
        # rank on reward (max)
        reward_sort = sorted(self.compare_dict.items(), key=lambda i: i[1][0], reverse=True)
        i = 1
        print("Ranking by Reward")
        for elem in reward_sort:
            print(f"Rank {i}: {elem[0]} with reward of {reward_sort[i-1][1][0]}")
            i += 1

        dist_sort = sorted(self.compare_dict.items(), key=lambda i: i[1][2])
        i = 1
        print("\n Ranking by Distance")
        for elem in dist_sort:
            print(f"Rank {i}: {elem[0]} with distance of {dist_sort[i-1][1][2]}")
            i += 1

        # rank on time (min)
        time_sort = sorted(self.compare_dict.items(), key=lambda i: i[1][3])
        i = 1
        print("\n Ranking by Travel Time")
        for elem in time_sort:
            print(f"Rank {i}: {elem[0]} with travel time of {time_sort[i-1][1][3]}")
            i += 1

        # rank on ratio of travelled and remaining time
        # rank on hubs (min)
        hub_sort = sorted(self.compare_dict.items(), key=lambda i: i[1][4])
        i = 1
        print("\n Ranking by Number of Hubs")
        for elem in hub_sort:
            print(f"Rank {i}: {elem[0]} with {hub_sort[i-1][1][4]} hubs")
            i += 1
        
        # display the route each agent took
        # TODO - do the visualization of Aga & Denisa here
    
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
        
        Comparer.compare_onetrip(self)
    
    def Average(lst):
        return sum(lst) / len(lst)

    def main():
        set_agents(self.num_agents,self.agents_arg)


# test comparer
w1 = BenchmarkWrapper("random")
w2 = BenchmarkWrapper("cost")
w3 = BenchmarkWrapper("PPO")
w4 = BenchmarkWrapper("DQN")
w5 = BenchmarkWrapper("Rainbow")
# c = Comparer(1,w5.name,w5)
# c = Comparer(2,w3.name,w5.name,w3,w5)
# c = Comparer(3,w1.name,w2.name,w3.name,w1,w2,w3)
c = Comparer(5,w1.name,w2.name,w3.name,w4.name,w5.name,w1,w2,w3,w4,w5)
c.establish_compare_onetrip()
# c.compare_multiple_trips()



