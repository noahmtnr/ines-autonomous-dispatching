# imports
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from BenchmarkWrapper import BenchmarkWrapper
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv
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


    def Average(lst):
        return sum(lst) / len(lst)

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
                # number of steps
                num_steps = current.get("steps")
                # booked own
                num_booked_own = current.get("count_bookowns")
                # share to own ratio
                ratio = current.get("ratio_share_to_own")

                # distance covered with shares
                dist_shares = current.get("dist_covered_shares")
                # print(dist_shares)
                # print(type(dist_shares))
                # distance covered with bookowns
                dist_bookowns = current.get("dist_covered_bookown")
                # compute share of distance covered with shares and bookowns
                ratio_dist_shares = float(dist_shares/dist)
                ratio_dist_bookowns = float(dist_bookowns/dist)

                # for one order
            self.compare_dict[key]= [reward,route,dist,time,num_hubs,num_booked_own,ratio,num_steps,ratio_dist_shares,ratio_dist_bookowns]

            # print(self.compare_dict)

            counter += 1
        
        Comparer.compare_onetrip(self)
    
    def compare_onetrip(self):

        # show overview of all compared agents
        print("\n Overview of Compared Agents")
        for elem in self.compare_dict.items():
            print(elem[0], " Agent: ", elem[1][7], " steps / ", elem[1][4], " hubs / ", elem[1][2], " distance / ", elem[1][3], " time / ", elem[1][0], " reward / ", elem[1][1], " route")

        # do ranking on each aspect
        print("\n Rankings of Compared Agents")
        # number of steps to compare

        # rank on hubs (min)
        hub_sort = sorted(self.compare_dict.items(), key=lambda i: i[1][4])
        i = 1
        print("\n Ranking by Number of Hubs")
        for elem in hub_sort:
            print(f"Rank {i}: {elem[0]} with {hub_sort[i - 1][1][4]} hubs")
            i += 1

        # rank on distance (min)
        dist_sort = sorted(self.compare_dict.items(), key=lambda i: i[1][2])
        i = 1
        print("\n Ranking by Distance")
        for elem in dist_sort:
            print(f"Rank {i}: {elem[0]} with distance of {dist_sort[i - 1][1][2]}")
            i += 1

        # whether has booked any own rides and how many (min)
        booked_own_sort = sorted(self.compare_dict.items(),key=lambda i: i[1][5])
        i = 1
        print("\n Ranking by Number of booked owns")
        for elem in booked_own_sort:
            print(f"Rank {i}: {elem[0]} with number of bookowns of {booked_own_sort[i - 1][1][5]}")
            i += 1

        # share to own ratio (max)
        ratio_sort = sorted(self.compare_dict.items(), key=lambda i: i[1][6], reverse=True)
        i = 1
        print("\n Ranking by Ratio of Shares to Booked Owns")
        for elem in ratio_sort:
            print(f"Rank {i}: {elem[0]} with ratio of {ratio_sort[i - 1][1][6]} shares to book-owns")
            i += 1

        # rank on reward (max)
        reward_sort = sorted(self.compare_dict.items(), key=lambda i: i[1][0], reverse=True)
        i = 1
        print("\n Ranking by Reward")
        for elem in reward_sort:
            print(f"Rank {i}: {elem[0]} with reward of {reward_sort[i-1][1][0]}")
            i += 1

        # rank on time (min)
        time_sort = sorted(self.compare_dict.items(), key=lambda i: i[1][3])
        i = 1
        print("\n Ranking by Travel Time")
        for elem in time_sort:
            print(f"Rank {i}: {elem[0]} with travel time of {time_sort[i-1][1][3]}")
            i += 1
        
        # display the route each agent took
        # TODO - do the visualization of Aga & Denisa here
        # Route bekommen (Liste von Hub IDs): self.compare_dict.items()[1
        # plotten mit Funktionsaufruf der Visualisierung
        """
        allRoutes = {}
        for elem in self.compare_dict.items():
            # route for current agent in the loop
            currentRoute =elem[1][1]
            currentAgentName = elem[0]
            allRoutes[currentAgentName] = currentRoute
        # visualize by calling method xy
        """

        # rank on ratio of distance reduced with shares (max)
        dist_reduce_sort = sorted(self.compare_dict.items(), key=lambda i: i[1][8], reverse=True)
        i = 1
        print("\n Ranking by Ratio Distance Reduced with Shares to Whole Distance")
        for elem in dist_reduce_sort:
            print(f"Rank {i}: {elem[0]} with ratio of {dist_reduce_sort[i-1][1][8]}")
            i += 1

    # compares multiple trips (by taking mean of results over the trips) for the agents
    def compare_multipletrips(self):
        self.compare_dict = {}
        i = 1

        # iterate over dictionary 
        for key in self.agents:
            print("Ausgabe Zweite")
            print(key, " : ", self.agents[key])
            array = self.agents[key].read_orders()

            # get results of all orders in respective arrays for each agent
            reward_array = []
            # route_array = []
            dist_array = []
            time_array = []
            hubs_array = []
            steps_array = []
            num_books_array = []
            ratio_array = []
            dist_shares_array = []
            route_array = []
            dist_bookowns_array = []

            print(f"Compare the agents for {len(array)} trips.")
            # get the only order
            for current in array:
                # get individual aspects out of dictionary
                # reward
                reward = current.get("reward")
                reward_array.append(reward)
                # route
                route = current.get("route")
                route_array.append(route) # necessary?
                # distance
                dist = current.get("dist")
                dist_array.append(dist)
                # remaining time
                time = current.get("time")
                try:
                    date = datetime.strptime(time, '%H:%M:%S.%f')
                except ValueError:
                    date = datetime.strptime(time, '%H:%M:%S')
                time_array.append(timedelta(hours=date.hour, minutes=date.minute, seconds=date.second).total_seconds())
                # travelled hubs
                num_hubs = current.get("hubs")
                hubs_array.append(num_hubs)
                # number of steps
                num_steps = current.get("steps")
                steps_array.append(num_steps)
                # booked own
                num_booked_own = current.get("count_bookowns")
                num_books_array.append(num_booked_own)
                # share to own ratio
                ratio = current.get("ratio_share_to_own")
                ratio_array.append(ratio)

                # distance covered with shares
                dist_shares = current.get("dist_covered_shares")
                # distance covered with bookowns
                dist_bookowns = current.get("dist_covered_bookown")
                # compute share of distance covered with shares and bookowns
                ratio_dist_shares = float(dist_shares/dist)
                ratio_dist_bookowns = float(dist_bookowns/dist)
                dist_shares_array.append(ratio_dist_shares)
                dist_bookowns_array.append(ratio_dist_bookowns)

            # print("Reward-Array: ", reward_array)
            # print("Count Books Array: ", num_books_array)
            # print("Time Array: ", time_array)
            # for i in time_array:
            #     print(type(i))
            # for one order
            # self.compare_dict[key]= [reward_array,route_array,dist_array,time_array,hubs_array,num_books_array,ratio_array,steps_array,dist_shares_array,dist_bookowns_array]

            # print(self.compare_dict)

            # compute the averages of each result array 
            # counter += 1
            reward_mean = Comparer.Average(reward_array)
            dist_mean = Comparer.Average(dist_array)
            time_mean = Comparer.Average(time_array)
            num_hubs_mean = Comparer.Average(hubs_array)
            num_books_mean = Comparer.Average(num_books_array)
            ratio_mean = Comparer.Average(ratio_array)
            steps_mean = Comparer.Average(steps_array)
            dist_shares_mean = Comparer.Average(dist_shares_array)
            dist_bookowns_mean = Comparer.Average(dist_bookowns_array)

            self.compare_dict[key] = [reward_mean,route_array,dist_mean,time_mean,num_hubs_mean,num_books_mean,ratio_mean,steps_mean,dist_shares_mean,dist_bookowns_mean]
            # print(self.compare_dict)
        
        # call comparer to get rankings
        Comparer.compare_onetrip(self)

    def main():
        set_agents(self.num_agents,self.agents_arg)


# possible agents to be compared
env = GraphEnv(use_config=True) # use normal GraphWorld for RLs
# RL agents
#w3 = BenchmarkWrapper("PPO",env)
# w4 = BenchmarkWrapper("DQN",env) # hat noch Fehler
w5 = BenchmarkWrapper("Rainbow",env)

from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattanBenchmark import GraphEnv # use Benchmark Graphworld for others
env = GraphEnv(use_config=True)
# benchmarks
w1 = BenchmarkWrapper("random",env)
# w2 = BenchmarkWrapper("cost",env)
w6 = BenchmarkWrapper("Shares",env)
w7 = BenchmarkWrapper("Bookown",env)
w8 = BenchmarkWrapper("SharesBookEnd",env)

# possible combinations of comparisons
# for one agent: Comparer(1,nameAgent,AgentObjekt)
# c = Comparer(1,w1.name,w1)
# c = Comparer(1,w2.name,w2)
# c = Comparer(1,w7.name,w7)
# c = Comparer(1,w6.name,w6)
# c = Comparer(1,w5.name,w5)
# c = Comparer(1,w8.name,w8)
# c = Comparer(3,w5.name,w6.name,w7.name,w5,w6,w7)
# c = Comparer(1,w4.name,w4)
# c = Comparer(2,w3.name,w5.name,w3,w5)
# c = Comparer(4,w1.name,w6.name,w7.name,w8.name,w1,w6,w7,w8)
# c = Comparer(3,w1.name,w2.name,w3.name,w1,w2,w3)
# c = Comparer(6,w1.name,w2.name,w3.name,w5.name,w6.name,w7.name,w1,w2,w3,w5,w6,w7)
# c = Comparer(5,w1.name,w2.name,w3.name,w4.name,w5.name,w1,w2,w3,w4,w5)
# c = Comparer(7,w1.name,w2.name,w3.name,w4.name,w5.name,w6.name,w7.name,w1,w2,w3,w4,w5,w6,w7)

# comparison of ALL benchmarks and rainbow
c = Comparer(5,w1.name,w5.name,w6.name,w7.name,w8.name,w1,w5,w6,w7,w8)

# compute the commparison
# c.establish_compare_onetrip() # call when only one order is placed (adapt row reads to 1 row in BenchmarkWrapper.read_orders!)
c.compare_multipletrips() # call when more than one order is placed (adapt row reads in BenchmarkWrapper.read_orders!)



