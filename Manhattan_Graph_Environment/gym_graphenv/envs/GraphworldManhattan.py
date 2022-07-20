from xml.dom.pulldom import parseString
import numpy as np
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import MarkerCluster
from datetime import datetime, timedelta
from array import array
import gym
from gym.utils import seeding
import random
import modin.pandas as pd
from gym import spaces
from pandas import Timestamp
import time
import pickle
import logging
import json
import os
import matplotlib.pyplot as plt
import networkx as nx 
from colorama import Fore, Back, Style
import sys
from PIL import Image, ImageDraw
import math 
import plotly.express as px
#from mpl_toolkits.basemap import Basemap
import plotly.express as px
RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1.
sys.path.insert(0,"")
# from config.definitions import ROOT_DIR
import statistics
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy



# CHANGES
from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.graphs.LearnGraph import LearnGraph
from Manhattan_Graph_Environment.OneHotVector import OneHotVector
from Manhattan_Graph_Environment.database_connection import DBConnection
# END OF CHANGES

class GraphEnv(gym.Env):

    REWARD_AWAY = -1
    REWARD_GOAL = 100
    WAIT_TIME_SECONDS = 300 
    WAIT_TIME_MINUTES= WAIT_TIME_SECONDS/60


    def __init__(self, use_config: bool = True ):
        DB_LOWER_BOUNDARY = '2016-01-01 00:00:00'
        DB_UPPER_BOUNDARY = '2016-01-14 23:59:59'
        self.LEARNGRAPH_FIRST_INIT_DONE = False
        print("USE CONFIG", use_config)

        self.done = False
        # f = open('graphworld_config.json')
        # data = json.load(f)
        # use_config=data['use_config']
        # print(use_config)

        if(use_config):
            self.env_config = self.read_config()
        else:
            self.env_config = None

        self.n_hubs = 120
        self.distance_matrix = None

        self.DB = DBConnection()

        manhattan_graph = ManhattanGraph(filename='simple', num_hubs=self.n_hubs)
        #manhattan_graph.setup_trips(self.START_TIME)
        self.manhattan_graph = manhattan_graph

        self.hubs = manhattan_graph.hubs

        self.trips = self.DB.getAvailableTrips(DB_LOWER_BOUNDARY, DB_UPPER_BOUNDARY)
        print(f"Initialized with {len(self.trips)} taxi rides within two weeks")
        print(f"Initialized with {len(self.hubs)} hubs")


        self.state = None
        self.state_of_delivery = DeliveryState.IN_DELIVERY
        self.allow_bookown = 0
        self.route_travel_distance=0


        self.action_space = gym.spaces.Discrete(self.n_hubs)

        self.observation_space = spaces.Dict(
            {
            # 'cost': gym.spaces.Box(low=np.zeros(70)-10, high=np.zeros(70)+10, shape=(70,), dtype=np.float64),
            'remaining_distance': gym.spaces.Box(low=np.zeros(self.n_hubs)-200000, high=np.zeros(self.n_hubs)+200000, shape=(self.n_hubs,), dtype=np.float64),
            'current_hub': gym.spaces.Box(low=0, high=1, shape=(self.n_hubs,), dtype=np.float64),
            'final_hub': gym.spaces.Box(low=0, high=1, shape=(self.n_hubs,), dtype=np.float64),
            'distinction': gym.spaces.Box(low=np.zeros(self.n_hubs)-1, high=np.zeros(self.n_hubs)+1, shape=(self.n_hubs,), dtype=np.float64),
            'allow_bookown': gym.spaces.Discrete(2)
        
        })
        self.rem_distance_values=[]
        self.rd_mean=120.2
        self.rd_stdev=5571.48

        self.distance_covered_with_shared=0
        self.distance_covered_with_ownrides=0

    def one_hot(self, pos):
        one_hot_vector = np.zeros(len(self.hubs))
        one_hot_vector[pos] = 1
        return one_hot_vector

    def reset(self):
        # two cases depending if we have env config
        #super().reset()

        #self.done = False
        resetExecutionStart = time.time()

        pickup_day = np.random.randint(low=1,high=14)
        pickup_hour =  np.random.randint(24)
        pickup_minute = np.random.randint(60) 
        self.route_taken = []
        self.START_TIME = datetime(2016,1,pickup_day,pickup_hour,pickup_minute,0).strftime('%Y-%m-%d %H:%M:%S')

        if (self.env_config == None or self.env_config == {}):
            print("Started Reset() without config")
            #self.final_hub = self.manhattan_graph.get_nodeids_list().index(random.sample(self.hubs,1)[0])
            self.final_hub = random.randint(0,self.n_hubs-1)
            #self.start_hub = self.manhattan_graph.get_nodeids_list().index(random.sample(self.hubs,1)[0])
            self.start_hub = random.randint(0,self.n_hubs-1)

            # just in case ;)
            if(self.start_hub == self.final_hub):
                self.start_hub = random.randint(0,self.n_hubs-1)

            self.position = self.start_hub

        # time for pickup
            self.pickup_time = datetime.strptime(self.START_TIME,'%Y-%m-%d %H:%M:%S')
            self.time = self.pickup_time
            self.total_travel_time = 0
            self.deadline=self.pickup_time+timedelta(hours=24)
            self.current_wait = 1 ## to avoid dividing by 0
        else:
            print("Started Reset() with config")
            self.final_hub = self.env_config['delivery_hub_index']

            self.start_hub = self.env_config['pickup_hub_index']

            self.position = self.start_hub

            self.pickup_time = self.env_config['pickup_timestamp']
            self.time = datetime.strptime(self.pickup_time, '%Y-%m-%d %H:%M:%S')
            self.total_travel_time = 0
            self.deadline=datetime.strptime(self.env_config['delivery_timestamp'], '%Y-%m-%d %H:%M:%S')
            self.current_wait = 0
        self.distance_covered_with_shared=0
        self.distance_covered_with_ownrides=0
        self.distance_reduced_with_shared=0 #to final hub
        self.distance_reduced_with_ownrides=0 #to final hub



        print(f"Reset initialized pickup: {self.position}")
        print(f"Reset initialized dropoff: {self.final_hub}")
        print(f"Reset initialized time: {self.time}")


        learn_graph = LearnGraph(n_hubs=self.n_hubs, manhattan_graph=self.manhattan_graph, final_hub=self.final_hub)
        self.learn_graph = learn_graph

        if(self.LEARNGRAPH_FIRST_INIT_DONE == False):
            self.distance_matrix = self.learn_graph.fill_distance_matrix()


        self.LEARNGRAPH_FIRST_INIT_DONE = True
        self.learn_graph.add_travel_cost_layer(self.availableTrips(), self.distance_matrix)
        self.learn_graph.add_remaining_distance_layer(current_hub=self.position, distance_matrix=self.distance_matrix)

        self.count_hubs = 0
        self.count_actions = 0
        self.count_wait = 0
        self.count_bookown = 0
        # whetherr it has booked any own
        self.booked_own = 0

        self.count_share = 0
        self.count_steps = 0
        self.action_choice = None

        self.old_position = self.start_hub
        self.current_trip = None

        self.own_ride = False
        self.has_waited=False
        reward=0

        if((self.deadline - self.time).total_seconds()/60 <= 120):
            self.allow_bookown = 1
        else:
            self.allow_bookown = 0

      
        # new metrics for shares and bookowns
        self.count_shared_available = 0
        self.boolean_shared_available = 0
        self.count_shared_available_useful = 0
        self.count_shared_taken_useful = 0
        self.boolean_useful_shares_available = 0
        self.orders_delivered_without_booked =  0
        # self.count_delive

        self.state = {
            # 'cost' : ((self.learn_graph.adjacency_matrix('cost')[self.position]-self.mean1)/self.stdev1).astype(np.float64),
            'remaining_distance': ((self.learn_graph.adjacency_matrix('remaining_distance')[self.position]-self.rd_mean)/self.rd_stdev).astype(np.float64),
            'current_hub' : self.one_hot(self.position).astype(np.float64),
            'final_hub' : self.one_hot(self.final_hub).astype(np.float64),
            'distinction' : self.learn_graph.adjacency_matrix('distinction')[self.position].astype(np.float64),
            'allow_bookown': self.allow_bookown,
            }
        self.rem_distance_values.extend(self.learn_graph.adjacency_matrix('remaining_distance')[self.position].astype(np.float64))

        self.shortest_distance=self.learn_graph.adjacency_matrix('remaining_distance')[self.position][self.final_hub]

        resetExecutionTime = (time.time() - resetExecutionStart)
        # print(f"Reset() Execution Time: {str(resetExecutionTime)}")
        return self.state

    def step(self, action: int):
        """ Executes an action based on the index passed as a parameter (only works with moves to direct neighbors as of now)
        Args:
            action (int): index of action to be taken from availableActions
        Returns:
            int: new position
            int: new reward
            boolean: isDone
        """

        startTime = time.time()

        #self.done = False

        # if agent is in time window 2 hours before deadline, we just move him to the final hub
        if((self.deadline - self.time).total_seconds()/60 <= 120):
            self.allow_bookown = 1
            action = self.final_hub
            print("Force Manual Delivery")
        else:
            self.allow_bookown = 0
            

        # determine whether shared ride was useful (= whether remaining distance was reduced)
        # compute the number of shared available actions and the number of useful shared available actions
        counter = 0
        boolean_available_temp = False
        boolean_useful_temp = False
        for hub in self.shared_rides_mask:
            if hub == 1:
                # self.boolean_shared_available = 1
                # count on step-base
                boolean_available_temp = True
                self.count_shared_available += 1
                # check whether remaining distance decreases with new position
                if self.state["remaining_distance"][counter] > 0:
                    # count on ride-base
                    self.count_shared_available_useful += 1
                    # self.boolean_useful_shares_available = 1
                    # count on step-base
                    boolean_useful_temp = True

        # set old position to current position before changing current position
        self.old_position = self.position
        step_duration = 0

        if self.validateAction(action):
            startTimeWait = time.time()
            self.count_steps +=1
            if(action == self.position):
            # action = wait
                step_duration = self.WAIT_TIME_SECONDS
                self.has_waited=True
                self.own_ride = False
                self.count_wait += 1
                self.action_choice = "Wait"
                # print("action == wait ")
                executionTimeWait = (time.time() - startTimeWait)
                # print(f"Time Wait: {str(executionTimeWait)}")
                route_taken = [self.manhattan_graph.get_nodeid_by_hub_index(self.position)]
                pass

            # action = share ride or book own ride
            else:
                pickup_nodeid = self.manhattan_graph.get_nodeid_by_hub_index(self.position)
                dropoff_nodeid = self.manhattan_graph.get_nodeid_by_hub_index(action)
                route=ox.shortest_path(self.manhattan_graph.inner_graph, pickup_nodeid,  dropoff_nodeid, weight='travel_time')
                route_taken = [pickup_nodeid]
                route_taken.extend(route)
                route_taken.append(dropoff_nodeid)
                route_travel_time = ox.utils_graph.get_route_edge_attributes(self.manhattan_graph.inner_graph,route,attribute='travel_time')
                self.route_travel_distance = sum(ox.utils_graph.get_route_edge_attributes(self.manhattan_graph.inner_graph,route,attribute='length'))


                if(self.learn_graph.wait_till_departure_times[(self.position,action)] == self.WAIT_TIME_SECONDS):
                    step_duration = sum(route_travel_time)+ self.WAIT_TIME_SECONDS#we add 5 minutes (300 seconds) so the taxi can arrive
                elif(self.learn_graph.wait_till_departure_times[(self.position,action)] != self.WAIT_TIME_SECONDS and self.learn_graph.wait_till_departure_times[(self.position,action)] != 0):
                    step_duration = sum(route_travel_time)
                    # TODO: String conversion von departure time besser direkt beim erstellen der Matrix
                    departure_time = datetime.strptime(self.learn_graph.wait_till_departure_times[(self.position,action)], '%Y-%m-%d %H:%M:%S')
                    self.current_wait = ( departure_time - self.time).seconds
                    step_duration += self.current_wait
                    self.time = departure_time

                
                print("Step Duration: ", step_duration)

                if(self.shared_rides_mask[action] == 1):
                    self.count_share += 1
                    self.action_choice = "Share"
                    print("action == share ")
                    # print(f"Rides Mask for Action {action}: {self.shared_rides_mask}")
                    
                    # check whether current action is useful
                    if self.state["remaining_distance"][action] > 0:
                        self.count_shared_taken_useful += 1
                    self.distance_covered_with_shared+=self.route_travel_distance
                    self.distance_reduced_with_shared+=self.learn_graph.adjacency_matrix('remaining_distance')[self.old_position][action]


                else:
                    self.count_bookown += 1
                    self.action_choice = "Book"
                    # print("action == book own ")
                    # print(f"Rides Mask for Action {action}: {self.shared_rides_mask}")
                    self.distance_covered_with_ownrides+=self.route_travel_distance
                    self.distance_reduced_with_ownrides+=self.learn_graph.adjacency_matrix('remaining_distance')[self.old_position][action]

                startTimeRide = time.time()
                self.has_waited=False
                self.count_hubs += 1


                self.old_position = self.position
                self.position = action

                executionTimeRide = (time.time() - startTimeRide)
                # print(f"Time Ride: {str(executionTimeRide)}")
                pass

        # brauchen wir nicht mehr oder??
        else:
            print("invalid action")
            #print("avail actions: ",self.available_actions)
            print("action: ",action)
            print("action space: ",self.action_space)

        self.time += timedelta(seconds=step_duration)
        # refresh travel cost layer after each step
        self.learn_graph.add_travel_cost_layer(self.availableTrips(), self.distance_matrix)
        self.learn_graph.add_remaining_distance_layer(current_hub=self.position, distance_matrix=self.distance_matrix)
        startTimeLearn = time.time()
        self.old_state = self.state
        self.state = {
            'remaining_distance': (((self.learn_graph.adjacency_matrix('remaining_distance')[self.position])-self.rd_mean)/self.rd_stdev).astype(np.float64),
            'current_hub' : self.one_hot(self.position).astype(np.float64),
            'final_hub' : self.one_hot(self.final_hub).astype(np.float64),
            'distinction' : self.learn_graph.adjacency_matrix('distinction')[self.position].astype(np.float64),
            'allow_bookown': self.allow_bookown,
            }

        # print("New State: ")        
        # print(self.state)

        print("End Time ", self.time)

        self.count_actions += 1

        reward, self.done, self.state_of_delivery = self.compute_reward(action)

        # if (self.done):
        #      self.mean_rd=sum(self.rem_distance_values)/len(self.rem_distance_values)
        #      #print("mean rd: ",self.mean_rd)
        #      self.sd_rd=statistics.stdev(self.rem_distance_values)
        #      #print("stdev rd: ",self.sd_rd)
        #self.state_of_delivery = state_of_delivery
        executionTime = (time.time() - startTime)

        # print(self.route_taken)

        # counting on step-base (not individual ride-base)
        if boolean_available_temp == True:
            self.boolean_shared_available += 1
        if boolean_useful_temp == True:
            self.boolean_useful_shares_available += 1

        # test prints for counters
        # print("Out of ", self.count_actions, " steps, in ", self.boolean_shared_available, " steps shared rides were available")
        """
        if self.boolean_shared_available == 1:
            print("In Step ", self.count_actions, " some share is available, number: ", self.count_shared_available)
        else:
            print("In Step ", self.count_actions, " there is no share available")
        """
        # print("In Step ", self.count_actions, " a useful share is available, number: ", self.boolean_useful_shares_available)            

        # print("Step End")
        return self.state, reward,  self.done, {"timestamp": self.time,"step_travel_time":step_duration,"distance":self.distance_matrix[self.old_position][self.position], "count_hubs":self.count_hubs, "action": self.action_choice, "hub_index": action, "route": route_taken}


    def compute_reward(self, action):
        # cost_of_action = self.learn_graph.adjacency_matrix('cost')[self.old_position][action]
        distance_gained = self.old_state['remaining_distance'][self.position]
        old_distinction = self.old_state['distinction']
        cost_of_action = self.learn_graph.adjacency_matrix('cost')[self.old_position][action]
        print(self.old_position, "->", action, distance_gained)
        self.done = False

        bookown = False
        wait = False
        share = False
        reward = 0

        if(old_distinction[action] == -1): # book own
            bookown = True
        elif(old_distinction[action] == 0): # wait
            wait = True
        else:
            share = True

        if(self.position == self.final_hub):
            self.done = True
        # came to final hub 
            if((self.deadline-self.time).total_seconds()/60 >= 120):
                # in time
                state_of_delivery = DeliveryState.DELIVERED_ON_TIME
                print(f"DELIVERED IN TIME AFTER {self.count_actions} ACTIONS (#wait: {self.count_wait}, #share: {self.count_share}, #book own: {self.count_bookown})")
                if(bookown == True):
                    if(self.allow_bookown == 0):
                        # strong punishment for bookown before 2h window before deadline
                        reward = old_distinction[action]*100000
                        reward += 10000
                    else:
                        reward = 0
                elif(share == True):
                    # high reward if agent comes to final hub with shared ride
                    reward = 100000
                    
            else:
                # in time delivered with delivery time < 2 hours to deadline
                state_of_delivery = DeliveryState.DELIVERED_ON_TIME
                print(f"MANUAL DELIVERY WITH {(self.deadline-self.time).total_seconds()/60} MINUTES TO DEADLINE - ACTIONS (#wait: {self.count_wait}, #share: {self.count_share}, #book own: {self.count_bookown})")
                reward = 0

        # did not come to final hub:
        else:
            # intermediate action
            self.done = False
            state_of_delivery = DeliveryState.IN_DELIVERY
            if(wait == True):
                # print("Action in Reward: Wait")
                # print("Time:", self.time)
                # print("Deadline:", self.deadline)
                reward = 0
            elif(bookown == True):
                # print("Action in Reward: Bookown")
                # print("Time:", self.time)
                # print("Deadline:", self.deadline)
                if(self.allow_bookown == 0):
                    reward = old_distinction[action]*100000
                else:
                    # kann eigentlich nicht sein dieser Case
                    reward = (distance_gained/100) * 1000
            elif(share == True):
                # print("Action in Reward: Share")
                # print("Time:", self.time)
                # print("Deadline:", self.deadline)
                reward = (distance_gained/100) * 1000 + old_distinction[action]*10000


        # # if delay is greater than 2 hours (=120 minutes), terminate training episode
        # if((self.time-self.deadline).total_seconds()/60 >= 120 or self.count_actions>200):
        #     self.done = True
        #     reward = - 10000
        #     state_of_delivery = DeliveryState.NOT_DELIVERED
        #     print("BOX WAS NOT DELIVERED until 2 hours after deadline")
        # # if box is delivered to final hub in time
        # if (self.position == self.final_hub and self.time <= self.deadline):
        #     print(f"DELIVERED IN TIME AFTER {self.count_actions} ACTIONS (#wait: {self.count_wait}, #share: {self.count_share}, #book own: {self.count_bookown}")
        #     reward = 10000
        #     self.done = True
        #     state_of_delivery = DeliveryState.DELIVERED_ON_TIME
        # # if box is delivered to final hub with delay
        # elif(self.position == self.final_hub and (self.time-self.deadline).total_seconds()/60 < 120): #self.time > self.deadline):
        #     overtime = self.time - self.deadline
        #     overtime = round(overtime.total_seconds() / 60)
        #     print(f"DELIVERED AFTER {self.count_actions} ACTIONS (#wait: {self.count_wait}, #share: {self.count_share}, #book own: {self.count_bookown} WITH DELAY: {overtime}")
        #     reward = 10000 - overtime
        #     self.done = True
        #     state_of_delivery = DeliveryState.DELIVERED_WITH_DELAY
        # # if box is not delivered to final hub
        # elif(self.done==False):
        #     # reward = distance_gained / 100 + old_distinction[action]*1000
        #     # print("book available",self.allow_bookown)
        #     # print("Distinction action available",old_distinction[action])
        #     if(self.allow_bookown == 0 and old_distinction[action] == -1 ):
        #          reward = old_distinction[action]*100000
        #     elif(self.allow_bookown == 1 and old_distinction[action] == -1):
        #         reward = (distance_gained/100) * 1000
        #     else:
        #         reward = (distance_gained/100) * 1000 + old_distinction[action]*1000
        #     # print(f"INTERMEDIATE STEP ACTIONS: (#wait: {self.count_wait}, #share: {self.count_share}, #book own: {self.count_bookown}")
        #     state_of_delivery = DeliveryState.IN_DELIVERY
        #     #done = False

        #print(self.old_position, "->", action, reward)
        print(f"Reward: {reward}")
        print(f"Action: {action}")
        #print(f"Old Distinction: {old_distinction}")
        #print(f"Rides Mask for Action {action}: {self.shared_rides_mask}")

        print("Done:", self.done)

        return reward, self.done, state_of_delivery

    def get_available_actions(self):
        """ Returns the available actions at the current position. Uses a simplified action space with moves to all direct neighbors allowed.
        Returns:
            list: list of nodeIds of direct neighbors
        """

        startTime = time.time()

        wait = [{'type': 'wait'}]
        ownRide = [{'type': 'ownRide'}]
        available_rides = list(self.availableTrips(10))

        executionTime = (time.time() - startTime)
        # print('get_available_actions() Execution time: ' + str(executionTime) + ' seconds')

        available_actions = [wait,ownRide,*available_rides]
        self.available_actions = available_actions

        return available_actions

    def availableTrips(self, time_window=5):
        """ Returns a list of all available trips at the current node and within the next 5 minutes. Includes the time of departure from the current node as well as the target node of the trip.
        Returns:
            list: [departure_time,target_node]
        """
        startTime = time.time()
        list_trips=[]
        position=self.manhattan_graph.get_nodeid_by_hub_index(self.position)
        position_str=str(position)
        final_hub_postion=self.manhattan_graph.get_nodeid_by_index(self.final_hub)

        start_timestamp=self.time
        end_timestamp = self.time + timedelta(minutes=time_window)

        trips = self.trips

        for tripId, nodeId, timestamp in trips:
            if(nodeId == position):
                if timestamp <= end_timestamp and timestamp >= start_timestamp:


                    route, times = self.DB.getRouteFromTrip(tripId)
                    isNotFinalNode = True
                    if isNotFinalNode:
                        index_in_route = route.index(position)
                        position_timestamp = times[index_in_route]
                        route_to_target_node=route[index_in_route::]
                        hubsOnRoute = any(node in route_to_target_node for node in self.hubs)
                        if hubsOnRoute:
                            route_hubs = [node for node in route_to_target_node if node in self.hubs]
                            for hub in route_hubs:
                                index_hub_in_route = route.index(hub)
                                index_hub_in_route += 1
                                route_to_target_hub = route[index_in_route:index_hub_in_route]
                                if(hub != position):
                                    trip = {'departure_time': position_timestamp, 'target_hub': hub, 'route': route_to_target_hub, 'trip_row_id': tripId}
                                    list_trips.append(trip)
        self.available_actions = list_trips

        # create index vector
        shared_rides_mask = np.zeros(self.n_hubs)
        for i in range(len(list_trips)):
            shared_rides_mask[self.manhattan_graph.get_hub_index_by_nodeid(list_trips[i]['target_hub'])] = 1

        self.shared_rides_mask = shared_rides_mask
        # print(shared_rides_mask)
        #print(list_trips)

        executionTime = (time.time() - startTime)
        # print('found '+ str(len(list_trips)) +' trips, ' + 'current time: ' + str(self.time))
        return list_trips

    def validateAction(self, action):
        return action < self.n_hubs

    def read_config(self):
        #filepath = os.path.join(ROOT_DIR,'env_config.pkl')
        #filepath = "/Users/noah/Desktop/Repositories/ines-autonomous-dispatching/Manhattan_Graph_Environment/env_config.pkl"
        filepath = "env_config.pkl"
        with open(filepath,'rb') as f:
            loaded_dict = pickle.load(f)
        self.env_config = loaded_dict
        return loaded_dict


    def render(self, visualize_actionspace: bool = False):
        """_summary_
        Args:
            visualize_actionspace (bool, optional): _description_. Defaults to False.
        Returns:
            _type_: _description_

        """

        shared_rides = list()
        shared_ids = list()
       
        for i, trip in enumerate(self.availableTrips()):
            shared_ids.append(trip['target_hub'])
          
        #print('Shared: ', shared_ids)

        all_hubs = self.hubs
        book_own_ids = list(set(all_hubs) - set(shared_ids))
        #print('Book owns: ', book_own_ids)
           
        position = self.manhattan_graph.get_nodeid_by_index(self.position)
        final = self.manhattan_graph.get_nodeid_by_index(self.final_hub)
        start = self.manhattan_graph.get_nodeid_by_index(self.start_hub)

        node_sizes = list()
        actions = []
        for n in self.manhattan_graph.nodes():
            if n == position:
                actions.append('position')
                node_sizes.append(120)
            else:
                if n == final:
                    actions.append('final')
                    node_sizes.append(120)
                else:
                    if n == start:
                        actions.append('start')
                        node_sizes.append(120)
                    else:
                        if n in shared_ids:
                            actions.append('shared')
                            node_sizes.append(120)
                        else:
                            if n in book_own_ids:
                                actions.append('book')
                                node_sizes.append(120)
                            else:
                                #colors.append('w')
                                node_sizes.append(0)
        df = pd.read_csv("ines-autonomous-dispatching/data/hubs/longlist.csv")
        df['actions']=actions

        graph = self.manhattan_graph.inner_graph

        px.set_mapbox_access_token(open("mapbox_token").read())
        #df = px.data.carshare()
        fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name ="id", color="actions", #size="car_hours",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)  
        fig.show()

        #fig, ax = ox.plot_graph(graph, figsize=(25,25), bgcolor='#89cff0',node_color=colors, node_size = node_sizes, edge_linewidth=1, show=False, save=True, filepath='try1.png')
        
        # nx.draw_circular(graph, with_labels = True)
        # plt.savefig("nx2.png")
        # gdf_nodes = ox.graph_to_gdfs(graph, edges=False, node_geometry=False)[["x", "y"]]
        # west, south = gdf_nodes.min()
        # east, north = gdf_nodes.max()

        

        # fig = px.scatter_geo(df,lat='latitude',lon='longitude', hover_name="id")
        # fig.update_layout(title = 'World map', title_x=0.5)
        # fig.show()

        # print('west', west)
        # print('east', east)
        # print('south', south)
        # print('north', north)

        # fig = plt.figure(figsize=(8, 8))
        # m = Basemap(projection='lcc', resolution=None,
        #             width=8E6, height=8E6, 
        #             lat_0=south, lon_0=west)
        # m.etopo(scale=0.5, alpha=0.5)

        # Map (long, lat) to (x, y) for plotting
        

        # rx = -486.913544
        # ry = 188.544021

        # rrx = 0.1517852
        # rry = 0.2168104

        # total_vertical = abs(north - south) 
        # total_horizontal = abs(east - west)

        # upper_mapx = 2712
        # upper_mapy = 5662

        # img = Image.open('try1.png')
        # d1 = ImageDraw.Draw(img)

        # for hub in self.hubs:
        #     print('Hub:', hub)
        #     index = self.manhattan_graph.get_hub_index_by_nodeid(hub)
        #     hub_x = self.manhattan_graph.get_node_by_index(index)['x']
        #     hub_y = self.manhattan_graph.get_node_by_index(index)['y']
        #     print(hub_x, hub_y)
            # x, y = m(hub_x, hub_y)
            # plt.plot(x, y, 'ok', markersize=5)
            # plt.text(x, y, str(index), fontsize=12)
            #ax.annotate(str(index), (hub_x, hub_y))

            # x1 = hub_x/rx
            # y1 = hub_y/ry

            # print(x1, y1)

            # final_x = x1/rrx
            # final_y = y1/rry

            #r= 6.371


            # final_x = -(hub_x)/total_horizontal * upper_mapx/1000
            # final_y = (hub_y)/total_vertical * upper_mapy/1000

            # final_x = -r * hub_x * math.cos(total_vertical/2)*10
            # final_y = r * hub_y*10

            #final_x = hub_x/west + 1
            #final_y = hub_y/north


            # final_x = hub_x + west
            # final_y = hub_y - south

            #print(final_x, final_y)
            #d1.text((final_x, final_y), str(index), fill=(255, 0, 0))
            
           
            # d1.text((28, 36), "Hello, TutorialsPoint!", fill=(255, 0, 0))
            # d1.text((0, 0), "Hello, 0", fill=(255, 0, 0))
            # d1.text((2000, 5000), "AAAAaaaaaa", fill=(255, 0, 0))

        #img.show()
        #img.save("try1.png")
            #fig.text(final_x, final_y, str(index), style = 'normal', fontsize = 10, color = "green")
        
        # fig.text(1, 1, '1,1', style = 'normal', fontsize = 10, color = "green")
        # fig.text(0, 0, '0,0', style = 'normal', fontsize = 10, color = "green")

        # fig.text(0.5, 0.5, '0.5,0.5', style = 'normal', fontsize = 10, color = "green")


        # current_pos_y = self.manhattan_graph.get_node_by_index(self.position)['y']
        # print("Legend HUBS:")
        # print(Fore.RED + 'RED - SHARED')
        # print(Fore.CYAN + 'TURQUISE - BOOK OWN')
        # print(Fore.MAGENTA + 'LILA - CURRENT')
        # print(Fore.GREEN + 'GREEN - START')
        # print(Fore.BLUE + 'BLUE - FINAL')
        # fig.show()
        # plt.show()


class DeliveryState:
    DELIVERED_ON_TIME, DELIVERED_WITH_DELAY, NOT_DELIVERED, IN_DELIVERY = range(4)

class CustomCallbacks(DefaultCallbacks):
    last_count_delivered_on_time = 0
    last_count_delivered_with_delay = 0
    last_count_not_delivered = 0
    count_not_delivered = 0
    count_delivered_with_delay = 0
    count_delivered_on_time = 0

    # new metrics for shares and bookowns
    count_shared_available = 0
    count_shared_available_useful = 0
    count_shared_taken = 0
    last_count_bookowns = 0
    count_bookowns = 0
    count_shared_taken_useful = 0
    boolean_useful_shares_available = 0
    boolean_shared_available = 0
    orders_delivered_without_booked = 0

    def on_algorithm_init(
        self,
        *,
        algorithm,
        **kwargs,
    ):
        """Callback run when a new trainer instance has finished setup.

        This method gets called at the end of Trainer.setup() after all
        the initialization is done, and before actually training starts.

        Args:
            trainer: Reference to the trainer instance.
            kwargs: Forward compatibility placeholder.
        """
        self.count_delivered_on_time = 0
        self.count_delivered_with_delay = 0
        self.count_not_delivered = 0

        # metrics for shares and bookowns
        self.count_shared_available = 0
        self.boolean_shared_available = 0
        self.count_shared_taken = 0
        self.count_bookowns = 0

        self.count_shared_available_useful = 0
        self.count_shared_taken_useful = 0
        self.boolean_useful_shares_available = 0
        self.orders_delivered_without_booked = 0

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.env = base_env.get_sub_environments(as_dict = True)[0]
        episode.custom_metrics["count_not_delivered"] = 0
        episode.custom_metrics["count_delivered_with_delay"] = 0
        episode.custom_metrics["count_delivered_on_time"] = 0

        # metrics for shares and bookowns
        #episode.custom_metrics["count_shared_available"] = 0
        #episode.custom_metrics["count_shared_taken"] = 0
        episode.custom_metrics["boolean_has_booked_any_own"] = 0
        episode.custom_metrics["count_shared_available_useful"] = 0
        #episode.custom_metrics["count_shared_taken_useful"] = 0        

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )

        episode.custom_metrics["count_wait"] = episode.env.count_wait
        episode.custom_metrics["count_bookown"] = episode.env.count_bookown
        episode.custom_metrics["count_share"] = episode.env.count_share
        episode.custom_metrics["count_steps"] = episode.env.count_steps
        episode.custom_metrics["share_wait"] = float(episode.env.count_wait / episode.env.count_actions)
        episode.custom_metrics["share_bookown"] = float(episode.env.count_bookown / episode.env.count_actions)
        episode.custom_metrics["share_share"] = float(episode.env.count_share / episode.env.count_actions)
        episode.custom_metrics["share_to_own_ratio"] = episode.env.count_share if episode.env.count_bookown == 0 else float(episode.env.count_share / episode.env.count_bookown)
        episode.custom_metrics["share_to_own_ratio"] = episode.env.count_share if episode.env.count_bookown == 0 else float(episode.env.count_share / episode.env.count_bookown)
        
        # metrics for shares and bookowns
        # ratio of shared taken when a shared is available
        if episode.env.boolean_shared_available == 0:
            episode.custom_metrics["shared_taken_to_shared_available"] = 0
        else:
            episode.custom_metrics["shared_taken_to_shared_available"] =  float(episode.env.count_share / episode.env.boolean_shared_available)
        # counting the shared availables (if one is available in a step, then +1)
        episode.custom_metrics["count_shared_available"] = episode.env.boolean_shared_available
        episode.custom_metrics["ratio_shared_available_to_all_steps"] = episode.env.boolean_shared_available / episode.env.count_steps
        # ratio: useful available shares (reducing remaining distance) of available shares
        if episode.env.count_shared_available == 0:
            episode.custom_metrics["shared_available_useful_to_shared_available"] = 0
        else:
            episode.custom_metrics["shared_available_useful_to_shared_available"] = float(episode.env.count_shared_available_useful/episode.env.count_shared_available)
        
        # counting the useful available shared rides
        episode.custom_metrics["count_shared_available_useful"] = episode.env.boolean_useful_shares_available
        # ratio: useful shares taken of useful shares available
        if episode.env.boolean_useful_shares_available == 0:
            episode.custom_metrics["shared_taken_useful_to_shared_available_useful"] = 0
        else:
            episode.custom_metrics["shared_taken_useful_to_shared_available_useful"] = float(episode.env.count_shared_taken_useful/episode.env.boolean_useful_shares_available)

        # displays 1 if any trip was booked and 0 if none was booked
        if episode.env.count_bookown > 0:
            episode.custom_metrics["boolean_has_booked_any_own"] = 1
        else:
            episode.custom_metrics["boolean_has_booked_any_own"] = 0

        if (episode.env.state_of_delivery == DeliveryState.DELIVERED_ON_TIME):
            self.count_delivered_on_time +=1
            episode.custom_metrics["count_delivered_on_time"] = self.count_delivered_on_time
            self.orders_delivered_without_booked += 1
        elif (episode.env.state_of_delivery == DeliveryState.DELIVERED_WITH_DELAY):
            self.count_delivered_with_delay +=1
            episode.custom_metrics["count_delivered_with_delay"] = self.count_delivered_with_delay
        elif (episode.env.state_of_delivery == DeliveryState.NOT_DELIVERED):
            self.count_not_delivered +=1
            episode.custom_metrics["count_not_delivered"] = self.count_not_delivered

        if (self.count_delivered_on_time==0):
            episode.custom_metrics["ratio_delivered_without_bookown_to_all_delivered"] = 0
        else:
            episode.custom_metrics["ratio_delivered_without_bookown_to_all_delivered"] = float(episode.env.orders_delivered_without_booked/self.count_delivered_on_time)

        # zum Vergleich ohne Abzug später
        # episode.custom_metrics["count_not_delivered_first"] = self.count_not_delivered

        #how much distance (in % of total distance) we don't have to ride with book own 
        episode.custom_metrics['bookown_distance_not_covered_share']=1-(episode.env.distance_covered_with_ownrides/episode.env.shortest_distance)
        #how much distance we don't have to ride with book own 
        episode.custom_metrics['bookown_distance_not_covered']=episode.env.shortest_distance-episode.env.distance_covered_with_ownrides
        
        episode.custom_metrics['distance_reduced_with_ownrides']=episode.env.distance_reduced_with_ownrides
        episode.custom_metrics['distance_reduced_with_shared']=episode.env.distance_reduced_with_shared


        episode.custom_metrics['distance_reduced_with_ownrides_share']=episode.env.distance_reduced_with_ownrides/episode.env.shortest_distance
        episode.custom_metrics['distance_reduced_with_shared_share']=episode.env.distance_reduced_with_shared/episode.env.shortest_distance



    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print(
            "trainer.train() result: {} -> {} episodes".format(
                trainer, result["episodes_this_iter"]
            )
        )
        # you can mutate the result dict to add new fields to return
        result["count_wait_min"] = result['custom_metrics']['count_wait_min']
        result["count_wait_max"] = result['custom_metrics']['count_wait_max']
        result["count_wait_mean"] = result['custom_metrics']['count_wait_mean']
        result["waiting_time_mean"] = result['custom_metrics']['count_wait_mean']*5

        result["count_bookown_min"] = result['custom_metrics']['count_bookown_min']
        result["count_bookown_max"] = result['custom_metrics']['count_bookown_max']
        result["count_bookown_mean"] = result['custom_metrics']['count_bookown_mean']
        result["count_share_min"] = result['custom_metrics']['count_share_min']
        result["count_share_max"] = result['custom_metrics']['count_share_max']
        result["count_share_mean"] = result['custom_metrics']['count_share_mean']
        result["count_steps_min"] = result['custom_metrics']['count_steps_min']
        result["count_steps_max"] = result['custom_metrics']['count_steps_max']
        result["count_steps_mean"] = result['custom_metrics']['count_steps_mean']

        result["share_wait_min"] = result['custom_metrics']['share_wait_min']
        result["share_wait_max"] = result['custom_metrics']['share_wait_max']
        result["share_wait_mean"] = result['custom_metrics']['share_wait_mean']
        result["share_bookown_min"] = result['custom_metrics']['share_bookown_min']
        result["share_bookown_max"] = result['custom_metrics']['share_bookown_max']
        result["share_bookown_mean"] = result['custom_metrics']['share_bookown_mean']
        result["share_share_min"] = result['custom_metrics']['share_share_min']
        result["share_share_max"] = result['custom_metrics']['share_share_max']
        result["share_share_mean"] = result['custom_metrics']['share_share_mean']
        
        result["share_to_own_ratio_min"] = result['custom_metrics']['share_to_own_ratio_min']
        result["share_to_own_ratio_max"] = result['custom_metrics']['share_to_own_ratio_max']
        result["share_to_own_ratio_mean"] = result['custom_metrics']['share_to_own_ratio_mean']

        result["count_delivered_on_time"] = result['custom_metrics']["count_delivered_on_time_max"] - CustomCallbacks.last_count_delivered_on_time
        result["count_delivered_with_delay"] = result['custom_metrics']["count_delivered_with_delay_max"] - CustomCallbacks.last_count_delivered_with_delay
        if result["count_delivered_with_delay"] < 0:
            result["count_delivered_with_delay"] = 0
        """
        print("COUNTER AUSGABE")
        print("Erg:", result['custom_metrics']["count_not_delivered_max"])
        print("Abzug", self.last_count_not_delivered)
        """

        result["count_not_delivered"] = result['custom_metrics']["count_not_delivered_max"] - CustomCallbacks.last_count_not_delivered
        # zum Vergleich ohne Abzug
        # result["count_not_delivered_first"] = result['custom_metrics']["count_not_delivered_max"]
        
        CustomCallbacks.last_count_not_delivered = CustomCallbacks.last_count_not_delivered + result["count_not_delivered"]
        CustomCallbacks.last_count_delivered_with_delay = CustomCallbacks.last_count_delivered_with_delay + result["count_delivered_with_delay"]
        CustomCallbacks.last_count_delivered_on_time = CustomCallbacks.last_count_delivered_on_time + result["count_delivered_on_time"]

        # metrics für shares and bookowns
        result["boolean_has_booked_any_own"] = result['custom_metrics']["boolean_has_booked_any_own_mean"] # - CustomCallbacks.last_count_bookowns
        # CustomCallbacks.last_count_bookowns = CustomCallbacks.last_count_bookowns + result["count_booked_own"]
        result["shared_taken_to_shared_available"] = result['custom_metrics']["shared_taken_to_shared_available_mean"]
        result["count_shared_available"] = result['custom_metrics']["count_shared_available_mean"]
        result["ratio_shared_available_to_all_steps"] = result['custom_metrics']["ratio_shared_available_to_all_steps_mean"]
        result["shared_available_useful_to_shared_available"] = result['custom_metrics']["shared_available_useful_to_shared_available_mean"]
        result["shared_taken_useful_to_shared_available_useful"] = result['custom_metrics']["shared_taken_useful_to_shared_available_useful_mean"]
        result["count_shared_available_useful"] = result['custom_metrics']["count_shared_available_useful_mean"]

        result["ratio_delivered_without_bookown_to_all_delivered"] = result['custom_metrics']["ratio_delivered_without_bookown_to_all_delivered_mean"]



        #metrics about bookown distance reduced and rem distance reduced
        result['bookown_distance_not_covered_share']=result['custom_metrics']['bookown_distance_not_covered_share_mean']
        result['bookown_distance_not_covered']=result['custom_metrics']['bookown_distance_not_covered_mean']
        result['distance_reduced_with_ownrides']=result['custom_metrics']['distance_reduced_with_ownrides_mean']
        result['distance_reduced_with_shared']=result['custom_metrics']['distance_reduced_with_shared_mean']
        result['distance_reduced_with_ownrides_share']=result['custom_metrics']['distance_reduced_with_ownrides_share_mean']
        result['distance_reduced_with_shared_share']=result['custom_metrics']['distance_reduced_with_shared_share_mean']

