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
from config.definitions import ROOT_DIR
from gym import spaces, logger
from gym.utils import seeding
from or_gym.utils import assign_env_config

from typing import Dict

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy

import sys
sys.path.insert(0,"")

# CHANGES
from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph
from Manhattan_Graph_Environment.graphs.LearnGraph import LearnGraph
from Manhattan_Graph_Environment.OneHotVector import OneHotVector
from Manhattan_Graph_Environment.database_connection import DBConnection
# END OF CHANGES

class GraphEnv(gym.Env):

    REWARD_AWAY = -1
    REWARD_GOAL = 100
    
    def __init__(self, use_config: bool = True ):
        DB_LOWER_BOUNDARY = '2016-01-01 00:00:00'
        DB_UPPER_BOUNDARY = '2016-01-14 23:59:59'
        self.LEARNGRAPH_FIRST_INIT_DONE = False
        print("USE CONFIG", use_config)

        self.done = False
        self.mask = False
        # f = open('graphworld_config.json')
        # data = json.load(f)
        # use_config=data['use_config']
        # print(use_config)

        if(use_config):
            self.env_config = self.read_config()
        else:
             self.env_config = None
    
        self.n_hubs = 70
        self.distance_matrix = None

        self.DB = DBConnection()

        manhattan_graph = ManhattanGraph(filename='simple', num_hubs=self.n_hubs)
        #manhattan_graph.setup_trips(self.START_TIME)
        self.manhattan_graph = manhattan_graph

        self.hubs = manhattan_graph.hubs

        self.trips = self.DB.getAvailableTrips(DB_LOWER_BOUNDARY, DB_UPPER_BOUNDARY)
        print(f"Initialized with {len(self.trips)} taxi rides within two weeks")


        self.state = None
        self.state_of_delivery = DeliveryState.IN_DELIVERY

      
        self.action_space = gym.spaces.Discrete(self.n_hubs) 
        obs_space = spaces.Dict(
                {
                'cost': gym.spaces.Box(low=np.zeros(70)-10, high=np.zeros(70)+10, shape=(70,), dtype=np.float64),
                'remaining_distance': gym.spaces.Box(low=np.zeros(70)-10, high=np.zeros(70)+10, shape=(70,), dtype=np.float64),
                'current_hub': gym.spaces.Box(low=0, high=1, shape=(70,), dtype=np.float64),
                'final_hub': gym.spaces.Box(low=0, high=1, shape=(70,), dtype=np.float64),
                'observations': gym.spaces.Box(low=np.zeros(70)-1, high=np.zeros(70)+1, shape=(70,), dtype=np.float64)
            })

        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.n_hubs,), dtype=np.int32),
                "avail_actions": spaces.Box(0, 1, shape=(self.n_hubs,), dtype=np.int16),
                "observations": obs_space
                })
        else:
            self.observation_space = spaces.Dict(
                {
                'cost': gym.spaces.Box(low=np.zeros(70)-10, high=np.zeros(70)+10, shape=(70,), dtype=np.float64),
                'remaining_distance': gym.spaces.Box(low=np.zeros(70)-10, high=np.zeros(70)+10, shape=(70,), dtype=np.float64),
                'current_hub': gym.spaces.Box(low=0, high=1, shape=(70,), dtype=np.float64),
                'final_hub': gym.spaces.Box(low=0, high=1, shape=(70,), dtype=np.float64),
                'observations': gym.spaces.Box(low=np.zeros(70)-1, high=np.zeros(70)+1, shape=(70,), dtype=np.float64)
            })
        self.mean1=6779.17
        self.mean2=13653.00
        self.stdev1=4433.51
        self.stdev2=6809.79

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
            self.deadline=self.pickup_time+timedelta(hours=12)
            self.current_wait = 1 ## to avoid dividing by 0
        else:
            print("Started Reset() with config")
            self.final_hub = self.env_config['delivery_hub_index']

            self.start_hub = self.env_config['pickup_hub_index']

            self.position = self.start_hub

            self.pickup_time = self.env_config['pickup_timestamp']
            self.time = datetime.strptime(self.pickup_time, '%Y-%m-%d %H:%M:%S')
            self.total_travel_time = 0
            self.deadline = datetime.strptime(self.env_config['delivery_timestamp'], '%Y-%m-%d %H:%M:%S')
            self.current_wait = 0


        
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
        self.count_share = 0
        self.count_steps = 0
        self.action_choice = None

        self.old_position = self.start_hub
        self.current_trip = None

        self.own_ride = False
        self.has_waited=False
        reward=0

        self.state = {
            'cost' : ((self.learn_graph.adjacency_matrix('cost')[self.position]-self.mean1)/self.stdev1).astype(np.float64),
            'remaining_distance': ((self.learn_graph.adjacency_matrix('remaining_distance')[self.position]-self.mean2)/self.stdev2).astype(np.float64),
            'current_hub' : self.one_hot(self.position).astype(np.float64), 
            'final_hub' : self.one_hot(self.final_hub).astype(np.float64),
            'observations' : self.learn_graph.adjacency_matrix('distinction')[self.position].astype(np.float64)
            }
        resetExecutionTime = (time.time() - resetExecutionStart)
        # print(f"Reset() Execution Time: {str(resetExecutionTime)}")
        self.update_action_mask()
        print("Masked State Return",self.state)
        return self.state


    def update_action_mask(self):
        if self.mask:
            mask = np.where(self.state['observations'] < 0,
            0, 1)
            mask_state = {
                "action_mask": mask,
                "avail_actions": np.ones(self.n_hubs, dtype=np.int16),
                "observations": self.state
                }
            self.state = mask_state     
        else:
            pass
          

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

        self.done =  False

        # set old position to current position before changing current position
        self.old_position = self.position
        step_duration = 0

        if self.validateAction(action):
            startTimeWait = time.time()
            self.count_steps +=1
            if(action == self.position):
            # action = wait
                step_duration = 300
                self.has_waited=True
                self.own_ride = False
                self.count_wait += 1
                self.action_choice = "Wait"
                print("action == wait ")
                executionTimeWait = (time.time() - startTimeWait)
                # print(f"Time Wait: {str(executionTimeWait)}")
                pass

            # action = share ride or book own ride
            else:
                if(self.shared_rides_mask[action] == 1):
                    self.count_share += 1
                    self.action_choice = "Share"
                    print("action == share ")
                    print(f"Rides Mask for Action {action}: {self.shared_rides_mask}")
                else:
                    self.count_bookown += 1
                    self.action_choice = "Book"
                    print("action == book own ")
                    print(f"Rides Mask for Action {action}: {self.shared_rides_mask}")
                startTimeRide = time.time()
                self.has_waited=False
                self.count_hubs += 1

                pickup_nodeid = self.manhattan_graph.get_nodeid_by_hub_index(self.position)
                dropoff_nodeid = self.manhattan_graph.get_nodeid_by_hub_index(action)

                route = ox.shortest_path(self.manhattan_graph.inner_graph, pickup_nodeid,  dropoff_nodeid, weight='travel_time')
                route_travel_time = ox.utils_graph.get_route_edge_attributes(self.manhattan_graph.inner_graph,route,attribute='travel_time')

                if(self.learn_graph.wait_till_departure_times[(self.position,action)] == 300):
                    step_duration = sum(route_travel_time)+300 #we add 5 minutes (300 seconds) so the taxi can arrive
                elif(self.learn_graph.wait_till_departure_times[(self.position,action)] != 300 and self.learn_graph.wait_till_departure_times[(self.position,action)] != 0):
                    step_duration = sum(route_travel_time)
                    # TODO: String conversion von departure time besser direkt beim erstellen der Matrix
                    departure_time = datetime.strptime(self.learn_graph.wait_till_departure_times[(self.position,action)], '%Y-%m-%d %H:%M:%S')
                    self.current_wait = ( departure_time - self.time).seconds
                    step_duration += self.current_wait
                    self.time = departure_time
                
                self.old_position = self.position
                self.position = action
                
                executionTimeRide = (time.time() - startTimeRide)
                # print(f"Time Ride: {str(executionTimeRide)}")
                pass 
        else:
            print("invalid action")
            #print("avail actions: ",self.available_actions)
            print("action: ",action)
            print("action space: ",self.action_space)

        
        # refresh travel cost layer after each step
        self.learn_graph.add_travel_cost_layer(self.availableTrips(), self.distance_matrix)
        self.learn_graph.add_remaining_distance_layer(current_hub=self.position, distance_matrix=self.distance_matrix)
        startTimeLearn = time.time()

        old_state = self.state

        self.state = {
        'cost' : ((self.learn_graph.adjacency_matrix('cost')[self.position]-self.mean1)/self.stdev1).astype(np.float64),
        'remaining_distance': ((self.learn_graph.adjacency_matrix('remaining_distance')[self.position]-self.mean2)/self.stdev2).astype(np.float64),
        'current_hub' : self.one_hot(self.position).astype(np.float64), 
        'final_hub' : self.one_hot(self.final_hub).astype(np.float64),
        'observations' : self.learn_graph.adjacency_matrix('distinction')[self.position].astype(np.float64)
        }
        # print("New State: ")        
        # print(self.state)

        self.time += timedelta(seconds=step_duration)

        self.count_actions += 1

        reward, self.done, state_of_delivery = self.compute_reward(action, old_state)
        
        self.state_of_delivery = state_of_delivery
        executionTime = (time.time() - startTime)
        self.update_action_mask()
        print("Masked State Step",self.state)
        return self.state, reward,  self.done, {"timestamp": self.time,"step_travel_time":step_duration,"distance":self.distance_matrix[self.old_position][self.position], "count_hubs":self.count_hubs, "action": self.action_choice, "hub_index": action}

    
    def compute_reward(self, action, old_state):
        old_distinction = old_state['observations']
        cost_of_action = self.learn_graph.adjacency_matrix('cost')[self.old_position][action]
        print(self.old_position, "->", action, cost_of_action)
        self.done = False
        # if delay is greater than 2 hours (=120 minutes), terminate training episode
        if((self.time-self.deadline).total_seconds()/60 >= 120 or self.count_actions>200):
            self.done = True
            reward = -10000
            state_of_delivery = DeliveryState.NOT_DELIVERED
            print("BOX WAS NOT DELIVERED until 2 hours after deadline")
        # if box is delivered to final hub in time
        if (self.position == self.final_hub and self.time <= self.deadline):
            print(f"DELIVERED IN TIME AFTER {self.count_actions} ACTIONS (#wait: {self.count_wait}, #share: {self.count_share}, #book own: {self.count_bookown}")
            reward = 10000
            self.done = True
            state_of_delivery = DeliveryState.DELIVERED_ON_TIME
        # if box is delivered to final hub with delay
        elif(self.position == self.final_hub and (self.time-self.deadline).total_seconds()/60 < 120): #self.time > self.deadline):
            overtime = self.time - self.deadline
            print(f"DELIVERED AFTER {self.count_actions} ACTIONS (#wait: {self.count_wait}, #share: {self.count_share}, #book own: {self.count_bookown} WITH DELAY: {overtime}")
            overtime = round(overtime.total_seconds()/60)
            reward = 10000 - overtime
            self.done = True
            state_of_delivery = DeliveryState.DELIVERED_WITH_DELAY
        # if box is not delivered to final hub
        elif(self.done==False):
            reward = old_distinction[action]*1000
            # print(f"INTERMEDIATE STEP ACTIONS: (#wait: {self.count_wait}, #share: {self.count_share}, #book own: {self.count_bookown}")
            state_of_delivery = DeliveryState.IN_DELIVERY
            #done = False

        print(f"Reward: {reward}")
        print(f"Action: {action}")
        print(f"Old Distinction: {old_distinction}")
        print(f"Rides Mask for Action {action}: {self.shared_rides_mask}")

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
        print(list_trips)

        executionTime = (time.time() - startTime)
        # print('found '+ str(len(list_trips)) +' trips, ' + 'current time: ' + str(self.time))
        return list_trips

    def validateAction(self, action):
        return action < self.n_hubs

    def read_config(self):
        filepath = os.path.join(ROOT_DIR,'env_config.pkl')
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
        current_pos_x = self.manhattan_graph.get_node_by_index(self.position)['x']
        current_pos_y = self.manhattan_graph.get_node_by_index(self.position)['y']
        final_hub_x = self.manhattan_graph.get_node_by_index(self.final_hub)['x']
        final_hub_y = self.manhattan_graph.get_node_by_index(self.final_hub)['y']
        start_hub_x = self.manhattan_graph.get_node_by_index(self.start_hub)['x']
        start_hub_y = self.manhattan_graph.get_node_by_index(self.start_hub)['y']
        
        # Create plot
        plot = ox.plot_graph_folium(self.manhattan_graph.inner_graph,fit_bounds=True, weight=2, color="#333333")


        #Place markers for the random hubs
        for hub in self.hubs:
            hub_node = self.manhattan_graph.get_node_by_nodeid(hub)
            hub_pos_x = hub_node['x']
            hub_pos_y = hub_node['y']
            popup = "HUB %d" % (hub)
            folium.Marker(location=[hub_pos_y, hub_pos_x],popup=popup, icon=folium.Icon(color='orange', prefix='fa', icon='cube')).add_to(plot)

        # Place markers for start, final and current position
        folium.Marker(location=[final_hub_y, final_hub_x], icon=folium.Icon(color='red', prefix='fa', icon='flag-checkered')).add_to(plot)
        folium.Marker(location=[start_hub_y, start_hub_x], popup = f"Pickup time: {self.pickup_time.strftime('%m/%d/%Y, %H:%M:%S')}", icon=folium.Icon(color='lightblue', prefix='fa', icon='caret-right')).add_to(plot)
        folium.Marker(location=[current_pos_y, current_pos_x], popup = f"Current time: {self.time.strftime('%m/%d/%Y, %H:%M:%S')}", icon=folium.Icon(color='lightgreen', prefix='fa',icon='cube')).add_to(plot)
        

        if(visualize_actionspace):
            for i, trip in enumerate(self.availableTrips()):
                target_hub=self.manhattan_graph.get_node_by_nodeid(trip['target_hub'])
                target_node_x = target_hub['x']
                target_node_y = target_hub['y']
                popup = "%s: go to node %d" % (i, trip['target_hub'])
                folium.Marker(location=[target_node_y, target_node_x], popup = popup, tooltip=str(i)).add_to(plot)
                ox.plot_route_folium(G=self.manhattan_graph.inner_graph,route=trip['route'],route_map=plot)
        # Plot
        # pos_to_final = nx.shortest_path(self.manhattan_graph.inner_graph, self.manhattan_graph.get_nodeid_by_index(self.start_hub), self.manhattan_graph.get_nodeid_by_index(self.final_hub), weight="travel_time")
        # if(not len(pos_to_final)< 2):

        return plot

class DeliveryState:
    DELIVERED_ON_TIME, DELIVERED_WITH_DELAY, NOT_DELIVERED, IN_DELIVERY = range(4)

class CustomCallbacks(DefaultCallbacks):
    last_count_delivered_on_time = 0
    last_count_delivered_with_delay = 0
    last_count_not_delivered = 0
    count_not_delivered = 0
    count_delivered_with_delay = 0
    count_delivered_on_time = 0

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

        if (episode.env.state_of_delivery == DeliveryState.DELIVERED_ON_TIME):
            self.count_delivered_on_time +=1
            episode.custom_metrics["count_delivered_on_time"] = self.count_delivered_on_time
        elif (episode.env.state_of_delivery == DeliveryState.DELIVERED_WITH_DELAY):
            self.count_delivered_with_delay +=1
            episode.custom_metrics["count_delivered_with_delay"] = self.count_delivered_with_delay
        elif (episode.env.state_of_delivery == DeliveryState.NOT_DELIVERED):
            self.count_not_delivered +=1
            episode.custom_metrics["count_not_delivered"] = self.count_not_delivered

        # zum Vergleich ohne Abzug später
        # episode.custom_metrics["count_not_delivered_first"] = self.count_not_delivered

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
