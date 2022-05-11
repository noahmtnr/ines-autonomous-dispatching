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

from ManhattanGraph import ManhattanGraph
from LearnGraph import LearnGraph
from OneHotVector import OneHotVector

class GraphEnv(gym.Env):

    REWARD_AWAY = -1
    REWARD_GOAL = 100
    pickup_day = 1
    pickup_hour =  np.random.randint(24)
    pickup_minute = np.random.randint(60) 
    START_TIME = datetime(2016,1,pickup_day,pickup_hour,pickup_minute,0)
    
    def __init__(self ,env_config = None):
        """_summary_

        Args:
            graph (nx.MultiDiGraph): graph
            start_hub (int): nodeId
            final_hub (int): nodeId

        """  
        self.env_config = env_config or {}

        # Creates an instance of StreetGraph with random trips and hubs
        # graph_meinheim = StreetGraph(filename='meinheim', num_trips=4000, fin_hub=self.final_hub, num_hubs=5)

        # manhattan_graph = ManhattanGraph(filename='simple', num_hubs=70)
        # manhattan_graph.setup_trips(self.START_TIME)

        self.hubs = manhattan_graph.hubs
        learn_graph = LearnGraph(n_hubs=self.n_hubs)

        self.graph = learn_graph

        # self.manhattan_graph = ManhattanGraph(filename='simple', num_hubs=70)
        # self.manhattan_graph.setup_trips(self.START_TIME)
        self.hubs = self.manhattan_graph.hubs
        self.graph = self.manhattan_graph

        self.state = None

        # Creates an instance of StreetGraph with random trips and hubs, just used for feeding the learn graph with real taxi trips later on
        # self.manhattan_graph = ManhattanGraph(filename='simple', num_hubs=70)
        # self.manhattan_graph.setup_trips(self.START_TIME)
        # self.hubs = manhattan_graph.hubs

        self.n_hubs = 70


        learn_graph = LearnGraph(n_hubs=self.n_hubs)

        self.graph = learn_graph
      
        self.action_space = gym.spaces.Discrete(self.n_hubs) 
        
        self.observation_space = spaces.Dict(dict(
            #layer_one = spaces.Box(low=0, high=100, shape=(1,self.n_hubs), dtype=np.int32),
            layer_one = gym.spaces.Discrete(self.n_hubs),
            current_hub = OneHotVector(self.n_hubs),
            final_hub = OneHotVector(self.n_hubs)
        ))

    
    def reset(self):
        # two cases depending if we have env config 
       
        if (self.env_config == {}):
            self.final_hub = self.graph.get_nodeids_list().index(random.sample(self.hubs,1)[0])
            self.start_hub = self.graph.get_nodeids_list().index(random.sample(self.hubs,1)[0])
            self.position = self.start_hub

        # time for pickup
            self.pickup_time = self.START_TIME
            self.time = self.pickup_time
            self.total_travel_time = 0
            self.deadline=self.pickup_time+timedelta(hours=12)
            self.current_wait = 1 ## to avoid dividing by 0
            self.manhattan_graph.setup_trips(self.START_TIME)
        else:
            
            self.final_hub= self.graph.get_index_by_nodeid(self.env_config['delivery_node_id'])
            self.start_hub= self.graph.get_index_by_nodeid(self.env_config['pickup_node_id'])
            self.position = self.start_hub

            self.pickup_time = self.env_config['pickup_timestamp']
            self.time = self.pickup_time
            self.total_travel_time = 0
            self.deadline=self.env_config['delivery_timestamp']
            self.current_wait = 0
            self.manhattan_graph.setup_trips(self.pickup_time)


        self.count_hubs = 0
        # old position is current position
        self.old_position = self.start_hub
        # current trip
        self.current_trip = None


        self.own_ride = False
        self.has_waited=False
        reward=0
        self.available_actions = self.get_available_actions()

        self.state = (self.learn_graph.adjacency_matrix('cost')[self.position],one_hot(self.position),one_hot(self.final_hub))
        return self.state

    def one_hot(pos):
        one_hot_vector = np.zeros(len(self.hubs))
        one_hot_vector[pos] = 1
        return one_hot_vector

    @property
    def action_space(self):
            num_actions = len(self.available_actions)
            return gym.spaces.Discrete(num_actions) 

    
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

        done =  False

        # set old position to current position before changing current position
        self.old_position = self.position
        availableActions = self.available_actions
        step_duration = 0

        if self.validateAction(action):
            if(action == self.position):
                step_duration = 300
                self.has_waited=True
                self.own_ride = False
                print("action == wait ")
                self.state = (self.learn_graph.adjacency_matrix('cost')[self.position],one_hot(self.position),one_hot(self.final_hub))
                pass
            else
                self.has_waited=False
                self.count_hubs += 1
                self.own_ride = True
                #create route to final hub
                route = ox.shortest_path(self.graph.inner_graph, self.graph.get_nodeids_list()[self.position],  self.graph.get_nodeids_list()[self.final_hub], weight='travel_time')
                route_travel_time = ox.utils_graph.get_route_edge_attributes(self.graph.inner_graph,route,attribute='travel_time')
                step_duration = sum(route_travel_time)+300 #we add 5 minutes (300 seconds) so the taxi can arrive
                self.old_position = self.position
                self.position=self.final_hub
                print("action ==  ownRide ")
                self.state = (self.learn_graph.adjacency_matrix('cost')[self.position],one_hot(self.position),one_hot(self.final_hub))
                pass 

            else:
                print("action ==  take taxi trip ")
                self.has_waited=False
                self.own_ride = False
                self.count_hubs += 1
                selected_trip = availableActions[action]
                self.current_trip = selected_trip

                if( self.graph.get_nodeids_list()[self.final_hub] in selected_trip['route']):
                    route = selected_trip['route']

                    self.position = self.final_hub
                    index_in_route = route.index( self.graph.get_nodeids_list()[self.final_hub])
                    route_to_final_hub=route[:index_in_route]
                    route_travel_time_to_final_hub = ox.utils_graph.get_route_edge_attributes(self.graph.inner_graph,route_to_final_hub,attribute='travel_time')
                    step_duration = sum(route_travel_time_to_final_hub)

                else:
                    self.position = self.graph.get_nodeids_list().index(selected_trip['target_hub'])
                    route_travel_time = ox.utils_graph.get_route_edge_attributes(self.graph.inner_graph,selected_trip['route'],attribute='travel_time')
                    step_duration = sum(route_travel_time)
                
                # Increase global time state by the time waited for the taxi to arrive at our location
                self.current_wait = (selected_trip['departure_time']-self.time).seconds
                self.time = selected_trip['departure_time']

                # Instead of cumulating trip duration here we add travel_time 
                # self.total_travel_time += timedelta(seconds=travel_time)
                print("action == ", action, " New Position", self.position)     
                
                self.state = (self.learn_graph.adjacency_matrix('cost')[self.position],one_hot(self.position),one_hot(self.final_hub))
        else:
            print("invalid action")
            print("avail actions: ",self.available_actions)
            print("action: ",action)
            print("action space: ",self.action_space)


        self.time += timedelta(seconds=step_duration)

        reward, done = self.compute_reward(done)

        executionTime = (time.time() - startTime)
        print('Step() Execution time: ' + str(executionTime) + ' seconds')

        return self.state, reward,  done, {}
        
    
    def compute_reward(self, done):
        """ Computes the reward for each step
        Args:
            bool: own_ride
        Returns:
            int: reward
            bool: done
        """
        startTime = time.time()

        # define weights for reward components
        # for the start: every component has the same weight
        num_comp = 6
        w = 1/num_comp
        # later: every component individually
        # w1 = ?, w2 = ?, ...

        reward = 0
        # if we reach the final hub
        if (self.position == self.final_hub):
            reward += self.REWARD_GOAL
            # if the agent books an own ride, penalize reward by 50
            #if self.own_ride:
            #    reward -= 80
            # if the box is not delivered in time, penalize reward by 1 for every minute over deadline
            #if (self.time > self.deadline):
            #    overtime = self.time-self.deadline
            #    overtime = round(overtime.total_seconds()/60)
            #    reward -= overtime
            done = True

        # if we do not reach the final hub, reward is -1
        else:
            # old reward: reward = self.REWARD_AWAY
            # 

            # if action was own ride
            if self.own_ride:
                 # punishment is the price for own ride and the emission costs for the distance
                path_travelled = ox.shortest_path(self.graph.inner_graph, self.graph.get_nodeids_list()[self.old_position],  self.graph.get_nodeids_list()[self.position], weight='travel_time')
                dist_travelled_list = ox.utils_graph.get_route_edge_attributes(self.graph.inner_graph,path_travelled,attribute='length')
                part_length = sum(dist_travelled_list)
                dist_travelled = 1000/part_length
                # choose random mobility provider (hereby modelling random availability for an own ride) and calculate trip costs
                providers = pd.read_csv("Provider.csv")
                id = random.randint(0,len(providers.index))
                price = providers['basic_cost'][id] + dist_travelled/1000 * providers['cost_per_km'][id]
                reward += dist_travelled - price

            # if action was wait
            elif (self.has_waited):
                # punishment is the time wasted on waiting relative to overall time that is available
                if self.time > self.deadline:
                    reward = self.deadline - self.time
                else:
                    reward -= 300/((self.deadline-self.pickup_time).total_seconds())

            # if action was to share ride
            else:
                df_id = self.current_trip['trip_row_id']
                # maximize difference between time constraint and relative travelled time
                time_diff = (self.deadline - self.time).seconds

                # minimize route distance travelled (calculate length of current (part-)trip and divide by total length of respective trip from dataframe)
                path_travelled = ox.shortest_path(self.graph.inner_graph, self.graph.get_nodeids_list()[self.old_position],  self.graph.get_nodeids_list()[self.position], weight='travel_time')
                dist_travelled_list = ox.utils_graph.get_route_edge_attributes(self.graph.inner_graph,path_travelled,attribute='length')
                part_length = sum(dist_travelled_list)
                dist_travelled = 1000/part_length

                # minimize distance to final hub (calculate difference between distance to final hub from the old position and the new position)
                oldpath_to_final = ox.shortest_path(self.graph.inner_graph, self.graph.get_nodeids_list()[self.old_position],  self.graph.get_nodeids_list()[self.final_hub], weight='travel_time')
                newpath_to_final = ox.shortest_path(self.graph.inner_graph, self.graph.get_nodeids_list()[self.position],  self.graph.get_nodeids_list()[self.final_hub], weight='travel_time')
                olddist_tofinal = sum(ox.utils_graph.get_route_edge_attributes(self.graph.inner_graph,oldpath_to_final,attribute='length'))
                newdist_tofinal = sum(ox.utils_graph.get_route_edge_attributes(self.graph.inner_graph,newpath_to_final,attribute='length'))
                dist_gained = olddist_tofinal - newdist_tofinal

                # minimize number of hops
                hops = -self.count_hubs

                # minimize waiting time (time between choosing ride and being picked up) 
                wait_time = 1/self.current_wait

                # minimize costs
                # get total route length
                total_length = self.graph.trips['route_length'][df_id]
                # get part route length: access variable part_length
                # calculate proportion
                prop = part_length/total_length
                # calculate price for this route part
                price = self.graph.trips['total_price'][df_id]*prop
                # if path length in dataframe is the same as path length of current trip -> get total costs, otherwise calculate partly costs

                passenger_num = self.graph.trips['passenger_count'][df_id]
                cost = -price/(passenger_num+1)
                # in the case of multiagent: cost = price(passenger_num+box_num+1)

                # add all to reward
                reward += w * time_diff + w * dist_travelled + w * dist_gained + w * hops + w * wait_time + w * cost
                # later: take into account each individual weight
                # reward += w1 * time_diff + w2 * dist + w3 * hops + w4 * wait_time + w5 * cost

        executionTime = (time.time() - startTime)
        print('Compute_reward() Execution time: ' + str(executionTime) + ' seconds')
        return reward, done



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
        print('get_available_actions() Execution time: ' + str(executionTime) + ' seconds')

        available_actions = [wait,ownRide,*available_rides]
        self.available_actions = available_actions
        return available_actions

    def availableTrips(self, time_window=5):
        """ Returns a list of all available trips at the current node and within the next 5 minutes. Includes the time of departure from the current node as well as the target node of the trip.

        Returns:
            list: [departure_time,target_node]
        """
        list_trips=[]
        position=self.graph.get_nodeid_by_index(self.position)
        position_str=str(position)
        final_hub_postion=self.graph.get_nodeid_by_index(self.final_hub)

        start_timestamp=self.time
        end_timestamp = self.time + timedelta(minutes=time_window)

        grid=self.graph.trips
        paths=grid['route_timestamps']
        
        row_id = -1
        for index in range(len(paths)):
            row_id += 1
            dict_route = grid['route_timestamps'][index]
            dict_route= eval(dict_route)
            for tupel_position in dict_route:
                startsInCurrentPosition = str(tupel_position) == position_str
                if(startsInCurrentPosition):
                    position_timestamp= datetime.strptime(str(dict_route[tupel_position]), "%Y-%m-%d %H:%M:%S")
                    inTimeframe = start_timestamp <= position_timestamp and end_timestamp >= position_timestamp
                    if inTimeframe:
                        trip_target_node = grid['dropoff_node'][index]
                        isNotFinalNode = str(tupel_position) != str(trip_target_node)
                        if(isNotFinalNode):
                            string_split = grid['route'][index].replace('[','').replace(']','').split(',')
                            route = [int(el) for el in string_split]
                            index_in_route = route.index(position)
                            route_to_target_node=route[index_in_route::]
                            hubsOnRoute = any(node in route_to_target_node for node in self.hubs)
                            if hubsOnRoute:
                                list_hubs = [node for node in self.hubs if node in route_to_target_node]
                                hubs_dict = dict((node, dict_route[node]) for node in list_hubs)
                                for hub in hubs_dict:
                                    index_hub_in_route = route.index(hub)
                                    index_hub_in_route += 1
                                    route_to_target_hub = route[index_in_route:index_hub_in_route]
                                    if(hub != position):
                                        trip = {'departure_time': position_timestamp, 'target_hub': hub, 'route': route_to_target_hub, 'trip_row_id': index}
                                        list_trips.append(trip)
        self.available_actions = list_trips
        print(list_trips)
        return list_trips

    def validateAction(self, action):
        return action < self.n_hubs




        

    def render(self, visualize_actionspace: bool = False):
        """_summary_

        Args:
            visualize_actionspace (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        current_pos_x = self.graph.get_node_by_index(self.position)['x']
        current_pos_y = self.graph.get_node_by_index(self.position)['y']
        final_hub_x = self.graph.get_node_by_index(self.final_hub)['x']
        final_hub_y = self.graph.get_node_by_index(self.final_hub)['y']
        start_hub_x = self.graph.get_node_by_index(self.start_hub)['x']
        start_hub_y = self.graph.get_node_by_index(self.start_hub)['y']
        
        # current_pos_x = list(self.graph.nodes())[self.position]['x']
        # current_pos_y = list(self.graph.nodes())[self.position]['y']
        # final_hub_x = list(self.graph.nodes())[self.final_hub]['x']
        # final_hub_y = list(self.graph.nodes())[self.final_hub]['y']
        # start_hub_x = list(self.graph.nodes())[self.start_hub]['x']
        # start_hub_y = list(self.graph.nodes())[self.start_hub]['y']
        
        # Create plot
        plot = ox.plot_graph_folium(self.graph.inner_graph,fit_bounds=True, weight=2, color="#333333")


        #Place markers for the random hubs
        for hub in self.hubs:
            hub_node = self.graph.get_node_by_nodeid(hub)
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
                target_hub=self.graph.get_node_by_nodeid(trip['target_hub'])
                target_node_x = target_hub['x']
                target_node_y = target_hub['y']
                popup = "%s: go to node %d" % (i, trip['target_hub'])
                folium.Marker(location=[target_node_y, target_node_x], popup = popup, tooltip=str(i)).add_to(plot)
                ox.plot_route_folium(G=self.graph.inner_graph,route=trip['route'],route_map=plot)
        # Plot
        # pos_to_final = nx.shortest_path(self.graph.inner_graph, self.graph.get_nodeid_by_index(self.start_hub), self.graph.get_nodeid_by_index(self.final_hub), weight="travel_time")
        # if(not len(pos_to_final)< 2):

        return plot

