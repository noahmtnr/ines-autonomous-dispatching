from xml.dom.pulldom import parseString
import numpy as np
from StreetGraph import StreetGraph
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import MarkerCluster
from datetime import datetime, timedelta
from array import array
from datetime import datetime, timedelta
import gym
from gym.utils import seeding
import random
import pandas as pd
from gym import spaces

class GraphEnv(gym.Env):

    REWARD_AWAY = -1
    REWARD_GOAL = 100
    
    def __init__(self,env_config = None):
        env_config = env_config or {}
        """_summary_

        Args:
            graph (nx.MultiDiGraph): graph
            start_hub (int): nodeId
            final_hub (int): nodeId

        """  
        # self.final_hub = 2
        # self.start_hub = 8
        # self.position = self.start_hub
        
        # pickup_day = 1
        # pickup_hour =  np.random.randint(24)
        # pickup_minute = np.random.randint(60)
        # self.pickup_time = datetime(2022,1,pickup_day,pickup_hour,pickup_minute,0)
        # #self.pickup_time = datetime(2022,1,1,1,1,0)

        # self.time = self.pickup_time
        # self.total_travel_time = 0
        # self.deadline=self.pickup_time+timedelta(hours=3)
        
        self.seed()
        self.reset()

        graph_meinheim=StreetGraph('meinheim')
        graph_meinheim_trips = StreetGraph('meinheim').trips

        self.graph = graph_meinheim
        self.graph.trips = graph_meinheim_trips

        #Creates a list of 5 random hubs
        self.hubs = random.sample(self.graph.nodes(),5) 
        final_hub = self.graph.get_nodeid_by_index(self.final_hub)
        if(final_hub not in self.hubs):
            self.hubs.append(final_hub)
        
        # if self.graph.inner_graph.has_node(self.start_hub):
        #     self.position = self.start_hub
        # else:
        #     return 'Initialized start hub was not found in graph'

        # if self.graph.inner_graph.has_node(final_hub):
        #     self.final_hub = final_hub
        # else:
        #     return 'Initialized final hub was not found in graph'
       

        #self.action_space = gym.spaces.Discrete(num_actions) 
        self.observation_space = gym.spaces.Discrete(len(self.graph.get_nodeids_list())) #num of nodes in the graph

       
    
    def reset(self):
        self.count = 0
        self.final_hub = 3
        self.start_hub = 6
        self.position = self.start_hub

        pickup_day = 1
        pickup_hour =  np.random.randint(24)
        pickup_minute = np.random.randint(60)
        self.pickup_time = datetime(2022,1,pickup_day,pickup_hour,pickup_minute,0)
      
        self.time = self.pickup_time
        self.total_travel_time = 0
        self.deadline=self.pickup_time+timedelta(hours=3)

        self.own_ride = False

        reward=0


        return self.position
    


    @property
    def action_space(self):
            num_actions = len(self.availableActions())
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

        self.count += 1
        done =  False

        old_position = self.graph.get_nodeids_list()[self.position]
        availableActions = self.availableActions()
        step_duration = 0
        print(availableActions)

        if self.validateAction(action):
             if(action == 0):
                 step_duration = 300
                 print("action == wait ")
                 pass
             elif(action==1):
                 self.own_ride = True
                 #create route to final hub
                 route = ox.shortest_path(self.graph.inner_graph, self.graph.get_nodeids_list()[self.position],  self.graph.get_nodeids_list()[self.final_hub], weight='travel_time')
                 route_travel_time = ox.utils_graph.get_route_edge_attributes(self.graph.inner_graph,route,attribute='travel_time')
                 step_duration = sum(route_travel_time)+300 #we add 5 minutes (300 seconds) so the taxi can arrive
                 self.position=self.final_hub
                 print("action ==  ownRide ")
                 pass 

             else:
                selected_trip = availableActions[action]

                if( self.graph.get_nodeids_list()[self.final_hub] in selected_trip['route']):
                    route = selected_trip['route']

                    self.position = self.final_hub
                    index_in_route = route.index( self.graph.get_nodeids_list()[self.final_hub])
                    route_to_final_hub=route[:index_in_route]
                    route_travel_time_to_final_hub = ox.utils_graph.get_route_edge_attributes(self.graph.inner_graph,route_to_final_hub,attribute='travel_time')
                    step_duration = sum(route_travel_time_to_final_hub)

                else:
                    self.position = self.graph.get_nodeids_list().index(selected_trip['target_node'])
                    route_travel_time = ox.utils_graph.get_route_edge_attributes(self.graph.inner_graph,selected_trip['route'],attribute='travel_time')
                    step_duration = sum(route_travel_time)
                
                # Increase global time state by travelled time (does not include waiting yet, in this case it should be +xx seconds)
                self.time = selected_trip['departure_time']

                # Instead of cumulating trip duration here we avel_time 
                # self.total_travel_time += timedelta(seconds=travel_time)
                print("action == ", action, " New Position", self.position)

        self.time += timedelta(seconds=step_duration)

        reward, done = self.compute_reward(done)

        return self.position, reward,  done, {}

    
    def compute_reward(self, done):
        """ Computes the reward for each step
        Args:
            bool: own_ride
        Returns:
            int: reward
            bool: done
        """
        reward = 0
        # if we reach the final hub
        if (self.position == self.final_hub):
            reward = self.REWARD_GOAL
            # if the agent books an own ride, penalize reward by 50
            if self.own_ride:
                reward -= 50
            # if the box is not delivered in time, penalize reward by 1 for every minute over deadline
            if (self.time > self.deadline):
                overtime = self.time-self.deadline
                overtime = round(overtime.total_seconds()/60)
                reward -= overtime
            done = True
        # if we do not reach the final hub, reward is -1
        else:
            reward = self.REWARD_AWAY

        return reward, done



    def availableActions(self):
        """ Returns the available actions at the current position. Uses a simplified action space with moves to all direct neighbors allowed.

        Returns:
            list: list of nodeIds of direct neighbors
        """
        wait = [{'type': 'wait'}]
        ownRide = [{'type': 'ownRide'}]
        available_rides = list(self.availableTrips())
        return [wait,ownRide,*available_rides]

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
        paths=grid['node_timestamps']
        
        for index in range(len(paths)):
            dict = grid['node_timestamps'][index]
            for tupel_position in dict:
                position_timestamp= datetime.strptime(str(dict[tupel_position]), "%Y-%m-%d %H:%M:%S")
                inTimeframe = start_timestamp <= position_timestamp and end_timestamp >= position_timestamp
                startsInCurrentPosition = str(tupel_position) == position_str
                trip_target_node = grid['dropoff_node'][index]
                isNotFinalNode = str(tupel_position) != str(trip_target_node)
                route = grid['route'][index]
                if startsInCurrentPosition and inTimeframe and isNotFinalNode:
                    index_in_route = route.index(position)
                    route_to_target_node=route[index_in_route::]
                    hubsOnRoute = any(node in route_to_target_node for node in self.hubs)
                    
                    # COMMENTED IF OUT UNTIL WE ALLOW MOVES TO ALL HUBS ON THE ROUTE TO FIND MORE TRIPS
                    #if hubsOnRoute:
                    trip = {'departure_time': position_timestamp, 'target_node': trip_target_node, 'route': route_to_target_node}
                    list_trips.append(trip)
        return list_trips

    def validateAction(self, action):
        return action < len(self.availableActions())




        

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



        # Place markers for start, final and current position
        folium.Marker(location=[final_hub_y, final_hub_x], icon=folium.Icon(color='red', prefix='fa', icon='flag-checkered')).add_to(plot)
        folium.Marker(location=[start_hub_y, start_hub_x], popup = f"Pickup time: {self.pickup_time.strftime('%m/%d/%Y, %H:%M:%S')}", icon=folium.Icon(color='lightblue', prefix='fa', icon='caret-right')).add_to(plot)
        folium.Marker(location=[current_pos_y, current_pos_x], popup = f"Current time: {self.time.strftime('%m/%d/%Y, %H:%M:%S')}", icon=folium.Icon(color='lightgreen', prefix='fa',icon='cube')).add_to(plot)
        
        #Place markers for the random hubs
        for hub in self.hubs:
            hub_node = self.graph.get_node_by_nodeid(hub)
            hub_pos_x = hub_node['x']
            hub_pos_y = hub_node['y']
            popup = "HUB %d" % (hub)
            folium.Marker(location=[hub_pos_y, hub_pos_x],popup=popup, icon=folium.Icon(color='orange', prefix='fa', icon='cube')).add_to(plot)

        if(visualize_actionspace):
            for i, trip in enumerate(self.availableTrips()):
                target_node_x = self.graph.nodes[trip['target_node']]['x']
                target_node_y = self.graph.nodes[trip['target_node']]['y']
                popup = "%s: go to node %d" % (i, trip['target_node'])
                folium.Marker(location=[target_node_y, target_node_x], popup = popup, tooltip=str(i)).add_to(plot)

        # Plot
        pos_to_final = nx.shortest_path(self.graph.inner_graph, self.graph.get_nodeid_by_index(self.start_hub), self.graph.get_nodeid_by_index(self.final_hub), weight="travel_time")
        if(not len(pos_to_final)< 2):
            ox.plot_route_folium(G=self.graph.inner_graph,route=pos_to_final,route_map=plot)

        return plot

