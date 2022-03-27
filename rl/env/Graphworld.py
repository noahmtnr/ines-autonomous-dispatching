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

graph_meinheim=StreetGraph('meinheim')
graph_meinheim_trips = StreetGraph('meinheim').trips

class GraphEnv(gym.Env):

    MAX_STEPS = 30000
    REWARD_AWAY = -1
    REWARD_GOAL = MAX_STEPS
    
    def __init__(self,env_config = None):
        env_config = env_config or {}
        """_summary_

        Args:
            graph (nx.MultiDiGraph): graph
            start_hub (int): nodeId
            final_hub (int): nodeId

        """  
        self.final_hub = 3
        self.start_hub = 6
        self.position = self.start_hub
        
        reward=0
        
        pickup_day = 1
        pickup_hour =  np.random.randint(24)
        pickup_minute = np.random.randint(60)
        self.pickup_time = datetime(2022,1,pickup_day,pickup_hour,pickup_minute,0)
        #self.pickup_time = datetime(2022,1,1,1,1,0)

        self.time = self.pickup_time
        self.total_travel_time = 0


        self.graph = graph_meinheim
        self.graph.trips = graph_meinheim_trips


        
        self.seed()
        self.reset()

        # if self.graph.has_node(self.start_hub):
        #     self.position = self.start_hub
        # else:
        #     return 'Initialized start hub was not found in graph'

        # if self.graph.has_node(final_hub):
        #     self.final_hub = final_hub
        # else:
        #     return 'Initialized final hub was not found in graph'
       

        #self.action_space = gym.spaces.Discrete(num_actions) 
        self.observation_space = gym.spaces.Discrete(len(list(self.graph.graph.nodes()))) #num of nodes in the graph
    
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
        print(self.position)
        print(self.availableActions())
        self.count += 1
        done = self.count >= self.MAX_STEPS

        old_position = list(self.graph.graph.nodes())[self.position]
        availableActions = self.availableActions()
        step_duration = 0

        if self.validateAction(action):
             if(action == 0):
                 step_duration = 300
                 print("action == 0 ")
                 pass
             else:
                selected_trip = availableActions[action-1]

                if( list(self.graph.graph.nodes())[self.final_hub] in selected_trip['route']):
                    route = selected_trip['route']

                    self.position = self.final_hub
                    index_in_route = route.index( list(self.graph.graph.nodes())[self.final_hub])
                    route_to_final_hub=route[:index_in_route]
                    #print(route_to_final_hub)
                    #print(route)
                    print(self.final_hub)
                    route_travel_time_to_final_hub = ox.utils_graph.get_route_edge_attributes(self.graph.graph,route_to_final_hub,attribute='travel_time')
                    step_duration = sum(route_travel_time_to_final_hub)
                    print("final node: ",self.position)

                else:
                    self.position = list(self.graph.graph.nodes()).index(selected_trip['target_node'])
                    route_travel_time = ox.utils_graph.get_route_edge_attributes(self.graph.graph,selected_trip['route'],attribute='travel_time')
                    step_duration = sum(route_travel_time)
                    print("not final node: ",self.position)
                
                # Increase global time state by travelled time (does not include waiting yet, in this case it should be +xx seconds)
                self.time = selected_trip['departure_time']

                # Instead of cumulating trip duration here we return travel_time 
                # self.total_travel_time += timedelta(seconds=travel_time)

        self.time += timedelta(seconds=step_duration)
            #else:
             #   pass
            #reward function

        if (self.position == self.final_hub):
                reward = self.REWARD_GOAL
                done = True
        else:
                reward = self.REWARD_AWAY

        

        return self.position, reward,  done, {}

    def availableActions(self):
        """ Returns the available actions at the current position. Uses a simplified action space with moves to all direct neighbors allowed.

        Returns:
            list: list of nodeIds of direct neighbors
        """
        rides = list(self.availableTrips())
        return rides

    def availableTrips(self, time_window=5):
        """ Returns a list of all available trips at the current node and within the next 5 minutes. Includes the time of departure from the current node as well as the target node of the trip.

        Returns:
            list: [departure_time,target_node]
        """
        list_trips=[]
        time_window=0
        while(len(list_trips)==0):
            time_window+=5
            position=list(self.graph.graph.nodes())[self.position]
            position_str=str(list(self.graph.graph.nodes())[self.position])
            start_timestamp=self.time
            end_timestamp = self.time + timedelta(minutes=time_window)
            grid=self.graph.trips

            final_hub_postion=list(self.graph.graph.nodes())[self.final_hub]
        
            paths=grid['node_timestamps']
            
            for index in range(len(paths)):
                dict = grid['node_timestamps'][index]
                for tupel_position in dict:
                    position_timestamp= datetime.strptime(str(dict[tupel_position]), "%Y-%m-%d %H:%M:%S")
                    inTimeframe = start_timestamp <= position_timestamp and end_timestamp >= position_timestamp
                    startsInCurrentPosition = str(tupel_position) == position_str
                    trip_target_node = grid['dropoff_node'][index]
                    isNotFinalNode = str(tupel_position) != str(trip_target_node)
                    

                    if startsInCurrentPosition and inTimeframe and isNotFinalNode:
                        route = grid['route'][index]
                        index_in_route = route.index(position)
                        route_to_target_node=route[index_in_route::]
                        #if final_hub_postion in route_to_target_node:
                        trip = {'departure_time': position_timestamp, 'target_node': trip_target_node, 'route': route_to_target_node}
                        list_trips.append(trip)


        return list_trips
    

    def validateAction(self, action):
        return action < len(self.availableTrips()) + 1

    
        

    def render(self, visualize_actionspace: bool = False):
        """_summary_

        Args:
            visualize_actionspace (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        current_pos_x = self.graph.nodes[list(self.graph.graph.nodes())[self.position]]['x']
        current_pos_y = self.graph.nodes[list(self.graph.graph.nodes())[self.position]]['y']
        final_hub_x = self.graph.nodes[list(self.graph.graph.nodes())[self.final_hub]]['x']
        final_hub_y = self.graph.nodes[list(self.graph.graph.nodes())[self.final_hub]]['y']
        start_hub_x = self.graph.nodes[list(self.graph.graph.nodes())[self.start_hub]]['x']
        start_hub_y = self.graph.nodes[list(self.graph.graph.nodes())[self.start_hub]]['y']

        # Create plot
        plot = ox.plot_graph_folium(self.graph,fit_bounds=True, weight=2, color="#333333")

        # Place markers for start, final and current position
        folium.Marker(location=[final_hub_y, final_hub_x], icon=folium.Icon(color='red', prefix='fa', icon='flag-checkered')).add_to(plot)
        folium.Marker(location=[start_hub_y, start_hub_x], popup = f"Pickup time: {self.pickup_time.strftime('%m/%d/%Y, %H:%M:%S')}", icon=folium.Icon(color='lightblue', prefix='fa', icon='caret-right')).add_to(plot)
        folium.Marker(location=[current_pos_y, current_pos_x], popup = f"Current time: {self.time.strftime('%m/%d/%Y, %H:%M:%S')}", icon=folium.Icon(color='lightgreen', prefix='fa',icon='cube')).add_to(plot)

        if(visualize_actionspace):
            for i, trip in enumerate(self.availableTrips()):
                target_node_x = self.graph.nodes[trip['target_node']]['x']
                target_node_y = self.graph.nodes[trip['target_node']]['y']
                popup = "%s: go to node %d" % (i, trip['target_node'])
                folium.Marker(location=[target_node_y, target_node_x], popup = popup, tooltip=str(i)).add_to(plot)

        # Plot
        pos_to_final = nx.shortest_path(self.graph, self.position, self.final_hub, weight="travel_time")
        if(not len(pos_to_final)< 2):
            ox.plot_route_folium(G=self.graph,route=pos_to_final,route_map=plot)

        return plot

    def reset(self):
        self.count = 0
        self.final_hub = 3
        self.start_hub = 6
        self.position = self.start_hub
        
        pickup_day = 1
        pickup_hour =  np.random.randint(24)
        pickup_minute = np.random.randint(60)
        
        self.pickup_time = datetime(2022,1,pickup_day,pickup_hour,pickup_minute,0)
        #self.pickup_time = datetime(2022,1,1,1,1,0)
        self.time = self.pickup_time
        self.total_travel_time = 0


        return self.position
    
