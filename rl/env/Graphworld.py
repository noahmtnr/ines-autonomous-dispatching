import numpy as np
import StreetGraph
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import MarkerCluster
from datetime import datetime, timedelta
class Environment:

    def __init__(self, graph: nx.MultiDiGraph, start_hub: int, final_hub: int, pickup_time: datetime): # TODO: add action space -> import gym.spaces -> action_space = Discrete(3)
        """_summary_

        Args:
            graph (nx.MultiDiGraph): graph
            start_hub (int): nodeId
            final_hub (int): nodeId

        """  
        self.graph = graph.graph
        self.graph.trips = graph.trips
        self.time = pickup_time
        self.pickup_time = pickup_time
        self.rel_time = 0

        if self.graph.has_node(start_hub):
            self.start_hub = start_hub
            self.position = start_hub
        else:
            return 'Initialized start hub was not found in graph'

        if self.graph.has_node(final_hub):
            self.final_hub = final_hub
        else:
            return 'Initialized final hub was not found in graph'

    def step(self, action):
        """ Executes an action based on the index passed as a parameter (only works with moves to direct neighbors as of now)

        Args:
            action (int): index of action to be taken from availableActions

        Returns:
            int: new position
            int: new reward
            boolean: isDone
        """
        old_position = self.position
        neighbors = self.availableActions()
        travel_time = 0

        if self.validateAction(action):
            self.position = neighbors[action]
            travel_time= self.graph.edges[(old_position, self.position, 0)]['travel_time']

            # Increase global time state by travelled time (does not include waiting yet, in this case it should be +xx seconds)
            self.time += timedelta(seconds=travel_time)

            # Instead of cumulating trip duration here we return travel_time 
            # self.rel_time += timedelta(seconds=travel_time)
        else:
            pass

        return self.position, self.reward(), travel_time, self.isDone()

    def availableActions(self):
        """ Returns the available actions at the current position. Uses a simplified action space with moves to all direct neighbors allowed.

        Returns:
            list: list of nodeIds of direct neighbors
        """
        neighbors = list(self.graph.neighbors(self.position))
        return neighbors

    def availableTrips(self):
        """ Returns a list of all available trips at the current node and within the next 5 minutes. Includes the time of departure from the current node as well as the target node of the trip.

        Returns:
            list: [departure_time,target_node]
        """
        position=str(self.position)
        start_timestamp=self.time
        time_window=5
        end_timestamp = self.time + timedelta(minutes=time_window)
        grid=self.graph.trips
        list=[]
        paths=grid['node_timestamps']
        
        for index in range(len(paths)):
            dict = grid['node_timestamps'][index]
            for tupel_position in dict:
                position_timestamp= datetime.strptime(dict[tupel_position], "%Y-%m-%d %H:%M:%S")
                inTimeframe = start_timestamp <= position_timestamp and end_timestamp >= position_timestamp
                samePosition = str(tupel_position) == str(position)
                isNotFinalNode = str(tupel_position) != grid['dropoff_node'][index]

                if samePosition and inTimeframe and isNotFinalNode:
                   list.append([position_timestamp,grid['dropoff_node'][index]])
                   # TODO slice and retrun dictionary in order to get only the route to the final node from the current node
        return list

    def validateAction(self, action):
        return action < len(self.availableActions())

    def isDone(self):
        return self.position == self.final_hub
    
    def reward(self): # TODO: extend function: should not return 0 reward if position is a second time on start_hub
        if self.isDone():
            return 10
        elif self.position == self.start_hub: 
            return 0
        else:
            return -1

    def render(self, visualize_actionspace: bool = False):
        """_summary_

        Args:
            visualize_actionspace (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        current_pos_x = self.graph.nodes[self.position]['x']
        current_pos_y = self.graph.nodes[self.position]['y']
        final_hub_x = self.graph.nodes[self.final_hub]['x']
        final_hub_y = self.graph.nodes[self.final_hub]['y']
        start_hub_x = self.graph.nodes[self.start_hub]['x']
        start_hub_y = self.graph.nodes[self.start_hub]['y']

        # Create plot
        plot = ox.plot_graph_folium(self.graph,fit_bounds=True, weight=2, color="#333333")

        # Place markers for start, final and current position
        folium.Marker(location=[final_hub_y, final_hub_x], icon=folium.Icon(color='red', prefix='fa', icon='flag-checkered')).add_to(plot)
        folium.Marker(location=[start_hub_y, start_hub_x], popup = f"Pickup time: {self.pickup_time.strftime('%m/%d/%Y, %H:%M:%S')}", icon=folium.Icon(color='lightblue', prefix='fa', icon='caret-right')).add_to(plot)
        folium.Marker(location=[current_pos_y, current_pos_x], popup = f"Current time: {self.time.strftime('%m/%d/%Y, %H:%M:%S')}", icon=folium.Icon(color='lightgreen', prefix='fa',icon='cube')).add_to(plot)

        if(visualize_actionspace):
            for i, target_node in enumerate(self.availableActions()):
                target_node_x = self.graph.nodes[target_node]['x']
                target_node_y = self.graph.nodes[target_node]['y']
                popup = "%s: go to node %d" % (i, target_node)
                folium.Marker(location=[target_node_y, target_node_x], popup = popup, tooltip=str(i)).add_to(plot)

        # Plot
        pos_to_final = nx.shortest_path(self.graph, self.position, self.final_hub, weight="travel_time")
        print(pos_to_final)
        if(not len(pos_to_final)< 2):
            ox.plot_route_folium(G=self.graph,route=pos_to_final,route_map=plot)

        return plot

    def reset(self):
        self.position = self.start_hub
        self.time = self.pickup_time
        self.rel_time = 0
        pass
