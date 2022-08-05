import os
from datetime import datetime, timedelta

import osmnx as ox
import pandas as pd

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))

#graph to be used: full.graphml (all nodes)
#if we use small_manhattan.graphml, we do not have all nodes which are in the trips and then we get Key Error
class ManhattanGraph:

    def __init__(self, filename, hubs):
        # filepath = os.path.join('data', 'graph', ("%s.graphml") % (filename))
        #filepath = "/Users/noah/Desktop/Repositories/ines-autonomous-dispatching/data/graph/simple.graphml"
        # filepath = "D:/ines-autonomous-dispatching/data/graph/simple.graphml"
        filepath = os.path.join(ROOT_DIR, 'data', 'graph', ("%s.graphml") % (filename))
        # filepath = "/Users/noah/Desktop/Repositories/ines-autonomous-dispatching/data/graph/simple.graphml"
        #filepath = "./data/graph/simple.graphml"

        self.inner_graph = ox.load_graphml(filepath)
        self.inner_graph = ox.add_edge_speeds(self.inner_graph,fallback=30)
        self.inner_graph = ox.add_edge_travel_times(self.inner_graph)
        ox.utils_graph.remove_isolated_nodes(self.inner_graph)
        self.hubs = hubs  

    def setup_trips(self, start_time: datetime):
        """Read trips information of Kaggle file in self.trips.

        Args:
            n (int, optional): Number of trips to be generated. Defaults to 2000.
        """
        
        #trips for simple graph, only the first 5000 rows
        filepath = os.path.join(ROOT_DIR, 'data', 'trips', 'preprocessed_trips.csv')
        #filepath = "data/trips/preprocessed_trips.csv"
        all_trips = pd.read_csv(filepath)
        self.trips = self.prefilter_trips(all_trips, start_time).reset_index(drop=True)
        route_length_column=[]
        for i in self.trips.index:
            current_route_string = self.trips.iloc[i]["route"]
            string_split = current_route_string.replace('[','').replace(']','').split(',')
            current_route = [int(el) for el in string_split]
            route_length = 0
            for j in range(len(current_route)-1):
                route_length += ox.distance.great_circle_vec(self.inner_graph.nodes()[current_route[j]]['y'], self.inner_graph.nodes()[current_route[j]]['x'],
                self.inner_graph.nodes()[current_route[j+1]]['y'], self.inner_graph.nodes()[current_route[j+1]]['x'])
            route_length_column.append(route_length)

        self.trips["route_length"]=route_length_column

        # add mobility providers randomly
        provider_column=[]
        totalprice_column=[]
        filepath = os.path.join(ROOT_DIR, 'data', 'others', 'Provider.csv')
        providers = pd.read_csv(filepath)
        for i in self.trips.index:
            provider_id = providers['id'].sample(n=1).iloc[0]
            provider_column.append(provider_id)
            selected_row = providers[providers['id']==provider_id]
            basic_price = selected_row['basic_cost']
            km_price = selected_row['cost_per_km']
            leng = self.trips.iloc[0]['route_length']
            # note that internal distance unit is meters in OSMnx
            total_price = basic_price + km_price*leng/1000
            totalprice_column.append(total_price.iloc[0])
        self.trips["provider"]=provider_column
        self.trips["total_price"]=totalprice_column


        self.trips.to_csv("trips_kaggle_providers.csv")

        return self.trips

    def prefilter_trips(self, all_trips, start_time: datetime):
        BOTTOM = str(start_time - timedelta(hours=2))
        TOP = str(start_time + timedelta(hours=24))
        return all_trips[(all_trips['pickup_datetime'] >= BOTTOM)&(all_trips['pickup_datetime'] <= TOP)]
    
    def nodes(self):
        return self.inner_graph.nodes()

    def edges(self):
        return self.inner_graph.edges()

    def get_nodeids_list(self):
        return list(self.nodes())

    def get_node_by_nodeid(self, nodeid: int):
        return self.nodes()[nodeid]

    def get_node_by_index(self, index: int):
        return self.get_node_by_nodeid(self.get_nodeids_list()[index])

    def get_nodeid_by_index(self, index: int):
        return self.get_nodeids_list()[index]

    def get_index_by_nodeid(self, nodeid: int):
        return self.get_nodeids_list().index(nodeid)
    
    def get_coordinates_of_node(self, node_id): 
        nodes = self.inner_graph.nodes()
        return [nodes[node_id]['x'], nodes[node_id]['y']]
        
    def get_nodeid_by_hub_index(self, hub_index: int):
        return self.hubs[hub_index] 

    def get_hub_index_by_nodeid(self, nodeid: int):
        if(nodeid in self.hubs):
            return self.hubs.index(nodeid)
        return ' '

    def get_hub_index_by_node_index(self, node_index: int):
        return self.get_hub_index_by_nodeid(self.get_nodeid_by_index(node_index))

    def get_nodeid_by_hub_index(self, hub_index: int):
        return self.hubs[hub_index]

    def get_node_by_hub_index(self, hub_index: int):
        return self.get_node_by_nodeid(self.get_nodeid_by_hub_index(hub_index))

    def get_node_index_by_hub_index(self, hub_index: int):
        return self.get_index_by_nodeid(self.get_nodeid_by_hub_index(hub_index))
    
    def get_coordinates_of_node(self, node_id): 
        nodes = self.inner_graph.nodes()
        return [nodes[node_id]['x'], nodes[node_id]['y']]
