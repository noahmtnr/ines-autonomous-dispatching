import osmnx as ox
import pandas as pd
import random

class ManhattenGraph:

    def __init__(self, filename, num_hubs):
        filepath = ("../../data/graph/%s.graphml") % (filename)
        self.inner_graph = ox.load_graphml(filepath)
        self.inner_graph = ox.add_edge_speeds(self.inner_graph,fallback=30)
        self.inner_graph = ox.add_edge_travel_times(self.inner_graph)
        ox.utils_graph.remove_isolated_nodes(self.inner_graph)
        fin_hub = random.sample(self.nodes(),1)
        self.generate_hubs(fin_hub, num_hubs)
        self.generate_random_trips()


    def generate_hubs(self, fin_hub, num_hubs: int = 5):
        """Generates random bubs within the graph

        Args:
            fin_hub (int): index_id of final hub
            num_hubs (int, optional): Number of hubs to create. Defaults to 5.

        Returns:
            self.hubs(list): List of hubs in graph
        """
        random.seed(42)
        hubs = random.sample(self.nodes(),num_hubs) 
        #final_hub = self.get_nodeid_by_index(fin_hub)
        if(fin_hub not in hubs):
            hubs.append(fin_hub)
        self.hubs = hubs

        return self.hubs

    def generate_random_trips(self):
        """Read trips information of Kaggle file in self.trips.

        Args:
            n (int, optional): Number of trips to be generated. Defaults to 2000.
        """
        
        self.trips = pd.read_csv('../../data/trips/trips_with_routes_timestamps.csv')

        return self.trips

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

