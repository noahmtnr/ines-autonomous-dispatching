import sys
sys.path.insert(0,"../..")

import osmnx as ox
import networkx as nx
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

from preprocessing.data_preprocessing import DataPreProcessing

class StreetGraph:

    def __init__(self, filename):
        filepath = ("../../data/graph/%s.graphml") % (filename)
        self.inner_graph = ox.load_graphml(filepath)
        self.inner_graph = ox.add_edge_speeds(self.inner_graph,fallback=30)
        self.inner_graph = ox.add_edge_travel_times(self.inner_graph)
        ox.utils_graph.remove_isolated_nodes(self.inner_graph)
        self.generate_random_trips(4000)


    def generate_random_trips(self, n: int = 2000):
        """Generates random trips within the graph and stores them in self.trips. The trips are randomly spread across January 2022.

        Args:
            n (int, optional): Number of trips to be generated. Defaults to 1000.
        """
        random.seed(42)

        graph = self.inner_graph

        gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)

        df = pd.DataFrame(gdf_nodes)
        df.reset_index(inplace=True)

        sequence = df["osmid"]
        sequence = sequence.to_numpy()
        sequence = sequence.tolist()

        random_nodes = random.choices(sequence, k=n)
        random_nodes2 = random.choices(sequence, k=n)

        trips = pd.DataFrame()
        count = 0
        trip_id = []
        nodeA = []
        nodeB = []
        for i in range(n):
            if random_nodes[i]==random_nodes2[i]:
                random_nodes2[i]=random.choice(sequence)
            trip_id.append(i)
        
        trips["tripid"] = trip_id
        trips["pickup_node"] = random_nodes
        trips["dropoff_node"] = random_nodes2

        pickup_day = [1 for i in range(n)]
        pickup_hour =  np.random.randint(24, size=n)
        pickup_minute = np.random.randint(60, size=n)
        pickup_datetimes = []

        for i in range(len(pickup_hour)):
            pickup_datetime=datetime(2022,1,1,pickup_hour[i],pickup_minute[i],0)
            pickup_datetimes.append(pickup_datetime)

        trips['pickup_day'] = pickup_day
        trips['pickup_hour'] = pickup_hour
        trips['pickup_minute'] = pickup_minute
        trips['pickup_datetime'] = pickup_datetimes

        routes = []
        dropoff_datetimes = []
        node_timestamps = []
        trip_durations = []
        for index, row in trips.iterrows():
            try:
                route = ox.shortest_path(
                    graph, trips.loc[index, "pickup_node"], trips.loc[index, "dropoff_node"], weight='travel_time'
                )
                routes.append(route)
                travel_times = ox.utils_graph.get_route_edge_attributes(graph,route,attribute='travel_time')
                pickup_datetime = trips.loc[index, "pickup_datetime"]
                dropoff_datetime = pickup_datetime + timedelta(seconds=sum(travel_times))
                trip_duration = (dropoff_datetime-pickup_datetime).total_seconds()
                
                timestamps_dict = DataPreProcessing.map_nodes_to_timestaps(route, pickup_datetime, dropoff_datetime, trip_duration)

                dropoff_datetimes.append(dropoff_datetime)
                trip_durations.append(trip_duration)
                node_timestamps.append(timestamps_dict)

            except Exception as e:
                print(e)
                trips.drop(index, inplace=True)

        trips["route"] = routes
        trips["dropoff_datetime"] = dropoff_datetimes
        trips["trip_duration"] = trip_durations
        trips["node_timestamps"] = node_timestamps
        trips.to_csv("trips_meinheim.csv")
        self.trips = trips

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