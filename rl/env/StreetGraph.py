import osmnx as ox
import networkx as nx
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta




class StreetGraph:

    def __init__(self, filename):
        filepath = ("../../data/graph/%s.graphml") % (filename)
        self.graph = ox.load_graphml(filepath)
        self.graph = ox.add_edge_speeds(self.graph,fallback=30)
        self.graph = ox.add_edge_travel_times(self.graph)
        ox.utils_graph.remove_isolated_nodes(self.graph)
        self.generateRandomTrips(1000)


    def generateRandomTrips(self, n: int = 1000):
        """Generates random trips within the graph and stores them in self.trips. The trips are randomly spread across January 2022.

        Args:
            n (int, optional): Number of trips to be generated. Defaults to 1000.
        """
        
        graph = self.graph

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

        for node1 in random_nodes:
            for node2 in random_nodes2:
                if node1 != node2:
                    if count < n:
                        trip_id.append(count)
                        nodeA.append(node1)
                        nodeB.append(node2)
                        count += 1

        trips["tripid"] = trip_id
        trips["pickup_node"] = nodeA
        trips["dropoff_node"] = nodeB

        pickup_day = np.random.randint(1,31, size=n)
        pickup_hour =  np.random.randint(24, size=n)
        pickup_minute = np.random.randint(60, size=n)
        pickup_datetimes = []

        for i in range(len(pickup_hour)):
            pickup_datetime=datetime(2022,1,pickup_day[i],pickup_hour[i],pickup_minute[i],0)
            pickup_datetimes.append(pickup_datetime)

        trips['pickup_day'] = pickup_day
        trips['pickup_hour'] = pickup_hour
        trips['pickup_minute'] = pickup_minute
        trips['pickup_datetime'] = pickup_datetime

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
                
                timestamps_dict = map_nodes_to_timestaps(route, pickup_datetime, dropoff_datetime, trip_duration)

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
        self.trips = trips



def map_nodes_to_timestaps(route_nodes, pickup_time, dropoff_time, duration):
    timestamps = []
    date_format_str = '%Y-%m-%d %H:%M:%S.%f'
    start_time = pd.to_datetime(pickup_time, format=date_format_str)
    end_time = pd.to_datetime(dropoff_time, format=date_format_str)

    if (len(route_nodes) > 1):
        time_between_nodes = duration / (len(route_nodes) - 1)
        delta = timedelta(seconds=time_between_nodes)
        timestamps = timestamp_range(start_time, end_time, delta, route_nodes)
    else:
        timestamps.append(dropoff_time)

    timestamps_mapping = dict(zip(route_nodes, timestamps))
    return timestamps_mapping


def timestamp_range(start_time, end_time, delta, route_nodes):
    timestamps = []

    while start_time < end_time:
        start_time_formatted = str(start_time.strftime("%Y-%m-%d %H:%M:%S"))
        timestamps.append(start_time_formatted)
        start_time += delta

    if len(timestamps) == len(route_nodes):
        end_time_formatted = str(end_time.strftime("%Y-%m-%d %H:%M:%S"))
        timestamps[-1] = str(end_time_formatted)
    else:
        end_time_formatted = str(end_time.strftime("%Y-%m-%d %H:%M:%S"))
        timestamps.append(end_time_formatted)
    return timestamps