import osmnx as ox
import networkx as nx
import pandas as pd
from datetime import timedelta
import time

class NYCCityPreProcessing:

    def setup_graph():
        graph = ox.io.load_graphml("data/graph/full.graphml")
        return graph

    def setup_trips(year: int, month: int) -> pd.DataFrame():
        # e.g., month = 1
        # e.g., year = 2009
        df = pd.read_csv(fr'C:\Users\dispatching-system\Downloads\NYCWebsite_Data\trips_{year}_{month}.csv')
        return df

    def map_trips_to_nodes(trips, graph):

        start_time = time.time()
        print("MAPPING STARTED")

        pickup_node = []
        dropoff_node = []
        pickup_distance = []
        dropoff_distance = []

        total_rows = len(trips)

        # Potentially optimizable by using nearest_nodes() with the entire trips list instead of this for loop (doc: https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.distance.nearest_nodes)
        for index, row in trips.iterrows():
            p_lat = trips.loc[index, "pickup_latitude"]
            p_long = trips.loc[index, "pickup_longitude"]
            d_lat = trips.loc[index, "dropoff_latitude"]
            d_long = trips.loc[index, "dropoff_longitude"]
            p_node, p_dist = ox.distance.nearest_nodes(
                graph, p_long, p_lat, return_dist=True
            )
            d_node, d_dist = ox.distance.nearest_nodes(
                graph, d_long, d_lat, return_dist=True
            )

            pickup_node.append(p_node)
            dropoff_node.append(d_node)
            pickup_distance.append(p_dist)
            dropoff_distance.append(d_dist)
            print("Rows mapped: ", round((index + 1) / total_rows * 100, 2), "%")

        trips["pickup_node"] = pickup_node
        trips["dropoff_node"] = dropoff_node
        trips["pickup_distance"] = pickup_distance
        trips["dropoff_distance"] = dropoff_distance

        print(
            "MAPPING DONE: ",
            str(len(trips)),
            "trips took --- %s seconds ---" % round((time.time() - start_time), 2),
        )
        return trips

    def map_routes_to_trips(trips, graph):

        routes = []
        for index, row in trips.iterrows():
            try:
                route = nx.shortest_path(
                    graph, trips.loc[index, "pickup_node"], trips.loc[index, "dropoff_node"]
                )
                routes.append(route)
            except Exception as e:
                trips.drop(index, inplace=True)

        trips["route"] = routes
        return trips

    def timestamp_range(graph, route_nodes, start_time, end_time):
        """
        Generates a list of timestamps for the list of route_nodes between start_time and end_time using the relative proportion of edge speed to the actual traxi trip duration

        :param graph: graph in which the route takes place
        :param route_nodes: all the nodes that are reached from starting point to destination
        :param start_time: stat time and the fime for the fist node of the route
        :param end_time: the end time for the route, time of last node of the route

        :return: a list of timestamps where each timestamp represents the time when a specific node is reached
        """
        timestamps = [start_time]

        edge_travel_times = NYCCityPreProcessing.relative_edge_travel_time(graph, route_nodes, start_time, end_time)

        for delta in edge_travel_times:
            timestamps.append(timestamps[-1] + timedelta(seconds=round(delta)))

        return timestamps

    def relative_edge_travel_time(graph, route_nodes, start_time, end_time):
        """
        Generates a list of timestamps for the list of route_nodes between start_time and end_time

        :param start_time: stat time and the fime for the fist node of the route
        :param end_time: the end time for the route, time of last node of the route
        :param delta: the time duration between each 2 nodes of the route
        :param route_nodes: all the nodes that are reached from starting point to destination
        :return: a list of timestamps where each timestamp represents the time when a specific node is reached
        """
        timestamps = [start_time]

        route_edge_travel_times = ox.utils_graph.get_route_edge_attributes(graph,route_nodes,attribute='travel_time')

        actual_travel_time = end_time-start_time
        free_flow_travel_time = round(sum(route_edge_travel_times))
        
        # work around for preventing division by zero
        if free_flow_travel_time == 0:
            free_flow_travel_time = 1

        actual_edge_travel_speed = []

        for x in route_edge_travel_times:
            def proportion_of_actual_travel_time(x): return x/free_flow_travel_time * actual_travel_time
            actual_edge_travel_speed.append(proportion_of_actual_travel_time(x).total_seconds())

        return actual_edge_travel_speed

    def map_nodes_to_timestaps(graph, route_nodes, pickup_time, dropoff_time, duration):
        """
        Maps the timestamp list with the route nodes to have for each route node the time when a particular node is reached

        :param route_nodes: list of nodes that are reached from starting point until destination
        :param pickup_time: start time of the trip
        :param dropoff_time: end time of the trip
        :param duration: duration in seconds of the trip
        :return: a dictionary mapping the list of nodes to the list of timestamps,
                each being mapped to the time this node is reached
        """
        timestamps = []
        date_format_str = '%Y-%m-%d %H:%M:%S.%f'
        start_time = pd.to_datetime(pickup_time, format=date_format_str)
        end_time = pd.to_datetime(dropoff_time, format=date_format_str)

        if (len(route_nodes) > 1):
            timestamps = NYCCityPreProcessing.timestamp_range(graph, route_nodes, start_time, end_time)
        else:
            timestamps.append(dropoff_time)

        timestamps_mapping = dict(zip(route_nodes, timestamps))
        return timestamps_mapping

    def map_routes_to_trips_with_timestamps(trips, graph):
        """
        Adds for the trips data the route and the timestamp for each node

        :param graph: the graph representing the roads and junctions
        :param trips: all the tips
        :return: the trips in addition having the route and the timestamp for each node from the route
        """
        routes = []
        node_timestamps = []
        for index, row in trips.iterrows():
            try:
                route = nx.shortest_path(
                    graph, trips.loc[index, "pickup_node"], trips.loc[index, "dropoff_node"]
                )
                routes.append(route)

                timestamps_dict = NYCCityPreProcessing.map_nodes_to_timestaps(graph, route, trips.loc[index, "pickup_datetime"],
                                                         trips.loc[index, "dropoff_datetime"]
                                                         , trips.loc[index, "trip_duration"])

                node_timestamps.append(timestamps_dict)

            except Exception as e:
                trips.drop(index, inplace=True)

        trips["route"] = routes
        trips["route_timestamps"] = node_timestamps
        return trips


    def generate_timestamps_for_route():
        graph = ox.io.load_graphml("data/graph/full.graphml")
        trips = pd.read_csv("data/trips/trips_with_nodes.csv")
        trips_with_timestamps = NYCCityPreProcessing.map_routes_to_trips_with_timestamps(trips, graph)
        trips_with_timestamps.to_csv("data/trips/trips_with_routes_timestamps.csv")


    def generate_route_for_trips():
        graph = ox.io.load_graphml("data/graph/full.graphml")
        trips = pd.read_csv("data/trips/trips_with_nodes.csv")
        
        trips_with_routes = NYCCityPreProcessing.map_routes_to_trips(trips, graph)
        trips_with_routes.to_csv("data/trips/trips_with_routes.csv")

# generate_timestamps_for_route()
# generate_route_for_trips()

