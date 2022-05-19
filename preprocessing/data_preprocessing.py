import osmnx as ox
import networkx as nx
import pandas as pd
from datetime import timedelta,datetime
import time

class DataPreProcessing:
    
    def setup_graph():
        graph = ox.io.load_graphml("data/graph/simple.graphml")
        return graph

    def setup_trips(nrows: int = None) -> pd.DataFrame():
        df = pd.read_csv(
            r"/Users/noah/OneDrive - Universität Mannheim/Uni/Mannheim/Team Project/nyc-taxi-trip-duration/train.csv",
            nrows=nrows,
        )
        return df

    def map_trips_to_nodes(graph, trips):

        start_time = time.time()

        pickup_node = []
        dropoff_node = []
        pickup_distance = []
        dropoff_distance = []

        total_rows = len(trips)

        # Potentially optimizable by using nearest_nodes() with the entire trips list instead of this for loop (doc: https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.distance.nearest_nodes)
        p_long = trips["pickup_longitude"]
        p_lat = trips["pickup_latitude"]
        d_lat = trips["dropoff_latitude"]
        d_long = trips["dropoff_longitude"]
        
        dropoff_nodes, dropoff_distances = ox.distance.nearest_nodes(
            graph, d_long, d_lat, return_dist=True
        )
        pickup_nodes, pickup_distances = ox.distance.nearest_nodes(
                graph, p_long, p_lat, return_dist=True
            )

        trips["pickup_node"] = pickup_nodes
        trips["dropoff_node"] = dropoff_nodes
        trips["pickup_distance"] = pickup_distances
        trips["dropoff_distance"] = dropoff_distances

        print(
            "MAPPING TRIPS TO NODES DONE: ",
            str(len(trips)),
            "trips took --- %s seconds ---" % round((time.time() - start_time), 2),
        )
        return trips

    def map_routes_to_trips(graph, trips):

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

        edge_travel_times = DataPreProcessing.relative_edge_travel_time(graph, route_nodes, start_time, end_time)

        for delta in edge_travel_times:
            timestamps.append(timestamps[-1] + timedelta(seconds=round(delta)))

        def to_string(timestamp):
            return timestamp.strftime('%Y-%m-%d %X')

        return list(map( to_string, timestamps))

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
            timestamps = DataPreProcessing.timestamp_range(graph, route_nodes, start_time, end_time)
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
        start_time = time.time()

        routes = []
        node_timestamps = []
        for index, row in trips.iterrows():
            try:
                route = nx.shortest_path(
                    graph, trips.loc[index, "pickup_node"], trips.loc[index, "dropoff_node"]
                )
                routes.append(route)

                timestamps_dict = DataPreProcessing.map_nodes_to_timestaps(graph, route, trips.loc[index, "pickup_datetime"],
                                                         trips.loc[index, "dropoff_datetime"]
                                                         , trips.loc[index, "trip_duration"])

                node_timestamps.append(timestamps_dict)

            except Exception as e:
                trips.drop(index, inplace=True)

        trips["route"] = routes
        trips["route_timestamps"] = node_timestamps

        print(
            "MAPPING ROUTES TO TRIPS WITH TIMESTAMPS DONE: ",
            str(len(trips)),
            "trips took --- %s seconds ---" % round((time.time() - start_time), 2),
        )

        return trips
    
    def map_oneRoute_to_oneTrip_with_timestamps(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, pickup_datetime, dropoff_datetime, trip_duration):
        """
        Adds for the trips data the route and the timestamp for each node

        :param graph: the graph representing the roads and junctions
        :param trips: all the tips
        :return: the trips in addition having the route and the timestamp for each node from the route
        """
        graph = ox.io.load_graphml("data/graph/simple.graphml")
        
        pickup_node = (pickup_longitude, pickup_latitude)
        dropoff_node = (dropoff_longitude, dropoff_latitude)
        pickup_node_id = ox.distance.nearest_nodes(graph,pickup_longitude, pickup_latitude)
        print(pickup_node_id)
        dropoff_node_id = ox.distance.nearest_nodes(graph, dropoff_longitude, dropoff_latitude)
        print(dropoff_node_id)
        route = nx.shortest_path(
            graph, pickup_node_id, dropoff_node_id
        )

        timestamps_dict = DataPreProcessing.map_nodes_to_timestaps(graph, route, pickup_datetime, dropoff_datetime, trip_duration)

        route_length = 0
        for j in range(len(route)-1):
                #print(self.inner_graph.nodes()[current_route[j]])
                route_length += ox.distance.great_circle_vec(graph.nodes()[route[j]]["y"], graph.nodes()[route[j]]["x"],
                graph.nodes()[route[j+1]]["y"], graph.nodes()[route[j+1]]["x"])
        print(route)
        print(timestamps_dict)

        return route, str(timestamps_dict), route_length, pickup_node_id, dropoff_node_id

    def get_coordinates_of_node(node_id): 
        # manhattangraph = ManhattanGraph(filename='simple', num_hubs=70)
        graph = ox.io.load_graphml("data/graph/simple.graphml")
        nodes = graph.nodes()
        coordinates = [nodes[node_id]['x'], nodes[node_id]['y']]
        return coordinates
    
    def getNearestNodeId(pickup_longitude, pickup_latitude):
        graph = ox.io.load_graphml("data/graph/simple.graphml")
        pickup_node_id = ox.distance.nearest_nodes(graph, pickup_longitude, pickup_latitude)
        return pickup_node_id

    def get_node_index_by_coordinates(longitude, latitude):
        graph = ox.io.load_graphml("data/graph/simple.graphml")
        node_id = ox.distance.nearest_nodes(graph, longitude, latitude)
        return list(graph.nodes()).index(node_id)

    def get_node_index_by_id(id):
        x, y = DataPreProcessing.get_coordinates_of_node(id)
        return DataPreProcessing.get_node_index_by_coordinates(x, y)
