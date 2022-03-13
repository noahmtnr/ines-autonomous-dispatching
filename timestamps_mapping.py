import osmnx as ox
import networkx as nx
import pandas as pd
from datetime import timedelta


def setup_graph():
    graph = ox.io.load_graphml("data/graph/full.graphml")
    return graph


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


def map_routes_to_trips(graph: nx.MultiDiGraph, trips: pd.DataFrame):
    routes = []
    node_timestamps = []
    for index, row in trips.iterrows():
        try:
            route = nx.shortest_path(
                graph, trips.loc[index, "pickup_node"], trips.loc[index, "dropoff_node"]
            )
            routes.append(route)

            timestamps_dict = map_nodes_to_timestaps(route, trips.loc[index, "pickup_datetime"], trips.loc[index, "dropoff_datetime"]
            , trips.loc[index, "trip_duration"])

            node_timestamps.append(timestamps_dict)

        except Exception as e:
            trips.drop(index, inplace=True)

    trips["route"] = routes
    trips["route_timestamps"] = node_timestamps
    return trips


graph = setup_graph()
trips = pd.read_csv("data//trips//trips_with_nodes.csv")
trips_with_routes = map_routes_to_trips(graph, trips)
trips_with_routes.to_csv("data/trips/trips_with_routes_timestamps.csv")
