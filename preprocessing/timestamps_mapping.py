from datetime import timedelta

import networkx as nx
import osmnx as ox
import pandas as pd


def setup_graph():
    graph = ox.io.load_graphml("data/graph/full.graphml")
    return graph


def timestamp_range(start_time, end_time, delta, route_nodes):
    """
    Generates a list of timestamps for the list of route_nodes between start_time and end_time

    :param start_time: stat time and the fime for the fist node of the route
    :param end_time: the end time for the route, time of last node of the route
    :param delta: the time duration between each 2 nodes of the route
    :param route_nodes: all the nodes that are reached from starting point to destination
    :return: a list of timestamps where each timestamp represents the time when a specific node is reached
    """
    timestamps = []

    while start_time < end_time:
        start_time_formatted = start_time.strftime("%Y-%m-%d %H:%M:%S")
        timestamps.append(start_time_formatted)
        start_time += delta

    if len(timestamps) == len(route_nodes):
        end_time_formatted = end_time.strftime("%Y-%m-%d %H:%M:%S")
        timestamps[-1] = end_time_formatted
    else:
        end_time_formatted = end_time.strftime("%Y-%m-%d %H:%M:%S")
        timestamps.append(end_time_formatted)
    return timestamps


def map_nodes_to_timestaps(route_nodes, pickup_time, dropoff_time, duration):
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
        time_between_nodes = duration / (len(route_nodes) - 1)
        delta = timedelta(seconds=time_between_nodes)
        timestamps = timestamp_range(start_time, end_time, delta, route_nodes)
    else:
        timestamps.append(dropoff_time)

    timestamps_mapping = dict(zip(route_nodes, timestamps))
    return timestamps_mapping


def map_nodes_to_timestaps_to_list(route_nodes, pickup_time, dropoff_time, duration):
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
        time_between_nodes = duration / (len(route_nodes) - 1)
        delta = timedelta(seconds=time_between_nodes)
        timestamps = timestamp_range(start_time, end_time, delta, route_nodes)
    else:
        timestamps.append(dropoff_time)

    return timestamps


def map_routes_to_trips(graph: nx.MultiDiGraph, trips: pd.DataFrame):
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

            timestamps_dict = map_nodes_to_timestaps(route, trips.loc[index, "pickup_datetime"],
                                                     trips.loc[index, "dropoff_datetime"]
                                                     , trips.loc[index, "trip_duration"])

            node_timestamps.append(timestamps_dict)

        except Exception as e:
            trips.drop(index, inplace=True)

    trips["route"] = routes
    trips["route_timestamps"] = node_timestamps
    return trips


def map_nodes_to_timestaps_to_list(route_nodes, pickup_time, dropoff_time, duration):
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
        time_between_nodes = duration / (len(route_nodes) - 1)
        delta = timedelta(seconds=time_between_nodes)
        timestamps = timestamp_range(start_time, end_time, delta, route_nodes)
    else:
        dropoff_time_formated = dropoff_time.strftime("%Y-%m-%d %H:%M:%S")
        timestamps.append(dropoff_time_formated)

    return timestamps


graph = setup_graph()
trips = pd.read_csv("data//trips//trips_with_nodes.csv")
trips_with_routes = map_routes_to_trips(graph, trips)
trips_with_routes.to_csv("data/trips/trips_with_routes_timestamps.csv")
