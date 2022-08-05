"""
This file preprocesses taxi data by adding timestamps and routes to it.
"""

import osmnx as ox
import pandas as pd

from data_preprocessing import DataPreProcessing


def generate_timestamps_for_route():
    """
    This function adds timpestamps to data that have already nodes.
    """
    graph = ox.io.load_graphml("data/graph/full.graphml")
    trips = pd.read_csv("data/trips/trips_with_nodes.csv")
    trips_with_timestamps = DataPreProcessing.map_routes_to_trips_with_timestamps(trips, graph)
    trips_with_timestamps.to_csv("data/trips/trips_with_routes_timestamps.csv")


def generate_route_for_trips():
    """    
    This function adds routes to data that have already nodes.
    """
    graph = ox.io.load_graphml("data/graph/full.graphml")
    trips = pd.read_csv("data/trips/trips_with_nodes.csv")

    trips_with_routes = DataPreProcessing.map_routes_to_trips(trips, graph)
    trips_with_routes.to_csv("data/trips/trips_with_routes.csv")


generate_timestamps_for_route()
generate_route_for_trips()
