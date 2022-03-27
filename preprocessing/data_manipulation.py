import osmnx as ox
import pandas as pd
from data_preprocessing import DataPreProcessing


def generate_timestamps_for_route():
    graph = ox.io.load_graphml("../data/graph/full.graphml")
    trips = pd.read_csv("../data/trips/trips_with_nodes.csv")
    data_processing = DataPreProcessing(graph, trips)
    trips_with_timestamps = data_processing.map_routes_to_trips_with_timestamps()
    trips_with_timestamps.to_csv("../data/trips/trips_with_routes_timestamps.csv")


def generate_route_for_trips():
    graph = ox.io.load_graphml("../data/graph/full.graphml")
    trips = pd.read_csv("../data/trips/trips_with_nodes.csv")
    data_processing = DataPreProcessing(graph, trips)
    trips_with_routes = data_processing.map_routes_to_trips()
    trips_with_routes.to_csv("../data/trips/trips_with_routes.csv")

generate_timestamps_for_route()
generate_route_for_trips()
