import osmnx as ox
import networkx as nx
import pandas as pd
import time


def setup_graph() -> pd.DataFrame():
    graph = ox.io.load_graphml("data/full.graphml")
    return graph


def setup_trips(nrows: int) -> pd.DataFrame():
    df = pd.read_csv(
        r"/Users/noah/OneDrive - Universität Mannheim/Uni/Mannheim/Team Project/nyc-taxi-trip-duration/train.csv",
        nrows=nrows,
    )
    return df


def match_trips_to_nodes(graph: nx.MultiDiGraph, trips: pd.DataFrame):

    start_time = time.time()
    print("MAPPING STARTED")

    pickup_node = []
    dropoff_node = []
    pickup_distance = []
    dropoff_distance = []

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

        # print("Coordinates: Lat ",p_lat,"; Long", p_long)
        # print("Pickup Node: ",p_node,"; Pickup Distance",p_dist,"Dropoff Node: ",d_node,"; Dropoff Distance",d_dist)
        # print(" ")
        pickup_node.append(p_node)
        dropoff_node.append(d_node)
        pickup_distance.append(p_dist)
        dropoff_distance.append(d_dist)

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


graph = setup_graph()
trips = setup_trips()
match_trips_to_nodes(graph, trips)
