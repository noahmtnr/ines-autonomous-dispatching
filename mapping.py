import osmnx as ox
import networkx as nx
import pandas as pd
import time


def setup_graph() -> pd.DataFrame():
    graph = ox.io.load_graphml("data/full.graphml")
    # graph = ox.graph_from_place('Manhattan, New York City, New York, USA', network_type='drive', simplify=False)
    # graph = ox.add_edge_speeds(graph)
    # graph = ox.add_edge_travel_times(graph)
    # ox.add_edge_travel_times(graph)
    # ox.save_graphml(graph, filepath="./data/full.graphml")
    return graph


def setup_trips(nrows: int = None) -> pd.DataFrame():
    df = pd.read_csv(
        r"/Users/noah/OneDrive - Universität Mannheim/Uni/Mannheim/Team Project/nyc-taxi-trip-duration/train.csv",
        nrows=nrows,
    )
    return df


def map_trips_to_nodes(graph: nx.MultiDiGraph, trips: pd.DataFrame):

    start_time = time.time()
    print("MAPPING STARTED")

    pickup_node = []
    dropoff_node = []
    pickup_distance = []
    dropoff_distance = []

    total_rows = len(trips)

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


def map_routes_to_trips(graph: nx.MultiDiGraph, trips: pd.DataFrame):
    # trips = trips.head()
    pickup = trips["pickup_node"]
    dropoff = trips["dropoff_node"]

    # routes = nx.shortest_path(
    #             graph, trips["pickup_node"], trips["dropoff_node"]
    #         )

    # print(routes)
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


graph = setup_graph()
# trips = setup_trips(5000)
# trips_with_nodes = map_trips_to_nodes(graph, trips)
# trips_with_nodes.to_csv("trips_with_nodes.csv")

trips = pd.read_csv("trips_with_nodes.csv")
trips_eith_routes = map_routes_to_trips(graph, trips)
trips_eith_routes.to_csv("trips_with_routes.csv")
