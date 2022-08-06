import random

import networkx as nx
import osmnx as ox
import pandas as pd

# Smallest Graph
# location = 'Meinheim, Bayern, Germany'
# graph = ox.graph_from_place(location, network_type='drive')

# Slightly bigger graph
# location_point = (40.7704797, -73.9838597)
# graph = ox.graph_from_point(location_point, dist=180)
# ox.save_graphml(graph, filepath="./data/graph/small_manhattan.graphml")

# Import local graphml
graph = ox.io.load_graphml("data/graph/meinheim.graphml")
# graph = ox.io.load_graphml("data/graph/small_manhattan.graphml")

ox.utils_graph.remove_isolated_nodes(graph)

# Uncomment to show graph
# fig, ax = ox.plot_graph(graph, node_color="r", node_size=10, figsize=(10, 20))
# fig.show()

gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)
print("# nodes: ", len(gdf_nodes))
print("# edges: ", len(gdf_edges))

df = pd.DataFrame(gdf_nodes)
df.reset_index(inplace=True)

sequence = df["osmid"]
sequence = sequence.to_numpy()
sequence = sequence.tolist()

random_nodes = random.choices(sequence, k=1000)
random_nodes2 = random.choices(sequence, k=1000)
random_hubs = random.choices(sequence, k=int(len(gdf_nodes) / 10))

isHub = []
for index, row in gdf_nodes.iterrows():
    if gdf_nodes.loc[index, "osmid"] in random_hubs:
        isHub.append(True)
    else:
        isHub.append(False)
gdf_nodes["isHub"] = isHub

trips = pd.DataFrame()
count = 0
trip_id = []
nodeA = []
nodeB = []
for node1 in random_nodes:

    for node2 in random_nodes2:
        if node1 != node2:
            if count < 1000:
                trip_id.append(count)
                nodeA.append(node1)
                nodeB.append(node2)
                count += 1

trips["tripid"] = trip_id
# trips["isHub"] = isHub
trips["pickup_node"] = nodeA
trips["dropoff_node"] = nodeB

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
print("# routes: ", len(routes))
print("Assigned %s hubs randomly: " % len(random_hubs), random_hubs)
