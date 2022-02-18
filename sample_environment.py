
import networkx as nx
import pandas as pd
import osmnx as ox
import folium
from IPython.display import IFrame
import random

# Uncomment to download new graph
# location = 'Meinheim, Bayern, Germany'
# graph = ox.graph_from_place(location, network_type='drive')

# Import local graphml
graph = ox.io.load_graphml("data/meinheim.graphml")

gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)

df = pd.DataFrame(gdf_nodes)
df.reset_index(inplace=True)

sequence = df['osmid']
sequence = sequence.to_numpy()
sequence = sequence.tolist()

random_nodes = random.choices(sequence, k=1000)
random_nodes2 = random.choices(sequence, k=1000)

trips = pd.DataFrame()
count = 0
trip_id = []
nodeA = []
nodeB = []
for node1 in random_nodes:
        for node2 in random_nodes2:
            if (node1 != node2): 
                if(count<1000):
                    trip_id.append(count)
                    nodeA.append(node1)
                    nodeB.append(node2)
                    count +=1  

trips['tripid'] = trip_id
trips['pickup_node'] = nodeA
trips['dropoff_node'] = nodeB


routes = []


for index, row in trips.iterrows():
    route = nx.shortest_path(graph, trips.loc[index, "pickup_node"], trips.loc[index, "dropoff_node"])
    routes.append(route)

trips['route'] = routes
print(trips)
# for trip in trips:
#     #print(trip[2])
#     #print(trip[1])
#     route = nx.shortest_path(graph, trips[trip,2], trips[trip,1])
#     # print(len(route))




