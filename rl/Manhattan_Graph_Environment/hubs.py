from ManhattanGraph import ManhattanGraph
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import folium
from folium.plugins import MarkerCluster
import numpy as np
#from IPython.display import display


manhattan_graph = ManhattanGraph(filename='simple', num_hubs=52, opt=0)
current_hubs = manhattan_graph.hubs
#print(current_hubs)
x = []
y = []

for i in current_hubs:
    x.append(manhattan_graph.get_node_by_nodeid(i)['x'])
    y.append(manhattan_graph.get_node_by_nodeid(i)['y'])
#print(x)
curr_hub_coordinates = pd.DataFrame()
curr_hub_coordinates['lon'] = x
curr_hub_coordinates['lat'] = y
curr_hub_coordinates = curr_hub_coordinates.to_numpy()
#print(curr_hub_coordinates)


# # plot current hubs:
# boulder_coords = location=[40.778, -73.953]
# #Create the map
# map_hubs = folium.Map(location = boulder_coords, zoom_start = 12)

# #Add 52 hubs to the map
# for i in range(52):
#     folium.Marker([curr_hub_coordinates[i,1],curr_hub_coordinates[i,0]], popup = f"hub {i}").add_to(map_hubs)

# #Display the map
# #display(map_hubs)
# import webbrowser
# map_hubs.save("cur_hubs_map.html")
# webbrowser.open("cur_hubs_map.html")
# #map_hubs.showMap()


# top nodes:

top_nodes = pd.read_csv('rl/Manhattan_Graph_Environment/top_nodes.csv')
top_nodes_list = top_nodes['nodes'].tolist()
#print(len(top_nodes_list))
#print(top_nodes_list)
#print(set(current_hubs).intersection(top_nodes_list)) # {42442937, 42446959} are contained in both lists
x2 = []
y2 = []
manhattan_graph2 = ManhattanGraph(filename='simple', num_hubs=70, opt=1)
for n in top_nodes_list:
    x2.append(manhattan_graph2.nodes()[n]['x'])
    y2.append(manhattan_graph2.nodes()[n]['y'])

new_hub_coordinates = pd.DataFrame()
new_hub_coordinates['lon'] = x2
new_hub_coordinates['lat'] = y2
new_hub_coordinates = new_hub_coordinates.to_numpy()
#print(new_hub_coordinates)

# # plot new hubs:
# boulder_coords = location=[40.778, -73.953]
# #Create the map
# map_hubs_new = folium.Map(location = boulder_coords, zoom_start = 12)

# #Add 70 hubs to the map
# for i in range(70):
#     folium.Marker([new_hub_coordinates[i,1],new_hub_coordinates[i,0]], popup = f"hub {i}").add_to(map_hubs_new)

# #Display the map
# #display(map_hubs)
# import webbrowser
# map_hubs_new.save("new_hubs_map.html")
# webbrowser.open("new_hubs_map.html")



# modify hub locations
modified_hubs = np.copy(curr_hub_coordinates)
modified_hubs[1] = new_hub_coordinates[58]
modified_hubs[48] = new_hub_coordinates[61]
modified_hubs[14] = new_hub_coordinates[46]
modified_hubs[13] = new_hub_coordinates[57]
modified_hubs[30] = new_hub_coordinates[50]
modified_hubs[10] = new_hub_coordinates[67]
modified_hubs[48] = new_hub_coordinates[41]

# append new hubs
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[23]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[45]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[65]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[19]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[28]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[43]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[37]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[25]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[5]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[2]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[31]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[3]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[21]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[35]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[38]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[64]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[69]], axis=0)
modified_hubs = np.append(modified_hubs, [new_hub_coordinates[39]], axis=0)
#print(modified_hubs)
#print(len(modified_hubs))
# print(modified_hubs[:,0])

modified_hubs_df = pd.DataFrame()
modified_hubs_df['latitude'] = modified_hubs[:,1]
modified_hubs_df['longitude'] = modified_hubs[:,0]
print(modified_hubs_df)

#modified_hubs_df.to_csv('new_hubs.csv',index=False)