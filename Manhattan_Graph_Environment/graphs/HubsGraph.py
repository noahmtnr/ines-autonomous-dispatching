import random
from datetime import datetime, timedelta

import osmnx as ox
import pandas as pd


#graph to be used: full.graphml (all nodes)
#if we use small_manhattan.graphml, we do not have all nodes which are in the trips and then we get Key Error
class HubsGraph:
    """Copy of an old state of ManhattanGraph used in API to map coordinates to closest hub. Not fully up to date.
    """

    def __init__(self, filename, num_hubs):
        filepath = ("data/graph/%s.graphml") % (filename)
        self.inner_graph = ox.load_graphml(filepath)
        fin_hub = random.sample(self.inner_graph.nodes(),1)

    def setup_trips(self, start_time: datetime):
     
        #trips for simple graph, only the first 5000 rows
        all_trips = pd.read_csv('data/trips/preprocessed_trips.csv', nrows=100)
        self.trips =  all_trips
        #compute trip length and add to csv
        #generate random passenger count between 1 and 4 and add to csv
        route_length_column=[]
        for i in self.trips.index:
            current_route_string = self.trips.iloc[i]["route"]
            string_split = current_route_string.replace('[','').replace(']','').split(',')
            current_route = [int(el) for el in string_split]
            route_length = 0
            for j in range(len(current_route)-1):
                route_length += ox.distance.great_circle_vec(self.inner_graph.nodes()[current_route[j]]['y'], self.inner_graph.nodes()[current_route[j]]['x'],
                self.inner_graph.nodes()[current_route[j+1]]['y'], self.inner_graph.nodes()[current_route[j+1]]['x'])
            route_length_column.append(route_length)

            self.trips["route_length"]=route_length_column

            provider_column=[]
            totalprice_column=[]
            x = pd.read_csv("Provider.csv")
            for i in self.trips.index:
                provider_id = x['id'].sample(n=1).iloc[0]
                provider_column.append(provider_id)
                selected_row = x[x['id']==provider_id]
                basic_price = selected_row['basic_cost']
                km_price = selected_row['cost_per_km']
                leng = self.trips.iloc[0]['route_length']
                # note that internal distance unit is meters in OSMnx
                total_price = basic_price + km_price*leng/1000
                totalprice_column.append(total_price.iloc[0])
            self.trips["provider"]=provider_column
            self.trips["total_price"]=totalprice_column


        self.trips.to_csv("trips_kaggle_providers.csv")

        return self.trips

    def prefilter_trips(self, all_trips, start_time: datetime):
        BOTTOM = str(start_time - timedelta(hours=2))
        TOP = str(start_time + timedelta(hours=24))
        return all_trips[(all_trips['pickup_datetime'] >= BOTTOM)&(all_trips['pickup_datetime'] <= TOP)]

    def nodes(self):
        return self.inner_graph.nodes()

    def edges(self):
        return self.inner_graph.edges()

    def get_nodeids_list(self):
        return list(self.nodes())

    def get_node_by_nodeid(self, nodeid: int):
        return self.nodes()[nodeid]

    def get_node_by_index(self, index: int):
        return self.get_node_by_nodeid(self.get_nodeids_list()[index])

    def get_nodeid_by_index(self, index: int):
        return self.get_nodeids_list()[index]

    def get_index_by_nodeid(self, nodeid: int):
        return self.get_nodeids_list().index(nodeid)

    def get_coordinates_of_node(self, node_id): 
        # manhattangraph = ManhattanGraph(filename='simple', num_hubs=70)
        nodes = self.inner_graph.nodes()
        return [nodes[node_id]['x'], nodes[node_id]['y']]
        
    def get_nodeid_by_hub_index(self, hub_index: int):
        return self.hubs[hub_index]

    def get_node_by_hub_index(self, hub_index: int):
        return self.get_node_by_nodeid(self.get_nodeid_by_hub_index(hub_index))

    def getNearestNode(self, pickup_longitude, pickup_latitude):
        pickup_node_id = ox.distance.nearest_nodes(self.inner_graph, pickup_longitude, pickup_latitude)
        return pickup_node_id

        # def get_coordinates_of_node_by_hub_index(self, node_index): 
        #     # manhattangraph = ManhattanGraph(filename='simple', num_hubs=70)
        #     nodes = self.inner_graph.nodes()
        #     get_node_by_index(node_index):

        #     return [nodes[node_id]['x'], nodes[node_id]['y']]       



            #self.generate_hubs(fin_hub, num_hubs)


        # def generate_hubs(self, fin_hub, num_hubs: int = 70):#, opt: int = 0):
        #     # the code below is for loading the hubs specified in data/trips/manual_hubs.csv
        #     """Generates random bubs within the graph

        #     Args:e
        #         fin_hub (int): index_id of final hub
        #         num_hubs (int, optional): Number of hubs to create. Defaults to 5.

        #     Returns:
        #         self.hubs(list): List of hubs in graph
        #     """
        #     # the code below is for mapping the pre-defined hubs (customer/store/trips) to nodes in the graph
        #     hubs_file = pd.read_csv("data/hubs/new_hubs.CSV")
        #     hubs = []
        #     i=0
        #     for row in hubs_file.index:    
        #         hubs.append(ox.get_nearest_node(self.inner_graph,(hubs_file.loc[row,"latitude"], hubs_file.loc[row,"longitude"])))
        #     # if opt == 0:
        #     #     # the code below is for mapping the pre-defined hubs (customer/store/trips) to nodes in the graph
        #     #     hubs_file = pd.read_csv("data/hubs/manual_hubs.CSV")
        #     #     hubs = []
        #     #     i=0
        #     #     for row in hubs_file.index:    
        #     #         hubs.append(ox.get_nearest_node(self.inner_graph,(hubs_file.loc[row,"latitude"], hubs_file.loc[row,"longitude"])))
        #     #         #hubs.append(hubs_file.loc[row])
        #     # else:
        #     #     hubs_file = pd.read_csv('rl/Manhattan_Graph_Environment/top_nodes.csv')
        #     #     hubs = []
        #     #     i=0
        #     #     for row in hubs_file.index:    
        #     #         hubs.append(hubs_file.loc[row])
            
        #     self.hubs = random.sample(hubs,num_hubs)
        #     #self.hubs = hubs
        #     return self.hubs

        def setup_trips(self, start_time: datetime):
            """Read trips information of Kaggle file in self.trips.

            Args:
                n (int, optional): Number of trips to be generated. Defaults to 2000.
            """
            
            #trips for simple graph, only the first 5000 rows
            all_trips = pd.read_csv('data/trips/preprocessed_trips.csv', nrows=100)
            print(all_trips.head())
            #self.trips = self.prefilter_trips(all_trips, start_time).reset_index(drop=True)
            self.trips =  all_trips
            #compute trip length and add to csv
            #generate random passenger count between 1 and 4 and add to csv
            route_length_column=[]
            for i in self.trips.index:
                current_route_string = self.trips.iloc[i]["route"]
                string_split = current_route_string.replace('[','').replace(']','').split(',')
                current_route = [int(el) for el in string_split]
                #print(current_route)
                route_length = 0
                for j in range(len(current_route)-1):
                    #print(self.inner_graph.nodes()[current_route[j]])
                    #print(current_route[j])
                    #print(self.get_node_by_nodeid(current_route[j]))
                    #print(self.nodes())
                    route_length += ox.distance.great_circle_vec(self.inner_graph.nodes()[current_route[j]]['y'], self.inner_graph.nodes()[current_route[j]]['x'],
                    self.inner_graph.nodes()[current_route[j+1]]['y'], self.inner_graph.nodes()[current_route[j+1]]['x'])
                route_length_column.append(route_length)

            self.trips["route_length"]=route_length_column

            # add mobility providers randomly
        #     provider_column=[]
        #     totalprice_column=[]
        #     x = pd.read_csv("Provider.csv")
        #     for i in self.trips.index:
        #         provider_id = x['id'].sample(n=1).iloc[0]
        #         provider_column.append(provider_id)
        #         selected_row = x[x['id']==provider_id]
        #         basic_price = selected_row['basic_cost']
        #         km_price = selected_row['cost_per_km']
        #         leng = self.trips.iloc[0]['route_length']
        #         # note that internal distance unit is meters in OSMnx
        #         total_price = basic_price + km_price*leng/1000
        #         totalprice_column.append(total_price.iloc[0])
        #     self.trips["provider"]=provider_column
        #     self.trips["total_price"]=totalprice_column


        #     self.trips.to_csv("trips_kaggle_providers.csv")

        #     return self.trips

        # def prefilter_trips(self, all_trips, start_time: datetime):
        #     BOTTOM = str(start_time - timedelta(hours=2))
        #     TOP = str(start_time + timedelta(hours=24))
        #     return all_trips[(all_trips['pickup_datetime'] >= BOTTOM)&(all_trips['pickup_datetime'] <= TOP)]
        
        # def nodes(self):
        #     return self.inner_graph.nodes()

        # def edges(self):
        #     return self.inner_graph.edges()

        # def get_nodeids_list(self):
        #     return list(self.nodes())

        # def get_node_by_nodeid(self, nodeid: int):
        #     return self.nodes()[nodeid]

        # def get_node_by_index(self, index: int):
        #     return self.get_node_by_nodeid(self.get_nodeids_list()[index])

        # def get_nodeid_by_index(self, index: int):
        #     return self.get_nodeids_list()[index]

        # def get_index_by_nodeid(self, nodeid: int):
        #     return self.get_nodeids_list().index(nodeid)
        
        # def get_coordinates_of_node(self, node_id): 
        #     # manhattangraph = ManhattanGraph(filename='simple', num_hubs=70)
        #     nodes = self.inner_graph.nodes()
        #     return [nodes[node_id]['x'], nodes[node_id]['y']]
            
        # def get_nodeid_by_hub_index(self, hub_index: int):
        #     return self.hubs[hub_index]

        # def get_node_by_hub_index(self, hub_index: int):
        #     return self.get_node_by_nodeid(self.get_nodeid_by_hub_index(hub_index))

        # # def get_coordinates_of_node_by_hub_index(self, node_index): 
        # #     # manhattangraph = ManhattanGraph(filename='simple', num_hubs=70)
        # #     nodes = self.inner_graph.nodes()
        # #     get_node_by_index(node_index):

        # #     return [nodes[node_id]['x'], nodes[node_id]['y']]       


