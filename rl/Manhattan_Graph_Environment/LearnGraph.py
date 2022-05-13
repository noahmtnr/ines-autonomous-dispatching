import networkx as nx
import osmnx as ox
from ManhattanGraph import ManhattanGraph
import numpy as np

class LearnGraph:
        
    def __init__(self, n_hubs: int, manhattan_graph):
        self.G=nx.complete_graph(n_hubs,nx.MultiDiGraph)
        self.manhattan_graph = manhattan_graph
        for node in self.G.nodes():
            self.G.add_edge(node, node)
        self.wait_till_departure_times = {}#np.zeros((70,70))
        ox.save_graphml(self.G, filepath="./data/graph/learn.graphml")

    def adjacency_matrix(self, layer: str = None):
        return nx.to_numpy_array(self.G, weight=layer)

    def add_travel_distance_layer(self):

        #for each edge in learn_graph calculate travel time and add as edge attribute with
        nx.set_edge_attributes(self.G, bb, "travel_distance") 

    def add_travel_cost_layer(self, available_trips):
        # self.manhattan_graph.setup_trips(self.START_TIME) #needed later for retrieving currently available trips
        # nx.set_edge_attributes(self.G, 100, "cost") 
        
        list_hubs = self.manhattan_graph.hubs
        edges = {}
        # wait
        for k in range(70):
            for l in range(70):
                if(k==l):
                    edges[(k,l,0)] = 2
                    self.wait_till_departure_times[(k,l)] = 0
                # book own ride
                else:
                    edges[(k,l,0)] = 50
                    self.wait_till_departure_times[(k,l)] = 300 # 5 minutes for book own ride wait

        for i in range(len(available_trips)):
            for j in range(len(available_trips[i]['route'])):
                # share ride
                if(available_trips[i]['route'][j] in list_hubs):
                    # print(f"Hub on route: {available_trips[i]['route'][j]}")
                    edges[(available_trips[i]['route'][0],available_trips[i]['route'][j],0)] = 5
                    pickup_nodeid = available_trips[i]['route'][0]
                    dropoff_nodeid = available_trips[i]['route'][j]
                    pickup_hub_index = self.manhattan_graph.get_hub_index_by_nodeid(pickup_nodeid)
                    dropoff_hub_index = self.manhattan_graph.get_hub_index_by_nodeid(dropoff_nodeid)
                    self.wait_till_departure_times[(pickup_hub_index,dropoff_hub_index)] = available_trips[i]['departure_time']
                    #self.wait_till_departure_times[pickup_hub_index,dropoff_hub_index] = 120
        
        #print(f"cost_edges: {edges}")
        nx.set_edge_attributes(self.G, edges, "cost")

        
        # edges = {}
        # for i in range(70):
        #     for j in range(70):
        #         if(i==2 and j==3 or i==3 and j==10 or i==10 and j==7):
        #             edges[(i,j,0)] = 5
        #         elif(i==j):
        #             edges[(i,j,0)] = 20
        #         else:
        #             edges[(i,j,0)] = 60
        # nx.set_edge_attributes(self.G, edges, "cost")      

    def add_remaining_distance_layer(self):

        pass 

