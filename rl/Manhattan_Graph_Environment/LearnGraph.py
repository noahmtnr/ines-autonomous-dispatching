import networkx as nx
import osmnx as ox
from ManhattanGraph import ManhattanGraph
import numpy as np

class LearnGraph:
        
    def __init__(self, n_hubs: int, manhattan_graph, final_hub):
        self.G=nx.complete_graph(n_hubs,nx.MultiDiGraph)
        self.manhattan_graph = manhattan_graph
        self.final_hub = final_hub
        for node in self.G.nodes():
            self.G.add_edge(node, node)
        self.wait_till_departure_times = {}#np.zeros((70,70))
        self.list_hubs = self.manhattan_graph.hubs
        ox.save_graphml(self.G, filepath="./data/graph/learn.graphml")

    def adjacency_matrix(self, layer: str = None):
        return nx.to_numpy_array(self.G, weight=layer)

    def add_travel_distance_layer(self):

        #for each edge in learn_graph calculate travel time and add as edge attribute with
        nx.set_edge_attributes(self.G, bb, "travel_distance") 

    def add_travel_cost_layer(self, available_trips):
        # self.manhattan_graph.setup_trips(self.START_TIME) #needed later for retrieving currently available trips
        # nx.set_edge_attributes(self.G, 100, "cost") 
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
                if(available_trips[i]['route'][j] in self.list_hubs):
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

    def add_remaining_distance_layer(self):
        distance_edges = {}
        for i in range(70):
            for j in range(70):
                pickup_nodeid = self.manhattan_graph.get_nodeid_by_hub_index(i)
                dropoff_nodeid = self.manhattan_graph.get_nodeid_by_hub_index(j)
                if(i==j):
                    path_travelled = ox.shortest_path(self.manhattan_graph.inner_graph, pickup_nodeid, self.final_hub, weight='travel_time')
                    dist_travelled = ox.utils_graph.get_route_edge_attributes(self.manhattan_graph.inner_graph,path_travelled,attribute='length')
                    distance_edges[(i,j)] = dist_travelled
                else:
                    route_to_intermediate_hub = ox.shortest_path(self.manhattan_graph.inner_graph, pickup_nodeid, dropoff_nodeid, weight='travel_time')
                    route_from_intermediate_to_final = ox.shortest_path(self.manhattan_graph.inner_graph, dropoff_nodeid, self.final_hub, weight='travel_time')
                    #total_route = route_to_intermediate_hub + route_from_intermediate_to_final
                    dist_travelled_intermediate = ox.utils_graph.get_route_edge_attributes(self.manhattan_graph.inner_graph, route_to_intermediate_hub, attribute='length')
                    dist_travelled_final = ox.utils_graph.get_route_edge_attributes(self.manhattan_graph.inner_graph, route_from_intermediate_to_final, attribute='length')
                    dist_travelled = dist_travelled_intermediate + dist_travelled_final
                    distance_edges[(i,j)] = dist_travelled

        #nx.set_edge_attributes(self.G, distance_edges, "remaining_distance")
        print(distance_edges)

