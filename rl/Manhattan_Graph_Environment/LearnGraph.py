import networkx as nx
import osmnx as ox
from ManhattanGraph import ManhattanGraph
import numpy as np

class LearnGraph:
        
    def __init__(self, n_hubs: int):
        self.G=nx.complete_graph(n_hubs,nx.MultiDiGraph)
        for node in self.G.nodes():
            self.G.add_edge(node, node)

        self.add_travel_cost_layer()
        ox.save_graphml(self.G, filepath="./data/graph/learn.graphml")

    def adjacency_matrix(self, layer: str = None):
        return nx.to_numpy_array(self.G, weight=layer)

    def add_travel_distance_layer(self):
        self.manhattan_graph = ManhattanGraph(filename='simple')

        #for each edge in learn_graph calculate travel time and add as edge attribute with
        nx.set_edge_attributes(self.G, bb, "travel_distance") 

    def add_travel_cost_layer(self):
        # self.manhattan_graph.setup_trips(self.START_TIME) #needed later for retrieving currently available trips
        # nx.set_edge_attributes(self.G, 100, "cost") 
        edges = {}
        for i in range(70):
            for j in range(70):
                if(i==2 and j==3 or i==3 and j==10 or i==10 and j==7):
                    edges[(i,j,0)] = 5
                elif(i==j):
                    edges[(i,j,0)] = 20
                else:
                    edges[(i,j,0)] =60
        nx.set_edge_attributes(self.G, edges, "cost")       

graph = LearnGraph(70)
