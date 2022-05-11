import networkx as nx
import osmnx as ox
from ManhattanGraph import ManhattanGraph

class LearnGraph:
        
    def __init__(self, n):
        self.G=nx.complete_graph(n,nx.MultiDiGraph)
        for node in self.G.nodes():
            self.G.add_edge(node, node)

        ox.save_graphml(self.G, filepath="./data/graph/learn.graphml")

    def adjacency_matrix(self):
        return nx.adjacency_matrix(self.G, weight=None)

    def add_travel_distance_layer(self):
        self.manhattan_graph = ManhattanGraph(filename='simple')

        #for each edge in learn_graph calculate travel time and add as edge attribute with
        nx.set_edge_attributes(G, bb, "travel_distance") 

    def add_travel_cost_layer(self):
        self.manhattan_graph.setup_trips(self.START_TIME)


graph = LearnGraph(70)
graph.adjacency_matrix()
graph.add_travel_distance_layer()
