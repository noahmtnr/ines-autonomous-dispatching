import osmnx as ox
import networkx as nx




class StreetGraph:

    def __init__(self, filename):
        filepath = ("../../data/graph/%s.graphml") % (filename)
        self.graph = ox.load_graphml(filepath)
        self.graph = ox.add_edge_speeds(self.graph,fallback=30)
        self.graph = ox.add_edge_travel_times(self.graph)
