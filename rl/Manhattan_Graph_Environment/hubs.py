from ManhattanGraph import ManhattanGraph
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

manhattan_graph = ManhattanGraph(filename='simple', num_hubs=52)
current_hubs = manhattan_graph.hubs
print(current_hubs)

top_nodes = pd.read_csv('top_nodes.csv')
print(top_nodes)