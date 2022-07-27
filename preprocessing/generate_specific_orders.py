import os 
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import dash
from dash import html
from dash import dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
#import dash_bootstrap_components as dbc
import sys
import ast
sys.path.insert(0,"")
#from graphs.ManhattanGraph import ManhattanGraph
import pickle
from Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv
env = GraphEnv(use_config=True)
import csv


#manual found redundant hubs 
pickup_hubs=[21,74,87,91,36,23,68,4,4,17,17]
delivery_hubs=[5,53,22,31,13,57,80,84,17,25,33]
pickup_timestamp=["2016-01-05 20:46:00","2016-01-13 00:20:00","2016-01-07 09:02:00","2016-01-06 03:19:00","2016-01-07 17:31:00","2016-01-09 15:15:00","2016-01-05 11:04:00", "2016-01-05 15:46:00", "2016-01-05 15:46:00", "2016-01-05 17:43:00", "2016-01-05 17:43:00"]
#delivery_timestamp=["2016-01-06 08:46:00","2016-01-13 12:20:00","2016-01-07 21:02:00","2016-01-06 15:19:00","2016-01-08 05:31:00","2016-01-10 03:15:00","2016-01-05 23:04:00","2016-01-06 03:46:00", "2016-01-06 03:46:00", "2016-01-06 05:43:00", "2016-01-06 05:43:00"]
date_format_str = '%Y-%m-%d %H:%M:%S'
test_orders=[]



for i in range(len(pickup_hubs)):
     pickup_hubid=env.manhattan_graph.get_nodeid_by_hub_index(pickup_hubs[i])
     delivery_hubid=env.manhattan_graph.get_nodeid_by_hub_index(delivery_hubs[i])
     deadline=datetime.strptime(pickup_timestamp[i], date_format_str)+timedelta(hours=24)
     test_order={'pickup_node_id':pickup_hubid, 'delivery_node_id':delivery_hubid, 'pickup_timestamp':pickup_timestamp[i],'delivery_timestamp':deadline}
     test_orders.append(test_order)
print(test_orders)

field_names= ['pickup_node_id', 'delivery_node_id', 'pickup_timestamp','delivery_timestamp']

with open('test_orders.csv', 'w', newline='') as csvfile:
     writer = csv.DictWriter(csvfile, fieldnames=field_names)
     writer.writeheader()
     writer.writerows(test_orders)




