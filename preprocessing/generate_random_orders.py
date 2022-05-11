import numpy as np
from datetime import datetime, timedelta
import random
import sys
import csv

sys.path.insert(0,"")

#here sth needs to be fixed
from rl.Manhattan_Graph_Environment.gym_graphenv.envs.GraphworldManhattan import GraphEnv
env=GraphEnv()

random_orders=[]

for i in range(1000):
    final_hub = env.graph.get_nodeids_list().index(random.sample(env.hubs,1)[0])
    start_hub = env.graph.get_nodeids_list().index(random.sample(env.hubs,1)[0])
    
    pickup_month =  np.random.randint(5)+1
    pickup_day = np.random.randint(28)+1
    pickup_hour =  np.random.randint(24)
    pickup_minute = np.random.randint(60) 

    pickup_time = datetime(2016,pickup_month,pickup_day,pickup_hour,pickup_minute,0)
    deadline=pickup_time+timedelta(hours=12)

    random_order={'pickup_node_id':start_hub, 'delivery_node_id':final_hub, 'pickup_timestamp':pickup_time,'delivery_timestamp':deadline}
    random_orders.append(random_order)

field_names= ['pickup_node_id', 'delivery_node_id', 'pickup_timestamp','delivery_timestamp']

with open('random_orders.csv', 'w', newline='') as csvfile:
     writer = csv.DictWriter(csvfile, fieldnames=field_names)
     writer.writeheader()
     writer.writerows(random_orders)