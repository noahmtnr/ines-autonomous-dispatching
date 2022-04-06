import osmnx as ox
import networkx as nx
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta


class StreetGraph:

    def __init__(self, filename, num_trips, fin_hub, num_hubs):
        filepath = ("../../data/graph/%s.graphml") % (filename)
        self.inner_graph = ox.load_graphml(filepath)
        self.inner_graph = ox.add_edge_speeds(self.inner_graph,fallback=30)
        self.inner_graph = ox.add_edge_travel_times(self.inner_graph)
        ox.utils_graph.remove_isolated_nodes(self.inner_graph)
        self.generate_random_trips(num_trips)
        self.generate_hubs(fin_hub, num_hubs)


    def generate_hubs(self, fin_hub, num_hubs: int = 5):
        """Generates random bubs within the graph

        Args:
            fin_hub (int): index_id of final hub
            num_hubs (int, optional): Number of hubs to create. Defaults to 5.

        Returns:
            self.hubs(list): List of hubs in graph
        """
        random.seed(42)
        hubs = random.sample(self.nodes(),num_hubs) 
        final_hub = self.get_nodeid_by_index(fin_hub)
        if(final_hub not in hubs):
            hubs.append(final_hub)
        self.hubs = hubs

        return self.hubs

    def generate_random_trips(self, n: int = 2000):
        """Generates random trips within the graph and stores them in self.trips. The trips are randomly spread across January 2022.

        Args:
            n (int, optional): Number of trips to be generated. Defaults to 2000.
        """
        random.seed(42)

        graph = self.inner_graph

        gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)

        df = pd.DataFrame(gdf_nodes)
        df.reset_index(inplace=True)

        sequence = df["osmid"]
        sequence = sequence.to_numpy()
        sequence = sequence.tolist()

        random_nodes = random.choices(sequence, k=n)
        random_nodes2 = random.choices(sequence, k=n)

        trips = pd.DataFrame()
        count = 0
        trip_id = []
        nodeA = []
        nodeB = []
        for i in range(n):
            if random_nodes[i]==random_nodes2[i]:
                random_nodes2[i]=random.choice(sequence)
            trip_id.append(i)
        
        trips["tripid"] = trip_id
        trips["pickup_node"] = random_nodes
        trips["dropoff_node"] = random_nodes2

        pickup_day = [1 for i in range(n)]
        pickup_hour =  np.random.randint(24, size=n)
        pickup_minute = np.random.randint(60, size=n)
        pickup_datetimes = []

        for i in range(len(pickup_hour)):
            pickup_datetime=datetime(2022,1,1,pickup_hour[i],pickup_minute[i],0)
            pickup_datetimes.append(pickup_datetime)

        trips['pickup_day'] = pickup_day
        trips['pickup_hour'] = pickup_hour
        trips['pickup_minute'] = pickup_minute
        trips['pickup_datetime'] = pickup_datetimes

        routes = []
        dropoff_datetimes = []
        node_timestamps = []
        trip_durations = []
        for index, row in trips.iterrows():
            try:
                route = ox.shortest_path(
                    graph, trips.loc[index, "pickup_node"], trips.loc[index, "dropoff_node"], weight='travel_time'
                )
                routes.append(route)
                travel_times = ox.utils_graph.get_route_edge_attributes(graph,route,attribute='travel_time')
                pickup_datetime = trips.loc[index, "pickup_datetime"]
                dropoff_datetime = pickup_datetime + timedelta(seconds=sum(travel_times))
                trip_duration = (dropoff_datetime-pickup_datetime).total_seconds()
                
                timestamps_dict = map_nodes_to_timestaps(route, pickup_datetime, dropoff_datetime, trip_duration)

                dropoff_datetimes.append(dropoff_datetime)
                trip_durations.append(trip_duration)
                node_timestamps.append(timestamps_dict)

            except Exception as e:
                print(e)
                trips.drop(index, inplace=True)

        trips["route"] = routes
        trips["dropoff_datetime"] = dropoff_datetimes
        trips["trip_duration"] = trip_durations
        trips["node_timestamps"] = node_timestamps

        # compute trip length and add to csv
        # generate random passenger count between 1 and 4 and add to csv
        route_length_column=[]
        passenger_count_column=[]
        for i in len(trips.index):
            current_route = trips.iloc[i]["route"]
            route_length = 0
            for j in len(current_route)-1:
                route_length += ox.distance.great_circle_vec(self.inner_graph[current_route[j]]['y'], self.inner_graph[current_route[j]]['x'],
                self.inner_graph[current_route[j+1]]['y'], self.inner_graph[current_route[j+1]]['x'])
            route_length_column.append(route_length)
            passenger_count_column.append(random.randint(1,4))

        trips["route_length"]=route_length_column
        trips["passenger_count"]=passenger_count_column

        
        # add mobility providers randomly
        provider_column=[]
        totalprice_column=[]
        x = pd.read_csv("Provider.csv")
        for i in len(trips.index):
            provider_id = x.sample(axis=0).loc[0]
            provider_column.append(provider_id)
            basic_price = x.iloc['basic_cost'][x.index[x['id']==provider_id]]
            km_price = x.iloc['cost_per_km'][x.index[x['id']==provider_id]]
            leng = x.iloc['route_length'][x.index[x['id']==provider_id]]
            # note that internal distance unit is meters in OSMnx
            total_price = basic_price + km_price*leng/1000
            totalprice_column.append(total_price)
        trips["provider"]=provider_column
        trips["total_price"]=totalprice_column



        trips.to_csv("trips_meinheim.csv")
        self.trips = trips

        return self.trips

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

def map_nodes_to_timestaps(route_nodes, pickup_time, dropoff_time, duration):
    timestamps = []
    date_format_str = '%Y-%m-%d %H:%M:%S.%f'
    start_time = pd.to_datetime(pickup_time, format=date_format_str)
    end_time = pd.to_datetime(dropoff_time, format=date_format_str)

    if (len(route_nodes) > 1):
        time_between_nodes = duration / (len(route_nodes) - 1)
        delta = timedelta(seconds=time_between_nodes)
        timestamps = timestamp_range(start_time, end_time, delta, route_nodes)
    else:
        timestamps.append(dropoff_time)

    timestamps_mapping = dict(zip(route_nodes, timestamps))
    return timestamps_mapping


def timestamp_range(start_time, end_time, delta, route_nodes):
    timestamps = []

    while start_time < end_time:
        start_time_formatted = str(start_time.strftime("%Y-%m-%d %H:%M:%S"))
        timestamps.append(start_time_formatted)
        start_time += delta

    if len(timestamps) == len(route_nodes):
        end_time_formatted = str(end_time.strftime("%Y-%m-%d %H:%M:%S"))
        timestamps[-1] = str(end_time_formatted)
    else:
        end_time_formatted = str(end_time.strftime("%Y-%m-%d %H:%M:%S"))
        timestamps.append(end_time_formatted)
    return timestamps