from Manhattan_Graph_Environment.graphs.ManhattanGraph import ManhattanGraph 
#from asyncio.windows_events import NULL
import sys
import matplotlib.pyplot as plot
from numpy import double
import osmnx as ox
import os
import urllib
import datetime
import folium
from folium import plugins
sys.path.insert(0,"")
from preprocessing.data_preprocessing import DataPreProcessing
from flask import Flask, jsonify, request, render_template, redirect
import pandas as pd
#import mysql.connector
import json
from folium.plugins import MarkerCluster
import osmnx as nx
import preprocessing.timestamps_mapping as timestamps_mapping



graph= ManhattanGraph('simple',70)
app = Flask(__name__)

input_example={'route': [1,2,2,6,5,5,5,7,9,15],'timestamps':[ datetime.datetime(2016, 5, 19, 23, 33, 46, 600000),datetime.datetime(2016, 5, 19, 23, 33, 46, 600000),datetime.datetime(2016, 5, 19, 23, 33, 46, 600000),datetime.datetime(2016, 5, 19, 23, 33, 46, 600000), datetime.datetime(2016, 5, 19, 23, 35, 46, 600000), datetime.datetime(2016, 5, 19, 23, 41, 46,600000), datetime.datetime(2016, 5, 19, 23, 46, 46, 600000),datetime.datetime(2016, 5, 19, 23, 56, 46, 600000),datetime.datetime(2016, 5, 19, 23, 33, 46, 600000),datetime.datetime(2016, 5, 19, 23, 33, 46, 600000)]}
input_example2={'route': [1,4,8,15],'timestamps':[datetime.datetime(2016, 5, 19, 23, 33, 46, 600000), datetime.datetime(2016, 5, 19, 23, 33, 46, 600000), datetime.datetime(2016, 5, 19, 23, 41, 56, 600000), datetime.datetime(2016, 5, 19, 23, 46, 46, 600000)]}
#input_example2={'route': [1,4,5],'timestamps':[ datetime.datetime(2016, 5, 19, 23, 33, 46, 600000), datetime.datetime(2016, 5, 19, 23, 41, 56, 600000), datetime.datetime(2016, 5, 19, 23, 46, 46, 600000)]}


real_input={'pickup_hub': 27, 'delivery_hub': 14, 'reward': -3182.772359, 'hubs': 29, 'route': [48, 19, 46, 47, 43, 15, 38, 24, 66, 4, 9, 62, 64, 3, 25, 58, 32, 25, 4, 37, 50, 50, 53, 67, 35, 15, 49, 65, 24, 17], 'time': '6:08:51.500000', 'dist': 182.77235900000005, 'time_until_deadline': datetime.datetime(2016, 5, 20, 4, 55, 8, 500000), 'timestamps': [datetime.datetime(2016, 5, 19, 23, 13, 24, 400000), datetime.datetime(2016, 
5, 19, 23, 22, 50, 100000), datetime.datetime(2016, 5, 19, 23, 33, 46, 600000), datetime.datetime(2016, 5, 19, 23, 41, 6, 600000), datetime.datetime(2016, 5, 19, 23, 51, 30, 600000), datetime.datetime(2016, 5, 20, 
0, 8, 15, 500000), datetime.datetime(2016, 5, 20, 0, 27, 47), datetime.datetime(2016, 5, 20, 0, 45, 49, 200000), datetime.datetime(2016, 5, 20, 0, 54, 21, 100000), datetime.datetime(2016, 5, 20, 1, 6, 52, 700000), 
datetime.datetime(2016, 5, 20, 1, 17, 37, 200000), datetime.datetime(2016, 5, 20, 1, 32, 20, 300000), datetime.datetime(2016, 5, 20, 1, 40, 23), datetime.datetime(2016, 5, 20, 1, 52, 2, 800000), datetime.datetime(2016, 5, 20, 2, 6, 11, 900000), datetime.datetime(2016, 5, 20, 2, 16, 44, 900000), datetime.datetime(2016, 5, 20, 2, 31, 42, 800000), datetime.datetime(2016, 5, 20, 2, 42, 43, 400000), datetime.datetime(2016, 5, 20, 2, 55, 54, 800000), datetime.datetime(2016, 5, 20, 3, 17, 40, 700000), datetime.datetime(2016, 5, 20, 3, 32, 59, 600000), datetime.datetime(2016, 5, 20, 3, 37, 59, 600000), datetime.datetime(2016, 5, 20, 3, 49, 1), datetime.datetime(2016, 5, 20, 3, 56, 5, 200000), datetime.datetime(2016, 5, 20, 4, 12, 37, 400000), datetime.datetime(2016, 5, 20, 4, 30, 44, 300000), datetime.datetime(2016, 5, 20, 4, 41, 23, 700000), datetime.datetime(2016, 5, 20, 4, 53, 24, 400000), datetime.datetime(2016, 5, 20, 5, 2, 9, 800000), datetime.datetime(2016, 5, 20, 5, 12, 51, 500000)]}

inputs=[input_example,input_example2] 
def preprocess_input(inputs):
    for input in inputs:
        date_format_str = '%Y-%m-%d %H:%M:%S.%f'
        nodes=[]
        all_timestamps=[]
        route_withids=[]
        for i in input["route"]:
            route_withids.append(graph.get_nodeid_by_hub_index(i))
        input.update({'route': route_withids})
        for i in range(len(input['route'])-1):
            route = nx.shortest_path(graph.inner_graph,input['route'][i] , input['route'][i+1])
            for j in route:
                nodes.append(j)
            duration=(input['timestamps'][i+1]-input['timestamps'][i]).total_seconds()
            timestamps=timestamps_mapping.map_nodes_to_timestaps_to_list(route,input["timestamps"][i],input["timestamps"][i+1],duration)
            for j in timestamps:
                #j=j.strftime("%Y-%m-%d %H:%M:%S")
                #print(timestamps)
                date= datetime.datetime.strptime(j,"%Y-%m-%d %H:%M:%S")
                date_str=date.strftime("%Y-%m-%d %H:%M:%S")
                j = pd.to_datetime(j, format=date_format_str)
                all_timestamps.append(date_str)
        input.update({'timestamps': all_timestamps})
        input['route_nodes']=nodes

        

preprocess_input(inputs)
input_example=inputs[0]
input_example2=inputs[1]

start_hub_coordinates=graph.get_coordinates_of_node(input_example["route"][0])
final_hub_coordinates=graph.get_coordinates_of_node(input_example["route"][len(input_example["route"])-1])




@app.route('/')
def index():
  return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    
    route, times, hubs = input_example["route_nodes"], input_example["timestamps"], input_example["route"]
    route2, times2, hubs2 = input_example2["route_nodes"], input_example2["timestamps"], input_example2["route"]


    lines = buildLines(route, times,"blue")
    lines2=buildLines(route2, times2, "red")

    return buildFolium(lines,lines2, route, route2, hubs, hubs2)

def buildFolium(lines,lines2,route, route2, hubs, hubs2):
    m_events = folium.Map(
    location=[40.776676, -73.971321],control_scale=True,
    zoom_start=12,
)

    dir = r"templates/ciao.html"
    dirname = os.path.dirname(__file__)
    save_location = os.path.join(dirname, dir)
    
    
    edges=len(lines)
    
    mocked_times=['2020-01-19 11:00:00'] * edges
    for i in range(len(mocked_times)-1):
       new_date= datetime.datetime.strptime(mocked_times[i], '%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=5)
       mocked_times[i+1]=str(new_date)
    

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": line["coordinates"],
            },
            "properties": {
                #"times": line["dates"],
                "times": mocked_times,
                "style": {
                    "color": line["color"],
                    "weight": line["weight"] if "weight" in line else 5,
                },
            #'icon': 'marker',
                # time must be like the following format : '2018-12-01T05:53:00'
            'popup': '<html> <head></head>  <body> comments </body> </html>'
            },
        }
        for line in lines]
    edges2=len(lines2)


    mocked_times2=['2020-01-19 11:00:00'] * edges2
    for i in range(len(mocked_times2)-1):
       new_date= datetime.datetime.strptime(mocked_times2[i], '%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=5)
       mocked_times2[i+1]=str(new_date)
    

    

    features2 = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": line["coordinates"],
            },
            "properties": {
                #"times": line["dates"],
                "times": mocked_times2,
                "style": {
                    "color": line["color"],
                    "weight": line["weight"] if "weight" in line else 5,
                },
            #'icon': 'marker',
                # time must be like the following format : '2018-12-01T05:53:00'
            'popup': '<html> <head></head>  <body> comments </body> </html>'
            },
        }
        for line in lines2]
    
    waited = 0
    waited2=False
    for i in range(len(hubs)-1):
        # case 1: if next hub is the same as current hub (i.e. wait) and we did not wait before on the same hub
        if (hubs[i]==hubs[i+1] and waited == 0):
            waited2=True
            waited = 1
            folium.Marker(location=graph.get_coordinates_of_node(hubs[i]), icon=folium.Icon(color='orange', prefix='fa', icon='cube'), popup = f'WAIT {waited}').add_to(m_events)
        # case 2: if next hub is the same as current hub (i.e. wait) and we did already wait at least once at the current hub
        elif (hubs[i]==hubs[i+1] and waited > 0):
            waited2=True
            waited += 1
            folium.Marker(location=graph.get_coordinates_of_node(hubs[i]), icon=folium.Icon(color='orange', prefix='fa', icon='cube'), popup = f'WAIT {waited}').add_to(m_events)
        # case 3: did not wait at the current hub 
        elif(waited2==False):
            waited = 0
            folium.Marker(location=graph.get_coordinates_of_node(hubs[i]), icon=folium.Icon(color='black', prefix='fa', icon='cube')).add_to(m_events)
        # case 4: waited at the current hub and now moving to next hub
        else:
            folium.Marker(location=graph.get_coordinates_of_node(hubs[i]), icon=folium.Icon(color='orange', prefix='fa', icon='cube'), popup = f'WAIT {waited}').add_to(m_events)
            waited = 0
            waited2=False
    waited = 0
    waited2=False
    for i in range(len(hubs2)-1):
        # if next hub is the same as current hub (i.e. wait) and we did not wait before on the same hub
        if (hubs2[i]==hubs2[i+1] and waited == 0):
            waited2=True
            waited = 1
            folium.Marker(location=graph.get_coordinates_of_node(hubs2[i]), icon=folium.Icon(color='orange', prefix='fa', icon='cube'), popup = f'WAIT {waited}').add_to(m_events)
        # if next hub is the same as current hub (i.e. wait) and we did already wait at least once at the current hub
        elif (hubs2[i]==hubs2[i+1] and waited > 0):
            waited2=True
            waited += 1
            folium.Marker(location=graph.get_coordinates_of_node(hubs2[i]), icon=folium.Icon(color='orange', prefix='fa', icon='cube'), popup = f'WAIT {waited}').add_to(m_events)
        # case 3 
        elif(waited2==False):
            waited = 0
            folium.Marker(location=graph.get_coordinates_of_node(hubs2[i]), icon=folium.Icon(color='black', prefix='fa', icon='cube')).add_to(m_events)
        # case 4: waited at the current hub and now moving to next hub
        else:
            folium.Marker(location=graph.get_coordinates_of_node(hubs2[i]), icon=folium.Icon(color='orange', prefix='fa', icon='cube'), popup = f'WAIT {waited}').add_to(m_events)
            waited = 0
            waited2=False
    folium.Marker(location=start_hub_coordinates, icon=folium.Icon(color='red', prefix='fa', icon='flag-checkered')).add_to(m_events)
    folium.Marker(location=final_hub_coordinates, icon=folium.Icon(color='green', prefix='fa', icon='caret-right')).add_to(m_events)


    plugins.TimestampedGeoJson(
         {
             "type": "FeatureCollection",
            "features": features,
        },
        period="PT1M",
        add_last_point=False,
    ).add_to(m_events)

    plugins.TimestampedGeoJson(
         {
             "type": "FeatureCollection",
            "features": features2,
        },
        period="PT1M",
        add_last_point=False,
    ).add_to(m_events)


    m_events.save(save_location)
    return render_template("ciao.html")
    #m_events


def buildLines(routes, timestamps,color):
    element = {}
    list_elements = []
    for i in range(len(routes)-1):
      element = {}
      element["coordinates"]= [DataPreProcessing.get_coordinates_of_node(routes[i]), DataPreProcessing.get_coordinates_of_node(routes[i+1])]
      #element["dates"] = [timestamps[i], timestamps[i+1]]
      element["color"] =  color
      list_elements.append(element)
    #print(list_elements)
    return list_elements

@app.route('/addTrip', methods=['POST'])
def addTrip():
    #content_type = request.headers.get('Content-Type')
  
    data = request.form
    pickup_longitude = double(data.get('pickup_long') or 0)
    pickup_latitude = double(data.get('pickup_lat') or 0)
    dropoff_longitude = double(data.get('dropoff_long') or 0)
    dropoff_latitude = double(data.get('dropoff_lat') or 0)
    # print(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
    pickup_datetime = datetime.strptime(data.get('pickup_date') or "", "%Y-%m-%d %H:%M:%S")
    dropoff_datetime = datetime.strptime(data.get('dropoff_date') or "", "%Y-%m-%d %H:%M:%S")
    trip_duration = (dropoff_datetime - pickup_datetime).total_seconds()
    route, timestamps, route_length, pickup_node, dropoff_node = DataPreProcessing.map_oneRoute_to_oneTrip_with_timestamps(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, pickup_datetime, dropoff_datetime, trip_duration)
    
    insertIntoTrips(data.get('id'), data.get('vendor_id'), pickup_datetime, dropoff_datetime, data.get('passenger_count'), pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, "N", trip_duration, pickup_node, dropoff_node, route_length, data.get('provider'), (float(data.get('total_price') or 0)))
    timestamps = timestamps.replace(': \'',' : \'')
    timestamps = timestamps.strip("{}")
    # print("Timestamps: ", timestamps)
    insertIntoTripsRoutes(data.get('id'), timestamps)
    return "Success"
   
    

# driver function
if __name__ == '__main__':
  
    app.run(debug=True)