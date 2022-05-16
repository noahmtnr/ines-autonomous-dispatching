from ManhattanGraph import ManhattanGraph 
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
import timestamps_mapping



graph= ManhattanGraph('simple',70)
app = Flask(__name__)

input_example={'route': [1,2,3],'timestamps':[ datetime.datetime(2016, 5, 19, 23, 33, 46, 600000), datetime.datetime(2016, 5, 19, 23, 41, 6, 600000), datetime.datetime(2016, 5, 19, 23, 51, 6, 600000)]}
input_example2={'route': [1,4,3],'timestamps':[ datetime.datetime(2016, 5, 19, 23, 33, 46, 600000), datetime.datetime(2016, 5, 19, 23, 41, 6, 600000), datetime.datetime(2016, 5, 19, 23, 51, 6, 600000)]}
#input_example={'route': [3579432156, 42428782, 42428770, 595314103, 7004393049, 42428760, 42421745, 42428751, 42428746, 42428742, 42428737, 42428728, 42428725, 42428723, 42428720, 42428716, 42428714, 42428711, 42428709, 42428705, 42428701, 42428695, 42428689, 42422006, 42428682, 42428678, 42428674, 42428670, 42421809, 42428663, 42428661, 42428657, 42421775, 42428653, 42428065, 42428648, 42428645, 42428643, 42428640, 42428637, 42428634, 1061531448, 9177424867, 42428610, 42428604, 42428601, 42428598, 42428595, 42428590, 42428588, 42428579, 42428575, 42421972, 42428570, 1825841655, 1825841704, 1825841743, 42428332, 42428329, 42428328, 42428321, 42428315, 42428313, 42428312, 42428310, 42428308, 42428307, 42428305, 42439996, 42439994, 42439990, 42428297, 42439984, 42439981, 42437363, 42439972, 42439968, 42439964, 42433611, 42428268, 42428264, 42428258, 42430311, 42452973, 42434946, 42437644], 'timestamps': ['2016-05-19 23:04:00', '2016-05-19 23:04:07', '2016-05-19 23:04:14', '2016-05-19 23:04:21', '2016-05-19 23:04:28', '2016-05-19 23:04:35', '2016-05-19 23:04:42', '2016-05-19 23:04:49', '2016-05-19 23:04:56', '2016-05-19 23:05:03', '2016-05-19 23:05:10', '2016-05-19 23:05:17', '2016-05-19 23:05:24', '2016-05-19 23:05:31', '2016-05-19 23:05:38', '2016-05-19 23:05:45', '2016-05-19 23:05:52', '2016-05-19 23:06:00', '2016-05-19 23:06:07', '2016-05-19 23:06:14', '2016-05-19 23:06:21', '2016-05-19 23:06:28', '2016-05-19 23:06:35', '2016-05-19 23:06:42', '2016-05-19 23:06:49', '2016-05-19 23:06:56', '2016-05-19 23:07:03', '2016-05-19 23:07:10', '2016-05-19 23:07:17', '2016-05-19 23:07:24', '2016-05-19 23:07:31', '2016-05-19 23:07:38', '2016-05-19 23:07:45', '2016-05-19 23:07:52', '2016-05-19 23:08:00', '2016-05-19 23:08:07', '2016-05-19 23:08:14', '2016-05-19 23:08:21', '2016-05-19 23:08:28', '2016-05-19 23:08:35', '2016-05-19 23:08:42', '2016-05-19 23:08:49', '2016-05-19 23:08:56', '2016-05-19 23:09:03', '2016-05-19 23:09:10', '2016-05-19 23:09:17', '2016-05-19 23:09:24', '2016-05-19 23:09:31', '2016-05-19 23:09:38', '2016-05-19 23:09:45', '2016-05-19 23:09:52', '2016-05-19 23:10:00', '2016-05-19 23:10:07', '2016-05-19 23:10:14', '2016-05-19 23:10:21', '2016-05-19 23:10:28', '2016-05-19 23:10:35', '2016-05-19 23:10:42', '2016-05-19 23:10:49', '2016-05-19 23:10:56', '2016-05-19 23:11:03', '2016-05-19 23:11:10', '2016-05-19 23:11:17', '2016-05-19 23:11:24', '2016-05-19 23:11:31', '2016-05-19 23:11:38', '2016-05-19 23:11:45', '2016-05-19 23:11:52', '2016-05-19 23:12:00', '2016-05-19 23:12:07', '2016-05-19 23:12:14', '2016-05-19 23:12:21', '2016-05-19 23:12:28', '2016-05-19 23:12:35', '2016-05-19 23:12:42', '2016-05-19 23:12:49', '2016-05-19 23:12:56', '2016-05-19 23:13:03', '2016-05-19 23:13:10', '2016-05-19 23:13:17', '2016-05-19 23:13:24', '2016-05-19 23:13:31', '2016-05-19 23:13:38', '2016-05-19 23:13:45', '2016-05-19 23:13:52', '2016-05-19 23:14:00']}
#input_example={'route': [3579432156, 42428782, 42428770], 'timestamps': ['2016-05-19 23:04:00', '2016-05-19 23:04:07', '2016-05-19 23:04:14']}
#input_example={'route': [1,2,3,4], 'timestamps': ["2016-05-19 23:04:00","2016-05-19 23:04:10","2016-05-19 23:04:20","2016-05-19 23:04:30"]}
#input_example2={'route': [1,2,3,4], 'timestamps': ["2016-05-19 23:04:00","2016-05-19 23:04:10","2016-05-19 23:04:20","2016-05-19 23:04:30"]}

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
            print(graph.get_nodeid_by_hub_index(i))
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
                print(j)
                date= datetime.datetime.strptime(j,"%Y-%m-%d %H:%M:%S")
                date_str=date.strftime("%Y-%m-%d %H:%M:%S")
                #j = pd.to_datetime(j, format=date_format_str)
                all_timestamps.append(date_str)
        input.update({'timestamps': all_timestamps})
        print("ALL",all_timestamps, type(all_timestamps[0]))
        input.update({'route': nodes})

        


preprocess_input(inputs)
print(inputs)
input_example=inputs[0]
input_example2=inputs[1]

start_hub_coordinates=graph.get_coordinates_of_node(input_example["route"][0])
final_hub_coordinates=graph.get_coordinates_of_node(input_example["route"][len(input_example["route"])-1])



@app.route('/')
def index():
  return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    # answear = {}

    # args = request.args
    # start_node_long = args.get('pickup_long')
    # start_node_lat = args.get('pickup_lat')
    # start_date = args.get('start_date')
  
    # new_start_node_long = float(start_node_long or 0)
    # # print(new_start_node_long)
    # new_start_node_lat = float(start_node_lat or 0) 
    # # print(new_start_node_lat)
    # new_start_node = DataPreProcessing.getNearestNodeId(new_start_node_long, new_start_node_lat)
    # # print(new_start_node)

    # new_start_date = urllib.parse.unquote(start_date)
  
    # start_date_format = datetime.strptime(new_start_date, "%Y-%m-%d %H:%M:%S")
    # end_date_format = start_date_format + timedelta(minutes=10)
    # print(new_start_node, new_start_date, end_date_format)
    #myList = getTrips(new_start_node, new_start_date, end_date_format)
    # route = []
    # times = []
    # for trip in myList:
    route, times = input_example["route"], input_example["timestamps"]
    route2, times2 = input_example2["route"], input_example2["timestamps"]

    #   
    #answear[trip] = {'route': route, 'timestamps': times}
    print(times)
    print(type(times[0]))
    lines = buildLines(route, times,"blue")
    lines2=buildLines(route2, times2, "red")

    return buildFolium(lines,lines2)

def buildFolium(lines,lines2):
    m_events = folium.Map(
    location=[40.776676, -73.971321],control_scale=True,
    zoom_start=12,
)

    dir = r"templates/ciao.html"
    dirname = os.path.dirname(_file_)
    save_location = os.path.join(dirname, dir)
    

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": line["coordinates"],
            },
            "properties": {
                "times": line["dates"],
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
    features2 = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": line["coordinates"],
            },
            "properties": {
                "times": line["dates"],
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
    folium.Marker(location=start_hub_coordinates, icon=folium.Icon(color='red', prefix='fa', icon='flag-checkered')).add_to(m_events)
    folium.Marker(location=final_hub_coordinates, icon=folium.Icon(color='green', prefix='fa', icon='caret-right')).add_to(m_events)
   # print("properties times", len(features["properties"]["times"]))
   # print("geometry coordinates",len(features["geometry"]["coordinates"]))

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
      element["dates"] = [timestamps[i], timestamps[i+1]]
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
if __name__ == '_main_':
  
    app.run(debug=True)
