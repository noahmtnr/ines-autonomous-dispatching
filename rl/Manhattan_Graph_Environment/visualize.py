from ManhattanGraph import ManhattanGraph 

#from asyncio.windows_events import NULL
import sys
import matplotlib.pyplot as plot
from numpy import double
import osmnx as ox
import os
from datetime import datetime
import urllib
from datetime import datetime, timedelta
from folium import plugins, folium
sys.path.insert(0,"")
#from preprocessing.data_preprocessing import DataPreProcessing
from flask import Flask, jsonify, request, render_template, redirect
import pandas as pd
#import mysql.connector
import json

from datetime import datetime

graph= ManhattanGraph('simple',70)
app = Flask(__name__)

input_example={'route': [1,2,3,4], 'timestamps': ["2016-05-19 23:04:00","2016-05-19 23:04:10","2016-05-19 23:04:20","2016-05-19 23:04:30"]}
route_withids=[]
for i in input_example["route"]:
    route_withids.append(graph.get_nodeid_by_hub_index(i))
input_example.update({'route': route_withids})



start_hub_coordinates=graph.get_coordinates_of_node(input_example["route"][0])
final_hub_coordinates=graph.get_coordinates_of_node(input_example["route"][len(input_example["route"])-1])



@app.route('/')
def index():
  return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    answear = {}

    args = request.args
    start_node_long = args.get('pickup_long')
    start_node_lat = args.get('pickup_lat')
    start_date = args.get('start_date')
  
  
    new_start_node_long = float(start_node_long or 0)
    # print(new_start_node_long)
    new_start_node_lat = float(start_node_lat or 0) 
    # print(new_start_node_lat)
    new_start_node = DataPreProcessing.getNearestNodeId(new_start_node_long, new_start_node_lat)
    # print(new_start_node)

    new_start_date = urllib.parse.unquote(start_date)
  
    start_date_format = datetime.strptime(new_start_date, "%Y-%m-%d %H:%M:%S")
    end_date_format = start_date_format + timedelta(minutes=10)
    # print(new_start_node, new_start_date, end_date_format)
    myList = getTrips(new_start_node, new_start_date, end_date_format)
    route = []
    times = []
    for trip in myList:
        route, times = getRouteFromTrip(trip)
        answear[trip] = {'route': route, 'timestamps': times}
    lines = buildLines(route, times)

    return buildFolium(lines)

def buildFolium(lines):
  m_events = folium.Map(
    location=[-73.971321,40.776676],control_scale=True,
    zoom_start=12,
)

  dir = r"templates\m_events.html"
  dirname = os.path.dirname(__file__)
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
            'icon': 'marker',
            # time must be like the following format : '2018-12-01T05:53:00'
            'popup': '<html> <head></head>  <body> comments </body> </html>'
        },
    }
    for line in lines]
    folium.Marker(location=start_hub_coordinates, icon=folium.Icon(color='red', prefix='fa', icon='flag-checkered')).add_to(m_events)
    folium.Marker(location=final_hub_coordinates, popup = f"Pickup time: {self.pickup_time.strftime('%m/%d/%Y, %H:%M:%S')}", icon=folium.Icon(color='lightblue', prefix='fa', icon='caret-right')).add_to(m_events)

  plugins.TimestampedGeoJson(
    {
        "type": "FeatureCollection",
        "features": features,
    },
    period="PT1M",
    add_last_point=True,
  ).add_to(m_events)
  m_events.save(save_location)
  return render_template("m_events.html")
  #m_events


def buildLines(routes, timestamps):
    element = {}
    list_elements = []
    for i in range(len(routes)-1):
      element = {}
      element["coordinates"]= [DataPreProcessing.get_coordinates_of_node(routes[i]), DataPreProcessing.get_coordinates_of_node(routes[i+1])]
      element["dates"] = [timestamps[i], timestamps[i+1]]
      element["color"] =  "blue"
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
  
    app.run(debug = True)