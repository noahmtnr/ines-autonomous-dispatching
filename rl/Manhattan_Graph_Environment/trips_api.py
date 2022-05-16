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
from rl.Manhattan_Graph_Environment.ManhattanGraph import ManhattanGraph

from preprocessing.data_preprocessing import DataPreProcessing
from flask import Flask, jsonify, request, render_template, redirect
import pandas as pd
import mysql.connector
from LearnGraph import LearnGraph
import json

from datetime import datetime
# creating a Flask app
mydb = mysql.connector.connect(
host="localhost",
user="Denisa",
password="Denisa_1700",
database="mannheimprojekt",
auth_plugin='mysql_native_password')
mycursor = mydb.cursor()


def getTrips(start_node, start_date, end_date):
   
    sql = "select id from TRIPS_ROUTES where route_node = %s and date_time between %s and %s"
    val = (start_node, start_date, end_date)
    tripsId_list=[]
    mycursor.execute(sql, val)
    for result in mycursor: 
        tripsId_list.append(result[0])
    return tripsId_list

def getRouteFromTrip(trip_id):
  sql = "select route_node, date_time from mannheimprojekt.TRIPS_ROUTES where id = %s order by date_time"
  val = (trip_id,)
  nodes_list=[]
  time_list = []
  mycursor.execute(sql, val)
  for result in mycursor: 
    nodes_list.append(result[0])
    time_list.append(result[1].strftime("%Y-%m-%d %H:%M:%S"))
  return nodes_list, time_list

def insertIntoTripsRoutes(tripId, listTrips):
  for element in listTrips.split(', '):
      a, b = element.split(' : ') 
      #new_b = datetime.fromisoformat(b)
      timestamp_b = datetime.strptime(str(b),"'%Y-%m-%d %H:%M:%S'")
      sql = "INSERT INTO TRIPS_ROUTES VALUES (%s, %s, %s)"
      val = (tripId, a, timestamp_b)
      mycursor.execute(sql, val)
  mydb.commit()

def insertIntoTrips(id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, store_and_fwd_flag, trip_duration, pickup_node, dropoff_node, route_length, provider, total_price):
  sql = "INSERT INTO TRIPS VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
  val = (id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, store_and_fwd_flag, trip_duration, pickup_node, dropoff_node, route_length, provider, total_price)
  mycursor.execute(sql, val)
  mydb.commit()

app = Flask(__name__)
  
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
      # print(type(times[0]))
      answear[trip] = {'route': route, 'timestamps': times}
    print(route, times)
    lines = buildLines(route, times)
    print(DataPreProcessing.get_coordinates_of_node(42444043))
    return buildFolium(lines)

@app.route('/order', methods=['GET'])
def addOrder():  
    data = request.args
    pickup_longitude = double(data.get('pickup_long') or 0)
    pickup_latitude = double(data.get('pickup_lat') or 0)
    dropoff_longitude = double(data.get('dropoff_long') or 0)
    dropoff_latitude = double(data.get('dropoff_lat') or 0)
    print(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
    pickup_datetime = datetime.strptime(data.get('start_date') or "", "%Y-%m-%d %H:%M:%S")
    dropoff_datetime = datetime.strptime(data.get('delivery_date') or "", "%Y-%m-%d %H:%M:%S")
    manhattangraph = ManhattanGraph(filename='simple', num_hubs=70)
    learngraph = LearnGraph(70, manhattangraph, 5)
    # start_node = DataPreProcessing.get_node_index_by_coordinates(pickup_longitude, pickup_latitude)
    # final_node = DataPreProcessing.get_node_index_by_coordinates(dropoff_longitude, dropoff_latitude)
    start_node = learngraph.getNearestNodeId(pickup_longitude,pickup_latitude)
    final_node = learngraph.getNearestNodeId(dropoff_longitude,dropoff_latitude)
    
    print(start_node, final_node)
    order={"pickup_node":start_node,"delivery_node":final_node,"pickup_timestamp":pickup_datetime , "delivery_timestamp":dropoff_datetime}

    result = proceed_order_random(order)
    lines = buildLines(result["route"],result["timestamps"])
    return buildFolium(lines)

def buildFolium(lines):
  graph = ox.io.load_graphml("data/graph/simple.graphml")
  plot = ox.plot_graph_folium(graph,fit_bounds=True, weight=2, color="#333333")
  m_events = folium.Map(
    location=[-73.971321,40.776676],control_scale=True,
    zoom_start=12,
)

  dir = r"templates\ciao.html"
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

  plugins.TimestampedGeoJson(
    {
        "type": "FeatureCollection",
        "features": features,
    },
    period="PT1M",
    add_last_point=True,
  ).add_to(m_events)
  m_events.save(save_location)
  return render_template("ciao.html")
  #m_events


def buildLines(routes, timestamps):
    element = {}
    list_elements = []
    for i in range(len(routes)-1):
      element = {}
      element["coordinates"]= [DataPreProcessing.get_coordinates_of_node(routes[i]), DataPreProcessing.get_coordinates_of_node(routes[i+1])]
      element["dates"] = [timestamps[i], timestamps[i+1]]
      print("Helo",[timestamps[i], timestamps[i+1]], type(timestamps[i]) )
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