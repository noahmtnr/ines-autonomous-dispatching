import sys
sys.path.insert(0,"")
from preprocessing.data_preprocessing import DataPreProcessing
# Using flask to make an api
# import necessary libraries and functions
from flask import Flask, jsonify, request
import pandas as pd
# from database_connection import getAvailableTrips
# from rl.Manhattan_Graph_Environment.database_connection import getAvailableTrips
import mysql.connector
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
  
@app.route('/search', methods=['GET'])
def search():
    answear = {}
    args = request.args
    start_node = request.args.get('start_node')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    myList = getTrips(start_node, start_date, end_date)

    for trip in myList:
        route, times = getRouteFromTrip(trip)
        answear[trip] = {'route': route, 'timestamps': times}
    print(myList)
    buildLines(route, times)
    return answear

@app.route('/addTrip', methods=['POST'])
def addTrip():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        jsonObj = request.json
        pickup_longitude = jsonObj['pickup_longitude']
        pickup_latitude = jsonObj['pickup_latitude']
        dropoff_longitude = jsonObj['dropoff_longitude']
        dropoff_latitude = jsonObj['dropoff_latitude']
        pickup_datetime = jsonObj['pickup_datetime']
        dropoff_datetime = jsonObj['dropoff_datetime']
        trip_duration = (datetime.strptime(dropoff_datetime,"%Y-%m-%d %H:%M:%S") - datetime.strptime(pickup_datetime,"%Y-%m-%d %H:%M:%S")).total_seconds()
        route, timestamps, route_length, pickup_node, dropoff_node = DataPreProcessing.map_oneRoute_to_oneTrip_with_timestamps(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, pickup_datetime, dropoff_datetime, trip_duration)
        
        insertIntoTrips(jsonObj['id'], jsonObj['vendor_id'], jsonObj['pickup_datetime'], jsonObj['dropoff_datetime'], jsonObj['passenger_count'], jsonObj['pickup_longitude'], jsonObj['pickup_latitude'], jsonObj['dropoff_longitude'], jsonObj['dropoff_latitude'], jsonObj['store_and_fwd_flag'], trip_duration, pickup_node, dropoff_node, route_length, jsonObj['provider'], jsonObj['total_price'])
        timestamps = timestamps.replace(': \'',' : \'')
        timestamps = timestamps.strip("{}")
        insertIntoTripsRoutes(jsonObj['id'], timestamps)
  
        return "Success"
    else:
        return 'Content-Type not supported!'
def buildLines(routes, timestamps):
  for i in range(len(routes)-1):
    

@app.route('/addOrder', methods=['POST'])
def addOrder():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        jsonObj = request.json
        pickup_longitude = jsonObj['pickup_longitude']
        pickup_latitude = jsonObj['pickup_latitude']
        dropoff_longitude = jsonObj['dropoff_longitude']
        dropoff_latitude = jsonObj['dropoff_latitude']
        pickup_datetime = jsonObj['pickup_datetime']
        dropoff_datetime = jsonObj['dropoff_datetime']
        trip_duration = (datetime.strptime(dropoff_datetime,"%Y-%m-%d %H:%M:%S") - datetime.strptime(pickup_datetime,"%Y-%m-%d %H:%M:%S")).total_seconds()
        route, timestamps, route_length, pickup_node, dropoff_node = DataPreProcessing.map_oneRoute_to_oneTrip_with_timestamps(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, pickup_datetime, dropoff_datetime, trip_duration)
        
        insertIntoTrips(jsonObj['id'], jsonObj['vendor_id'], jsonObj['pickup_datetime'], jsonObj['dropoff_datetime'], jsonObj['passenger_count'], jsonObj['pickup_longitude'], jsonObj['pickup_latitude'], jsonObj['dropoff_longitude'], jsonObj['dropoff_latitude'], jsonObj['store_and_fwd_flag'], trip_duration, pickup_node, dropoff_node, route_length, jsonObj['provider'], jsonObj['total_price'])
        timestamps = timestamps.replace(': \'',' : \'')
        timestamps = timestamps.strip("{}")
        insertIntoTripsRoutes(jsonObj['id'], timestamps)
        return "Success"
    else:
        return 'Content-Type not supported!'
    

  

# driver function
if __name__ == '__main__':
  
    app.run(debug = True)