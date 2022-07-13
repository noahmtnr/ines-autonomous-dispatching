from time import time
import pandas as pd
from datetime import datetime
import mysql.connector
import time


class DBConnection:
  def __init__(self):
    try:
      self.mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="mannheimprojekt",
        auth_plugin='mysql_native_password'
      )
      print("Using local db")
    except:
      self.mydb = mysql.connector.connect(
        host="mannheimprojekt.mysql.database.azure.com",
        user="mannheim",
        password="Projekt2022",
        database="mannheimprojekt",
        auth_plugin='mysql_native_password'
      )
      print("Using remote db")
    self.mycursor = self.mydb.cursor()

  def initialiazeTables(self):

    self.mycursor.execute(
            """ create table TRIPS (
        id varchar(30) primary key,
        vendor_id int,
        pickup_datetime timestamp,
        dropoff_datetime timestamp,
        passenger_count int,
        pickup_longitude float,
        pickup_latitude float,
        dropoff_longitude float,
        dropoff_latitude float,
        store_and_fwd_flag varchar(2),
        trip_duration int,
        pickup_node bigint,
        dropoff_node bigint,
        pickup_distance float,
        dropoff_distance float,
        route_length float,
        provider int,
        total_price float)""")

    self.mycursor.execute("""
        create table TRIPS_ROUTES (
          id varchar(30),
          route_node bigint,
          date_time timestamp,
          foreign key (id) references TRIPS(id),
          primary key (id, route_node, date_time)
          )
          """)


  def insertIntoTripsRoutes(self,tripId, listTrips):
    for element in listTrips.split(', '):
        a, b = element.split(' : ') 
        #new_b = datetime.fromisoformat(b)
        timestamp_b = datetime.strptime(str(b),"'%Y-%m-%d %H:%M:%S'")
        sql = "INSERT INTO TRIPS_ROUTES VALUES (%s, %s, %s)"
        val = (tripId, a, timestamp_b)
        self.mycursor.execute(sql, val)
    self.mydb.commit()

  def insertIntoTrips(self,id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, store_and_fwd_flag, trip_duration, pickup_node, dropoff_node, pickup_distance, dropoff_distance, route_length, provider, total_price):
    sql = "INSERT INTO TRIPS VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    val = (id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, store_and_fwd_flag, trip_duration, pickup_node, dropoff_node, pickup_distance, dropoff_distance, route_length, provider, total_price)
    self.mycursor.execute(sql, val)
    self.mydb.commit()

  def getAvailableTripsAtNode(self,start_node, start_date, end_date):
    startTime = time.time()
    sql = "select * from PREFILTERED_TRIPS_VIEW_01 where route_node = %s and date_time between %s and %s"
    val = (start_node, start_date, end_date)
    tripsId_list=[]
    nodes_list=[]
    timestamp_list=[]
    self.mycursor.execute(sql, val)
      
    result = self.mycursor.fetchall()
    executionTime = (time.time() - startTime)
    print('DB: getAvailableTrips() Execution time: ' + str(executionTime) + ' seconds')
    return result

  def getAvailableTrips(self, start_date, end_date):
    startTime = time.time()
    sql = "select * from PREFILTERED_TRIPS_VIEW_01 where date_time between %s and %s"
    val = (start_date, end_date)
    tripsId_list=[]
    nodes_list=[]
    timestamp_list=[]
    self.mycursor.execute(sql, val)
      
    result = self.mycursor.fetchall()
    executionTime = (time.time() - startTime)
    print('DB: getAvailableTrips() Execution time: ' + str(executionTime) + ' seconds')
    print("loaded "+ str(len(result)) + " trips")
    return result

  def getRouteFromTrip(self,trip_id):
    startTime = time.time()
    sql = "select route_node, date_time from mannheimprojekt.TRIPS_ROUTES where id = %s order by date_time"
    val = (trip_id,)
    nodes_list=[]
    time_list = []
    self.mycursor.execute(sql, val)
    for result in self.mycursor: 
      nodes_list.append(result[0])
      time_list.append(result[1].strftime("%Y-%m-%d %H:%M:%S"))
    executionTime = (time.time() - startTime)
    # print('DB: getRouteFromTrip() Execution time: ' + str(executionTime) + ' seconds')
    return nodes_list, time_list


  def populateDatabase(self):
    self.initialiazeTables()
    full_df = pd.read_csv('rl/Graph_Environment/trips_kaggle_providers.csv')
    
    #data for each trip
    trips_df = full_df[['id','vendor_id','pickup_datetime','dropoff_datetime','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','store_and_fwd_flag','trip_duration','pickup_node','dropoff_node','pickup_distance','dropoff_distance','route_length','provider','total_price']]
    
    trips_df.apply(lambda row: self.insertIntoTrips(row['id'], row['vendor_id'], row['pickup_datetime'], 
    row['dropoff_datetime'], row['passenger_count'], row['pickup_longitude'], row['pickup_latitude'],
    row['dropoff_longitude'], row['dropoff_latitude'], row['store_and_fwd_flag'], row['trip_duration'],
    row['pickup_node'], row['dropoff_node'], row['pickup_distance'], row['dropoff_distance'], row['route_length'],
    row['provider'], row['total_price']), axis=1)
    #trip with route and timestamps
    rest_df = full_df[['id','route_timestamps']]
    rest_df['route_timestamps'] = rest_df['route_timestamps'].apply(lambda x: x.replace(': \'',' : \''))
    rest_df['route_timestamps'] = (rest_df['route_timestamps'].apply(lambda x: x.strip("{}")))
    rest_df.apply(lambda row: self.insertIntoTripsRoutes(row['id'], row['route_timestamps']), axis=1)


  def writeHubsToDB(self, hubs):
    sql = "INSERT INTO HUBS VALUES ( %s )"
    for hub in hubs:
      
        values = (int(hub),)
        self.mycursor.execute(sql, values)
    
    self.mydb.commit()

  def getAllHubs(self):
    sql = "SELECT * FROM HUBS"

    self.mycursor.execute(sql)
    result = self.mycursor.fetchall()
    print(result)
    return result


  def close_connection(self):
    self.cursor.close()
    self.mydb.close()

