import time
from datetime import datetime
from time import time

import mysql.connector
import pandas as pd


class DBConnection:
    def __init__(self):
        # try:
        #   self.mydb = mysql.connector.connect(
        #     host="localhost",
        #     user="root",
        #     password="root",
        #     database="mannheimprojekt",
        #     auth_plugin='mysql_native_password'
        #   )
        #   print("Using local db")
        # except:
        self.mydb = mysql.connector.connect(
            host="mannheimprojekt.mysql.database.azure.com",
            user="mannheim",
            password="Projekt2022",
            database="mannheimprojekt",
            auth_plugin='mysql_native_password'
        )
        print("Using remote db")
        self.mycursor = self.mydb.cursor()

    def insert_into_trips_routes(self, tripId, listTrips):
        """ Helper function to fill trips_routes table

        Args:
            tripId (_type_): _description_
            listTrips (_type_): _description_
        """
        for element in listTrips.split(', '):
            a, b = element.split(' : ')
            timestamp_b = datetime.strptime(str(b), "'%Y-%m-%d %H:%M:%S'")
            sql = "INSERT INTO TRIPS_ROUTES VALUES (%s, %s, %s)"
            val = (tripId, a, timestamp_b)
            self.mycursor.execute(sql, val)
        self.mydb.commit()

    def insert_into_trips(self, id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count, pickup_longitude,
                          pickup_latitude, dropoff_longitude, dropoff_latitude, store_and_fwd_flag, trip_duration,
                          pickup_node, dropoff_node, pickup_distance, dropoff_distance, route_length, provider,
                          total_price):
        """ Helper function to write trip into trips table"""

        sql = "INSERT INTO TRIPS VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count, pickup_longitude, pickup_latitude,
               dropoff_longitude, dropoff_latitude, store_and_fwd_flag, trip_duration, pickup_node, dropoff_node,
               pickup_distance, dropoff_distance, route_length, provider, total_price)
        self.mycursor.execute(sql, val)
        self.mydb.commit()

    def fetch_available_trips_at_node(self, start_node, start_date, end_date):
        """Fetches all available trips at a node for a given time window defined by start and end timestamps. Still missing a mapping function that maps the query to the correct partition of the prefiltered_trips_view. Currently only a few 2 weeks windows can be queried (see README>Database).

        Args:
            start_date (String): lower time boundary to query for
            end_date (String): upper time boundary to query for

        Returns:
            list: available trips
        """
        startTime = time.time()
        sql = "select * from PREFILTERED_TRIPS_VIEW_01 where route_node = %s and date_time between %s and %s"
        val = (start_node, start_date, end_date)
        tripsId_list = []
        nodes_list = []
        timestamp_list = []
        self.mycursor.execute(sql, val)

        result = self.mycursor.fetchall()
        executionTime = (time.time() - startTime)
        print('DB: fetch_all_available_trips() Execution time: ' + str(executionTime) + ' seconds')
        return result

    def fetch_all_available_trips(self, start_date, end_date):
        """Fetches all available trips for a given time window defined by start and end timestamps. Still missing a mapping function that maps the query to the correct partition of the prefiltered_trips_view. Currently only a few 2 weeks windows can be queried (see README>Database).

        Args:
            start_date (String): lower time boundary to query for
            end_date (String): upper time boundary to query for

        Returns:
            list: available trips
        """
        startTime = time.time()
        sql = "select * from PREFILTERED_TRIPS_VIEW_01 where date_time between %s and %s"
        val = (start_date, end_date)
        tripsId_list = []
        nodes_list = []
        timestamp_list = []
        self.mycursor.execute(sql, val)

        result = self.mycursor.fetchall()
        executionTime = (time.time() - startTime)
        print('DB: fetch_all_available_trips() Execution time: ' + str(executionTime) + ' seconds')
        print("loaded " + str(len(result)) + " trips")
        return result

    def fetch_route_from_trip(self, trip_id):
        """Gets detailed route information for a given trip_id like pickup, dropoff coordinates, passenger count, total price.

        Args:
            trip_id (int)

        Returns:
            _type_: _description_
        """
        startTime = time.time()
        sql = "select route_node, date_time from mannheimprojekt.TRIPS_ROUTES where id = %s order by date_time"
        val = (trip_id,)
        nodes_list = []
        time_list = []
        self.mycursor.execute(sql, val)
        for result in self.mycursor:
            nodes_list.append(result[0])
            time_list.append(result[1].strftime("%Y-%m-%d %H:%M:%S"))
        executionTime = (time.time() - startTime)
        # print('DB: fetch_route_from_trip() Execution time: ' + str(executionTime) + ' seconds')
        return nodes_list, time_list

    def populate_database(self):
        """Reads raw trips from file and writes them into trips table
        """
        full_df = pd.read_csv('rl/Graph_Environment/trips_kaggle_providers.csv')

        # data for each trip
        trips_df = full_df[
            ['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime', 'passenger_count', 'pickup_longitude',
             'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'trip_duration',
             'pickup_node', 'dropoff_node', 'pickup_distance', 'dropoff_distance', 'route_length', 'provider',
             'total_price']]

        trips_df.apply(lambda row: self.insert_into_trips(row['id'], row['vendor_id'], row['pickup_datetime'],
                                                          row['dropoff_datetime'], row['passenger_count'],
                                                          row['pickup_longitude'], row['pickup_latitude'],
                                                          row['dropoff_longitude'], row['dropoff_latitude'],
                                                          row['store_and_fwd_flag'], row['trip_duration'],
                                                          row['pickup_node'], row['dropoff_node'], row['pickup_distance'],
                                                          row['dropoff_distance'], row['route_length'],
                                                          row['provider'], row['total_price']), axis=1)
        # trip with route and timestamps
        rest_df = full_df[['id', 'route_timestamps']]
        rest_df['route_timestamps'] = rest_df['route_timestamps'].apply(lambda x: x.replace(': \'', ' : \''))
        rest_df['route_timestamps'] = (rest_df['route_timestamps'].apply(lambda x: x.strip("{}")))
        rest_df.apply(lambda row: self.insert_into_trips_routes(row['id'], row['route_timestamps']), axis=1)

    def write_hubs_to_db(self, hubs):
        """Helper function to write into hubs table.

        Args:
            hubs (list(int)): List of hub ids
        """
        sql = "INSERT INTO HUBS VALUES ( %s )"
        for hub in hubs:
            values = (int(hub),)
            self.mycursor.execute(sql, values)

        self.mydb.commit()

    def fetch_all_hubs(self):
        """Retrieves all hubs

        Returns:
            list(int): list of hub ids
        """
        sql = "SELECT * FROM HUBS ORDER BY id ASC"

        self.mycursor.execute(sql)
        result = self.mycursor.fetchall()
        processed_hubs = []

        for hub in result:
            processed_hubs.append(hub[0])

        return processed_hubs

    def close_connection(self):
        self.cursor.close()
        self.mydb.close()
