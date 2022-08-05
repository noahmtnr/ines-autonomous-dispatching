'''
This file prepares taxi data to a format suitable for saving them in a database.
'''

# imports and load dataset
import numpy as np
import osmnx as ox
import pandas as pd

from data_preprocessing import DataPreProcessing  # map_routes_to_trips_with_timestamps


# apply preprocessing to specific data of one month in a year
def preprocess(month,year):
    """
    Applies preprocessing to specific data of one month in a year.

    Args:
        month (int): chosen month
        year (int): chosen year

      
        pickup_days = []
        pickup_hours = []
        pickup_minutes = []
        for index, row in df.iterrows():
            pickup_days.append(row['Trip_Pickup_DateTime'].day)
            pickup_hours.append(row['Trip_Pickup_DateTime'].hour)
            pickup_minutes.append(row['Trip_Pickup_DateTime'].minute)
        df['pickup_day']=pickup_days
        df['pickup_hour']=pickup_hours
        df['pickup_minute']=pickup_minutes
    

    Returns:
        pandas.DataFrame: preprocessed data from a chosen month
    """    
    # depending on month and year, load the respective dataset
    # month = '03' # example: March
    # year = '2009' # example: year 2009
    filename = f'C:/Users/dispatching-system/Downloads/NYCWebsite_Data/unprocessed_data/yellow_tripdata_{year}-{month}.csv'
    df = pd.read_csv(filename)
    print(df.head())

    # define columns to drop
    to_drop=['Rate_Code', 'store_and_forward', 'Payment_Type', 'surcharge','mta_tax', 'Tolls_Amt', 'Fare_Amt']
    df.drop(to_drop,inplace=True,axis=1)
    
    # get actual price for tip by: total_amt - tip_amt  [because box does not pay tips]
    df['total_price']=df['Total_Amt']-df['Tip_Amt']
    df.drop(columns=['Total_Amt','Tip_Amt'])

    # transform distance from miles to meters
    df['route_length']=df['Trip_Distance']* 1609.344
    df.drop('Trip_Distance',inplace=True,axis=1)

    # extract day, hour and minute from datetime
    df['Trip_Pickup_DateTime']=pd.to_datetime(df['Trip_Pickup_DateTime'])
    df['Trip_Dropoff_DateTime']=pd.to_datetime(df['Trip_Dropoff_DateTime'])

  

    # rename columns to be similar to those of Kaggle
    df.rename(columns={'vendor_name': 'vendor_id', 'Passenger_Count': 'passenger_count', 'Trip_Pickup_DateTime': 'pickup_datetime', 'Trip_Dropoff_DateTime': 'dropoff_datetime', 'Start_Lon': 'pickup_longitude', 'Start_Lat': 'pickup_latitude', 'End_Lon': 'dropoff_longitude', 'End_Lat': 'dropoff_latitude'},inplace=True)

    
    # remove trips with passenger_count > 4 (more do normally not fit in car)
    df.drop(df.index[df['passenger_count']>4],inplace=True)

    # remove trips with price > 200 (even if 4 people take car and each pay 50 that seems a lot)
    df.drop(df.index[df['total_price']>200],inplace=True)

    # define coordinate limits
    # e.g., all points with latitude smaller than 40.70 and greater than 40.90 are definitely outside of Manhattan
    manhattan_lat_limits = np.array([40.70, 40.90])
    manhattan_lon_limits = np.array([-74.016, -73.9102])

    # remove outlier trips from original dataframe
    df = df[(df['pickup_latitude']   >= manhattan_lat_limits[0] ) & (df['pickup_latitude']   <= manhattan_lat_limits[1]) ]
    df = df[(df['dropoff_latitude']  >= manhattan_lat_limits[0] ) & (df['dropoff_latitude']  <= manhattan_lat_limits[1]) ]
    df = df[(df['pickup_longitude']  >= manhattan_lon_limits[0]) & (df['pickup_longitude']  <= manhattan_lon_limits[1])]
    df = df[(df['dropoff_longitude'] >= manhattan_lon_limits[0]) & (df['dropoff_longitude'] <= manhattan_lon_limits[1])]

    return df

# apply pre-processing to all months in all years (2009, 01 until 2017, 12)
"""
def run():
    # loop for arguments month and year
    # only 2009 to 2017 because later only taxi areas provided not exact coordinates for pickup and dropoff
    for y in range(2009,2017):
        year = f'{y}'
        for m in range(1,12):
            if len(m)==1:
                month = f'0{m}'
            else:
                month = str(m)
            
            preprocessed_df = preprocess(month,year)
            graph = setup_graph()

            trips_with_nodes = map_trips_to_nodes(preprocessed_df,graph)
            trips_with_routes = map_routes_to_trips(preprocessed_df,graph)
            trips_with_routes_timestamps = map_routes_to_trips_with_timestamps(preprocessed_df,graph)

            trips_with_routes_timestamps.rename(columns={'provider_name': 'provider'})
            # route length and total price already are in the data
            # mobility providers are already present

            # write into csv
            trips_with_routes_timestamps.to_csv(f"../../Files_For_Database/trip_data_{year}_{month}.csv")
"""

def run():
    year = '2009'
    for m in range(1,12):
        if len(str(m))==1:
            month = f'0{m}'
        else:
            month = str(m)
            
        preprocessed_df = preprocess(month,year)
        graph = ox.io.load_graphml("data/graph/full.graphml") #setup_graph()

        #import sklearn
        trips_with_nodes = DataPreProcessing.map_trips_to_nodes(preprocessed_df,graph)
        #trips_with_routes = DataPreProcessing.map_routes_to_trips(preprocessed_df,graph)
        trips_with_routes_timestamps = DataPreProcessing.map_routes_to_trips_with_timestamps(trips_with_nodes,graph)

        #trips_with_routes_timestamps.rename(columns={'provider_name': 'provider'})
        # route length and total price already are in the data
        # mobility providers are already present

        # write into csv
        trips_with_routes_timestamps.to_csv(f"../../Files_For_Database/trip_data_{year}_{month}.csv")
run()