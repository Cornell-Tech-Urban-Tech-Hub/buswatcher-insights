import os
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from data_pipeline.data_downloader import get_data_dict
from data_pipeline.data_processing import (
    remove_non_normalProgreess_observations,
    remove_unique_trip_ids_with_high_pct_nan_passenger_count_readings,
    remove_delinquent_stops,
    add_stop_positions,
    add_estimated_seconds_to_next_stop,
    fill_nan_estimated_seconds_to_next_stop,
    remove_stops_not_in_stops_dict
) 

parser = argparse.ArgumentParser()

parser.add_argument(
    'read_file', 
    help='Name and extension of file to be processed',
    type=str
)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Arguments", args)

    read_file = args.read_file
    route = read_file.split('_')[0]
    save_file_processed = read_file
    save_file_stops = route + '.json'
    read_path = 'data/Bus/Segment Data - Raw'
    save_path_processed = 'data/Bus/Segment Data - Processed'
    save_path_stops = 'data/Bus/Route Data'

    df = pd.read_csv(os.path.join(read_path, read_file))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['next_stop_eta'] = pd.to_datetime(df['next_stop_eta'])
    df = df.sort_values('timestamp')

    remove_non_normalProgreess_observations(df)
    remove_unique_trip_ids_with_high_pct_nan_passenger_count_readings(df, 1.0)
    stops_dict, delinquent_stops_dict = get_data_dict(df, route)
    remove_delinquent_stops(df, delinquent_stops_dict)
    remove_stops_not_in_stops_dict(df, stops_dict)
    add_stop_positions(df, stops_dict)
    add_estimated_seconds_to_next_stop(df)
    fill_nan_estimated_seconds_to_next_stop(df)
    print()

    keep = [
        'route', 
        'direction', 
        'trip_id', 
        'service_date', 
        'vehicle_id', 
        'timestamp', 
        'lat', 
        'lon', 
        'bearing', 
        'prior_stop_id', 
        'next_stop_id', 
        'next_stop_id_pos', 
        'next_stop_d_along_route', 
        'next_stop_est_sec', 
        'passenger_count'
    ]

    df = df[keep]

    if not os.path.exists(save_path_processed):
        os.makedirs(save_path_processed)
    if not os.path.exists(save_path_stops):
        os.makedirs(save_path_stops)
    with open(os.path.join(save_path_stops, save_file_stops), 'w') as f:
        json.dump(stops_dict, f)
        print(f'route data saved to: {os.path.join(save_path_stops, save_file_stops)}')
    df.to_csv(os.path.join(save_path_processed, save_file_processed), index=False)
    print(f'processed segment data saved to: {os.path.join(save_path_processed, save_file_processed)}')
    print()
  