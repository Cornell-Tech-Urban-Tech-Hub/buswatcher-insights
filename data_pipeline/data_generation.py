import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from data_pipeline.data_processing import (
    remove_vehicles_that_never_report_passenger_count,
    add_unique_trip_id,
) 

def generate_segment_data_df(df):
    remove_vehicles_that_never_report_passenger_count(df)
    add_unique_trip_id(df)
    unique_trip_ids = list(set(df['unique_trip_id']))
    segment_data_dict = {}
    i = 0
    for unique_trip_id in unique_trip_ids:
        unique_trip_id_df = df.copy()
        unique_trip_id_df = unique_trip_id_df[unique_trip_id_df['unique_trip_id'] == unique_trip_id]
        unique_trip_id_stops = list(set(unique_trip_id_df['next_stop_id']))
        for unique_trip_id_stop in unique_trip_id_stops:
            unique_trip_id_stop_df = unique_trip_id_df.copy()
            if not pd.isna(unique_trip_id_stop):
                unique_trip_id_stop_df = unique_trip_id_stop_df[unique_trip_id_stop_df['next_stop_id'] == unique_trip_id_stop]
                unique_trip_id_stop_df.reset_index(drop=True, inplace=True)
                observation_count = unique_trip_id_stop_df.shape[0]
                duration = unique_trip_id_stop_df.timestamp.max() - unique_trip_id_stop_df.timestamp.min()
                middle = observation_count // 2
                segment_data = unique_trip_id_stop_df.loc[middle].to_dict()
                segment_data['observation_count'] = observation_count
                segment_data['duration'] = duration
                segment_data_dict[i] = segment_data
                i += 1
            else:
                unique_trip_id_stop_df = unique_trip_id_stop_df[unique_trip_id_stop_df['next_stop_id'].isna() == True]
                unique_trip_id_stop_df.reset_index(drop=True, inplace=True)
                unique_trip_id_stop_dict = unique_trip_id_stop_df.to_dict('index')
                for index in unique_trip_id_stop_dict:
                    segment_data = unique_trip_id_stop_dict[index]
                    segment_data['observation_count'] = np.nan
                    segment_data['duration'] = np.nan
                    segment_data_dict[i] = segment_data
                    i += 1
    segment_data_df = pd.DataFrame.from_dict(segment_data_dict, orient='index')
    return segment_data_df