import pandas as pd

def remove_vehicles_that_never_report_passenger_count(df):
    vehicles = set(df['vehicle_id'])
    for vehicle in vehicles:
        vehicle_data = df[df['vehicle_id'] == vehicle]
        num_non_nan_passenger_counts = vehicle_data['passenger_count'].notna().sum() 
        if num_non_nan_passenger_counts == 0:
            drop = df[df['vehicle_id'] == vehicle].index
            df.drop(index=drop, inplace=True)
    df.reset_index(drop=True, inplace=True) 

def add_unique_trip_id(df):
    df['unique_trip_id'] = df['trip_id'] + '-' + df['service_date'] + '-' + df['vehicle_id']    

def remove_non_normalProgreess_observations(df):
    drop = df[df['progress_rate'] != 'normalProgress'].index
    df.drop(index=drop, inplace=True) 

def remove_unique_trip_ids_with_high_pct_nan_passenger_count_readings(df, pct):
    unique_trip_ids = set(df['unique_trip_id'])
    for uuid in unique_trip_ids:
        temp = df[df['unique_trip_id'] == uuid]
        num_nan =  temp['passenger_count'].isna().sum()
        if num_nan / temp.shape[0] >= pct:
            drop = df[df['unique_trip_id'] == uuid].index
            df.drop(index=drop, inplace=True)

def remove_delinquent_stops(df, delinquent_stops_dict):
    for direction in [0, 1]:
        for delinquent_stop in delinquent_stops_dict[direction]:
            drop = df[df['next_stop_id'] == delinquent_stop].index 
            df.drop(index=drop, inplace=True) 

def add_stop_positions(df, stops_dict):
    # zip direction and next_stop_id to look up position of next_stop_id along appropriate route stop sequence
    df['direction-next_stop_id'] = list(zip(df['direction'], df['next_stop_id']))

    # look up position of next_stop_id along appropriate route stop sequence and drop direction-next_stop_id column (not needed)
    df['next_stop_id_pos'] = df['direction-next_stop_id'].apply(lambda x: stops_dict[x[0]].index(x[1]))
    df.drop(columns='direction-next_stop_id', inplace=True)

    # prepend None to route stop sequence to look up prior_stop_id using next_stop_id position and drop direction-stop_sequence column (not needed)
    stops_dict[1].insert(0, None)
    stops_dict[0].insert(0, None)
    df['direction-stop_sequence'] = list(zip(df['direction'], df['next_stop_id_pos']))
    df['prior_stop_id'] = df['direction-stop_sequence'].apply(lambda x: stops_dict[x[0]][x[1]])
    df.drop(columns='direction-stop_sequence', inplace=True)

def add_estimated_seconds_to_next_stop(df):
    # compute estimated number of seconds > 0 between stops
    df['next_stop_est_sec'] = (df['next_stop_eta'] - df['timestamp']).dt.seconds

def fill_nan_estimated_seconds_to_next_stop(df):
    # check for missing values in next_stop_est_sec and replace pursuant to strategies described below...
    num_missing = df['next_stop_est_sec'].isna().sum()
    if num_missing > 0:
        print(f'{num_missing} missing values in next_stop_est_sec... attempting to replace with median unique_trip_id value')
        uuids = list(df['unique_trip_id'])
        segment_times = list(df['next_stop_est_sec'])
        for (i, (uuid, segment_time)) in enumerate(zip(uuids, segment_times)):
            if pd.isna(segment_time):
                replacement_value = df[df['unique_trip_id'] == uuid]['next_stop_est_sec'].median()
                if not pd.isna(replacement_value):
                    segment_times[i] = replacement_value
                    num_missing -= 1
        df['next_stop_est_sec'] = segment_times
    if num_missing > 0:
        print(f'{num_missing} missing values in next_stop_est_sec... attempting to replace with median trip_id value')
        trips = list(df['trip_id'])
        for (i, (trip, segment_time)) in enumerate(zip(trips, segment_times)):
            if pd.isna(segment_time):
                replacement_value = df[df['trip_id'] == trip]['next_stop_est_sec'].median()
                if not pd.isna(replacement_value):
                    segment_times[i] = replacement_value
                    num_missing -= 1
        df['next_stop_est_sec'] = segment_times    
    if num_missing > 0:
        print(f'{num_missing} missing values in next_stop_est_sec... attempting to replace with median next_stop_id value')
        segments = list(df['next_stop_id'])
        for (i, (segment, segment_time)) in enumerate(zip(segments, segment_times)):
            if pd.isna(segment_time):
                replacement_value = df[df['next_stop_id'] == segment]['next_stop_est_sec'].median()
                if not pd.isna(replacement_value):
                    segment_times[i] = replacement_value
                    num_missing -= 1
        df['next_stop_est_sec'] = segment_times
    if num_missing > 0:
        print(f'{num_missing} missing values in next_stop_est_sec... replacing with global average')
        replacement_value = df['next_stop_est_sec'].median()
        for (i, segment_time) in enumerate(segment_times):
            if pd.isna(segment_time):
                segment_times[i] = replacement_value
                num_missing -= 1
        segment_times = list(df['next_stop_est_sec'])
    assert num_missing == 0
    print(f'\nsuccessfully replaced all missing values in next_stop_est_sec!')
