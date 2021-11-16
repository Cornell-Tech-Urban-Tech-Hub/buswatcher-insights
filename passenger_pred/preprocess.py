import pandas as pd

def add_uid(df):
    """
    Add unique identifier to a pd.DataFrame loaded using methods from load.py. The
    uid is made up of the date, trip_id and vehicle_id. 
    
    pd.DataFrame df
    returns df
    """
    df['uid'] = df['service_date'] + "_" + df['trip_id'] + "_" + df['vehicle_id']
    return df
  
  
def remove_duplicate_stops(df):
    """
    Return a DataFrame like df, but retaining only one datapoint per stop for each uid.
    """
    df = df.drop_duplicates(subset=['uid', 'next_stop_id'], keep='last', ignore_index=True)
    df = df.reset_index(drop=True)
    return df

def add_time_features(df, timestamp_str="timestamp"):
    """
    Given a pd.DataFrame with a timestamp column, return a pd.DataFrame like df, but with
    additional columns "hour", "day", "dow" (day of week)
    
    pd.DataFrame df
    String timestamp_str: name of timestamp column in df
    returns df
    """
    df['timestamp_dt'] = pd.to_datetime(df.timestamp)
    df['hour'] = df['timestamp_dt'].dt.hour
    df['day'] = df['timestamp_dt'].dt.day
    df['dow'] = df['timestamp_dt'].dt.weekday
    return df
