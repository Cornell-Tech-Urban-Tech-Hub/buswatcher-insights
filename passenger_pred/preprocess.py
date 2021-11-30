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
    df['timestamp_dt'] = pd.to_datetime(df.timestamp, utc=True)
    df['hour'] = df['timestamp_dt'].dt.hour
    df['day'] = df['timestamp_dt'].dt.day
    df['dow'] = df['timestamp_dt'].dt.weekday
    return df

def remove_unplanned_alert(alert_df):
    id_df = alert_df.id.str.split(":", expand=True)
    keep_index = id_df[id_df[1] == 'alert'].index
    alert_df = alert_df.loc[keep_index]
    return alert_df

def add_averages(df_train, df_test, stop_column='next_stop'):
    mean_hour = df_train.groupby([stop_column, 'hour']).passenger_count.mean()
    mean_dow = df_train.groupby([stop_column, 'dow']).passenger_count.mean()
    mean_day = df_train.groupby([stop_column, 'day']).passenger_count.mean()
    
    df_train = df_train.join(mean_hour, on=[stop_column, 'hour'], rsuffix='_mean_hr')
    df_train = df_train.join(mean_dow, on=[stop_column, 'dow'], rsuffix='_mean_dow')
    df_train = df_train.join(mean_day, on=[stop_column, 'day'], rsuffix='_mean_day')

    df_test = df_test.join(mean_hour, on=[stop_column, 'hour'], rsuffix='_mean_hr')
    df_test = df_test.join(mean_dow, on=[stop_column, 'dow'], rsuffix='_mean_dow')
    df_test = df_test.join(mean_day, on=[stop_column, 'day'], rsuffix='_mean_day')
    return df_train, df_test
