import pandas as pd
# from experiment_pipeline.feature_engineering import (
#     add_stop_stats
# )

bus_features_ls = [
    'vehicle_id',
    'next_stop_id_pos',
    'next_stop_est_sec',
    'month',    
    'DoW',  
    'hour',
    'minute',    
    'trip_id_comp_SDon_bool',
    'trip_id_comp_3_dig_id',
    # 'day',                   # always drop
    # 'year',                  # always drop
    # 'trip_id_comp_6_dig_id', # always drop
    # 'timestamp'              # always drop
]

weather_features_ls = [
    'Precipitation',
    'Cloud Cover',
    'Relative Humidity',
    'Heat Index',
    'Max Wind Speed'
]

def bus_pos_and_obs_time(train, test, dependent_variable, stop_stats):
    # select features
    feature_set_bus = ['next_stop_id_pos', 'DoW','hour']
    feature_set_weather = []
    feature_set = feature_set_bus + feature_set_weather + [dependent_variable]
    non_features = list(set(train.columns) - set(feature_set))
    train.drop(columns=non_features, inplace=True)
    test.drop(columns=non_features, inplace=True)
    # partition
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]
    return train_x, train_y, test_x, test_y

def bus_features(train, test, dependent_variable, stop_stats):
    # select features
    feature_set_bus = bus_features_ls
    feature_set_weather = []
    feature_set = feature_set_bus + feature_set_weather + [dependent_variable]
    non_features = list(set(train.columns) - set(feature_set))
    train.drop(columns=non_features, inplace=True)
    test.drop(columns=non_features, inplace=True)
    # partition
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]
    return train_x, train_y, test_x, test_y

def bus_features_with_stop_stats(train, test, dependent_variable, stop_stats):
    # select features
    feature_set_bus = bus_features_ls
    feature_set_weather = []
    feature_set = feature_set_bus + feature_set_weather + [dependent_variable]
    non_features = list(set(train.columns) - set(feature_set))
    train.drop(columns=non_features, inplace=True)
    test.drop(columns=non_features, inplace=True)
    # add stop stats
    train, test = add_stop_stats(train, test, stop_stats)
    # partition
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]
    return train_x, train_y, test_x, test_y

def bus_and_weather_features(train, test, dependent_variable, stop_stats):
    # select features
    feature_set_bus = bus_features_ls
    feature_set_weather = weather_features_ls
    feature_set = feature_set_bus + feature_set_weather + [dependent_variable]
    non_features = list(set(train.columns) - set(feature_set))
    train.drop(columns=non_features, inplace=True)
    test.drop(columns=non_features, inplace=True)
    # partition
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]
    return train_x, train_y, test_x, test_y

def bus_and_weather_features_with_stop_stats(train, test, dependent_variable, stop_stats):
    # select features
    feature_set_bus = bus_features_ls
    feature_set_weather = weather_features_ls
    feature_set = feature_set_bus + feature_set_weather + [dependent_variable]
    non_features = list(set(train.columns) - set(feature_set))
    train.drop(columns=non_features, inplace=True)
    test.drop(columns=non_features, inplace=True)
    # add stop stats
    train, test = add_stop_stats(train, test, stop_stats)
    # partition
    train_x = train.drop(columns=[dependent_variable])
    train_y = train[dependent_variable]
    test_x = test.drop(columns=[dependent_variable])
    test_y = test[dependent_variable]
    return train_x, train_y, test_x, test_y

# helper
def add_stop_stats(train, test, stop_stats):
    train['avg_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    train['std_stop_passengers'] = train['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])
    test['avg_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'mean')].loc[x])
    test['std_stop_passengers'] = test['next_stop_id_pos'].apply(lambda x: stop_stats[('passenger_count', 'std')].loc[x])
    return train, test