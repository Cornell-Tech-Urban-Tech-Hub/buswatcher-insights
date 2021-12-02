import pandas as pd
from experiment_pipeline.feature_engineering import (
    normalize,
    encode_cyclical_time_features,
    discretize_weather_feature
)

bus_features_ls = [
    'vehicle_id',
    'next_stop_id_pos',
    'next_stop_est_sec',
    'DoW',  
    'hour',
    'minute',    
    'trip_id_comp_SDon_bool',
    'trip_id_comp_3_dig_id',
    # 'day',                   # always drop
    # 'month',                 # always drop
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

def normalized_and_encoded_bus_and_weather_features_with_stop_stats(train, test, dependent_variable, stop_stats):
    # select features
    feature_set_bus = bus_features_ls
    feature_set_weather = weather_features_ls
    feature_set = feature_set_bus + feature_set_weather + [dependent_variable]
    non_features = list(set(train.columns) - set(feature_set))
    train.drop(columns=non_features, inplace=True)
    test.drop(columns=non_features, inplace=True)
    # add stop stats
    train, test = add_stop_stats(train, test, stop_stats)
    # encode cyclical time features
    encode_cyclical_time_features(train, test, drop_cols=True)
    # discretize weather featrues
    discretize_weather_feature(train, test, 'Precipitation', n_bins=2, encode='ordinal', strategy='uniform', drop_cols=True)
    discretize_weather_feature(train, test, 'Cloud Cover', n_bins=3, encode='ordinal', strategy='uniform', drop_cols=True)
    discretize_weather_feature(train, test, 'Relative Humidity', n_bins=3, encode='ordinal', strategy='uniform', drop_cols=True)
    discretize_weather_feature(train, test, 'Heat Index', n_bins=3, encode='ordinal', strategy='uniform', drop_cols=True)
    discretize_weather_feature(train, test, 'Max Wind Speed', n_bins=3, encode='ordinal', strategy='uniform', drop_cols=True)
    # normalize data
    train, test = normalize(train, test)
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