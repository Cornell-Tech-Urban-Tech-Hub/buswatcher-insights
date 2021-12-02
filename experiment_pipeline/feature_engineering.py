import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer

def compute_stop_stats(train, test):
    # compute stop statistics using train set only
    # 25th Percentile
    def q25(x):
        return x.quantile(0.25)
    # 25th Percentile (median)
    def q50(x):
        return x.quantile(0.50)
    # 75th Percentile
    def q75(x):
        return x.quantile(0.75)
    stop_stats = train[['next_stop_id_pos', 'passenger_count']].groupby('next_stop_id_pos').agg({'passenger_count':['mean', 'std', q25, q50, q75]})
    return stop_stats

def normalize(train, test):
    scaler = MinMaxScaler()
    scaler.fit(train)
    train = pd.DataFrame(scaler.transform(train), index=train.index, columns=train.columns)
    test = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
    return train, test

def encode_cyclical_time_features(train, test, drop_cols=True):
    if drop_cols:
        drop = ['DoW', 'hour', 'minute', 'time']
    else:
        drop = ['time']
    train['time'] = train['hour'] * 3600 + train['minute'] * 60
    train['DoW_sin'] = np.sin(2 * np.pi * train['DoW'] / train['DoW'].max())
    train['DoW_cos'] = np.cos(2 * np.pi * train['DoW'] / train['DoW'].max())
    train['time_sin'] = np.sin(2 * np.pi * train['time'] / train['time'].max())
    train['time_cos'] = np.cos(2 * np.pi * train['time'] / train['time'].max())
    train.drop(columns=drop, inplace=True)

    test['time'] = test['hour'] * 3600 + test['minute'] * 60
    test['DoW_sin'] = np.sin(2 * np.pi * test['DoW'] / test['DoW'].max())
    test['DoW_cos'] = np.cos(2 * np.pi * test['DoW'] / test['DoW'].max())
    test['time_sin'] = np.sin(2 * np.pi * test['time'] / test['time'].max())
    test['time_cos'] = np.cos(2 * np.pi * test['time'] / test['time'].max())
    test.drop(columns=drop, inplace=True)

def discretize_weather_feature(train, test, weather_feature, n_bins, encode='ordinal', strategy='uniform', drop_cols=True):
    discretized_weather_feature = f'd_{weather_feature}'
    est = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    est.fit(np.array(train[weather_feature]).reshape(-1, 1))
    train[discretized_weather_feature] = est.transform(np.array(train[weather_feature]).reshape(-1, 1))
    test[discretized_weather_feature] = est.transform(np.array(test[weather_feature]).reshape(-1, 1))
    if drop_cols:
        train.drop(columns=weather_feature, inplace=True)
        test.drop(columns=weather_feature, inplace=True)
