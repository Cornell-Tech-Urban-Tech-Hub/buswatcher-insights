import os
import pickle
import argparse
import pandas as pd
from datetime import datetime
from experiment_pipeline.evaluation import Evaluation
from experiment_pipeline.feature_engineering import compute_stop_stats
from experiment_pipeline.feature_sets import (
    bus_pos_and_obs_time,
    bus_features, 
    bus_features_with_stop_stats, 
    bus_and_weather_features,
    bus_and_weather_features_with_stop_stats,
    normalized_and_encoded_bus_and_weather_features_with_stop_stats
)
from experiment_pipeline.utils import custom_train_test_split
from experiment_pipeline.data_loader import load_global_feature_set
from sklearn.linear_model import (
  LinearRegression, 
  Lasso, 
  LassoCV, 
  Ridge,
  RidgeCV
)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def run_experiment(
    global_feature_set,
    feature_extractor_fn,
    model,  
    stop_id_ls,
    dependent_variable="passenger_count",
    split_heuristic="datetime",
    test_size=0.1,
    split_datetime=datetime(year=2021, month=9, day=17, hour=0, minute=0),
    test_period="1D",
    refit_interval=None,
    random_state=0,
    experiment_name=None,
    experiment_dir="./saved_experiments/"
):
    train, test = custom_train_test_split(
        global_feature_set, 
        split_heuristic=split_heuristic, 
        test_size=test_size, 
        split_datetime=split_datetime,
        test_period=test_period, 
        random_state=random_state
    )

    stop_stats = compute_stop_stats(train, test)
    train_x, train_y, test_x, test_y = feature_extractor_fn(train, test, dependent_variable, stop_stats)
    # Fit
    print("Fitting model...")
    model.fit(train_x, train_y)
    if refit_interval is None:
        # Inference
        print("Inference...")
        train_preds = model.predict(train_x)
        test_preds = model.predict(test_x)

        train['passenger_count_pred'] = train_preds
        test['passenger_count_pred'] = test_preds
    else:
        # Run inference once on initial train set    
        train_preds = model.predict(train_x)
        train['passenger_count_pred'] = train_preds
        print(f"Refitting every {refit_interval}")
        initial_split = split_datetime
        refit_test_sets = []
        total_refits = pd.Timedelta(test_period) / pd.Timedelta(refit_interval)
        counter = 0
        while split_datetime < initial_split + pd.Timedelta(test_period):
            print(f"Refitting {counter/int(total_refits):.0%}...")
            train_refit, test_refit = custom_train_test_split(
                global_feature_set, 
                split_heuristic=split_heuristic, 
                test_size=test_size, 
                split_datetime=split_datetime,
                test_period=refit_interval, 
                random_state=random_state
            )
            if (len(test_refit) > 0):
                train_x, train_y, test_x, test_y = feature_extractor_fn(train_refit, test_refit, dependent_variable, stop_stats)
                refit_test_sets.append(test_refit)
                model.fit(train_x, train_y)
                test_preds = model.predict(test_x)
                test_refit['passenger_count_pred'] = test_preds
            else:
                print("No test data found for refit test_period:", split_datetime, refit_interval)

            split_datetime += pd.Timedelta(refit_interval)
            counter += 1

        test = pd.concat(refit_test_sets)

    # Eval
    eval_instance = Evaluation(global_feature_set=global_feature_set, train=train, test=test, stop_id_ls=stop_id_ls, stop_stats=stop_stats, model=model)

    if experiment_name is not None:
        with open(os.path.join(experiment_dir, f"{experiment_name}.pickle"), "wb") as f:
            pickle.dump({
                "split_datetime": split_datetime,
                "test_period": test_period,
                "split_heuristic": split_heuristic,
                "model": model,
                "global_feature_set": global_feature_set,
                "train": train,
                "test": test,
                "stop_id_ls": stop_id_ls,
                "stop_stats": stop_stats
            }, f)

    return eval_instance
    
def load_pickled_experiment(location):
    """ 
        Loads a pickled experiment from a location 
        and instantiates the evaluation class
    """

    with open(location, "rb") as f:
        loaded_experiment = pickle.load(f)

    return Evaluation(
        global_feature_set=loaded_experiment["global_feature_set"], 
        train=loaded_experiment["train"], 
        test=loaded_experiment["test"], 
        stop_id_ls=loaded_experiment["stop_id_ls"], 
        stop_stats=loaded_experiment["stop_stats"],
        model=loaded_experiment["model"]
    )
        

parser = argparse.ArgumentParser()

parser.add_argument(
    'data_dir',
    default="../data/",
    type=str
)

parser.add_argument(
    'route', 
    help='The route number, i.e. "B46"',
    type=str
)

parser.add_argument(
    'station', 
    help='The weather station, i.e. "JFK"',
    type=str
)

parser.add_argument(
    'direction', 
    help='The route direction, i.e. 0 or 1',
    type=int
)

parser.add_argument(
    '-r',
    '--refit_interval',
    default=None,
    help='Refit interval specified as a pandas timedelta'
)

parser.add_argument(
    '-t',
    '--test_period',
    default='1D',
    help='Time period for testing specified as a pandas timedelta'
)

parser.add_argument(
    '-n',
    '--experiment_name',
    help='Experiment name to pickle the eval class in saved experiments'
)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Arguments", args)

    data_dir = args.data_dir
    route_str = args.route
    station_str = args.station
    direction_int = args.direction

    ## Prepare globlal feature set
    df_route, stop_id_ls = load_global_feature_set(
        data_dir, 
        route_str, 
        station_str, 
        direction_int
    )

    ## run experiment
    experiment_eval = run_experiment(
        global_feature_set=df_route,
        feature_extractor_fn=bus_and_weather_features,
        model=XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=64, random_state=0),
        stop_id_ls=stop_id_ls,
        dependent_variable="passenger_count",
        split_heuristic="datetime",
        test_period=args.test_period,
        refit_interval=args.refit_interval,
        random_state=0,
        experiment_name=args.experiment_name
    )

    print("-- Evaluation on train --")
    model_pred_eval, mean_pred_eval = experiment_eval.regression_metrics("train", pretty_print=True)
    print()
    print("-- Evaluation on test --")
    model_pred_eval, mean_pred_eval = experiment_eval.regression_metrics("test", pretty_print=True)
