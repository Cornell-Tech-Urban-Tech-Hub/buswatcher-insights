import streamlit as st
import pandas as pd
from run_experiment import load_pickled_experiment

SAVED_EXPERIMENT_DIR = "saved_experiments/"
DATA_DIR = "data/streamlit/"

def st_demo_weather():
    eval_lr_bus = load_pickled_experiment(SAVED_EXPERIMENT_DIR + 'Linear-Bus.pickle')
    eval_xg_bus = load_pickled_experiment(SAVED_EXPERIMENT_DIR + 'XGBoost-Bus.pickle')
    eval_xg_bus_weather = load_pickled_experiment(SAVED_EXPERIMENT_DIR + 'XGBoost-BusWeather.pickle')

    lr_bus_pred_metrics, mean_pred_metrics = eval_lr_bus.regression_metrics("test", pretty_print=False)
    xg_bus_pred_metrics, _ = eval_xg_bus.regression_metrics("test", pretty_print=False)
    predict_training_mean_MAE = mean_pred_metrics[0]
    predict_training_mean_R2 = mean_pred_metrics[2]
    lr_bus_MAE = lr_bus_pred_metrics[0]
    lr_bus_R2 = lr_bus_pred_metrics[2]
    xg_bus_MAE = xg_bus_pred_metrics[0]
    xg_bus_R2 = xg_bus_pred_metrics[2]
    
    xg_bus_pred_metrics, _ = eval_xg_bus.regression_metrics("test", pretty_print=False)
    xg_bus_weather_pred_metrics, _ = eval_xg_bus_weather.regression_metrics("test", pretty_print=False)
    xg_bus_MAE = xg_bus_pred_metrics[0]
    xg_bus_R2 = xg_bus_pred_metrics[2]
    xg_bus_weather_MAE = xg_bus_weather_pred_metrics[0]
    xg_bus_weather_R2 = xg_bus_weather_pred_metrics[2]

    st.write("""
        ## Motivation
        The motivation behind a lot of this work was to understand how severe weather impacts bus ridership in NYC. Let’s see 
        how the tools we developed can be used to gain insight into that question.
    """)

    st.write("""
        ## Establishing a Baseline
        A sensible first step in assessing the extent to which weather conditions influence the number of people who ride the 
        bus on a given day (in the case of our analysis, the B46 between 8/1 and 9/30/2021) is to establish a baseline predictive 
        model trained in the absence of weather features (e.g., bus position, observation time, and certain timetable details). 
        Then, we can compare the performance of our baseline against that of a substantially similar model trained on exactly the 
        same data plus some additional weather features (e.g., precipitation, temperature, and humidity). If the augmented model 
        performs better than our baseline, we can say that the inclusion of weather features improves our model’s ability to predict 
        bus ridership. Conversely, if the augmented model performs inline with (or worse than) our baseline, we might begin to question 
        if weather has anything to do with people’s decision to ride the bus or, at the very least, attempt to diagnose why our data 
        (or representation thereof) did not lend itself to the prediction task.
    """)

    st.write("""
        ### An Aside on the Prediction Task
        Just to we are all on the same page, the specific task we are training our model to perform is the prediction of the number 
        of passengers on a specific vehicle at a specific time and place. To help illustrate that, here are the first five rows of 
        the combined bus and weather training data.
    """)

    st.dataframe(eval_xg_bus_weather.train.head())

    st.write("""
    We sourced our weather data from the station at JFK, which we accessed through the [VisualCrossing Weather API](https://www.visualcrossing.com/weather-api)
    and joined with the cleaned and processed BusWatcher data on the DateTime column of each dataframe. 
    """)

    st.write(f"""
        ### Model Selection
        #### Linear
        Our first attempt at establishing a baseline model was to train a first-degree, L1-regularized linear regressor the first 6 of 
        our 8 weeks of bus data, then test on the final 2 weeks. To evaluate performance, we looked primarily at mean absolute error (MAE) 
        on basis of interpretability with respect to the prediction task at hand (i.e., how many people are currently on the bus). 
        Overall, our linear model performed surprisingly well for a baseline, with an MAE of {lr_bus_MAE:.1f} on the test set, meaning 
        that, on average, it was able to correctly predict occupancy to within roughly {int(round(lr_bus_MAE))} people relative to the 
        number of riders actually observed. When you stop and think about it, that’s not bad at all. In fact, you probably wouldn’t even 
        notice whether there were 8 more or 8 fewer passengers on a given bus (particularly the larger articulated ones that serve the B46). 
        Indeed, we can see our model does a pretty good job of capturing the ebb end flow of ridership over the course of a day.
    """)

    fig_weekday, fig_weekend, fig_datetime = eval_lr_bus.plot_passenger_count_by_time_of_day('train', segment=None, agg='sum')
    st.write(fig_weekday)
    st.write(fig_weekend)

    st.write("""
        However, that’s not all we care about when evaluating the performance of a regression model. We also want to understand how well it 
        captures the variance in the data. We can see below that our baseline struggles mightily here.
    """)

    fig_weekday, fig_weekend, fig_datetime = eval_lr_bus.plot_passenger_count_by_time_of_day('train', segment=None, agg='mean')
    st.write(fig_weekday)
    st.write(fig_weekend)
    
    st.write(f"""
        This is confirmed by an abysmal R^2 score of {lr_bus_R2:.2f}, meaning that as promising as things were looking for us a moment ago, our 
        baseline is essentially no better than a naive model that always predicts the mean passenger count observed in the training data.
        
        We can do better...
    """)
    
    st.write("""
        #### Gradient Boosted Tree
        To address underfitting, we decided to experiment with a more expressive model class for our second attempt at establishing a baseline, 
        namely Gradient Boosted Trees (XGBoost was the particular implementation we used). Right away, we see a marked improvement in both 
        average error and explanation of variance:
    """)

    index = ['Predict Training Mean', 'Linear', 'XGBoost']
    columns = ['MAE', 'R^2']

    summary_results = pd.DataFrame(
        [
            [predict_training_mean_MAE, predict_training_mean_R2],
            [lr_bus_MAE, lr_bus_R2],
            [xg_bus_MAE, xg_bus_R2]
        ],
        index=index,
        columns=columns
    )

    st.dataframe(summary_results)
    
    agg = st.selectbox("Aggregation Method", options=['Sum', 'Mean'])
    agg = agg.lower()
    fig_weekday, fig_weekend, fig_datetime = eval_xg_bus.plot_passenger_count_by_time_of_day('test', segment=None, agg=agg)
    st.write(fig_weekday)
    st.write(fig_weekend)

    st.write("""
        Despite the improvement in overall fit and prediction variance, our model was still not able to achieve the level of expressiveness we’d 
        like to have seen on a per-stop basis:
    """)

    segments = eval_xg_bus.stop_id_ls
    segment = st.selectbox("Stop", options=segments)
    segment = eval_xg_bus.stop_id2stop_pos[segment]
    agg = st.selectbox("Aggregation Method", options=['Mean', 'Sum'])
    agg = agg.lower()
    fig_weekday, fig_weekend, fig_datetime = eval_xg_bus.plot_passenger_count_by_time_of_day('test', segment=segment, agg=agg)
    st.write(fig_weekday)
    st.write(fig_weekend)

    st.write("""
        Admittedly, there are a lot of potential reasons that one might see this kind of behavior but since we believe it speaks more to 
        higher-level decisions made around problem formulation, data modeling, and training procedures than algorithm selection, we’ll press 
        forward with XGBoost as our baseline for now and leave the discussion of our approach’s shortcomings for a later section.
    """)

    st.write("""
        ## Adding Weather Features
        We can now train a new instance of our baseline model on bus and weather data and compare the results:
    """)

    index = ['XGBoost - Bus', 'XGBoost - Bus + Weather']
    columns = ['MAE', 'R^2']

    summary_results = pd.DataFrame(
        [
            [xg_bus_MAE, xg_bus_R2],
            [xg_bus_weather_MAE, xg_bus_weather_R2],
        ],
        index=index,
        columns=columns
    )

    st.dataframe(summary_results)

    st.write("""
        What we find is that adding weather features actually diminishes our model’s predictive capacity—how disappointing!

        Now, if, hypothetically speaking, you were trying to predict bus occupancy using time, location, and weather data for a course project, 
        you might consider, instead, shifting the focus of your work toward the development of an open-source repo that others can use to push 
        the ball forward, for example. :P

        But let’s be good sports about it and try and gain some insight into what’s going on…

        A good place to start would be to look at the correlations that exist between the various features:
    """)
    
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
    subsets_dict = {
        'All':bus_features_ls + weather_features_ls + ['passenger_count'],
        'Bus':bus_features_ls + ['passenger_count'],
        'Weather':weather_features_ls + ['passenger_count']
    }
    subset = st.selectbox("Feature Subset", options=['All', 'Bus', 'Weather'])
    subset = subsets_dict[subset]
    fig_corr = eval_xg_bus_weather.plot_feature_correlation(subset=subset)
    st.write(fig_corr)

    st.write(f"""
        Interestingly, heat index and relative humidity are the two most highly-correlated features with passenger count.  While this implies 
        that the inclusion of such features would improve the predictions of a linear model, our current baseline has learned non-linear 
        relationships between the features that are more relevant to the prediction task than the linear relationships described in the 
        correlation matrix above. Indeed, although adding weather features to our preliminary linear model, for instance, would see MAE and R^2 
        scores improve to 7.6 and 0.02, from {lr_bus_MAE:.1f} and {lr_bus_R2:.2f}, respectively, it still vastly underperforms the current XGBoost 
        baseline.

        To (begin to) get a sense for the non-linear relationships learned by our baseline, we can inspect feature importance, which, in the context 
        of XGBoost, is a measure of information gain:
    """)
    
    importance, mae, me, r2 = eval_xg_bus_weather.plot_feature_importance(ablate_features=False)
    st.write(importance)

    st.write("""
        Going a step further, we can see how each the inclusion of each successive feature improves or diminishes model performance:
    """)

    importance, mae, me, r2 = eval_xg_bus_weather.plot_feature_importance(ablate_features=True)
    st.write(mae)
    st.write(r2)

    st.write("""
        The upshot is that not only are weather features of minimal importance to our baseline, it can achieve essentially peak performance using 
        only just time and location!
    
        But why is this? Intuitively, weather should have some impact on bus ridership. To help gain some insight into what is going on, let’s 
        look at our training and testing data:
    """)

    st.write("""
        ### Train
    """)
    fig_weekday, fig_weekend, fig_datetime = eval_xg_bus_weather.gt_pred_scatter('train', plot='datetime', errors='large', n=1000, s=0, y_axis='gt', overlay_weather=True)
    st.write(fig_datetime)

    st.write("""
        ### Test
    """)
    fig_weekday, fig_weekend, fig_datetime = eval_xg_bus_weather.gt_pred_scatter('test', plot='datetime', errors='large', n=1000, s=0, y_axis='gt', overlay_weather=True)
    st.write(fig_datetime)
    st.write("""
        We can see that a potential issue from an evaluation perspective is that there are very few weather events in our testing data! 
        How is a model that learns, for example, that fewer people take the bus on hot and humid days supposed to perform on a test set that 
        doesn’t have any hot and humid days?
        
        As with the low variance observed in our baseline model’s stop-specific predictions, we view its inability to glean information related 
        to the prediction task from weather features as more of a high-level issue than one of model selection. Moreover, we believe there exist 
        significant room to improve the approach to this problem than the one outlined above.
    """)

    st.write("""
        ## Error Analysis
        Hypothetically speaking, if you had a reasonably well-functioning model, one of the things you might want to do as part of your 
        evaluation / fine-tuning of it is to inspect where it is making mistakes. 

        A first step might be to see how your model’s predictions compare to ground truth observations:
    """)

    y_axis_dict = {
        'Ground Truth':'gt',
        'Prediction':'pred',
    }
    data = st.selectbox("Dataset", options=['Test', 'Train'], key=1)
    data = data.lower()
    errors = st.selectbox("Errors", options=['Large', 'Small', 'All'], key=1)
    errors = errors.lower()
    n = st.selectbox("Number", options=[5000, 1000, 500, 100, 50], key=1)
    fig_weekday, fig_weekend, fig_datetime = eval_xg_bus_weather.gt_pred_scatter(data=data, plot='simple', errors=errors, n=n)
    st.write(fig_weekday)
    st.write(fig_weekend)

    st.write("""
        But because time and space are dimensions to consider (i.e., date, time of day, bus stop, etc.), being able to disambiguate errors 
        made along these dimensions is critical for model development
    """)

    y_axis_dict = {
        'Ground Truth':'gt',
        'Prediction':'pred',
    }
    data = st.selectbox("Dataset", options=['Test', 'Train'], key=2)
    data = data.lower()
    plot = st.selectbox("Plot By", options=['Stop', 'Hour', 'DateTime'])
    plot = plot.lower()
    errors = st.selectbox("Errors", options=['Large', 'Small', 'All'], key=2)
    errors = errors.lower()
    n = st.selectbox("Number", options=[5000, 1000, 500, 100, 50], key=2)
    y_axis = st.selectbox("Y Axis", options=['Ground Truth', 'Prediction'])
    y_axis = y_axis_dict[y_axis]
    fig_weekday, fig_weekend, fig_datetime = eval_xg_bus_weather.gt_pred_scatter(data=data, plot=plot, errors=errors, n=n, y_axis=y_axis, overlay_weather=True)
    if plot == 'datetime':
        st.write(fig_datetime)
    else:
        st.write(fig_weekday)
        st.write(fig_weekend)

    st.write("""
        ## Parting Thoughts
        Though the weather result was clearly disappointing, we think there exists significant room to improve the methods outlined 
        in this demo. 
        
        First of all, our model does not incorporate the inherently sequential nature of the data. The challenge with this kind of 
        representation, however, is unlike conventional timeseries data, ours is made up of many shorter sequences (i.e., individual 
        buses making stops along a route) that overlap one another and generally don’t individually span the entire time window you 
        are looking at. Moreover, because our data is collected at fixed intervals that in many cases exceed the amount of time it 
        takes a bus to travel between two stops, it is generally not possible to establish a 1:1 mapping between set of datapoints 
        collected from an individual vehicle to the full sequence of stops it is scheduled to make (at least not directly from the 
        data or without significant interpolation). 
        
        Second, we likely didn’t look at a long enough period of time to “see” enough weather events to learn anything meaningful. 
        Additionally, the route we focused on, the B46 runs through the heart of Brooklyn, so commuters who rely on it arguably don’t 
        have the kind of transit alternatives (and may therefore be less sensitive to weather conditions) as those who take the M103, 
        for instance, which parallel to the 6 train along Lexington Ave. in Manhattan for a significant portion of its route. 
        
        Third, by formulating the task as predicting of the number of passengers on a specific vehicle at a specific time and place, we 
        preclude the use of information about the state of the network elsewhere at inference time. A potentially interesting alternative 
        to our problem formulation would be to think about the task as predicting the edge weights of a directed acyclic graph, where edge 
        weights are passenger counts aggregated over some period of time.
    """)








    