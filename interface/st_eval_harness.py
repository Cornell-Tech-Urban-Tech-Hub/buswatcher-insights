import streamlit as st
import os
from experiment_pipeline.data_loader import load_global_feature_set
import pandas as pd
from interface.st_utils import SAVED_EXPERIMENT_DIR
from run_experiment import load_pickled_experiment

def st_eval_harness():
    st.header("Experimentation & Evaluation Harness")
    st.write("""
        The repository includes an experiment and evaluation harness built on 
        top of [scikit-learn](https://scikit-learn.org/stable/), which allows
        anyone to easily run an experiment.

        [Explore the weather demo notebook which showcases the API](https://github.com/alexanderamy/BusPassengerPred/blob/main/notebooks/demo.ipynb)
    """)


    ## run experiment

    st.write("""

    ##### run_experiment API
    ```python
    experiment_eval = run_experiment(
        global_feature_set=df_route,
        feature_extractor_fn=bus_and_weather_features,
        model=XGBRegressor(...),
        stop_id_ls=stop_id_ls,
        dependent_variable="passenger_count",
        test_period="14D",
        refit_interval="1D",
        experiment_name="test"
    )
    ```

    The entrypoint to the evaluation harness is the [run_experiment](https://github.com/Cornell-Tech-Urban-Tech-Hub/buswatcher-insights/blob/main/run_experiment.py) API which allows users to pass a custom model, feature set, feature extractor and testing period.
    The method runs an experiment on the specified testing period and refits the model iteratively according to the refit interval. It returns an `Evaluation` class.

    #### Evaluation class
    The [Evaluation class](https://github.com/Cornell-Tech-Urban-Tech-Hub/buswatcher-insights/blob/main/experiment_pipeline/evaluation.py) has a set of evaluation methods and plotting methods built-in to standardize evaluation across models.
    """)

    st.header("Showcase")
    st.write("Select to explore an experiment - loads experiment with according description & evaluation of results")
    experiments = os.listdir(SAVED_EXPERIMENT_DIR)
    selected_experiment = st.selectbox("Select experiment", options=experiments)
    experiment_eval = load_pickled_experiment(SAVED_EXPERIMENT_DIR + selected_experiment)

    fig1, fig2, fig3 = experiment_eval.plot_passenger_count_by_time_of_day('test')
    st.write(fig1)
