import streamlit as st
import os
from experiment_pipeline.data_loader import load_global_feature_set
import pandas as pd
from interface.st_utils import SAVED_EXPERIMENT_DIR
from run_experiment import load_pickled_experiment

def st_eval_harness():
    st.header("Experimentation & Evaluation Harness")
    st.write("""
        ...
    """)

    st.header("Experiments")
    st.write("Select to explore an experiment - loads experiment with according description & evaluation of results")
    experiments = os.listdir(SAVED_EXPERIMENT_DIR)
    selected_experiment = st.selectbox("Select experiment", options=experiments)
    experiment_eval = load_pickled_experiment(SAVED_EXPERIMENT_DIR + selected_experiment)

    fig1, fig2, fig3 = experiment_eval.plot_passenger_count_by_time_of_day('test')
    st.write(fig1)
