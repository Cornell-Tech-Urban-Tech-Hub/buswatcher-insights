import streamlit as st
import pandas as pd
import os

from run_experiment import load_pickled_experiment

SAVED_EXPERIMENT_DIR = "saved_experiments/"
DATA_DIR = "data/streamlit/"

def load_bus_segment_data(route, processed=True):
    try:
        df_route = pd.read_pickle(os.path.join(
            DATA_DIR,
            "processed" if processed else "raw",
            f"{route}_2021-10-18.pickle"
        ))
    except Exception as e:
        st.error(f"Failed to load data for route {route}: {e}")
        return
    return df_route

@st.cache(allow_output_mutation=True)
def st_load_experiment(name):
    return load_pickled_experiment(os.path.join(
        SAVED_EXPERIMENT_DIR,
        name
    ))
