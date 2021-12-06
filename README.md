# BusWatcher Insights
**BusWatcher Insights** is an open-source project which aims to build a set of tools for analyzing and predicting bus passenger counts in New York City, powered by [NYC BusWatcher](https://github.com/Cornell-Tech-Urban-Tech-Hub/nycbuswatcher) and other publicly available Urban data sources. It is an open-source collaboration between students at Cornell Tech and [the Urban Tech Hub](https://urban.tech.cornell.edu/).

This repository provides utility methods for fetching data, running experiments for building prediction models and a standardized evaluation suite.
## Install package
Locally, install the `passenger_pred` package by first cloning this repository, and then running the following command:
`pip install -e .`

Within Python, the package can then be loaded using
`import passenger_predict`.

## Report directory
This section lists the different reports that we have written, and describes how they relate to each other.

# Experiment Pipeline

### Setup
1. `conda create -n bus_prediction python=3.8`
2. `conda activate bus_prediction` 
3. `pip install -r requirements.txt`
4. Get `Bus` data folder from the shared drive and put it into the `data/` folder.


### Run an experiment
1. `python run_experiment.py ./data B46 JFK 1 -t 4D -n test`

Or see the `notebooks/` folder for an example.

# Streamlit Interface
1. Install dependencies `pip install -r requirements.txt`
2. Run streamlit app `streamlit run st_main.py`