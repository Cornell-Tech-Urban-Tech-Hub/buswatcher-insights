import random
import pandas as pd
import streamlit as st
from interface.st_utils import load_bus_segment_data

def st_data():
    # TODO:
    # - Say something about the value of open city data
    # - Link to notebook, api documentation for the data pipeline
    # - Mention Bx12 and M15 (/other bus routes) in the intro
    # - Say something about other cities

    st.write("""
    ## Data Collection
    We leveraged Dr. Anthony Townsend’s [public NYCBusWatcher API](https://github.com/Cornell-Tech-Urban-Tech-Hub/nycbuswatcher) 
    to collect once-a-minute observations of the location, occupancy, and other data reported by NYC buses via NYC Transit's BusTime API.

    Due to the fixed once-a-minute collection intervals, one of our first preprocessing steps was to filter the data such that each route segment
    (i.e., the space between stops) corresponded with at most one observation per unique trip ID (i.e., the sequence of stops a particular vehicle is 
    scheduled to make on a given day). We made this decision on the basis of passenger count (our ultimate regression target) not changing between stops. 
    The process (visualized conceptually below) had the net effect of reducing the total number of observations in our dataset by about 40%.
    """)

    st.image("./interface/images/data-segments.png")

    # TODO: streamlit map… maybe incorporate functionality to show raw vs. processed data for a given unique trip id?
    # st.write("""
    # On a map this looks like:
    # """)

    routes = ["B46", "Bx12", "M15"]
    route = st.selectbox("Select a bus route", options=routes)
    df_raw = load_bus_segment_data(route, processed=False)
    df_processed = load_bus_segment_data(route, processed=True)
    df_raw['uuid'] = df_raw['trip_id'] + '-' + df_raw['service_date'] + '-' + df_raw['vehicle_id']
    df_processed['uuid'] = df_processed['trip_id'] + '-' + df_processed['service_date'] + '-' + df_processed['vehicle_id']

    st.dataframe(df_raw)
    st.write(len(set(df_raw['uuid'])))
    st.write(df_processed.shape[0])
    st.write(len(set(df_processed['uuid'])))

    # for uuid in set(df_processed['uuid']):
    #     if uuid not in set(df_raw['uuid']):
    #         st.write(uuid)

    random_uuid = random.choice(list(set(df_raw['uuid'])))

    # st.write(set(df_processed['uuid']).intersection(set(df_raw['uuid'])))
    # st.write(set(df_raw['uuid']))

    st.dataframe(df_raw.head())
    st.write(df_raw.shape[0])


    summary_data_dict = {
        'B46':{
            'raw_observations':1101901,
            'raw_vehicles':176,
            'raw_trips':20909,
            'processed_observations':233099,
            'processed_vehicles':84,
            'processed_trips':6696,
        },
        'Bx12':{
            'raw_observations':562680,
            'raw_vehicles':292,
            'raw_trips':18210,
            'processed_observations':14028,
            'processed_vehicles':81,
            'processed_trips':1489,
        },
        'M15':{
            'raw_observations':1099743,
            'raw_vehicles':138,
            'raw_trips':16399,
            'processed_observations':82873,
            'processed_vehicles':28,
            'processed_trips':2025,
        },
    }

    raw_observations = summary_data_dict[route]['raw_observations']
    raw_vehicles = summary_data_dict[route]['raw_vehicles']
    raw_trips = summary_data_dict[route]['raw_trips']
    processed_observations = summary_data_dict[route]['processed_observations']
    processed_vehicles = summary_data_dict[route]['processed_vehicles']
    processed_trips = summary_data_dict[route]['processed_trips']

    index = ['Raw', 'Processed']
    columns = ['Observations', 'Vehicles', 'Trips']

    summary_data = pd.DataFrame(
        [
            [raw_observations, raw_vehicles, raw_trips],
            [processed_observations, processed_vehicles, processed_trips],
        ],
        index=index,
        columns=columns
    )

    st.dataframe(summary_data)

    st.write(df_raw[df_raw['uuid'] == random_uuid].shape[0])
    
    st.write("""
    ### Raw
    """)

    st.map(
        data=df_raw[df_raw['uuid'] == random_uuid]
    )

    st.write("""
    ### Processed
    """)

    st.map(
        data=df_processed[df_processed['uuid'] == random_uuid]
    )

    st.write("""
    The next important decision we made was to think about the two directions a bus could travel along a given route 
    (e.g., uptown vs. downtown) as two distinct routes. 
    This decision facilitated a shift from 100+-dimensional one-hot encodings of segment IDs, 
    to one-dimensional ordinal representations corresponding to their positions along a particular 
    route in a given direction that dramatically reduced the training times and increased the interpretability of our downstream models. 
    """)


    st.write("""
    ## Joining with Weather Data
    Finally, we associated each cleaned and processed BusWatcher datapoint with the nearest sub-hourly weather 
    observation collected by the station at JFK and accessed through the VisualCrossing Weather API.
    """)