import streamlit as st
from interface.st_data import st_data
from interface.st_demo_weather import st_demo_weather
from interface. st_eval_harness import st_eval_harness

# Main page
def render_landing_page():
    st.write("""
        **BusWatcher Insights** is an open-source project which
        aims to build a set of tools for analyzing and predicting bus passenger counts in New York City,
        powered by [NYC BusWatcher](https://github.com/Cornell-Tech-Urban-Tech-Hub/nycbuswatcher) and
        other publicly available Urban data sources.
        It is an open-source collaboration between students at Cornell Tech 
    and the Urban Tech Hub.
    """)

    st.write("""
        The [public repository](https://github.com/Cornell-Tech-Urban-Tech-Hub/buswatcher-insights/) is a set up to showcase the best performing models and
        give insight into experiments. It provides utility methods for fetching data, 
        running experiments for building prediction models and a standardized evaluation suite.
    """)

    st.write("""
        Use the navigation on the left to learn more or explore experiment demos.
    """)

    st.write("**[Github repository](https://github.com/Cornell-Tech-Urban-Tech-Hub/buswatcher-insights/)**")

PAGES = {
    "Introduction": render_landing_page,
    "Data Pipeline": st_data,
    "Experiment & Evaluation Harness": st_eval_harness,
    "Demo: Weather Data": st_demo_weather,
}

PAGE_OPTIONS = list(PAGES.keys())

# Sidebar
st.sidebar.header("BusWatcher | Insights")
st.sidebar.write("**By the [Urban Tech Hub at Cornell Tech](https://urban.tech.cornell.edu/)**")
page_selection = st.sidebar.radio("Page navigation", PAGE_OPTIONS)
page = PAGES[page_selection]
page()
st.sidebar.write("""
**Contributors:**  
Alexander Amy  
Anton Abilov  
Lars Kouwenhoven  
Sanket Shah  
""")