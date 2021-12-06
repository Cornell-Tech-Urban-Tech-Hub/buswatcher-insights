import streamlit as st
from interface.st_data import st_data
from interface.st_demo_weather import st_demo_weather
from interface. st_eval_harness import st_eval_harness

# Main page
def render_landing_page():
    st.header("BusWatcher | Insights")
    st.subheader("Data Science Toolset")
    st.write("""
             BusWatcher Insights is an open source library that aims to provide researchers a 
             scaffold for data engineering and analysis. Currently it's main focus is on passenger 
             count data from the NYC bys system. The public repository is a set up to showcase 
             the best performing models and give insight into experiments. It provides utility methods for fetching data, running experiments for building prediction models and a
             standardized evaluation suite.
             
             The work presented here was born of an endeavor to study the impact of weather data 
             on bus rider volume. Over the course of the project, the team had built up a substantial 
             data engineering code base. After many challenges with the initial study direction, 
             the team decided to refactor and rebuild the code base into a user-friendly public 
             library. 
             
             The library is modular and sufficiently flexible to serve a data science workflow, 
             allowing a researcher to audit data, perform feature selection, compare models, and 
             present findings, in a manageable, piecemeal fashion without much overhead.
             BusWatcher Insights is powered by NYC BusWatcher and other publicly available Urban 
             data sources. It is a collaboration between students at Cornell Tech and the Urban 
             Tech Hub.
            
             Use the navigation on the left to learn more or explore experiment demos.
             """
             )

    st.write("**[Github repository](https://github.com/Cornell-Tech-Urban-Tech-Hub/buswatcher-insights/)**")
    st.image("./interface/images/workflow.png")

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