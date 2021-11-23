import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

def load_data_range(
    start_date, end_date, route, api_url='http://api.buswatcher.org/api/v2'
):
    """
    Loads data from start_date until end_date (exclusive) for route
    start_date: String YYYY-MM-DD
    end_date: String YYYY-MM-DD
    String route
    String api_url: base url of API
    """
    rows = []
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    dates = pd.date_range(start_date, end_date - timedelta(days=1), freq='d')
    for date in dates:
        print("Working on", date)
        year = str(date.year)
        month = str(date.month)
        day = str(date.day)
        for hour in range(24):
            hour = str(hour)
            url = f'{api_url}/nyc/{year}/{month}/{day}/{hour}/{route}/buses/'
            try:
                shipment = requests.get(url).json()
                for bus_dict in shipment['buses']:
                    rows.append(bus_dict)
            except Exception as e:
                print(e)

    df = pd.DataFrame.from_dict(rows, orient='columns')
    
    return df


def load_alerts(list_of_files):
    all_alerts = []
    for file_name in list_of_files:
        with open (file_name, 'r') as f:
            data = json.load(f)
            all_alerts += data
    alert_df = pd.DataFrame(all_alerts)
    return alert_df
