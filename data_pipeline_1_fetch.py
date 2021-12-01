import argparse
import datetime
from data_pipeline.data_downloader import save_data

parser = argparse.ArgumentParser()

parser.add_argument(
    'route', 
    help='The route number, i.e. "B46"',
    type=str
)

parser.add_argument(
    '-m',
    '--months', 
    nargs="+",
    default = [datetime.date.today().month],
    help='The month(s), i.e. 8 for August (separate by space); defaults to current month',
    type=int
)

parser.add_argument(
    '-y',
    '--years', 
    nargs="+",
    default = [datetime.date.today().year],
    help='The year(s) (separate by space); defaults to current year',
    type=int
)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Arguments", args)

    route = args.route
    months = args.months
    years = args.years
    
    # fetch data
    save_data(
        route=route, 
        months=months,  
        years=years, 
    )
