import os
import argparse
import datetime
from data_pipeline.data_downloader import get_shipments

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

    save_path = 'data/Bus/API Call'
    today = datetime.date.today()
    save_file = f'{route}_{today}.geojson'

    gdf = get_shipments(route, months=months, years=years)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gdf.to_file(os.path.join(save_path, save_file), driver='GeoJSON')
    print(f'bus data saved to: {os.path.join(save_path, save_file)}')
    print()
