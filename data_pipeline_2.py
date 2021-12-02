import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from data_pipeline.data_generation import (
    generate_segment_data_df,
) 

parser = argparse.ArgumentParser()

parser.add_argument(
    'read_file', 
    help='Name and extension of file to be processed',
    type=str
)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Arguments", args)

    read_file = args.read_file
    save_file, _ = root, ext = os.path.splitext(read_file)
    save_file = save_file + '.csv'
    read_path = 'data/Bus/API Call'
    save_path = 'data/Bus/Segment Data - Raw'

    df = gpd.read_file(os.path.join(read_path, read_file), ignore_geometry=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    segment_data_df = generate_segment_data_df(df)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    segment_data_df.to_csv(os.path.join(save_path, save_file), index=False)
    print(f'segment data saved to: {os.path.join(save_path, save_file)}')
    print()