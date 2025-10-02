# IMPORT PACKAGES
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import shapefile
import os

import netCDF4 as nc
from netCDF4 import Dataset
from shapely.geometry import Point, shape, box, mapping
from shapely.vectorized import contains
from shapely.strtree import STRtree

from rasterio.coords import BoundingBox
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.features import shapes

def average_precip_single(shp_input, nc_dir, date_str):
    yyyymm = date_str.replace('-', '')
    filename = f"MERRA2_400.tavgM_2d_lnd_Nx.{yyyymm}.nc4"
    filepath = os.path.join(nc_dir, filename)

    if not os.path.exists(filepath):
        return np.nan

    try:
        shp = gpd.read_file(shp_input).to_crs('EPSG:4326')
        geom = shp.geometry.unary_union

        with nc.Dataset(filepath) as dataset:
            precip = dataset.variables['PRECTOTLAND'][:]
            lat = dataset.variables['lat'][:]
            lon = dataset.variables['lon'][:]

        if precip.ndim == 3:
            precip = precip.mean(axis=0)

        minlon, minlat, maxlon, maxlat = geom.bounds
        lat_mask = (lat >= minlat) & (lat <= maxlat)
        lon_mask = (lon >= minlon) & (lon <= maxlon)

        lat_filtered = lat[lat_mask]
        lon_filtered = lon[lon_mask]
        precip_filtered = precip[np.ix_(lat_mask, lon_mask)]

        lon_grid, lat_grid = np.meshgrid(lon_filtered, lat_filtered)
        lon_flat = lon_grid.ravel()
        lat_flat = lat_grid.ravel()
        precip_flat = precip_filtered.ravel()

        mask = contains(geom, lon_flat, lat_flat)
        if not np.any(mask):
            return np.nan

        return np.mean(precip_flat[mask])

    except Exception as e:
        print(f"Error for {date_str}, {shp_input}: {e}")
        return np.nan


# DEFINE INPUTS FOR FUNCTION
opened = []

csv_path = '/global/scratch/users/arvalcarcel/CSMUB/RESULTS/CSV/'
masterlist = '/global/scratch/users/arvalcarcel/CSMUB/RESULTS/ALL_STATIONS_FINAL_REVISED.csv'

# Load the master list of stations
stations_df = pd.read_csv(masterlist)

# Extract station numbers, areas, and latitudes
station_num = stations_df['grdc_no']
station_area = stations_df['area']
station_lat = stations_df['lat']
station_shp = stations_df['shapefile_code']
station_avgslope = stations_df['avg_slope']
station_maxslope = stations_df['max_slope']
station_aridity = stations_df['avg_aridity']

# Map station numbers to areas and latitudes
station_area_map = dict(zip(station_num, station_area))
station_lat_map = dict(zip(station_num, station_lat))
station_shp_map = dict(zip(station_num, station_shp))
station_avgslope_map = dict(zip(station_num, station_avgslope))
station_maxslope_map = dict(zip(station_num, station_maxslope))
station_aridity_map = dict(zip(station_num, station_aridity))

# Generate the list of file paths
arrayFile = [os.path.join(csv_path, f"{station_no}.csv") for station_no in station_num]

# Initialize a list to store opened DataFrames
for file in arrayFile:
    station_no = os.path.basename(file).split('.')[0]
    # print(station_no)# Extract station number from the filename
    if os.path.exists(file):  # Check if file exists
        df = pd.read_csv(file, index_col=None, header=0)
        station_no_int = int(station_no)  # Convert station number to integer for lookup
        df['GRDC_No'] = station_no_int  # Add the station number as a new column
        df['shp_code'] = station_shp_map.get(station_no_int, None)
        df['Area'] = station_area_map.get(station_no_int, None)  # Add the Area column
        df['Latitude'] = station_lat_map.get(station_no_int, None)  # Add the latitude column
        df['Avg Slope'] = station_avgslope_map.get(station_no_int, None)
        df['Max Slope'] = station_maxslope_map.get(station_no_int, None)
        df['Aridity'] = station_aridity_map.get(station_no_int, None)
        opened.append(df)

# Combine all DataFrames into one
total_df = pd.concat(opened, axis=0, ignore_index=True)
stations_df['P'] = np.nan
# Print or save the resulting DataFrame
# print(total_df)


for i in range(0,len(total_df)):
# i = 0
    
    date = total_df['Date']
    number = total_df['GRDC_No'][i]
    shp_log = total_df['shp_code']
    
    # Read the shapefiles
    shapefile1 = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/STATIONS/{number}/{number}.shp' # delineated shapefile
    shapefile2 = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/SHAPEFILES/{number}/{number}.shp' # GRDC shapefile
    
    if shp_log[i] == 1:
        shapefile = shapefile1
    
    elif shp_log[i] == 2:
        shapefile = shapefile2
    
    # print(shapefile)
    
    precip_folder = '/global/scratch/users/arvalcarcel/CSMUB/DATA/PRECIP/'
    date_input = date[i]
    
    # Compute avg precip for this date and shapefile
    avg = average_precip_single(shapefile, precip_folder, date_input)*1e6
    
    # Assign result back to the matching rows
    stations_df['P'][i] = avg

results_csv = f"/global/scratch/users/arvalcarcel/CSMUB/RESULTS/ALL_STATIONS_REVISED_PRECIP.csv"
stations_df.to_csv(results_csv)
# print(pd.read_csv(results_csv))


