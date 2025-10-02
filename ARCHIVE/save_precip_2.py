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

def calc_avg_precip(shp_input, nc_folder, date_input):
    filename = f"IMERG-Final.CLIM.2001-2022.{date_input}.V07B.nc4"
    nc_input = os.path.join(nc_folder, filename)
    
    # Open NetCDF file and extract variables
    with nc.Dataset(nc_input) as dataset:
        precip = dataset.variables['precipitation'][:]  # shape (time, lat, lon) or (lat, lon)
        # print(precip.shape)
        latitude = dataset.variables['lat'][:]  # shape (lat,)
        longitude = dataset.variables['lon'][:]  # shape (lon,)
    
    # Load and reproject the shapefile
    shp = gpd.read_file(shp_input).to_crs('EPSG:4326')
    minlon, minlat, maxlon, maxlat = shp.geometry.total_bounds
    
    # Limit the NetCDF data to the bounding box of the shapefile
    lat_mask = (latitude >= minlat) & (latitude <= maxlat)
    lon_mask = (longitude >= minlon) & (longitude <= maxlon)
    
    # Filter latitude and longitude based on the mask
    lat_filtered = latitude[lat_mask]
    lon_filtered = longitude[lon_mask]
    # print(lat_filtered,lon_filtered)
    
    # Step 1: Create a grid of filtered points
    lon_grid, lat_grid = np.meshgrid(lon_filtered, lat_filtered)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    
    # Check the dimensionality of precip
    if precip.ndim == 2:  # If it's (lat, lon)
        precip_filtered = precip[:,lat_mask][lon_mask,:]  # Apply both lat and lon masks
    elif precip.ndim == 3:  # If it's (time, lat, lon)
        precip_filtered = precip[:, lat_mask, :][:, :, lon_mask]
    
    # Flatten the filtered precipitation array (time dimension included if 3D)
    precip_flat = precip_filtered.ravel()
    
    # Step 2: Load the shapefile and get combined geometry
    shapefile_geom = shp.geometry.unary_union  # Combine all geometries in the shapefile
    
    # Step 3: Identify points intersecting the shapefile
    intersects_mask = contains(shapefile_geom, points[:, 0], points[:, 1])
    
    # Step 4: Filter the points and precipitation values
    filtered_points = points[intersects_mask]
    filtered_precip = precip_flat[intersects_mask]
    
    avg = np.mean(filtered_precip)
    
    return avg * 1e6  # Return in mm

opened = []

file_path = '/global/scratch/users/arvalcarcel/CSMUB/RESULTS/ALL_STATIONS_ALL_MONTHS.csv'
stations_df = pd.read_csv(file_path)

# Example call
precip = np.zeros(len(stations_df))
for i in range(0, len(stations_df)):  # Only processing the first station for now
    date = stations_df['Date']
    number = stations_df['GRDC_No'][i]
    shp_log = stations_df['SHP']
    
    # Read the shapefiles based on shp_log
    shapefile1 = f'/global/home/users/arvalcarcel/ondemand/data/dem/{number}/{number}.shp' 
    shapefile2 = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/SHAPEFILES/{number}/{number}.shp'
    
    if shp_log[i] == 1:
        shapefile = shapefile1
    elif shp_log[i] == 2:
        shapefile = shapefile2
    
    precip_folder = '/global/scratch/users/arvalcarcel/CSMUB/DATA/PRECIP/RESAMPLED/'
    date_input = date[i][-2:]
    
    # Compute average precipitation for this date and shapefile
    avg = calc_avg_precip(shapefile, precip_folder, date_input)
    
    # Assign result back to the matching row
    precip[i] = avg

stations_df['P'] = precip

final_path = '/global/scratch/users/arvalcarcel/CSMUB/RESULTS/ALL_STATIONS_ALL_MONTHS_PRECIP.csv'
stations_df.to_csv(final_path, index=False)