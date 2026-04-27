# IMPORT PACKAGES
import pysheds
from pysheds.grid import Grid
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import fiona
import xarray as xr
import shapefile
import math
from scipy import stats
import os
from numpy import random
import netCDF4 as nc
from netCDF4 import Dataset
from shapely.geometry import Point, shape, box, mapping, Polygon, MultiPolygon
from shapely.vectorized import contains
from shapely.strtree import STRtree
import matplotlib.path as mpath
from scipy.ndimage import sobel
from rasterio.transform import rowcol
from rasterio.coords import BoundingBox
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.features import shapes

def get_stationdata(station_file):
    # Load data
    df_Q = pd.read_csv(station_file, delimiter=';', encoding='utf-8', skiprows=36)
    
    # Clean column names
    df_Q.columns = df_Q.columns.str.replace(' ', '')
    
    # Convert dates to datetime (keep full daily precision)
    df_Q['YYYY-MM-DD'] = pd.to_datetime(df_Q['YYYY-MM-DD'], format='%Y-%m-%d')
    
    # Crop by date
    start_date = pd.to_datetime('2018-08-31')
    end_date = pd.to_datetime('2025-12-31')
    df_filtered = df_Q[(df_Q['YYYY-MM-DD'] >= start_date) & (df_Q['YYYY-MM-DD'] <= end_date)]
    
    # Extract values
    stream_gauge = df_filtered['Value'].values
    
    # Reset index for dates
    dates = df_filtered['YYYY-MM-DD'].reset_index(drop=True)
    dates = dates.reset_index()
    
    return stream_gauge, dates



def waterpx_count(shp_input, nc_input):

  # Open NetCDF file and extract variables
  with nc.Dataset(nc_input) as dataset:
      watermask = dataset.variables['watermask'][:]
      latitude = dataset.variables['lat'][:]
      longitude = dataset.variables['lon'][:]

  # Load and reproject the shapefile
  shp = gpd.read_file(shp_input).to_crs('EPSG:4326')
  minlon, minlat, maxlon, maxlat = shp.geometry.total_bounds

  # Limit the NetCDF data to the bounding box of the shapefile
  lat_mask = (latitude >= minlat) & (latitude <= maxlat)
  lon_mask = (longitude >= minlon) & (longitude <= maxlon)

  watermask = watermask[lat_mask, :][:, lon_mask]
  lat_filtered = latitude[lat_mask]
  lon_filtered = longitude[lon_mask]

  # Step 1: Create a grid of filtered points
  lon_grid, lat_grid = np.meshgrid(lon_filtered, lat_filtered)
  points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

  # Flatten the watermask array to align with points
  watermask_flat = watermask.ravel()

  # Step 2: Load the shapefile and get combined geometry
  shapefile_geom = shp.geometry.unary_union  # Combine all geometries in the shapefile

  # Step 3: Identify points intersecting the shapefile
  intersects_mask = contains(shapefile_geom, points[:, 0], points[:, 1])

  # Step 4: Filter the points and watermask values
  filtered_points = points[intersects_mask]
  filtered_watermask = watermask_flat[intersects_mask]

  # Step 5: Create geometries for intersecting points
  geometries = [Point(lon, lat) for lon, lat in filtered_points]

  # Step 6: Create a GeoDataFrame
  gdf = gpd.GeoDataFrame({'watermask': filtered_watermask}, geometry=geometries, crs='EPSG:4326')


  # markercolormap2= colors.ListedColormap(['white', 'black','blue'])

  # # Assuming gdf and water_percent are defined, and markercolormap2 is valid
  # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
  # gdf.plot(column='watermask', ax=ax, vmin=0, vmax=3, legend=True, markersize=5, cmap=markercolormap2)
  # ax.set_title(nc_input)
  # ax.set_xlabel('Longitude')
  # ax.set_ylabel('Latitude')
  # plt.show()

  # Step 1: Filter the GeoDataFrame where 'watermask' is equal to 2
  filtered_gdf = gdf[gdf['watermask'] == 1]

  water_pixels = len(filtered_gdf)
  total_pixels = len(gdf)

  water_percent = (water_pixels / total_pixels) * 100

  return water_pixels, water_percent, total_pixels

def calc_avg_precip(shp_input, date_input):
    precip_folder = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/PRECIP/RESAMPLED/'
    filename = f"IMERG-Final.CLIM.2001-2022.{date_input}.V07B.nc4"
    
    nc_input = os.path.join(precip_folder, filename)
    
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

    return avg

# DEFINE INPUTS FOR FUNCTION
masterlist = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/GRDC_Stations_UPDATED.csv'
stations_df = pd.read_csv(masterlist)
station_num = stations_df['grdc_no']
final_result = pd.DataFrame()
print(f"Loaded {len(station_num)} stations.") # as of Nov 2025: should be 578 stations total when loading all

import warnings
warnings.filterwarnings("ignore")
# Loop over selected stations
for s in range(110,len(station_num)):

    station = station_num[s]
    print("Running station ", station)

    data = stations_df.iloc[s]
    number = station
    region = data['wmo_reg']
    river = data['river']
    name = data['station']
    lat = data['lat']
    lon = data['long']
    area = data['area_delin']
    altitude = data['altitude']

    # Paths
    q_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/sls_GRDC_shp/GRDC_shp/{number}_Q_Day.Cmd.txt' # GRDC monthly streamgauge readings as .txt files
    shp_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/DEM/STATIONS/{station}/{station}.shp'
    dem_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/DEM/STATIONS/{station}_dem.tif'

    # Load streamflow and dates
    results_df = pd.DataFrame(stations_df.iloc[[s]])
    stream_gauge, dates = get_stationdata(q_file)  # returns daily timestamps

    # Initialize water pixel arrays
    water_px = np.zeros(len(dates))
    water_pcnt = np.zeros(len(dates))
    water_area = np.zeros(len(dates))
    tot_px = np.zeros(len(dates))

    # NEW: dir logic
    ncdf_list = []
    for i in range(0,len(dates)):
        date = dates['YYYY-MM-DD'][i]
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)

        # Using cutoff to swap
        cutoff = pd.to_datetime('2025-07-27')
        if date < cutoff:
            directory = "/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_from_PODAAC/Daily/Daily_For_Trend"
        else:
            directory = f"/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/sls_CYGNSS_DAILY_WATERMASK/{date.year}"

        date_str = date.strftime('%Y-%m-%d')
        filename = f"cyg.ddmi.{date_str}.l3.uc-berkeley-watermask-daily.a32.d33.nc"

        # Check if file exists using scandir for speed
        file_exists = any(entry.name == filename and entry.is_file() for entry in os.scandir(directory))
        if file_exists:
            ncdf_list.append(os.path.join(directory, filename))
        else:
            print(f"File not found: {os.path.join(directory, filename)}")

    # Calculate water pixels
    for f, filename in enumerate(ncdf_list):
        pixel_count, pixel_percent, total_pixels = waterpx_count(shp_file, filename)
        water_px[f] = pixel_count
        water_pcnt[f] = pixel_percent
        tot_px[f] = total_pixels
        water_area[f] = (pixel_percent * area) / 100

    # Compute monthly precipitation once per month
    monthly_precip = {}
    dates_pd = pd.to_datetime(dates['YYYY-MM-DD'])
    precip = np.zeros(len(dates))

    for idx, date in enumerate(dates_pd):
        month_str = str(date)[5:7]    # YYYY-MM
        if month_str not in monthly_precip:
            monthly_precip[month_str] = calc_avg_precip(shp_file, month_str)
        precip[idx] = monthly_precip[month_str]

    # Save Individual Results   
    df_final = pd.DataFrame({'Date': dates['YYYY-MM-DD'], 'Q': stream_gauge,'SWE': water_pcnt, 'SWE_scaled': water_area, 'P': precip})
    df_final.to_csv(f"/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/RESULTS/sls_Daily/{number}.csv")
    print("Finished running station ", number)
    
