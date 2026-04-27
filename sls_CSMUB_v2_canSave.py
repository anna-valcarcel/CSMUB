# =========================
# IMPORT PACKAGES
# =========================
print('importing packages')
import pysheds
from pysheds.grid import Grid
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
import rasterio
import time
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
import warnings
warnings.filterwarnings("ignore")

# Parallelization packages
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
print('imported packages')

# =========================
# FUNCTIONS
# =========================
def get_stationdata(station_file):
    df_Q = pd.read_csv(station_file, delimiter=';', encoding='utf-8', skiprows=36)
    df_Q.columns = df_Q.columns.str.replace(' ', '')
    df_Q['YYYY-MM-DD'] = pd.to_datetime(df_Q['YYYY-MM-DD'], format='%Y-%m-%d')
    start_date = pd.to_datetime('2018-08-31')
    end_date = pd.to_datetime('2025-12-31')
    df_filtered = df_Q[(df_Q['YYYY-MM-DD'] >= start_date) & (df_Q['YYYY-MM-DD'] <= end_date)]
    try:
        stream_gauge = df_filtered['Value'].values
    except: 
        stream_gauge = df_filtered[' Value'].values
    dates = df_filtered['YYYY-MM-DD'].reset_index(drop=True).reset_index()
    return stream_gauge, dates

def waterpx_count(shp_input, nc_input):
    with nc.Dataset(nc_input) as dataset:
        watermask = dataset.variables['watermask'][:]
        latitude = dataset.variables['lat'][:]
        longitude = dataset.variables['lon'][:]

    shp = gpd.read_file(shp_input).to_crs('EPSG:4326')
    minlon, minlat, maxlon, maxlat = shp.geometry.total_bounds

    lat_mask = (latitude >= minlat) & (latitude <= maxlat)
    lon_mask = (longitude >= minlon) & (longitude <= maxlon)
    watermask = watermask[lat_mask, :][:, lon_mask]
    lat_filtered = latitude[lat_mask]
    lon_filtered = longitude[lon_mask]

    lon_grid, lat_grid = np.meshgrid(lon_filtered, lat_filtered)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    watermask_flat = watermask.ravel()
    shapefile_geom = shp.geometry.unary_union
    intersects_mask = contains(shapefile_geom, points[:, 0], points[:, 1])
    filtered_points = points[intersects_mask]
    filtered_watermask = watermask_flat[intersects_mask]
    geometries = [Point(lon, lat) for lon, lat in filtered_points]
    gdf = gpd.GeoDataFrame({'watermask': filtered_watermask}, geometry=geometries, crs='EPSG:4326')

    filtered_gdf = gdf[gdf['watermask'] == 1]
    water_pixels = len(filtered_gdf)
    total_pixels = len(gdf)
    water_percent = (water_pixels / total_pixels) * 100

    return water_pixels, water_percent, total_pixels

def calc_avg_precip(shp_input, date_input):
    precip_folder = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/PRECIP/RESAMPLED/'
    filename = f"IMERG-Final.CLIM.2001-2022.{date_input}.V07B.nc4"
    nc_input = os.path.join(precip_folder, filename)

    with nc.Dataset(nc_input) as dataset:
        precip = dataset.variables['precipitation'][:]
        latitude = dataset.variables['lat'][:]
        longitude = dataset.variables['lon'][:]

    shp = gpd.read_file(shp_input).to_crs('EPSG:4326')
    minlon, minlat, maxlon, maxlat = shp.geometry.total_bounds
    lat_mask = (latitude >= minlat) & (latitude <= maxlat)
    lon_mask = (longitude >= minlon) & (longitude <= maxlon)
    lat_filtered = latitude[lat_mask]
    lon_filtered = longitude[lon_mask]

    lon_grid, lat_grid = np.meshgrid(lon_filtered, lat_filtered)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    if precip.ndim == 2:
        precip_filtered = precip[:, lat_mask][lon_mask, :]
    elif precip.ndim == 3:
        precip_filtered = precip[:, lat_mask, :][:, :, lon_mask]

    precip_flat = precip_filtered.ravel()
    shapefile_geom = shp.geometry.unary_union
    intersects_mask = contains(shapefile_geom, points[:, 0], points[:, 1])
    filtered_precip = precip_flat[intersects_mask]
    avg = np.mean(filtered_precip)

    return avg

# =========================
# MAIN STATION PROCESSING FUNCTION
# =========================
def process_station(s, station_num, stations_df):
    start_time = time.time() # record time
    station = station_num[s]
    # Exit if station already been run
    save_path = f"/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/RESULTS/sls_Daily/{station}_parallel.csv"
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"Skipping station {station} (already processed).")
        return

    save_path2 = f"/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/RESULTS/sls_Daily/{station}_parallel2.csv"
    if os.path.exists(save_path2) and os.path.getsize(save_path2) > 0:
        print(f"Skipping station {station} (already processed).")
        return
    print(f"Station {station} has not yet been processed. Processing now...")

    data = stations_df.iloc[s]
    number = station
    region = data['wmo_reg']
    river = data['river']
    name = data['station']
    lat = data['lat']
    lon = data['long']
    area = data['area_delin']
    altitude = data['altitude']
    print(f"Processing station {number}")
    q_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/sls_GRDC_shp/GRDC_shp/{number}_Q_Day.Cmd.txt'
    shp_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/DEM/STATIONS/{station}/{station}.shp'
    dem_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/DEM/STATIONS/{station}_dem.tif'

    # Load streamflow and dates
    stream_gauge, dates = get_stationdata(q_file)

    water_px = np.zeros(len(dates))
    water_pcnt = np.zeros(len(dates))
    water_area = np.zeros(len(dates))
    tot_px = np.zeros(len(dates))

    # Gather NetCDF files
    ncdf_list = []
    for i in range(len(dates)):
        date = pd.to_datetime(dates['YYYY-MM-DD'][i])
        cutoff = pd.to_datetime('2025-07-27')
        if date < cutoff:
            directory = "/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_from_PODAAC/Daily/Daily_For_Trend"
        else:
            directory = f"/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/sls_CYGNSS_DAILY_WATERMASK/{date.year}"

        date_str = date.strftime('%Y-%m-%d')
        filename = f"cyg.ddmi.{date_str}.l3.uc-berkeley-watermask-daily.a32.d33.nc"

        file_exists = any(entry.name == filename and entry.is_file() for entry in os.scandir(directory))
        if file_exists:
            ncdf_list.append(os.path.join(directory, filename))
        else:
            print(f"File not found: {os.path.join(directory, filename)}")

    # Parallel processing for water pixels per NetCDF file
    results = Parallel(n_jobs=4)(
        delayed(waterpx_count)(shp_file, f) for f in ncdf_list
    )
    for idx, (pixel_count, pixel_percent, total_pixels) in enumerate(results):
        water_px[idx] = pixel_count
        water_pcnt[idx] = pixel_percent
        tot_px[idx] = total_pixels
        water_area[idx] = (pixel_percent * area) / 100

    # Compute monthly precipitation per station
    dates_pd = pd.to_datetime(dates['YYYY-MM-DD'])
    months = sorted(set(str(date)[5:7] for date in dates_pd))
    monthly_precip_values = Parallel(n_jobs=4)(
        delayed(calc_avg_precip)(shp_file, month) for month in months
    )
    monthly_precip = dict(zip(months, monthly_precip_values))
    precip = np.array([monthly_precip[str(date)[5:7]] for date in dates_pd])

    # Save results
    df_final = pd.DataFrame({
        'Date': dates['YYYY-MM-DD'],
        'Q': stream_gauge,
        'SWE': water_pcnt,
        'SWE_scaled': water_area,
        'P': precip
    })
    save_path3 = f"/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/RESULTS/sls_Daily/{station}_parallel_rerun.csv"
    print(f'Saving to {save_path3}')
    df_final.to_csv(save_path3, index=False)
    print(f'Saved')
    end_time = time.time()  # End timer
    elapsed = end_time - start_time
    print(f"Finished running station {number} in {elapsed:.2f} seconds")

# =========================
# MAIN SCRIPT
# =========================
'''
if __name__ == "__main__":
    print('script starting')
    masterlist = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/GRDC_Stations_UPDATED.csv'
    stations_df = pd.read_csv(masterlist)
    station_num = stations_df['grdc_no']

    print(f"Loaded {len(station_num)} stations.")

    # Parallel processing over stations
    num_processes = min(cpu_count(), 8)  # Adjust depending on your cluster
    with Pool(num_processes) as pool:
        pool.starmap(process_station, [(s, station_num, stations_df) for s in range(0, len(station_num))])
'''
if __name__ == "__main__":
    print('script starting, with save failed stations logic')
    masterlist = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/GRDC_Stations_UPDATED.csv'
    stations_df = pd.read_csv(masterlist)
    station_num = stations_df['grdc_no']

    print(f"Loaded {len(station_num)} stations.")
    num_processes = min(cpu_count(), 8)

    timeout_seconds = 7200  # 2hr per station
    failed_stations = []
    stations_missed=[0,1]

    with Pool(num_processes) as pool:
        results = []
        for s in stations_missed:#range(len(station_num)):
            r = pool.apply_async(process_station, (s, station_num, stations_df))
            results.append((s, r))

        for s, r in results:
            try:
                r.get(timeout=timeout_seconds)
            except Exception as e:
                print(f"Station {station_num[s]} timed out or crashed. Skipping. Error: {e}")
                failed_stations.append(station_num[s])
    
                fail_path = "/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/RESULTS/sls_Daily/failed_stations5.txt"
                with open(fail_path, "w") as f:
                    for st in failed_stations:
                        f.write(str(st) + "\n")

                print(f"Saved {len(failed_stations)} failed stations to {fail_path}")
