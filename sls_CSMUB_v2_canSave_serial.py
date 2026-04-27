# =========================
# IMPORT PACKAGES
# =========================
print('importing packages')
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
from shapely.vectorized import contains
import warnings
warnings.filterwarnings("ignore")
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

def waterpx_count(shp_geom, bbox, nc_data):
    """Count water pixels within the shapefile for a single NetCDF dataset (preloaded arrays)."""
    watermask = nc_data['watermask']
    lat = nc_data['lat']
    lon = nc_data['lon']

    minlon, minlat, maxlon, maxlat = bbox
    lat_mask = (lat >= minlat) & (lat <= maxlat)
    lon_mask = (lon >= minlon) & (lon <= maxlon)
    watermask = watermask[lat_mask, :][:, lon_mask]
    lat_filtered = lat[lat_mask]
    lon_filtered = lon[lon_mask]

    lon_grid, lat_grid = np.meshgrid(lon_filtered, lat_filtered)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    watermask_flat = watermask.ravel()
    intersects_mask = contains(shp_geom, points[:, 0], points[:, 1])
    filtered_watermask = watermask_flat[intersects_mask]

    water_pixels = np.sum(filtered_watermask == 1)
    total_pixels = len(filtered_watermask)
    water_percent = (water_pixels / total_pixels * 100) if total_pixels > 0 else 0
    return water_pixels, water_percent, total_pixels

def calc_avg_precip(shp_geom, bbox, nc_file_path):
    """Average precipitation over the shapefile for a single NetCDF file."""
    with nc.Dataset(nc_file_path) as dataset:
        precip = dataset.variables['precipitation'][:]
        lat = dataset.variables['lat'][:]
        lon = dataset.variables['lon'][:]

    minlon, minlat, maxlon, maxlat = bbox
    lat_mask = (lat >= minlat) & (lat <= maxlat)
    lon_mask = (lon >= minlon) & (lon <= maxlon)
    lat_filtered = lat[lat_mask]
    lon_filtered = lon[lon_mask]

    lon_grid, lat_grid = np.meshgrid(lon_filtered, lat_filtered)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    if precip.ndim == 2:
        precip_filtered = precip[:, lat_mask][lon_mask, :]
    elif precip.ndim == 3:
        precip_filtered = precip[:, lat_mask, :][:, :, lon_mask]

    precip_flat = precip_filtered.ravel()
    intersects_mask = contains(shp_geom, points[:, 0], points[:, 1])
    filtered_precip = precip_flat[intersects_mask]
    avg = np.mean(filtered_precip) if len(filtered_precip) > 0 else np.nan
    return avg

# =========================
# PROCESS SINGLE STATION (SERIAL)
# =========================
def process_station(s, station_num, stations_df, nc_cache, precip_folder):
    station = station_num[s]
    print(f'Processing Station {station}')
    start_time = time.time()

    # Skip if already processed
    save_path = f"/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/RESULTS/sls_Daily/{station}_parallel.csv"
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"Skipping station {station} (already processed).")
        return

    data = stations_df.iloc[s]
    area = data['area_delin']
    shp_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/DEM/STATIONS/{station}/{station}.shp'
    shp = gpd.read_file(shp_file).to_crs('EPSG:4326')
    shp_geom = shp.geometry.unary_union
    bbox = shp.geometry.total_bounds

    # Streamflow and dates
    q_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/sls_GRDC_shp/GRDC_shp/{station}_Q_Day.Cmd.txt'
    stream_gauge, dates = get_stationdata(q_file)
    dates_pd = pd.to_datetime(dates['YYYY-MM-DD'])

    water_px = np.zeros(len(dates))
    water_pcnt = np.zeros(len(dates))
    water_area = np.zeros(len(dates))
    tot_px = np.zeros(len(dates))

    # Select relevant CYGNSS datasets from cache
    ncdf_list = [f for f in nc_cache if any(date.strftime('%Y-%m-%d') in os.path.basename(f) for date in dates_pd)]

    for idx, nc_file in enumerate(ncdf_list):
        nc_data = nc_cache[nc_file]
        if nc_data is None:
            continue
        pixel_count, pixel_percent, total_pixels = waterpx_count(shp_geom, bbox, nc_data)
        water_px[idx] = pixel_count
        water_pcnt[idx] = pixel_percent
        tot_px[idx] = total_pixels
        water_area[idx] = (pixel_percent * area) / 100

    # Precipitation per month
    months = sorted(set(str(date)[5:7] for date in dates_pd))
    monthly_precip_values = []
    for month in months:
        nc_file = os.path.join(precip_folder, f"IMERG-Final.CLIM.2001-2022.{month}.V07B.nc4")
        monthly_precip_values.append(calc_avg_precip(shp_geom, bbox, nc_file))
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
    save_path2 = f"/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/RESULTS/sls_Daily/{station}_parallel2.csv"
    df_final.to_csv(save_path2, index=False)

    elapsed = time.time() - start_time
    print(f"Finished station {station} in {elapsed:.2f} seconds")

# =========================
# MAIN SCRIPT (SERIAL WITH FULL ARRAY CACHE)
# =========================
if __name__ == "__main__":
    masterlist = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/GRDC_Stations_UPDATED.csv'
    stations_df = pd.read_csv(masterlist)
    station_num = stations_df['grdc_no']
    print(f"Loaded {len(station_num)} stations.")

    # Build complete list of all CYGNSS files
    cutoff = pd.to_datetime('2025-07-27')
    all_ncdfs = set()
    directory_cache = {}
    for station in station_num:
        q_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/sls_GRDC_shp/GRDC_shp/{station}_Q_Day.Cmd.txt'
        _, dates = get_stationdata(q_file)
        dates_pd = pd.to_datetime(dates['YYYY-MM-DD'])
        for date in dates_pd:
            directory = ("/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_from_PODAAC/Daily/Daily_For_Trend"
                         if date < cutoff else
                         f"/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/sls_CYGNSS_DAILY_WATERMASK/{date.year}")
            if directory not in directory_cache:
                try:
                    directory_cache[directory] = set(f.name for f in os.scandir(directory) if f.is_file())
                except:
                    directory_cache[directory] = set()
            fname = f"cyg.ddmi.{date.strftime('%Y-%m-%d')}.l3.uc-berkeley-watermask-daily.a32.d33.nc"
            if fname in directory_cache[directory]:
                all_ncdfs.add(os.path.join(directory, fname))
    all_ncdfs = sorted(list(all_ncdfs))
    print(f"Found {len(all_ncdfs)} CYGNSS files.")

    # Precipitation folder
    precip_folder = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/PRECIP/RESAMPLED/'

    # =========================
    # Load CYGNSS arrays into memory
    # =========================
    nc_cache = {}
    for idx, nc_file in enumerate(all_ncdfs):
        try:
            with nc.Dataset(nc_file, 'r') as ds:
                nc_cache[nc_file] = {
                    'watermask': ds.variables['watermask'][:],
                    'lat': ds.variables['lat'][:],
                    'lon': ds.variables['lon'][:]
                }
        except:
            nc_cache[nc_file] = None
        if (idx + 1) % 200 == 0:
            print(f"Loaded {idx + 1} CYGNSS files into cache...")
    print("CYGNSS cache loaded.")

    # =========================
    # Serial processing over all stations
    # =========================
    for s in range(len(station_num)):
        process_station(s, station_num, stations_df, nc_cache, precip_folder)
