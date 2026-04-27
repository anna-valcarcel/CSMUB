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
  # Remove spaces in column names entirely
  df_Q = pd.read_csv(station_file, delimiter=';',encoding='utf-8',skiprows=38)
  # print(df_Q.head)

  df_Q.columns = df_Q.columns.str.replace(' ', '')
  df_Q['YYYY-MM-DD'] = df_Q['YYYY-MM-DD'].str.slice(0, 7)

  # Convert 'YYYY-MM-DD' column to datetime format
  df_Q['YYYY-MM-DD'] = pd.to_datetime(df_Q['YYYY-MM-DD'], format='%Y-%m')

  # Convert cutoff_date to datetime format
  start_date = pd.to_datetime('2018-08', format='%Y-%m')
  end_date = pd.to_datetime('2025-11', format='%Y-%m')

  # Filter the DataFrame to keep rows after '2018-08'
  df_filtered = df_Q[df_Q['YYYY-MM-DD'] >= start_date]
  df_filtered = df_filtered[df_filtered['YYYY-MM-DD'] <= end_date]

  # print(df_filtered.head)

  stream_gauge = df_filtered['Calculated'].values
  # print(len(stream_gauge))

  df_filtered['YYYY-MM-DD'] = df_filtered['YYYY-MM-DD'].dt.strftime('%Y-%m')
  dates = df_filtered['YYYY-MM-DD']
  dates = dates.reset_index()
  # print(dates)

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

# def calculate_average_slope(dem_path, shapefile_path):
    # Load shapefile
    gdf = gpd.read_file(shapefile_path)
    # print("CRS of shapefile:", gdf.crs)
    shapes = gdf.geometry.values

    # Load DEM and mask it with the shapefile
    with rasterio.open(dem_path) as src:
        dem_data, transform = mask(src, shapes, crop=True)
        dem_data = dem_data[0].astype('float32')  # Convert to float

        if src.nodata is not None:
            dem_data[dem_data == src.nodata] = np.nan

        # Resolution in degrees
        xres_deg, yres_deg = src.res

        # Estimate latitude at center of masked area
        bounds = src.bounds
        center_lat = (bounds.top + bounds.bottom) / 2

        # Convert resolution from degrees to meters
        meters_per_degree_lat = 111320  # approx constant
        meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))

        xres_m = xres_deg * meters_per_degree_lon
        yres_m = yres_deg * meters_per_degree_lat

    # Compute slope using Sobel filters (rise/run)
    dzdx = sobel(dem_data, axis=1, mode='nearest') / (8 * xres_m)
    dzdy = sobel(dem_data, axis=0, mode='nearest') / (8 * yres_m)

    # Slope in degrees
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)

    # Average slope
    avg_slope = np.nanmean(slope_deg)
    max_slope = np.nanmax(slope_deg)
    
    return avg_slope, max_slope

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

# def landcover_percentage(shp_input):

    landcover_map = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/landcover_mosaic.tif'
    gdf = gpd.read_file(shp_input)

    # File paths
    geotiff_path = landcover_map
    
    # Open the GeoTIFF
    with rasterio.open(geotiff_path) as src:
        # Reproject shapefile to match raster CRS
        gdf = gdf.to_crs(src.crs)
    
        # Convert shapefile geometry to GeoJSON-like format
        shapes = [mapping(geom) for geom in gdf.geometry]
    
        # Mask raster with shapefile
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, filled=True)
        out_meta = src.meta.copy()
    
        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    # Masked array: out_image is (1, height, width) with nodata masked
    masked_array = np.ma.masked_array(out_image, out_image == src.nodata)
    
    # Alternatively, use the mask from the output:
    valid_pixels = masked_array.compressed()

    # Count occurrences of each band value (1-20)
    unique, counts = np.unique(valid_pixels, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    # Compute percentage of total valid pixels
    total_pixels = valid_pixels.size
    percentages = {i: (pixel_counts[i] / total_pixels * 100) if i in pixel_counts else 0 for i in range(1, 21)}

    # Convert percentages to a DataFrame row
    landcover_row = pd.DataFrame([percentages])
    
    landcover_row.columns = [f"LC_{col}" for col in landcover_row.columns]

    landcover_row["number"] = number  # Add 'number' as a separate column
    landcover_row.set_index("number", inplace=True)  # Set 'number' as the index

    return landcover_row

# def soiltype_percentage(shp_input):

    nc_file = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/GLDASp5_soiltexture_025d.nc4'
    
    # Load the NetCDF data
    with nc.Dataset(nc_file) as dataset:
        soil = dataset.variables['GLDAS_soiltex'][:]  # Soil data (masked)
        latitude = dataset.variables['lat'][:]  # Latitude values
        longitude = dataset.variables['lon'][:]  # Longitude values
        soil_masked = dataset.variables['GLDAS_soiltex'][:]
        soil_masked = np.ma.masked_equal(soil_masked, -9999)  # Mask invalid values (assuming -9999 is used for missing data)

    # Remove the singleton time dimension (if present)
    soil = soil_masked.squeeze()  # Removes the first dimension if it's size 1

    # Check the shape after squeezing
    # print(f"Shape after squeezing: {soil.shape}")

    # Convert the masked array to a regular numpy array with NaN for masked values
    soil = soil.filled(np.nan)

    # # Check if the entire soil data is NaN
    # if np.all(np.isnan(soil)):
    #     print("Warning: All soil data is NaN after masking. Please check the data source or bounds.")
    # else:
    #     print(f"Data contains valid values. Shape: {soil.shape}")

    # Create an xarray Dataset for easy interpolation
    data_set = xr.Dataset({"soil": (["lat", "lon"], soil)},
                    coords={"lat": latitude, "lon": longitude})

    # Load the shapefile and reproject it to EPSG:4326 if needed
    shp = gpd.read_file(shp_input).to_crs('EPSG:4326')
    minlon, minlat, maxlon, maxlat = shp.geometry.total_bounds

    # Crop the data to the bounds of the shapefile
    lat_mask = (latitude >= minlat) & (latitude <= maxlat)
    lon_mask = (longitude >= minlon) & (longitude <= maxlon)
    cropped_soil = soil[lat_mask, :][:, lon_mask]

    # Check if the cropped data is empty or full of NaNs
    if np.all(np.isnan(cropped_soil)):
        print("Warning: Cropped soil data is empty or full of NaNs. Please check your shapefile bounds or data.")
    else:
        print(f"Cropped data has valid values. Shape: {cropped_soil.shape}")

    # Create new latitude and longitude arrays for the desired resolution
    dx_new = 0.1  # Desired resolution
    newlon = np.arange(minlon, maxlon, dx_new)
    newlat = np.arange(minlat, maxlat, dx_new)

    # Interpolate the data to the new grid
    data_set_interp = data_set.interp(lat=newlat, lon=newlon)

    # Extract the 'soil' data as a NumPy array from the xarray Dataset
    soil = data_set_interp['soil'].values

    # Get interpolated lat/lon arrays
    lat_vals = data_set_interp['lat'].values
    lon_vals = data_set_interp['lon'].values

    # Create meshgrid of coordinates
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    # Flatten the grids for masking
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    soil_flat = soil.flatten()

    # Get the first (or union) geometry from the shapefile
    geom = shp.unary_union  # For multi-polygons

    # Mask: only points within the shapefile geometry
    mask = contains(geom, lon_flat, lat_flat)

    # Apply mask to soil data
    soil_in_geom = soil_flat[mask]

    # Filter out NaNs and values outside 1–16
    soil_in_geom = soil_in_geom[~np.isnan(soil_in_geom)]
    soil_in_geom = soil_in_geom[(soil_in_geom >= 1) & (soil_in_geom <= 16)]

    # Ensure integer values for counting
    soil_in_geom = soil_in_geom.astype(int)

    # Count valid values
    unique, counts = np.unique(soil_in_geom, return_counts=True)
    soil_counts = dict(zip(unique, counts))

    # Total number of valid soil class pixels in the shape
    total_points = np.sum(counts)

    # Calculate percentage for each class (1–16)
    percentages = {i: (soil_counts.get(i, 0) / total_points * 100) for i in range(1, 17)}

    # Debug: check sum of percentages
    # print(f"Sum of percentages: {sum(percentages.values()):.2f}%")

    # Display the results as a pandas Series for better readability
    percentages_df = pd.Series(percentages).sort_index()

    percentages_df = percentages_df.T

    most_common = percentages_df.idxmax()

    # print(most_common)
    # Convert percentages to a DataFrame row
    soil_row = pd.DataFrame([percentages])
    soil_row.columns = [f"ST_{col}" for col in soil_row.columns]
    soil_row["number"] = number  # Add 'number' as a separate column
    soil_row.set_index("number", inplace=True)  # Set 'number' as the index

    return soil_row


# DEFINE INPUTS FOR FUNCTION
masterlist = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/GRDC_Stations_UPDATED.csv'
# final_path = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/RESULTS/ALL_STATIONS__DAILY_FINAL_2_00.csv'

stations_df = pd.read_csv(masterlist)
# print(stations_df)
station_num = stations_df['grdc_no']
# station_num

final_result = pd.DataFrame()
print(f"Loaded {len(station_num)} stations.") # as of Nov 2025: should be 578 stations total when loading all

import warnings
warnings.filterwarnings("ignore")

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
    
    q_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/STATIONS/{number}_Q_Month.txt' # GRDC monthly streamgauge readings as .txt files
    shp_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/DEM/STATIONS/{station}/{station}.shp' # all station delineated shapefiles saved in subfolders named by station number
    dem_file = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/DATA/DEM/STATIONS/{station}_dem.tif' # cropped .tif files for 15 arc-second DEM saved in subfolders named by station number
    
    results_df = pd.DataFrame(stations_df.iloc[[s]])
    
    stream_gauge, dates = get_stationdata(q_file)
    
    # assign directory
    directory = "/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/MONTHLY_WATERMASK/"
    file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Calculate water pixel count
    ncdf_list = []
    water_px = np.zeros(len(dates))
    water_pcnt = np.zeros(len(dates)) 
    water_area = np.zeros(len(dates))
    tot_px = np.zeros(len(dates))
    
    for i in range(0,len(dates)):
        date = dates['YYYY-MM-DD'][i]
        ncdf_name = f'/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/MONTHLY_WATERMASK/cyg.ddmi.{date}.l3.uc-berkeley-watermask-monthly.a31.d32.nc'
        ncdf_list.append(ncdf_name)
    
    for f, filename in enumerate(ncdf_list):
      if os.path.isfile(filename):
        pixel_count, pixel_percent, total_pixels = waterpx_count(shp_file, filename)
        water_px[f] = pixel_count
        water_pcnt[f] = pixel_percent
        tot_px[f] = total_pixels
        water_area[f] = ((pixel_percent*area)/100)

    # print("water px count calculated")
    
    # Get Other Parameters: Precipitation Climatology
    precip = np.zeros(len(dates))
    date_inputs = dates['YYYY-MM-DD']
    for index, date in enumerate(date_inputs):
        date_input = date[-2:]
        # Compute average precipitation for this date and shapefile
        precip[index] = calc_avg_precip(shp_file, date_input)

    # Save Individual Results   
    df_final = pd.DataFrame({'Date': dates['YYYY-MM-DD'], 'Q': stream_gauge,'SWE': water_pcnt, 'SWE_scaled': water_area, 'P': precip})
    df_final.to_csv(f"/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/RESULTS/CSV/DAILY/{number}.csv")
    # print("Individual results:")
    # print(df_final)
    
    # # Calculate Statistics
    # try:
    #     slope, intercept, r, p, se = stats.linregress(stream_gauge, water_pcnt)
    #     r2 = r**2
    # except ValueError:
    #     r2 = np.nan
    # results_df['r2'] = r2

    # # Get Other Parameters: Slope, Landcover, Soil Type
    # average_slope, maximum_slope = calculate_average_slope(dem_file, shp_file)
    # results_df['avg_slope'] = average_slope
    # results_df['max_slope'] = maximum_slope
    # landcover_classes = landcover_percentage(shp_file).reset_index(drop=True)
    # # print(landcover_classes)
    # soil_types = soiltype_percentage(shp_file).reset_index(drop=True)
    # # print(soil_types)
    # results_df = results_df.reset_index(drop=True)

    # # Concatenate horizontally (side by side)
    # combined_row = pd.concat([results_df, landcover_classes, soil_types], axis=1)
    
    # # Append that single row to the final results
    # final_result = pd.concat([final_result, combined_row], ignore_index=True)
    print("Finished running station ", number)
    # final_result.to_csv(final_path, index=False)
    # print(final_result.head())

# final_path_comp = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CSMUB/RESULTS/ALL_STATIONS_FINAL_2_00_COMPLETE.csv'
# final_result.to_csv(final_path_comp, index=False)
