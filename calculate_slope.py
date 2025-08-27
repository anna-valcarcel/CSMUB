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

import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from scipy.ndimage import sobel

# DEFINE INPUTS FOR FUNCTION
masterlist = '/global/scratch/users/arvalcarcel/CSMUB/RESULTS/ALL_STATIONS_FINAL_REVISED.csv'

full_df = pd.read_csv(masterlist)
station_num = full_df['grdc_no']
full_df['average_slope'] = np.nan

print(f"Loaded {len(station_num)} stations.")

for i in range(0,len(station_num)):
    data = full_df.iloc[i]
    number = data['grdc_no']
    region = data['wmo_reg']
    river = data['river']
    name = data['station']
    lat = data['lat']
    lon = data['long']
    area = data['area']
    altitude = data['altitude']
    shp_log = data['shapefile_code']

    if region == 1:
        dem = '/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/hyd_af_dem_30s.tif'
    elif region == 2:
        dem = '/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/hyd_as_dem_30s.tif'
    elif region == 3:
        dem = '/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/hyd_sa_dem_30s.tif'
    elif region == 4:
        dem = '/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/hyd_na_dem_30s.tif'
    elif region == 5:
        dem = '/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/hyd_au_dem_30s.tif'
    elif region == 6:
        dem = '/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/hyd_eu_dem_30s.tif'
    # output_tif = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/STATIONS/{number}_dem.tif'


    # Read the shapefiles
    shapefile1 = f'/global/home/users/arvalcarcel/ondemand/data/dem/{number}/{number}.shp' # delineated shapefile
    shapefile2 = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/SHAPEFILES/{number}/{number}.shp' # GRDC shapefile

    if shp_log == 1:
        shapefile = shapefile1

    elif shp_log == 2:
        shapefile = shapefile2

    def calculate_average_slope(dem_path, shapefile_path):
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

    # Example usage
    average_slope, maximum_slope = calculate_average_slope(dem, shapefile)
    # print(f"Average slope within the area: {average_slope:.2f} degrees")

    # Merge total_df with landcover_df on GRDC_No
    # Store the average slope in the DataFrame
    full_df.at[i, 'avg_slope'] = average_slope
    full_df.at[i, 'max_slope'] = maximum_slope

print(full_df.head())

full_df.to_csv(masterlist, index=False)
print(pd.read_csv(masterlist))