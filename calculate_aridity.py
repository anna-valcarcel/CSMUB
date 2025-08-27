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
    
    tif = '/global/scratch/users/arvalcarcel/CSMUB/DATA/aridity_index.tif'


    # Read the shapefiles
    shapefile1 = f'/global/home/users/arvalcarcel/ondemand/data/dem/{number}/{number}.shp' # delineated shapefile
    shapefile2 = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/SHAPEFILES/{number}/{number}.shp' # GRDC shapefile

    if shp_log == 1:
        shapefile = shapefile1

    elif shp_log == 2:
        shapefile = shapefile2

    def calculate_average_aridity(tif_path, shapefile_path):
        # Load shapefile
        gdf = gpd.read_file(shapefile_path)
        print("CRS of shapefile:", gdf.crs)
        shapes = gdf.geometry.values
    
        # Load DEM and mask it with the shapefile
        with rasterio.open(tif_path) as src:
            aridity_data, transform = mask(src, shapes, crop=True)
            aridity_data = aridity_data[0].astype('float32')  # Convert to float
    
            if src.nodata is not None:
                aridity_data[aridity_data == src.nodata] = np.nan
    
        # Average slope
        aridity_index = np.nanmean(aridity_data)
        return aridity_index

    # Example usage
    aridity_index = calculate_average_aridity(tif, shapefile)
    # print(f"Average aridity index within the area: {aridity_index:.2f}")

    # Merge total_df with landcover_df on GRDC_No
    # Store the average slope in the DataFrame
    full_df.at[i, 'avg_aridity'] = aridity_index

print(full_df.head())

full_df.to_csv(masterlist, index=False)
print(pd.read_csv(masterlist))