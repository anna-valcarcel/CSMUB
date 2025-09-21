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

# DEFINE INPUTS FOR FUNCTION
masterlist = '/global/scratch/users/arvalcarcel/CSMUB/RESULTS/ALL_STATIONS_FINAL_REVISED.csv'

full_df = pd.read_csv(masterlist)
station_num = full_df['grdc_no']
df_landcover = pd.DataFrame()

print(f"Loaded {len(station_num)} stations.")

for i in range(0,len(station_num)):
# i = 0:len(station_num):
    data = full_df.iloc[i]
    number = data['grdc_no']
    shp_log = data['shapefile_code']
    # print(shp_log)

    # Read the shapefiles
    shapefile1 = f'/global/home/users/arvalcarcel/ondemand/data/dem/{number}/{number}.shp' # delineated shapefile
    shapefile2 = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/SHAPEFILES/{number}/{number}.shp' # GRDC shapefile

    if shp_log == 1:
        shapefile = shapefile1

    elif shp_log == 2:
        shapefile = shapefile2

    landcover_map = '/global/scratch/users/arvalcarcel/CSMUB/DATA/landcover_mosaic.tif'
    gdf = gpd.read_file(shapefile)

    # File paths
    geotiff_path = landcover_map
    
    # Load shapefile
    gdf = gpd.read_file(shapefile)
    
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
    new_row = pd.DataFrame([percentages])
    new_row["number"] = number  # Add 'number' as a separate column
    new_row.set_index("number", inplace=True)  # Set 'number' as the index

    # Append to df_landcover
    df_landcover = pd.concat([df_landcover, new_row])

results_csv = f"/global/scratch/users/arvalcarcel/CSMUB/RESULTS/CSV/ALL_STATIONS_LANDCOVER.csv"
df_landcover.to_csv(results_csv)
print(pd.read_csv(results_csv))