# IMPORT PACKAGES
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import shapefile
import os
import time
import netCDF4 as nc
from affine import Affine  
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box

def netcdf_totif(netcdf, shp_input, output_tif):

    # Load NetCDF file and extract variables
    with nc.Dataset(netcdf) as dataset:
        watermask = dataset.variables['watermask'][:]
        latitude = dataset.variables['lat'][:]
        longitude = dataset.variables['lon'][:]
    
    # Load and reproject the shapefile
    shp = gpd.read_file(shp_input).to_crs('EPSG:4326')
    
    # Clip NetCDF data to shapefile bounding box
    minlon, minlat, maxlon, maxlat = shp.geometry.total_bounds
    lat_mask = (latitude >= minlat) & (latitude <= maxlat)
    lon_mask = (longitude >= minlon) & (longitude <= maxlon)
    
    watermask = watermask[lat_mask, :][:, lon_mask]
    lat_filtered = latitude[lat_mask]
    lon_filtered = longitude[lon_mask]
    
    # Create a mask based on shapefile geometry
    transform = Affine(
        (lon_filtered[-1] - lon_filtered[0]) / len(lon_filtered), 0, lon_filtered[0],
        0, (lat_filtered[-1] - lat_filtered[0]) / len(lat_filtered), lat_filtered[0]
    )
    
    shapes = [geom for geom in shp.geometry]
    mask = geometry_mask(shapes, transform=transform, invert=True, out_shape=watermask.shape)
    
    # Apply mask to the watermask
    filtered_watermask = np.where(mask, watermask, np.nan)

    # Export to GeoTIFF
    with rasterio.open(
        output_tif,
        'w',
        driver='GTiff',
        height=filtered_watermask.shape[0],
        width=filtered_watermask.shape[1],
        count=1,
        dtype=filtered_watermask.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(filtered_watermask, 1)


# DEFINE INPUTS FOR FUNCTION
result_list = '/global/home/users/arvalcarcel/ondemand/results/ALL_STATIONS_FINAL.csv'
stations_df = pd.read_csv(result_list)

for s in range(260,len(stations_df)):
# for s in range(0,1):
    stations_df = pd.read_csv(result_list)
    data = stations_df.iloc[s]
    # print(data)
    number = data['grdc_no']
    shp_code = data['shapefile_code']
    # print(shp_code)

    # CHOOSE SHAPEFILE BASED ON CODE NUMBER
    # Define the path for the new folder
    new_folder_path = f'/global/home/users/arvalcarcel/ondemand/data/dem/{number}/'

    # Define shapefile name from outputs
    shp_name = new_folder_path + f'{number}.shp'

    if shp_code == 1:
        shapefile = shp_name # CSMUB generated shp
    elif shp_code == 2:
        shapefile = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/SHAPEFILES/{number}/{number}.shp' # GRDC shp

    # print(shapefile)
    csv_name = f'/global/home/users/arvalcarcel/ondemand/results/csv/{number}.csv'
    station_result = pd.read_csv(csv_name)
    dates = station_result['Date']

    ncdf_list = []

    for i in range(0,len(dates)):
        date = dates[i]
        ncdf_name = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/CYGNSS/cyg.ddmi.{date}.l3.uc-berkeley-watermask-monthly.a31.d32.nc'
        ncdf_list.append(ncdf_name)


    # LOOP THROUGH ALL MONTHS FOR EACH BASIN
    for f, filename in enumerate(ncdf_list):
        if os.path.isfile(filename):
            new_folder_path = f'/global/scratch/users/arvalcarcel/CSMUB/RESULTS/MONTHLY_TIF/{number}/'
            os.makedirs(new_folder_path, exist_ok=True)
            
            # Create new geotiff
            tif_out = new_folder_path + f'{number}_watermask_' + dates[f] + '.tif'
            netcdf_totif(filename,shapefile,tif_out)

    # time.sleep(120)


print('finished: ', s)
