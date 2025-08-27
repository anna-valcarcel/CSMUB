import matplotlib.pyplot as plt
import pandas as pd
# import netcdf4
import geopandas as gpd
import rasterio
import netCDF4 as nc
import numpy as np
from shapely.geometry import Point
from shapely.vectorized import contains
import xarray as xr

# Open the NetCDF file
nc_file = '/global/scratch/users/arvalcarcel/CSMUB/DATA/GLDASp5_soiltexture_025d.nc4'

# DEFINE INPUTS FOR FUNCTION
masterlist = '/global/scratch/users/arvalcarcel/CSMUB/RESULTS/ALL_STATIONS_FINAL_REVISED.csv'

full_df = pd.read_csv(masterlist)
station_num = full_df['grdc_no']

df_soil = pd.DataFrame()

for i in range(0,len(station_num)):
# i = 0
    data = full_df.iloc[i]
    number = data['grdc_no']
    name = data['station']
    lat = data['lat']
    lon = data['long']

    shp_log = data['shapefile_code']
    # print(shp_log)
    # output_tif = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/STATIONS/{number}_dem.tif'

    # Read the shapefiles
    shapefile1 = f'/global/home/users/arvalcarcel/ondemand/data/dem/{number}/{number}.shp' # delineated shapefile
    shapefile2 = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/SHAPEFILES/{number}/{number}.shp' # GRDC shapefile

    if shp_log == 1:
        shapefile = shapefile1

    elif shp_log == 2:
        shapefile = shapefile2

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
    print(f"Shape after squeezing: {soil.shape}")

    # Convert the masked array to a regular numpy array with NaN for masked values
    soil = soil.filled(np.nan)

    # Check if the entire soil data is NaN
    if np.all(np.isnan(soil)):
        print("Warning: All soil data is NaN after masking. Please check the data source or bounds.")
    else:
        print(f"Data contains valid values. Shape: {soil.shape}")

    # Create an xarray Dataset for easy interpolation
    # The shape should now be (600, 1440)
    data_set = xr.Dataset({"soil": (["lat", "lon"], soil)},
                    coords={"lat": latitude, "lon": longitude})

    # Load the shapefile and reproject it to EPSG:4326 if needed
    shp = gpd.read_file(shapefile).to_crs('EPSG:4326')
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

    print(most_common)
    # Convert percentages to a DataFrame row
    new_row = pd.DataFrame([percentages])
    new_row["number"] = number  # Add 'number' as a separate column
    new_row.set_index("number", inplace=True)  # Set 'number' as the index

    # Append to df_landcover
    df_soil = pd.concat([df_soil, new_row])

results_csv = f"/global/scratch/users/arvalcarcel/CSMUB/RESULTS/CSV/ALL_STATIONS_SOILTYPE.csv"
df_soil.to_csv(results_csv)
print(pd.read_csv(results_csv))