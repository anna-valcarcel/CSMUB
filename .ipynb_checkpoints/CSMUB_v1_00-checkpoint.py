# CYGNSS Streamflow Modeling for Ungauged Basins (CSMUB)

# This code combines the delineation of watersheds and the analysis of the CYGNSS watermask files to determine the surface water extent of each basin on a monthly time scale using the functions `crop_dem`, `process_basin` and `waterpx_count`. In addition, the function `read_gauge` loads the empirical streamflow data from text files downloaded from the Global Runoff Data Centre (GRDC) [link text](https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser#dataDownload/Home). Lastly, it outputs a .csv file with the station name and the results of the analysis.
#
# ---
#
# Author(s): Anna Valcarcelc
#
# Last Updated: November 19, 2024

#########################################################################
# 0.0 - IMPORT PACKAGES #
#########################################################################

# get_ipython().system('python3.6 -m pip install pysheds netCDF4 fiona geopandas xarray pyshp')
# python3.6 -m pip install pysheds netCDF4 fiona geopandas xarray pyshp

# mkdir /global/scratch/users/arvalcarcel/tmp
# export TMPDIR=/global/scratch/users/arvalcarcel/tmp # set TMPDIR

import pysheds
from pysheds.grid import Grid
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import fiona
import shapefile
import math
from scipy import stats
import os

import netCDF4 as nc
from netCDF4 import Dataset
from shapely.geometry import Point, shape, box
from shapely.vectorized import contains
from shapely.strtree import STRtree
import matplotlib.path as mpath
from matplotlib.ticker import MultipleLocator
from rasterio.coords import BoundingBox
from rasterio.mask import mask
from rasterio.plot import show
# from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#########################################################################
# 1.0 DEFINE FUNCTIONS #
#########################################################################

def crop_dem(input_tif, output_tif, center_lon, center_lat,area):

    # # Define the path to the input and output TIFF files

    if area < 10000:
      width = np.sqrt(area)*4 # width extending from center point to each edge of square
    else:
      width = np.sqrt(area)*2

    width = int(width)
    degrees = np.round(width/111.32,2)

    xmin = center_lon - degrees
    xmax = center_lon + degrees
    ymin = center_lat - degrees
    ymax = center_lat + degrees

    # Define the bounding box coordinates (left, bottom, right, top)
    bbox = (xmin, ymin, xmax, ymax)  # Replace with actual coordinates

    # Open the source TIFF file
    with rasterio.open(input_tif) as src:
        # Convert the bounding box to a GeoJSON-style geometry for rasterio.mask.mask
        bbox_geom = {
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]],
                [bbox[0], bbox[3]],
                [bbox[2], bbox[3]],
                [bbox[2], bbox[1]],
                [bbox[0], bbox[1]]
            ]]
        }

        # Crop the image using the bounding box geometry
        out_image, out_transform = mask(src, [bbox_geom], crop=True)
        out_meta = src.meta.copy()

        # Update metadata to reflect the new cropped area
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            'crs': src.crs  # Ensure the CRS remains the same
        })

        # # Save the cropped data to a new GeoTIFF file with updated profile
        # profile = src.profile
        # profile.update({
        #     'height': height,
        #     'width': width,
        #     'transform': transform,
        #     'crs': src.crs  # Ensure CRS is preserved
        # })

        with rasterio.open(output_tif, 'w', **out_meta) as dst:
            dst.write(out_image)


    # Calculate the extent of the image in geographic coordinates
    left = out_transform.c  # xmin
    right = out_transform.c + out_transform.a * out_image.shape[2]  # xmax
    top = out_transform.f  # ymax
    bottom = out_transform.f + out_transform.e * out_image.shape[1]  # ymin

    extent = (left, right, bottom, top)



def process_basin(tif_input, pour_lon, pour_lat,shp_name,number):
# ----------------------------
    # LOAD THE TIF FILE
    dset = rasterio.open(tif_input,mode='r+')
    # dset.nodata = -32767

    # OPTIONAL: change elevation values of ocean
    # rdbl = dset.read(1)
    # for i in range(0,len(rdbl)):
    #     list1 = rdbl[i]
    #     for j in range(0,len(list1)):
    #         if list1[j] == 32767:
    #             list1[j] = -1
    #     rdbl[i] = list1

    dset.close()


    # ----------------------------

    # Read elevation raster
    grid = Grid.from_raster(tif_input)
    dem = grid.read_raster(tif_input)
    # # Open and process the raster
    # with rasterio.open(tif_input, "r") as src:
    #     profile = src.profile
    #     data = src.read(1)

    #     # Replace NoData values
    #     data[data == 32767.0] = -99.0

    #     # Update the profile to reflect the new NoData value
    #     profile.update(dtype="float32", nodata=-99.0)

    # new_tif = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/STATIONS/{number}/{number}_fixed.tif'
    # # Write the updated raster
    # with rasterio.open(new_tif, "w", **profile) as dst:
    #     dst.write(data, 1)

    # # Debug to confirm changes
    # with rasterio.open(new_tif) as src:
    #     print("Updated Data type:", src.dtypes[0])
    #     print("Updated NoData value:", src.nodata)
    #     print("Updated Min/Max values:", data.min(), data.max())
    #     print("Number of NoData cells:", np.sum(data == -99.0))

    # # Step 1: Initialize the grid and read raster
    # grid = Grid.from_raster(new_tif, data_name="dem", nodata=np.float32(-99.0))

    # # Step 2: Explicitly read raster and set nodata value
    # dem = grid.read_raster(new_tif, nodata=np.float32(-99.0))

    # Condition DEM
    # ----------------------
    # Fill pits in DEM
    pit_filled_dem = grid.fill_pits(dem)

    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)

    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)
    print(inflated_dem.dtype)

    # Determine D8 flow directions from DEM
    # ----------------------
    # Specify directional mapping
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Compute flow directions
    # -------------------------------------
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

    # Calculate flow accumulation
    # --------------------------
    acc = grid.accumulation(fdir, dirmap=dirmap)
    acc[dem==0] = 0

    # Delineate a catchment
    # ---------------------
    # Specify pour point
    x = pour_lon
    y = pour_lat
    # print(x,y)

    # # Snap pour point to high accumulation cell
    # x_snap, y_snap = grid.snap_to_mask(acc > 1000000, (x, y))
    # # print(x_snap,y_snap)

    # Snap pour point to high accumulation cell
    acc_max = int(acc.max())
    power_ten = len(str(acc_max))
    snap_pt = 10**(int(power_ten) - 1)
    x_snap, y_snap = grid.snap_to_mask(acc > snap_pt, (x, y))

    # snap_pt = round(acc_max*0.9,-3)
    # x_snap, y_snap = grid.snap_to_mask(acc > snap_pt, (x, y))

    # Delineate the catchment
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap,
                          xytype='coordinate')


    # Crop and plot the catchment
    # ---------------------------
    # Clip the bounding box to the catchment
    grid.clip_to(catch)
    catch_view = grid.view(catch, dtype=np.uint8)

    # Create a vector representation of the catchment mask
    shapes = grid.polygonize(catch_view)

    # Specify schema
    schema = {
            'geometry': 'Polygon',
            'properties': {'LABEL': 'float:16'}
    }

    # Write shapefile
    shp_output = shp_name
    with fiona.open(shp_output, 'w',
                    driver='ESRI Shapefile',
                    crs=grid.crs.srs,
                    schema=schema) as c:
        i = 0
        for shape, value in shapes:
            rec = {}
            rec['geometry'] = shape
            rec['properties'] = {'LABEL' : str(value)}
            rec['id'] = str(i)
            c.write(rec)
            i += 1

    # fig, ax = plt.subplots(figsize=(8,6))
    # fig.patch.set_alpha(0)
    # plt.grid('on', zorder=0)
    # im = ax.imshow(acc, extent=catch.extent, zorder=1,
    #               cmap='ocean',
    #               norm=colors.LogNorm(1, acc.max()),
    #               interpolation='bilinear')
    # plt.colorbar(im, ax=ax, label='Upstream Cells')
    # plt.imshow(np.where(catch_view,catch_view, np.nan), extent=grid.extent,zorder=2, cmap='Greys',alpha=0.3)
    # plt.scatter(x_snap,y_snap, color='red', s=100, marker='o', zorder=3)
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.tight_layout()
    # plt.show()

    return shp_output



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
  end_date = pd.to_datetime('2024-09', format='%Y-%m')

  # Filter the DataFrame to keep rows after '2018-08'
  df_filtered = df_Q[df_Q['YYYY-MM-DD'] >= start_date]
  df_filtered = df_filtered[df_filtered['YYYY-MM-DD'] <= end_date]

  stream_gauge = df_filtered['Calculated'].values

  df_filtered['YYYY-MM-DD'] = df_filtered['YYYY-MM-DD'].dt.strftime('%Y-%m')
  dates = df_filtered['YYYY-MM-DD']
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


  markercolormap2= colors.ListedColormap(['white', 'black','blue'])

  # Step 1: Filter the GeoDataFrame where 'watermask' is equal to 2
  filtered_gdf = gdf[gdf['watermask'] == 1]

  water_pixels = len(filtered_gdf)
  total_pixels = len(gdf)

  water_percent = (water_pixels / total_pixels) * 100

  # if stream == -999 or stream < -100 :
  #     water_percent = np.nan
  #     stream = np.nan


  return water_pixels, water_percent, total_pixels



#########################################################################
# 2.0 READ MASTERFILE #
#########################################################################

masterlist = '/global/scratch/users/arvalcarcel/CSMUB/DATA/GRDC_Stations_AllMonthly.csv'

stations_df = pd.read_csv(masterlist)
station_num = stations_df['grdc_no']

monthly_path = '/global/scratch/users/arvalcarcel/CSMUB/DATA/STATIONS/'

station_files = [os.path.join(monthly_path, f"{station_no}_Q_Month.txt") for station_no in station_num]

stations_df['shapefile_code'] = np.ones(len(station_files)) * 999
stations_df['r2'] = np.ones(len(station_files)) * 999
stations_df['NaNs'] = np.ones(len(station_files)) * 999

print(f"Loaded {len(station_files)} stations.")





############################# START OF LOOP #############################


#########################################################################
# 2.1 CROP CONTINENT SCALE DEM #
#########################################################################


for s in range(260,len(stations_df)):
# for s in range(44,45):
    data = stations_df.iloc[s]
    
    number = data['grdc_no']
    region = data['wmo_reg']
    river = data['river']
    country = data['country']
    name = data['station']
    lat = data['lat']
    lon = data['long']
    area = data['area']
    altitude = data['altitude']
    
    
    # determine what continent DEM to use
    # 1 - africa, 2 - asia, 3 - SA, 4 - NA/CA/caribbean, 5 - SW pacific, 6 - europe
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

    # Define the path for the new folder
    new_folder_path = f'/global/home/users/arvalcarcel/ondemand/data/dem/{number}/'
    
    # Create the folder
    # os.makedirs(new_folder_path, exist_ok=True)
    
    output_tif = f'/global/home/users/arvalcarcel/ondemand/data/dem/{number}/{number}_dem.tif' # save the cropped tif file
    
    # crop_dem(dem, output_tif, lon, lat,area)
    
    
    #########################################################################
    # 2.2 DELINEATE BASIN #
    #########################################################################
    
    
    # LOAD THE TIF FILE
    tif_input = output_tif # open the croppe tif file and delineate basin from it
    pour_point = [lon,lat]
    
    # Define the path for the new folder
    new_folder_path = f'/global/home/users/arvalcarcel/ondemand/data/dem/{number}/'
    
    # Create the folder
    # os.makedirs(new_folder_path, exist_ok=True)
    
    # Create new shapefile
    shp_name = new_folder_path + f'{number}.shp'
    shp_output = shp_name
    # shp_output = process_basin(tif_input, lon, lat,shp_name,number)
    
    #########################################################################
    # 2.3 DETERMINE SHAPEFILE #
    #########################################################################
    
    
    # Read the shapefiles
    shapefile1 = gpd.read_file(shp_output) # delineated shapefile
    shapefile2 = gpd.read_file(f'/global/scratch/users/arvalcarcel/CSMUB/DATA/SHAPEFILES/{number}/{number}.shp') # GRDC shapefile
    
    # Step 3: Reproject to a projected CRS (e.g., UTM Zone 33N, replace EPSG:32633 as appropriate)
    shapefile1 = shapefile1.to_crs(epsg=32633)
    shapefile2 = shapefile2.to_crs(epsg=32633)
    
    # Ensure CRS consistency
    assert shapefile1.crs == shapefile2.crs, "CRS mismatch between shapefiles!"
    
    # Validate geometries
    shapefile1['geometry'] = shapefile1.geometry.buffer(0)
    shapefile2['geometry'] = shapefile2.geometry.buffer(0)
    
    # Filter geometries that potentially intersect
    possible_intersections = shapefile1[shapefile1.geometry.apply(lambda g: shapefile2.geometry.intersects(g).any())]
    
    # Perform the intersection
    intersection = gpd.overlay(possible_intersections, shapefile2, how='intersection',keep_geom_type=False)
    
    # Calculate area
    intersection_area = intersection.geometry.area.sum() / 10**6  # Convert m² to km²

    inter_pcnt = (intersection_area/area)*100
    
    # print('area: ', area)
    # print('intersection area: ', intersection_area)
    # print('percent: ', inter_pcnt)
    
    if 200 > inter_pcnt >= 50:
        shapefile = shp_output
        stations_df['shapefile_code'].iloc[s] = 1
    else:
        shapefile = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/SHAPEFILES/{number}/{number}.shp'
        stations_df['shapefile_code'].iloc[s] = 2
    
    # print(shp_log[s])
    
    #########################################################################
    # 2.3 IMPORT STATION DATA #
    #########################################################################
    
    
    q_file = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/STATIONS/{number}_Q_Month.txt'
    stream_gauge, dates = get_stationdata(q_file)
    
    
    
    #########################################################################
    # 2.4 CALCULATE WATER PIXEL PERCENT #
    #########################################################################
    
    
    directory = "/global/scratch/users/arvalcarcel/CSMUB/DATA/CYGNSS/"
    file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    
    # Create empty numpy arrays with the correct size
    ncdf_list = []
    water_px = np.zeros(len(dates))
    water_pcnt = np.zeros(len(dates)) # CHANGE TO DATES
    tot_px = np.zeros(len(dates))
    
    
    for i in range(0,len(dates)):
        date = dates['YYYY-MM-DD'][i]
        ncdf_name = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/CYGNSS/cyg.ddmi.{date}.l3.uc-berkeley-watermask-monthly.a31.d32.nc'
        ncdf_list.append(ncdf_name)
    
    
    for f, filename in enumerate(ncdf_list):
        if os.path.isfile(filename):
            pixel_count, pixel_percent, total_pixels = waterpx_count(shp_output, filename)
            water_px[f] = pixel_count
            water_pcnt[f] = pixel_percent
            tot_px[f] = total_pixels
    
    
    
    
    #########################################################################
    # 2.5 EXPORT AND SAVE RESULTS #
    #########################################################################
    

    # Identify indices where stream_gauge has the value -999
    invalid_indices = [i for i, value in enumerate(stream_gauge) if value == -999]
    stations_df['NaNs'].iloc[s] = len(invalid_indices)
    # print('invalid indices : ',invalid_indices, ' of ', len(stream_gauge))
    # print('total nans: ', nan_no[s])

    # Remove items at those indices from all three lists
    stream_gauge = [value for i, value in enumerate(stream_gauge) if i not in invalid_indices]
    dates = [value for i, value in enumerate(dates['YYYY-MM-DD']) if i not in invalid_indices]
    water_pcnt = [value for i, value in enumerate(water_pcnt) if i not in invalid_indices]
    water_area = [x * area for x in water_pcnt]

    df_final = pd.DataFrame({'Date': dates, 'Q': stream_gauge, 'SWE': water_pcnt, 'SWE_scaled': water_area})
    
    df_final.to_csv(f"/global/home/users/arvalcarcel/ondemand/results/csv/{number}.csv")
    
    def check(list):
        return all(i == list[0] for i in list)

    if check(stream_gauge) == False:
        slope, intercept, r, p, se = stats.linregress(stream_gauge, water_pcnt)
        r2 = r**2
        stations_df['r2'].iloc[s] = r2
    
    
        # fig, ax1 = plt.subplots(figsize=(10,6))
        # ax1.plot(df_final['Date'],df_final['Q'], color='b', label='Streamflow')
        # ax1.set_xlabel('Date')
        # ax1.set_ylabel('Streamflow [m^3/s]', color='b')
        # ax1.set_title(f'{river} ({number}), {country}: Streamflow vs Watermask Timeseries')
        # ax1.set_xticks(df_final['Date'])
        # ax1.set_xticklabels(df_final['Date'])
        # plt.xticks(rotation=60)
        
        # ax2 = ax1.twinx()
        # ax2.plot(df_final['Date'],df_final['SWE'], color='r', label='Watermask')
        # ax2.set_ylabel('Watermask Px Percent', color='r')
        # # major ticks every 10
        # ax1.xaxis.set_major_locator(MultipleLocator(3))
        
        # # minor ticks at every point
        # ax1.xaxis.set_minor_locator(MultipleLocator(1))
        # # Adding the legend inside the plot
        # lines, labels = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines + lines2, labels + labels2, loc='best', ncol=1)
        # plt.savefig(f"/global/home/users/arvalcarcel/ondemand/results/timeseries/{number}.png",bbox_inches = "tight")
        # plt.show()

    else:
        stations_df['r2'].iloc[s] = np.nan
        stations_df['NaNs'].iloc[s] = np.nan
    

    # stations_df['shapefile_code'][s] = shp_log[s]
    # stations_df['r2'][s] = r_squared[s]
    # stations_df['NaNs'][s] = nan_no[s]

    stations_df.to_csv(f"/global/home/users/arvalcarcel/ondemand/results/ALL_STATIONS_FINAL.csv")
    print(s)
    ############################## END OF LOOP ##############################



print('finished: ', s)

#########################################################################
# END OF CODE #
#########################################################################
