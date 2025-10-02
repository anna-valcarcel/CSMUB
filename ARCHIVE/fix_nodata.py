import rasterio
import numpy as np
from pysheds.grid import Grid

# Input file
number = 1159100
tif_input = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/STATIONS/{number}/{number}_dem.tif'  # Replace with your input file path

# Open and process the raster
with rasterio.open(tif_input, "r") as src:
    profile = src.profile
    data = src.read(1)

    # Replace NoData values
    data[data == 32767.0] = -99.0

    # Update the profile to reflect the new NoData value
    profile.update(dtype="float32", nodata=-99.0)

new_tif = f'/global/scratch/users/arvalcarcel/CSMUB/DATA/DEM/STATIONS/{number}/{number}_fixed.tif'
# Write the updated raster
with rasterio.open(new_tif, "w", **profile) as dst:
    dst.write(data, 1)

# Debug to confirm changes
with rasterio.open(new_tif) as src:
    print("Updated Data type:", src.dtypes[0])
    print("Updated NoData value:", src.nodata)
    print("Updated Min/Max values:", data.min(), data.max())
    print("Number of NoData cells:", np.sum(data == -99.0))

# Define the nodata value explicitly
nodata_value = np.float32(-99.0)

# Initialize the grid
grid = Grid.from_raster(new_tif, data_name="dem", nodata=nodata_value)
# Access the data
try:
    dem = grid.read_raster(new_tif, data_name="dem")
    print("DEM loaded successfully.")
    print("  Data type:", dem.dtype)
    print("  Shape:", dem.shape)
    print("  Min value:", dem.min())
    print("  Max value:", dem.max())
    print("  Number of NoData cells:", np.sum(dem == -99.0))
except Exception as e:
    print("Error reading raster:", e)

# Debug the internal metadata
print("Grid metadata:")
for key, value in grid.__dict__.items():
    print(f"  {key}: {value}")

# Access the data
try:
    dem = grid.read_raster(new_tif, data_name="dem")
    print("DEM loaded successfully.")
    print("  Data type:", dem.dtype)
    print("  Shape:", dem.shape)
    print("  Min value:", dem.min())
    print("  Max value:", dem.max())
    print("  Number of NoData cells:", np.sum(dem == -99.0))
except Exception as e:
    print("Error reading raster:", e)

print("Any NaN values:", np.isnan(dem).any())
print("Any infinity values:", np.isinf(dem).any())

print("Number of masked (NoData) cells:", np.sum(grid.mask))

try:
    fdir = grid.flowdir(dem, nodata_cells=(dem == -99.0))
    print("Flow direction computed successfully.")
except Exception as e:
    print("Error during flow direction computation:", e)

try:
    fdir = grid.flowdir(dem, nodata_cells=(dem == -99.0), verbose=True)
    print("Flow direction computed successfully.")
except Exception as e:
    print("Error during flow direction computation:", e)

masked_dem = np.ma.masked_equal(dem, -99.0)
try:
    fdir = grid.flowdir(masked_dem.data, nodata_cells=masked_dem.mask)
    print("Flow direction computed successfully.")
except Exception as e:
    print("Error during flow direction computation with masked array:", e)
