import xarray as xr
import numpy as np
import os

def resample_ncf(input_ncf, output_ncf, new_resolution=0.1):
    ds = xr.open_dataset(input_ncf)
    
    # Identify your data variable (update this as needed)
    data_var = list(ds.data_vars)[0]

    # Get old lat/lon values
    old_lat = ds['lat']
    old_lon = ds['lon']

    # Build new lat/lon arrays with desired resolution
    new_lat = np.arange(float(old_lat.min()), float(old_lat.max()), new_resolution)
    new_lon = np.arange(float(old_lon.min()), float(old_lon.max()), new_resolution)

    # Interpolate to new grid
    ds_resampled = ds.interp(lat=new_lat, lon=new_lon, method='linear')

    # Save to NetCDF
    ds_resampled.to_netcdf(output_ncf)

    ds.close()
    ds_resampled.close()

directory = "/global/scratch/users/arvalcarcel/CSMUB/DATA/PRECIP/"
q = [f for f in os.listdir(directory) if f.endswith(".nc4")]
input_ncf = [os.path.join(directory, qx) for qx in q]
print(input_ncf)
directory_out = "/global/scratch/users/arvalcarcel/CSMUB/DATA/PRECIP/RESAMPLED/"
output_ncf = [os.path.join(directory_out, qx) for qx in q]
print(output_ncf)
for i in range(len(q)):
    resample_ncf(input_ncf[i], output_ncf[i], new_resolution=0.1)
