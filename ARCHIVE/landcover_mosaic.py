import rasterio
import os
from rasterio.merge import merge
from rasterio.plot import show
import numpy as np
# import glob

# assign directory
directory = "/global/scratch/users/arvalcarcel/CSMUB/DATA/LANDCOVER/RESAMPLED/"
src_files_to_mosaic = []
q = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
dem_fps = [os.path.join(directory, qx) for qx in q]
print(dem_fps)

for fp in dem_fps:
    src = rasterio.open(fp)
    src_files_to_mosaic.append(src)

# print(src_files_to_mosaic)

mosaic, out_trans = merge(src_files_to_mosaic)
print('mosaic completed')

out_fp = "/global/scratch/users/arvalcarcel/CSMUB/DATA/LANDCOVER/RESAMPLED/landcover_mosaic.tif"
out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans, "crs": src.crs})
with rasterio.open(out_fp, "w", **out_meta) as dest:
    dest.write(mosaic)
# show(mosaic, cmap='terrain')