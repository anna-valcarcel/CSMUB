import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
import os

def resample_tif(input_tif, output_tif, new_resolution=0.01, compress='LZW'):
    """
    Resample a GeoTIFF file to 0.01-degree resolution (EPSG:4326).
    
    Parameters:
        input_tif (str): Path to the input GeoTIFF.
        output_tif (str): Path to save the output GeoTIFF.
        new_resolution (float): Target resolution in degrees (default 0.01).
        compress (str): Compression type for output (default 'LZW').
    """
    with rasterio.open(input_tif) as src:
        # Extract bounding box
        left, bottom, right, top = src.bounds
        
        # Calculate new transform and shape
        transform, width, height = calculate_default_transform(
            src_crs=src.crs,
            dst_crs=src.crs,
            width=src.width,
            height=src.height,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            resolution=new_resolution
        )

        # Define output profile
        profile = src.profile.copy()
        profile.update({
            'transform': transform,
            'width': width,
            'height': height,
            'compress': compress,
            'dtype': src.dtypes[0]  # Maintain same data type
        })

        # Create output dataset
        with rasterio.open(output_tif, 'w', **profile) as dst:
            for i in range(1, src.count + 1):  # Loop over all bands
                data = src.read(i)
                resampled_data = np.empty((height, width), dtype=profile['dtype'])
                
                # Resample the band
                reproject(
                    source=data,
                    destination=resampled_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.average  # Use average for downsampling
                )
                
                dst.write(resampled_data, i)

# Example usage
directory = "/global/scratch/users/arvalcarcel/CSMUB/DATA/LANDCOVER/"
q = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
input_tifs = [os.path.join(directory, qx) for qx in q]
# print(input_tifs)
directory_out =  "/global/scratch/users/arvalcarcel/CSMUB/DATA/LANDCOVER/RESAMPLED/"
output_tifs = [os.path.join(directory_out, qx) for qx in q]

for i in range(0,len(q)):
    resample_tif(input_tifs[i], output_tifs[i])
    # show(mosaic, cmap='terrain')
