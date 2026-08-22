import xarray as xr
import numpy as np
from scipy import stats

# Setup paths
input_file = "sample_output/tillage_revised.nc4"
output_file = "sample_output/tillage_single_layer_old.nc4"

def _calc_mode(arr):
    """
    Helper function to calculate mode along the last axis.
    We use axis=-1 because apply_ufunc moves the 'core dim' (crop) to the last axis.
    """
    # nan_policy='omit' ignores NaNs so they don't become the mode
    mode_result = stats.mode(arr, axis=-1, nan_policy='omit', keepdims=False)
    
    # Check if the result is empty or all-NaN (stats.mode behavior varies by version)
    # If the mode is NaN or empty, we default to a fill value (e.g., -9999 or NaN)
    result = mode_result.mode
    
    # If result has shape (..., 1), squeeze it slightly or just return
    return result

def collapse_crops_to_map():
    print(f"Opening {input_file}...")
    
    # 1. Load the dataset
    # "chunks='auto'" automatically aligns with the file's internal structure 
    # to prevent the "specified chunks separate stored chunks" warning.
    ds = xr.open_dataset(input_file, chunks='auto')

    # 2. Identify crop variables
    crop_vars = [v for v in ds.data_vars if v.endswith('_till')]
    print(f"Found {len(crop_vars)} crop variables to merge.")

    # 3. Stack variables
    print("Stacking variables... (Lazy evaluation)")
    stack = ds[crop_vars].to_array(dim='crop')

    # === THE FIX IS HERE ===
    # We must re-chunk so that the 'crop' dimension is NOT split.
    # -1 means "include the entire size of this dimension in one chunk".
    # We leave lat/lon as 'auto' to keep memory usage low.
    stack = stack.chunk({'crop': -1})

    # 4. Collapse dimensions using MODE
    print("Calculating the combined map (Finding dominant tillage class)...")
    
    # apply_ufunc allows us to apply the scipy function in parallel on the dask chunks
    combined_map = xr.apply_ufunc(
        _calc_mode,
        stack,
        input_core_dims=[['crop']], # The dimension we are collapsing
        dask='parallelized',        # Enable Dask parallel processing
        output_dtypes=[np.float32]  # output data type
    )

    # 5. Create clean output Dataset
    output_ds = xr.Dataset(
        data_vars={
            'tillage_class': (['lat', 'lon'], combined_map.data)
        },
        coords={
            'lat': ds.lat,
            'lon': ds.lon
        },
        attrs={
            'description': 'Dominant tillage map (Mode) from 42 individual crop variables',
            'classes': '1=Conventional, 2=Conservation, 3=Rotational',
            'logic_fix': 'Changed from MAX to MODE to prevent Rotational bias.'
        }
    )

    # 6. Save to disk with Compression
    print(f"Saving combined map to {output_file}...")
    
    # Compression settings
    encoding = {'tillage_class': {'zlib': True, 'complevel': 5, '_FillValue': -9999}}
    
    # compute=True triggers the actual calculation
    output_ds.to_netcdf(output_file, encoding=encoding, compute=True)
    
    print("\n" + "="*30)
    print("SUCCESS: Single layer created.")
    print(f"File saved: {output_file}")
    print("="*30)

if __name__ == "__main__":
    collapse_crops_to_map()