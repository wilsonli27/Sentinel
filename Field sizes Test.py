import rasterio
import numpy as np
from pathlib import Path

path_input = Path("D:/Users/Wilson/Downloads/Sentinel/")
field_file = path_input / "lesiv_2018_field_sizes" / "Global Field Sizes" / "dominant_field_size_categories.tif"

print("FIELD SIZE DIAGNOSTIC - READING FROM CENTER")
print("="*60)

with rasterio.open(field_file) as src:
    print(f"File info:")
    print(f"  Shape: {src.shape}")
    print(f"  Resolution: {src.res}")
    
    # Read from CENTER where cropland likely exists
    center_y = src.height // 2
    center_x = src.width // 2
    
    # Read 2000x2000 window from center
    window = ((center_y - 1000, center_y + 1000), 
              (center_x - 1000, center_x + 1000))
    
    sample = src.read(1, window=window)
    
    print(f"\nSample from CENTER ({center_y}, {center_x}):")
    print(f"  Min: {np.nanmin(sample)}")
    print(f"  Max: {np.nanmax(sample)}")
    print(f"  Mean: {np.nanmean(sample):.2f}")
    print(f"  Non-zero cells: {np.sum(sample > 0):,} / {sample.size:,}")
    print(f"  Unique values: {len(np.unique(sample))}")
    
    unique_vals = sorted(np.unique(sample))
    print(f"  All unique values: {unique_vals}")
    
    # Also check global stats
    print(f"\nReading entire file (this may take a moment)...")
    full_data = src.read(1)
    print(f"  Global min: {np.nanmin(full_data)}")
    print(f"  Global max: {np.nanmax(full_data)}")
    print(f"  Global unique values: {len(np.unique(full_data))}")
    print(f"  All unique: {sorted(np.unique(full_data))}")