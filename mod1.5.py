"""
MODULE 1.5: THE COUNTRY FIXER
Surgically repairs the missing country allocation map without touching 
the heavy 26GB crop data files.
"""

import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── 1. Setup Absolute Paths ───────────────────────────────────────────────────
path_global = Path(r"D:\Users\Wilson\Downloads\Sentinel\mod1_global_output_250m")
# Hardcoded absolute path to guarantee it finds your vector data
path_gpkg = Path(r"D:\Users\Wilson\Downloads\Sentinel\output\country_allocation_fixed.gpkg")

print("=" * 60)
print("MODULE 1.5: COUNTRY ALLOCATION REPAIR")
print("=" * 60)

print("Loading Country Vector Data (This may take a moment)...")
if not path_gpkg.exists():
    print(f"❌ ERROR: Cannot find the Geopackage at {path_gpkg}")
    print("Please verify the file exists before continuing.")
    exit()
    
gdf = gpd.read_file(path_gpkg, layer="countries")

# ── 2. Steal the 250m Grid Profile from Crop Mix ──────────────────────────────
with rasterio.open(path_global / "crop_mix.tif") as src:
    profile = src.profile.copy()
    global_transform = src.transform
    blocks = list(src.block_windows(1))
    
total_blocks = len(blocks)

# ── 3. Burn the Vector Data into the Raster Grid ──────────────────────────────
print(f"Burning Country Codes to 250m Grid across {total_blocks} chunks...")

with rasterio.open(path_global / "spam_alloc_rast.tif", 'w', **profile) as out_alloc:
    with rasterio.Env(GDAL_CACHEMAX=2000000000):
        for idx, (ji, window) in enumerate(blocks):
            if idx % 100 == 0:
                print(f"  Processing Block {idx}/{total_blocks}...", end='\r', flush=True)

            # Get the geographic bounding box of the current chunk
            win_transform = rasterio.windows.transform(window, global_transform)
            win_bounds = rasterio.windows.bounds(window, global_transform)
            
            # Fast spatial filter: only get polygons touching this specific chunk
            local_gdf = gdf.cx[win_bounds[0]:win_bounds[2], win_bounds[1]:win_bounds[3]]
            
            if not local_gdf.empty:
                # Burn the integer country codes into the raster pixels
                shapes = ((geom, val) for geom, val in zip(local_gdf.geometry, local_gdf["country_code"]))
                alloc_data = rasterize(
                    shapes=shapes, 
                    out_shape=(window.height, window.width), 
                    transform=win_transform, 
                    fill=np.nan, 
                    dtype=np.float32
                )
            else:
                # Ocean / Open Water
                alloc_data = np.full((window.height, window.width), np.nan, dtype=np.float32)
                
            out_alloc.write(alloc_data, 1, window=window)

print("\n\n✓ Country Allocation Map Fixed! You may now run Module 3.")