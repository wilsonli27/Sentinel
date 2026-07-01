"""
MODULE 2: CROP MIX CALCULATION (GLOBAL EDITION - TRUE MASS PRESERVATION)
Bypasses the 4GB file limit using BIGTIFF=YES and Deflate compression.
Removed the biased high-income/field-size masks to preserve African/Global South smallholders.
"""
import numpy as np
import rasterio
from pathlib import Path
import warnings
import gc

warnings.filterwarnings("ignore")

path_global = Path(r"D:\Users\Wilson\Downloads\Sentinel\mod1_global_output_250m")
print("=" * 60)
print("MODULE 2: GLOBAL CROP MIX (TRUE MASS PRESERVATION EDITION)")
print("=" * 60)

min_area_threshold = 0.000001  
annual_indices = [0, 1, 2, 3, 4, 6, 12, 13, 16] + list(range(23, 42)) + [22]
grain_indices = [0, 2, 3, 4, 5, 6, 8] + list(range(14, 29))

try:
    src_ta     = rasterio.open(path_global / "SPAM_TA_Disaggregated.tif")
    src_fields = rasterio.open(path_global / "fields_interpol.tif")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# BIGTIFF profiles
out_meta_1 = src_fields.profile.copy()
out_meta_1.update({"BIGTIFF": "YES", "compress": "deflate", "tiled": True})

out_meta_22 = src_ta.profile.copy()
out_meta_22.update({"count": 22, "BIGTIFF": "YES", "compress": "deflate", "tiled": True})

out_crop_mix = rasterio.open(path_global / "crop_mix.tif", "w", **out_meta_1)
out_grains   = rasterio.open(path_global / "grains_stack.tif", "w", **out_meta_22)

blocks = list(src_ta.block_windows(1))
total_blocks = len(blocks)

with rasterio.Env(GDAL_CACHEMAX=2000000000, GDAL_NUM_THREADS='ALL_CPUS'):
    for idx, (ji, window) in enumerate(blocks):
        if idx % 100 == 0:
            print(f"  Processing Block {idx}/{total_blocks}...", end='\r', flush=True)

        ta_data = src_ta.read(window=window)
        
        # 🚀 Safe Pre-Scanner: Check the actual crops
        if np.all(np.isnan(ta_data)) or np.nanmax(ta_data) == 0:
            empty_1 = np.full((1, window.height, window.width), np.nan, dtype=np.float32)
            empty_22 = np.full((22, window.height, window.width), np.nan, dtype=np.float32)
            out_crop_mix.write(empty_1, window=window)
            out_grains.write(empty_22, window=window)
            continue

        # 🚀 TRUE MASS PRESERVATION: No arbitrary income/size deletions.
        # We extract the CA-eligible annuals directly from the raw SPAM data.
        annuals_all_stack = ta_data[annual_indices].copy()

        grains_stack = annuals_all_stack[grain_indices]
        grains_sum = np.nansum(grains_stack, axis=0)
        ta_sum     = np.nansum(ta_data, axis=0)

        ta_sum[ta_sum < min_area_threshold] = np.nan
        grains_sum[grains_sum < min_area_threshold] = 0.0

        crop_mix = grains_sum / ta_sum
        crop_mix[~np.isfinite(crop_mix)] = np.nan
        crop_mix[crop_mix > 1.5] = np.nan

        out_crop_mix.write(crop_mix, 1, window=window)
        out_grains.write(grains_stack, window=window)

        del ta_data, annuals_all_stack, grains_stack, crop_mix
        if idx % 50 == 0: gc.collect()

for src in [src_ta, src_fields, out_crop_mix, out_grains]: src.close()
print("\n✓ GLOBAL MOD 2 COMPLETE!")