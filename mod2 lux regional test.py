"""
MOD 2 - REGIONAL FIX: LUXEMBOURG (Float-Patched)
Lowers the area threshold to save smallholder/European farms 
and generates localized crop mix and grain stack files.
"""
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

path_global = Path(r"D:\Users\Wilson\Downloads\Sentinel\mod1_global_output_250m")
left, bottom, right, top = 5.5, 49.3, 6.7, 50.3

print("=" * 60)
print("MODULE 2 REGIONAL FIX: LUXEMBOURG")
print("=" * 60)

# 🚀 THE CRITICAL FIX: Dropped to save all small farms
min_area_threshold = 0.000001

annual_indices = [0, 1, 2, 3, 4, 6, 12, 13, 16] + list(range(23, 42)) + [22]
grain_indices = [0, 2, 3, 4, 5, 6, 8] + list(range(14, 29))

try:
    src_ta     = rasterio.open(path_global / "SPAM_TA_Disaggregated.tif")
    src_fields = rasterio.open(path_global / "fields_interpol.tif")
    src_high   = rasterio.open(path_global / "high_raster.tif")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

target_window = from_bounds(left, bottom, right, top, src_ta.transform)
win_transform = rasterio.windows.transform(target_window, src_ta.transform)

# Force dimensions to integers to prevent Numpy TypeErrors
h = int(target_window.height)
w = int(target_window.width)

print("Loading Mod 1 Data for Luxembourg...")
ta_data = src_ta.read(window=target_window)
fields  = src_fields.read(1, window=target_window)
high    = src_high.read(1, window=target_window)

print("Applying Ultra-Low Threshold to Save Small Farms...")
large_mask = fields >= 2.5
small_high_mask = (fields < 2.5) & (high == 400)

annuals_large = ta_data.copy()
annuals_large[:, ~large_mask] = np.nan

high_income_small = ta_data.copy()
high_income_small[:, ~small_high_mask] = np.nan

# Used the integer dimensions (h, w) here:
annuals_all_stack = np.zeros((len(annual_indices), h, w), dtype=np.float32)

for i, crop_idx in enumerate(annual_indices):
    l_arr = annuals_large[crop_idx]
    s_arr = high_income_small[crop_idx]
    combined = np.stack([l_arr, s_arr])
    arr_sum = np.nansum(combined, axis=0)
    both_nan = np.isnan(l_arr) & np.isnan(s_arr)
    arr_sum[both_nan] = np.nan
    annuals_all_stack[i] = arr_sum

grains_stack = annuals_all_stack[grain_indices]
grains_sum = np.nansum(grains_stack, axis=0)
ta_sum     = np.nansum(ta_data, axis=0)

ta_sum[ta_sum < min_area_threshold] = np.nan
grains_sum[grains_sum < min_area_threshold] = 0.0

crop_mix = grains_sum / ta_sum
crop_mix[~np.isfinite(crop_mix)] = np.nan
crop_mix[crop_mix > 1.5] = np.nan

print(f"  -> SURVIVOR COUNT (Valid Pixels): {np.sum(~np.isnan(crop_mix)):,}")

out_meta_1 = src_fields.profile.copy()
out_meta_1.update({"height": h, "width": w, "transform": win_transform})

out_meta_22 = src_ta.profile.copy()
out_meta_22.update({"count": 22, "height": h, "width": w, "transform": win_transform})

with rasterio.open(path_global / "lux_crop_mix.tif", "w", **out_meta_1) as dest:
    dest.write(crop_mix, 1)

with rasterio.open(path_global / "lux_grains_stack.tif", "w", **out_meta_22) as dest:
    dest.write(grains_stack)

print("✅ Saved lux_crop_mix.tif and lux_grains_stack.tif!")