import rasterio
from rasterio.windows import from_bounds, Window
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Define Paths
path_raw = Path("path_input/test_output_250m/")
file_disagg = Path("mod1_final_output_250m/SPAM_TA_Disaggregated.tif")

# Target a 100km x 100km box in Iowa (~400 x 400 pixels)
target_lon, target_lat = -93.5, 42.0
bbox = (target_lon - 0.45, target_lat - 0.45, target_lon + 0.45, target_lat + 0.45)

print("==================================================")
print(" MASS PRESERVATION DIAGNOSTIC (100km x 100km Box) ")
print("==================================================")

# 1. Get the NEW Disaggregated Mass (Band 3 is Maize)
try:
    with rasterio.open(file_disagg) as src_disagg:
        win_disagg = from_bounds(*bbox, transform=src_disagg.transform)
        win_disagg = win_disagg.round_offsets().round_lengths()
        
        disagg_maize = src_disagg.read(3, window=win_disagg)
        disagg_total = np.nansum(disagg_maize)
        max_pixel_ha = np.nanmax(disagg_maize)
except Exception as e:
    print(f"❌ Could not read Disaggregated file: {e}")
    disagg_total, max_pixel_ha = 0, 0

# 2. Get the ORIGINAL SPAM Mass (from the raw GEE shards)
raw_files = list(path_raw.glob("spam2020_V2r0_global_A_MAIZ_A*.tif"))
raw_total = 0.0

for fp in raw_files:
    with rasterio.open(fp) as src_raw:
        # Check if the GEE shard intersects our Iowa bounding box
        if not (src_raw.bounds.right <= bbox[0] or src_raw.bounds.left >= bbox[2] or 
                src_raw.bounds.top <= bbox[1] or src_raw.bounds.bottom >= bbox[3]):
            
            win_raw = from_bounds(*bbox, transform=src_raw.transform)
            win_raw = win_raw.round_offsets().round_lengths()
            win_raw = win_raw.intersection(Window(0, 0, src_raw.width, src_raw.height))
            
            if win_raw.width > 0 and win_raw.height > 0:
                raw_data = src_raw.read(1, window=win_raw)
                
                # Because GEE duplicated the 10km values across 1,600 small 250m pixels,
                # we must divide the sum of the raw array by 1,600 to find the TRUE ground mass.
                valid_data = np.where(raw_data > 0, raw_data, 0)
                raw_total += np.nansum(valid_data) / 1600.0

print(f"1. ORIGINAL SPAM MAIZE MASS : {raw_total:,.2f} hectares")
print(f"2. NEW DISAGGREGATED MASS   : {disagg_total:,.2f} hectares")

diff = abs(raw_total - disagg_total)
pct_diff = (diff / max(raw_total, 1)) * 100

print(f"-> DIFFERENCE               : {diff:,.2f} ha ({pct_diff:.4f}%)")

print("\n--- ML CNN OVERFLOW CHECK ---")
print(f"Maximum Hectares in a single 250m Pixel: {max_pixel_ha:.2f} ha")
if max_pixel_ha > 6.25:
    print("✅ SUCCESS: Pixels successfully exceeded the 6.25ha physical limit.")
    print("   (ESA underestimated dirt here, so the algorithm squeezed the SPAM data")
    print("   to conserve total mass, creating dense anchor points for your CNN).")
else:
    print("⚠️ NOTE: Max pixel is <= 6.25ha. ESA found enough physical dirt to perfectly")
    print("   fit all crops without exceeding physical limits in this specific box.")