import numpy as np
import rasterio
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────
path_global = Path(r"D:\Users\Wilson\Downloads\Sentinel\mod1_global_output_250m")
master_tif = path_global / "Global_Tillage_Classes_Master.tif"

print("=" * 60)
print("PHYSICAL PIXEL VERIFICATION DIAGNOSTIC")
print("=" * 60)

total_grid_pixels = 0
valid_data_pixels = 0

try:
    with rasterio.open(master_tif) as src:
        blocks = list(src.block_windows(1))
        total_blocks = len(blocks)
        
        print(f"Scanning {total_blocks} chunks to count physical data points...\n")
        
        for idx, (ji, window) in enumerate(blocks):
            if idx % 500 == 0:
                print(f"  Scanning block {idx}/{total_blocks}...", end='\r', flush=True)
                
            data = src.read(window=window)
            
            # A pixel is valid if the sum of its 6 bands > 0
            # Using np.nansum to handle any NoData/NaNs safely
            pixel_sums = np.nansum(data, axis=0)
            
            # Count the pixels
            valid_count = np.sum(pixel_sums > 0)
            total_count = pixel_sums.size
            
            valid_data_pixels += valid_count
            total_grid_pixels += total_count

    empty_pixels = total_grid_pixels - valid_data_pixels
    
    print("\n\n" + "=" * 60)
    print("FINAL HARDWARE PIXEL COUNT")
    print("=" * 60)
    
    valid_pct = (valid_data_pixels / total_grid_pixels) * 100 if total_grid_pixels > 0 else 0
    empty_pct = (empty_pixels / total_grid_pixels) * 100 if total_grid_pixels > 0 else 0
    
    print(f"Total Pixels in Global Grid : {total_grid_pixels:>18,}")
    print(f"Empty Pixels (Ocean/Desert) : {empty_pixels:>18,}  ({empty_pct:>6.2f}%)")
    print(f"Valid Pixels (Farms/Data)   : {valid_data_pixels:>18,}  ({valid_pct:>6.2f}%)")
    print("=" * 60)
    
    if valid_data_pixels > 0:
        print(f"\n✅ SUCCESS: You have {valid_data_pixels:,} physical pixels containing actual data.")
        print("Your map is NOT empty. QGIS is just struggling to render them all at once.")
    else:
        print("\n❌ WARNING: Zero valid pixels found. The map is completely empty.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")