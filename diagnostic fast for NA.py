import rasterio
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

tif_path = Path("mod1_final_output_250m/SPAM_TA_Disaggregated.tif") 

print(f"Executing Continental Radar Sweep on: {tif_path.name}")
print("This will take ~10 seconds. Scanning millions of pixels...")

max_ha = 0.0

try:
    with rasterio.open(tif_path) as src:
        total_blocks = len(list(src.block_windows(1)))
        
        for i, (ji, window) in enumerate(src.block_windows(1)):
            if i % 100 == 0:
                print(f"  Scanning chunk {i}/{total_blocks}...", end='\r')
                
            # Read all 42 bands for this specific block
            data = src.read(window=window)
            
            # Sum all crops to get total agriculture in each pixel
            total_area = np.nansum(data, axis=0)
            
            # Find the absolute max in this block
            block_max = np.nanmax(total_area)
            if block_max > max_ha:
                max_ha = block_max
                
    print("\n\n" + "="*50)
    print(f"GLOBAL MAXIMUM PIXEL AREA: {max_ha:.4f} hectares")
    print("="*50)

    if max_ha > 0.0 and max_ha <= 6.25:
        print("✅ SUCCESS! The data is present globally, and NO pixels overlap beyond 6.25ha.")
    elif max_ha == 0.0:
        print("❌ The file is actually empty. We have a write bug.")
    else:
        print("⚠️ Pixels are still overlapping beyond limits.")

except Exception as e:
    print(f"Error: {e}")