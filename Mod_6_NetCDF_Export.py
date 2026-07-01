"""
MODULE 6: FINAL TILLAGE MERGE (GLOBAL MASTER LAYER)
Aggregates the 42 crop-specific bands from Mod 4 and Mod 3 
into a single, clean 6-band Master TIFF representing total 
area (in hectares) for each of the 6 global tillage practices.
"""

import numpy as np
import rasterio
from pathlib import Path
import warnings
import time
import gc

warnings.filterwarnings("ignore")

# ── 1. Setup & Paths ──────────────────────────────────────────────────────────
path_global = Path(r"D:\Users\Wilson\Downloads\Sentinel\mod1_global_output_250m")

print("=" * 60)
print("MODULE 6: FINAL GLOBAL TILLAGE MERGE")
print("=" * 60)

# ── 2. Open Input Datasets ────────────────────────────────────────────────────
try:
    src_trad_ann = rasterio.open(path_global / "traditional_annual_tillage.tif")
    src_trad_rot = rasterio.open(path_global / "traditional_rotational_tillage.tif")
    src_rot      = rasterio.open(path_global / "rotational_tillage.tif")
    src_reduced  = rasterio.open(path_global / "reduced_tillage.tif")
    src_conv     = rasterio.open(path_global / "conventional_annual_tillage.tif")
    src_ca       = rasterio.open(path_global / "Conservation_Agriculture.tif")
except Exception as e:
    print(f"❌ Error locating Mod 4 or Mod 3 input datasets: {e}")
    exit()

# ── 3. Configure the Master Output Profile ────────────────────────────────────
# We base it on one of the inputs, but force it to exactly 6 bands.
master_profile = src_trad_ann.profile.copy()
master_profile.update(
    count=6, 
    BIGTIFF="YES", 
    compress="deflate", 
    tiled=True, 
    nodata=np.nan,
    predictor=3 # Keeps the final file perfectly lightweight
)

# ── 4. Open Output Writer ─────────────────────────────────────────────────────
out_master = rasterio.open(path_global / "Global_Tillage_Classes_Master.tif", 'w', **master_profile)

# Set band descriptions so they are beautifully labeled in QGIS
out_master.set_band_description(1, 'Traditional Annual (ha)')
out_master.set_band_description(2, 'Traditional Rotational (ha)')
out_master.set_band_description(3, 'Rotational (ha)')
out_master.set_band_description(4, 'Reduced (ha)')
out_master.set_band_description(5, 'Conventional Annual (ha)')
out_master.set_band_description(6, 'Conservation Agriculture (ha)')

blocks = list(src_trad_ann.block_windows(1))
total_blocks = len(blocks)

print(f"Merging {total_blocks} global chunks into the Master Layer...\n")

skipped_empty = 0
start_time = time.time()

# ── 5. Block-by-Block Processing Engine ───────────────────────────────────────
# Fast, RAM-safe processing limits
with rasterio.Env(GDAL_CACHEMAX=500000000, GDAL_NUM_THREADS='2'):
    for idx, (ji, window) in enumerate(blocks):
        
        if idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Merged {idx}/{total_blocks} | Empty Skipped: {skipped_empty} | Run Time: {elapsed:.1f}s", end='\r', flush=True)

        # 🚀 THE SKIPPER: We use the smallest file (Trad Rot) as a quick ocean scout
        scout = src_trad_rot.read(window=window)
        if np.all(np.isnan(scout)) or np.nanmax(scout) == 0:
            # If scout is empty, do a fast secondary check on Reduced (the largest)
            scout_two = src_reduced.read(window=window)
            if np.all(np.isnan(scout_two)) or np.nanmax(scout_two) == 0:
                skipped_empty += 1
                del scout, scout_two
                continue
            del scout_two
        del scout

        # ── READ ALL FILES ──
        trad_ann = src_trad_ann.read(window=window)
        trad_rot = src_trad_rot.read(window=window)
        rot      = src_rot.read(window=window)
        reduced  = src_reduced.read(window=window)
        conv     = src_conv.read(window=window)
        ca       = src_ca.read(window=window)

        # ── COLLAPSE TO 6 BANDS ──
        # Sum across the crop bands (axis=0) to get total area per tillage class
        b1_trad_ann = np.nansum(trad_ann, axis=0)
        b2_trad_rot = np.nansum(trad_rot, axis=0)
        b3_rot      = np.nansum(rot, axis=0)
        b4_reduced  = np.nansum(reduced, axis=0)
        b5_conv     = np.nansum(conv, axis=0)
        b6_ca       = np.nansum(ca, axis=0)

        # ── STACK AND CLEAN ──
        master_stack = np.stack([
            b1_trad_ann, 
            b2_trad_rot, 
            b3_rot, 
            b4_reduced, 
            b5_conv, 
            b6_ca
        ]).astype(np.float32)

        # Re-apply the NoData mask so empty land doesn't show as 0
        master_stack[master_stack == 0] = np.nan

        # ── WRITE TO DISK ──
        out_master.write(master_stack, window=window)

        # Keep RAM completely clear
        del trad_ann, trad_rot, rot, reduced, conv, ca, master_stack
        del b1_trad_ann, b2_trad_rot, b3_rot, b4_reduced, b5_conv, b6_ca
        
        if idx % 100 == 0:
            gc.collect()

# ── 6. Cleanup ────────────────────────────────────────────────────────────────
for src in [src_trad_ann, src_trad_rot, src_rot, src_reduced, src_conv, src_ca, out_master]:
    src.close()

print(f"\n\n✓ MOD 6 COMPLETE: Global Tillage Master Layer generated in {(time.time() - start_time)/60:.1f} minutes!")