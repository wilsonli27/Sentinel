"""
MODULE 5: TILLAGE SUMS & STATISTICS (GLOBAL RASTERIO EDITION)
Calculates global and country-specific hectare totals for all 6 tillage practices.
Features Multi-Core Decompression and Fail-Fast reading to bypass empty landmasses.
"""

import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# ── 1. Setup & Paths ──────────────────────────────────────────────────────────
path_global = Path(r"D:\Users\Wilson\Downloads\Sentinel\mod1_global_output_250m")

print("=" * 60)
print("MODULE 5: GLOBAL TILLAGE SUMS (MULTI-THREAD OPTIMIZED)")
print("=" * 60)

# ── 2. Initialize Master Counters ─────────────────────────────────────────────
global_sums = {
    "total_cropland": 0.0,
    "conventional_annual": 0.0,
    "traditional_annual": 0.0,
    "reduced_tillage": 0.0,
    "rotational_tillage": 0.0,
    "traditional_rotational": 0.0,
    "conservation_agriculture": 0.0,
    "scenario_ca_area": 0.0
}

country_sums = defaultdict(lambda: {k: 0.0 for k in global_sums})

# ── 3. Open All Data Sources ──────────────────────────────────────────────────
try:
    src_ta       = rasterio.open(path_global / "SPAM_TA_Disaggregated.tif")
    src_alloc    = rasterio.open(path_global / "spam_alloc_rast.tif")
    src_conv     = rasterio.open(path_global / "conventional_annual_tillage.tif")
    src_trad_ann = rasterio.open(path_global / "traditional_annual_tillage.tif")
    src_reduced  = rasterio.open(path_global / "reduced_tillage.tif")
    src_rot      = rasterio.open(path_global / "rotational_tillage.tif")
    src_trad_rot = rasterio.open(path_global / "traditional_rotational_tillage.tif")
    src_ca       = rasterio.open(path_global / "Conservation_Agriculture.tif")
    src_scen     = rasterio.open(path_global / "scenario_ca_area.tif")
except Exception as e:
    print(f"❌ Error locating input datasets: {e}")
    exit()

blocks = list(src_ta.block_windows(1))
total_blocks = len(blocks)

print(f"Auditing {total_blocks} global chunks using ALL CPU CORES...")

# ── 4. The Block-by-Block Auditing Engine ─────────────────────────────────────
# 🚀 OPTIMIZATION 1: UNLOCK ALL CPU CORES FOR DECOMPRESSION 🚀
with rasterio.Env(GDAL_CACHEMAX=2000000000, GDAL_NUM_THREADS='ALL_CPUS'):
    for idx, (ji, window) in enumerate(blocks):
        if idx % 50 == 0:
            print(f"  Scanning Block {idx}/{total_blocks}...", end='\r', flush=True)

        # 🚀 OPTIMIZATION 2: THE FAIL-FAST WATERFALL 🚀
        # Step A: Check for Country Codes
        alloc = src_alloc.read(1, window=window)
        valid_mask = ~np.isnan(alloc) & (alloc > 0)
        
        if not np.any(valid_mask):
            continue # Skip Ocean instantly
            
        # Step B: Check for Actual Cropland
        ta = src_ta.read(window=window)
        if np.all(np.isnan(ta)) or np.nanmax(ta) == 0:
            continue # Skip Deserts/Ice instantly. Saves reading 6 heavy files!

        # Step C: Only read the heavy files if crops exist
        data_dict = {
            "total_cropland": ta,
            "conventional_annual": src_conv.read(window=window),
            "traditional_annual": src_trad_ann.read(window=window),
            "reduced_tillage": src_reduced.read(window=window),
            "rotational_tillage": src_rot.read(window=window),
            "traditional_rotational": src_trad_rot.read(window=window),
            "conservation_agriculture": src_ca.read(window=window),
            "scenario_ca_area": src_scen.read(window=window)
        }

        alloc_valid = alloc[valid_mask].astype(int)

        for key, arr in data_dict.items():
            # 1. Update Global Hectares
            chunk_total = np.nansum(arr)
            global_sums[key] += float(chunk_total)
            
            # 2. Update Country Hectares
            flat_sum = np.nansum(arr, axis=0)
            flat_valid = flat_sum[valid_mask]
            
            # Ultra-fast NumPy Grouping
            counts = np.bincount(alloc_valid, weights=flat_valid)
            active_ccs = np.nonzero(counts)[0]
            
            for cc in active_ccs:
                country_sums[cc][key] += float(counts[cc])

# ── 5. Format and Save Results ────────────────────────────────────────────────
for src in [src_ta, src_alloc, src_conv, src_trad_ann, src_reduced, src_rot, src_trad_rot, src_ca, src_scen]:
    src.close()

print("\n\n" + "="*60)
print("AUDIT COMPLETE: EXPORTING STATISTICS")
print("="*60)

# ── A. Global Table ──
df_global = pd.DataFrame.from_dict(global_sums, orient='index', columns=['Global_Hectares'])
df_global['Global_Mha'] = df_global['Global_Hectares'] / 1_000_000

print("\n--- GLOBAL TILLAGE TOTALS ---")
print(df_global.round(2))

global_csv = path_global / "tab_tillage_types_area.csv"
df_global.to_csv(global_csv)
print(f"✓ Saved Global Totals: {global_csv}")

# ── B. Country Table ──
df_country = pd.DataFrame.from_dict(country_sums, orient='index')
df_country.index.name = 'Country_Code'
df_country = df_country.sort_index()

country_csv = path_global / "tillage_per_country.csv"
df_country.to_csv(country_csv)
print(f"✓ Saved Country Totals: {country_csv}")