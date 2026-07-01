"""
MODULE 3 & 3b: LOGIT MODEL & CA ALLOCATION (GLOBAL 2-PASS EDITION)
Pass 1: Calculates Logit probability and determines country-specific thresholds.
Pass 2: Allocates 22-band Conservation Agriculture based on exact FAOSTAT quotas.
"""

import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── 1. Setup & Paths ──────────────────────────────────────────────────────────
path_global = Path(r"D:\Users\Wilson\Downloads\Sentinel\mod1_global_output_250m")
path_input = path_global.parent  

# UPDATED: Pointing to the correct CA dataset
faostat_file = path_input / "FAOSTAT_data_en_3-29-2026 (1).csv"

print("=" * 60)
print("MODULE 3/3b: GLOBAL LOGIT & CA ALLOCATION (2-PASS ENGINE)")
print("=" * 60)

# ── 2. Open Master Files & Copy Profiles ──────────────────────────────────────
try:
    src_crop_mix = rasterio.open(path_global / "crop_mix.tif")
    src_grains   = rasterio.open(path_global / "grains_stack.tif")
    src_arid     = rasterio.open(path_global / "aridity_res.tif")
    src_fields   = rasterio.open(path_global / "fields_interpol.tif")
    src_alloc    = rasterio.open(path_global / "spam_alloc_rast.tif")
except Exception as e:
    print(f"❌ Error locating Mod 1/2 outputs: {e}")
    exit()

profile_1band = src_crop_mix.profile.copy()
profile_22band = src_grains.profile.copy()

blocks = list(src_crop_mix.block_windows(1))
total_blocks = len(blocks)

# ── 3. PASS 1: Calculate Logit & Extract Thresholds ───────────────────────────
print(f"\n[PASS 1 OF 2]: Calculating Logit & Scanning {total_blocks} global chunks...")

out_logit = rasterio.open(path_global / "logit_ref.tif", 'w', **profile_1band)

# Mathematical constants for the Logit Equation
kvalue = np.array([0.25, 1/60, -5, 10])
xmid   = np.array([20, 12, 0.65, 0.5])
logit_f = - (xmid[0]*kvalue[0]) - (xmid[1]*kvalue[1]) - (xmid[2]*kvalue[2]) - (xmid[3]*kvalue[3])

valid_alloc = []
valid_logit = []
valid_pot_ca = []

with rasterio.Env(GDAL_CACHEMAX=2000000000):
    for idx, (ji, window) in enumerate(blocks):
        if idx % 100 == 0:
            print(f"  Pass 1 Scanning Block {idx}/{total_blocks}...", end='\r', flush=True)

        crop_mix = src_crop_mix.read(1, window=window)
        
        # Ocean/Desert Skipper
        if np.all(np.isnan(crop_mix)):
            out_logit.write(np.full((window.height, window.width), np.nan, dtype=np.float32), 1, window=window)
            continue
            
        arid   = src_arid.read(1, window=window)
        fields = src_fields.read(1, window=window)
        alloc  = src_alloc.read(1, window=window)
        grains = src_grains.read(window=window)

        # Logit Math
        b = (kvalue[0]*fields) + (kvalue[1]*12) + (kvalue[2]*arid) + (kvalue[3]*crop_mix)
        logit = 1 / (1 + np.exp(-(b + logit_f)))
        logit[np.isnan(crop_mix)] = np.nan
        
        out_logit.write(logit, 1, window=window)

        # Collect data for threshold solving
        valid_mask = ~np.isnan(logit) & (alloc > 0)
        if np.any(valid_mask):
            valid_alloc.append(alloc[valid_mask].astype(np.int32))
            valid_logit.append(logit[valid_mask])
            pot_ca = np.nansum(grains[:, valid_mask], axis=0) * logit[valid_mask]
            valid_pot_ca.append(pot_ca)

out_logit.close()

# ── 4. THE SOLVER: Determine Country Cutoff Scores in RAM ────────────────────
print("\n\n[THE SOLVER]: Calculating country-specific Logit thresholds...")

if not valid_alloc:
    print("❌ No valid cropland found. Exiting.")
    exit()

all_alloc = np.concatenate(valid_alloc)
all_logit = np.concatenate(valid_logit)
all_pot_ca = np.concatenate(valid_pot_ca)

del valid_alloc, valid_logit, valid_pot_ca 

# Load the Correct FAOSTAT Target Data
try:
    ca_fao = pd.read_csv(faostat_file)
    # Using the standard M49 Area Code from FAOSTAT
    ca_fao['country_code'] = ca_fao.get('Area Code (M49)')
    ca_fao_latest = ca_fao.sort_values('Year').groupby('Area').last().reset_index()
except Exception as e:
    print(f"⚠️ FAOSTAT file missing or malformed ({e}). Using 10% fallback for all countries.")
    ca_fao_latest = pd.DataFrame()

country_thresholds = {}
df_pixels = pd.DataFrame({'alloc': all_alloc, 'logit': all_logit, 'pot_ca': all_pot_ca})
grouped = df_pixels.groupby('alloc')

for country_code, group in grouped:
    target_area = 0
    if not ca_fao_latest.empty:
        match = ca_fao_latest[ca_fao_latest['country_code'] == country_code]
        if not match.empty:
            # FIX: Multiply by 1000 to convert from '1000 ha' to standard hectares
            target_area = match['Value'].values[0] * 1000
            
    if target_area == 0:
        target_area = group['pot_ca'].sum() * 0.10 
        
    group_sorted = group.sort_values(by='logit', ascending=False)
    cumsum = group_sorted['pot_ca'].cumsum()
    cutoff_idx = cumsum.searchsorted(target_area)
    
    if cutoff_idx >= len(group_sorted):
        cutoff_logit = group_sorted['logit'].min() 
    else:
        cutoff_logit = group_sorted.iloc[cutoff_idx]['logit']
        
    country_thresholds[country_code] = cutoff_logit

print(f"✓ Solved dynamic thresholds for {len(country_thresholds)} countries.")

# ── 5. PASS 2: Write Conservation Agriculture to Disk ─────────────────────────
print(f"\n[PASS 2 OF 2]: Allocating 22-Band CA to {total_blocks} chunks...")

out_ca = rasterio.open(path_global / "Conservation_Agriculture.tif", 'w', **profile_22band)
out_scenario = rasterio.open(path_global / "scenario_ca_area.tif", 'w', **profile_22band)
out_pot_ca = rasterio.open(path_global / "pot_ca_area.tif", 'w', **profile_22band)

src_logit = rasterio.open(path_global / "logit_ref.tif")

with rasterio.Env(GDAL_CACHEMAX=2000000000):
    for idx, (ji, window) in enumerate(blocks):
        if idx % 100 == 0:
            print(f"  Pass 2 Writing Block {idx}/{total_blocks}...", end='\r', flush=True)

        logit = src_logit.read(1, window=window)
        
        if np.all(np.isnan(logit)):
            empty_22 = np.full((22, window.height, window.width), np.nan, dtype=np.float32)
            out_ca.write(empty_22, window=window)
            out_scenario.write(empty_22, window=window)
            out_pot_ca.write(empty_22, window=window)
            continue

        grains = src_grains.read(window=window)
        alloc  = src_alloc.read(1, window=window)
        
        pot_ca_stack = grains * logit
        ca_out = np.zeros_like(pot_ca_stack)
        
        for country_code, cutoff in country_thresholds.items():
            winning_pixels = (alloc == country_code) & (logit >= cutoff)
            
            if np.any(winning_pixels):
                ca_out[:, winning_pixels] = pot_ca_stack[:, winning_pixels]
                
        ca_out[np.isnan(grains)] = np.nan
        
        remaining_potential = pot_ca_stack - ca_out
        remaining_potential[remaining_potential < 0] = 0
        scenario_out = ca_out + (remaining_potential * 0.5)
        
        out_ca.write(ca_out, window=window)
        out_scenario.write(scenario_out, window=window)
        out_pot_ca.write(pot_ca_stack, window=window)

src_logit.close()
out_ca.close()
out_scenario.close()
out_pot_ca.close()

for src in [src_crop_mix, src_grains, src_arid, src_fields, src_alloc]:
    src.close()

print(f"\n\n✓ GLOBAL LOGIT & CA ALLOCATION COMPLETE! Outputs saved to {path_global}")