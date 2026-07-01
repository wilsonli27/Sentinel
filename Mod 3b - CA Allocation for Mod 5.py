"""
MODULE 3b: FAST CA ALLOCATION (SKIPPING PASS 1)
Reads existing Logit data, applies the M49 -> ISO3 -> 1-239 translation, 
and directly allocates the Conservation Agriculture pixels.
"""

import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── 1. Setup & Paths ──────────────────────────────────────────────────────────
path_global = Path(r"D:\Users\Wilson\Downloads\Sentinel\mod1_global_output_250m")
path_input = path_global.parent  

faostat_file = path_input / "FAOSTAT_data_en_3-29-2026 (1).csv"
gpkg_path = path_input / "output" / "country_allocation_fixed.gpkg"

print("=" * 60)
print("MODULE 3b: FAST CA ALLOCATION (SKIPPING PASS 1 MATH)")
print("=" * 60)

# ── 2. Open Files & Rebuild RAM Arrays ────────────────────────────────────────
src_logit  = rasterio.open(path_global / "logit_ref.tif")
src_grains = rasterio.open(path_global / "grains_stack.tif")
src_alloc  = rasterio.open(path_global / "spam_alloc_rast.tif")

profile_22band = src_grains.profile.copy()
blocks = list(src_logit.block_windows(1))
total_blocks = len(blocks)

print("\n[STEP 1]: Rebuilding RAM arrays from existing Logit map...")
valid_alloc, valid_logit, valid_pot_ca = [], [], []

with rasterio.Env(GDAL_CACHEMAX=2000000000):
    for idx, (ji, window) in enumerate(blocks):
        if idx % 100 == 0:
            print(f"  Scanning Block {idx}/{total_blocks}...", end='\r', flush=True)

        logit = src_logit.read(1, window=window)
        if np.all(np.isnan(logit)):
            continue
            
        alloc  = src_alloc.read(1, window=window)
        grains = src_grains.read(window=window)

        valid_mask = ~np.isnan(logit) & (alloc > 0)
        if np.any(valid_mask):
            valid_alloc.append(alloc[valid_mask].astype(np.int32))
            valid_logit.append(logit[valid_mask])
            pot_ca = np.nansum(grains[:, valid_mask], axis=0) * logit[valid_mask]
            valid_pot_ca.append(pot_ca)

all_alloc = np.concatenate(valid_alloc)
all_logit = np.concatenate(valid_logit)
all_pot_ca = np.concatenate(valid_pot_ca)
del valid_alloc, valid_logit, valid_pot_ca 

# ── 3. THE BRIDGE: Translating IDs and Solving Thresholds ─────────────────────
print("\n\n[STEP 2]: Translating Country IDs and Solving Quotas...")

# Load the numbering system from your Geopackage (1-239 IDs)
gdf_countries = gpd.read_file(gpkg_path, layer="countries", ignore_geometry=True)
iso3_to_map_id = dict(zip(gdf_countries['iso3'], gdf_countries['country_code']))

# Load UN standard crosswalk
url = "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
un_crosswalk = pd.read_csv(url)
m49_to_iso3 = dict(zip(un_crosswalk['country-code'], un_crosswalk['alpha-3']))

try:
    ca_fao = pd.read_csv(faostat_file)
    ca_fao['country_code_m49'] = ca_fao.get('Area Code (M49)')
    ca_fao_latest = ca_fao.sort_values('Year').groupby('Area').last().reset_index()
    
    # Translate M49 -> ISO3 -> Map ID
    ca_fao_latest['iso3'] = ca_fao_latest['country_code_m49'].map(m49_to_iso3)
    ca_fao_latest['map_id'] = ca_fao_latest['iso3'].map(iso3_to_map_id)
    ca_fao_latest = ca_fao_latest.dropna(subset=['map_id'])
except Exception as e:
    print(f"⚠️ FAOSTAT Error: {e}")
    ca_fao_latest = pd.DataFrame()

country_thresholds = {}
df_pixels = pd.DataFrame({'alloc': all_alloc, 'logit': all_logit, 'pot_ca': all_pot_ca})
grouped = df_pixels.groupby('alloc')

for map_id, group in grouped:
    target_area = 0
    if not ca_fao_latest.empty:
        match = ca_fao_latest[ca_fao_latest['map_id'] == map_id]
        if not match.empty:
            target_area = match['Value'].values[0] * 1000  # The 1000 ha fix
            
    if target_area == 0:
        target_area = group['pot_ca'].sum() * 0.10 
        
    group_sorted = group.sort_values(by='logit', ascending=False)
    cumsum = group_sorted['pot_ca'].cumsum()
    cutoff_idx = cumsum.searchsorted(target_area)
    
    if cutoff_idx >= len(group_sorted):
        country_thresholds[map_id] = group_sorted['logit'].min() 
    else:
        country_thresholds[map_id] = group_sorted.iloc[cutoff_idx]['logit']

print(f"✓ Solved dynamic thresholds for {len(country_thresholds)} countries.")

# ── 4. WRITE PASS: Allocating the pixels ──────────────────────────────────────
print(f"\n[STEP 3]: Allocating 22-Band CA to {total_blocks} chunks...")

out_ca = rasterio.open(path_global / "Conservation_Agriculture.tif", 'w', **profile_22band)
out_scenario = rasterio.open(path_global / "scenario_ca_area.tif", 'w', **profile_22band)
out_pot_ca = rasterio.open(path_global / "pot_ca_area.tif", 'w', **profile_22band)

with rasterio.Env(GDAL_CACHEMAX=2000000000):
    for idx, (ji, window) in enumerate(blocks):
        if idx % 100 == 0:
            print(f"  Writing Block {idx}/{total_blocks}...", end='\r', flush=True)

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
        
        for map_id, cutoff in country_thresholds.items():
            winning_pixels = (alloc == map_id) & (logit >= cutoff)
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
src_grains.close()
src_alloc.close()
out_ca.close()
out_scenario.close()
out_pot_ca.close()

print("\n\n✓ FAST ALLOCATION COMPLETE!")