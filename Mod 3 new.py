"""
MODULE 3: GLOBAL CA PIPELINE (MEMORY OPTIMIZED EDITION)
Dynamically calculates the Logit, scales overflowing pixels,
and solves quotas using a low-RAM country-by-country loop.
Throttled to 2 Threads / 500MB RAM to prevent OS freezing.
Outputs enforce BigTIFF/Deflate to prevent file size crashes.
"""
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd
from pathlib import Path
import warnings
import gc

warnings.filterwarnings("ignore")

# ── 1. Setup & Paths ──────────────────────────────────────────────────────────
path_global = Path(r"D:\Users\Wilson\Downloads\Sentinel\mod1_global_output_250m")
path_input = path_global.parent  
faostat_file = path_input / "FAOSTAT_cons_ag.csv"
gpkg_path = path_input / "output" / "country_allocation_fixed.gpkg"

print("=" * 60)
print("MODULE 3: GLOBAL CA ALLOCATION (LOW-RAM PIPELINE)")
print("=" * 60)

try:
    src_crop_mix = rasterio.open(path_global / "crop_mix.tif")
    src_grains   = rasterio.open(path_global / "grains_stack.tif")
    src_arid     = rasterio.open(path_global / "aridity_res.tif")
    src_fields   = rasterio.open(path_global / "fields_interpol.tif")
    src_alloc    = rasterio.open(path_global / "spam_alloc_rast.tif")
except Exception as e:
    print(f"❌ Error locating input files: {e}")
    exit()

# ── 2. Force BigTIFF on all profiles ──────────────────────────────────────────
profile_1band = src_crop_mix.profile.copy()
profile_1band.update({"BIGTIFF": "YES", "compress": "deflate", "tiled": True})

profile_22band = src_grains.profile.copy()
profile_22band.update({"BIGTIFF": "YES", "compress": "deflate", "tiled": True})

blocks = list(src_crop_mix.block_windows(1))
total_blocks = len(blocks)

kvalue = np.array([0.25, 1/60, -5, 10])
xmid   = np.array([20, 12, 0.65, 0.5])
logit_f = - (xmid[0]*kvalue[0]) - (xmid[1]*kvalue[1]) - (xmid[2]*kvalue[2]) - (xmid[3]*kvalue[3])

# ── 3. PASS 1: Dynamic Logit & Scaled Potential CA ────────────────────────────
print("\n[STEP 1]: Calculating Dynamic Global Logit & Scaling Areas...")
out_logit = rasterio.open(path_global / "logit_global_fixed.tif", 'w', **profile_1band)

valid_alloc, valid_logit, valid_pot_ca = [], [], []

with rasterio.Env(GDAL_CACHEMAX=500000000, GDAL_NUM_THREADS='2'):
    for idx, (ji, window) in enumerate(blocks):
        if idx % 100 == 0:
            print(f"  Scanning Block {idx}/{total_blocks}...", end='\r', flush=True)

        crop_mix = src_crop_mix.read(1, window=window)
        if np.all(np.isnan(crop_mix)):
            out_logit.write(np.full((1, window.height, window.width), np.nan, dtype=np.float32), window=window)
            continue
            
        arid   = src_arid.read(1, window=window)
        fields = src_fields.read(1, window=window)
        alloc  = src_alloc.read(1, window=window)
        grains = src_grains.read(window=window)

        # Calculate Logit
        b = (kvalue[0]*fields) + (kvalue[1]*12) + (kvalue[2]*arid) + (kvalue[3]*crop_mix)
        logit = 1 / (1 + np.exp(-(b + logit_f)))
        logit[np.isnan(crop_mix)] = np.nan
        out_logit.write(logit, 1, window=window)

        # Proportional Scaler
        grains[np.isnan(grains)] = 0.0
        grains[grains < 0] = 0.0
        
        pixel_sum = np.sum(grains, axis=0)
        overflow_mask = pixel_sum > 6.25
        
        if np.any(overflow_mask):
            scale = np.ones_like(pixel_sum)
            scale[overflow_mask] = 6.25 / pixel_sum[overflow_mask]
            grains = grains * scale

        grains[grains == 0] = np.nan

        valid_mask = ~np.isnan(logit) & (alloc > 0)
        if np.any(valid_mask):
            valid_alloc.append(alloc[valid_mask].astype(np.int32))
            valid_logit.append(logit[valid_mask])
            pot_ca = np.nansum(grains[:, valid_mask], axis=0) 
            valid_pot_ca.append(pot_ca)

out_logit.close() 
all_alloc = np.concatenate(valid_alloc)
all_logit = np.concatenate(valid_logit)
all_pot_ca = np.concatenate(valid_pot_ca)
del valid_alloc, valid_logit, valid_pot_ca 
gc.collect()

# ── 4. GPKG Bridge & Quota Solver (MEMORY OPTIMIZED) ──────────────────────────
print("\n\n[STEP 2]: Reading GPKG Bridge & Solving Quotas (Low RAM Mode)...")
gdf_countries = gpd.read_file(gpkg_path, layer="countries", ignore_geometry=True)
iso3_to_map_id = dict(zip(gdf_countries['iso3'].str.upper(), gdf_countries['country_code'].astype(int)))

ca_fao = pd.read_csv(faostat_file)
ca_fao['country_code_m49'] = ca_fao['Area Code (M49)'].fillna(-1).astype(int)
ca_fao_latest = ca_fao.sort_values('Year').groupby('Area').last().reset_index()

fao_iso3_map = {
    'Canada': 'CAN', 'Croatia': 'HRV', 'Czechia': 'CZE', 'Denmark': 'DNK', 
    'Ecuador': 'ECU', 'France': 'FRA', 'Germany': 'DEU', 'Hungary': 'HUN', 
    'Lithuania': 'LTU', 'Luxembourg': 'LUX', 'Netherlands (Kingdom of the)': 'NLD', 
    'Panama': 'PAN', 'Poland': 'POL', 'Portugal': 'PRT', 'Spain': 'ESP', 
    'United States of America': 'USA', 'Zambia': 'ZMB'
}
ca_fao_latest['iso3'] = ca_fao_latest['Area'].map(fao_iso3_map)

url = "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
try:
    un_crosswalk = pd.read_csv(url)
    m49_to_iso3 = dict(zip(un_crosswalk['country-code'].fillna(-1).astype(int), un_crosswalk['alpha-3'].astype(str).str.upper()))
    missing_iso3 = ca_fao_latest['iso3'].isna()
    ca_fao_latest.loc[missing_iso3, 'iso3'] = ca_fao_latest.loc[missing_iso3, 'country_code_m49'].map(m49_to_iso3)
except Exception:
    pass

ca_fao_latest['map_id'] = ca_fao_latest['iso3'].map(iso3_to_map_id)
ca_fao_latest = ca_fao_latest.dropna(subset=['map_id'])

country_thresholds = {}
unique_map_ids = np.unique(all_alloc)

# 🚀 THE FIX: Isolate and solve one country at a time
for map_id in unique_map_ids:
    country_mask = all_alloc == map_id
    c_logit = all_logit[country_mask]
    c_pot = all_pot_ca[country_mask]
    
    target_area = 0
    match = ca_fao_latest[ca_fao_latest['map_id'] == map_id]
    if not match.empty:
        target_area = match['Value'].values[0] * 1000 
            
    if target_area == 0:
        target_area = c_pot.sum() * 0.10 
        
    df_country = pd.DataFrame({'logit': c_logit, 'pot_ca': c_pot})
    df_country = df_country.sort_values(by='logit', ascending=False)
    cumsum = df_country['pot_ca'].cumsum()
    cutoff_idx = cumsum.searchsorted(target_area)
    
    if cutoff_idx >= len(df_country):
        country_thresholds[map_id] = df_country['logit'].min() 
    else:
        country_thresholds[map_id] = df_country.iloc[cutoff_idx]['logit']

    del country_mask, c_logit, c_pot, df_country
    
print(f"✓ Solved exact thresholds for {len(country_thresholds)} global territories.")
del all_alloc, all_logit, all_pot_ca
gc.collect()

# ── 5. PASS 2: Allocate & Write 22-Band Output ────────────────────────────────
print(f"\n[STEP 3]: Allocating 22-Band CA to {total_blocks} chunks...")
src_logit_fixed = rasterio.open(path_global / "logit_global_fixed.tif")

out_ca = rasterio.open(path_global / "Conservation_Agriculture.tif", 'w', **profile_22band)
out_scenario = rasterio.open(path_global / "scenario_ca_area.tif", 'w', **profile_22band)
out_pot_ca = rasterio.open(path_global / "pot_ca_area.tif", 'w', **profile_22band)

with rasterio.Env(GDAL_CACHEMAX=500000000, GDAL_NUM_THREADS='2'):
    for idx, (ji, window) in enumerate(blocks):
        if idx % 100 == 0:
            print(f"  Writing Block {idx}/{total_blocks}...", end='\r', flush=True)

        logit = src_logit_fixed.read(1, window=window)
        if np.all(np.isnan(logit)):
            empty_22 = np.full((22, window.height, window.width), np.nan, dtype=np.float32)
            out_ca.write(empty_22, window=window)
            out_scenario.write(empty_22, window=window)
            out_pot_ca.write(empty_22, window=window)
            continue

        grains = src_grains.read(window=window)
        alloc  = src_alloc.read(1, window=window)
        
        grains[np.isnan(grains)] = 0.0
        grains[grains < 0] = 0.0
        
        pixel_sum = np.sum(grains, axis=0)
        overflow_mask = pixel_sum > 6.25
        if np.any(overflow_mask):
            scale = np.ones_like(pixel_sum)
            scale[overflow_mask] = 6.25 / pixel_sum[overflow_mask]
            grains = grains * scale
            
        grains[grains == 0] = np.nan
        
        ca_out = np.zeros_like(grains)
        
        for map_id, cutoff in country_thresholds.items():
            winning_pixels = (alloc == map_id) & (logit >= cutoff)
            if np.any(winning_pixels):
                ca_out[:, winning_pixels] = grains[:, winning_pixels]
                
        ca_out[np.isnan(grains)] = np.nan
        
        remaining_potential = grains - ca_out
        remaining_potential[remaining_potential < 0] = 0
        scenario_out = ca_out + (remaining_potential * 0.5)
        
        out_ca.write(ca_out, window=window)
        out_scenario.write(scenario_out, window=window)
        out_pot_ca.write(grains, window=window) 
        
        del logit, grains, alloc, ca_out, scenario_out
        if idx % 50 == 0: gc.collect()

src_logit_fixed.close()
for src in [src_crop_mix, src_grains, src_arid, src_fields, src_alloc, out_ca, out_scenario, out_pot_ca]: src.close()

print("\n\n✓ GLOBAL CA ALLOCATION COMPLETE!")