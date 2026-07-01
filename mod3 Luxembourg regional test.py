"""
MOD 3 - REGIONAL TEST: LUXEMBOURG (ALL 4 OUTPUTS)
End-to-end Logit Pass 1 + Pass 2 Allocation.
Outputs the Logit map, CA, Scenario, and Potential CA TIFFs.
"""
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import pandas as pd
import geopandas as gpd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── 1. Setup & Paths ──────────────────────────────────────────────────────────
path_global = Path(r"D:\Users\Wilson\Downloads\Sentinel\mod1_global_output_250m")
path_input = path_global.parent  

faostat_file = path_input / "FAOSTAT_cons_ag.csv"
gpkg_path = path_input / "output" / "country_allocation_fixed.gpkg"

# Exact Bounding Box for Luxembourg
left, bottom, right, top = 5.5, 49.3, 6.7, 50.3

print("=" * 60)
print("MODULE 3 REGIONAL TEST: LUXEMBOURG (4 OUTPUTS)")
print("=" * 60)

# ── 2. Open Files & Extract Window ────────────────────────────────────────────
try:
    src_crop_mix = rasterio.open(path_global / "crop_mix.tif")
    src_grains   = rasterio.open(path_global / "grains_stack.tif")
    src_arid     = rasterio.open(path_global / "aridity_res.tif")
    src_fields   = rasterio.open(path_global / "fields_interpol.tif")
    src_alloc    = rasterio.open(path_global / "spam_alloc_rast.tif")
except Exception as e:
    print(f"❌ Error locating input files: {e}")
    exit()

target_window = from_bounds(left, bottom, right, top, src_crop_mix.transform)
win_transform = rasterio.windows.transform(target_window, src_crop_mix.transform)

print("Loading data for Luxembourg...")
crop_mix = src_crop_mix.read(1, window=target_window)
grains   = src_grains.read(window=target_window)
arid     = src_arid.read(1, window=target_window)
fields   = src_fields.read(1, window=target_window)
alloc    = src_alloc.read(1, window=target_window)

# ── 3. ID Bridge ──────────────────────────────────────────────────────────────
print("\nReading GPKG to match raster ID for Luxembourg...")
gdf_countries = gpd.read_file(gpkg_path, layer="countries", ignore_geometry=True)
iso3_to_map_id = dict(zip(gdf_countries['iso3'].str.upper(), gdf_countries['country_code'].astype(int)))
map_id_lux = iso3_to_map_id.get("LUX")

ca_fao = pd.read_csv(faostat_file)
lux_fao = ca_fao[ca_fao['Area'] == 'Luxembourg'].sort_values('Year').iloc[-1]
target_area_lux = lux_fao['Value'] * 1000  # Convert to standard hectares

print(f"  -> Luxembourg ISO3: LUX")
print(f"  -> Luxembourg Raster Map ID: {map_id_lux}")
print(f"  -> FAOSTAT Target CA Area: {target_area_lux:,.1f} ha")

# ── 4. PASS 1: Calculate Logit ────────────────────────────────────────────────
print("\nCalculating Logit Probabilities...")
kvalue = np.array([0.25, 1/60, -5, 10])
xmid   = np.array([20, 12, 0.65, 0.5])
logit_f = - (xmid[0]*kvalue[0]) - (xmid[1]*kvalue[1]) - (xmid[2]*kvalue[2]) - (xmid[3]*kvalue[3])

b = (kvalue[0]*fields) + (kvalue[1]*12) + (kvalue[2]*arid) + (kvalue[3]*crop_mix)
logit = 1 / (1 + np.exp(-(b + logit_f)))
logit[np.isnan(crop_mix)] = np.nan

# ── 5. SANITIZER & SOLVER ─────────────────────────────────────────────────────
print("\nSolving Thresholds...")
invalid_mask = (grains > 6.25) | (grains < 0)
grains[invalid_mask] = np.nan

valid_mask = ~np.isnan(logit) & (alloc == map_id_lux)
pot_ca_lux = np.nansum(grains[:, valid_mask], axis=0) 
logit_lux = logit[valid_mask]

if len(logit_lux) == 0:
    print("❌ ERROR: No valid cropland pixels found for Luxembourg inside this bounding box.")
    exit()

df_pixels = pd.DataFrame({'logit': logit_lux, 'pot_ca': pot_ca_lux})
df_pixels = df_pixels.sort_values(by='logit', ascending=False)
df_pixels['cumsum'] = df_pixels['pot_ca'].cumsum()

cutoff_idx = df_pixels['cumsum'].searchsorted(target_area_lux)
if cutoff_idx >= len(df_pixels):
    cutoff = df_pixels['logit'].min()
else:
    cutoff = df_pixels.iloc[cutoff_idx]['logit']

print(f"  -> Calculated Logit Cutoff: {cutoff:.6f}")

# ── 6. PASS 2: Allocate CA, Scenario, and Potential CA ────────────────────────
print("\nAllocating All Layers...")
ca_out = np.zeros_like(grains)
winning_pixels = (alloc == map_id_lux) & (logit >= cutoff)

if np.any(winning_pixels):
    ca_out[:, winning_pixels] = grains[:, winning_pixels]
ca_out[np.isnan(grains)] = np.nan

# Calculate Scenario Layer
remaining_potential = grains - ca_out
remaining_potential[remaining_potential < 0] = 0
scenario_out = ca_out + (remaining_potential * 0.5)

total_assigned_pixels = np.sum(winning_pixels)
total_allocated_ha = np.nansum(ca_out)

print("\n" + "=" * 60)
print(f"⭐ REGIONAL PROOF OF CONCEPT RESULTS ⭐")
print(f"Target CA Quota          : {target_area_lux:,.1f} ha")
print(f"Actual Allocated CA      : {total_allocated_ha:,.1f} ha")
print(f"Total Assigned CA Pixels : {total_assigned_pixels:,}")
print(f"Max Hectares in 1 Pixel  : {np.nanmax(ca_out):.4f} ha")
print("=" * 60)

# ── 7. Write All 4 Outputs to Disk ────────────────────────────────────────────

# Create 22-band metadata for crops
out_meta_22band = src_grains.profile.copy()
out_meta_22band.update({
    "height": target_window.height,
    "width": target_window.width,
    "transform": win_transform
})

# Create 1-band metadata for the Logit
out_meta_1band = src_crop_mix.profile.copy()
out_meta_1band.update({
    "height": target_window.height,
    "width": target_window.width,
    "transform": win_transform
})

# Write the new Logit map
with rasterio.open(path_global / "lux_logit_ref.tif", "w", **out_meta_1band) as dest:
    dest.write(logit, 1)

# Write the 3 crop maps
with rasterio.open(path_global / "lux_Conservation_Agriculture.tif", "w", **out_meta_22band) as dest:
    dest.write(ca_out)
    
with rasterio.open(path_global / "lux_scenario_ca_area.tif", "w", **out_meta_22band) as dest:
    dest.write(scenario_out)
    
with rasterio.open(path_global / "lux_pot_ca_area.tif", "w", **out_meta_22band) as dest:
    dest.write(grains)

print(f"Saved all 4 regional files to {path_global}!")