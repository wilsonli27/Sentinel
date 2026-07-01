"""
MODULE 4: TILLAGE CLASSIFICATION (SPARSE TIFF OPTIMIZED)
Systematically allocates physical crop area into 5 distinct tillage practices.
Includes RAM limits, fixed Dasymetric scanners, and Sparse TIFF optimization 
to drastically reduce file sizes and skip oceans instantly.
"""

import numpy as np
import rasterio
from pathlib import Path
import warnings
import gc
import time

warnings.filterwarnings("ignore")

# ── 1. Setup & Paths ──────────────────────────────────────────────────────────
path_global = Path(r"D:\Users\Wilson\Downloads\Sentinel\mod1_global_output_250m")

print("=" * 60)
print("MODULE 4: GLOBAL TILLAGE CLASSIFICATION (SPARSE TIFF EDITION)")
print("=" * 60)

# ── INDICES (Strict Porwollik Mapping) ──
annual_idx = [0, 1, 2, 3, 4, 6, 12, 13, 16] + list(range(23, 42)) + [22] # 29 crops
perm_idx = [5, 7, 8, 9, 10, 11, 14, 15, 17, 18, 19, 20, 21]              # 13 crops
grain_indices = [0, 2, 3, 4, 5, 6, 8] + list(range(14, 29))              # 22 CA crops

SMALL_FIELD_THRESHOLD = 2.5

# ── 2. Open Master Files & Copy Profiles ──────────────────────────────────────
try:
    src_ta     = rasterio.open(path_global / "SPAM_TA_Disaggregated.tif")
    src_ca     = rasterio.open(path_global / "Conservation_Agriculture.tif")
    src_flat   = rasterio.open(path_global / "flat_area.tif")
    src_fields = rasterio.open(path_global / "fields_interpol.tif")
    src_high   = rasterio.open(path_global / "high_raster.tif")
    src_soil   = rasterio.open(path_global / "soilgrid_res.tif")
except Exception as e:
    print(f"❌ Error locating input datasets: {e}")
    exit()

# 🚀 SPARSE TIFF UPGRADE: Force nodata=np.nan so unwritten blocks default to empty space
prof_42 = src_ta.profile.copy()
prof_42.update(BIGTIFF="YES", compress="deflate", tiled=True, nodata=np.nan)

prof_29 = src_ta.profile.copy()
prof_29.update(count=len(annual_idx), BIGTIFF="YES", compress="deflate", tiled=True, nodata=np.nan)

prof_13 = src_ta.profile.copy()
prof_13.update(count=len(perm_idx), BIGTIFF="YES", compress="deflate", tiled=True, nodata=np.nan)

# ── 3. Open Output Writers ────────────────────────────────────────────────────
out_trad_ann = rasterio.open(path_global / "traditional_annual_tillage.tif", 'w', **prof_29)
out_trad_rot = rasterio.open(path_global / "traditional_rotational_tillage.tif", 'w', **prof_13)
out_reduced  = rasterio.open(path_global / "reduced_tillage.tif", 'w', **prof_42)
out_rot      = rasterio.open(path_global / "rotational_tillage.tif", 'w', **prof_13)
out_conv     = rasterio.open(path_global / "conventional_annual_tillage.tif", 'w', **prof_42)

blocks = list(src_ta.block_windows(1))
total_blocks = len(blocks)

print(f"Processing {total_blocks} global chunks through the Classification Engine...\n")

skipped_empty = 0
start_time = time.time()

# ── 4. Block-by-Block Processing Engine ───────────────────────────────────────
with rasterio.Env(GDAL_CACHEMAX=500000000, GDAL_NUM_THREADS='2'):
    for idx, (ji, window) in enumerate(blocks):
        
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Classified {idx}/{total_blocks} | Empty Blocks Skipped: {skipped_empty} | Run Time: {elapsed:.1f}s", end='\r', flush=True)

        ta = src_ta.read(window=window)
        
        # 🚀 THE SPEED FIX: If empty, do absolutely nothing. GDAL will leave a sparse hole.
        if np.all(np.isnan(ta)) or np.nanmax(ta) == 0:
            skipped_empty += 1
            del ta
            continue

        ta[np.isnan(ta)] = 0.0
        ta[ta < 0] = 0.0

        ca_data = src_ca.read(window=window)
        flat    = src_flat.read(1, window=window)
        fields  = src_fields.read(1, window=window)
        high    = src_high.read(1, window=window)
        soil    = src_soil.read(1, window=window)

        # ── MASKS ──
        flat_mask = ~np.isnan(flat)
        ta_flat_out = ta.copy()
        ta_flat_out[:, flat_mask] = np.nan

        small_mask = fields < SMALL_FIELD_THRESHOLD
        large_mask = fields >= SMALL_FIELD_THRESHOLD
        high_mask  = ~np.isnan(high)

        # ── A. TRADITIONAL ANNUAL (29 bands) ──
        valid_trad_ann = small_mask & ~high_mask
        trad_ann = ta_flat_out[annual_idx].copy()
        trad_ann[:, ~valid_trad_ann] = np.nan
        out_trad_ann.write(trad_ann, window=window)

        # ── B. TRADITIONAL ROTATIONAL (13 bands) ──
        trad_rot = ta_flat_out[perm_idx].copy()
        trad_rot[:, ~valid_trad_ann] = np.nan
        trad_rot[trad_rot == 0] = np.nan
        out_trad_rot.write(trad_rot, window=window)

        # ── C. INITIAL REDUCED (42 bands) ──
        pre_reduced = ta.copy()
        pre_reduced[:, ~flat_mask] = np.nan

        # ── D. ROTATIONAL (13 bands) ──
        perms = ta_flat_out[perm_idx].copy()
        
        perm_large = perms.copy()
        perm_large[:, ~large_mask] = np.nan
        
        perm_small_high = perms.copy()
        valid_small_high = small_mask & high_mask
        perm_small_high[:, ~valid_small_high] = np.nan
        
        pre_rot = np.zeros_like(perm_large)
        for i in range(len(perm_idx)):
            comb = np.stack([perm_large[i], perm_small_high[i]])
            pre_rot[i] = np.nansum(comb, axis=0)
            both_nan = np.isnan(perm_large[i]) & np.isnan(perm_small_high[i])
            pre_rot[i][both_nan] = np.nan

        flat_15_20 = (soil >= 15) & (soil < 20)
        
        rot = pre_rot.copy()
        rot[:, flat_15_20] = np.nan
        rot[rot == 0] = np.nan
        out_rot.write(rot, window=window)

        redu_rot = pre_rot.copy()
        redu_rot[:, ~flat_15_20] = np.nan

        del perms, perm_large, perm_small_high, pre_rot

        # ── E. CONVENTIONAL ANNUAL (42 bands) ──
        pre_conv = ta.copy()
        
        for i in range(42):
            sub_amt = np.zeros((window.height, window.width), dtype=np.float32)
            
            if i in perm_idx:
                idx_p = perm_idx.index(i)
                sub_amt += np.nan_to_num(trad_rot[idx_p]) + np.nan_to_num(rot[idx_p])
                
            if i in annual_idx:
                idx_a = annual_idx.index(i)
                sub_amt += np.nan_to_num(trad_ann[idx_a])
                
            sub_amt += np.nan_to_num(pre_reduced[i])
            
            if i in grain_indices:
                idx_ca = grain_indices.index(i)
                sub_amt += np.nan_to_num(ca_data[idx_ca])
                
            pre_conv[i] -= sub_amt

        pre_conv[pre_conv <= 0] = np.nan 
        
        mech_shallow = pre_conv.copy()
        mech_shallow[:, ~flat_15_20] = np.nan
        
        conv_final = pre_conv.copy()
        conv_final[:, flat_15_20] = np.nan
        out_conv.write(conv_final, window=window)

        # ── F. FINAL REDUCED (42 bands) ──
        redu_rot_42 = np.zeros_like(ta)
        redu_rot_42[perm_idx] = redu_rot
        
        reduced_final = np.zeros_like(ta)
        for i in range(42):
            comb = np.stack([
                np.nan_to_num(pre_reduced[i]), 
                np.nan_to_num(mech_shallow[i]), 
                np.nan_to_num(redu_rot_42[i])
            ])
            reduced_final[i] = np.sum(comb, axis=0)
            all_nan = np.isnan(pre_reduced[i]) & np.isnan(mech_shallow[i]) & np.isnan(redu_rot_42[i])
            reduced_final[i][all_nan] = np.nan
            reduced_final[i][reduced_final[i] == 0] = np.nan
            
        out_reduced.write(reduced_final, window=window)

        # 🚀 MANUAL GARBAGE COLLECTION
        del flat, ta, ca_data, fields, high, soil, flat_mask, ta_flat_out
        del small_mask, large_mask, high_mask, valid_trad_ann
        del trad_ann, trad_rot, pre_reduced, valid_small_high
        del flat_15_20, rot, redu_rot, pre_conv, mech_shallow, conv_final
        del redu_rot_42, reduced_final, comb, all_nan
        
        if idx % 50 == 0:
            gc.collect()

# ── 5. Cleanup ────────────────────────────────────────────────────────────────
for src in [src_ta, src_ca, src_flat, src_fields, src_high, src_soil]:
    src.close()

out_trad_ann.close()
out_trad_rot.close()
out_reduced.close()
out_rot.close()
out_conv.close()

print(f"\n\n✓ GLOBAL TILLAGE CLASSIFICATION COMPLETE in {(time.time() - start_time)/60:.1f} minutes!")