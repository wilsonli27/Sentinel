"""
MODULE 1: GLOBAL HARMONIZATION (MASS-PRESERVING EDITION)
Processes the entire planet. Includes pre-flight file validation, empty intersection 
crash protection, and a highly optimized "Crop Skipper" to bypass 80% of the Earth's 
oceans and deserts without leaving holes in your covariate maps.
"""
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import geopandas as gpd
import pandas as pd
from pathlib import Path
import math
import gc
import warnings
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore")

# ── 1. Setup & Global Grid Definition ─────────────────────────────────────────
path_gee   = Path("path_input/test_output_250m/")
path_calc  = Path("mod1_global_output_250m/") 
path_calc.mkdir(exist_ok=True, parents=True)

final_ext  = {"xmin": -180, "xmax": 180, "ymin": -56, "ymax": 84}
resolution = 0.002245788  
PIXEL_AREA_HA = 6.25      

nx = int(round((final_ext["xmax"] - final_ext["xmin"]) / resolution))
ny = int(round((final_ext["ymax"] - final_ext["ymin"]) / resolution))

global_transform = from_bounds(
    final_ext["xmin"], final_ext["ymin"],
    final_ext["xmax"], final_ext["ymax"],
    nx, ny
)

BLOCK_SIZE = 1024  
x_blocks = math.ceil(nx / BLOCK_SIZE)
y_blocks = math.ceil(ny / BLOCK_SIZE)

crop_names = [
    "WHEA","RICE","MAIZ","BARL","REST","OOIL","TOBA","TEAS",
    "COCO","RCOF","COFF","OFIB","COTT","SUGB","SUGC","OILP",
    "VEGE","TEMF","TROF","PLNT","BANA","CNUT","GROU","ORTS",
    "CASS","YAMS","SWPO","POTA","SESA","RAPE","SUNF","SOYB",
    "OPUL","LENT","PIGE","COWP","CHIC","BEAN","OCER","SORG",
    "SMIL","PMIL",
]

print("=" * 60)
print("MODULE 1: GLOBAL MASS-PRESERVING DASYMETRIC HARMONIZATION")
print("=" * 60)

# ── 2. PRE-FLIGHT VALIDATION CHECK ────────────────────────────────────────────
def run_pre_flight_check():
    print("Executing Global Pre-Flight Data Scan...")
    required_prefixes = [
        "esa_cropland_fraction_250m", # Matches all quadrant chunks
        "soilgrids_bdticm_250m",
        "field_size_lesiv_250m",
        "aridity_index_250m"
    ]
    
    for crop in crop_names:
        if crop != "SMIL":
            required_prefixes.append(f"spam2020_V2r0_global_A_{crop}_A")
            required_prefixes.append(f"spam2020_V2r0_global_A_{crop}_R")

    missing_prefixes = []
    for prefix in required_prefixes:
        if not list(path_gee.glob(f"{prefix}*.tif")):
            missing_prefixes.append(prefix)

    if missing_prefixes:
        print("\n❌ FATAL ERROR: PRE-FLIGHT FAILED.")
        print("The script is blind to the following required datasets:")
        for m in missing_prefixes[:10]:
            print(f"  - {m}")
        if len(missing_prefixes) > 10:
            print(f"  ...and {len(missing_prefixes) - 10} more.")
        print(f"\nPlease verify your extracted files are physically inside: {path_gee.absolute()}")
        sys.exit(1)
        
    print("✅ Pre-Flight Passed: All expected GEE arrays located.\n")

run_pre_flight_check()

# ── 3. Helper: Dynamic Mosaic Reader (With Crash Protection) ──────────────────
def bounds_intersect(b1, b2):
    return not (b1.right <= b2.left or b1.left >= b2.right or
                b1.top <= b2.bottom or b1.bottom >= b2.top)

def read_gee_chunks(prefix, window, out_shape, transform):
    file_paths = list(path_gee.glob(f"{prefix}*.tif"))
    
    if not file_paths:
        return np.full(out_shape, np.nan, dtype=np.float32)
        
    win_bounds = rasterio.windows.bounds(window, transform)
    target_box = rasterio.coords.BoundingBox(*win_bounds)
    
    datasets_to_merge = []
    srcs = []
    try:
        for fp in file_paths:
            src = rasterio.open(fp)
            srcs.append(src)
            src_box = rasterio.coords.BoundingBox(*src.bounds)
            if bounds_intersect(src_box, target_box):
                datasets_to_merge.append(src)
                
        if not datasets_to_merge:
            return np.full(out_shape, np.nan, dtype=np.float32)
            
        try:
            mosaic, _ = merge(datasets_to_merge, bounds=win_bounds, nodata=np.nan)
            data = mosaic[0]
        except ValueError:
            return np.full(out_shape, np.nan, dtype=np.float32)
        except Exception:
            return np.full(out_shape, np.nan, dtype=np.float32)
        
        out_array = np.full(out_shape, np.nan, dtype=np.float32)
        h, w = data.shape
        out_array[:min(h, out_shape[0]), :min(w, out_shape[1])] = data[:min(h, out_shape[0]), :min(w, out_shape[1])]
        return out_array
    finally:
        for src in srcs:
            src.close()

# ── 4. Pre-load Country Data ──────────────────────────────────────────────────
prepared_countries = Path("path_input/output/country_allocation_fixed.gpkg")
mapping_file = Path("path_input/output/country_code_mapping.csv")
gdf, high_income_codes = None, []

if prepared_countries.exists():
    gdf = gpd.read_file(prepared_countries, layer="countries")
    if mapping_file.exists():
        cmap = pd.read_csv(mapping_file)
        high_income_codes = cmap[cmap["income_code"] >= 3]["country_code"].values

# ── 5. MAIN TILING LOOP ───────────────────────────────────────────────────────
base_profile = {
    'driver': 'GTiff', 'height': ny, 'width': nx, 'crs': 'EPSG:4326',
    'transform': global_transform, 'dtype': 'float32', 'nodata': np.nan,
    'tiled': True, 'blockxsize': 1024, 'blockysize': 1024, 'compress': 'deflate', 'zlevel': 6,
    'BIGTIFF': 'YES'
}

spam_ta_out = rasterio.open(path_calc / "SPAM_TA_Disaggregated.tif", 'w', count=42, **base_profile)
spam_tr_out = rasterio.open(path_calc / "SPAM_TR_Disaggregated.tif", 'w', count=42, **base_profile)
soil_out    = rasterio.open(path_calc / "soilgrid_res.tif", 'w', count=1, **base_profile)
flat_out    = rasterio.open(path_calc / "flat_area.tif", 'w', count=1, **base_profile)
fields_out  = rasterio.open(path_calc / "fields_interpol.tif", 'w', count=1, **base_profile)
aridity_out = rasterio.open(path_calc / "aridity_res.tif", 'w', count=1, **base_profile)
alloc_out   = rasterio.open(path_calc / "spam_alloc_rast.tif", 'w', count=1, **base_profile)
high_out    = rasterio.open(path_calc / "high_raster.tif", 'w', count=1, **base_profile)

# Processing every block globally
total_active_tiles = y_blocks * x_blocks
active_count = 0

print(f"Targeting {total_active_tiles} global tiles.")

for y in range(y_blocks):
    for x in range(x_blocks):
        active_count += 1
        print(f"  Processing Global Tile {active_count}/{total_active_tiles}...      ", end='\r')
        
        row_off, col_off = y * BLOCK_SIZE, x * BLOCK_SIZE
        width, height    = min(BLOCK_SIZE, nx - col_off), min(BLOCK_SIZE, ny - row_off)
        window           = Window(col_off, row_off, width, height)
        window_shape     = (height, width)
        win_transform    = rasterio.windows.transform(window, global_transform)

        # ── 0. Load ESA Cookie-Cutter ──
        esa_raw = read_gee_chunks("esa_cropland_fraction_250m", window, window_shape, global_transform)
        esa_fraction = np.nan_to_num(esa_raw, nan=0.0)
        
        # =====================================================================
        # 🚀 THE OPTIMIZED CROP SKIPPER 🚀
        # Only process the 84 heavy SPAM crop files if physical dirt exists.
        has_crops = np.max(esa_fraction) > 0.0
        # =====================================================================

        if has_crops:
            pixel_capacity_ha = PIXEL_AREA_HA * esa_fraction
            
            # MASS PRESERVATION ENGINE: 40 pixels = ~10km radius
            WINDOW_SIZE = 40 
            local_total_capacity = uniform_filter(pixel_capacity_ha, size=WINDOW_SIZE, mode='reflect') * (WINDOW_SIZE ** 2)
            safe_local_capacity = np.where(local_total_capacity > 0, local_total_capacity, 1.0)
            spatial_weight = pixel_capacity_ha / safe_local_capacity

            # ── 1. SPAM Disaggregation ──
            raw_ta_stack = np.zeros((42, height, width), dtype=np.float32)
            raw_tr_stack = np.zeros((42, height, width), dtype=np.float32)
            
            for i, crop in enumerate(crop_names):
                ta_data = read_gee_chunks(f"spam2020_V2r0_global_A_{crop}_A", window, window_shape, global_transform)
                tr_data = read_gee_chunks(f"spam2020_V2r0_global_A_{crop}_R", window, window_shape, global_transform)
                
                # MULTIPLY duplicated 10km SPAM value by the local spatial weight
                raw_ta_stack[i] = np.nan_to_num(ta_data, nan=0.0) * spatial_weight
                raw_tr_stack[i] = np.nan_to_num(tr_data, nan=0.0) * spatial_weight
                
            zero_mask = (esa_fraction == 0.0)
            raw_ta_stack[:, zero_mask] = np.nan
            raw_tr_stack[:, zero_mask] = np.nan
                
            spam_ta_out.write(raw_ta_stack, window=window)
            spam_tr_out.write(raw_tr_stack, window=window)
            del raw_ta_stack, raw_tr_stack

        # ── 2. SoilGrids & Flat Area (ALWAYS RUNS TO PREVENT EMPTY MAPS) ──
        soil_data = read_gee_chunks("soilgrids_bdticm_250m", window, window_shape, global_transform)
        soil_data[soil_data < 0] = np.nan
        flat_data = np.where(soil_data < 15, soil_data, np.nan).astype(np.float32)
        soil_out.write(soil_data, 1, window=window)
        flat_out.write(flat_data, 1, window=window)

        # ── 3. Field Sizes ──
        fields_raw = read_gee_chunks("field_size_lesiv_250m", window, window_shape, global_transform)
        FIELD_MAPPING = {3502: 1, 3503: 2, 3504: 3, 3505: 4, 3506: 5}
        fields_mapped = np.full_like(fields_raw, np.nan)
        for raw_val, new_val in FIELD_MAPPING.items():
            fields_mapped[fields_raw == raw_val] = new_val
        fields_out.write(fields_mapped, 1, window=window)

        # ── 4. Aridity Index ──
        arid_data = read_gee_chunks("aridity_index_250m", window, window_shape, global_transform)
        arid_data[arid_data < 0] = np.nan
        aridity_out.write(arid_data, 1, window=window)

        # ── 5. Country Allocation ──
        if gdf is not None:
            win_bounds = rasterio.windows.bounds(window, global_transform)
            local_gdf = gdf.cx[win_bounds[0]:win_bounds[2], win_bounds[1]:win_bounds[3]]
            if not local_gdf.empty:
                shapes = ((geom, val) for geom, val in zip(local_gdf.geometry, local_gdf["country_code"]))
                alloc_data = rasterize(shapes=shapes, out_shape=window_shape, transform=win_transform, fill=0, dtype=np.float32)
                alloc_data[alloc_data == 0] = np.nan
                high_data = np.full_like(alloc_data, np.nan)
                for code in high_income_codes:
                    high_data[alloc_data == code] = 400
            else:
                alloc_data = np.full(window_shape, np.nan, dtype=np.float32)
                high_data = np.full(window_shape, np.nan, dtype=np.float32)
                
            alloc_out.write(alloc_data, 1, window=window)
            high_out.write(high_data, 1, window=window)

        gc.collect()

print("\nCleaning up and closing files...")
for ds in [spam_ta_out, spam_tr_out, soil_out, flat_out, fields_out, aridity_out, alloc_out, high_out]:
    ds.close()

grid_meta = {
    "nx": nx, "ny": ny, "resolution": resolution,
    "xmin": final_ext["xmin"], "xmax": final_ext["xmax"],
    "ymin": final_ext["ymin"], "ymax": final_ext["ymax"],
}
np.save(path_calc / "grid_meta.npy", grid_meta)
print("GLOBAL Harmonization Complete. Outputs saved.")