"""
Data Harmonization Script - FIXED VERSION
Prevents vertical squashing by clipping to extent before resampling.
Switched from Bilinear to Nearest resampling to preserve absolute hectare area.
"""

import numpy as np
import rasterio
from rasterio.warp import Resampling
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.features import rasterize
from rasterio.transform import from_origin
import geopandas as gpd
import pandas as pd
from pathlib import Path
from scipy.ndimage import distance_transform_edt

#### 1. Setup ####
path_input = Path("D:/Users/Wilson/Downloads/Sentinel/")
path_calc = Path("sample_output/")
path_calc.mkdir(exist_ok=True, parents=True)

# Define target grid extent and resolution
final_ext = {'xmin': -180, 'xmax': 180, 'ymin': -56, 'ymax': 84}
resolution = 1/12  # 0.083333...°
nx = int((final_ext['xmax'] - final_ext['xmin']) / resolution)
ny = int((final_ext['ymax'] - final_ext['ymin']) / resolution)

print("="*60)
print("DATA HARMONIZATION - ALIGNMENT FIXED")
print("="*60)
print(f"Target Grid: {nx} x {ny} cells")
print(f"Extent: {final_ext['ymin']} to {final_ext['ymax']} Latitude")
print("="*60)

def read_clipped(src_path, resampling_method=Resampling.nearest):
    """Helper function to clip to extent and resample, with a fix for unreferenced files."""
    try:
        with rasterio.open(src_path) as src:
            transform = src.transform
            
            # FIX: If the file is not georeferenced (Identity transform), 
            # assume it's a standard global -180 to 180, -90 to 90 map
            if transform == rasterio.Affine.identity():
                pixel_width = 360.0 / src.width
                pixel_height = 180.0 / src.height
                transform = from_origin(-180.0, 90.0, pixel_width, pixel_height)

            # Use from_bounds instead of src.window to allow passing the transform explicitly
            window = from_bounds(
                final_ext['xmin'], final_ext['ymin'], 
                final_ext['xmax'], final_ext['ymax'], 
                transform=transform
            )
            
            # Read and resample
            data = src.read(
                1, 
                window=window, 
                out_shape=(ny, nx), 
                resampling=resampling_method
            ).astype(np.float32)
            
            return data
            
    except Exception as e:
        print(f"  ❌ Error reading {src_path.name}: {e}")
        return np.full((ny, nx), np.nan, dtype=np.float32)

#### 2. Load SPAM2020 cropland ####
print("\n=== Loading SPAM2020 cropland (Clipped) ===")
spam_folder = path_input / "spam2020v2r0_global_physical_area.geotiff" / "spam2020v2r0_global_physical_area"

crop_names = ['WHEA', 'RICE', 'MAIZ', 'BARL', 'REST', 'OOIL', 'TOBA', 'TEAS', 
              'COCO', 'RCOF', 'ACOF', 'OFIB', 'COTT', 'SUGB', 'SUGC', 'OILP', 
              'VEGE', 'TEMF', 'TROF', 'PLNT', 'BANA', 'CNUT', 'GROU', 'OTRS', 
              'CASS', 'YAMS', 'SWPO', 'POTA', 'SESA', 'RAPE', 'SUNF', 'SOYB', 
              'OPUL', 'LENT', 'PIGE', 'COWP', 'CHIC', 'BEAN', 'OCER', 'SORG', 
              'SMIL', 'PMIL']

SPAM_TA = np.zeros((42, ny, nx), dtype=np.float32)
SPAM_TR = np.zeros((42, ny, nx), dtype=np.float32)

for i, crop in enumerate(crop_names):
    ta_file = spam_folder / f"spam2020_V2r0_global_A_{crop}_A.tif"
    tr_file = spam_folder / f"spam2020_V2r0_global_A_{crop}_R.tif"
    
    # FIX: Changed from Resampling.bilinear to Resampling.nearest for physical area
    if ta_file.exists():
        SPAM_TA[i] = read_clipped(ta_file, Resampling.nearest)
    if tr_file.exists():
        SPAM_TR[i] = read_clipped(tr_file, Resampling.nearest)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/42 crops")

np.save(path_calc / "SPAM_TA.npy", SPAM_TA)
np.save(path_calc / "SPAM_TR.npy", SPAM_TR)
print(f"✓ SPAM Aligned: {SPAM_TA.shape}")

#### 3. Load SoilGrids ####
print("\n=== Loading SoilGrids (Clipped) ===")
soil_file = path_input / "BDTICM_M_250m_ll.tif"

# Using Resampling.average for downscaling high-res soil data
soilgrid_res = read_clipped(soil_file, Resampling.average)
soilgrid_res[soilgrid_res < 0] = np.nan

# Create flat area mask (depth < 15 cm)
flat_area = np.where(soilgrid_res < 15, soilgrid_res, np.nan)

np.save(path_calc / "soilgrid_res.npy", soilgrid_res)
np.save(path_calc / "flat_area.npy", flat_area)
print(f"✓ Soil Aligned: Range {np.nanmin(soilgrid_res):.0f}-{np.nanmax(soilgrid_res):.0f} cm")

#### 4. Load Field Sizes ####
print("\n=== Loading Field Sizes (Clipped) ===")
field_file = path_input / "lesiv_2018_field_sizes" / "Global Field Sizes" / "dominant_field_size_categories.tif"

# Use Nearest Neighbor for categorical data
fields_raw = read_clipped(field_file, Resampling.nearest)

FIELD_MAPPING = {3502: 1, 3503: 2, 3504: 3, 3505: 4, 3506: 5}
fields_interpol = np.zeros_like(fields_raw)
for old_val, new_val in FIELD_MAPPING.items():
    fields_interpol[fields_raw == old_val] = new_val

fields_interpol[fields_interpol == 0] = np.nan

# Gap filling
mask = np.isnan(fields_interpol)
if np.any(mask):
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    fields_interpol[mask] = fields_interpol[tuple(indices[:, mask])]

np.save(path_calc / "fields_interpol.npy", fields_interpol)
print(f"✓ Fields Aligned and Gap-filled")

#### 5. Load Erosion ####
print("\n=== Loading Erosion (Clipped) ===")
erosion_file = path_input / "Data2_SOIL_DISPLACEMENT_ESTIMATE_2019_1" / "Data2_SOIL_DISPLACEMENT_ESTIMATE_2019_1" / "SOIL_DISPLACEMENT_ESTIMATE_2019.tif"

if not erosion_file.exists():
    erosion_file_ovr = erosion_file.with_suffix('.tif.ovr')
    if erosion_file_ovr.exists():
        print("  ⚠️ Main .tif not found, using .ovr (Warning: .ovr files often lack geodata)")
        erosion_file = erosion_file_ovr

gladis_hdr = read_clipped(erosion_file, Resampling.average)

gladis_hdr[gladis_hdr < 0] = np.nan
gladis_hdr[gladis_hdr > 12000] = np.nan

np.save(path_calc / "gladis_hdr.npy", gladis_hdr)
print(f"✓ Erosion Aligned")

#### 6. Load Aridity ####
print("\n=== Loading Aridity (Clipped) ===")
ari_file = path_input / "Global-AI_ET0_annual_v3" / "Global-AI_ET0_v3_annual" / "ai_v3_yr.tif"

aridity_res = read_clipped(ari_file, Resampling.bilinear)
if np.nanmax(aridity_res) > 100:
    aridity_res = aridity_res / 10000.0

aridity_res[aridity_res < 0] = np.nan
np.save(path_calc / "aridity_res.npy", aridity_res)
print(f"✓ Aridity Aligned")

#### 7. Country Allocation ####
print("\n=== Generating Country Raster ===")
prepared_countries = path_input / "output" / "country_allocation_fixed.gpkg"

if not prepared_countries.exists():
    alloc_rast = np.full((ny, nx), 1, dtype=np.float32)
    high_raster = np.full((ny, nx), 400, dtype=np.float32)
else:
    gdf = gpd.read_file(prepared_countries, layer='countries')
    transform = transform_from_bounds(final_ext['xmin'], final_ext['ymin'],
                                      final_ext['xmax'], final_ext['ymax'], nx, ny)
    
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['country_code']))
    alloc_rast = rasterize(shapes=shapes, out_shape=(ny, nx), transform=transform, fill=0, dtype=np.float32)
    alloc_rast[alloc_rast == 0] = np.nan
    
    mapping_file = path_input / "output" / "country_code_mapping.csv"
    if mapping_file.exists():
        country_mapping = pd.read_csv(mapping_file)
        high_income_codes = country_mapping[country_mapping['income_code'] >= 3]['country_code'].values
        high_raster = np.full_like(alloc_rast, np.nan)
        for code in high_income_codes:
            high_raster[alloc_rast == code] = 400
    else:
        high_raster = np.full_like(alloc_rast, 400)

np.save(path_calc / "spam_alloc_rast.npy", alloc_rast)
np.save(path_calc / "high_raster.npy", high_raster)

print("\n" + "="*60)
print("HARMONIZATION COMPLETE")
print("="*60)