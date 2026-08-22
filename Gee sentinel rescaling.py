import ee

# 1. Initialize
print("Initialising Earth Engine...")
ee.Initialize(project='gvrsf-2026')
print("✓ Authenticated.")

GEE_ASSET_ROOT_SPAM = "projects/gvrsf-2026/assets/tillage_inputs"
SOILGRIDS_ASSET     = "projects/gvrsf-2026/assets/BDTICM_M_250m_ll"
ARIDITY_ASSET       = "projects/gvrsf-2026/assets/ai_v3_yr_fixed"
FIELD_SIZE_ASSET    = "projects/gvrsf-2026/assets/dominant_field_size_categories"

DRIVE_FOLDER   = "gee_outputs_250m"
EXPORT_SCALE_M = 250 
# Bounding box for the entire globe
REGION = ee.Geometry.Rectangle([-180, -56, 180, 84], 'EPSG:4326', False)

# --- RESCALING FUNCTIONS ---
def rescale_continuous(image):
    """For Aridity and SoilGrids: Bilinear interpolation for smooth continuous data."""
    projection = ee.Projection('EPSG:4326').atScale(EXPORT_SCALE_M)
    return image.resample('bilinear').reproject(crs=projection)

def rescale_categorical(image):
    """For Field Sizes: Modal reduction to prevent 'fake' interpolated classes."""
    projection = ee.Projection('EPSG:4326').atScale(EXPORT_SCALE_M)
    return image.reduceResolution(reducer=ee.Reducer.mode(), maxPixels=65536).reproject(crs=projection)

def rescale_area(image):
    """For SPAM physical area: Nearest neighbor (GEE Default) to preserve actual cell values."""
    projection = ee.Projection('EPSG:4326').atScale(EXPORT_SCALE_M)
    # Dropped .resample('near') — GEE defaults to nearest neighbor automatically
    return image.reproject(crs=projection)

def _export(img, name):
    task = ee.batch.Export.image.toDrive(
        image=img.toFloat(),
        description=f"export_{name}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        region=REGION,
        scale=EXPORT_SCALE_M,
        crs='EPSG:4326',
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"  ▶ Submitted: {name}")

# --- 1. SPAM Discovery & Export ---
print("\n=== Processing SPAM Collection ===")
try:
    spam_col = ee.ImageCollection(GEE_ASSET_ROOT_SPAM)
    assets = spam_col.toList(200).getInfo()
    print(f"Found {len(assets)} SPAM images.")
    
    for asset in assets:
        asset_id = asset['id']
        name = asset_id.split('/')[-1]
        img = ee.Image(asset_id)
        
        # Use the corrected area rescaling for SPAM
        rescaled_spam = rescale_area(img)
        _export(rescaled_spam, name)
except Exception as e:
    print(f"  ✗ SPAM Collection error: {e}")

# --- 2. SoilGrids (Custom Upload) ---
print("\n=== Processing SoilGrids ===")
try:
    bedrock = ee.Image(SOILGRIDS_ASSET).divide(10).rename("bdticm")
    rescaled_bedrock = rescale_continuous(bedrock)
    _export(rescaled_bedrock, "soilgrids_bdticm_250m")
except Exception as e:
    print(f"  ✗ SoilGrids error: {e}. Check {SOILGRIDS_ASSET}")

# --- 3. Aridity Index (Custom Upload) ---
print("\n=== Processing Aridity Index ===")
try:
    aridity = ee.Image(ARIDITY_ASSET).divide(10000).rename("aridity")
    rescaled_ai = rescale_continuous(aridity)
    _export(rescaled_ai, "aridity_index_250m")
except Exception as e:
    print(f"  ✗ Aridity error: {e}. Check {ARIDITY_ASSET}")

# --- 4. Field Sizes (Categorical Custom Upload) ---
print("\n=== Processing Field Sizes ===")
try:
    field_img = ee.Image(FIELD_SIZE_ASSET).rename("field_size")
    rescaled_fields = rescale_categorical(field_img)
    _export(rescaled_fields, "field_size_lesiv_250m")
except Exception as e:
    print(f"  ✗ Field Size error: {e}. Check {FIELD_SIZE_ASSET}")

print("\nAll tasks submitted! Head over to the Earth Engine Code Editor to monitor the Tasks tab.")