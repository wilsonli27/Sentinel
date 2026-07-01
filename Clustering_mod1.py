import ee
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
PROJECT_ID = 'gvrsf-2026'

try:
    ee.Initialize(project=PROJECT_ID)
    print(f"Successfully initialized GEE with project: {PROJECT_ID}")
except Exception as e:
    print(f"Initialization Error: {e}")
    print("Try running 'ee.Authenticate()' in your terminal first.")

# ASSET PATH
OUTPUT_ASSET_ID = f"projects/{PROJECT_ID}/assets/Tillage_Time_Markers"

# SETTINGS
YEAR = 2023
GRID_SIZE = 5.0
LAT_MIN, LAT_MAX = -55, 70
LON_MIN, LON_MAX = -180, 180

# WORKER SETTINGS:
# Keep at 3. Sumatra is heavy; more workers will just choke the bandwidth.
MAX_WORKERS = 3 

# CLUSTERING PARAMS
ARABLE_MIN_PERCENT = 1.0  
NDVI_MAX_THRESHOLD = 0.35 
CLUSTER_P1_K = 12
CLUSTER_P2_K = 4
OUTLIER_THRESHOLD = 0.03

# ==========================================
# 2. CORE LOGIC
# ==========================================

def retry_on_error(retries=3, delay=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            for i in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "Too many" in str(e) or "Internal" in str(e):
                        if i < retries:
                            time.sleep(current_delay + random.uniform(0, 2))
                            continue
                    raise e
            return func(*args, **kwargs)
        return wrapper
    return decorator

class TillageTimeIdentifier:
    def __init__(self):
        self.worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()
        self.cropland = self.worldcover.eq(40)

    @retry_on_error()
    def is_tile_arable(self, region):
        """Pre-check: Skip tiles that have no cropland."""
        stats = self.cropland.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5000, 
            maxPixels=1e8,
            tileScale=4
        ).getInfo()
        val = stats.get('Map')
        return (val if val else 0) * 100 > ARABLE_MIN_PERCENT

    def preprocess_s2(self, img):
        qa = img.select('cs_cdf')
        scl = img.select('SCL')
        mask = qa.gte(0.65).And(scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)))
        
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndti = img.normalizedDifference(['B11', 'B12']).rename('NDTI')
        
        date = ee.Date(img.get('system:time_start'))
        mjd_val = date.millis().divide(86400000).add(2440587.5).subtract(2400000.5)
        mjd = ee.Image.constant(mjd_val).rename('MJD').toFloat()
        neg_ndvi = ndvi.multiply(-1).rename('neg_NDVI')
        
        return img.updateMask(mask).addBands([ndvi, ndti, mjd, neg_ndvi])

    @retry_on_error()
    def get_derf_candidate(self, region):
        """Finds the single lowest NDVI moment per pixel."""
        # FIX FOR SUMATRA: Stricter cloud filter (< 60%).
        # This drastically reduces the number of images GEE has to scan in the tropics.
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)) \
            .linkCollection(ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'), ['cs_cdf']) \
            .map(self.preprocess_s2)
            
        if s2.size().getInfo() == 0:
            return None
            
        derf_pixel = s2.qualityMosaic('neg_NDVI')
        
        band_names = derf_pixel.bandNames().getInfo()
        if 'NDVI' not in band_names:
            return None

        valid_mask = self.cropland \
            .And(derf_pixel.select('NDVI').lt(NDVI_MAX_THRESHOLD)) \
            .And(derf_pixel.select('NDTI').mask())
            
        return derf_pixel.updateMask(valid_mask)

    @retry_on_error()
    def cluster_time_periods(self, derf_img, region):
        input_bands = derf_img.select(['MJD'])
        
        # 1. Fast Sample for clustering
        # tileScale 16 is crucial for handling large 5x5 degree tiles
        sample_p1 = input_bands.sample(
            region=region, scale=500, numPixels=5000, geometries=True, tileScale=16
        )
        
        if sample_p1.size().getInfo() < 100: return None
        
        # 2. Cluster the POINTS to get the histogram (Fast)
        clusterer_p1 = ee.Clusterer.wekaKMeans(CLUSTER_P1_K).train(sample_p1)
        
        clustered_samples = sample_p1.cluster(clusterer_p1)
        
        # Use aggregate_histogram on points (instant) instead of reduceRegion on image (slow)
        hist = clustered_samples.aggregate_histogram('cluster').getInfo()
        
        if not hist: return None
        
        total = sum(hist.values())
        valid_ids = [int(k) for k, v in hist.items() if (v/total) > OUTLIER_THRESHOLD]
        if len(valid_ids) < 2: return None
        
        # 3. Apply the rules to the Image
        result_p1 = input_bands.cluster(clusterer_p1)
        valid_mask = result_p1.remap(valid_ids, [1]*len(valid_ids), 0).eq(1)
        
        # --- Pass 2: Refined Seasonality ---
        clean_input = input_bands.updateMask(valid_mask)
        sample_p2 = clean_input.sample(
            region=region, scale=500, numPixels=5000, tileScale=16
        )
        
        if sample_p2.size().getInfo() < 10: return None
        
        clusterer_p2 = ee.Clusterer.wekaKMeans(CLUSTER_P2_K).train(sample_p2)
        final_clusters = clean_input.cluster(clusterer_p2).rename('cluster_id')
        
        return clean_input.addBands(final_clusters)

    def process_tile(self, coords):
        min_x, min_y, max_x, max_y = coords
        tile_id = f"Tile_X{int(min_x)}_Y{int(min_y)}"
        region = ee.Geometry.Rectangle(coords)
        
        # --- CRITICAL FIX: SKIP IF ASSET ALREADY EXISTS ---
        # This allows you to restart the script without re-processing finished tiles.
        asset_path = f"{OUTPUT_ASSET_ID}/{tile_id}"
        try:
            # If this call succeeds, the asset exists -> Skip it.
            ee.data.getAsset(asset_path)
            return f"Already Exists: {tile_id}"
        except:
            pass # Asset does not exist -> Proceed.
        # --------------------------------------------------

        try:
            # 1. Arable Check
            if not self.is_tile_arable(region):
                return f"Skipped: {tile_id} (No Cropland)"

            # 2. Get Derf Mosaic
            derf_img = self.get_derf_candidate(region)
            if derf_img is None:
                return f"Skipped: {tile_id} (No clear Sentinel-2 images)"

            # 3. Cluster
            result_img = self.cluster_time_periods(derf_img, region)
            
            if result_img is None:
                return f"Skipped: {tile_id} (Not enough bare soil pixels)"

            # 4. Export
            # FIX: Changed scale to 30m. 
            # 20m exports for 5x5 degree tiles near equator are too large and cause stalls.
            task = ee.batch.Export.image.toAsset(
                image=result_img.clip(region),
                description=f"TM_{tile_id}",
                assetId=asset_path,
                region=region,
                scale=30,         
                maxPixels=1e13     
            )
            task.start()
            return f"Started Export: {tile_id}"
            
        except Exception as e:
            return f"Error {tile_id}: {str(e)}"

# ==========================================
# 3. EXECUTION BLOCK
# ==========================================

def generate_global_grid():
    tiles = []
    for lat in range(LAT_MIN, LAT_MAX, int(GRID_SIZE)):
        for lon in range(LON_MIN, LON_MAX, int(GRID_SIZE)):
            tiles.append((lon, lat, lon + GRID_SIZE, lat + GRID_SIZE))
    return tiles

def main():
    print(f"{'='*50}")
    print("MODULE 1: IDENTIFYING DERF TIMEFRAMES (SUMATRA FIX)")
    print(f"{'='*50}")
    
    processor = TillageTimeIdentifier()
    grid = generate_global_grid()
    print(f"Generated {len(grid)} tiles to check.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(processor.process_tile, tile): tile for tile in grid}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            # We print everything that isn't a "Skip" to keep the log clean
            # If it's "Already Exists", we print it once every 50 tiles just to show life
            if "Skipped" in result:
                pass
            elif "Already Exists" in result:
                if i % 100 == 0: print(f"[{i+1}/{len(grid)}] Skipping completed tiles...")
            else:
                print(f"[{i+1}/{len(grid)}] {result}")

if __name__ == "__main__":
    main()