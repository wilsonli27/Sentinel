import ee
import time
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize Earth Engine
try:
    ee.Initialize(project='590577866979')
except Exception as e:
    print(f"Error: {e}")
    print("Please run ee.Authenticate() and ee.Initialize()")

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_FOLDER = "Global_Tillage_Clusters_v1"
YEAR = 2023

# Spatial Tiling Settings
GRID_SIZE = 5.0          
LAT_MIN, LAT_MAX = -55, 70 
LON_MIN, LON_MAX = -180, 180

# Arable Land Thresholds
ARABLE_MIN_PERCENT = 1.0  

# Clustering & Derf Settings
NDVI_MAX_THRESHOLD = 0.35 
CLUSTER_P1_K = 12         
CLUSTER_P2_K = 4          
OUTLIER_THRESHOLD = 0.03  

# Performance - DRAMATICALLY REDUCED
# GEE limits concurrent interactive requests. 
# 4 is usually the safe limit for heavy aggregations.
MAX_WORKERS = 4          

# ==========================================
# UTILITIES
# ==========================================
def retry_on_error(retries=3, delay=5, backoff=2):
    """
    Decorator to retry GEE operations that fail due to concurrency limits.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            for i in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    # Check for specific GEE errors that are transient
                    if "Too many concurrent" in error_msg or "Too many requests" in error_msg or "Internal" in error_msg:
                        if i < retries:
                            sleep_time = current_delay + random.uniform(0, 2) # Add jitter
                            # print(f"  > Hit limit. Retrying in {sleep_time:.1f}s...")
                            time.sleep(sleep_time)
                            current_delay *= backoff
                            continue
                    raise e
            return func(*args, **kwargs)
        return wrapper
    return decorator

class GlobalTillageProcessor:
    def __init__(self):
        self.worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()
        self.cropland = self.worldcover.eq(40)

    def mask_clouds_strict(self, img):
        qa = img.select('cs_cdf')
        scl = img.select('SCL')
        cloud_mask = qa.gte(0.65)
        scl_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        return img.updateMask(cloud_mask.And(scl_mask))

    def add_indices_and_time(self, img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndti = img.normalizedDifference(['B11', 'B12']).rename('NDTI')
        sti = img.select('B11').divide(img.select('B12')).rename('STI')
        
        date = ee.Date(img.get('system:time_start'))
        mjd_val = date.millis().divide(86400000).add(2440587.5).subtract(2400000.5)
        mjd = ee.Image.constant(mjd_val).rename('MJD').toFloat()
        
        neg_ndvi = ndvi.multiply(-1).rename('neg_NDVI')
        return img.addBands([ndvi, ndti, sti, mjd, neg_ndvi])

    def get_derf_mosaic(self, region):
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(f'{YEAR}-01-01', f'{YEAR}-12-31') \
            .linkCollection(
                ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'), 
                ['cs_cdf']
            ) \
            .map(self.mask_clouds_strict) \
            .map(self.add_indices_and_time)

        derf_raw = s2.qualityMosaic('neg_NDVI')

        valid_mask = self.cropland \
            .And(derf_raw.select('NDVI').lt(NDVI_MAX_THRESHOLD)) \
            .And(derf_raw.select('NDTI').mask())

        return derf_raw.updateMask(valid_mask)

    # Apply retry logic to the heavy clustering function
    @retry_on_error(retries=5, delay=10)
    def two_pass_clustering(self, derf_image, region):
        input_bands = derf_image.select(['MJD'])

        # Optimization: Use tileScale 4 to avoid memory errors during sampling
        sample_p1 = input_bands.sample(
            region=region,
            scale=100,
            numPixels=5000, 
            geometries=True,
            seed=42,
            tileScale=4 
        )
        
        # Check size locally
        if sample_p1.size().getInfo() < 100:
            return None 

        # --- PASS 1 ---
        clusterer_p1 = ee.Clusterer.wekaKMeans(CLUSTER_P1_K).train(sample_p1)
        result_p1 = input_bands.cluster(clusterer_p1).rename('cluster_p1')

        # --- FILTER ---
        # Optimization: Added tileScale=4 to reduceRegion
        hist = result_p1.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=region,
            scale=200, 
            maxPixels=1e9,
            tileScale=4 
        ).get('cluster_p1').getInfo()

        if not hist: return None

        total_pixels = sum(hist.values())
        valid_ids = []
        for cid, count in hist.items():
            if (count / total_pixels) > OUTLIER_THRESHOLD:
                valid_ids.append(int(cid))
        
        if len(valid_ids) < 2: return None

        valid_mask = result_p1.remap(valid_ids, [1]*len(valid_ids), 0).eq(1)

        # --- PASS 2 ---
        clean_input = input_bands.updateMask(valid_mask)
        
        sample_p2 = clean_input.sample(
            region=region,
            scale=100,
            numPixels=5000,
            seed=43,
            tileScale=4
        )
        
        if sample_p2.size().getInfo() < 50: return None

        clusterer_p2 = ee.Clusterer.wekaKMeans(CLUSTER_P2_K).train(sample_p2)
        final_clusters = clean_input.cluster(clusterer_p2).rename('season_cluster')

        return final_clusters

    @retry_on_error(retries=3, delay=2)
    def is_tile_arable(self, region):
        """
        Checks if a tile has arable land. 
        OPTIMIZATION: increased scale to 5000m for faster check.
        """
        stats = self.cropland.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5000, # Increased from 1000 to speed up pre-check
            maxPixels=1e8,
            tileScale=2
        ).getInfo()
        
        val = stats.get('Map')
        if val is None:
            val = 0
            
        return (val * 100) > ARABLE_MIN_PERCENT

    def process_tile(self, min_x, min_y, max_x, max_y):
        tile_id = f"X{int(min_x)}_Y{int(min_y)}"
        region = ee.Geometry.Rectangle([min_x, min_y, max_x, max_y])
        
        try:
            # 1. Arable Check
            if not self.is_tile_arable(region):
                return f"Skipped: {tile_id} (No Arable Land)"

            # 2. Get Derf (Lazy evaluation, no getInfo here)
            derf_img = self.get_derf_mosaic(region)

            # 3. Cluster (Contains getInfo calls)
            clusters = self.two_pass_clustering(derf_img, region)
            
            if clusters is None:
                return f"Skipped: {tile_id} (Clustering Failed/Not Enough Data)"

            # 4. Export
            final_image = derf_img.select(['B2','B3','B4','B8','B11','B12','NDTI','STI','MJD']) \
                .addBands(clusters) \
                .clip(region) \
                .int16()

            desc = f"Tillage_{tile_id}_{YEAR}"
            
            task = ee.batch.Export.image.toDrive(
                image=final_image,
                description=desc,
                folder=OUTPUT_FOLDER,
                region=region,
                scale=20, 
                maxPixels=1e10,
                crs='EPSG:4326',
                fileFormat='GeoTIFF',
                formatOptions={'cloudOptimized': True}
            )
            task.start()
            return f"Started Export: {desc}"
            
        except Exception as e:
            # If it still fails after internal retries, we return the error
            return f"Error {tile_id}: {str(e)}"

def generate_global_grid():
    tiles = []
    for lat in range(LAT_MIN, LAT_MAX, int(GRID_SIZE)):
        for lon in range(LON_MIN, LON_MAX, int(GRID_SIZE)):
            tiles.append((lon, lat, lon + GRID_SIZE, lat + GRID_SIZE))
    return tiles

def main():
    processor = GlobalTillageProcessor()
    grid = generate_global_grid()
    
    print(f"{'='*50}")
    print(f"Global Tillage Processing Started (Fixed Version)")
    print(f"Grid Size: {GRID_SIZE} deg | Total Tiles: {len(grid)}")
    print(f"Parallel Workers: {MAX_WORKERS}")
    print(f"{'='*50}\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(processor.process_tile, *tile): tile for tile in grid}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if "Started" in result or "Error" in result:
                print(f"[{i+1}/{len(grid)}] {result}")
            if i % 50 == 0:
                print(f"[{i+1}/{len(grid)}] Progress update...")

    print("\nAll tasks submitted. Check GEE Task Manager.")

if __name__ == "__main__":
    main()