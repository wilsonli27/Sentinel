"""
MODULE 3.0: CA MEGA-CLUSTER BOOSTER
- STRICTLY CONSERVATION AGRICULTURE: Ignores Conv/Trad tiles entirely.
- MEGA-CLUSTER RADAR: Uses a 16km moving window to guarantee >40% regional CA density.
- DYNAMIC NDVI ANCHOR: Scans the entire year to find the barest (lowest NDVI) 
  planting date for each epicenter automatically.
- Exports strict 64x64 patches with 13 Input Bands (No MJD).
"""
import ee
import random
import time

# ==========================================
# 1. INITIALIZATION & CONFIG
# ==========================================
PROJECT_ID = 'gvrsf-2026'

try:
    ee.Initialize(project=PROJECT_ID)
    print(f"GEE Initialized: {PROJECT_ID}")
except Exception as e:
    print(f"Init Error: {e}")

MOD6_ASSET_ID  = f"projects/{PROJECT_ID}/assets/Global_Tillage_Classes_Master"
DRIVE_FOLDER   = "CNN_Tillage_Patches_Hybrid"

TARGET_CA_QUOTA = 150 
PATCH_SIZE_METERS = 8000 # 8000m radius = 16x16km box = exactly 64x64 pixels at 250m

# ==========================================
# 2. THE GEOGRAPHIC ENGINE
# ==========================================
class CABoosterSampler:
    def __init__(self):
        print("Loading Master Tillage Map...")
        self.master = ee.Image(MOD6_ASSET_ID)
        
        b_trad_ann = self.master.select([0])
        b_trad_rot = self.master.select([1])
        b_rot      = self.master.select([2])
        b_reduced  = self.master.select([3])
        b_conv_ann = self.master.select([4])
        b_ca       = self.master.select([5])

        cons_area = b_ca.add(b_reduced).rename('Conservational')
        conv_area = b_conv_ann.add(b_rot).rename('Conventional')
        trad_area = b_trad_ann.add(b_trad_rot).rename('Traditional')

        grouped = ee.Image([cons_area, conv_area, trad_area])
        total = grouped.reduce(ee.Reducer.sum())
        self.soft_labels = grouped.divide(total).unmask(0).toFloat()
        
        # Raw CA Fraction (0.0 to 1.0)
        self.ca_fraction = cons_area.divide(total).unmask(0)

    def get_barest_date(self, lon, lat):
        """
        Scans the entire year for this specific coordinate to find the exact date 
        with the lowest cloud-free NDVI (the planting/pre-emergence phase).
        """
        point = ee.Geometry.Point([lon, lat])
        
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(point) \
            .filterDate('2023-01-01', '2023-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .linkCollection(ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'), ['cs_cdf'])
        
        def add_mean_ndvi(img):
            # Strict cloud mask
            qa = img.select('cs_cdf')
            mask = qa.gte(0.65)
            ndvi = img.normalizedDifference(['B8', 'B4']).updateMask(mask).rename('NDVI')
            
            # Get the average NDVI of the farm surrounding the epicenter
            mean_ndvi = ndvi.reduceRegion(ee.Reducer.mean(), point.buffer(500), 50).get('NDVI')
            return img.set('mean_ndvi', mean_ndvi)
        
        # Filter out extreme negatives (snow/water) and sort ascending to find the barest dirt
        s2_sorted = s2.map(add_mean_ndvi).filter(ee.Filter.gt('mean_ndvi', 0.05)).sort('mean_ndvi', True)
        
        try:
            barest_img = ee.Image(s2_sorted.first())
            return barest_img.date().millis().getInfo()
        except Exception:
            # Fallback to a standard spring/summer date if the algorithm hits a massive cloud bank year
            return ee.Date('2023-05-15').millis().getInfo()

    def generate_mega_clusters(self):
        print("\n[PHASE 1] Launching 16km Spatial Radar for CA Mega-Clusters...")
        circle_kernel = ee.Kernel.circle(radius=8000, units='meters')
        
        ca_regional_density = self.ca_fraction.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=circle_kernel
        )

        # ISOLATE REGIONS > 40% CA DENSITY
        mega_cluster_mask = ca_regional_density.gte(0.40)
        hotspot_zone = ee.Image.constant(1).updateMask(mega_cluster_mask).rename('cluster_val')

        print(f"\n[PHASE 2] Executing Global Drop for {TARGET_CA_QUOTA} points...")
        try:
            ca_pts = hotspot_zone.stratifiedSample(
                numPoints=TARGET_CA_QUOTA,
                classBand='cluster_val',
                region=ee.Geometry.BBox(-180, -56, 180, 84),
                scale=5000, 
                geometries=True
            ).getInfo()['features']
        except Exception as e:
            raise ValueError(f"CRITICAL: Failed to sample mega-clusters. {e}")

        print(f"  ✓ Found {len(ca_pts)} CA Epicenters.")
        print(f"\n[PHASE 3] Scanning 2023 timeline for lowest NDVI dates (Please wait, calculating...)")
        
        ca_list = []
        for i, f in enumerate(ca_pts):
            lon, lat = f['geometry']['coordinates']
            
            # Print progress every 25 points so you know it hasn't crashed
            if (i+1) % 25 == 0:
                print(f"  ...Found planting dates for {i+1}/{len(ca_pts)} epicenters.")
                
            barest_ms = self.get_barest_date(lon, lat)
            ca_list.append({
                'coords': [lon, lat], 
                'date': barest_ms,
                'class': 'Cons_Boost'
            })
            
        return ca_list

    def extract_patch(self, lon, lat, time_start_ms):
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(PATCH_SIZE_METERS).bounds()
        date = ee.Date(time_start_ms)

        # ── SENTINEL-2 (OPTICAL) ──
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(region) \
            .filterDate(date.advance(-15, 'day'), date.advance(15, 'day')) \
            .linkCollection(ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'), ['cs_cdf']) \
            
        def mask_s2(img):
            qa = img.select('cs_cdf')
            scl = img.select('SCL')
            mask = qa.gte(0.65).And(scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)))
            ndvi = img.normalizedDifference(['B8', 'B4']).multiply(10000).rename('NDVI')
            ndti = img.normalizedDifference(['B11', 'B12']).multiply(10000).rename('NDTI')
            sti = img.select('B11').divide(img.select('B12')).multiply(10000).rename('STI')
            return img.select(['B2','B3','B4','B8','B8A','B11','B12']).addBands([ndvi, ndti, sti]).updateMask(mask)

        # 🚀 THE S2 FALLBACK INJECTION
        s2_fallback = ee.Image.constant([0]*10).rename(['B2','B3','B4','B8','B8A','B11','B12','NDVI','NDTI','STI']).updateMask(0)
        s2_comp = s2.map(mask_s2).merge(ee.ImageCollection([s2_fallback])).mean().toInt16()

        # ── SENTINEL-1 (RADAR) ──
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(region) \
            .filterDate(date.advance(-15, 'day'), date.advance(15, 'day')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filter(ee.Filter.eq('instrumentMode', 'IW'))

        # 🚀 THE S1 FALLBACK INJECTION
        s1_fallback = ee.Image.constant([0, 0]).rename(['VV', 'VH']).updateMask(0)
        s1_comp = s1.select(['VV', 'VH']).merge(ee.ImageCollection([s1_fallback])).mean().multiply(1000).toInt16() 

        cluster_id = ee.Image.constant(1).rename('cluster_id').toInt8()
        
        # Combine the 13 Input Features (S2 + S1 + Cluster) + 3 Soft Labels
        features = s2_comp.addBands([s1_comp, cluster_id])
        final_stack = features.addBands(self.soft_labels).clip(region).toFloat()

        return final_stack, region

# ==========================================
# 3. EXECUTION
# ==========================================
def main():
    print("="*60)
    print("INITIATING CA MEGA-CLUSTER BOOSTER")
    print("="*60)

    sampler = CABoosterSampler()
    
    try:
        master_list = sampler.generate_mega_clusters()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        return
        
    print(f"\nQueueing {len(master_list)} ultra-dense CA patches to Drive...")
    
    queued = 0
    for i, item in enumerate(master_list):
        try:
            lon, lat = item['coords']
            date_ms = item['date']
            
            img_stack, region = sampler.extract_patch(lon, lat, date_ms)
            
            # Labeled as Booster so you know they are the new high-density tiles
            filename = f"Patch_Booster_{i:04d}_CONS_S1S2"
            
            task = ee.batch.Export.image.toDrive(
                image=img_stack,
                description=filename,
                folder=DRIVE_FOLDER,
                fileNamePrefix=filename,
                region=region,
                scale=250, 
                maxPixels=1e8,
                fileFormat='GeoTIFF'
            )
            task.start()
            queued += 1
            
            if queued % 50 == 0:
                print(f" ✓ Queued {queued} tasks...")
                
            time.sleep(0.1) 
            
        except Exception as e:
            pass 

    print("\n" + "="*60)
    print(f"ALL DONE! Successfully queued {queued} patches.")
    print("Watch the Task Manager: https://code.earthengine.google.com/tasks")
    print("="*60)

if __name__ == "__main__":
    main()