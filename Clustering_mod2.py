"""
MODULE 2.5: HYBRID PATCH EXPORTER (MASTER MASK EDITION)
- Phase 1: Probes the 287 plowing tiles for Conv/Trad and harvests valid dates.
- Phase 2: Uses the Master TIF as a strict global mask to instantly pull 500 CA points.
- Steals planting dates from Phase 1 to prevent calendar cheating.
- Exports strict 64x64 patches with 13 Input Bands (No MJD).
"""
import ee
import re
import time
import random

# ==========================================
# 1. INITIALIZATION & CONFIG
# ==========================================
PROJECT_ID = 'gvrsf-2026'

try:
    ee.Initialize(project=PROJECT_ID)
    print(f"GEE Initialized: {PROJECT_ID}")
except Exception as e:
    print(f"Init Error: {e}")

INPUT_ASSET_ID = f"projects/{PROJECT_ID}/assets/Tillage_Time_Markers" 
MOD6_ASSET_ID  = f"projects/{PROJECT_ID}/assets/Global_Tillage_Classes_Master"
DRIVE_FOLDER   = "CNN_Tillage_Patches_Hybrid"

TARGET_QUOTA_PER_CLASS = 500 
PATCH_SIZE_METERS = 8000 # 8000m radius = 16x16km box = exactly 64x64 pixels at 250m

# ==========================================
# 2. THE GEOGRAPHIC ENGINE
# ==========================================
class HybridSampler:
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

    def get_global_tiles(self):
        assets = ee.data.listAssets({'parent': INPUT_ASSET_ID})
        valid_tiles = []
        for asset in assets.get('assets', []):
            name = asset['id'].split('/')[-1]
            match = re.search(r"X(-?\d+)_Y(-?\d+)", name)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                bounds = ee.Geometry.Rectangle([x, y, x + 5.0, y + 5.0])
                valid_tiles.append({'id': asset['id'], 'bounds': bounds})
        return valid_tiles

    def generate_balanced_lists(self):
        valid_tiles = self.get_global_tiles()
        random.shuffle(valid_tiles) 
        
        conv_list, trad_list = [], []
        valid_dates = []
        
        print(f"\n[PHASE 1] Probing 287 tiles for Conv & Trad ({TARGET_QUOTA_PER_CLASS} each)...")
        
        for idx, tile in enumerate(valid_tiles):
            if len(conv_list) >= TARGET_QUOTA_PER_CLASS and len(trad_list) >= TARGET_QUOTA_PER_CLASS:
                break
                
            print(f"  Probing tile {idx+1}/{len(valid_tiles)}...", end='\r')
            
            tile_img = ee.Image(tile['id'])
            tile_mjd = tile_img.select('MJD')
            tile_bounds = tile['bounds']
            
            try:
                # ── PROBE FOR CONVENTIONAL ──
                if len(conv_list) < TARGET_QUOTA_PER_CLASS:
                    conv_mask = self.soft_labels.select('Conventional').gte(0.40).And(tile_mjd.mask())
                    conv_pts = self.soft_labels.select('Conventional').addBands(tile_mjd).updateMask(conv_mask).sample(
                        region=tile_bounds, scale=250, numPixels=50, geometries=True
                    ).getInfo()['features']
                    
                    for f in conv_pts:
                        mjd = f['properties'].get('MJD')
                        if mjd:
                            ms_time = (mjd - 40587.0) * 86400000.0 
                            valid_dates.append(ms_time)
                            conv_list.append({'coords': f['geometry']['coordinates'], 'date': ms_time, 'class': 'Conv'})

                # ── PROBE FOR TRADITIONAL ──
                if len(trad_list) < TARGET_QUOTA_PER_CLASS:
                    trad_mask = self.soft_labels.select('Traditional').gte(0.40).And(tile_mjd.mask())
                    trad_pts = self.soft_labels.select('Traditional').addBands(tile_mjd).updateMask(trad_mask).sample(
                        region=tile_bounds, scale=250, numPixels=50, geometries=True
                    ).getInfo()['features']
                    
                    for f in trad_pts:
                        mjd = f['properties'].get('MJD')
                        if mjd:
                            ms_time = (mjd - 40587.0) * 86400000.0
                            valid_dates.append(ms_time)
                            trad_list.append({'coords': f['geometry']['coordinates'], 'date': ms_time, 'class': 'Trad'})
                            
            except Exception:
                continue 

        conv_list = conv_list[:TARGET_QUOTA_PER_CLASS]
        trad_list = trad_list[:TARGET_QUOTA_PER_CLASS]
        print(f"\n  ✓ Found {len(conv_list)} Conv | {len(trad_list)} Trad.")

        # ── PHASE 2: GLOBAL CONSERVATION PROBE (USING MASTER MASK) ──
        print(f"\n[PHASE 2] Executing Global Probe for {TARGET_QUOTA_PER_CLASS} Conservation points...")
        ca_list = []
        
        if not valid_dates:
            raise ValueError("CRITICAL: No valid dates found in Phase 1. Cannot synchronize cameras.")

        try:
            # 🚀 Create a binary integer mask directly from the Master TIF
            ca_mask = self.soft_labels.select('Conservational').gte(0.40)
            ca_int_mask = ee.Image.constant(1).updateMask(ca_mask).rename('class_val')

            # 🚀 Use stratifiedSample to instantly pull points from the verified pixels globally
            ca_pts = ca_int_mask.stratifiedSample(
                numPoints=TARGET_QUOTA_PER_CLASS,
                classBand='class_val',
                region=ee.Geometry.BBox(-180, -56, 180, 84), 
                scale=2500, # 2.5km scale avoids memory limits while finding the pixels fast
                geometries=True
            ).getInfo()['features']

            for f in ca_pts:
                ca_list.append({
                    'coords': f['geometry']['coordinates'], 
                    'date': random.choice(valid_dates), # Steal a valid plowing date
                    'class': 'Cons'
                })
        except Exception as e:
            print(f"  ❌ CA Global Probe Failed: {e}")

        print(f"  ✓ Found {len(ca_list)} Conservation points.")

        master_list = conv_list + trad_list + ca_list
        random.shuffle(master_list)
        return master_list

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
    print("INITIATING HYBRID PATCH EXPORTER (MASTER MASK EDITION)")
    print("="*60)

    sampler = HybridSampler()
    
    try:
        master_list = sampler.generate_balanced_lists()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        return
        
    print(f"\nQueueing {len(master_list)} perfectly balanced patches to Drive...")
    
    queued = 0
    for i, item in enumerate(master_list):
        try:
            lon, lat = item['coords']
            date_ms = item['date']
            
            img_stack, region = sampler.extract_patch(lon, lat, date_ms)
            
            filename = f"Patch_{i:04d}_{item['class']}_S1S2"
            
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