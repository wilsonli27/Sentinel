import ee
import geemap
import os
import numpy as np
import tensorflow as tf

# 1. Initialize GEE
try:
    ee.Initialize(project='gvrsf-2026')
    print("GEE Initialized.")
except:
    ee.Authenticate()
    ee.Initialize(project='gvrsf-2026')

# 2. Configuration
ASSET_PATH = "projects/gvrsf-2026/assets/Global_Tillage_Spectral_v1"
OUTPUT_DIR = "data"  # Local folder to save chips
CLASSES = ["Conventional", "Conservational", "Rotational"]
PATCH_SIZE = 64  # Per research paper 
SAMPLES_PER_CLASS = 1000  # Total training images we want per class

# Create directories
for c in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

# 3. Patch Extraction Function
def download_chips_from_gee():
    print("Scanning GEE Assets for valid training patches...")
    
    # Load your collection
    col = ee.ImageCollection(ASSET_PATH)
    
    # Get list of assets
    asset_list = col.aggregate_array('system:id').getInfo()
    print(f"Found {len(asset_list)} source tiles. Extracting valid chips...")

    for class_name in CLASSES:
        count = 0
        # Filter collection for this class (based on filename logic from Mod 3)
        # We look for assets containing the class name
        class_assets = [a for a in asset_list if class_name in a]
        
        if not class_assets:
            print(f"Warning: No assets found for {class_name}")
            continue

        print(f"  Processing class: {class_name} ({len(class_assets)} tiles available)")
        
        # Shuffle assets to get a random distribution
        np.random.shuffle(class_assets)
        
        for asset_id in class_assets:
            if count >= SAMPLES_PER_CLASS: break
            
            try:
                img = ee.Image(asset_id)
                
                # We need to find where the pixels ARE (Valid Mask)
                # We sample random points inside the valid region
                # Since your assets are masked, we use the mask itself.
                mask = img.select(0).mask() 
                
                # Sample points where mask == 1
                points = mask.sample(
                    region=img.geometry(), 
                    scale=30, 
                    numPixels=50,  # Grab 50 chips per tile
                    geometries=True
                )
                
                # Download patches around these points
                size_list = points.size().getInfo()
                if size_list == 0: continue
                
                # Convert points to feature collection
                point_list = points.toList(size_list)
                
                for i in range(size_list):
                    if count >= SAMPLES_PER_CLASS: break
                    
                    pt = ee.Feature(point_list.get(i))
                    # Create 64x64 buffer (approx 2km box)
                    buffer = pt.geometry().buffer(PATCH_SIZE * 30 / 2).bounds()
                    
                    # Export the pixels
                    # We select specific bands. 
                    # Assuming Mod 2 output: B2, B3, B4, B8, B8A, B11, B12, NDVI, NDTI, STI, MJD, ClusterID
                    patch = geemap.ee_to_numpy(
                        img.select(['B2','B3','B4','B8','B8A','B11','B12','NDVI','NDTI','STI']), 
                        region=buffer
                    )
                    
                    # Validate shape (must be 64x64 or close)
                    if patch is not None and patch.shape[0] >= 60 and patch.shape[1] >= 60:
                        # Resize strictly to 64x64 to avoid errors
                        patch_resized = tf.image.resize(patch, [PATCH_SIZE, PATCH_SIZE]).numpy()
                        
                        # Save as .npy (Better than JPG for spectral data)
                        save_path = os.path.join(OUTPUT_DIR, class_name, f"chip_{count}.npy")
                        np.save(save_path, patch_resized)
                        count += 1
                        print(f"    Saved {class_name} chip {count}/{SAMPLES_PER_CLASS}", end='\r')
                        
            except Exception as e:
                print(f"    Skipping tile error: {e}")
                continue
        print(f"\n  Finished {class_name}.")

# EXECUTE DATA LOADING
download_chips_from_gee()