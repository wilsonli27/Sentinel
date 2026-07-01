import ee
import geemap
import os

# ================= CONFIGURATION =================
PROJECT_ID = 'gvrsf-2026'

# GEE ASSET PATHS
# The output from Module 2 (Spectral Images)
SENTINEL_COLLECTION = f"projects/{PROJECT_ID}/assets/Global_Tillage_Spectral_v1"
# The Tillage Map you uploaded (Step 1 from previous instruction)
TILLAGE_ASSET = f"projects/{PROJECT_ID}/assets/Global_Tillage_Map"

# LOCAL OUTPUT PATH (Base Directory)
# The script will create sub-folders 'Conventional', 'Conservational', 'Rotational' here.
BASE_OUTPUT_DIR = r"D:\Users\Wilson\Downloads\Sentinel\CNN_Training_Data"

# CLASS DEFINITIONS (Strict Mapping)
CLASSES = {
    "Conventional":   [1, 2],
    "Conservational": [3, 4],
    "Rotational":     [5, 6]
}

# ================= MAIN LOGIC =================
def main():
    # 1. Initialize GEE
    try:
        ee.Initialize(project=PROJECT_ID)
        print("GEE Initialized successfully.")
    except Exception as e:
        print(f"Init Error: {e}")
        print("Try running 'ee.Authenticate()' in your terminal.")
        return

    # 2. Setup Local Directories
    for class_name in CLASSES.keys():
        class_dir = os.path.join(BASE_OUTPUT_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"Created directory: {class_dir}")

    # 3. Load Collections
    try:
        s2_col = ee.ImageCollection(SENTINEL_COLLECTION)
        # Ensure we can read the collection
        tile_list = s2_col.aggregate_array('system:index').getInfo()
        
        tillage_map = ee.Image(TILLAGE_ASSET)
        # Verify Tillage Map loads
        _ = tillage_map.id().getInfo() 
        
    except Exception as e:
        print(f"Error accessing GEE assets: {e}")
        return

    print(f"Found {len(tile_list)} Sentinel tiles to process.")

    # 4. Processing Loop
    for i, tile_id in enumerate(tile_list):
        print(f"\nProcessing Tile [{i+1}/{len(tile_list)}]: {tile_id}")
        
        try:
            # Fetch the specific Sentinel image
            img = s2_col.filter(ee.Filter.eq('system:index', tile_id)).first()
            region = img.geometry() # Use the tile's own boundary
            
            # --- Iterate through each Tillage Class ---
            for class_name, values in CLASSES.items():
                
                # A. Construct the Mask
                # Start with a mask of 0s
                class_mask = ee.Image(0)
                # Combine conditions: (tillage == 1) OR (tillage == 2) ...
                for v in values:
                    class_mask = class_mask.Or(tillage_map.eq(v))
                
                # B. Apply Mask
                # This makes everything NOT in the class invisible (NoData)
                # It preserves the 12 spectral bands only where the mask is 1
                masked_img = img.updateMask(class_mask)
                
                # C. Check for Data Existence (Optimization)
                # We count valid pixels to avoid downloading empty black images
                try:
                    stats = class_mask.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=region,
                        scale=1000, # Fast, coarse check
                        maxPixels=1e9
                    ).getInfo()
                    
                    # If the sum is effectively 0, the class isn't here.
                    pixel_hits = list(stats.values())[0]
                    if pixel_hits < 10: # Threshold to ignore tiny noise
                        # print(f"  - No {class_name} pixels found. Skipping.")
                        continue
                except:
                    # If reduceRegion fails (e.g. tile completely outside), skip safely
                    continue

                # D. Define Output Path
                # Save into the specific class folder
                out_dir = os.path.join(BASE_OUTPUT_DIR, class_name)
                out_filename = f"{tile_id}_{class_name}.tif"
                out_path = os.path.join(out_dir, out_filename)
                
                if os.path.exists(out_path):
                    print(f"  -> Skipping {class_name} (File exists)")
                    continue

                print(f"  -> Downloading {class_name}...")
                
                # E. Download
                geemap.download_ee_image(
                    image=masked_img,
                    filename=out_path,
                    region=region,
                    scale=30,      # Match the resolution of Mod 2 export
                    crs='EPSG:4326',
                    dtype='float32'
                )
                
        except Exception as e:
            print(f"  [!] Error processing {tile_id}: {e}")

if __name__ == "__main__":
    main()