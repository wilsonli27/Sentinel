import os
import math
import numpy as np
import rasterio
import tensorflow as tf

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "Global_Tillage_ResNet50_Dice.h5"
PATCH_DIR = r"D:\Users\Wilson\Downloads\Sentinel\image_cnn\CNN_Tillage_Patches_Hybrid"
PATCH_SIZE = 64

# Our target UI regions and their rough Lat/Lng coordinates
target_regions = {
    "North American Corn Belt": {"coords": [41.0, -90.0], "best_file": None, "min_dist": float('inf')},
    "Canadian Prairies": {"coords": [51.0, -105.0], "best_file": None, "min_dist": float('inf')},
    "Indo-Gangetic Plain": {"coords": [27.0, 80.0], "best_file": None, "min_dist": float('inf')},
    "South American Cerrado": {"coords": [-12.0, -50.0], "best_file": None, "min_dist": float('inf')},
    "Australian Wheatbelt": {"coords": [-32.0, 118.0], "best_file": None, "min_dist": float('inf')}
}

# The Haversine formula to calculate true distance across the Earth's curve
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dLat, dLon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ==========================================
# 2. PHASE 1: SCAN FOLDER FOR CLOSEST MATCHES
# ==========================================
print(f"Scanning directory: {PATCH_DIR}")
print("Searching for the closest patches to our 5 targets...")

valid_files = [f for f in os.listdir(PATCH_DIR) if f.endswith('.tif')]

for filename in valid_files:
    file_path = os.path.join(PATCH_DIR, filename)
    try:
        with rasterio.open(file_path) as src:
            # Get the geographic bounds of the patch
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            
            # Check how close this patch is to our 5 target regions
            for region, data in target_regions.items():
                target_lat, target_lon = data['coords']
                dist = haversine(center_lat, center_lon, target_lat, target_lon)
                
                # If this is the closest one we've seen so far, save it!
                if dist < data['min_dist']:
                    data['min_dist'] = dist
                    data['best_file'] = file_path
    except Exception:
        pass # Skip corrupted files

print("\nFound the best candidates!")

# ==========================================
# 3. PHASE 2: RUN THE AI MODEL
# ==========================================
print("Loading Dual-Head ResNet-50...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("\n" + "="*50)
print(" PASTE THIS JAVASCRIPT INTO YOUR INDEX.HTML FILE")
print("="*50 + "\n")

for region_name, data in target_regions.items():
    file_path = data['best_file']
    
    if not file_path:
        print(f"// ERROR: Could not find any valid files for {region_name}")
        continue
        
    with rasterio.open(file_path) as src:
        # Extract the actual coordinates of the patch we found
        bounds = src.bounds
        found_lat = round((bounds.bottom + bounds.top) / 2, 4)
        found_lon = round((bounds.left + bounds.right) / 2, 4)

        img_data = src.read()
        img_data = np.transpose(img_data, (1, 2, 0))
        img_data = np.nan_to_num(img_data, nan=0.0)
        
        # Center crop to 64x64
        h, w, _ = img_data.shape
        start_y = (h - PATCH_SIZE) // 2
        start_x = (w - PATCH_SIZE) // 2
        X_patch = img_data[start_y:start_y+PATCH_SIZE, start_x:start_x+PATCH_SIZE, :13].astype(np.float32)
        
        # Apply your Custom Multimodal Scaling
        X_patch[:, :, :10] /= 10000.0   
        X_patch[:, :, 10:12] /= 30000.0 
        
        # Apply your NDVI Cropland Mask
        ndvi = X_patch[:, :, 7]
        cropland_mask = np.expand_dims((ndvi > -0.1) & (ndvi < 0.8), axis=-1)
        X_patch = X_patch * cropland_mask
        
        # Add the Batch Dimension for Keras
        X_batch = np.expand_dims(X_patch, axis=0)
        
# Run Prediction
        predictions = model.predict(X_batch, verbose=0)
        
        # 1. Grab the 64x64x3 map from the batch
        segmentation_map = predictions[0]
        
        # 2. Average all spatial pixels (axes 0 and 1) to get the 3 global fractions
        raw_fractions = np.mean(segmentation_map, axis=(0, 1))
        
        # Now raw_fractions is a perfect 1D array of 3 numbers! (e.g., [0.60, 0.10, 0.30])
        cons_pct = int(round(raw_fractions[0] * 100))
        conv_pct = int(round(raw_fractions[1] * 100))
        trad_pct = int(round(raw_fractions[2] * 100))
        conv_pct = int(round(raw_fractions[1] * 100))
        trad_pct = int(round(raw_fractions[2] * 100))
        
        # Ensure it perfectly equals 100%
        total = cons_pct + conv_pct + trad_pct
        if total != 100:
            conv_pct += (100 - total)

        # Print the exact JS object formatting
        print(f'"{region_name}": {{')
        print(f'    coords: [{found_lat}, {found_lon}],')
        print(f'    predictions: [{cons_pct}, {conv_pct}, {trad_pct}], // Actual AI Output from: {os.path.basename(file_path)}')
        
        # Keep the steps logic identical
        if cons_pct < 30:
            print('    steps: ["Subsidize no-till drill equipment rentals.", "Implement tax credits for off-season cover crops.", "Host community agronomy workshops."]')
        elif cons_pct > 60:
            print('    steps: ["Publish regional case studies on drought resilience.", "Optimize targeted fertilizer application.", "Establish the region as a global benchmark."]')
        else:
            print('    steps: ["Maintain carbon credit payouts to reward good behavior.", "Monitor soil moisture retention levels.", "Expand successful pilot programs."]')
        print(f'}},')