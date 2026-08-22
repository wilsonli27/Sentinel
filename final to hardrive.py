import ee
import time

# Initialize GEE
try:
    ee.Initialize(project='gvrsf-2026')
    print("GEE Initialized successfully.")
except Exception as e:
    print(f"Init Error: {e}")

ASSET_FOLDER = 'projects/gvrsf-2026/assets/Global_Tillage_SoftLabels_3Class_v1'
DRIVE_FOLDER_NAME = 'CNN_Tillage_Tiles'  

def export_assets_to_drive():
    print(f"Scanning {ASSET_FOLDER} for finished tiles...")
    
    try:
        assets = ee.data.listAssets({'parent': ASSET_FOLDER}).get('assets', [])
    except Exception as e:
        print(f"Error reading assets: {e}")
        return

    if not assets:
        print("No assets found in the collection.")
        return

    print(f"Found {len(assets)} tiles. Queuing exports to Google Drive...\n")
    
    for asset in assets:
        asset_id = asset['id']
        filename = asset_id.split('/')[-1]
        
        # 🚀 THE FIX: Cast all 15 bands to Float32 before GeoTIFF conversion
        img = ee.Image(asset_id).toFloat()
        
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=f"Drive_Export_{filename}",
            folder=DRIVE_FOLDER_NAME,
            fileNamePrefix=filename,
            scale=250, 
            maxPixels=1e13,
            fileFormat='GeoTIFF'
        )
        
        task.start()
        print(f"✓ Queued to Drive: {filename}")
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("ALL DRIVE EXPORTS INITIATED.")
    print("Watch them process here: https://code.earthengine.google.com/tasks")
    print("=" * 60)

if __name__ == "__main__":
    export_assets_to_drive()