import ee

print("Initialising Earth Engine...")
ee.Initialize(project='gvrsf-2026')

DRIVE_FOLDER   = "gee_outputs_250m"
EXPORT_SCALE_M = 250 

# Define the four global quadrants to prevent GEE timeouts
QUADRANTS = {
    "NW": ee.Geometry.Rectangle([-180, 0, 0, 84], 'EPSG:4326', False),
    "NE": ee.Geometry.Rectangle([0, 0, 180, 84], 'EPSG:4326', False),
    "SW": ee.Geometry.Rectangle([-180, -56, 0, 0], 'EPSG:4326', False),
    "SE": ee.Geometry.Rectangle([0, -56, 180, 0], 'EPSG:4326', False)
}

print("\n=== Generating ESA Cropland Fraction (CHUNKED GLOBAL) ===")

try:
    esa_worldcover = ee.Image("ESA/WorldCover/v100/2020")
    cropland_mask = esa_worldcover.eq(40)
    projection = ee.Projection('EPSG:4326').atScale(EXPORT_SCALE_M)
    
    cropland_fraction = (cropland_mask
                         .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=65536)
                         .reproject(crs=projection))

    for quad_name, region in QUADRANTS.items():
        print(f"  ▶ Submitting {quad_name} Quadrant...")
        task = ee.batch.Export.image.toDrive(
            image=cropland_fraction.toFloat(),
            description=f"export_esa_cropland_fraction_250m_{quad_name}",
            folder=DRIVE_FOLDER,
            fileNamePrefix=f"esa_cropland_fraction_250m_{quad_name}",
            region=region,
            scale=EXPORT_SCALE_M,
            crs='EPSG:4326',
            maxPixels=1e13,
            fileFormat='GeoTIFF'
        )
        task.start()
        
    print("  ✓ All 4 Global Quadrants queued successfully!")

except Exception as e:
    print(f"  ✗ Failed: {e}")