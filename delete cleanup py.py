"""
MODULE: CA MEGA-CLUSTER DETECTOR
Scans the global Master TIF to find high-density Conservation Agriculture hotbeds.
Uses a regional moving window (reduceNeighborhood) to guarantee that the 
extracted points sit in the middle of massive continuous CA farms.
"""
import ee

# ==========================================
# 1. INITIALIZATION & CONFIG
# ==========================================
PROJECT_ID = 'gvrsf-2026'

try:
    ee.Initialize(project=PROJECT_ID)
    print(f"GEE Initialized: {PROJECT_ID}")
except Exception as e:
    print(f"Init Error: {e}")

MOD6_ASSET_ID = f"projects/{PROJECT_ID}/assets/Global_Tillage_Classes_Master"

def main():
    print("="*60)
    print("INITIATING GLOBAL CA MEGA-CLUSTER DETECTOR")
    print("="*60)
    
    print("\n1. Loading Master Data & Calculating Soft Labels...")
    master = ee.Image(MOD6_ASSET_ID)
    
    # Calculate Soft Labels exactly as before
    b_trad_ann = master.select([0])
    b_trad_rot = master.select([1])
    b_rot      = master.select([2])
    b_reduced  = master.select([3])
    b_conv_ann = master.select([4])
    b_ca       = master.select([5])

    cons_area = b_ca.add(b_reduced)
    grouped = ee.Image([cons_area, b_conv_ann.add(b_rot), b_trad_ann.add(b_trad_rot)])
    total = grouped.reduce(ee.Reducer.sum())
    
    # This is our raw CA fraction (0.0 to 1.0) for every 250m pixel
    ca_fraction = cons_area.divide(total).unmask(0)

    print("2. Launching 16km Spatial Radar (Neighborhood Reducer)...")
    # We draw an 8000m radius circle (16km width) around EVERY pixel on earth.
    # It calculates the average CA density of that massive circle.
    circle_kernel = ee.Kernel.circle(radius=8000, units='meters')
    
    ca_regional_density = ca_fraction.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=circle_kernel
    )

    print("3. Isolating Epicenters (>40% Regional CA Density)...")
    # Finding a single pixel that is 100% CA is easy. 
    # Finding a 16km circle that averages >40% CA is incredibly rare. These are our Mega-Clusters.
    mega_cluster_mask = ca_regional_density.gte(0.40)
    
    # Create a binary layer for stratified sampling
    hotspot_zone = ee.Image.constant(1).updateMask(mega_cluster_mask).rename('cluster_val')

    print("4. Executing Global Stratified Drop (Target: 250 Points)...")
    try:
        # We sample at a 5km scale to bypass memory limits and scatter the points globally
        hotspots = hotspot_zone.stratifiedSample(
            numPoints=250,
            classBand='cluster_val',
            region=ee.Geometry.BBox(-180, -56, 180, 84),
            scale=5000,
            geometries=True
        ).getInfo()['features']
        
        print(f"\n✅ SUCCESS: Found {len(hotspots)} Mega-Cluster Epicenters!")
        
        # Print a few coordinates to verify they scattered globally
        print("\nSample Coordinates (Longitude, Latitude):")
        for i in range(min(5, len(hotspots))):
            coords = hotspots[i]['geometry']['coordinates']
            print(f"  - Epicenter {i+1}: {coords[0]:.4f}, {coords[1]:.4f}")
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    main()