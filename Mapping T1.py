"""
Bihar NDTI Verification Map Generator - Simplified Version
No Google Drive authentication required - uses direct GEE sampling
"""

import ee
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from matplotlib.colors import BoundaryNorm, ListedColormap

# Initialize Google Earth Engine
PROJECT_ID = '590577866979'  # Replace with your Google Cloud Project ID

try:
    ee.Initialize(project=PROJECT_ID)
    print("✅ Google Earth Engine initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing GEE: {e}")
    print("Please ensure you have:")
    print("1. Run 'earthengine authenticate'")
    print("2. Created a Google Cloud Project")
    print("3. Set the correct PROJECT_ID")
    exit()


class LandType:
    """Get land type using ESA WorldCover dataset."""
    def __init__(self, GEE_project_id='tlg-erosion1', EE_initialized=True):
        if not EE_initialized: 
            ee.Authenticate()
            ee.Initialize(project=GEE_project_id)
        
        worldcover = ee.ImageCollection('ESA/WorldCover/v200')
        self.worldcover = worldcover

    def get_land_cover_for_region(self, Geometry):
        """Get land cover data for specified geometry"""
        try:
            worldcover_image = self.worldcover.first()
            clipped = worldcover_image.clip(Geometry)
            return {'image': clipped}
        except Exception as e:
            print(f"Error getting land cover: {e}")
            return None

    def Map_LandType(self, landcover_image):
        """Create cropland mask (ESA class 40)"""
        try:
            cropland_mask = landcover_image.eq(40).rename('cropland_mask')
            return cropland_mask
        except Exception as e:
            print(f"Error mapping land type: {e}")
            return None


def get_bihar_polygon():
    """Create Bihar polygon geometry (13 vertices)"""
    try:
        coordinates = [[
            [83.0, 27.5], [84.0, 27.6], [85.5, 27.4], [87.0, 26.9],
            [87.5, 26.5], [87.6, 26.0], [87.0, 25.2], [86.0, 24.6],
            [84.8, 24.3], [83.3, 24.4], [83.0, 25.0], [82.7, 25.8],
            [83.0, 27.5]
        ]]
        return ee.Geometry.Polygon(coordinates)
    except Exception as e:
        print(f"Error creating Bihar polygon: {e}")
        return None


def calculate_ndti(image):
    """
    Calculate Normalized Difference Tillage Index (NDTI)
    NDTI = (B11 - B12) / (B11 + B12)
    B11 = SWIR1 (1610 nm), B12 = SWIR2 (2190 nm)
    """
    try:
        b11 = image.select('B11')  # SWIR1
        b12 = image.select('B12')  # SWIR2
        ndti = b11.subtract(b12).divide(b11.add(b12)).rename('NDTI')
        ndti = ndti.clamp(-1, 1)
        return image.addBands(ndti)
    except Exception as e:
        print(f"Error calculating NDTI: {e}")
        return image


def create_ndti_colormap():
    """
    Create custom colormap for NDTI values
    Focus on 0.0 to 0.4 range where most cropland NDTI values occur
    """
    # NDTI color scheme: negative (blue) -> low (green) -> high (red)
    color_codes = [
        '#0000FF',  # -1.0 to -0.2: Deep blue (water/very wet)
        '#4169E1',  # -0.2 to 0.0: Blue (wet soil)
        '#006400',  # 0.0 to 0.05: Dark green (very low tillage)
        '#228B22',  # 0.05 to 0.10: Green
        '#32CD32',  # 0.10 to 0.15: Lime green
        '#90EE90',  # 0.15 to 0.20: Light green
        '#ADFF2F',  # 0.20 to 0.25: Yellow-green
        '#FFFF00',  # 0.25 to 0.30: Yellow
        '#FFD700',  # 0.30 to 0.35: Gold
        '#FFA500',  # 0.35 to 0.40: Orange
        '#FF6347',  # 0.40 to 0.50: Tomato
        '#FF0000',  # 0.50 to 0.60: Red (high tillage)
        '#8B0000'   # 0.60 to 1.0: Dark red (very high tillage)
    ]
    
    color_bnds = [-1.0, -0.2, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 
                  0.30, 0.35, 0.40, 0.50, 0.60, 1.0]
    
    tick_labels = [f'{color_bnds[i]:.2f}-{color_bnds[i+1]:.2f}' 
                   for i in range(len(color_bnds) - 1)]
    tick_labels[-1] = '0.60+'  # Last bin is open-ended
    
    tick_locs = [(color_bnds[i] + color_bnds[i+1]) / 2 
                 for i in range(len(color_bnds) - 1)]
    
    cmap = ListedColormap(color_codes)
    norm = BoundaryNorm(color_bnds, len(color_codes))
    
    return cmap, norm, tick_locs, tick_labels, color_bnds


def sample_ndti_grid(image, region, scale):
    """
    Sample NDTI data using sampleRegion method
    Creates a grid of points and samples NDTI values
    Returns: data_array, lons, lats, bounds
    """
    try:
        print("\n📊 Sampling NDTI data from GEE...")
        
        # Get bounds
        bounds_coords = region.bounds().coordinates().get(0).getInfo()
        min_lon = min([coord[0] for coord in bounds_coords])
        max_lon = max([coord[0] for coord in bounds_coords])
        min_lat = min([coord[1] for coord in bounds_coords])
        max_lat = max([coord[1] for coord in bounds_coords])
        
        # Calculate grid dimensions
        # Approximate degrees per meter at this latitude
        center_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon = 111320 * np.cos(np.radians(center_lat))
        meters_per_degree_lat = 110540
        
        lon_span_m = (max_lon - min_lon) * meters_per_degree_lon
        lat_span_m = (max_lat - min_lat) * meters_per_degree_lat
        
        lon_steps = max(10, int(lon_span_m / scale))
        lat_steps = max(10, int(lat_span_m / scale))
        
        # GEE has a 5000 element limit for sampleRegions
        total_points = lon_steps * lat_steps
        if total_points > 4900:
            print(f"  ⚠️  Grid too large ({total_points} points), reducing...")
            aspect_ratio = lon_steps / lat_steps
            max_points = 4900
            lat_steps = int(np.sqrt(max_points / aspect_ratio))
            lon_steps = int(lat_steps * aspect_ratio)
            total_points = lon_steps * lat_steps
            print(f"  ✅ Reduced to {lon_steps}x{lat_steps} = {total_points} points")
        
        lons = np.linspace(min_lon, max_lon, lon_steps)
        lats = np.linspace(max_lat, min_lat, lat_steps)
        
        print(f"  📍 Creating {lon_steps}x{lat_steps} sampling grid...")
        
        # Create point geometries
        points = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                point = ee.Geometry.Point([lon, lat])
                feature = ee.Feature(point, {'lat_idx': i, 'lon_idx': j})
                points.append(feature)
        
        points_collection = ee.FeatureCollection(points)
        
        # Sample at points
        print("  🔍 Sampling NDTI values...")
        sampled = image.select('NDTI').sampleRegions(
            collection=points_collection,
            scale=scale,
            geometries=True
        )
        
        # Extract data
        print("  📥 Retrieving sampled data...")
        sampled_list = sampled.getInfo()['features']
        
        # Initialize array with NaN
        data_array = np.full((len(lats), len(lons)), np.nan)
        
        # Fill in sampled values
        valid_count = 0
        for feature in sampled_list:
            props = feature['properties']
            value = props.get('NDTI')
            lat_idx = props.get('lat_idx')
            lon_idx = props.get('lon_idx')
            
            if value is not None and lat_idx is not None and lon_idx is not None:
                data_array[lat_idx, lon_idx] = value
                valid_count += 1
        
        print(f"  ✅ Sampled {valid_count}/{total_points} valid points")
        
        if valid_count < total_points * 0.1:
            print("  ⚠️  Warning: Less than 10% valid samples")
        
        bounds = (min_lon, max_lon, min_lat, max_lat)
        return data_array, lons, lats, bounds
        
    except Exception as e:
        print(f"❌ Error sampling NDTI: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def plot_ndti_map(data_array, lons, lats, bounds, stats_dict, 
                  output_path, title, show_satellite=True):
    """
    Create comprehensive NDTI visualization map
    """
    try:
        print("\n🗺️  Creating NDTI map...")
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add satellite background if requested
        if show_satellite:
            print("  📡 Adding satellite imagery...")
            try:
                imagery = cimgt.GoogleTiles(style='satellite')
                ax.add_image(imagery, 8)  # Zoom level
            except Exception as e:
                print(f"  ⚠️  Could not load satellite imagery: {e}")
                show_satellite = False
        
        if not show_satellite:
            # Add basic map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        
        # Set extent
        ax.set_extent(list(bounds), crs=ccrs.PlateCarree())
        
        # Create meshgrid
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        
        # Get colormap
        cmap, norm, tick_locs, tick_labels, color_bnds = create_ndti_colormap()
        
        # Plot NDTI data
        print("  🎨 Rendering NDTI heatmap...")
        im = ax.pcolormesh(
            lon_mesh, lat_mesh, data_array,
            cmap=cmap, norm=norm, alpha=0.7,
            transform=ccrs.PlateCarree()
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05, aspect=30)
        cbar.set_label('NDTI Value', rotation=270, labelpad=25, fontsize=12, fontweight='bold')
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(tick_labels, fontsize=9)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.3, linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        
        # Title
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Statistics box
        stats_text = (
            f"NDTI Statistics\n"
            f"{'─'*25}\n"
            f"Mean:   {stats_dict['mean']:.6f}\n"
            f"Min:    {stats_dict['min']:.6f}\n"
            f"Max:    {stats_dict['max']:.6f}\n"
            f"Std:    {stats_dict['std']:.6f}\n"
            f"{'─'*25}\n"
            f"Images: {stats_dict['image_count']}\n"
            f"Period: {stats_dict['period']}"
        )
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='black', linewidth=1.5)
        )
        
        # Distribution histogram (inset)
        ax_hist = fig.add_axes([0.72, 0.15, 0.15, 0.15])
        valid_data = data_array[~np.isnan(data_array)]
        
        if len(valid_data) > 0:
            ax_hist.hist(valid_data, bins=50, color='steelblue', 
                        alpha=0.7, edgecolor='black')
            ax_hist.set_xlabel('NDTI', fontsize=8)
            ax_hist.set_ylabel('Frequency', fontsize=8)
            ax_hist.set_title('NDTI Distribution', fontsize=9, fontweight='bold')
            ax_hist.tick_params(labelsize=7)
            ax_hist.grid(True, alpha=0.3)
            
            # Add mean line
            ax_hist.axvline(stats_dict['mean'], color='red', 
                          linestyle='--', linewidth=2, label='Mean')
            ax_hist.legend(fontsize=7)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Map saved to: {output_path}")
        
        plt.show()
        
        return fig
        
    except Exception as e:
        print(f"❌ Error creating map: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_ndti_calculation(start_date='2017-09-01', end_date='2018-05-01',
                           scale=200, output_dir='./ndti_verification'):
    """
    Main function to verify NDTI calculations with visualization
    Uses direct sampling (no Google Drive authentication needed)
    
    Args:
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (YYYY-MM-DD)
        scale: Sampling resolution in meters (200m recommended for speed)
        output_dir: Directory to save output files
    """
    import os
    
    print("="*80)
    print("🌾 BIHAR NDTI VERIFICATION WITH MAPPING")
    print("="*80)
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"📏 Scale: {scale}m")
    print(f"📂 Output: {output_dir}")
    print()
    
    # Create output directory early to catch any permission issues
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Output directory ready: {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"❌ Failed to create output directory: {e}")
        return None, None
    
    try:
        # 1. Get Bihar polygon
        print("📍 Creating Bihar polygon (13 vertices)...")
        bihar_polygon = get_bihar_polygon()
        if bihar_polygon is None:
            raise ValueError("Failed to create Bihar polygon")
        print("  ✅ Bihar polygon created")
        
        # 2. Get cropland mask
        print("\n🌱 Creating cropland mask...")
        lt = LandType(EE_initialized=True)
        result = lt.get_land_cover_for_region(bihar_polygon)
        if result is None:
            raise ValueError("Failed to get land cover data")
        cropland_mask = lt.Map_LandType(result['image'])
        print("  ✅ Cropland mask created")
        
        # 3. Get Sentinel-2 data
        print(f"\n🛰️  Fetching Sentinel-2 data...")
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        
        # Filter by date and region
        collection = (s2
            .filterBounds(bihar_polygon)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
            .linkCollection(csPlus, ['cs_cdf']))
        
        initial_count = collection.size().getInfo()
        print(f"  📊 Initial images: {initial_count}")
        
        if initial_count == 0:
            raise ValueError("No images found in date range")
        
        # 4. Apply cloud masking and calculate NDTI
        print("\n☁️  Applying cloud masking and calculating NDTI...")
        
        def mask_and_calculate(img):
            # Minimal cloud masking
            cs = img.select('cs_cdf')
            cloud_mask = cs.gte(0.40)
            
            scl = img.select('SCL')
            saturated_mask = scl.neq(1)
            
            combined_mask = cloud_mask.And(saturated_mask)
            masked = img.updateMask(combined_mask)
            
            # Apply cropland mask
            landtype_mask = cropland_mask.reproject(
                crs=img.select('B4').projection(), scale=scale
            )
            landtype_valid = landtype_mask.eq(1)
            masked = masked.updateMask(landtype_valid)
            
            # Calculate NDTI
            return calculate_ndti(masked)
        
        ndti_collection = collection.map(mask_and_calculate)
        final_count = ndti_collection.size().getInfo()
        print(f"  ✅ Usable images: {final_count}")
        
        if final_count == 0:
            raise ValueError("No usable images after processing")
        
        # 5. Calculate temporal mean NDTI
        print("\n📈 Calculating temporal mean NDTI...")
        ndti_mean = ndti_collection.select('NDTI').mean()
        
        # 6. Get statistics
        print("  📊 Computing statistics...")
        stats = ndti_mean.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.minMax(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True
            ),
            geometry=bihar_polygon,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True
        ).getInfo()
        
        stats_dict = {
            'mean': stats.get('NDTI_mean', 0),
            'min': stats.get('NDTI_min', 0),
            'max': stats.get('NDTI_max', 0),
            'std': stats.get('NDTI_stdDev', 0),
            'image_count': final_count,
            'period': f'{start_date} to {end_date}'
        }
        
        print("\n" + "="*80)
        print("📊 NDTI STATISTICS:")
        print("="*80)
        print(f"Mean NDTI:  {stats_dict['mean']:.6f}")
        print(f"Min NDTI:   {stats_dict['min']:.6f}")
        print(f"Max NDTI:   {stats_dict['max']:.6f}")
        print(f"Std Dev:    {stats_dict['std']:.6f}")
        print(f"Images:     {stats_dict['image_count']}")
        print()
        
        # 7. Sample NDTI data and create maps
        print("📍 Sampling NDTI data for mapping...")
        data_array, lons, lats, bounds = sample_ndti_grid(
            ndti_mean, bihar_polygon, scale
        )
        
        if data_array is not None and not np.all(np.isnan(data_array)):
            print(f"✅ Data array shape: {data_array.shape}")
            print(f"   Valid pixels: {np.sum(~np.isnan(data_array))}/{data_array.size}")
            
            # Create map with satellite background
            map_path_sat = os.path.join(output_dir, 'bihar_ndti_satellite.png')
            print(f"\n🎨 Creating satellite map...")
            fig1 = plot_ndti_map(
                data_array, lons, lats, bounds, stats_dict,
                map_path_sat,
                f'Bihar NDTI Map - {start_date} to {end_date}',
                show_satellite=True
            )
            
            # Create map without satellite
            map_path_basic = os.path.join(output_dir, 'bihar_ndti_basic.png')
            print(f"\n🎨 Creating basic map...")
            fig2 = plot_ndti_map(
                data_array, lons, lats, bounds, stats_dict,
                map_path_basic,
                f'Bihar NDTI Map - {start_date} to {end_date}',
                show_satellite=False
            )
            
            print("\n" + "="*80)
            print("✅ VERIFICATION COMPLETE!")
            print("="*80)
            print(f"📁 Output directory: {os.path.abspath(output_dir)}")
            print(f"🗺️  Maps generated:")
            print(f"   • {os.path.basename(map_path_sat)}")
            print(f"   • {os.path.basename(map_path_basic)}")
            
            return stats_dict, data_array
        else:
            print("❌ Failed to sample NDTI data or no valid data returned")
            print("   This might happen if:")
            print("   - All pixels were masked (clouds/non-cropland)")
            print("   - Scale is too coarse")
            print("   - Region has no valid data for this period")
            return stats_dict, None
        
    except Exception as e:
        print(f"\n❌ Error in verification: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    print("🚀 Starting Bihar NDTI Verification with Mapping")
    print("   (Simplified version - no Google Drive authentication required)")
    print()
    
    # Run verification with default parameters
    # RESOLUTION OPTIONS:
    # scale=300: Very fast, blocky (good for testing)
    # scale=200: Fast, somewhat blocky (default)
    # scale=100: Slower, much better detail (RECOMMENDED)
    # scale=50:  Slow, very detailed (may hit GEE limits)
    # scale=10:  Very slow, maximum detail (Sentinel-2 native resolution)
    
    stats, data = verify_ndti_calculation(
        start_date='2017-09-01',
        end_date='2018-05-01',
        scale=10,  # Changed to 100m for better detail!
        output_dir='./ndti_verification'
    )
    
    if stats:
        print("\n✅ Verification successful!")
        print("Check the output directory for maps and statistics")
    else:
        print("\n❌ Verification failed. Check error messages above.")