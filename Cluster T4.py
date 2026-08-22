"""
Enhanced Cropland Phenology Clustering Analysis for Bihar, India
Combines temporal NDVI clustering with advanced visualization and multi-season detection
Period: 2019 (full year for complete seasonal cycle)
Features:
- Single vs. biannual cropping season detection
- Full time-series visualization with uncertainty envelopes
- Crop-free period identification
- Enhanced statistical analysis per cluster
"""

import ee
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import seaborn as sns
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Visualization settings
sns.set_context("talk")
sns.set_style("ticks")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
rcParams['axes.linewidth'] = 1.2 
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['lines.markeredgewidth'] = 1

# Initialize Google Earth Engine
PROJECT_ID = '590577866979'

try:
    ee.Initialize(project=PROJECT_ID)
    print("Google Earth Engine initialized successfully!")
except Exception as e:
    print(f"Error initializing GEE: {e}")

class LandType:
    """Land type classification using ESA WorldCover dataset."""
    def __init__(self, GEE_project_id='tlg-erosion1', DataRes=0.00009, EE_initialized=True):
        if not EE_initialized: 
            ee.Authenticate()
            ee.Initialize(project=GEE_project_id)
        
        worldcover = ee.ImageCollection('ESA/WorldCover/v200')
        self.worldcover = worldcover
        self.DataRes = DataRes

    def get_land_cover_for_region(self, Geometry):
        try:
            worldcover_image = self.worldcover.first()
            clipped = worldcover_image.clip(Geometry)
            return {'image': clipped}
        except Exception as e:
            print(f"Error getting land cover: {e}")
            return None

    def Map_LandType(self, landcover_image):
        try:
            cropland_mask = landcover_image.eq(40).rename('cropland_mask')
            return cropland_mask
        except Exception as e:
            print(f"Error mapping land type: {e}")
            return None

def get_bihar_polygon():
    """Create Bihar polygon geometry based on actual state boundaries"""
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

def calculate_ndvi(image):
    """Calculate NDVI: (NIR - Red) / (NIR + Red)"""
    try:
        nir = image.select('B8')
        red = image.select('B4')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        ndvi = ndvi.clamp(-1, 1)
        return image.addBands(ndvi)
    except Exception as e:
        print(f"Error calculating NDVI: {e}")
        return image

def add_modified_julian_date(image):
    """Add Modified Julian Date (MJD) as image property"""
    try:
        date = ee.Date(image.get('system:time_start'))
        unix_millis = date.millis()
        days_since_epoch = unix_millis.divide(86400000)
        julian_date = days_since_epoch.add(2440587.5)
        modified_julian_date = julian_date.subtract(2400000.5)
        return image.set('MJD', modified_julian_date)
    except Exception as e:
        print(f"Error adding MJD: {e}")
        return image

def mjd_to_doy(mjd, year=2019):
    """Convert Modified Julian Date to Day of Year"""
    # MJD 0 = Nov 17, 1858
    # Jan 1, 2019 = MJD 58484
    mjd_jan1 = 58484
    doy = mjd - mjd_jan1 + 1
    return doy

def detect_peaks_in_timeseries(doy_values, ndvi_values, prominence=0.05):
    """
    Detect peaks in NDVI time series to identify cropping seasons
    Returns number of significant peaks and their properties
    """
    try:
        # Find peaks with minimum prominence
        peaks, properties = find_peaks(ndvi_values, prominence=prominence, distance=30)
        
        return {
            'n_peaks': len(peaks),
            'peak_indices': peaks,
            'peak_doys': doy_values[peaks] if len(peaks) > 0 else [],
            'peak_ndvi': ndvi_values[peaks] if len(peaks) > 0 else [],
            'prominences': properties['prominences'] if len(peaks) > 0 else []
        }
    except Exception as e:
        print(f"Error detecting peaks: {e}")
        return {'n_peaks': 0, 'peak_indices': [], 'peak_doys': [], 'peak_ndvi': [], 'prominences': []}

def identify_crop_free_periods(doy_values, ndvi_values, threshold=0.3, min_duration=20):
    """
    Identify periods where NDVI is consistently low (bare soil)
    Returns list of (start_doy, end_doy) tuples
    """
    try:
        low_ndvi = ndvi_values < threshold
        crop_free_periods = []
        
        in_period = False
        period_start = None
        
        for i, is_low in enumerate(low_ndvi):
            if is_low and not in_period:
                period_start = doy_values[i]
                in_period = True
            elif not is_low and in_period:
                period_end = doy_values[i-1]
                if period_end - period_start >= min_duration:
                    crop_free_periods.append((period_start, period_end))
                in_period = False
        
        # Handle case where period extends to end
        if in_period and len(doy_values) > 0:
            period_end = doy_values[-1]
            if period_end - period_start >= min_duration:
                crop_free_periods.append((period_start, period_end))
        
        return crop_free_periods
    except Exception as e:
        print(f"Error identifying crop-free periods: {e}")
        return []

class PhenologyProcessor:
    """Enhanced Phenology Processing with Advanced Analysis"""
    
    def __init__(self, geometry, start_date, end_date, Verbose=True, SentRes=10):
        try:
            self.LT = LandType(EE_initialized=True)
            self.AoI_geom = geometry
            
            if self.AoI_geom is None:
                raise ValueError("Failed to create area of interest geometry")
            
            result = self.LT.get_land_cover_for_region(Geometry=self.AoI_geom)
            if result is None:
                raise ValueError("Failed to get land cover data")
                
            self.RegionMap = self.LT.Map_LandType(result['image'])
            if self.RegionMap is None:
                raise ValueError("Failed to create cropland mask")
            
            self.start_date = start_date
            self.end_date = end_date
            self.Verbose = Verbose
            self.SentRes = SentRes
            self.phenology_results = {}
            self.cluster_timeseries = {}
            
        except Exception as e:
            print(f"Error initializing PhenologyProcessor: {e}")
            raise

    def Pull_Process_Sentinel_data(self, QA_BAND='cs_cdf', CLEAR_THRESHOLD=0.65):
        """Process Sentinel-2 data with strict cloud masking"""
        try:
            def mask_clouds_strict(img):
                cs = img.select(QA_BAND)
                cloud_mask = cs.gte(CLEAR_THRESHOLD)
                scl = img.select('SCL')
                scl_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
                combined_mask = cloud_mask.And(scl_mask)
                return img.updateMask(combined_mask)
            
            def apply_landtype_mask(image):
                landtype_mask = self.RegionMap.reproject(
                    crs=image.select('B4').projection(), 
                    scale=self.SentRes
                )
                landtype_valid_mask = landtype_mask.eq(1)
                return image.updateMask(landtype_valid_mask)

            if self.Verbose:
                print(f"\n  Processing Bihar: {self.start_date} to {self.end_date}")
            
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

            filtered_s2 = (s2
                .filterBounds(self.AoI_geom)
                .filterDate(self.start_date, self.end_date)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
                .linkCollection(csPlus, [QA_BAND])
                .map(mask_clouds_strict)
                .map(apply_landtype_mask)
                .map(calculate_ndvi)
                .map(add_modified_julian_date))
            
            if self.Verbose:
                print(f"    ✅ Collection ready for temporal compositing")
            
            return filtered_s2
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return ee.ImageCollection([])

    def create_temporal_composites(self, collection, composite_days=30):
        """Create temporal composites"""
        try:
            if self.Verbose:
                print(f"\n  Creating {composite_days}-day temporal composites...")
            
            start_date = ee.Date(self.start_date)
            end_date = ee.Date(self.end_date)
            n_days = end_date.difference(start_date, 'day')
            n_composites = n_days.divide(composite_days).ceil()
            composite_indices = ee.List.sequence(0, n_composites.subtract(1))
            
            def create_composite(i):
                i = ee.Number(i)
                composite_start = start_date.advance(i.multiply(composite_days), 'day')
                composite_end = composite_start.advance(composite_days, 'day')
                
                period_collection = collection.filterDate(composite_start, composite_end)
                mean_ndvi = period_collection.select('NDVI').mean()
                mean_mjd = period_collection.aggregate_mean('MJD')
                
                return mean_ndvi.set({
                    'MJD': mean_mjd,
                    'system:time_start': composite_start.millis(),
                    'composite_start': composite_start.format('YYYY-MM-dd'),
                    'composite_end': composite_end.format('YYYY-MM-dd')
                })
            
            composites = ee.ImageCollection(composite_indices.map(create_composite))
            composites = composites.filter(ee.Filter.notNull(['MJD']))
            
            if self.Verbose:
                n_comp = composites.size().getInfo()
                print(f"    ✅ Created {n_comp} temporal composites")
            
            return composites
            
        except Exception as e:
            print(f"Error creating temporal composites: {e}")
            return ee.ImageCollection([])

    def calculate_phenology_metrics(self, collection):
        """Calculate peak_time, min_time, and mean_time for each pixel"""
        try:
            if self.Verbose:
                print(f"\n  Calculating phenology metrics...")
            
            def add_mjd_band(image):
                mjd = ee.Number(image.get('MJD'))
                mjd_band = ee.Image.constant(mjd).toFloat().rename('MJD')
                return image.addBands(mjd_band)
            
            collection_with_mjd_band = collection.map(add_mjd_band)
            
            # Peak time
            peak_time_image = collection_with_mjd_band.qualityMosaic('NDVI').select('MJD').rename('peak_time')
            
            # Min time
            def invert_ndvi(image):
                inverted = image.select('NDVI').multiply(-1).rename('NDVI_inverted')
                return image.addBands(inverted)
            
            collection_inverted = collection_with_mjd_band.map(invert_ndvi)
            min_time_image = collection_inverted.qualityMosaic('NDVI_inverted').select('MJD').rename('min_time')
            
            # Mean time
            mean_time_image = collection_with_mjd_band.select('MJD').mean().rename('mean_time')
            mean_ndvi = collection.select('NDVI').mean().rename('mean_NDVI')
            
            phenology_image = peak_time_image.addBands(min_time_image).addBands(mean_time_image).addBands(mean_ndvi)
            
            if self.Verbose:
                print(f"    ✅ Calculated phenology metrics")
            
            return phenology_image
            
        except Exception as e:
            print(f"Error calculating phenology metrics: {e}")
            return None

    def create_clusters(self, phenology_image, n_bins=15, cluster_type='peak_time'):
        """Assign pixels to bins based on phenology metrics"""
        try:
            if self.Verbose:
                print(f"\n  Creating {n_bins} clusters based on {cluster_type}...")
            
            metric_band = phenology_image.select(cluster_type)
            
            stats = metric_band.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=self.AoI_geom,
                scale=100,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=8
            ).getInfo()
            
            min_val = stats[f'{cluster_type}_min']
            max_val = stats[f'{cluster_type}_max']
            
            if self.Verbose:
                print(f"    {cluster_type} range: {min_val:.2f} to {max_val:.2f} MJD")
            
            bin_width = (max_val - min_val) / n_bins
            cluster_image = metric_band.subtract(min_val).divide(bin_width).floor().rename('cluster')
            cluster_image = cluster_image.clamp(0, n_bins - 1)
            
            result_image = phenology_image.addBands(cluster_image)
            
            histogram = cluster_image.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=self.AoI_geom,
                scale=100,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=8
            ).getInfo()
            
            cluster_stats = histogram.get('cluster', {})
            cluster_stats = {int(float(k)): v for k, v in cluster_stats.items()}
            
            if self.Verbose:
                print(f"    ✅ Created {n_bins} clusters")
            
            return {
                'image': result_image,
                'n_bins': n_bins,
                'cluster_type': cluster_type,
                'min_val': min_val,
                'max_val': max_val,
                'bin_width': bin_width,
                'cluster_stats': cluster_stats
            }
            
        except Exception as e:
            print(f"Error creating clusters: {e}")
            return None

    def extract_cluster_timeseries(self, collection, cluster_result, sample_size=1000):
        """
        Extract NDVI time series for each cluster using server-side statistics
        Returns mean, std, and percentiles for visualization
        OPTIMIZED: Uses reduceRegion instead of sampling to avoid reprojection issues
        """
        try:
            if self.Verbose:
                print(f"\n  Extracting time series for each cluster...")
            
            cluster_image = cluster_result['image'].select('cluster')
            n_bins = cluster_result['n_bins']
            
            # Get list of composites with MJD
            composite_list = collection.toList(collection.size())
            n_composites = collection.size().getInfo()
            
            mjd_values = []
            for i in range(n_composites):
                img = ee.Image(composite_list.get(i))
                mjd = img.get('MJD').getInfo()
                mjd_values.append(mjd)
            
            mjd_values = np.array(mjd_values)
            doy_values = mjd_to_doy(mjd_values)
            
            # Sort by DOY
            sort_idx = np.argsort(doy_values)
            doy_values = doy_values[sort_idx]
            mjd_values = mjd_values[sort_idx]
            
            cluster_data = {}
            
            for cluster_id in range(n_bins):
                if self.Verbose:
                    print(f"    Processing cluster {cluster_id}...")
                
                # Create mask for this cluster
                cluster_mask = cluster_image.eq(cluster_id)
                
                # Extract NDVI statistics for each composite using server-side operations
                mean_values = []
                std_values = []
                p25_values = []
                p75_values = []
                p05_values = []
                p95_values = []
                
                for idx in sort_idx:
                    img = ee.Image(composite_list.get(int(idx)))
                    ndvi_band = img.select('NDVI')
                    
                    # Mask NDVI to only this cluster
                    masked_ndvi = ndvi_band.updateMask(cluster_mask)
                    
                    # Calculate statistics using reduceRegion with LARGE scale to prevent timeout
                    # Use scale=500 (500m pixels) instead of 100m to massively reduce computation
                    stats = masked_ndvi.reduceRegion(
                        reducer=ee.Reducer.mean().combine(
                            reducer2=ee.Reducer.stdDev(),
                            sharedInputs=True
                        ).combine(
                            reducer2=ee.Reducer.percentile([5, 25, 75, 95]),
                            sharedInputs=True
                        ),
                        geometry=self.AoI_geom,
                        scale=500,  # INCREASED from 100 to 500 to prevent reprojection error
                        maxPixels=1e9,
                        bestEffort=True,
                        tileScale=16  # INCREASED from 8 to 16 for better performance
                    ).getInfo()
                    
                    # Extract values (with fallback for missing data)
                    mean_values.append(stats.get('NDVI_mean', np.nan))
                    std_values.append(stats.get('NDVI_stdDev', 0))
                    p05_values.append(stats.get('NDVI_p5', np.nan))
                    p25_values.append(stats.get('NDVI_p25', np.nan))
                    p75_values.append(stats.get('NDVI_p75', np.nan))
                    p95_values.append(stats.get('NDVI_p95', np.nan))
                
                # Convert to numpy arrays
                mean_ndvi = np.array(mean_values)
                std_ndvi = np.array(std_values)
                p25 = np.array(p25_values)
                p75 = np.array(p75_values)
                p05 = np.array(p05_values)
                p95 = np.array(p95_values)
                
                cluster_data[cluster_id] = {
                    'doy': doy_values,
                    'mean': mean_ndvi,
                    'std': std_ndvi,
                    'p25': p25,
                    'p75': p75,
                    'p05': p05,
                    'p95': p95,
                    'n_pixels': cluster_result['cluster_stats'].get(cluster_id, 0)
                }
            
            self.cluster_timeseries = cluster_data
            
            if self.Verbose:
                print(f"    ✅ Extracted time series for {n_bins} clusters")
            
            return cluster_data
            
        except Exception as e:
            print(f"Error extracting cluster timeseries: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def process(self, composite_days=30, n_bins=15, cluster_type='peak_time', extract_timeseries=True):
        """Full processing pipeline"""
        try:
            print("\n" + "="*80)
            print("ENHANCED PHENOLOGY PROCESSING PIPELINE")
            print("="*80)
            
            collection = self.Pull_Process_Sentinel_data()
            composites = self.create_temporal_composites(collection, composite_days)
            
            n_composites = composites.size().getInfo()
            if n_composites == 0:
                print("❌ No composites created")
                return None
            
            phenology_image = self.calculate_phenology_metrics(composites)
            if phenology_image is None:
                return None
            
            cluster_result = self.create_clusters(phenology_image, n_bins, cluster_type)
            if cluster_result is None:
                return None
            
            self.phenology_results = cluster_result
            
            # Extract time series if requested
            if extract_timeseries:
                self.extract_cluster_timeseries(composites, cluster_result)
            
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            
            return cluster_result
            
        except Exception as e:
            print(f"Error in processing pipeline: {e}")
            return None

def analyze_cluster_seasonality(cluster_data):
    """
    Analyze each cluster for single vs. biannual cropping patterns
    Returns detailed analysis for each cluster
    """
    analyses = {}
    
    for cluster_id, data in cluster_data.items():
        doy = data['doy']
        mean_ndvi = data['mean']
        
        # Detect peaks
        peak_info = detect_peaks_in_timeseries(doy, mean_ndvi, prominence=0.05)
        
        # Identify crop-free periods
        crop_free = identify_crop_free_periods(doy, mean_ndvi, threshold=0.3, min_duration=20)
        
        # Classify cropping pattern
        if peak_info['n_peaks'] == 0:
            pattern = 'No clear pattern'
        elif peak_info['n_peaks'] == 1:
            pattern = 'Single-season cropping'
        elif peak_info['n_peaks'] == 2:
            pattern = 'Potential biannual cropping'
        else:
            pattern = f'Complex pattern ({peak_info["n_peaks"]} peaks)'
        
        analyses[cluster_id] = {
            'pattern': pattern,
            'n_peaks': peak_info['n_peaks'],
            'peak_doys': peak_info['peak_doys'],
            'peak_ndvi': peak_info['peak_ndvi'],
            'crop_free_periods': crop_free,
            'n_pixels': data['n_pixels']
        }
    
    return analyses

def plot_enhanced_visualization(processor, cluster_result, start_date, end_date, analyses):
    """
    Create comprehensive visualization with:
    - Large cluster distribution chart
    - Top 3 clusters with time series
    - Clear summary statistics
    SIMPLIFIED: Focus on most important information
    """
    cluster_data = processor.cluster_timeseries
    n_bins = cluster_result['n_bins']
    
    # Filter out clusters with insufficient data
    valid_clusters = {}
    for cluster_id, data in cluster_data.items():
        mean_list = data['mean']
        try:
            mean_clean = [x if x is not None else np.nan for x in mean_list]
            mean_array = np.array(mean_clean, dtype=float)
            
            if np.any(~np.isnan(mean_array)):
                valid_clusters[cluster_id] = data
        except Exception as e:
            print(f"    Warning: Skipping cluster {cluster_id} due to data issues: {e}")
            continue
    
    if len(valid_clusters) == 0:
        print("❌ No valid cluster data for visualization")
        return
    
    print(f"    Visualizing top 3 clusters from {len(valid_clusters)} valid clusters...")
    
    # Get top 3 clusters by pixel count
    cluster_stats = cluster_result['cluster_stats']
    top_clusters = sorted(cluster_stats.items(), key=lambda x: x[1], reverse=True)[:3]
    top_cluster_ids = [cid for cid, _ in top_clusters]
    
    # Build summary statistics
    single_season = sum(1 for a in analyses.values() if a['n_peaks'] == 1)
    biannual = sum(1 for a in analyses.values() if a['n_peaks'] == 2)
    complex_pattern = sum(1 for a in analyses.values() if a['n_peaks'] > 2)
    no_pattern = sum(1 for a in analyses.values() if a['n_peaks'] == 0)
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(28, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35, 
                         width_ratios=[1.2, 1], height_ratios=[1, 1])
    
    # Title
    fig.suptitle(f'Enhanced Bihar Phenology Clustering Analysis\n{start_date} to {end_date}', 
                 fontsize=28, fontweight='bold', y=0.98)
    
    # ===== TOP LEFT: LARGE CLUSTER DISTRIBUTION =====
    ax_dist = fig.add_subplot(gs[0, 0])
    
    bins = sorted(cluster_stats.keys())
    counts = [cluster_stats[i] for i in bins]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(bins)))
    bars = ax_dist.bar(bins, counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
    
    # Highlight top 3 clusters
    for i, cluster_id in enumerate(bins):
        if cluster_id in top_cluster_ids:
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(3)
    
    ax_dist.set_xlabel('Cluster ID', fontweight='bold', fontsize=20)
    ax_dist.set_ylabel('Pixel Count', fontweight='bold', fontsize=20)
    ax_dist.set_title('Cluster Distribution (Top 3 Highlighted in Red)', 
                     fontweight='bold', fontsize=22, pad=15)
    ax_dist.grid(True, alpha=0.3, axis='y', linewidth=1.5)
    ax_dist.tick_params(labelsize=18, width=2, length=8)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax_dist.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height/1000)}K',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # ===== TOP RIGHT: SUMMARY STATISTICS =====
    ax_summary = fig.add_subplot(gs[0, 1])
    ax_summary.axis('off')
    
    summary_text = "SEASONALITY SUMMARY\n" + "="*45 + "\n\n"
    summary_text += f"Total Clusters: {len(cluster_stats)}\n"
    summary_text += f"Valid Clusters: {len(valid_clusters)}\n\n"
    summary_text += "PATTERN CLASSIFICATION\n" + "-"*45 + "\n"
    summary_text += f"Single-season:  {single_season:2d} clusters\n"
    summary_text += f"Biannual:       {biannual:2d} clusters\n"
    summary_text += f"Complex:        {complex_pattern:2d} clusters\n"
    summary_text += f"No pattern:     {no_pattern:2d} clusters\n\n"
    
    total_pixels = sum(cluster_stats.values())
    summary_text += "TOP 3 CLUSTERS\n" + "="*45 + "\n\n"
    
    for rank, (cluster_id, count) in enumerate(top_clusters, 1):
        pct = 100 * count / total_pixels
        analysis = analyses.get(cluster_id, {})
        pattern = analysis.get('pattern', 'Unknown')
        pattern_short = pattern.replace('Single-season cropping', 'Single-season')
        
        summary_text += f"#{rank} Cluster {cluster_id}:\n"
        summary_text += f"   {int(count):,} pixels ({pct:.1f}%)\n"
        summary_text += f"   {pattern_short}\n"
        
        if 'peak_doys' in analysis and len(analysis['peak_doys']) > 0:
            peaks = ', '.join([str(int(d)) for d in analysis['peak_doys']])
            summary_text += f"   Peak DOY: {peaks}\n"
        summary_text += "\n"
    
    ax_summary.text(0.05, 0.95, summary_text,
                   transform=ax_summary.transAxes,
                   fontsize=16,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=1.5', 
                            facecolor='#E8F4F8',
                            edgecolor='#4A90E2',
                            linewidth=3,
                            alpha=0.9))
    
    # ===== BOTTOM ROW: TOP 3 CLUSTER TIME SERIES =====
    for plot_idx, cluster_id in enumerate(top_cluster_ids):
        if cluster_id not in valid_clusters:
            continue
        
        # Position: bottom left (0,0), bottom center (1,0), or span bottom right
        if plot_idx == 0:
            ax = fig.add_subplot(gs[1, 0])
        elif plot_idx == 1:
            ax = fig.add_subplot(gs[1, 1])
        else:
            # Third cluster - create a new row or skip
            continue
    
        data = cluster_data[cluster_id]
        analysis = analyses[cluster_id]
        
        # Convert to numpy arrays, handling None values
        doy = np.array(data['doy'])
        mean_list = [x if x is not None else np.nan for x in data['mean']]
        mean = np.array(mean_list, dtype=float)
        std_list = [x if x is not None else 0.0 for x in data['std']]
        std = np.array(std_list, dtype=float)
        
        # Create masks for valid data
        valid_mask = ~np.isnan(mean)
        
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, f'Cluster {cluster_id}\nNo valid data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=18)
            ax.set_xlim([0, 365])
            ax.set_ylim([0, 1])
            continue
        
        # Filter to valid data only
        doy_valid = doy[valid_mask]
        mean_valid = mean[valid_mask]
        std_valid = std[valid_mask]
        std_valid = np.nan_to_num(std_valid, nan=0.0)
        
        # Plot mean line
        ax.plot(doy_valid, mean_valid, 'k-', linewidth=3.5, label='Mean NDVI', zorder=10)
        
        # Plot ±1σ envelope
        lower_1sig = np.maximum(mean_valid - std_valid, 0)
        upper_1sig = np.minimum(mean_valid + std_valid, 1)
        ax.fill_between(doy_valid, lower_1sig, upper_1sig,
                        color='blue', alpha=0.4, label='±1σ', zorder=5)
        
        # Plot ±2σ envelope
        lower_2sig = np.maximum(mean_valid - 2*std_valid, 0)
        upper_2sig = np.minimum(mean_valid + 2*std_valid, 1)
        ax.fill_between(doy_valid, lower_2sig, upper_2sig,
                        color='blue', alpha=0.2, label='±2σ', zorder=3)
        
        # Mark peaks
        if len(analysis['peak_doys']) > 0:
            peak_doys_array = np.array(analysis['peak_doys'])
            peak_ndvi_array = np.array(analysis['peak_ndvi'])
            valid_peaks_mask = np.array([doy in doy_valid for doy in peak_doys_array])
            if np.any(valid_peaks_mask):
                ax.plot(peak_doys_array[valid_peaks_mask], peak_ndvi_array[valid_peaks_mask], 
                       'r*', markersize=25, markeredgewidth=2, markeredgecolor='darkred',
                       label='Peaks', zorder=15)
        
        # Mark crop-free periods
        for period_idx, period in enumerate(analysis['crop_free_periods']):
            label = 'Crop-free' if period_idx == 0 else ''
            ax.axvspan(period[0], period[1], alpha=0.3, color='orange', label=label, zorder=1)
        
        # Formatting
        ax.set_xlabel('Day of Year', fontweight='bold', fontsize=18)
        ax.set_ylabel('NDVI', fontweight='bold', fontsize=18)
        
        pattern_text = analysis["pattern"]
        pixels_text = f'({int(data["n_pixels"]):,} pixels)'
        rank = top_cluster_ids.index(cluster_id) + 1
        ax.set_title(f'#{rank} - Cluster {cluster_id}: {pattern_text}\n{pixels_text}', 
                    fontweight='bold', fontsize=20, pad=12)
        
        ax.set_xlim([0, 365])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.35, linewidth=1.2)
        ax.tick_params(labelsize=16, width=2, length=6)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=14, framealpha=0.95, edgecolor='black', 
                 fancybox=True, shadow=True)
    
    plt.savefig('bihar_enhanced_phenology_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Enhanced visualization saved as 'bihar_enhanced_phenology_analysis.png'")
    plt.show()

def main_enhanced_analysis():
    """Main processing pipeline with enhanced analysis"""
    
    start_date = '2019-01-01'
    end_date = '2019-12-31'
    
    print("🌾 Enhanced Bihar Phenology Clustering Analysis")
    print("=" * 80)
    print(f"📍 Region: Bihar, India")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🔬 Features:")
    print("   • Single vs. biannual cropping detection")
    print("   • Crop-free period identification")
    print("   • Time-series visualization with uncertainty")
    print("   • Peak detection and pattern analysis")
    print()
    
    try:
        bihar_polygon = get_bihar_polygon()
        if bihar_polygon is None:
            raise ValueError("Failed to create Bihar polygon")
        
        print("🗺️ Bihar Polygon Created (13 vertices)")
        print()
        
        # Initialize processor
        print("🔄 Starting phenology processing...")
        processor = PhenologyProcessor(
            geometry=bihar_polygon,
            start_date=start_date,
            end_date=end_date,
            Verbose=True
        )
        
        # Run processing pipeline
        result = processor.process(
            composite_days=30,
            n_bins=15,
            cluster_type='peak_time',
            extract_timeseries=True
        )
        
        if result and processor.cluster_timeseries:
            print("\n" + "="*80)
            print("✅ PROCESSING COMPLETED!")
            print("="*80)
            
            # Analyze seasonality patterns
            print("\n🔍 Analyzing crop seasonality patterns...")
            analyses = analyze_cluster_seasonality(processor.cluster_timeseries)
            
            # Print detailed analysis
            print("\n" + "="*80)
            print("SEASONALITY ANALYSIS RESULTS")
            print("="*80)
            
            single_season = []
            biannual = []
            complex_patterns = []
            
            for cluster_id, analysis in analyses.items():
                print(f"\nCluster {cluster_id} ({int(analysis['n_pixels']):,} pixels):")
                print(f"  Pattern: {analysis['pattern']}")
                print(f"  Number of peaks: {analysis['n_peaks']}")
                
                if len(analysis['peak_doys']) > 0:
                    peak_info = ", ".join([f"DOY {int(d)} (NDVI={v:.3f})" 
                                          for d, v in zip(analysis['peak_doys'], 
                                                         analysis['peak_ndvi'])])
                    print(f"  Peak details: {peak_info}")
                
                if len(analysis['crop_free_periods']) > 0:
                    print(f"  Crop-free periods:")
                    for start, end in analysis['crop_free_periods']:
                        print(f"    - DOY {int(start)} to {int(end)} ({int(end-start)} days)")
                else:
                    print(f"  Crop-free periods: None detected")
                
                # Categorize
                if analysis['n_peaks'] == 1:
                    single_season.append(cluster_id)
                elif analysis['n_peaks'] == 2:
                    biannual.append(cluster_id)
                elif analysis['n_peaks'] > 2:
                    complex_patterns.append(cluster_id)
            
            # Summary statistics
            print("\n" + "="*80)
            print("SUMMARY STATISTICS")
            print("="*80)
            print(f"Total clusters analyzed: {len(analyses)}")
            print(f"Single-season cropping: {len(single_season)} clusters")
            if single_season:
                print(f"  Cluster IDs: {single_season}")
            print(f"Biannual cropping: {len(biannual)} clusters")
            if biannual:
                print(f"  Cluster IDs: {biannual}")
            print(f"Complex patterns: {len(complex_patterns)} clusters")
            if complex_patterns:
                print(f"  Cluster IDs: {complex_patterns}")
            
            # Calculate total pixels in each category
            total_pixels = sum(a['n_pixels'] for a in analyses.values())
            single_pixels = sum(analyses[c]['n_pixels'] for c in single_season)
            biannual_pixels = sum(analyses[c]['n_pixels'] for c in biannual)
            complex_pixels = sum(analyses[c]['n_pixels'] for c in complex_patterns)
            
            print(f"\nPixel distribution:")
            print(f"  Single-season: {int(single_pixels):,} ({100*single_pixels/total_pixels:.1f}%)")
            print(f"  Biannual: {int(biannual_pixels):,} ({100*biannual_pixels/total_pixels:.1f}%)")
            print(f"  Complex: {int(complex_pixels):,} ({100*complex_pixels/total_pixels:.1f}%)")
            
            # Generate visualization
            print("\n📊 Generating enhanced visualization...")
            plot_enhanced_visualization(processor, result, start_date, end_date, analyses)
            
            # Additional warnings for biannual patterns
            if len(biannual) > 0:
                print("\n⚠️  WARNING: Biannual cropping patterns detected!")
                print("   These clusters show two distinct growing seasons.")
                print("   Interpretation notes:")
                print("   - Peak timing may represent primary or secondary crop")
                print("   - Crop-free periods between peaks indicate harvest/tillage")
                print("   - Consider sub-clustering these groups for detailed analysis")
            
            return result, analyses
        else:
            print("❌ Failed to complete phenology analysis")
            return None, None
        
    except Exception as e:
        print(f"❌ Error in main analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    try:
        print("🚀 Starting Enhanced Bihar Phenology Clustering Analysis")
        print("🌾 Region: Bihar, India")
        print("📅 Period: 2019 (Full year)")
        print("🎯 Goal: Classify cropland by phenology with seasonality detection")
        print()
        
        result, analyses = main_enhanced_analysis()
        
        if result and analyses:
            print("\n" + "="*80)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("✅ Key Outputs:")
            print("   • Temporal NDVI composites created")
            print("   • Phenology metrics calculated (peak, min, mean timing)")
            print("   • Pixels clustered into distinct phenology groups")
            print("   • Seasonality patterns analyzed (single vs. biannual)")
            print("   • Crop-free periods identified")
            print("   • Time-series with uncertainty envelopes visualized")
            print()
            print("📊 Output: bihar_enhanced_phenology_analysis.png")
            print()
            print("💡 Next Steps:")
            print("   1. Review biannual clusters for detailed sub-analysis")
            print("   2. Validate crop-free periods against ground truth")
            print("   3. Consider seasonal stratification for model training")
            print("   4. Export cluster assignments for spatial analysis")
        else:
            print("\n❌ Analysis incomplete - check error messages above")
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        import traceback
        traceback.print_exc()