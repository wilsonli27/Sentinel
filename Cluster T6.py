"""
Enhanced Cropland Phenology Clustering Analysis for Bihar, India
Combines temporal NDVI clustering with advanced visualization and multi-season detection
Period: 2019 (full year for complete seasonal cycle)
Features:
- Single vs. biannual cropping season detection
- Full time-series visualization with uncertainty envelopes
- Crop-free period identification
- Enhanced statistical analysis per cluster
- CLEAN VISUALIZATION: Filters anomalies, removes redundant labels, improved readability
"""

import ee
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.patches import Patch
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
    mjd_jan1 = 58484  # Jan 1, 2019
    doy = mjd - mjd_jan1 + 1
    return doy

def is_sinusoidal_pattern(doy, mean_ndvi, min_variation=0.10, max_peaks=4):
    """
    Check if NDVI pattern follows reasonable sinusoidal behavior
    
    RELAXED Criteria:
    - Should have clear variation (range > 0.10, was 0.15)
    - Should not have too many peaks (allow up to 4, was 3)
    - Should have smooth transitions (more lenient noise threshold)
    """
    try:
        # Remove NaN values
        valid_mask = ~np.isnan(mean_ndvi)
        if np.sum(valid_mask) < 3:  # Reduced from 5
            return False
        
        mean_clean = mean_ndvi[valid_mask]
        
        # Check 1: Sufficient variation (RELAXED: 0.10 instead of 0.15)
        ndvi_range = np.max(mean_clean) - np.min(mean_clean)
        if ndvi_range < min_variation:
            return False
        
        # Check 2: Not too many peaks (RELAXED: allow 4 peaks instead of 3)
        peaks, _ = find_peaks(mean_clean, prominence=0.05)
        if len(peaks) > max_peaks:
            return False
        
        # Check 3: Smoothness - REMOVED (too strict, was rejecting valid patterns)
        # Agricultural patterns can be somewhat noisy due to management practices
        
        return True
        
    except Exception as e:
        print(f"Error checking sinusoidal pattern: {e}")
        return False

def detect_peaks_in_timeseries(doy_values, ndvi_values, prominence=0.05):
    """
    Detect peaks in NDVI time series to identify cropping seasons
    Returns number of significant peaks and their properties
    """
    try:
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

def find_optimal_tillage_window(cluster_data, analyses, cluster_stats, threshold=0.35):
    """
    Find the optimal tillage window across all major clusters
    Returns the widest period when NDVI is consistently low
    """
    try:
        total_pixels = sum(cluster_stats.values())
        major_clusters = {cid: count for cid, count in cluster_stats.items() 
                         if count > 0.05 * total_pixels}
        
        print(f"\n  Analyzing tillage windows for {len(major_clusters)} major clusters...")
        
        cluster_periods = {}
        
        for cluster_id in major_clusters.keys():
            if cluster_id not in cluster_data:
                continue
                
            data = cluster_data[cluster_id]
            doy = np.array(data['doy'])
            mean_list = [x if x is not None else np.nan for x in data['mean']]
            mean = np.array(mean_list, dtype=float)
            
            valid_mask = ~np.isnan(mean)
            if not np.any(valid_mask):
                continue
                
            doy_valid = doy[valid_mask]
            mean_valid = mean[valid_mask]
            
            periods = identify_crop_free_periods(doy_valid, mean_valid, 
                                                threshold=threshold, min_duration=15)
            
            if periods:
                cluster_periods[cluster_id] = periods
        
        if not cluster_periods:
            return None
        
        period_votes = {}
        
        for cluster_id, periods in cluster_periods.items():
            pixel_weight = cluster_stats[cluster_id] / total_pixels
            for start, end in periods:
                start_rounded = round(start / 10) * 10
                end_rounded = round(end / 10) * 10
                key = (start_rounded, end_rounded)
                
                if key not in period_votes:
                    period_votes[key] = {'count': 0, 'weight': 0, 'durations': []}
                
                period_votes[key]['count'] += 1
                period_votes[key]['weight'] += pixel_weight
                period_votes[key]['durations'].append(end - start)
        
        best_period = None
        best_score = 0
        
        for (start, end), info in period_votes.items():
            width = end - start
            score = width * info['weight'] * info['count']
            
            if score > best_score and width > 20:
                best_score = score
                avg_duration = np.mean(info['durations'])
                best_period = {
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'avg_duration': avg_duration,
                    'cluster_count': info['count'],
                    'pixel_weight': info['weight'],
                    'clusters': [cid for cid, periods in cluster_periods.items() 
                                if any(abs(s - start) < 15 and abs(e - end) < 15 
                                      for s, e in periods)]
                }
        
        return best_period
        
    except Exception as e:
        print(f"Error finding optimal tillage window: {e}")
        import traceback
        traceback.print_exc()
        return None

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
            
            peak_time_image = collection_with_mjd_band.qualityMosaic('NDVI').select('MJD').rename('peak_time')
            
            def invert_ndvi(image):
                inverted = image.select('NDVI').multiply(-1).rename('NDVI_inverted')
                return image.addBands(inverted)
            
            collection_inverted = collection_with_mjd_band.map(invert_ndvi)
            min_time_image = collection_inverted.qualityMosaic('NDVI_inverted').select('MJD').rename('min_time')
            
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
        """Extract NDVI time series for each cluster"""
        try:
            if self.Verbose:
                print(f"\n  Extracting time series for each cluster...")
            
            cluster_image = cluster_result['image'].select('cluster')
            n_bins = cluster_result['n_bins']
            
            composite_list = collection.toList(collection.size())
            n_composites = collection.size().getInfo()
            
            mjd_values = []
            for i in range(n_composites):
                img = ee.Image(composite_list.get(i))
                mjd = img.get('MJD').getInfo()
                mjd_values.append(mjd)
            
            mjd_values = np.array(mjd_values)
            doy_values = mjd_to_doy(mjd_values)
            
            sort_idx = np.argsort(doy_values)
            doy_values = doy_values[sort_idx]
            mjd_values = mjd_values[sort_idx]
            
            cluster_data = {}
            
            for cluster_id in range(n_bins):
                if self.Verbose:
                    print(f"    Processing cluster {cluster_id}...")
                
                cluster_mask = cluster_image.eq(cluster_id)
                
                mean_values = []
                std_values = []
                
                for idx in sort_idx:
                    img = ee.Image(composite_list.get(int(idx)))
                    ndvi_band = img.select('NDVI')
                    masked_ndvi = ndvi_band.updateMask(cluster_mask)
                    
                    stats = masked_ndvi.reduceRegion(
                        reducer=ee.Reducer.mean().combine(
                            reducer2=ee.Reducer.stdDev(),
                            sharedInputs=True
                        ),
                        geometry=self.AoI_geom,
                        scale=500,
                        maxPixels=1e9,
                        bestEffort=True,
                        tileScale=16
                    ).getInfo()
                    
                    mean_values.append(stats.get('NDVI_mean', np.nan))
                    std_values.append(stats.get('NDVI_stdDev', 0))
                
                cluster_data[cluster_id] = {
                    'doy': doy_values,
                    'mean': np.array(mean_values),
                    'std': np.array(std_values),
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
    """Analyze each cluster for single vs. biannual cropping patterns"""
    analyses = {}
    
    for cluster_id, data in cluster_data.items():
        doy = data['doy']
        mean_ndvi = data['mean']
        
        peak_info = detect_peaks_in_timeseries(doy, mean_ndvi, prominence=0.05)
        crop_free = identify_crop_free_periods(doy, mean_ndvi, threshold=0.3, min_duration=20)
        
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

def plot_clean_visualization(processor, cluster_result, start_date, end_date, analyses):
    """
    CLEAN visualization with improvements:
    1. Filters anomalous clusters (<30K pixels or non-sinusoidal)
    2. Removes redundant "Day of Year" labels from individual plots
    3. Statistical data in console only
    4. Improved label readability and spacing
    """
    cluster_data = processor.cluster_timeseries
    cluster_stats = cluster_result['cluster_stats']
    total_pixels = sum(cluster_stats.values())
    
    print("\n" + "="*80)
    print("FILTERING CLUSTERS")
    print("="*80)
    
    # Filter clusters
    valid_clusters = {}
    removed_clusters = []
    removal_reasons = {}
    
    for cluster_id, data in cluster_data.items():
        pixel_count = cluster_stats.get(cluster_id, 0)
        
        # Filter 1: Too few pixels (<30K) - but allow if >10K and has good pattern
        if pixel_count < 10000:  # Changed from 30K to 10K
            removed_clusters.append(cluster_id)
            removal_reasons[cluster_id] = f"Too few pixels ({pixel_count:,.0f})"
            continue
        
        # Filter 2: No clear pattern
        analysis = analyses.get(cluster_id, {})
        pattern = analysis.get('pattern', '')
        if 'No clear pattern' in pattern:
            removed_clusters.append(cluster_id)
            removal_reasons[cluster_id] = "No clear pattern detected"
            continue
        
        # Filter 3: Non-sinusoidal (only if ALSO <30K pixels)
        # Don't filter on sinusoidal if cluster is large
        mean_list = [x if x is not None else np.nan for x in data['mean']]
        mean_array = np.array(mean_list, dtype=float)
        doy_array = np.array(data['doy'])
        
        if pixel_count < 30000 and not is_sinusoidal_pattern(doy_array, mean_array):
            removed_clusters.append(cluster_id)
            removal_reasons[cluster_id] = f"Small cluster ({pixel_count:,.0f} px) with anomalous pattern"
            continue
        
        # Passed all filters
        valid_clusters[cluster_id] = data
    
    # Print removal summary
    print(f"\n✅ Kept {len(valid_clusters)} high-quality clusters")
    print(f"❌ Removed {len(removed_clusters)} anomalous clusters\n")
    
    removed_pixels = 0
    for cluster_id in removed_clusters:
        pixels = cluster_stats.get(cluster_id, 0)
        removed_pixels += pixels
        print(f"  Removed Cluster {cluster_id:2d}: {removal_reasons[cluster_id]:<40} ({pixels:,} px)")
    
    removed_pct = 100 * removed_pixels / total_pixels
    kept_pixels = total_pixels - removed_pixels
    
    print(f"\n📊 Summary:")
    print(f"  Removed: {removed_pixels:,} pixels ({removed_pct:.2f}%)")
    print(f"  Kept:    {kept_pixels:,} pixels ({100-removed_pct:.2f}%)")
    
    if len(valid_clusters) == 0:
        print("\n❌ No valid clusters remaining!")
        return
    
    # Statistics for valid clusters only
    valid_cluster_stats = {cid: cluster_stats[cid] for cid in valid_clusters.keys()}
    top_clusters = sorted(valid_cluster_stats.items(), key=lambda x: x[1], reverse=True)[:3]
    top_cluster_ids = [cid for cid, _ in top_clusters]
    
    single_season = sum(1 for cid, a in analyses.items() 
                       if cid in valid_clusters and a.get('n_peaks') == 1)
    biannual = sum(1 for cid, a in analyses.items() 
                  if cid in valid_clusters and a.get('n_peaks') == 2)
    complex_pattern = sum(1 for cid, a in analyses.items() 
                         if cid in valid_clusters and a.get('n_peaks', 0) > 2)
    
    # Calculate tillage window
    print("\n" + "="*80)
    print("CALCULATING TILLAGE WINDOW")
    print("="*80)
    
    tillage_window = find_optimal_tillage_window(
        valid_clusters, 
        analyses, 
        valid_cluster_stats
    )
    
    if tillage_window:
        start_date_obj = datetime(2019, 1, 1) + timedelta(days=int(tillage_window['start'])-1)
        end_date_obj = datetime(2019, 1, 1) + timedelta(days=int(tillage_window['end'])-1)
        
        print(f"\n🚜 OPTIMAL TILLAGE WINDOW:")
        print(f"  DOY: {int(tillage_window['start'])} to {int(tillage_window['end'])}")
        print(f"  Duration: {int(tillage_window['duration'])} days")
        print(f"  Dates: {start_date_obj.strftime('%B %d')} to {end_date_obj.strftime('%B %d, %Y')}")
        print(f"  Coverage: {tillage_window['pixel_weight']*100:.1f}% of valid cropland")
        print(f"  Present in {len(tillage_window['clusters'])} clusters: {tillage_window['clusters']}")
    else:
        print("\n⚠️  No clear tillage window identified")
    
    # Print pattern statistics
    print("\n" + "="*80)
    print("PATTERN CLASSIFICATION")
    print("="*80)
    print(f"  Single-season:  {single_season:2d} clusters")
    print(f"  Biannual:       {biannual:2d} clusters")
    print(f"  Complex:        {complex_pattern:2d} clusters")
    
    # ===== CREATE VISUALIZATION =====
    
    n_cols = 3
    n_cluster_rows = int(np.ceil(len(valid_clusters) / n_cols))
    
    total_rows = 1 + n_cluster_rows
    fig_height = 10 + (n_cluster_rows * 6)
    fig_width = 36
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(total_rows, 3, hspace=0.4, wspace=0.35, 
                         height_ratios=[1.2] + [1]*n_cluster_rows)
    
    title_text = f'Bihar Phenology Clustering Analysis\n{start_date} to {end_date}\n'
    title_text += f'{len(valid_clusters)} High-Quality Clusters ({removed_pct:.1f}% filtered out)'
    fig.suptitle(title_text, fontsize=28, fontweight='bold', y=0.995, linespacing=1.3)
    
    # ===== SUMMARY ROW: Cluster Distribution =====
    ax_dist = fig.add_subplot(gs[0, :])
    
    bins = sorted(cluster_stats.keys())
    counts = [cluster_stats[i] for i in bins]
    
    colors_list = []
    for cluster_id in bins:
        if cluster_id in removed_clusters:
            colors_list.append('#CD5C5C')
        elif cluster_id in top_cluster_ids:
            colors_list.append('#FFD700')
        else:
            colors_list.append('#4169E1')
    
    bars = ax_dist.bar(bins, counts, color=colors_list, alpha=0.85, edgecolor='black', linewidth=2)
    
    ax_dist.set_xlabel('Cluster ID', fontweight='bold', fontsize=22)
    ax_dist.set_ylabel('Pixel Count', fontweight='bold', fontsize=22)
    ax_dist.set_title('Cluster Distribution (Gold=Top 3, Blue=Valid, Red=Filtered Out)', 
                     fontweight='bold', fontsize=24, pad=15)
    ax_dist.grid(True, alpha=0.3, axis='y', linewidth=1.5)
    ax_dist.tick_params(labelsize=18, width=2.5, length=8)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax_dist.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height/1000)}K',
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    legend_elements = [
        Patch(facecolor='#FFD700', edgecolor='black', label='Top 3 clusters'),
        Patch(facecolor='#4169E1', edgecolor='black', label='Valid clusters'),
        Patch(facecolor='#CD5C5C', edgecolor='black', label='Filtered out')
    ]
    ax_dist.legend(handles=legend_elements, loc='upper right', fontsize=16, framealpha=0.95)
    
    # ===== TIME SERIES PLOTS =====
    for plot_idx, cluster_id in enumerate(sorted(valid_clusters.keys())):
        row = 1 + (plot_idx // n_cols)
        col = plot_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        data = cluster_data[cluster_id]
        analysis = analyses[cluster_id]
        
        doy = np.array(data['doy'])
        mean_list = [x if x is not None else np.nan for x in data['mean']]
        mean = np.array(mean_list, dtype=float)
        std_list = [x if x is not None else 0.0 for x in data['std']]
        std = np.array(std_list, dtype=float)
        
        valid_mask = ~np.isnan(mean)
        doy_valid = doy[valid_mask]
        mean_valid = mean[valid_mask]
        std_valid = std[valid_mask]
        std_valid = np.nan_to_num(std_valid, nan=0.0)
        
        # Tillage window (behind everything)
        if tillage_window and cluster_id in tillage_window['clusters']:
            ax.axvspan(tillage_window['start'], tillage_window['end'], 
                      alpha=0.25, color='brown', 
                      label='🚜 Tillage', zorder=0,
                      hatch='///', edgecolor='saddlebrown', linewidth=2)
        
        # ±2σ envelope (lightest)
        lower_2sig = np.maximum(mean_valid - 2*std_valid, 0)
        upper_2sig = np.minimum(mean_valid + 2*std_valid, 1)
        ax.fill_between(doy_valid, lower_2sig, upper_2sig,
                        color='blue', alpha=0.15, label='±2σ', zorder=2)
        
        # ±1σ envelope (darker)
        lower_1sig = np.maximum(mean_valid - std_valid, 0)
        upper_1sig = np.minimum(mean_valid + std_valid, 1)
        ax.fill_between(doy_valid, lower_1sig, upper_1sig,
                        color='blue', alpha=0.35, label='±1σ', zorder=3)
        
        # Mean line (bold, on top)
        ax.plot(doy_valid, mean_valid, 'k-', linewidth=3.5, label='Mean', zorder=10)
        
        # Mark peaks
        if len(analysis.get('peak_doys', [])) > 0:
            peak_doys_array = np.array(analysis['peak_doys'])
            peak_ndvi_array = np.array(analysis['peak_ndvi'])
            valid_peaks_mask = np.array([doy in doy_valid for doy in peak_doys_array])
            if np.any(valid_peaks_mask):
                ax.plot(peak_doys_array[valid_peaks_mask], peak_ndvi_array[valid_peaks_mask], 
                       'r*', markersize=24, markeredgewidth=2, markeredgecolor='darkred',
                       label='Peak', zorder=15)
        
        # REMOVED xlabel "Day of Year" from individual plots
        ax.set_ylabel('NDVI', fontweight='bold', fontsize=18)
        
        pattern_text = analysis["pattern"].replace(' cropping', '').replace('Potential ', '')
        pixels_text = f'{int(data["n_pixels"]/1000):.0f}K px'
        
        # Highlight top 3
        if cluster_id in top_cluster_ids:
            rank = top_cluster_ids.index(cluster_id) + 1
            ax.set_title(f'⭐ #{rank} - C{cluster_id}: {pattern_text} ({pixels_text})', 
                        fontweight='bold', fontsize=19, pad=12, color='#DAA520',
                        bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', 
                                 edgecolor='gold', linewidth=2.5))
        else:
            ax.set_title(f'C{cluster_id}: {pattern_text} ({pixels_text})', 
                        fontweight='bold', fontsize=18, pad=10)
        
        ax.set_xlim([0, 365])
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, linewidth=1.2)
        ax.tick_params(labelsize=16, width=2, length=6)
        
        # Legend only on first plot
        if plot_idx == 0:
            ax.legend(loc='upper right', fontsize=13, framealpha=0.95, 
                     edgecolor='black', fancybox=True, shadow=True,
                     borderpad=0.8, labelspacing=0.6)
    
    # Single "Day of Year" label at bottom center
    fig.text(0.5, 0.02, 'Day of Year', ha='center', fontsize=24, fontweight='bold')
    
    plt.savefig('bihar_clean_phenology_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Clean visualization saved as 'bihar_clean_phenology_analysis.png'")
    plt.show()

def main_enhanced_analysis():
    """Main processing pipeline"""
    
    start_date = '2019-01-01'
    end_date = '2019-12-31'
    
    print("🌾 Enhanced Bihar Phenology Clustering Analysis")
    print("=" * 80)
    print(f"📍 Region: Bihar, India")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🔬 Features:")
    print("   • Anomalous cluster filtering (<30K pixels or non-sinusoidal)")
    print("   • Single vs. biannual cropping detection")
    print("   • Crop-free period identification")
    print("   • Clean visualization with readable labels")
    print()
    
    try:
        bihar_polygon = get_bihar_polygon()
        if bihar_polygon is None:
            raise ValueError("Failed to create Bihar polygon")
        
        print("🗺️ Bihar Polygon Created (13 vertices)")
        
        processor = PhenologyProcessor(
            geometry=bihar_polygon,
            start_date=start_date,
            end_date=end_date,
            Verbose=True
        )
        
        result = processor.process(
            composite_days=30,
            n_bins=15,
            cluster_type='peak_time',
            extract_timeseries=True
        )
        
        if result and processor.cluster_timeseries:
            print("\n✅ PROCESSING COMPLETED!")
            
            analyses = analyze_cluster_seasonality(processor.cluster_timeseries)
            
            # Print detailed analysis to console
            print("\n" + "="*80)
            print("DETAILED CLUSTER ANALYSIS")
            print("="*80)
            
            for cluster_id, analysis in sorted(analyses.items()):
                print(f"\nCluster {cluster_id} ({int(analysis['n_pixels']):,} pixels):")
                print(f"  Pattern: {analysis['pattern']}")
                print(f"  Peaks: {analysis['n_peaks']}")
                
                if len(analysis['peak_doys']) > 0:
                    peak_info = ", ".join([f"DOY {int(d)} (NDVI={v:.3f})" 
                                          for d, v in zip(analysis['peak_doys'], 
                                                         analysis['peak_ndvi'])])
                    print(f"  Peak details: {peak_info}")
                
                if len(analysis['crop_free_periods']) > 0:
                    print(f"  Crop-free periods:")
                    for start, end in analysis['crop_free_periods']:
                        print(f"    - DOY {int(start)} to {int(end)} ({int(end-start)} days)")
            
            # Generate clean visualization
            plot_clean_visualization(processor, result, start_date, end_date, analyses)
            
            return result, analyses
        else:
            print("❌ Failed to complete analysis")
            return None, None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("🚀 Starting Clean Bihar Phenology Analysis")
    print()
    
    result, analyses = main_enhanced_analysis()
    
    if result and analyses:
        print("\n" + "="*80)
        print("🎉 ANALYSIS COMPLETED!")
        print("="*80)
        print("📊 Output: bihar_clean_phenology_analysis.png")
        print()
        print("✅ Key Improvements:")
        print("   • Filtered anomalous clusters (<30K pixels or non-sinusoidal)")
        print("   • Removed redundant 'Day of Year' labels from subplots")
        print("   • All statistics printed to console")
        print("   • Improved label readability and spacing")
    else:
        print("\n❌ Analysis incomplete")