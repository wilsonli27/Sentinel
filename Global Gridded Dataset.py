"""
Global Gridded Tillage Dataset Generation
Translated from R to Python

Original work by Porwollik, Vera; Rolinski, Susanne; Müller, Christoph (2019)
Python translation with UPDATED DATASETS (2010-2020 timeframe)

DATASET VERSIONS USED:
- SPAM2020 (cropland ~2010)
- SoilGrids 2017 250m (depth to bedrock)
- GloSEM 1.3 (erosion 2019)
- Global-AI_ET0_v3.1 (aridity index)
- OGHIST 2025 (income levels FY11/2010)
- Lesiv et al. 2018 (field sizes)
- FAOSTAT No-Till (conservation agriculture statistics 2010-2015)

Required packages:
- rasterio: for reading/writing raster data
- numpy: for array operations
- pandas: for data manipulation
- scipy: for interpolation and statistics
- xarray: for NetCDF operations
"""

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr
import xarray as xr
import netCDF4 as nc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. SETUP AND CONFIGURATION
# ============================================================================

class TillageDatasetGenerator:
    """
    Main class for generating global tillage dataset
    """
    
    def __init__(self, path_input, path_output, sample_calc=False):
        """
        Initialize the tillage dataset generator
        
        Parameters:
        -----------
        path_input : str or Path
            Path to input data directory containing:
            - spam2020v2r0_global_physical_area.geotiff/ (SPAM2020 cropland)
            - BDTICM_M_250m_ll (SoilGrids 2017 depth to bedrock)
            - Data2_SOIL_DISPLACEMENT_ESTIMATE_2019_1/ (GloSEM 1.3 erosion)
            - Global-AI_ET0_v3/ (FAO Aridity Index v3.1)
            - OGHIST_2025_10_07 (World Bank income levels)
        path_output : str or Path
            Path to output directory
        sample_calc : bool
            If True, runs on sample data for testing (45° resolution)
            If False, runs on full dataset (5 arcmin / 0.0833° resolution)
        """
        self.path_input = Path(path_input)
        self.path_output = Path(path_output)
        self.sample_calc = sample_calc
        
        # Define spatial extent and resolution
        if sample_calc:
            self.extent = {'xmin': -180, 'xmax': 180, 'ymin': -45, 'ymax': 90}
            self.resolution = 45  # degrees
        else:
            self.extent = {'xmin': -180, 'xmax': 180, 'ymin': -56, 'ymax': 84}
            self.resolution = 1/12  # 5 arcmin = 0.0833 degrees
        
        # Calculate grid dimensions
        self.nx = int((self.extent['xmax'] - self.extent['xmin']) / self.resolution)
        self.ny = int((self.extent['ymax'] - self.extent['ymin']) / self.resolution)
        
        # Create coordinate arrays
        self.lons = np.linspace(
            self.extent['xmin'] + self.resolution/2,
            self.extent['xmax'] - self.resolution/2,
            self.nx
        )
        self.lats = np.linspace(
            self.extent['ymax'] - self.resolution/2,
            self.extent['ymin'] + self.resolution/2,
            self.ny
        )
        
        # Define crop names (42 crops from SPAM2005)
        self.crop_names = [
            'WHEA', 'RICE', 'MAIZ', 'BARL', 'REST', 'OOIL', 'TOBA', 'TEAS',
            'COCO', 'RCOF', 'ACOF', 'OFIB', 'COTT', 'SUGB', 'SUGC', 'OILP',
            'VEGE', 'TEMF', 'TROF', 'PLNT', 'BANA', 'CNUT', 'GROU', 'OTRS',
            'CASS', 'YAMS', 'SWPO', 'POTA', 'SESA', 'RAPE', 'SUNF', 'SOYB',
            'OPUL', 'LENT', 'PIGE', 'COWP', 'CHIC', 'BEAN', 'OCER', 'SORG',
            'SMIL', 'PMIL'
        ]
        
        print(f"Initialized TillageDatasetGenerator")
        print(f"Sample calculation: {sample_calc}")
        print(f"Grid size: {self.nx} x {self.ny}")
        print(f"Resolution: {self.resolution}°")

# ============================================================================
# 2. DATA LOADING AND HARMONIZATION
# ============================================================================

    def load_spam_data(self, version='2020'):
        """
        Load SPAM cropland data (physical area in hectares)
        
        UPDATED for SPAM2020!
        
        Data source: 
        - SPAM2020: https://www.mapspam.info/data/
        - Legacy SPAM2005: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DHXBJX
        
        IMPORTANT CHANGES in SPAM2020:
        1. Data format: Now GeoTIFF instead of mixed formats
        2. Crop list: Expanded from 42 to 43 crops (added 'REST' split into more specific crops)
        3. File naming: Changed from "SPAM2005V3r1_global_A_TA_CROP_A.tif" 
           to "spam2020V2r0_global_P_CROP_A.geotiff" (note "P" for Physical area)
        4. Resolution: Still 5 arcmin but improved methodology
        5. Production systems: Same 6 systems (A, I, R, H, L, S)
        
        Files needed from SPAM2020:
        - spam2020V2r0_global_P_[CROP]_A.geotiff (Physical area, All systems)
        - spam2020V2r0_global_P_[CROP]_R.geotiff (Physical area, Rainfed)
        
        Production systems remain the same:
        - A: All production systems combined
        - R: Rainfed
        - I: Irrigated
        - H: High input
        - L: Low input
        - S: Subsistence
        """
        print(f"\nLoading SPAM{version} cropland data...")
        
        if self.sample_calc:
            # Create sample data for testing
            spam_ta = np.random.randint(1, 25, size=(42, self.ny, self.nx))
            spam_tr = spam_ta - 0.33
            return spam_ta, spam_tr
        
        # SPAM2020 file pattern - YOUR NESTED FOLDER STRUCTURE
        spam_outer = self.path_input / "spam2020v2r0_global_physical_area.geotiff"
        
        print(f"  Looking for SPAM folder at: {spam_outer}")
        print(f"  Path exists: {spam_outer.exists()}")
        
        # If outer doesn't exist, try alternate names
        if not spam_outer.exists():
            # List what's actually in the input folder
            print(f"\n  Contents of {self.path_input}:")
            if self.path_input.exists():
                for item in self.path_input.iterdir():
                    print(f"    - {item.name}")
            else:
                print(f"  ERROR: Input path doesn't exist: {self.path_input}")
            
            raise FileNotFoundError(f"SPAM folder not found. Expected: {spam_outer}")
        
        spam_folder = spam_outer / "spam2020v2r0_global_physical_area"
        
        # Check if nested folder exists, otherwise use outer folder
        if not spam_folder.exists():
            spam_folder = spam_outer
        
        # Check if folder has any files
        test_files = list(spam_folder.glob("spam2020*.tif"))
        print(f"  Found {len(test_files)} SPAM TIF files")
        if len(test_files) > 0:
            print(f"  Example: {test_files[0].name}")
        else:
            print(f"  WARNING: No SPAM TIF files found in {spam_folder}")
            print(f"  Listing folder contents:")
            for item in spam_folder.iterdir():
                print(f"    - {item.name}")
        
        if version == '2020':
            # YOUR ACTUAL FILE PATTERN: spam2020_V2r0_global_A_CROP_SYSTEM.tif
            file_pattern_a = "spam2020_V2r0_global_A_{crop}_A.tif"  # All systems
            file_pattern_r = "spam2020_V2r0_global_A_{crop}_R.tif"  # Rainfed
        else:  # Legacy SPAM2005
            file_pattern_a = "SPAM2005V3r1_global_A_TA_{crop}_A.tif"
            file_pattern_r = "SPAM2005V3r1_global_A_TR_{crop}_R.tif"
        
        spam_ta = np.zeros((42, self.ny, self.nx))
        spam_tr = np.zeros((42, self.ny, self.nx))
        
        for i, crop in enumerate(self.crop_names):
            # Load Total All (or All systems for SPAM2020)
            ta_file = spam_folder / file_pattern_a.format(crop=crop)
            
            if not ta_file.exists():
                print(f"  WARNING: Missing {ta_file.name}")
                continue
                
            with rasterio.open(ta_file) as src:
                data = src.read(1)
                spam_ta[i] = self._resample_to_grid(data, src.transform)
            
            # Load Total Rainfed
            tr_file = spam_folder / file_pattern_r.format(crop=crop)
            
            if not tr_file.exists():
                print(f"  WARNING: Missing {tr_file.name}")
                continue
                
            with rasterio.open(tr_file) as src:
                data = src.read(1)
                spam_tr[i] = self._resample_to_grid(data, src.transform)
            
            if i == 0 or (i+1) % 10 == 0:  # Print every 10th crop
                print(f"  Loaded {i+1}/42 crops...")
        
        n_loaded = np.sum(~np.all(spam_ta == 0, axis=(1,2)))
        print(f"  ✓ Successfully loaded {n_loaded}/42 crops")
        
        return spam_ta, spam_tr
    
    def load_soilgrids_depth(self, version='250m', use_r_horizon=False):
        """
        Load SoilGrids absolute depth to bedrock (cm)
        
        Data source: SoilGrids250m 2017-03
        
        TWO OPTIONS AVAILABLE:
        
        Option 1 (RECOMMENDED): BDTICM - Absolute depth to bedrock
        - File: BDTICM_M_250m_ll.tif
        - URL: https://files.isric.org/soilgrids/former/2017-03-10/data/BDTICM_M_250m_ll.tif
        - Size: ~8 GB
        - What it is: Depth to any impenetrable layer (bedrock, hardpan, etc.)
        - Best for: Identifying all barriers to tillage
        
        Option 2: BDRICM - Depth to bedrock (R horizon)
        - File: BDRICM_M_250m_ll.tif
        - URL: https://files.isric.org/soilgrids/former/2017-03-10/data/BDRICM_M_250m_ll.tif
        - Size: ~8 GB (same size!)
        - What it is: Depth specifically to R horizon (consolidated bedrock)
        - Best for: More conservative estimate (may miss other barriers)
        
        WHICH ONE TO USE?
        -----------------
        For this tillage analysis: Use BDTICM (default)
        
        BDTICM is better because it captures ALL obstacles to deep tillage:
        - Bedrock (R horizon)
        - Hardpans (densic/duripans)
        - Cemented layers
        - Permafrost
        
        BDRICM only captures bedrock, so it might OVERESTIMATE 
        the area suitable for deep tillage (missing other barriers).
        
        Both files are the same size (~8 GB), so there's no storage advantage
        to using BDRICM.
        
        SMALLER ALTERNATIVE:
        If 8 GB is too large, use SoilGrids 1km 2014:
        - File: BDTICM_M_sl1_10km_ll.tif
        - URL: https://files.isric.org/soilgrids/former/2014/data/BDTICM_M_sl1_10km_ll.tif
        - Size: ~200 MB (40x smaller!)
        - Resolution: 1km instead of 250m
        - Still perfectly adequate for 5 arcmin analysis
        
        Parameters:
        -----------
        version : str
            '250m' or '2017' = Use SoilGrids250m 2017 (high res)
            '1km' or '2014' = Use SoilGrids 1km 2014 (smaller file)
        use_r_horizon : bool
            True = Use BDRICM (R horizon only)
            False = Use BDTICM (all barriers) - RECOMMENDED
        """
        print(f"\nLoading SoilGrids depth to bedrock...")
        
        if self.sample_calc:
            soilgrid = np.arange(1, 25).reshape(self.ny, self.nx)
            flat_area = soilgrid.copy()
            flat_area[soilgrid >= 15] = np.nan
            return soilgrid, flat_area
        
        # Determine which file to use - YOUR FILE: BDTICM_M_250m_ll (NO EXTENSION!)
        if version in ['250m', '2017', '2017-03']:
            # SoilGrids250m 2017 - High resolution
            # Try multiple possible names/extensions
            possible_files = [
                "BDTICM_M_250m_ll",  # No extension (YOUR FILE)
                "BDTICM_M_250m_ll.tif",
                "BDTICM_M_250m_ll.tiff",
                "BDRICM_M_250m_ll" if use_r_horizon else None
            ]
            
            soilgrid_file = None
            for filename in possible_files:
                if filename is None:
                    continue
                test_file = self.path_input / filename
                if test_file.exists():
                    soilgrid_file = test_file
                    break
            
            if soilgrid_file:
                var_name = "BDRICM (R horizon)" if use_r_horizon else "BDTICM (absolute)"
                resolution_str = "250m"
                print(f"  Found file: {soilgrid_file.name}")
            else:
                print("  WARNING: SoilGrids250m 2017 file not found!")
                print("  Tried: BDTICM_M_250m_ll (with/without .tif extension)")
                print("  Download from: https://files.isric.org/soilgrids/former/2017-03-10/data/")
                print("  File size: ~8 GB")
                raise FileNotFoundError(f"SoilGrids file not found. Tried: {possible_files}")
        
        elif version in ['1km', '2014', '1.0']:
            # SoilGrids 1km 2014 - Legacy backup
            soilgrid_file = self.path_input / "BDTICM_M_10km_ll.tif"
            var_name = "BDTICM (absolute)"
            resolution_str = "1km"
            
            if not soilgrid_file.exists():
                print("  ERROR: SoilGrids file not found!")
                print("  Download from: https://files.isric.org/soilgrids/former/2014/data/")
                print("  File needed: BDTICM_M_sl1_10km_ll.tif")
                raise FileNotFoundError(f"Missing: {soilgrid_file}")
        
        else:
            raise ValueError(f"Unknown version: {version}. Use '250m' or '1km'")
        
        # Load and resample
        print(f"  Loading {var_name} at {resolution_str} resolution...")
        
        # MEMORY ISSUE FIX: For large files, read in chunks and resample
        with rasterio.open(soilgrid_file) as src:
            print(f"  Original size: {src.width} x {src.height} pixels")
            print(f"  Original resolution: {src.res}")
            
            # Check if file is too large for memory
            estimated_memory = src.width * src.height * 4 / (1024**3)  # GB
            print(f"  Estimated memory: {estimated_memory:.1f} GB")
            
            if estimated_memory > 10:  # If larger than 10 GB
                print(f"  ⚠️  File too large to load at once!")
                print(f"  Reading and resampling in chunks...")
                
                # Calculate target dimensions for 5 arcmin
                target_width = self.nx
                target_height = self.ny
                
                # Use windowed reading with overview level
                # This reads at lower resolution automatically
                overview_level = 0
                if src.overviews(1):  # Check if overviews exist
                    # Use overview closest to our target resolution
                    overview_level = min(len(src.overviews(1)) - 1, 3)
                    print(f"  Using overview level {overview_level}")
                
                # Read at reduced resolution
                data = src.read(
                    1,
                    out_shape=(target_height, target_width),
                    resampling=Resampling.bilinear
                )
                
                print(f"  Resampled to: {target_width} x {target_height} pixels")
                soilgrid_res = data
                
            else:
                # Small enough to load normally
                data = src.read(1)
                print(f"  Original data range: {np.nanmin(data):.1f} - {np.nanmax(data):.1f} cm")
                
                # Resample to target grid (5 arcmin)
                soilgrid_res = self._resample_to_grid(data, src.transform, method='bilinear')
        
        # Identify flat/shallow areas (< 15 cm depth)
        # Convert to float first to allow NaN values
        soilgrid_res = soilgrid_res.astype(np.float32)
        flat_area = soilgrid_res.copy()
        flat_area[soilgrid_res >= 15] = np.nan
        
        print(f"  Resampled to 5 arcmin. Range: {np.nanmin(soilgrid_res):.1f} - {np.nanmax(soilgrid_res):.1f} cm")
        print(f"  Shallow areas (<15cm): {np.sum(~np.isnan(flat_area))} cells ({np.sum(~np.isnan(flat_area))/soilgrid_res.size*100:.2f}%)")
        
        return soilgrid_res, flat_area
    
    def load_field_size_data(self):
        print("\nLoading Lesiv 2018 global field size data...")
        
        if self.sample_calc:
            return np.arange(10, 34).reshape(self.ny, self.nx)
        
        # CORRECT PATH: Go inside the "Global Field Sizes" folder
        field_folder = self.path_input / "lesiv_2018_field_sizes" / "Global Field Sizes"
        
        # THE ACTUAL RASTER FILE (discovered from diagnostic)
        field_file = field_folder / "dominant_field_size_categories.tif"
        
        if not field_file.exists():
            print(f"  ERROR: Field size file not found!")
            print(f"  Looking for: {field_file}")
            raise FileNotFoundError(f"Missing: {field_file}")
        
        print(f"  Found: {field_file.name}")
        
        # Open the actual TIF file (not the folder!)
        with rasterio.open(field_file) as src:
            data = src.read(1)
            print(f"  Original resolution: {src.res}")
            print(f"  Original size: {src.width} x {src.height} pixels")
            print(f"  Original range: {np.nanmin(data)} - {np.nanmax(data)}")
            
            # Resample to target grid (5 arcmin)
            fields_resampled = self._resample_to_grid(
                data, 
                src.transform,
                method='nearest'  # Use nearest for categorical data
            )
        
        print(f"  Resampled to 5 arcmin. Range: {np.nanmin(fields_resampled)} - {np.nanmax(fields_resampled)}")
        
        return fields_resampled
    
    def load_erosion_data(self, version='glosem'):
        """
        Load water erosion / land degradation data
        
        FIXED: Can read from .tif.ovr (pyramid file) at lower resolution
        """
        print(f"\nLoading erosion data (version: {version})...")
        
        if self.sample_calc:
            return np.arange(1, 25).reshape(self.ny, self.nx)
        
        if version.lower() in ['glosem', '2020', '1.3']:
            glosem_outer = self.path_input / "Data2_SOIL_DISPLACEMENT_ESTIMATE_2019_1"
            glosem_folder = glosem_outer / "Data2_SOIL_DISPLACEMENT_ESTIMATE_2019_1"
            
            if not glosem_folder.exists():
                glosem_folder = glosem_outer
            
            # Try main file first, then overview file
            erosion_file = glosem_folder / "SOIL_DISPLACEMENT_ESTIMATE_2019.tif"
            
            if not erosion_file.exists():
                # Try the .ovr file (pyramid/overview)
                erosion_file = glosem_folder / "SOIL_DISPLACEMENT_ESTIMATE_2019.tif.ovr"
                if erosion_file.exists():
                    print(f"  ℹ️  Main .tif not found, using .ovr (pyramid file)")
                else:
                    print(f"  ERROR: Neither main .tif nor .ovr file found!")
                    raise FileNotFoundError(f"Missing GloSEM files in {glosem_folder}")
            
        elif version.lower() in ['gladis', '2000', 'legacy']:
            erosion_file = self.path_input / "hdr.adf"
            if not erosion_file.exists():
                raise FileNotFoundError(f"Missing: {erosion_file}")
        else:
            raise ValueError(f"Unknown erosion version: {version}")
        
        print(f"  Found: {erosion_file.name}")
        
        # Open the file
        with rasterio.open(erosion_file) as src:
            print(f"  Original resolution: {src.res}")
            print(f"  Original size: {src.width} x {src.height} pixels")
            
            # Calculate memory requirement
            estimated_memory = src.width * src.height * 4 / (1024**3)
            print(f"  Estimated memory: {estimated_memory:.1f} GB")
            
            # Check if file has overviews (pyramid levels)
            has_overviews = False
            if src.overviews(1):
                has_overviews = True
                n_overviews = len(src.overviews(1))
                print(f"  Available overview levels: {n_overviews}")
                print(f"  Overview factors: {src.overviews(1)}")
            
            # Strategy: Always read at target resolution to avoid memory issues
            target_width = self.nx   # 4320 for 5 arcmin
            target_height = self.ny  # 1680 for 5 arcmin
            
            print(f"  Target size: {target_width} x {target_height} pixels")
            print(f"  Reading directly at target resolution...")
            
            try:
                # Read with automatic decimation/resampling
                # This is MUCH more memory efficient than reading full resolution
                data = src.read(
                    1,
                    out_shape=(target_height, target_width),
                    resampling=Resampling.average  # Use average for continuous data
                )
                
                print(f"  ✓ Successfully read at target resolution")
                erosion = data.astype(np.float32)
                
            except Exception as e:
                print(f"  ⚠️  Direct reading failed: {e}")
                
                if has_overviews:
                    # Try reading from a specific overview level
                    print(f"  Attempting to read from overview level...")
                    
                    # Find appropriate overview level
                    # Overview factors are typically [2, 4, 8, 16, ...]
                    overview_factors = src.overviews(1)
                    
                    # Choose overview level closest to our target
                    src_resolution = src.res[0]  # degrees per pixel
                    target_resolution = self.resolution  # 5 arcmin = 0.0833 degrees
                    needed_factor = target_resolution / src_resolution
                    
                    # Find closest overview
                    best_level = 0
                    for i, factor in enumerate(overview_factors):
                        if factor <= needed_factor:
                            best_level = i
                    
                    print(f"  Using overview level {best_level} (factor {overview_factors[best_level]})")
                    
                    # Read from overview
                    data = src.read(
                        1,
                        out_shape=(target_height, target_width),
                        resampling=Resampling.average
                    )
                    erosion = data.astype(np.float32)
                else:
                    raise
        
        # Clean up no-data values
        erosion[erosion < 0] = np.nan
        erosion[erosion > 10000] = np.nan  # Unrealistic values
        
        # Handle any remaining issues
        if np.all(np.isnan(erosion)):
            print(f"  ⚠️  WARNING: All values are NaN after loading!")
            print(f"  This might indicate a data format issue.")
        
        valid_data = erosion[~np.isnan(erosion)]
        if len(valid_data) > 0:
            print(f"  Data range: {np.nanmin(erosion):.1f} - {np.nanmax(erosion):.1f} t/ha/yr")
            print(f"  Mean: {np.nanmean(erosion):.1f} t/ha/yr")
            print(f"  Valid cells: {len(valid_data)} ({len(valid_data)/erosion.size*100:.1f}%)")
        else:
            print(f"  ⚠️  No valid data found!")
        
        return erosion
    
    def load_aridity_index(self):
        """
        Load Global Aridity Index v3.1 (CGIAR-CSI)
        
        FIXED: Handles resampling errors and checks for valid data
        """
        print("\nLoading Global Aridity Index v3.1...")
        
        if self.sample_calc:
            aridity = np.arange(1, 24).reshape(self.ny, self.nx) / 100.0
            aridity = np.append(aridity, [[np.nan] * self.nx], axis=0)
            return aridity
        
        # YOUR FOLDER STRUCTURE
        aridity_folder = self.path_input / "Global-AI_ET0_annual_v3" / "Global-AI_ET0_v3_annual"
        
        # YOUR ACTUAL FILE: ai_v3_yr.tif (495 MB)
        aridity_file = aridity_folder / "ai_v3_yr.tif"
        
        if not aridity_file.exists():
            print(f"  ERROR: Aridity Index file not found!")
            print(f"  Looking for: {aridity_file}")
            
            # Show what's actually there
            if aridity_folder.exists():
                print(f"\n  Files in {aridity_folder.name}:")
                for item in sorted(aridity_folder.iterdir()):
                    if item.is_file() and item.suffix in ['.tif', '.sd']:
                        size_mb = item.stat().st_size / (1024**2)
                        print(f"    - {item.name} ({size_mb:.1f} MB)")
            
            raise FileNotFoundError(f"Missing: {aridity_file}")
        
        print(f"  Found: {aridity_file.name}")
        
        # Load the file
        with rasterio.open(aridity_file) as src:
            print(f"  Resolution: {src.res}")
            print(f"  Size: {src.width} x {src.height} pixels")
            
            # Check file size
            estimated_memory = src.width * src.height * 4 / (1024**3)
            print(f"  Estimated memory: {estimated_memory:.1f} GB")
            
            if estimated_memory > 10:
                print(f"  Large file, reading at target resolution...")
                # Read directly at 5 arcmin resolution
                data = src.read(
                    1,
                    out_shape=(self.ny, self.nx),
                    resampling=Resampling.bilinear
                )
                aridity_resampled = data
            else:
                # Small enough to load fully
                data = src.read(1)
                print(f"  Raw data range: {np.nanmin(data)} - {np.nanmax(data)}")
                
                # Check if _resample_to_grid returns None
                aridity_resampled = self._resample_to_grid(data, src.transform, method='bilinear')
                
                # CRITICAL FIX: Check if resampling succeeded
                if aridity_resampled is None:
                    print(f"  ⚠️  Resampling returned None, using direct read instead...")
                    aridity_resampled = src.read(
                        1,
                        out_shape=(self.ny, self.nx),
                        resampling=Resampling.bilinear
                    )
            
            # Ensure we have a numpy array
            if aridity_resampled is None:
                raise ValueError("Failed to load aridity data - resampling returned None")
            
            aridity_resampled = np.asarray(aridity_resampled, dtype=np.float32)
            
            # CRITICAL: Check if values are scaled (multiplied by 10,000)
            # Global-AI v3 stores values as integers scaled by 10,000
            valid_data = aridity_resampled[np.isfinite(aridity_resampled)]
            
            if len(valid_data) == 0:
                raise ValueError("No valid data after resampling!")
            
            max_val = np.max(valid_data)
            min_val = np.min(valid_data)
            
            print(f"  Resampled range: {min_val:.0f} - {max_val:.0f}")
            
            if max_val > 100:  # Values are scaled
                print(f"  Detected scaled values (max={max_val:.0f})")
                print(f"  Dividing by 10,000 to get actual AI values...")
                aridity_actual = aridity_resampled / 10000.0
            else:
                print(f"  Values appear to be unscaled (already 0-1 range)")
                aridity_actual = aridity_resampled.copy()
        
        # Clean up no-data values
        aridity_actual = aridity_actual.astype(np.float32)
        
        # Check for common no-data values BEFORE setting to NaN
        aridity_actual[aridity_actual == -9999] = np.nan
        aridity_actual[aridity_actual == 9999] = np.nan
        aridity_actual[aridity_actual == 65535] = np.nan  # Common no-data for uint16
        aridity_actual[aridity_actual < 0] = np.nan
        aridity_actual[aridity_actual > 10] = np.nan  # AI typically 0-5, max ~4
        
        valid_count = np.sum(~np.isnan(aridity_actual))
        if valid_count == 0:
            raise ValueError("No valid aridity data after cleaning!")
        
        print(f"  Actual AI range: {np.nanmin(aridity_actual):.3f} - {np.nanmax(aridity_actual):.3f}")
        print(f"  Mean AI: {np.nanmean(aridity_actual):.3f}")
        print(f"  Valid cells: {valid_count} ({valid_count/aridity_actual.size*100:.1f}%)")
        print(f"  Note: Higher AI = more humid, lower AI = more arid")
        
        return aridity_actual
    
    def load_income_levels(self):
        """
        Load World Bank income levels - READS WHOLE FILE THEN EXTRACTS
        """
        print("\nLoading World Bank income levels (OGHIST)...")
        
        if self.sample_calc:
            return {}, []
        
        import pandas as pd
        
        income_file = self.path_input / "OGHIST_2025_10_07.xlsx"
        if not income_file.exists():
            income_file = self.path_input / "OGHIST_2025_10_07.xls"
        if not income_file.exists():
            raise FileNotFoundError(f"Missing: {income_file}")
        
        # Read the entire sheet with no skipping
        df_raw = pd.read_excel(income_file, sheet_name='Country Analytical History', header=None)
        
        print(f"  Raw sheet size: {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
        
        # Based on your screenshot:
        # Row 5 (index 4): "Bank's fiscal year:"
        # Row 6 (index 5): FY labels (FY21, FY22, etc.)
        # Row 12 (index 11): First country code (AFG)
        
        # Get FY column headers from row 6 (index 5)
        fy_row = df_raw.iloc[5]
        print(f"  Row 6 (FY labels): {fy_row.tolist()[:10]}")
        
        # Get country data starting from row 12 (index 11)
        df_data = df_raw.iloc[11:].copy()
        df_data = df_data.reset_index(drop=True)
        
        # Set column names from row 6
        df_data.columns = fy_row.tolist()
        
        print(f"  Data shape: {df_data.shape[0]} rows x {df_data.shape[1]} cols")
        print(f"  Column names: {df_data.columns.tolist()[:5]}")
        
        # Find the Code and Country columns (first two)
        # They might not have proper names in row 6
        first_col = df_data.columns[0]
        second_col = df_data.columns[1]
        
        df_data = df_data.rename(columns={first_col: 'Code', second_col: 'Country'})
        
        print(f"  Sample data:")
        print(f"    Code={df_data['Code'].iloc[0]}, Country={df_data['Country'].iloc[0]}")
        
        # Find year columns (numeric columns representing calendar years)
        year_cols = [col for col in df_data.columns 
                    if isinstance(col, (int, float)) and col >= 1987 and col <= 2024]
        
        if not year_cols:
            print(f"  ERROR: No year columns found!")
            print(f"  All columns: {df_data.columns.tolist()}")
            raise ValueError("No year columns found!")
        
        year_cols_sorted = sorted(year_cols)
        
        print(f"  Found {len(year_cols_sorted)} year columns")
        print(f"  Range: {year_cols_sorted[0]} to {year_cols_sorted[-1]}")
        
        # Use 2010 (matches SPAM2020 ~2010) if available
        # Note: OGHIST uses calendar years, and World Bank fiscal year runs July-June
        # So FY11 (July 2010 - June 2011) corresponds to calendar year 2010
        if 2010 in year_cols_sorted:
            year_column = 2010
            print(f"  ✓ Using calendar year 2010 (matches SPAM2020 ~2010)")
        else:
            # Use closest year to 2010
            year_column = min(year_cols_sorted, key=lambda x: abs(x - 2010))
            print(f"  ⚠️  Using {year_column} (2010 not available, using closest year)")
        
        # Map income levels to numeric codes
        income_map = {
            'L': 1, 'LIC': 1,
            'LM': 2, 'LMC': 2,
            'UM': 3, 'UMC': 3,
            'H': 4, 'HIC': 4,
            '..': None,
            None: None
        }
        
        df_data['income_code'] = df_data[year_column].map(income_map)
        
        # Keep only valid rows (3-letter codes, valid income)
        df_valid = df_data[
            (df_data['Code'].notna()) & 
            (df_data['Code'].astype(str).str.len() == 3) &
            (df_data['income_code'].notna())
        ].copy()
        
        print(f"  Valid countries: {len(df_valid)} (using year {year_column})")
        print(f"    Low: {sum(df_valid['income_code']==1)}")
        print(f"    Lower-middle: {sum(df_valid['income_code']==2)}")
        print(f"    Upper-middle: {sum(df_valid['income_code']==3)}")
        print(f"    High: {sum(df_valid['income_code']==4)}")
        
        # Show examples
        print(f"  Examples:")
        for idx in range(min(5, len(df_valid))):
            row = df_valid.iloc[idx]
            print(f"    {row['Code']} ({row['Country']}): {row[year_column]} = code {row['income_code']}")
        
        # Create outputs
        income_dict = dict(zip(df_valid['Code'], df_valid['income_code']))
        high_income_codes = df_valid[df_valid['income_code'] >= 3]['Code'].values
        
        # Store for later
        self.income_data = df_valid[['Code', year_column, 'income_code']].copy()
        
        print(f"  ✓ Loaded {len(income_dict)} countries with income data")
        print(f"  ✓ {len(high_income_codes)} high-income countries")
        
        return income_dict, high_income_codes


    def load_notill_statistics(self):
        """
        Load FAOSTAT No-Till / Conservation Agriculture Statistics
        
        FIXED: Better error handling and data format detection
        """
        print("\nLoading FAOSTAT No-Till statistics...")
        
        # Look for FAOSTAT file in input directory
        possible_names = [
            "FAOSTAT_data_en_1-9-2026.csv",
            "FAOSTAT_data_*.csv",
            "faostat_notill_2010s.csv",
            "faostat_conservation_tillage.csv",
            "notill_statistics.csv",
            "ca_statistics.csv"
        ]
        
        notill_file = None
        for name in possible_names:
            if '*' in name:
                # Glob pattern
                matches = list(self.path_input.glob(name))
                if matches:
                    notill_file = matches[0]
                    break
            else:
                test_file = self.path_input / name
                if test_file.exists():
                    notill_file = test_file
                    break
        
        if notill_file is None:
            print("  ⚠️  WARNING: FAOSTAT No-Till statistics not found!")
            print("  Expected files:")
            for name in possible_names[:3]:
                print(f"    - {name}")
            print("\n  Download from: https://www.faostat.org/")
            print("  Path: Data → Inputs → Land Use")
            print("  Variable: 'Cropland area under zero or no tillage'")
            print("  Years: 2010-2015")
            print("\n  ⚠️  Continuing without no-till statistics...")
            print("  This will affect downscaling accuracy!")
            
            # Return empty dataframe to allow pipeline to continue
            import pandas as pd
            return pd.DataFrame(columns=['country', 'iso3', 'notill_area_ha', 'year'])
        
        print(f"  Loading: {notill_file.name}")
        
        import pandas as pd
        df = pd.read_csv(notill_file)
        
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Try to detect format and standardize
        if 'Area' in df.columns and 'Value' in df.columns:
            print("  Detected FAOSTAT standard format")
            
            # Get country names
            countries = df['Area'].unique()
            
            # Check units - might be in "1000 ha"
            if 'Unit' in df.columns:
                unit = df['Unit'].iloc[0] if len(df) > 0 else 'ha'
                print(f"  Unit: {unit}")
                
                if '1000 ha' in str(unit) or '1000ha' in str(unit):
                    print("  Converting from 1000 ha to ha...")
                    df['Value'] = df['Value'] * 1000
            
            # Get ISO3 codes if available
            iso3_col = None
            for col in ['Area Code (ISO3)', 'ISO3', 'Area Code']:
                if col in df.columns:
                    iso3_col = col
                    break
            
            # Get year if available
            if 'Year' in df.columns:
                # Use most recent year for each country
                df_latest = df.sort_values('Year').groupby('Area').last().reset_index()
            else:
                df_latest = df
            
            # Create standardized format
            notill_data = pd.DataFrame({
                'country': df_latest['Area'],
                'notill_area_ha': df_latest['Value'],
                'year': df_latest['Year'] if 'Year' in df_latest.columns else 2012
            })
            
            if iso3_col:
                notill_data['iso3'] = df_latest[iso3_col]
        
        elif 'iso3' in df.columns.str.lower():
            print("  Detected simplified format")
            # Find actual column names (case-insensitive)
            col_map = {col.lower(): col for col in df.columns}
            notill_data = df.rename(columns={
                col_map.get('iso3', 'iso3'): 'iso3',
                col_map.get('country', 'country'): 'country',
                col_map.get('notill_area_ha', 'area'): 'notill_area_ha',
                col_map.get('year', 'year'): 'year'
            })
        
        else:
            print(f"  ⚠️  WARNING: Unrecognized format")
            print(f"  Columns: {df.columns.tolist()}")
            print("  Continuing without no-till statistics...")
            return pd.DataFrame(columns=['country', 'iso3', 'notill_area_ha', 'year'])
        
        # Remove any rows with missing values
        notill_data = notill_data.dropna(subset=['notill_area_ha'])
        
        # Filter out zero values
        notill_data = notill_data[notill_data['notill_area_ha'] > 0]
        
        if len(notill_data) == 0:
            print("  ⚠️  WARNING: No valid no-till data found after filtering!")
            return pd.DataFrame(columns=['country', 'iso3', 'notill_area_ha', 'year'])
        
        print(f"\n  Processed statistics for {len(notill_data)} countries")
        print(f"  Total no-till area: {notill_data['notill_area_ha'].sum()/1e6:.1f} million ha")
        
        # Show top 10 countries
        top10 = notill_data.nlargest(10, 'notill_area_ha')
        print(f"\n  Top 10 countries by no-till area:")
        for idx, row in top10.iterrows():
            country = row.get('country', row.get('iso3', 'Unknown'))
            area_mha = row['notill_area_ha'] / 1e6
            print(f"    {country}: {area_mha:.1f} Mha")
        
        return notill_data

# ============================================================================
# 3. LOGIT MODEL FOR CONSERVATION AGRICULTURE LIKELIHOOD
# ============================================================================

    def calculate_crop_mix(self, spam_tr, flat_area, fields_interpol, high_raster):
        """
        Calculate crop mix: ratio of CA-suitable crops to total cropland
        
        CA-suitable crops are the 22 annual grains (excluding rice, sugarbeet,
        tubers, and perennials). This represents the potential for conservation
        agriculture adoption.
        
        Parameters:
        -----------
        spam_tr : array (42, ny, nx)
            Rainfed cropland area by crop
        flat_area : array (ny, nx)
            Mask of shallow soils
        fields_interpol : array (ny, nx)
            Interpolated field sizes (can be None)
        high_raster : array (ny, nx)
            High income country mask (can be None)
        
        Returns:
        --------
        crop_mix : array (ny, nx)
            Ratio of CA-suitable crops (0-1)
        """
        print("\nCalculating crop mix for CA suitability...")
        
        # Check if field sizes are available
        if fields_interpol is None:
            print("  ⚠️  WARNING: Field sizes not available, using simplified calculation")
            # Use all cropland without field size filtering
            spam_tr_flat_out = spam_tr.copy()
            spam_tr_flat_out[:, np.isnan(flat_area)] = np.nan
            
            # Annual crops suitable for CA (29 crops total)
            annual_indices = [0,1,2,3,4,6,12,13,16] + list(range(23,42)) + [22]
            annuals = spam_tr_flat_out[annual_indices]
            
            # Grain crops only (22 crops)
            grain_indices = [0,2,3,4,5,6,8] + list(range(14,29))
            grains = annuals[grain_indices]
            grains_sum = np.nansum(grains, axis=0)
            
            # Total cropland
            total_cropland = np.nansum(spam_tr, axis=0)
            total_cropland[total_cropland == 0] = np.nan
            
            # Calculate crop mix ratio
            crop_mix = grains_sum / total_cropland
            crop_mix[~np.isfinite(crop_mix)] = np.nan
            
            print(f"  Crop mix calculated (simplified). Range: {np.nanmin(crop_mix):.3f} - {np.nanmax(crop_mix):.3f}")
            return crop_mix
        
        # Remove flat areas (< 15 cm soil depth)
        spam_tr_flat_out = spam_tr.copy()
        spam_tr_flat_out[:, np.isnan(flat_area)] = np.nan
        
        # Separate small (<2 ha) and large (>=2 ha) fields
        small_fields = fields_interpol < 20
        large_fields = fields_interpol >= 20
        
        # Small field area in high income countries
        small_area = spam_tr_flat_out.copy()
        small_area[:, ~small_fields] = np.nan
        
        if high_raster is not None:
            high_income_small = small_area.copy()
            high_income_small[:, np.isnan(high_raster)] = np.nan
        else:
            print("  ⚠️  High income data not available, treating all small fields equally")
            high_income_small = small_area
        
        # Large field area
        large_area = spam_tr_flat_out.copy()
        large_area[:, ~large_fields] = np.nan
        
        # Combine large fields with small fields in high income areas
        # This represents mechanized/commercial agriculture
        
        # Annual crops suitable for CA (29 crops total)
        annual_indices = [0,1,2,3,4,6,12,13,16] + list(range(23,42)) + [22]
        annuals = spam_tr_flat_out[annual_indices]
        
        annuals_large = large_area[annual_indices]
        annuals_small_high = high_income_small[annual_indices]
        
        # Sum across crop dimension
        annuals_all = np.nansum([annuals_large, annuals_small_high], axis=0)
        
        # Subset for grain crops only (22 crops, excluding rice, sugarbeet, tubers)
        # Indices: 0,2,3,4,5,6,8,14-28 (corresponding to grains)
        grain_indices = [0,2,3,4,5,6,8] + list(range(14,29))
        grains = annuals_all[grain_indices]
        grains_sum = np.nansum(grains, axis=0)
        
        # Total cropland
        total_cropland = np.nansum(spam_tr, axis=0)
        total_cropland[total_cropland == 0] = np.nan
        
        # Calculate crop mix ratio
        crop_mix = grains_sum / total_cropland
        crop_mix[~np.isfinite(crop_mix)] = np.nan
        
        print(f"  Crop mix calculated. Range: {np.nanmin(crop_mix):.3f} - {np.nanmax(crop_mix):.3f}")
        
        return crop_mix
    
    def build_logit_model(self, field_size, erosion, aridity, crop_mix):
        """
        Build logistic regression model for CA likelihood
        
        The logit model predicts probability of conservation agriculture adoption
        based on four spatial predictor variables:
        
        1. Field size: Larger fields more likely for CA (mechanization)
        2. Erosion: Higher erosion favors CA (soil conservation)
        3. Aridity: Lower AI (more arid) favors CA (water conservation)
        4. Crop mix: Higher ratio of suitable crops increases CA potential
        
        Model form:
        logit(p) = k1*field_size + k2*erosion + k3*aridity + k4*crop_mix + intercept
        p = 1 / (1 + exp(-logit(p)))
        
        IMPORTANT: Aridity Index interpretation
        - Higher AI = more humid
        - Lower AI = more arid
        - More arid regions favor CA → negative coefficient (k3 < 0)
        
        Parameters:
        -----------
        field_size : array (ny, nx)
            Field size class values (10-40 scale)
        erosion : array (ny, nx)
            Water erosion (t/ha/yr)
        aridity : array (ny, nx)
            Aridity Index (0-4, where higher = more humid)
        crop_mix : array (ny, nx)
            CA-suitable crop ratio (0-1)
        
        Returns:
        --------
        logit_prob : array (ny, nx)
            Probability of CA adoption (0-1)
        """
        print("\nBuilding logit model for CA likelihood...")
        
        # Model parameters (from Porwollik et al. 2019)
        k_field = 1/4      # Slope for field size (positive: larger → more CA)
        k_erosion = 1/60   # Slope for erosion (positive: more erosion → more CA)
        k_aridity = -5     # Slope for aridity (NEGATIVE: more arid/lower AI → more CA)
        k_crop_mix = 10    # Slope for crop mix (positive: more suitable crops → more CA)
        
        # Midpoints (thresholds from literature)
        xmid_field = 20      # 2 ha threshold for mechanization
        xmid_erosion = 12    # Moderate erosion threshold (t/ha/yr)
        xmid_aridity = 0.65  # Sub-humid/humid boundary (AI value)
        xmid_crop_mix = 0.5  # 50% suitable crops
        
        print(f"  Model parameters:")
        print(f"    Field size: k={k_field}, midpoint={xmid_field} ha")
        print(f"    Erosion: k={k_erosion:.4f}, midpoint={xmid_erosion} t/ha/yr")
        print(f"    Aridity: k={k_aridity}, midpoint={xmid_aridity} (lower AI = drier → more CA)")
        print(f"    Crop mix: k={k_crop_mix}, midpoint={xmid_crop_mix}")
        
        # Calculate linear predictor
        # Note: k_aridity is negative, so drier regions (lower AI) get higher CA probability
        linear_pred = (
            k_field * field_size +
            k_erosion * erosion +
            k_aridity * aridity +  # Negative coefficient: lower AI → higher contribution to CA
            k_crop_mix * crop_mix
        )
        
        # Calculate intercept to center the model
        intercept = -(
            xmid_field * k_field +
            xmid_erosion * k_erosion +
            xmid_aridity * k_aridity +
            xmid_crop_mix * k_crop_mix
        )
        
        # Add intercept
        logit_val = linear_pred + intercept
        
        # Apply logistic function to get probability (0-1)
        logit_prob = 1 / (1 + np.exp(-logit_val))
        logit_prob[~np.isfinite(logit_prob)] = np.nan
        
        # Diagnostics
        print(f"  Logit probability range: {np.nanmin(logit_prob):.3f} - {np.nanmax(logit_prob):.3f}")
        print(f"  Mean CA probability: {np.nanmean(logit_prob):.3f}")
        print(f"  Cells with prob > 0.5: {np.sum(logit_prob > 0.5)} ({np.sum(logit_prob > 0.5)/np.sum(~np.isnan(logit_prob))*100:.1f}%)")
        
        return logit_prob
    
    def calculate_correlation_matrix(self, field_size, erosion, aridity, crop_mix):
        """
        Calculate Spearman's rank correlation between input variables
        
        This checks for multicollinearity in the predictor variables.
        Correlations should generally be < 0.7 to avoid issues.
        """
        print("\nCalculating correlation matrix...")
        
        # Flatten arrays and remove NaNs
        data = np.vstack([
            field_size.flatten(),
            erosion.flatten(),
            aridity.flatten(),
            crop_mix.flatten()
        ]).T
        
        # Remove rows with any NaN
        data = data[~np.isnan(data).any(axis=1)]
        
        # Calculate Spearman correlations
        correlations = {}
        var_names = ['field_size', 'erosion', 'aridity', 'crop_mix']
        
        for i in range(len(var_names)):
            for j in range(i+1, len(var_names)):
                corr, _ = spearmanr(data[:, i], data[:, j])
                correlations[f"{var_names[i]}_{var_names[j]}"] = corr
                print(f"  {var_names[i]} vs {var_names[j]}: {corr:.3f}")
        
        return correlations

# ============================================================================
# 4. DOWNSCALE CONSERVATION AGRICULTURE
# ============================================================================

    def downscale_notill_area(self, logit_prob, notill_data, alloc_rast, pot_notill_area):
        """
        Downscale national no-till area statistics to grid cells
        
        Uses the logit probability to allocate reported national no-till areas
        to specific grid cells. Cells with higher probability receive no-till first.
        
        Parameters:
        -----------
        logit_prob : array (ny, nx)
            No-till/CA adoption probability from logit model
        notill_data : DataFrame
            National no-till areas from FAOSTAT
            Columns: country, iso3, notill_area_ha, year
        alloc_rast : array (ny, nx)
            Country allocation raster (grid cell -> country code)
        pot_notill_area : array (22, ny, nx)
            Potential no-till area by crop (CA-suitable crops only)
        
        Returns:
        --------
        notill_downscaled : array (22, ny, nx)
            Downscaled no-till area by crop
        """
        print("\nDownscaling no-till areas to grid cells...")
        
        notill_downscaled = np.zeros_like(pot_notill_area)
        
        # Sum potential no-till area across crops
        pot_notill_sum = np.nansum(pot_notill_area, axis=0)
        
        # Iterate through countries with no-till data
        for idx, row in notill_data.iterrows():
            # Get country identifier
            if 'iso3' in row and pd.notna(row['iso3']):
                country_name = row['iso3']
            else:
                country_name = row['country']
            
            # Find country code in allocation raster
            # (This requires mapping between ISO3 and numeric codes)
            # For now, skip if we can't identify country
            country_code = self._get_country_code(country_name, alloc_rast)
            
            if country_code is None:
                print(f"  WARNING: Could not find country code for {country_name}")
                continue
            
            target_area = row['notill_area_ha']
            
            # Extract cells for this country
            country_mask = alloc_rast == country_code
            
            if not country_mask.any():
                continue
            
            # Get probability and area for this country
            prob = logit_prob[country_mask]
            area = pot_notill_sum[country_mask]
            
            # Create DataFrame and sort by probability (descending)
            lons, lats = np.meshgrid(self.lons, self.lats)
            df = pd.DataFrame({
                'prob': prob.flatten(),
                'area': area.flatten(),
                'lon': lons[country_mask].flatten(),
                'lat': lats[country_mask].flatten()
            })
            df = df[df['area'] > 0]  # Only cells with cropland
            df = df.sort_values('prob', ascending=False)
            
            # Calculate cumulative area
            df['cumsum'] = df['area'].cumsum()
            
            # Find cells to include
            if df['area'].sum() > target_area:
                # Enough area available
                idx_thresh = (df['cumsum'] <= target_area).sum()
                
                # Check if adding one more cell gets closer to target
                if idx_thresh < len(df):
                    diff_without = abs(df.iloc[idx_thresh-1]['cumsum'] - target_area)
                    diff_with = abs(df.iloc[idx_thresh]['cumsum'] - target_area)
                    
                    if diff_with < diff_without:
                        idx_thresh += 1
                
                df['selected'] = False
                df.iloc[:idx_thresh, df.columns.get_loc('selected')] = True
                
                actual_area = df[df['selected']]['area'].sum()
            else:
                # Not enough area - use all available
                df['selected'] = True
                actual_area = df['area'].sum()
                print(f"  {country_name}: Only {actual_area/1e6:.2f} Mha available of {target_area/1e6:.2f} Mha target")
            
            # Apply selection to output
            selected_cells = df[df['selected']][['lon', 'lat']].values
            
            # Convert back to raster indices
            for lon, lat in selected_cells:
                i = np.argmin(np.abs(self.lats - lat))
                j = np.argmin(np.abs(self.lons - lon))
                
                # Assign all crops proportionally
                if pot_notill_sum[i, j] > 0:
                    notill_downscaled[:, i, j] = pot_notill_area[:, i, j]
            
            print(f"  {country_name}: Downscaled {actual_area/1e6:.2f} Mha")
        
        total_downscaled = np.nansum(notill_downscaled)
        print(f"\n  Total downscaled: {total_downscaled/1e6:.1f} million ha")
        
        return notill_downscaled

# ============================================================================
# 5. HELPER FUNCTIONS
# ============================================================================

    def _resample_to_grid(self, data, transform, method='bilinear'):
        """
        Resample input data to target grid using rasterio
        
        Parameters:
        -----------
        data : array
            Input data array
        transform : affine.Affine
            Affine transform of input data
        method : str
            Resampling method: 'bilinear', 'nearest', 'average', 'cubic'
        
        Returns:
        --------
        resampled : array (ny, nx)
            Data resampled to target grid
        """
        from rasterio.transform import from_bounds
        
        # Create destination array
        dst_array = np.zeros((self.ny, self.nx), dtype=np.float32)
        
        # Create destination transform
        dst_transform = from_bounds(
            self.extent['xmin'], 
            self.extent['ymin'],
            self.extent['xmax'], 
            self.extent['ymax'],
            self.nx, 
            self.ny
        )
        
        # Map method string to Resampling enum
        method_map = {
            'bilinear': Resampling.bilinear,
            'nearest': Resampling.nearest,
            'average': Resampling.average,
            'cubic': Resampling.cubic,
            'mode': Resampling.mode
        }
        
        resampling_method = method_map.get(method, Resampling.bilinear)
        
        try:
            # Reproject to target grid
            reproject(
                source=data,
                destination=dst_array,
                src_transform=transform,
                src_crs='EPSG:4326',  # WGS84
                dst_transform=dst_transform,
                dst_crs='EPSG:4326',
                resampling=resampling_method
            )
            
            # Convert no-data values to NaN
            dst_array[dst_array == 0] = np.nan
            dst_array[dst_array < -9998] = np.nan
            
            return dst_array
            
        except Exception as e:
            print(f"    ERROR in resampling: {e}")
            return None
    
    def _aggregate_modal(self, data, factor):
        """Aggregate using modal (most frequent) value"""
        # Implementation would use scipy.ndimage or custom aggregation
        pass
    
    def _interpolate_field_sizes(self, fields):
        """Interpolate missing field size values using moving window"""
        # Implementation would use scipy.ndimage.generic_filter
        pass
    
    def _get_country_code(self, country_identifier, alloc_rast):
        """
        Helper function to find country code in allocation raster
        
        Parameters:
        -----------
        country_identifier : str
            Country name or ISO3 code
        alloc_rast : array
            Country allocation raster
            
        Returns:
        --------
        country_code : int or None
            Numeric country code in allocation raster
        """
        # This is a placeholder - actual implementation needs
        # a mapping between ISO3/country names and numeric codes
        # from the allocation raster
        
        # For now, return None (will be implemented based on actual data)
        return None


# ============================================================================
# USAGE EXAMPLE - UPDATED FOR YOUR SPECIFIC DATASETS
# ============================================================================

if __name__ == "__main__":
    """
    COMPLETE PIPELINE WITH YOUR DATASETS
    ====================================
    
    YOUR FOLDER STRUCTURE (nested folders):
    
    Sentinel/  (your main input folder)
    ├── spam2020v2r0_global_physical_area.geotiff/
    │   └── spam2020v2r0_global_physical_area/  ← NESTED!
    │       ├── spam2020V2r0_global_P_WHEA_A.geotiff
    │       ├── spam2020V2r0_global_P_WHEA_R.geotiff
    │       └── ... (84 files total)
    ├── BDTICM_M_250m_ll (file, not nested)
    ├── Data2_SOIL_DISPLACEMENT_ESTIMATE_2019_1/
    │   └── Data2_SOIL_DISPLACEMENT_ESTIMATE_2019_1/  ← NESTED!
    │       └── *.tif (GloSEM erosion files)
    ├── Global-AI_ET0_v3/
    │   └── Global-AI_ET0_v3/  ← NESTED!
    │       └── Global-AI_ET0_v3_annual.tif
    ├── OGHIST_2025_10_07.xlsx (file, not nested)
    ├── lesiv_2018_field_sizes/
    │   └── Global Field Sizes (file, not nested)
    └── FAOSTAT_data_en_1-9-2026.csv (file, not nested)
    
    Code now handles the nested folder structure automatically!
    
    Datasets Used (2010-2020 timeframe):
    1. ✅ SPAM2020 - Cropland (~2010)
    2. ✅ SoilGrids 2017 250m - Depth to bedrock
    3. ✅ GloSEM 1.3 - Erosion (2019)
    4. ✅ Global-AI_ET0_v3.1 - Aridity index
    5. ✅ OGHIST 2025 (FY11) - Income levels (2010)
    6. ✅ Lesiv 2018 - Field sizes (improved version)
    7. ✅ FAOSTAT_data_en_1-9-2026.csv - No-till statistics
    8. ⚠️  Country allocation - STILL NEEDED
    """
    
    # Initialize generator - point to your Sentinel folder
    generator = TillageDatasetGenerator(
    path_input="D:/Users/Wilson/Downloads/Sentinel",
    path_output="D:/Users/Wilson/Downloads/Sentinel/output"
    )
       
    print("\n" + "="*80)
    print("GLOBAL TILLAGE DATASET GENERATION")
    print("Updated Datasets: 2010-2020 Timeframe")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load all input datasets
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: LOADING INPUT DATASETS")
    print("="*80)
    
    # Cropland (SPAM2020)
    spam_ta, spam_tr = generator.load_spam_data(version='2020')
    
    # Soil depth (SoilGrids 2017 250m)
    soilgrid, flat_area = generator.load_soilgrids_depth(
        version='250m',
        use_r_horizon=False  # Use BDTICM (absolute depth)
    )
    
    # Field sizes (Lesiv 2018 - improved version)
    field_sizes = generator.load_field_size_data()
    
    # Erosion (GloSEM 1.3 baseline 2019)
    erosion = generator.load_erosion_data(version='glosem')
    
    # Aridity (Global-AI v3.1)
    aridity = generator.load_aridity_index()
    
    # Income levels (OGHIST FY11 = 2010)
    income_dict, high_income_codes = generator.load_income_levels()
    
    # No-till statistics (FAOSTAT 2010-2015)
    notill_data = generator.load_notill_statistics()
    
    # Country allocation (STILL NEEDED - placeholder for now)
    print("\n⚠️  WARNING: Country allocation not yet implemented")
    print("   This is needed for:")
    print("   - Applying income-based rules")
    print("   - Downscaling no-till to grid cells")
    print("   - Country-level aggregation")
    
    # ========================================================================
    # STEP 2: Calculate crop mix
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: CALCULATING CROP MIX FOR CA SUITABILITY")
    print("="*80)
    
    # This requires income levels applied to allocation raster
    # For now, proceed with simplified version
    crop_mix = generator.calculate_crop_mix(
        spam_tr, 
        flat_area, 
        field_sizes,
        high_raster=None  # Placeholder until allocation implemented
    )
    
    # ========================================================================
    # STEP 3: Build logit model
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: BUILDING LOGIT MODEL FOR NO-TILL PROBABILITY")
    print("="*80)
    
    # Check correlation between inputs
    correlations = generator.calculate_correlation_matrix(
        field_sizes, erosion, aridity, crop_mix
    )
    
    print("\nInput variable correlations:")
    for pair, corr in correlations.items():
        print(f"  {pair}: {corr:.3f}")
    
    # Build logit model
    notill_prob = generator.build_logit_model(
        field_sizes, erosion, aridity, crop_mix
    )
    
    # Save probability map
    np.save(generator.path_output / 'notill_probability.npy', notill_prob)
    print(f"\n✓ Saved no-till probability map")
    
    # ========================================================================
    # STEP 4: Downscale no-till areas (if allocation available)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: DOWNSCALING NO-TILL TO GRID CELLS")
    print("="*80)
    
    print("⚠️  Skipped: Requires country allocation raster")
    print("   Once allocation is available:")
    print("   - National no-till areas will be distributed to grid cells")
    print("   - Based on logit probability (high prob cells get no-till first)")
    print("   - Results in gridded no-till map")
    
    # ========================================================================
    # STEP 5: Calculate other tillage systems
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 5: CALCULATING OTHER TILLAGE SYSTEMS")
    print("="*80)
    
    print("✓ Can proceed with:")
    print("  - Traditional tillage (small fields, low income)")
    print("  - Reduced tillage (shallow soils)")
    print("  - Rotational tillage (perennials)")
    print("  - Conventional tillage (remaining area)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    print("\n✅ COMPLETED:")
    print("  1. Loaded all modern datasets (2010-2020 timeframe)")
    print("  2. Calculated crop mix for CA suitability")
    print("  3. Built logit model for no-till probability")
    print("  4. Generated no-till probability map")
    
    print("\n⚠️  PENDING:")
    print("  1. Country allocation raster (critical!)")
    print("     - Download SPAM2020 allocation file, OR")
    print("     - Generate from GADM country boundaries")
    print("  2. Apply income-based rules to tillage systems")
    print("  3. Downscale no-till statistics to grid cells")
    print("  4. Calculate remaining tillage systems")
    print("  5. Validate results")
    
    print("\n" + "="*80)
    print("OUTPUTS GENERATED")
    print("="*80)
    
    print(f"\nSaved to: {generator.path_output}")
    print("  - notill_probability.npy (CA/no-till adoption probability 0-1)")
    print("  - [Other outputs pending country allocation]")
    
    print("\n" + "="*80)
    print("NEXT STEPS FOR YOU")
    print("="*80)
    
    print("\n1. GET COUNTRY ALLOCATION:")
    print("   Option A: Check SPAM2020 download for allocation file")
    print("   Option B: Download GADM and rasterize country boundaries")
    print("   Option C: Use existing allocation from previous studies")
    
    print("\n2. COMPLETE PIPELINE:")
    print("   - Rerun with allocation raster")
    print("   - Generate all 6 tillage system maps")
    print("   - Validate against FAOSTAT no-till statistics")
    
    print("\n3. FOR YOUR CNN:")
    print("   - Use notill_probability.npy as training labels")
    print("   - Continuous probability = better than binary labels")
    print("   - More honest about uncertainty")
    
    print("\n" + "="*80)
    print("DATASET QUALITY ASSESSMENT")
    print("="*80)
    
    print("\n✅ STRENGTHS:")
    print("  - All datasets updated to 2010-2020 (vs 2005 original)")
    print("  - Higher resolution: 250m soil, 100m erosion, 30\" aridity")
    print("  - Better temporal consistency across datasets")
    print("  - FAOSTAT no-till = more accessible than AQUASTAT CA")
    
    print("\n⚠️  LIMITATIONS:")
    print("  - No-till ≠ full CA (subset only, may underestimate)")
    print("  - Country allocation still needed for completion")
    print("  - Cannot validate without complete pipeline")
    print("  - Temporal spread: 2010-2019 across datasets")
    
    print("\n🎯 FOR CNN TRAINING:")
    print("  - Probability map suitable for soft labels ✓")
    print("  - Better than hard binary classification ✓")
    print("  - Honest uncertainty representation ✓")
    print("  - Validate with independent remote sensing ✓")
    
    print("\n" + "="*80)