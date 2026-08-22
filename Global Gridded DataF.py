"""
Complete Global Tillage Dataset Generation - Python Translation
================================================================

Translated from Porwollik et al. (2019) R code to Python
Updated for modern datasets (2010-2020 timeframe)

Uses VECTOR-BASED country allocation (no rasterization needed!)

Author: Python translation with modernized datasets
Original: Porwollik, Vera; Rolinski, Susanne; Müller, Christoph (2019)
Citation: http://doi.org/10.5880/PIK.2019.010
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.ndimage import generic_filter
from scipy.stats import spearmanr
import xarray as xr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class GlobalTillageDataset:
    """
    Complete pipeline for generating global gridded tillage dataset
    """
    
    def __init__(self, path_input, path_output, sample_calc=False):
        self.path_input = Path(path_input)
        self.path_output = Path(path_output)
        self.path_output.mkdir(exist_ok=True, parents=True)
        self.sample_calc = sample_calc
        
        # Grid setup
        if sample_calc:
            self.extent = {'xmin': -180, 'xmax': 180, 'ymin': -45, 'ymax': 90}
            self.resolution = 45
        else:
            self.extent = {'xmin': -180, 'xmax': 180, 'ymin': -56, 'ymax': 84}
            self.resolution = 1/12  # 5 arcmin
        
        self.nx = int((self.extent['xmax'] - self.extent['xmin']) / self.resolution)
        self.ny = int((self.extent['ymax'] - self.extent['ymin']) / self.resolution)
        
        # Create coordinate arrays
        self.lons = np.linspace(self.extent['xmin'] + self.resolution/2,
                                self.extent['xmax'] - self.resolution/2, self.nx)
        self.lats = np.linspace(self.extent['ymax'] - self.resolution/2,
                                self.extent['ymin'] + self.resolution/2, self.ny)
        
        # 42 crops from SPAM
        self.crop_names = [
            'WHEA', 'RICE', 'MAIZ', 'BARL', 'REST', 'OOIL', 'TOBA', 'TEAS',
            'COCO', 'RCOF', 'ACOF', 'OFIB', 'COTT', 'SUGB', 'SUGC', 'OILP',
            'VEGE', 'TEMF', 'TROF', 'PLNT', 'BANA', 'CNUT', 'GROU', 'OTRS',
            'CASS', 'YAMS', 'SWPO', 'POTA', 'SESA', 'RAPE', 'SUNF', 'SOYB',
            'OPUL', 'LENT', 'PIGE', 'COWP', 'CHIC', 'BEAN', 'OCER', 'SORG',
            'SMIL', 'PMIL'
        ]
        
        print(f"\n{'='*80}")
        print(f"GLOBAL TILLAGE DATASET GENERATOR")
        print(f"{'='*80}")
        print(f"Grid: {self.nx} x {self.ny} ({self.resolution}°)")
        print(f"Sample mode: {sample_calc}")
    
    # ========================================================================
    # 1. DATA LOADING
    # ========================================================================
    
    def load_spam_data(self):
        """Load SPAM2020 cropland data"""
        print(f"\nLoading SPAM2020 cropland...")
        
        if self.sample_calc:
            spam_ta = np.tile(np.arange(1, 25), (42, 1)).reshape(42, self.ny, self.nx)
            spam_tr = spam_ta - 0.33
            return spam_ta, spam_tr
        
        spam_folder = self.path_input / "spam2020v2r0_global_physical_area.geotiff" / "spam2020v2r0_global_physical_area"
        
        spam_ta = np.zeros((42, self.ny, self.nx), dtype=np.float32)
        spam_tr = np.zeros((42, self.ny, self.nx), dtype=np.float32)
        
        for i, crop in enumerate(self.crop_names):
            ta_file = spam_folder / f"spam2020_V2r0_global_A_{crop}_A.tif"
            tr_file = spam_folder / f"spam2020_V2r0_global_A_{crop}_R.tif"
            
            if ta_file.exists():
                with rasterio.open(ta_file) as src:
                    spam_ta[i] = self._resample(src.read(1), src.transform)
            
            if tr_file.exists():
                with rasterio.open(tr_file) as src:
                    spam_tr[i] = self._resample(src.read(1), src.transform)
            
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i+1}/42 crops...")
        
        print(f"  ✓ Loaded {np.sum(~np.all(spam_ta==0, axis=(1,2)))}/42 crops")
        return spam_ta, spam_tr
    
    def load_soilgrids(self):
        """
        Load SoilGrids depth to bedrock
        
        CRITICAL FIX: In R code, flat_area is used to IDENTIFY shallow soils,
        not to remove them. They are allocated to "reduced" tillage later.
        """
        print(f"\nLoading SoilGrids depth to bedrock...")
        
        if self.sample_calc:
            soil = np.random.uniform(0, 200, (self.ny, self.nx))
            flat = soil.copy()
            flat[soil >= 15] = np.nan
            return soil, flat
        
        soil_file = self.path_input / "BDTICM_M_250m_ll.tif"
        
        with rasterio.open(soil_file) as src:
            print(f"  Reading at target resolution...")
            soil = src.read(1, out_shape=(self.ny, self.nx), 
                           resampling=Resampling.bilinear).astype(np.float32)
        
        soil[soil < 0] = np.nan
        
        # Create flat mask: soil depth < 15cm
        # In R: flat_area <- soil_depth; flat_area[flat_area >= 15] <- NA
        flat = soil.copy()
        flat[flat >= 15] = np.nan
        
        print(f"  ✓ Range: {np.nanmin(soil):.0f}-{np.nanmax(soil):.0f} cm")
        print(f"  DEBUG: Pixels < 15cm: {np.sum(~np.isnan(flat)):,} ({100*np.sum(~np.isnan(flat))/flat.size:.1f}%)")
        
        return soil, flat
    
    def load_fields(self):
        """Load field sizes at lower resolution if needed"""
        print(f"\nLoading field sizes...")
        
        if self.sample_calc:
            return np.tile(np.arange(10, 34), (self.ny // 4, self.nx // 4))[:self.ny, :self.nx]
        
        field_file = self.path_input / "lesiv_2018_field_sizes" / "Global Field Sizes" / "dominant_field_size_categories.tif"
        
        with rasterio.open(field_file) as src:
            # Read at 1/4 resolution first
            ny_low = self.ny // 4
            nx_low = self.nx // 4
            
            fields_low = src.read(1, out_shape=(ny_low, nx_low),
                                resampling=Resampling.nearest).astype(np.float32)
        
        fields_low[fields_low < 0] = np.nan
        
        # Interpolate low-res gaps (much faster)
        print(f"  Interpolating at low resolution...")
        fields_low = self._interpolate_fields_optimized(fields_low)
        
        # Upscale to target resolution
        print(f"  Upscaling to target resolution...")
        from scipy.ndimage import zoom
        fields = zoom(fields_low, 4, order=0)  # Nearest neighbor
        
        # Ensure correct size (zoom might be slightly off)
        if fields.shape != (self.ny, self.nx):
            fields = fields[:self.ny, :self.nx]
        
        print(f"  ✓ Range: {np.nanmin(fields):.0f}-{np.nanmax(fields):.0f}")
        return fields
    
    def load_erosion(self):
        """Load GloSEM erosion data"""
        print(f"\nLoading GloSEM erosion...")
        
        if self.sample_calc:
            return np.tile(np.arange(1, 25), (self.ny // 4, self.nx // 4))[:self.ny, :self.nx]
        
        erosion_folder = self.path_input / "Data2_SOIL_DISPLACEMENT_ESTIMATE_2019_1" / "Data2_SOIL_DISPLACEMENT_ESTIMATE_2019_1"
        erosion_file = erosion_folder / "SOIL_DISPLACEMENT_ESTIMATE_2019.tif"
        
        if not erosion_file.exists():
            erosion_file = erosion_folder / "SOIL_DISPLACEMENT_ESTIMATE_2019.tif.ovr"
        
        with rasterio.open(erosion_file) as src:
            erosion = src.read(1, out_shape=(self.ny, self.nx),
                             resampling=Resampling.average).astype(np.float32)
        
        erosion[erosion < 0] = np.nan
        erosion[erosion > 10000] = np.nan
        
        print(f"  ✓ Range: {np.nanmin(erosion):.1f}-{np.nanmax(erosion):.1f} t/ha/yr")
        return erosion
    
    def load_aridity(self):
        """Load Global Aridity Index"""
        print(f"\nLoading aridity index...")
        
        if self.sample_calc:
            ari = np.tile(np.arange(1, 24), (self.ny // 4, self.nx // 4))[:self.ny, :self.nx] / 100.0
            return ari
        
        ari_file = self.path_input / "Global-AI_ET0_annual_v3" / "Global-AI_ET0_v3_annual" / "ai_v3_yr.tif"
        
        with rasterio.open(ari_file) as src:
            ari = src.read(1, out_shape=(self.ny, self.nx),
                          resampling=Resampling.bilinear).astype(np.float32)
        
        # Scale if needed (Global-AI v3 uses scaling factor of 10000)
        if np.nanmax(ari) > 100:
            ari = ari / 10000.0
        
        ari[ari < 0] = np.nan
        ari[ari > 10] = np.nan
        
        print(f"  ✓ Range: {np.nanmin(ari):.3f}-{np.nanmax(ari):.3f}")
        return ari
    
    def load_country_allocation(self):
        """Load country allocation using vector GeoPackage"""
        print(f"\nLoading country allocation (vector)...")
        
        if self.sample_calc:
            return np.full((self.ny, self.nx), 240, dtype=np.uint16)
        
        # Load the vector country data
        countries_gpkg = self.path_output / "country_allocation_fixed.gpkg"
        
        if not countries_gpkg.exists():
            raise FileNotFoundError(
                f"Country allocation not found!\n"
                f"Run the country allocation script first to generate:\n"
                f"  {countries_gpkg}"
            )
        
        gdf = gpd.read_file(countries_gpkg, layer='countries')
        
        # Rasterize using country_code
        print(f"  Rasterizing {len(gdf)} countries...")
        
        transform = from_bounds(self.extent['xmin'], self.extent['ymin'],
                               self.extent['xmax'], self.extent['ymax'],
                               self.nx, self.ny)
        
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['country_code']))
        
        alloc = rasterize(shapes=shapes, out_shape=(self.ny, self.nx),
                         transform=transform, fill=0, dtype=np.uint16)
        
        alloc = alloc.astype(np.float32)
        alloc[alloc == 0] = np.nan
        
        print(f"  ✓ Rasterized to {self.nx}x{self.ny}")
        
        # Also load the mapping
        mapping = pd.read_csv(self.path_output / "country_code_mapping.csv")
        self.country_mapping = mapping
        
        return alloc
    
    def load_income_levels(self):
        """Load and process income level data"""
        print(f"\nLoading income levels...")
        
        if self.sample_calc:
            return np.full((self.ny, self.nx), 4, dtype=np.float32)  # All high income
        
        # Load mapping if not already loaded
        if not hasattr(self, 'country_mapping'):
            self.country_mapping = pd.read_csv(self.path_output / "country_code_mapping.csv")
        
        # Create high income mask (income_code >= 3)
        high_income_codes = self.country_mapping[
            self.country_mapping['income_code'] >= 3
        ]['country_code'].values
        
        # Load allocation raster
        alloc = self.load_country_allocation()
        
        # Create high income raster
        high_raster = np.full_like(alloc, np.nan)
        for code in high_income_codes:
            high_raster[alloc == code] = 400
        
        print(f"  ✓ {len(high_income_codes)} high/upper-middle income countries")
        return high_raster
    
    def load_ca_statistics(self):
        """Load Conservation Agriculture/No-Till statistics"""
        print(f"\nLoading CA/No-Till statistics...")
        
        if self.sample_calc:
            return pd.DataFrame({
                'country': ['SAMPLE'],
                'iso3': ['SAM'],
                'ca_area_ha': [10000],
                'country_code': [240]
            })
        
        # Try to find FAOSTAT file
        fao_files = list(self.path_input.glob("FAOSTAT*.csv"))
        
        if not fao_files:
            print(f"  ⚠️ No FAOSTAT file found, using mapping countries as placeholder")
            # Use countries from mapping as placeholder
            df = self.country_mapping[['iso3', 'country_code']].copy()
            df['ca_area_ha'] = 1000000  # 1M ha placeholder
            return df
        
        df = pd.read_csv(fao_files[0])
        
        # Process FAOSTAT format
        if 'Area' in df.columns and 'Value' in df.columns:
            # Standard FAOSTAT format
            if 'Unit' in df.columns and '1000' in str(df['Unit'].iloc[0]):
                df['Value'] = df['Value'] * 1000
            
            df_latest = df.sort_values('Year').groupby('Area').last().reset_index() if 'Year' in df.columns else df
            
            result = pd.DataFrame({
                'country': df_latest['Area'],
                'ca_area_ha': df_latest['Value']
            })
            
            # Merge with country codes
            result = result.merge(self.country_mapping[['iso3', 'country_code']], 
                                 left_on='country', right_on='iso3', how='left')
        
        result = result.dropna(subset=['ca_area_ha', 'country_code'])
        result = result[result['ca_area_ha'] > 0]
        
        print(f"  ✓ {len(result)} countries with CA data")
        print(f"  Total: {result['ca_area_ha'].sum()/1e6:.1f} Mha")
        
        return result
    
    # ========================================================================
    # 2. CROP MIX CALCULATION
    # ========================================================================
    
    def calculate_crop_mix(self, spam_ta, spam_tr, flat_area, fields, high_raster):
        """
        Calculate crop mix: ratio of CA-suitable crops to total cropland
        
        Following original R code logic:
        1. Remove flat areas (<15cm depth)
        2. Separate small (<2ha) and large (>=2ha) fields
        3. For annuals: use large fields + small fields in high income countries
        4. Subset to 22 CA-suitable grain crops
        5. Calculate ratio to total cropland
        """
        print(f"\nCalculating crop mix...")
        
        # Step 1: Remove flat areas
        spam_tr_flat_out = spam_tr.copy()
        for i in range(42):
            spam_tr_flat_out[i][np.isnan(flat_area)] = np.nan
        
        # Step 2: Separate by field size
        fields_small = fields < 20
        fields_large = fields >= 20
        
        # Small fields
        small = spam_tr_flat_out.copy()
        for i in range(42):
            small[i][~fields_small] = 0
        
        # Small fields in high income
        high_income_small = small.copy()
        for i in range(42):
            high_income_small[i][np.isnan(high_raster)] = np.nan
        
        # Large fields
        annuals_large = spam_tr_flat_out.copy()
        for i in range(42):
            annuals_large[i][~fields_large] = np.nan
        
        # Step 3: Build annuals stack (29 crops)
        # Indices: 0,1,2,3,4,6,12,13,16,23-41,22
        annual_indices = [0,1,2,3,4,6,12,13,16] + list(range(23,42)) + [22]
        
        annuals_all = np.zeros((29, self.ny, self.nx), dtype=np.float32)
        for i, idx in enumerate(annual_indices):
            annuals_all[i] = np.nansum([annuals_large[idx], high_income_small[idx]], axis=0)
        
        # Step 4: Subset to 22 CA-suitable grains (excluding rice, sugarbeet, tubers)
        # Grains indices in annuals_all: 0,2,3,4,5,6,8,14-28 (22 crops)
        grain_indices = [0,2,3,4,5,6,8] + list(range(14,29))
        grains_sum = np.nansum(annuals_all[grain_indices], axis=0)
        
        # Step 5: Calculate ratio
        spam_ta_sum = np.nansum(spam_ta, axis=0)
        spam_ta_sum[spam_ta_sum == 0] = np.nan
        
        crop_mix = grains_sum / spam_ta_sum
        crop_mix[~np.isfinite(crop_mix)] = np.nan
        
        print(f"  ✓ Range: {np.nanmin(crop_mix):.3f}-{np.nanmax(crop_mix):.3f}")
        
        # Save intermediate for CA calculation
        self.grains_stack = annuals_all[grain_indices]  # 22 crops
        
        return crop_mix
    
    # ========================================================================
    # 3. LOGIT MODEL
    # ========================================================================
    
    def build_logit_model(self, fields, erosion, aridity, crop_mix):
        """Build logit model for CA likelihood"""
        print(f"\nBuilding logit model...")
        
        # Interpolate erosion and aridity where needed
        erosion_interp = erosion.copy()
        erosion_interp[np.isnan(erosion_interp) & ~np.isnan(crop_mix)] = 12
        
        aridity_interp = aridity.copy()
        aridity_interp = self._interpolate_missing(aridity_interp, crop_mix)
        aridity_interp[np.isnan(aridity_interp) & ~np.isnan(crop_mix)] = 0.65
        
        # Model parameters (from original R code)
        k = np.array([1/4, 1/60, -5, 10])  # field, erosion, aridity, crop_mix
        xmid = np.array([20, 12, 0.65, 0.5])
        
        # Calculate logit
        b = (k[0] * fields + k[1] * erosion_interp + 
             k[2] * aridity_interp + k[3] * crop_mix)
        f = -np.sum(xmid * k)
        d = b + f
        e = -d
        
        logit = 1 / (1 + np.exp(e))
        logit[~np.isfinite(logit)] = np.nan
        
        print(f"  ✓ Range: {np.nanmin(logit):.3f}-{np.nanmax(logit):.3f}")
        
        # Calculate correlations
        print(f"\n  Input variable correlations:")
        vars_data = np.column_stack([
            fields.flatten(),
            erosion_interp.flatten(),
            aridity_interp.flatten(),
            crop_mix.flatten()
        ])
        mask = ~np.isnan(vars_data).any(axis=1)
        vars_clean = vars_data[mask]
        
        var_names = ['field', 'erosion', 'aridity', 'crop_mix']
        for i in range(4):
            for j in range(i+1, 4):
                corr, _ = spearmanr(vars_clean[:, i], vars_clean[:, j])
                print(f"    {var_names[i]} vs {var_names[j]}: {corr:.3f}")
        
        return logit
    
    # ========================================================================
    # 4. DOWNSCALE CA
    # ========================================================================
    
    def downscale_ca(self, logit, ca_stats, alloc, pot_ca_area):
        """
        Downscale national CA statistics to grid cells
        
        Following original R code algorithm:
        1. For each country, get cells and their logit probability
        2. Sort cells by probability (descending)
        3. Cumulative sum of area until target reached
        4. Choose closest match (slightly under or slightly over)
        """
        print(f"\nDownscaling CA to grid cells...")
        
        ca_downscaled = np.zeros_like(pot_ca_area)
        
        for idx, row in ca_stats.iterrows():
            country_code = row['country_code']
            target_area = row['ca_area_ha']
            country_name = row.get('iso3', row.get('country', f'Code_{country_code}'))
            
            # Get cells for this country
            mask = alloc == country_code
            if not mask.any():
                continue
            
            # Extract data
            lons_grid, lats_grid = np.meshgrid(self.lons, self.lats)
            
            prob_vals = logit[mask]
            area_vals = pot_ca_area.sum(axis=0)[mask]  # Sum across 22 crops
            lon_vals = lons_grid[mask]
            lat_vals = lats_grid[mask]
            
            # Create dataframe
            df = pd.DataFrame({
                'lon': lon_vals,
                'lat': lat_vals,
                'prob': prob_vals,
                'area': area_vals
            })
            
            df = df[df['area'] > 0].sort_values('prob', ascending=False).reset_index(drop=True)
            
            if len(df) == 0:
                continue
            
            # Cumulative sum
            df['cumsum'] = df['area'].cumsum()
            total_available = df['area'].sum()
            
            if total_available > target_area:
                # Find threshold
                io = (df['cumsum'] <= target_area).sum()
                
                if io > 0 and io < len(df):
                    # Check if adding one more cell is closer
                    diff_without = abs(df.iloc[io-1]['cumsum'] - target_area)
                    diff_with = abs(df.iloc[io]['cumsum'] - target_area)
                    
                    if diff_with < diff_without:
                        io += 1
                
                df['selected'] = False
                df.loc[:io-1, 'selected'] = True
            else:
                # Use all available
                df['selected'] = True
                print(f"    {country_name}: {total_available/1e6:.2f} Mha available < {target_area/1e6:.2f} Mha target")
            
            # Apply selection
            selected = df[df['selected']]
            
            for _, cell in selected.iterrows():
                i = np.argmin(np.abs(self.lats - cell['lat']))
                j = np.argmin(np.abs(self.lons - cell['lon']))
                
                ca_downscaled[:, i, j] = pot_ca_area[:, i, j]
            
            actual = selected['area'].sum()
            print(f"    {country_name}: {actual/1e6:.2f} Mha")
        
        total = np.nansum(ca_downscaled)
        print(f"\n  ✓ Total downscaled: {total/1e6:.1f} Mha")
        
        return ca_downscaled
    
    # ========================================================================
    # 5. CALCULATE TILLAGE SYSTEMS
    # ========================================================================
    
    def calculate_traditional_annual(self, spam_ta, flat_area, fields, high_raster):
        """Traditional annual tillage: small fields in low income countries"""
        print(f"\nCalculating traditional annual tillage...")
        
        spam_ta_flat_out = spam_ta.copy()
        for i in range(42):
            spam_ta_flat_out[i][np.isnan(flat_area)] = np.nan
        
        # Small fields
        small = spam_ta_flat_out.copy()
        for i in range(42):
            small[i][fields >= 20] = np.nan
        
        # Low income (NOT high income)
        small_low = small.copy()
        for i in range(42):
            small_low[i][~np.isnan(high_raster)] = np.nan
        
        # Subset to 29 annuals
        annual_indices = [0,1,2,3,4,6,12,13,16] + list(range(23,42)) + [22]
        trad_annual = small_low[annual_indices]
        
        total = np.nansum(trad_annual)
        print(f"  ✓ {total/1e6:.1f} Mha")
        
        return trad_annual
    
    def calculate_traditional_rotational(self, spam_ta, flat_area, fields, high_raster):
        """Traditional rotational tillage: perennials on small fields in low income"""
        print(f"\nCalculating traditional rotational tillage...")
        
        spam_ta_flat_out = spam_ta.copy()
        for i in range(42):
            spam_ta_flat_out[i][np.isnan(flat_area)] = np.nan
        
        small = spam_ta_flat_out.copy()
        for i in range(42):
            small[i][fields >= 20] = np.nan
        
        small_low = small.copy()
        for i in range(42):
            small_low[i][~np.isnan(high_raster)] = np.nan
        
        # 13 perennials: indices 5,7,8,9,10,11,14,15,17,18,19,20,21
        peren_indices = [5,7,8,9,10,11,14,15,17,18,19,20,21]
        trad_rot = small_low[peren_indices]
        
        total = np.nansum(trad_rot)
        print(f"  ✓ {total/1e6:.1f} Mha")
        
        return trad_rot
    
    def calculate_rotational(self, spam_ta, flat_area, fields, high_raster, soil):
        """Rotational tillage: perennials on large fields or small+high income"""
        print(f"\nCalculating rotational tillage...")
        
        spam_ta_flat_out = spam_ta.copy()
        for i in range(42):
            spam_ta_flat_out[i][np.isnan(flat_area)] = np.nan
        
        peren_indices = [5,7,8,9,10,11,14,15,17,18,19,20,21]
        perms = spam_ta_flat_out[peren_indices]
        
        # Large fields
        perm_large = perms.copy()
        for i in range(len(peren_indices)):
            perm_large[i][fields < 20] = np.nan
        
        # Small fields in high income
        perm_small = perms.copy()
        for i in range(len(peren_indices)):
            perm_small[i][fields >= 20] = np.nan
        
        perm_small_high = perm_small.copy()
        for i in range(len(peren_indices)):
            perm_small_high[i][np.isnan(high_raster)] = np.nan
        
        # Combine
        pre_rot = np.nansum([perm_large, perm_small_high], axis=0)
        
        # Exclude areas with 15-20 cm depth (go to reduced)
        rot = pre_rot.copy()
        for i in range(len(peren_indices)):
            rot[i][(soil >= 15) & (soil < 20)] = np.nan
        
        total = np.nansum(rot)
        print(f"  ✓ {total/1e6:.1f} Mha")
        
        return rot
    
    def calculate_reduced(self, spam_ta, flat_area, conv_annual, rot, soil):
        """Reduced tillage: shallow soils (< 20cm depth)"""
        print(f"\nCalculating reduced tillage...")
        
        # Start with flat areas (<15cm)
        pre_reduced = spam_ta.copy()
        for i in range(42):
            pre_reduced[i][np.isnan(flat_area)] = 0
            pre_reduced[i][~np.isnan(flat_area)] = spam_ta[i][~np.isnan(flat_area)]
        
        # Add areas from conventional (15-20cm depth)
        for i in range(42):
            mask = (soil >= 15) & (soil < 20)
            pre_reduced[i][mask] = conv_annual[i][mask]
        
        # Add areas from rotational (15-20cm depth)
        peren_indices = [5,7,8,9,10,11,14,15,17,18,19,20,21]
        for i, idx in enumerate(peren_indices):
            mask = (soil >= 15) & (soil < 20)
            if i < len(rot):
                pre_reduced[idx][mask] = rot[i][mask]
        
        total = np.nansum(pre_reduced)
        print(f"  ✓ {total/1e6:.1f} Mha")
        
        return pre_reduced
    
    def calculate_conventional_annual(self, spam_ta, perms_sum, reduced, trad_annual, ca_down, soil):
        """Conventional annual tillage: remaining annual cropland"""
        print(f"\nCalculating conventional annual tillage...")
        
        # Expand arrays to 42 crops
        r = np.full((self.ny, self.nx), np.nan)
        
        # Expand perennials
        peren_indices = [5,7,8,9,10,11,14,15,17,18,19,20,21]
        perms_ext = np.full((42, self.ny, self.nx), 0.0)
        for i, idx in enumerate(peren_indices):
            perms_ext[idx] = perms_sum[i] if i < len(perms_sum) else 0
        
        # Expand traditional annual (29 crops)
        annual_indices = [0,1,2,3,4,6,12,13,16] + list(range(23,42)) + [22]
        trad_ext = np.full((42, self.ny, self.nx), 0.0)
        for i, idx in enumerate(annual_indices):
            if i < len(trad_annual):
                trad_ext[idx] = trad_annual[i]
        
        # Expand CA (22 grain crops)
        grain_indices = [0,2,3,4,6,12,16,22] + list(range(28,42))  # Adjusted
        ca_ext = np.full((42, self.ny, self.nx), 0.0)
        for i, idx in enumerate(grain_indices):
            if i < len(ca_down):
                ca_ext[idx] = ca_down[i]
        
        # Subtract from total
        conv = spam_ta - perms_ext - reduced - trad_ext - ca_ext
        conv[conv < 0] = 0
        
        # Remove 15-20cm depth areas (go to reduced)
        for i in range(42):
            conv[i][(soil >= 15) & (soil < 20)] = np.nan
        
        total = np.nansum(conv)
        print(f"  ✓ {total/1e6:.1f} Mha")
        
        return conv
    
    # ========================================================================
    # 6. HELPER FUNCTIONS
    # ========================================================================
    
    def _resample(self, data, transform, method='bilinear'):
        """Resample to target grid"""
        dst = np.zeros((self.ny, self.nx), dtype=np.float32)
        dst_transform = from_bounds(self.extent['xmin'], self.extent['ymin'],
                                    self.extent['xmax'], self.extent['ymax'],
                                    self.nx, self.ny)
        
        method_map = {
            'bilinear': Resampling.bilinear,
            'nearest': Resampling.nearest,
            'average': Resampling.average
        }
        
        reproject(source=data, destination=dst, src_transform=transform,
                 src_crs='EPSG:4326', dst_transform=dst_transform,
                 dst_crs='EPSG:4326', resampling=method_map.get(method, Resampling.bilinear))
        
        dst[dst == 0] = np.nan
        dst[dst < -9998] = np.nan
        
        return dst
    
    def _interpolate_fields(self, fields):
        """Interpolate missing field sizes using moving window"""
        
        def fill_na(window):
            center_idx = len(window) // 2
            if np.isnan(window[center_idx]):
                valid = window[window != 0]
                valid = valid[~np.isnan(valid)]
                if len(valid) > 0:
                    return np.round(np.mean(valid))
            return window[center_idx]
        
        # Multiple passes with increasing window size
        for window_size in [51, 101, 191]:
            fields = generic_filter(fields, fill_na, size=window_size, mode='constant', cval=np.nan)
        
        # Fill any remaining with neutral value
        fields[np.isnan(fields)] = 20
        
        return fields
    
    def _interpolate_missing(self, data, mask):
        """Interpolate missing values where mask is not NaN"""
        
        def fill_na(window):
            center_idx = len(window) // 2
            if np.isnan(window[center_idx]):
                valid = window[window != 999]
                valid = valid[~np.isnan(valid)]
                if len(valid) > 0:
                    return np.mean(valid)
            return window[center_idx]
        
        data_copy = data.copy()
        data_copy[np.isnan(data_copy)] = 999
        data_interp = generic_filter(data_copy, fill_na, size=51, mode='constant', cval=999)
        data_interp[data_interp == 999] = np.nan
        
        return data_interp
    
    def _interpolate_fields_optimized(self, fields):
        """
        Memory-efficient interpolation using uniform_filter
        Much faster than generic_filter
        """
        from scipy.ndimage import uniform_filter
        
        # Step 1: Fast uniform filter for most gaps
        mask = ~np.isnan(fields)
        
        if mask.sum() == 0:
            print(f"  ⚠️ No valid field data, using default value 20")
            return np.full_like(fields, 20.0)
        
        # Fill NaNs temporarily for uniform filter
        fields_filled = fields.copy()
        fields_filled[~mask] = 0
        
        # Apply uniform filter with progressively larger windows
        for window_size in [11, 21, 41]:
            # Weighted average (only counts valid neighbors)
            smoothed = uniform_filter(fields_filled, size=window_size, mode='constant')
            counts = uniform_filter(mask.astype(float), size=window_size, mode='constant')
            
            # Avoid division by zero
            counts[counts < 0.01] = np.nan
            weighted = smoothed / counts
            
            # Fill gaps
            needs_fill = np.isnan(fields_filled) | (fields_filled == 0)
            fields_filled[needs_fill] = weighted[needs_fill]
        
        # Step 2: Fill any remaining with nearest neighbor
        still_missing = np.isnan(fields_filled) | (fields_filled == 0)
        
        if still_missing.sum() > 0:
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(still_missing, return_distances=False, return_indices=True)
            fields_filled[still_missing] = fields_filled[tuple(indices)][still_missing]
        
        # Step 3: Fill any remaining with neutral value
        fields_filled[np.isnan(fields_filled) | (fields_filled == 0)] = 20

        valid_categories = np.array([10, 20, 30, 40, 50])
        fields_rounded = np.zeros_like(fields)
        for cat in valid_categories:
            distances = np.abs(fields - cat)
            mask = np.all([np.abs(fields - other) >= distances for other in valid_categories if other != cat], axis=0)
            fields_rounded[mask] = cat
        fields = fields_rounded
        
        return fields_filled
    
    # ========================================================================
    # 7. MAIN EXECUTION
    # ========================================================================
    
    def run_complete_pipeline(self):
        """Execute complete tillage dataset generation"""
        
        print(f"\n{'='*80}")
        print(f"STEP 1: LOAD INPUT DATA")
        print(f"{'='*80}")
        
        spam_ta, spam_tr = self.load_spam_data()
        soil, flat_area = self.load_soilgrids()
        fields = self.load_fields()
        erosion = self.load_erosion()
        aridity = self.load_aridity()
        alloc = self.load_country_allocation()
        high_raster = self.load_income_levels()
        ca_stats = self.load_ca_statistics()
        
        print(f"\n{'='*80}")
        print(f"STEP 2: CALCULATE CROP MIX")
        print(f"{'='*80}")
        
        crop_mix = self.calculate_crop_mix(spam_ta, spam_tr, flat_area, fields, high_raster)
        
        # Save potential CA area (grains_stack from crop_mix calculation)
        pot_ca_area = self.grains_stack  # 22 crops
        
        print(f"\n{'='*80}")
        print(f"STEP 3: BUILD LOGIT MODEL")
        print(f"{'='*80}")
        
        logit = self.build_logit_model(fields, erosion, aridity, crop_mix)
        
        print(f"\n{'='*80}")
        print(f"STEP 4: DOWNSCALE CONSERVATION AGRICULTURE")
        print(f"{'='*80}")
        
        ca_downscaled = self.downscale_ca(logit, ca_stats, alloc, pot_ca_area)
        
        print(f"\n{'='*80}")
        print(f"STEP 5: CALCULATE TILLAGE SYSTEMS")
        print(f"{'='*80}")
        
        trad_annual = self.calculate_traditional_annual(spam_ta, flat_area, fields, high_raster)
        trad_rot = self.calculate_traditional_rotational(spam_ta, flat_area, fields, high_raster)
        rot = self.calculate_rotational(spam_ta, flat_area, fields, high_raster, soil)
        
        # Calculate conventional (needs perms_sum)
        perms_sum = trad_rot + rot  # Combine traditional and modern rotational
        
        conv = self.calculate_conventional_annual(spam_ta, perms_sum, 
                                                   np.zeros_like(spam_ta),  # Placeholder for reduced
                                                   trad_annual, ca_downscaled, soil)
        
        reduced = self.calculate_reduced(spam_ta, flat_area, conv, rot, soil)
        
        # Recalculate conventional with actual reduced
        conv = self.calculate_conventional_annual(spam_ta, perms_sum, reduced,
                                                   trad_annual, ca_downscaled, soil)
        
        print(f"\n{'='*80}")
        print(f"STEP 6: SAVE OUTPUTS")
        print(f"{'='*80}")
        
        # Save all outputs
        outputs = {
            'conservation_agriculture': ca_downscaled,
            'traditional_annual': trad_annual,
            'traditional_rotational': trad_rot,
            'rotational': rot,
            'reduced': reduced,
            'conventional_annual': conv
        }
        
        for name, data in outputs.items():
            np.save(self.path_output / f"{name}.npy", data)
            total = np.nansum(data)
            print(f"  ✓ {name}: {total/1e6:.1f} Mha")
        
        # Summary table
        summary = pd.DataFrame({
            'Tillage_System': list(outputs.keys()),
            'Area_Mha': [np.nansum(data)/1e6 for data in outputs.values()]
        })
        summary.to_csv(self.path_output / "tillage_summary.csv", index=False)
        
        print(f"\n  ✓ Saved summary to tillage_summary.csv")
        
        print(f"\n{'='*80}")
        print(f"COMPLETE!")
        print(f"{'='*80}")
        print(f"\nTotal cropland: {np.nansum(spam_ta)/1e6:.1f} Mha")
        print(f"Accounted for: {summary['Area_Mha'].sum():.1f} Mha")
        print(f"Difference: {np.nansum(spam_ta)/1e6 - summary['Area_Mha'].sum():.1f} Mha")

        print(f"\n{'='*80}")
        print(f"DEBUG: DATA LOADED")
        print(f"{'='*80}")
        print(f"SPAM total: {np.nansum(spam_ta)/1e6:.1f} Mha")
        print(f"Flat pixels (<15cm): {np.sum(~np.isnan(flat_area)):,}")
        print(f"Small fields (<20): {np.sum(fields < 20):,}")
        print(f"Large fields (>=20): {np.sum(fields >= 20):,}")
        print(f"High income pixels: {np.sum(~np.isnan(high_raster)):,}")
        print(f"Low income pixels: {np.sum(np.isnan(high_raster)):,}")
        
        # Test: How much cropland is on small fields in low income countries?
        test = spam_ta.copy()
        for i in range(42):
            test[i][fields >= 20] = np.nan  # Remove large fields
            test[i][~np.isnan(high_raster)] = np.nan  # Remove high income
        print(f"\nCropland on SMALL fields + LOW income: {np.nansum(test)/1e6:.1f} Mha")
        
        # Test: How much on large fields?
        test2 = spam_ta.copy()
        for i in range(42):
            test2[i][fields < 20] = np.nan
        print(f"Cropland on LARGE fields: {np.nansum(test2)/1e6:.1f} Mha")
        # END DEBUG BLOCK
        
        return outputs, summary


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Initialize
    pipeline = GlobalTillageDataset(
        path_input="D:/Users/Wilson/Downloads/Sentinel",
        path_output="D:/Users/Wilson/Downloads/Sentinel/output",
        sample_calc=False
    )
    
    # Run complete pipeline
    outputs, summary = pipeline.run_complete_pipeline()
    
    print(f"\n{'='*80}")
    print(f"OUTPUTS SAVED TO: {pipeline.path_output}")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  - conservation_agriculture.npy (22 crops)")
    print(f"  - traditional_annual.npy (29 crops)")
    print(f"  - traditional_rotational.npy (13 crops)")
    print(f"  - rotational.npy (13 crops)")
    print(f"  - reduced.npy (42 crops)")
    print(f"  - conventional_annual.npy (42 crops)")
    print(f"  - tillage_summary.csv")
    
    print(f"\n{'='*80}")
    print(f"NEXT STEPS")
    print(f"{'='*80}")
    print(f"\n1. Validate outputs:")
    print(f"   - Check total areas match SPAM2020 cropland")
    print(f"   - Compare CA area to FAOSTAT statistics")
    print(f"   - Visualize spatial patterns")
    
    print(f"\n2. Generate categorical map:")
    print(f"   - Convert continuous areas to dominant tillage system per cell")
    print(f"   - Categories: 1=conventional, 2=traditional_annual, etc.")
    
    print(f"\n3. Use for CNN training:")
    print(f"   - Load .npy files as training labels")
    print(f"   - Pair with Sentinel-2 imagery")
    print(f"   - Train multi-class or multi-label classifier")