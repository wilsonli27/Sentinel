"""
Global Gridded Tillage Dataset Generation - MEMORY-SAFE VERSION
Complete ready-to-paste code with all fixes applied
"""

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

def get_memory_usage():
    """Get current process memory in GB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    except:
        return 0

class TillageDatasetGenerator:
    """Memory-safe tillage dataset generator"""
    
    def __init__(self, path_input, path_output, sample_calc=False):
        self.path_input = Path(path_input)
        self.path_output = Path(path_output)
        self.path_output.mkdir(exist_ok=True, parents=True)
        self.sample_calc = sample_calc
        
        if sample_calc:
            self.extent = {'xmin': -180, 'xmax': 180, 'ymin': -45, 'ymax': 90}
            self.resolution = 45
        else:
            self.extent = {'xmin': -180, 'xmax': 180, 'ymin': -56, 'ymax': 84}
            self.resolution = 1/12
        
        self.nx = int((self.extent['xmax'] - self.extent['xmin']) / self.resolution)
        self.ny = int((self.extent['ymax'] - self.extent['ymin']) / self.resolution)
        
        self.lons = np.linspace(
            self.extent['xmin'] + self.resolution/2,
            self.extent['xmax'] - self.resolution/2, self.nx)
        self.lats = np.linspace(
            self.extent['ymax'] - self.resolution/2,
            self.extent['ymin'] + self.resolution/2, self.ny)
        
        self.crop_names = [
            'WHEA', 'RICE', 'MAIZ', 'BARL', 'REST', 'OOIL', 'TOBA', 'TEAS',
            'COCO', 'RCOF', 'ACOF', 'OFIB', 'COTT', 'SUGB', 'SUGC', 'OILP',
            'VEGE', 'TEMF', 'TROF', 'PLNT', 'BANA', 'CNUT', 'GROU', 'OTRS',
            'CASS', 'YAMS', 'SWPO', 'POTA', 'SESA', 'RAPE', 'SUNF', 'SOYB',
            'OPUL', 'LENT', 'PIGE', 'COWP', 'CHIC', 'BEAN', 'OCER', 'SORG',
            'SMIL', 'PMIL']
        
        print(f"Memory-Safe Tillage Generator")
        print(f"Grid: {self.nx} x {self.ny}, Resolution: {self.resolution}°")
        print(f"Memory: {get_memory_usage():.2f} GB")

    def load_spam_data(self, version='2020'):
        """Load SPAM - reads at target resolution"""
        print(f"\nLoading SPAM{version}...")
        
        if self.sample_calc:
            spam_ta = np.random.randint(1, 25, size=(42, self.ny, self.nx)).astype(np.float32)
            return spam_ta, spam_ta * 0.67
        
        spam_folder = self.path_input / "spam2020v2r0_global_physical_area.geotiff"
        if not spam_folder.exists():
            raise FileNotFoundError(f"Missing: {spam_folder}")
        
        nested = spam_folder / "spam2020v2r0_global_physical_area"
        if nested.exists():
            spam_folder = nested
        
        spam_ta = np.zeros((42, self.ny, self.nx), dtype=np.float32)
        spam_tr = np.zeros((42, self.ny, self.nx), dtype=np.float32)
        
        for i, crop in enumerate(self.crop_names):
            ta_file = spam_folder / f"spam2020_V2r0_global_A_{crop}_A.tif"
            tr_file = spam_folder / f"spam2020_V2r0_global_A_{crop}_R.tif"
            
            if not ta_file.exists():
                continue
            
            try:
                with rasterio.open(ta_file) as src:
                    spam_ta[i] = src.read(1, out_shape=(self.ny, self.nx),
                                         resampling=Resampling.average).astype(np.float32)
                if tr_file.exists():
                    with rasterio.open(tr_file) as src:
                        spam_tr[i] = src.read(1, out_shape=(self.ny, self.nx),
                                             resampling=Resampling.average).astype(np.float32)
                if (i+1) % 10 == 0:
                    print(f"  {i+1}/42 crops...")
                    gc.collect()
            except Exception as e:
                print(f"  Error {crop}: {e}")
        
        print(f"  ✓ Loaded {np.sum(~np.all(spam_ta==0, axis=(1,2)))}/42")
        gc.collect()
        return spam_ta, spam_tr

    def load_soilgrids_depth(self, version='250m', use_r_horizon=False):
        """Load SoilGrids"""
        print("\nLoading SoilGrids...")
        
        if self.sample_calc:
            data = np.arange(1, 25).reshape(self.ny, self.nx).astype(np.float32)
            flat = data.copy()
            flat[data >= 15] = np.nan
            return data, flat
        
        files = ["BDTICM_M_250m_ll", "BDTICM_M_250m_ll.tif"]
        soil_file = None
        for f in files:
            if (self.path_input / f).exists():
                soil_file = self.path_input / f
                break
        
        if not soil_file:
            raise FileNotFoundError("SoilGrids not found")
        
        with rasterio.open(soil_file) as src:
            print(f"  Reading at target resolution...")
            data = src.read(1, out_shape=(self.ny, self.nx),
                           resampling=Resampling.bilinear).astype(np.float32)
        
        data[data < -9998] = np.nan
        flat = data.copy()
        flat[data >= 15] = np.nan
        print(f"  Range: {np.nanmin(data):.1f}-{np.nanmax(data):.1f} cm")
        gc.collect()
        return data, flat

    def load_field_size_data(self):
        """Load field sizes"""
        print("\nLoading field sizes...")
        
        if self.sample_calc:
            return np.arange(10, 34).reshape(self.ny, self.nx).astype(np.float32)
        
        field_file = self.path_input / "lesiv_2018_field_sizes" / "Global Field Sizes" / "dominant_field_size_categories.tif"
        
        if not field_file.exists():
            print("  ⚠️  Not found, using simplified method")
            return None
        
        try:
            with rasterio.open(field_file) as src:
                data = src.read(1, out_shape=(self.ny, self.nx),
                               resampling=Resampling.mode).astype(np.float32)
            print(f"  ✓ Range: {np.nanmin(data):.0f}-{np.nanmax(data):.0f}")
            gc.collect()
            return data
        except:
            print("  ⚠️  Error loading, using simplified method")
            return None

    def load_erosion_data(self, version='glosem'):
        """Load erosion"""
        print("\nLoading erosion...")
        
        if self.sample_calc:
            return np.arange(1, 25).reshape(self.ny, self.nx).astype(np.float32)
        
        folder = self.path_input / "Data2_SOIL_DISPLACEMENT_ESTIMATE_2019_1"
        nested = folder / "Data2_SOIL_DISPLACEMENT_ESTIMATE_2019_1"
        if nested.exists():
            folder = nested
        
        erosion_file = folder / "SOIL_DISPLACEMENT_ESTIMATE_2019.tif"
        if not erosion_file.exists():
            erosion_file = folder / "SOIL_DISPLACEMENT_ESTIMATE_2019.tif.ovr"
        
        with rasterio.open(erosion_file) as src:
            data = src.read(1, out_shape=(self.ny, self.nx),
                           resampling=Resampling.average).astype(np.float32)
        
        data[data < 0] = np.nan
        data[data > 10000] = np.nan
        print(f"  Range: {np.nanmin(data):.1f}-{np.nanmax(data):.1f}")
        gc.collect()
        return data

    def load_aridity_index(self):
        """Load aridity"""
        print("\nLoading aridity...")
        
        if self.sample_calc:
            return (np.arange(1, 24).reshape(self.ny, self.nx) / 100.0).astype(np.float32)
        
        aridity_file = self.path_input / "Global-AI_ET0_annual_v3" / "Global-AI_ET0_v3_annual" / "ai_v3_yr.tif"
        
        with rasterio.open(aridity_file) as src:
            data = src.read(1, out_shape=(self.ny, self.nx),
                           resampling=Resampling.bilinear).astype(np.float32)
        
        if np.nanmax(data) > 100:
            data = data / 10000.0
        
        data[data < 0] = np.nan
        data[data > 10] = np.nan
        print(f"  Range: {np.nanmin(data):.3f}-{np.nanmax(data):.3f}")
        gc.collect()
        return data

    def load_income_levels(self):
        """Load income levels"""
        print("\nLoading income levels...")
        
        if self.sample_calc:
            return {}, []
        
        income_file = self.path_input / "OGHIST_2025_10_07.xlsx"
        df_raw = pd.read_excel(income_file, sheet_name='Country Analytical History', header=None)
        
        fy_row = df_raw.iloc[5]
        df_data = df_raw.iloc[11:].copy()
        df_data.columns = fy_row.tolist()
        df_data = df_data.rename(columns={df_data.columns[0]: 'Code', df_data.columns[1]: 'Country'})
        
        year_cols = [c for c in df_data.columns if isinstance(c, (int, float)) and 1987 <= c <= 2024]
        year_column = 2010 if 2010 in year_cols else min(year_cols, key=lambda x: abs(x-2010))
        
        income_map = {'L': 1, 'LIC': 1, 'LM': 2, 'LMC': 2, 'UM': 3, 'UMC': 3, 'H': 4, 'HIC': 4}
        df_data['income_code'] = df_data[year_column].map(income_map)
        
        df_valid = df_data[(df_data['Code'].notna()) & 
                           (df_data['Code'].astype(str).str.len() == 3) &
                           (df_data['income_code'].notna())]
        
        print(f"  ✓ {len(df_valid)} countries")
        return dict(zip(df_valid['Code'], df_valid['income_code'])), df_valid[df_valid['income_code'] >= 3]['Code'].values

    def load_notill_statistics(self):
        """Load no-till stats"""
        print("\nLoading no-till stats...")
        
        notill_file = self.path_input / "FAOSTAT_data_en_1-9-2026.csv"
        if not notill_file.exists():
            print("  ⚠️  Not found")
            return pd.DataFrame(columns=['country', 'notill_area_ha', 'year'])
        
        df = pd.read_csv(notill_file)
        if 'Area' in df.columns and 'Value' in df.columns:
            if 'Unit' in df.columns and '1000 ha' in str(df['Unit'].iloc[0]):
                df['Value'] *= 1000
            data = pd.DataFrame({'country': df['Area'], 'notill_area_ha': df['Value']})
            data = data[data['notill_area_ha'] > 0]
            print(f"  ✓ {len(data)} countries, {data['notill_area_ha'].sum()/1e6:.1f} Mha")
            return data
        return pd.DataFrame(columns=['country', 'notill_area_ha'])

    def calculate_crop_mix(self, spam_tr, flat_area, fields, high_raster):
        """Calculate crop mix"""
        print("\nCalculating crop mix...")
        
        spam_tr_flat = spam_tr.copy()
        spam_tr_flat[:, np.isnan(flat_area)] = np.nan
        
        annual_idx = [0,1,2,3,4,6,12,13,16] + list(range(23,42)) + [22]
        grain_idx = [0,2,3,4,5,6,8] + list(range(14,29))
        
        grains = spam_tr_flat[annual_idx][grain_idx]
        grains_sum = np.nansum(grains, axis=0)
        total = np.nansum(spam_tr, axis=0)
        total[total == 0] = np.nan
        
        crop_mix = grains_sum / total
        crop_mix[~np.isfinite(crop_mix)] = np.nan
        
        print(f"  Range: {np.nanmin(crop_mix):.3f}-{np.nanmax(crop_mix):.3f}")
        gc.collect()
        return crop_mix

    def build_logit_model(self, field_size, erosion, aridity, crop_mix):
        """Build logit model"""
        print("\nBuilding logit model...")
        
        linear = (0.25 * field_size + erosion/60 - 5 * aridity + 10 * crop_mix)
        intercept = -(20*0.25 + 12/60 - 5*0.65 + 10*0.5)
        prob = 1 / (1 + np.exp(-(linear + intercept)))
        prob[~np.isfinite(prob)] = np.nan
        
        print(f"  Prob range: {np.nanmin(prob):.3f}-{np.nanmax(prob):.3f}")
        gc.collect()
        return prob

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MEMORY-SAFE TILLAGE DATASET GENERATION")
    print("="*70)
    
    generator = TillageDatasetGenerator(
        path_input="D:/Users/Wilson/Downloads/Sentinel",
        path_output="D:/Users/Wilson/Downloads/Sentinel/output"
    )
    
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    spam_ta, spam_tr = generator.load_spam_data()
    print(f"💾 Memory: {get_memory_usage():.2f} GB")
    
    soilgrid, flat_area = generator.load_soilgrids_depth()
    print(f"💾 Memory: {get_memory_usage():.2f} GB")
    
    field_sizes = generator.load_field_size_data()
    print(f"💾 Memory: {get_memory_usage():.2f} GB")
    
    erosion = generator.load_erosion_data()
    print(f"💾 Memory: {get_memory_usage():.2f} GB")
    
    aridity = generator.load_aridity_index()
    print(f"💾 Memory: {get_memory_usage():.2f} GB")
    
    income_dict, high_income = generator.load_income_levels()
    notill_data = generator.load_notill_statistics()
    
    print("\n" + "="*70)
    print("CALCULATING")
    print("="*70)
    
    crop_mix = generator.calculate_crop_mix(spam_tr, flat_area, field_sizes, None)
    
    if field_sizes is None:
        field_sizes = np.full((generator.ny, generator.nx), 20.0, dtype=np.float32)
    
    notill_prob = generator.build_logit_model(field_sizes, erosion, aridity, crop_mix)
    
    output_file = generator.path_output / 'notill_probability.npy'
    np.save(output_file, notill_prob)
    
    print(f"\n✓ COMPLETE!")
    print(f"  Saved: {output_file}")
    print(f"  Final memory: {get_memory_usage():.2f} GB")
    print("="*70)