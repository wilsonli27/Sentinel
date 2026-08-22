"""
Create Country Allocation Raster from GADM or Natural Earth
Run this FIRST to generate the allocation raster needed for full tillage classification
"""

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from pathlib import Path
import pandas as pd

def create_allocation_raster(input_path, output_path, method='gadm'):
    """
    Create country allocation raster at 5 arcmin resolution
    
    Parameters:
    -----------
    input_path : str
        Path to Sentinel folder
    output_path : str  
        Path to output folder
    method : str
        'gadm' - Use GADM gpkg file (download first)
        'naturalearth' - Use Natural Earth shapefile (download first)
        'online' - Download Natural Earth automatically (requires internet)
    """
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print("CREATING COUNTRY ALLOCATION RASTER")
    print(f"{'='*70}")
    
    # Define grid (same as tillage dataset)
    extent = {'xmin': -180, 'xmax': 180, 'ymin': -56, 'ymax': 84}
    resolution = 1/12  # 5 arcmin
    
    nx = int((extent['xmax'] - extent['xmin']) / resolution)
    ny = int((extent['ymax'] - extent['ymin']) / resolution)
    
    print(f"Grid: {nx} x {ny} = {nx*ny:,} cells")
    print(f"Resolution: {resolution}° (5 arcmin)")
    
    # Load country boundaries
    print(f"\nLoading country boundaries (method: {method})...")
    
    if method == 'gadm':
        # GADM gpkg file
        gadm_file = input_path / "gadm_410.gpkg"
        if not gadm_file.exists():
            raise FileNotFoundError(
                f"GADM file not found: {gadm_file}\n"
                f"Download from: https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-gpkg.zip"
            )
        print(f"  Reading GADM from: {gadm_file.name}")
        gdf = gpd.read_file(gadm_file, layer='ADM_0')
        
    elif method == 'naturalearth':
        # Natural Earth shapefile
        ne_file = input_path / "ne_10m_admin_0_countries" / "ne_10m_admin_0_countries.shp"
        if not ne_file.exists():
            raise FileNotFoundError(
                f"Natural Earth file not found: {ne_file}\n"
                f"Download from: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/"
            )
        print(f"  Reading Natural Earth from: {ne_file.name}")
        gdf = gpd.read_file(ne_file)
        
    elif method == 'online':
        # Download Natural Earth automatically
        print(f"  Downloading Natural Earth data...")
        url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip"
        gdf = gpd.read_file(url)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"  ✓ Loaded {len(gdf)} countries")
    
    # Get ISO3 codes and create numeric mapping
    if 'ISO_A3' in gdf.columns:
        iso_col = 'ISO_A3'
    elif 'iso_a3' in gdf.columns:
        iso_col = 'iso_a3'
    elif 'GID_0' in gdf.columns:
        iso_col = 'GID_0'
    else:
        raise ValueError("Cannot find ISO3 code column")
    
    # Clean and sort
    gdf = gdf[gdf[iso_col].notna()].copy()
    gdf = gdf[gdf[iso_col] != '-99']  # Remove invalid codes
    gdf[iso_col] = gdf[iso_col].str.upper()
    
    # Create numeric country codes (sorted alphabetically by ISO3)
    unique_iso3 = sorted(gdf[iso_col].unique())
    iso3_to_numeric = {iso3: i+1 for i, iso3 in enumerate(unique_iso3)}
    gdf['country_code'] = gdf[iso_col].map(iso3_to_numeric)
    
    print(f"  Mapped {len(iso3_to_numeric)} unique country codes")
    print(f"  Range: 1 to {len(iso3_to_numeric)}")
    
    # Create transform
    transform = from_bounds(
        extent['xmin'], extent['ymin'],
        extent['xmax'], extent['ymax'],
        nx, ny
    )
    
    # Rasterize countries
    print(f"\nRasterizing countries to 5 arcmin grid...")
    print(f"  This may take 2-5 minutes...")
    
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['country_code']))
    
    alloc_raster = rasterize(
        shapes=shapes,
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype=np.uint16,
        all_touched=False  # Only cells with center in country
    )
    
    # Set ocean to NaN
    alloc_raster = alloc_raster.astype(np.float32)
    alloc_raster[alloc_raster == 0] = np.nan
    
    print(f"  ✓ Rasterized")
    print(f"  Valid cells: {np.sum(~np.isnan(alloc_raster)):,} ({np.sum(~np.isnan(alloc_raster))/alloc_raster.size*100:.1f}%)")
    
    # Save as GeoTIFF
    output_file = output_path / "country_allocation_5arcmin.tif"
    
    with rasterio.open(
        output_file, 'w',
        driver='GTiff',
        height=ny, width=nx,
        count=1, dtype=np.float32,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(alloc_raster, 1)
    
    print(f"\n✓ SAVED: {output_file}")
    print(f"  File size: {output_file.stat().st_size / (1024**2):.1f} MB")
    
    # Save ISO3 mapping
    mapping_df = pd.DataFrame({
        'country_code': range(1, len(unique_iso3) + 1),
        'iso3': unique_iso3
    })
    
    mapping_file = output_path / "country_code_mapping.csv"
    mapping_df.to_csv(mapping_file, index=False)
    print(f"  Mapping: {mapping_file}")
    
    # Load World Bank income data and add to mapping
    try:
        income_file = input_path / "OGHIST_2025_10_07.xlsx"
        if income_file.exists():
            print(f"\n  Adding income levels from OGHIST...")
            
            df_raw = pd.read_excel(income_file, sheet_name='Country Analytical History', header=None)
            fy_row = df_raw.iloc[5]
            df_data = df_raw.iloc[11:].copy()
            df_data.columns = fy_row.tolist()
            df_data = df_data.rename(columns={df_data.columns[0]: 'Code', df_data.columns[1]: 'Country'})
            
            year_cols = [c for c in df_data.columns if isinstance(c, (int, float)) and 1987 <= c <= 2024]
            year_column = 2010 if 2010 in year_cols else min(year_cols, key=lambda x: abs(x-2010))
            
            income_map = {'L': 1, 'LIC': 1, 'LM': 2, 'LMC': 2, 'UM': 3, 'UMC': 3, 'H': 4, 'HIC': 4}
            df_data['income_code'] = df_data[year_column].map(income_map)
            
            df_valid = df_data[
                (df_data['Code'].notna()) & 
                (df_data['Code'].astype(str).str.len() == 3) &
                (df_data['income_code'].notna())
            ][['Code', 'income_code']].copy()
            
            # Merge with mapping
            mapping_df = mapping_df.merge(
                df_valid.rename(columns={'Code': 'iso3'}),
                on='iso3',
                how='left'
            )
            
            mapping_with_income = output_path / "country_code_mapping_with_income.csv"
            mapping_df.to_csv(mapping_with_income, index=False)
            print(f"  ✓ Added income levels for {df_valid.shape[0]} countries")
            print(f"  Saved: {mapping_with_income}")
            
    except Exception as e:
        print(f"  ⚠️  Could not add income levels: {e}")
    
    print(f"\n{'='*70}")
    print("ALLOCATION RASTER READY!")
    print(f"{'='*70}")
    print(f"\nYou now have:")
    print(f"  1. {output_file.name} - The allocation raster")
    print(f"  2. {mapping_file.name} - ISO3 to numeric code mapping")
    print(f"\nNext step: Run the full tillage classification code")
    
    return alloc_raster, mapping_df


if __name__ == "__main__":
    """
    USAGE:
    
    METHOD 1 - Download GADM first (most accurate):
    1. Download: https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-gpkg.zip
    2. Extract gadm_410.gpkg to Sentinel folder
    3. Run: method='gadm'
    
    METHOD 2 - Download Natural Earth first (smaller):
    1. Download: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
    2. Extract to Sentinel/ne_10m_admin_0_countries/
    3. Run: method='naturalearth'
    
    METHOD 3 - Automatic download (requires internet):
    Run: method='online'
    """
    
    # CHOOSE YOUR METHOD:
    alloc, mapping = create_allocation_raster(
        input_path="D:/Users/Wilson/Downloads/Sentinel",
        output_path="D:/Users/Wilson/Downloads/Sentinel/output",
        method='online'  # or 'gadm' or 'naturalearth'
    )
    
    print("\n✓ COMPLETE! Now you can run the full tillage classification.")