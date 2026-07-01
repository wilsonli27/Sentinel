"""
Prepare Country Allocation Data (Vector-based, no rasterization)
This is much faster and less memory-intensive than rasterization
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import fiona

def prepare_country_allocation():
    """Prepare country allocation data without rasterization"""
    
    input_path = Path("D:/Users/Wilson/Downloads/Sentinel")
    output_path = Path("D:/Users/Wilson/Downloads/Sentinel/output")
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("PREPARING COUNTRY ALLOCATION DATA (Vector-based)")
    print("="*70)
    
    # Load GADM
    gadm_file = input_path / "gadm_410-gpkg" / "gadm_410.gpkg"
    print(f"Loading GADM from: {gadm_file}")
    
    # Find correct layer
    try:
        layers = fiona.listlayers(str(gadm_file))
        print(f"\nAvailable layers:")
        for layer in layers:
            print(f"  - {layer}")
        
        level0_layer = None
        for layer in layers:
            if 'ADM_0' in layer.upper() or layer.upper() == 'LEVEL0':
                level0_layer = layer
                break
        
        if level0_layer is None:
            level0_layer = layers[0] if layers else 'ADM_ADM_0'
            print(f"\n⚠ Using first layer: {level0_layer}")
        else:
            print(f"\n✓ Using layer: {level0_layer}")
        
        gdf = gpd.read_file(gadm_file, layer=level0_layer)
        
    except Exception as e:
        print(f"\nTrying without layer specification...")
        gdf = gpd.read_file(gadm_file)
    
    print(f"✓ Loaded {len(gdf)} countries")
    print(f"  Columns: {list(gdf.columns)}")
    
    # Find ISO column
    iso_column = None
    for col in ['GID_0', 'COUNTRY', 'ISO', 'ISO3', 'SOVEREIGN']:
        if col in gdf.columns:
            iso_column = col
            break
    
    if iso_column is None:
        print(f"\n❌ Could not find ISO3 code column")
        raise ValueError("Cannot find country code column")
    
    print(f"✓ Using column '{iso_column}' for country codes")
    
    # Clean and prepare
    gdf = gdf[gdf[iso_column].notna()].copy()
    gdf['iso3'] = gdf[iso_column].str.upper()
    
    # Add numeric codes
    unique_iso3 = sorted(gdf['iso3'].unique())
    iso3_to_numeric = {iso3: i+1 for i, iso3 in enumerate(unique_iso3)}
    gdf['country_code'] = gdf['iso3'].map(iso3_to_numeric)
    
    print(f"✓ Mapped {len(unique_iso3)} countries")
    
    # Simplify geometry for faster processing (optional)
    print("\nSimplifying geometries for faster processing...")
    gdf['geometry'] = gdf['geometry'].simplify(0.01, preserve_topology=True)
    
    # Keep only needed columns
    gdf = gdf[['iso3', 'country_code', 'geometry']].copy()
    
    # Add income levels
    print("\nAdding income level data...")
    income_file = input_path / "OGHIST_2025_10_07.xlsx"
    
    if income_file.exists():
        df_raw = pd.read_excel(income_file, sheet_name='Country Analytical History', header=None)
        df_data = df_raw.iloc[11:].copy()
        df_data.columns = df_raw.iloc[5].tolist()
        df_data = df_data.rename(columns={df_data.columns[0]: 'Code', df_data.columns[1]: 'Country'})
        
        year_cols = [c for c in df_data.columns if isinstance(c, (int, float)) and 1987 <= c <= 2024]
        year_column = 2010 if 2010 in year_cols else min(year_cols, key=lambda x: abs(x-2010))
        
        income_map = {'L': 1, 'LIC': 1, 'LM': 2, 'LMC': 2, 'UM': 3, 'UMC': 3, 'H': 4, 'HIC': 4}
        df_data['income_code'] = df_data[year_column].map(income_map)
        df_valid = df_data[(df_data['Code'].notna()) & 
                           (df_data['Code'].astype(str).str.len() == 3) &
                           (df_data['income_code'].notna())][['Code', 'income_code']].copy()
        df_valid = df_valid.rename(columns={'Code': 'iso3'})
        
        gdf = gdf.merge(df_valid, on='iso3', how='left')
        print(f"✓ Added income levels for {gdf['income_code'].notna().sum()} countries")
    else:
        print(f"⚠ Income file not found, skipping income levels")
        gdf['income_code'] = None
    
    # Save vector file (much faster to work with)
    output_vector = output_path / "country_allocation.gpkg"
    gdf.to_file(output_vector, driver='GPKG', layer='countries')
    print(f"\n✓ Saved vector file: {output_vector}")
    
    # Save mapping table
    mapping_df = gdf[['country_code', 'iso3', 'income_code']].drop_duplicates().sort_values('country_code')
    mapping_file = output_path / "country_code_mapping.csv"
    mapping_df.to_csv(mapping_file, index=False)
    print(f"✓ Saved mapping: {mapping_file}")
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print(f"  Countries: {len(unique_iso3)}")
    print(f"  Output: {output_vector}")
    print("\n  Use spatial join with your tillage data instead of raster overlay")
    print("="*70)
    
    return gdf

def spatial_join_example():
    """Example of how to use the vector allocation with point data"""
    print("\n" + "="*70)
    print("EXAMPLE: Spatial join with tillage points")
    print("="*70)
    print("""
# Load your tillage points
import geopandas as gpd
tillage_gdf = gpd.read_file('your_tillage_points.shp')

# Load country allocation
countries = gpd.read_file('output/country_allocation.gpkg', layer='countries')

# Spatial join (much faster than raster operations)
tillage_with_countries = gpd.sjoin(tillage_gdf, countries, how='left', predicate='within')

# Now you have iso3, country_code, and income_code for each tillage point
print(tillage_with_countries[['iso3', 'country_code', 'income_code']].head())
    """)

if __name__ == "__main__":
    gdf = prepare_country_allocation()
    print("\n")
    spatial_join_example()