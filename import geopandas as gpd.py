"""
Fix Country Allocation GeoPackage
Dissolves 356k+ duplicate geometries into 263 unique countries
"""

import geopandas as gpd
from pathlib import Path

print("="*80)
print("FIXING COUNTRY ALLOCATION")
print("="*80)

# Paths
input_gpkg = Path("D:/Users/Wilson/Downloads/Sentinel/output/country_allocation.gpkg")
output_gpkg = Path("D:/Users/Wilson/Downloads/Sentinel/output/country_allocation_fixed.gpkg")

print(f"\nInput:  {input_gpkg}")
print(f"Output: {output_gpkg}")

# Load the problematic data
print(f"\nLoading original file...")
gdf = gpd.read_file(input_gpkg, layer='countries')
print(f"  ✓ Loaded: {len(gdf):,} rows")
print(f"  Unique countries: {gdf['country_code'].nunique()}")

# Show the problem
print(f"\nProblem: Each country is split into many small polygons")
example = gdf[gdf['iso3'] == 'AFG']
print(f"  Example: Afghanistan (AFG) has {len(example)} separate geometries")

# Dissolve by country_code
print(f"\nDissolving geometries by country_code...")
gdf_fixed = gdf.dissolve(by='country_code', aggfunc='first').reset_index()

print(f"  ✓ After dissolve: {len(gdf_fixed):,} rows")
print(f"  ✓ Unique countries: {gdf_fixed['country_code'].nunique()}")

# Verify
print(f"\nVerification:")
print(f"  Before: {len(gdf):,} geometries")
print(f"  After:  {len(gdf_fixed):,} geometries")
print(f"  Reduction: {len(gdf) - len(gdf_fixed):,} duplicates removed")

# Check Afghanistan example
example_fixed = gdf_fixed[gdf_fixed['iso3'] == 'AFG']
print(f"\n  Afghanistan now has {len(example_fixed)} geometry (was {len(example)})")

# Show sample
print(f"\n📊 First 10 countries after fix:")
print(gdf_fixed[['iso3', 'country_code', 'income_code']].head(10))

# Save the fixed version
print(f"\nSaving fixed file...")
gdf_fixed.to_file(output_gpkg, layer='countries', driver='GPKG')
print(f"  ✓ Saved to: {output_gpkg}")

# Also save mapping
mapping = gdf_fixed[['iso3', 'country_code', 'income_code']].copy()
mapping_csv = Path("D:/Users/Wilson/Downloads/Sentinel/output/country_code_mapping.csv")
mapping.to_csv(mapping_csv, index=False)
print(f"  ✓ Updated mapping: {mapping_csv}")

print(f"\n{'='*80}")
print(f"FIX COMPLETE!")
print(f"{'='*80}")
print(f"\nNext steps:")
print(f"1. Update your load_country_allocation() method to use:")
print(f"   'country_allocation_fixed.gpkg'")
print(f"2. Re-run your main pipeline")
print(f"\nCode change needed in GlobalTillageDataset.load_country_allocation():")
print(f"""
# OLD:
countries_gpkg = self.path_output / "country_allocation.gpkg"

# NEW:
countries_gpkg = self.path_output / "country_allocation_fixed.gpkg"
""")