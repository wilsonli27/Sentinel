"""
VISUALIZE ALL TILLAGE CROPS AND CATEGORIES
Creates individual maps for each crop showing their tillage categories
"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
path_calc = Path("sample_output/")
input_file = path_calc / "tillage.nc4"
output_path = path_calc / "crop_maps"
output_path.mkdir(exist_ok=True)

print("="*60)
print("VISUALIZING TILLAGE BY CROP")
print("="*60)

# Open NetCDF file
nc = Dataset(input_file, 'r')
print(f"\n✓ Loaded: {input_file}")

# Get coordinates
lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:]
print(f"Grid: {len(lon)} lon × {len(lat)} lat")

# Get all tillage variables
tillage_vars = [v for v in nc.variables.keys() if v not in ['lon', 'lat']]
print(f"Found {len(tillage_vars)} tillage variables\n")

# Tillage category labels (matching Mod_6_NetCDF_Export.py)
category_labels = {
    1: 'Conventional annual tillage',
    2: 'Traditional annual tillage', 
    3: 'Reduced tillage',
    4: 'Conservation Agriculture (CA)',
    5: 'Rotational tillage (perennials)',
    6: 'Traditional rotational tillage (perennials)',
    7: 'Scenario CA area'
}

# Crop-specific category ranges
crop_category_ranges = {
    # Tubers and rice (categories 1-3)
    'rice_till': (1, 3),
    'sugb_till': (1, 3),
    'orts_till': (1, 3),
    'cass_till': (1, 3),
    'yams_till': (1, 3),
    'swpo_till': (1, 3),
    'pota_till': (1, 3),
    
    # Perennials (categories 5-6)
    'ooil_till': (5, 6),
    'teas_till': (5, 6),
    'coco_till': (5, 6),
    'rcof_till': (5, 6),
    'acof_till': (5, 6),
    'ofib_till': (5, 6),
    'sugc_till': (5, 6),
    'oilp_till': (5, 6),
    'temf_till': (5, 6),
    'trof_till': (5, 6),
    'plnt_till': (5, 6),
    'bana_till': (5, 6),
    'cnut_till': (5, 6),
    
    # Scenario CA
    'scenario_ca_area': (7, 7),
    
    # All other annuals (grains) use categories 1-4
}

####CREATE INDIVIDUAL CROP MAPS####
print("Creating individual crop maps...")

for i, var_name in enumerate(tillage_vars):
    # Load data
    data = nc.variables[var_name][:]
    long_name = nc.variables[var_name].long_name
    units = nc.variables[var_name].units
    
    # Skip if no data
    if np.all(np.isnan(data)):
        print(f"  {i+1}/{len(tillage_vars)}: {var_name} - No data, skipping")
        continue
    
    # Determine valid category range for this crop
    if var_name in crop_category_ranges:
        vmin, vmax = crop_category_ranges[var_name]
    else:
        # Default: annual grains (1-4)
        vmin, vmax = 1, 4
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot with discrete colors for categories
    im = ax.imshow(data, cmap='tab10', vmin=1, vmax=7, interpolation='nearest')
    
    ax.set_title(f'{long_name}\n({var_name}) - {units}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Add colorbar with category labels
    cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5, 6, 7])
    cbar.set_label('Tillage Category', fontsize=12)
    
    # Get unique categories present in this crop
    unique_cats = np.unique(data[~np.isnan(data)]).astype(int)
    cat_text = '\n'.join([f'{cat}: {category_labels[cat]}' for cat in unique_cats])
    
    # Add text box with categories and valid range
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, f'Valid range: {vmin}-{vmax}\n\nCategories present:\n{cat_text}', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save individual map
    safe_name = var_name.replace('/', '_')
    plt.savefig(output_path / f'{i+1:02d}_{safe_name}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  {i+1}/{len(tillage_vars)}: {var_name} - Saved")

print(f"\n✓ All individual maps saved to: {output_path}")


####CREATE OVERVIEW GRID####
print("\nCreating overview grid...")

# Calculate grid dimensions (try to make it roughly square)
n_crops = len(tillage_vars)
n_cols = int(np.ceil(np.sqrt(n_crops)))
n_rows = int(np.ceil(n_crops / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*3))
axes = axes.flatten() if n_crops > 1 else [axes]

for i, var_name in enumerate(tillage_vars):
    data = nc.variables[var_name][:]
    
    # Plot
    im = axes[i].imshow(data, cmap='tab10', vmin=1, vmax=7, interpolation='nearest')
    axes[i].set_title(var_name, fontsize=8)
    axes[i].axis('off')

# Hide empty subplots
for i in range(len(tillage_vars), len(axes)):
    axes[i].axis('off')

# Add single colorbar
cbar = fig.colorbar(im, ax=axes, orientation='horizontal', 
                     fraction=0.02, pad=0.04, ticks=[1, 2, 3, 4, 5, 6, 7])
cbar.set_label('Tillage Category', fontsize=12)
cbar.ax.set_xticklabels(['1: Conv', '2: Trad', '3: Redu', 
                          '4: CA', '5: Rot', '6: TradRot', '7: ScenCA'], 
                         fontsize=8, rotation=45)

plt.suptitle('All Crops - Tillage Categories Overview\n' + 
             'Categories: 1-4 (grains), 1-3 (tubers/rice), 5-6 (perennials), 7 (scenario)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path / 'ALL_CROPS_OVERVIEW.png', dpi=200, bbox_inches='tight')
print(f"✓ Saved overview: {output_path / 'ALL_CROPS_OVERVIEW.png'}")

plt.show()


####CREATE SUMMARY TABLE####
print("\nCreating summary table...")

summary_data = []
for var_name in tillage_vars:
    data = nc.variables[var_name][:]
    long_name = nc.variables[var_name].long_name
    
    # Count pixels per category
    unique_cats = np.unique(data[~np.isnan(data)])
    
    row = {'Crop': var_name, 'Long Name': long_name}
    for cat in range(1, 8):
        count = np.sum(data == cat)
        row[f'Cat_{cat}'] = count
    
    # Total pixels with tillage
    row['Total_Pixels'] = np.sum(~np.isnan(data))
    summary_data.append(row)

# Save as text file
with open(output_path / 'tillage_summary.txt', 'w') as f:
    f.write("TILLAGE CATEGORY SUMMARY BY CROP\n")
    f.write("="*100 + "\n\n")
    f.write(f"{'Crop':<20} {'Long Name':<40} {'Cat1':<8} {'Cat2':<8} {'Cat3':<8} {'Cat4':<8} {'Cat5':<8} {'Cat6':<8} {'Cat7':<8} {'Total':<8}\n")
    f.write("-"*100 + "\n")
    
    for row in summary_data:
        f.write(f"{row['Crop']:<20} {row['Long Name']:<40} ")
        f.write(f"{row['Cat_1']:<8} {row['Cat_2']:<8} {row['Cat_3']:<8} {row['Cat_4']:<8} ")
        f.write(f"{row['Cat_5']:<8} {row['Cat_6']:<8} {row['Cat_7']:<8} {row['Total_Pixels']:<8}\n")
    
    f.write("\n" + "="*100 + "\n")
    f.write("Category Legend (from Mod_6_NetCDF_Export.py):\n")
    f.write("  Annual crops (grains): categories 1-4\n")
    f.write("  Tubers and rice: categories 1-3\n")
    f.write("  Perennial crops: categories 5-6\n")
    f.write("  Scenario: category 7\n\n")
    for cat, label in category_labels.items():
        f.write(f"  {cat}: {label}\n")

print(f"✓ Saved summary: {output_path / 'tillage_summary.txt'}")

# Close NetCDF
nc.close()

print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)
print(f"\nOutput location: {output_path}")
print(f"\nGenerated:")
print(f"  - {len(tillage_vars)} individual crop maps")
print(f"  - 1 overview grid (ALL_CROPS_OVERVIEW.png)")
print(f"  - 1 summary table (tillage_summary.txt)")
print("\nTillage Categories (from Mod_6_NetCDF_Export.py):")
print("  Annual crops (grains): 1-4")
print("  Tubers and rice: 1-3")
print("  Perennial crops: 5-6")
print("  Scenario: 7")
print()
for cat, label in category_labels.items():
    print(f"  {cat} = {label}")
print("="*60)