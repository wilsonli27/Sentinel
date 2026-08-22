"""
Field Size File Diagnostic Tool
Run this to identify the exact issue with your file
"""
import os
from pathlib import Path

# Your path
field_folder = Path("D:/Users/Wilson/Downloads/Sentinel/lesiv_2018_field_sizes")

print("="*80)
print("FIELD SIZE FILE DIAGNOSTIC")
print("="*80)

# Check if folder exists
print(f"\n1. Folder exists: {field_folder.exists()}")

if field_folder.exists():
    print(f"\n2. Contents of {field_folder.name}:")
    
    all_items = list(field_folder.iterdir())
    print(f"   Total items: {len(all_items)}")
    
    for item in all_items:
        # Get file info
        is_file = item.is_file()
        is_dir = item.is_dir()
        size_mb = item.stat().st_size / (1024**2) if is_file else 0
        
        # Check if it's really a raster file
        name_lower = item.name.lower()
        
        print(f"\n   📁 {item.name}")
        print(f"      Type: {'FILE' if is_file else 'FOLDER'}")
        print(f"      Size: {size_mb:.1f} MB")
        print(f"      Full path: {item}")
        
        # Try to detect raster format
        if is_file:
            # Check extension
            suffix = item.suffix.lower()
            print(f"      Extension: '{suffix}' {f'({suffix})' if suffix else '(NONE - THIS IS THE PROBLEM!)'}")
            
            # Try to read first few bytes to detect format
            try:
                with open(item, 'rb') as f:
                    magic_bytes = f.read(8)
                    magic_hex = ' '.join(f'{b:02x}' for b in magic_bytes)
                    print(f"      Magic bytes: {magic_hex}")
                    
                    # Detect format from magic bytes
                    if magic_bytes[:2] == b'II' or magic_bytes[:2] == b'MM':
                        print(f"      ✓ Detected: GeoTIFF/TIFF format")
                    elif magic_bytes[:6] == b'EHFA_H':
                        print(f"      ✓ Detected: ERDAS IMAGINE (.img) format")
                    elif b'HDF' in magic_bytes:
                        print(f"      ✓ Detected: HDF format")
                    else:
                        print(f"      ⚠️  Unknown raster format")
            except Exception as e:
                print(f"      ⚠️  Could not read file: {e}")
        
        elif is_dir:
            # Check if it's an ESRI Grid folder
            sub_items = list(item.iterdir())
            has_adf = any(f.suffix.lower() == '.adf' for f in sub_items if f.is_file())
            if has_adf:
                print(f"      ✓ Contains .adf files - likely ESRI Grid format")
                print(f"      Files: {[f.name for f in sub_items if f.suffix.lower() == '.adf']}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Check for the specific "Global Field Sizes" item
target = field_folder / "Global Field Sizes"
if target.exists():
    if target.is_dir():
        print("\n⚠️  'Global Field Sizes' is a FOLDER, not a file!")
        print("   Solution: Look inside the folder for the actual raster file")
        sub_files = list(target.glob("*.*"))
        if sub_files:
            print(f"\n   Files inside folder:")
            for f in sub_files:
                print(f"   - {f.name} ({f.stat().st_size / (1024**2):.1f} MB)")
    else:
        print("\n✓ 'Global Field Sizes' is a FILE")
        if not target.suffix:
            print("\n⚠️  BUT it has NO EXTENSION!")
            print("   Solution: Try adding .tif or .img extension")
        else:
            print(f"   Extension: {target.suffix}")
else:
    print("\n❌ 'Global Field Sizes' not found!")

print("\n" + "="*80)