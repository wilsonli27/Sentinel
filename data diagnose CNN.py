import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Pointing directly to your Hybrid export folder containing all patches
TIFF_DIR = r"D:\Users\Wilson\Downloads\Sentinel\image_cnn\CNN_Tillage_Patches_Hybrid"
CLASSES = ['Conservational', 'Conventional', 'Traditional']

def analyze_class_proportions():
    tiff_files = [f for f in os.listdir(TIFF_DIR) if f.endswith('.tif')]
    
    if not tiff_files:
        print(f"❌ No TIFF files found in: {TIFF_DIR}")
        return
    
    # Check how many of the new Booster patches made it into the folder
    booster_count = sum(1 for f in tiff_files if "Booster" in f)
    print(f"Found {len(tiff_files)} total patches ({booster_count} Booster CA Patches). Scanning pixel data...\n")

    # Dictionary to hold the absolute sum of fractional pixels
    total_pixels = {0: 0.0, 1: 0.0, 2: 0.0}

    # ==========================================
    # 2. FILE SCANNER
    # ==========================================
    for idx, file in enumerate(tiff_files):
        path = os.path.join(TIFF_DIR, file)
        
        with rasterio.open(path) as src:
            # rasterio is 1-indexed. Labels sit at the very end of the 16-band stack
            labels = src.read([14, 15, 16]) 
            
            # np.nansum safely ignores any empty/edge pixels (NaNs)
            total_pixels[0] += np.nansum(labels[0])
            total_pixels[1] += np.nansum(labels[1])
            total_pixels[2] += np.nansum(labels[2])

        # Print progress so it doesn't flood your terminal
        if (idx + 1) % 100 == 0 or (idx + 1) == len(tiff_files):
            print(f"  ✓ Scanned {idx + 1}/{len(tiff_files)} patches...")

    # ==========================================
    # 3. STATISTICAL REPORT
    # ==========================================
    grand_total = sum(total_pixels.values())
    
    print("\n" + "="*50)
    print("FINAL PIXEL-LEVEL PROPORTIONS (GROUND TRUTH)")
    print("="*50)
    
    for c in range(3):
        pct = (total_pixels[c] / grand_total) * 100
        print(f"{CLASSES[c]}: {pct:.2f}%  (Total Fractional Area: {total_pixels[c]:,.1f})")

    # Calculate new class weights for the neural network automatically
    print("\n" + "="*50)
    print("NEW RECOMMENDED FOCAL LOSS CLASS WEIGHTS")
    print("="*50)
    for c in range(3):
        pct = (total_pixels[c] / grand_total) * 100
        if pct > 0:
            weight = 100.0 / (3.0 * pct) # Weight balancing formula
            print(f"{CLASSES[c]} Weight: {weight:.2f}")

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    labels_plot = [f"{CLASSES[c]}\n({total_pixels[c]/grand_total*100:.1f}%)" for c in range(3)]
    sizes = [total_pixels[c] for c in range(3)]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e'] # Green, Blue, Orange

    plt.figure(figsize=(8, 8))
    
    # Donut chart style
    wedges, texts, autotexts = plt.pie(
        sizes, labels=labels_plot, colors=colors, autopct='%1.1f%%', 
        startangle=140, pctdistance=0.85, explode=(0.05, 0, 0)
    )
    
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title(f"Actual Pixel-Level Class Distribution\nAcross {len(tiff_files)} Boosted Patches", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("Hybrid_Data_Diagnostic_Boosted.png", dpi=300)
    print("\n✅ Saved visual report to 'Hybrid_Data_Diagnostic_Boosted.png'")
    plt.show()

if __name__ == "__main__":
    analyze_class_proportions()