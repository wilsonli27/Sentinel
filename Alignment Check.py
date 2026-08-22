import numpy as np
import matplotlib.pyplot as plt

def check_alignment(file_path, xmin=-180, ymax=84, res=1/12):
    data = np.load(file_path)
    if len(data.shape) == 3:
        data = np.nansum(data, axis=0) # Collapse crop dimension
    
    ny, nx = data.shape
    
    # Calculate where the data 'starts' based on the first non-NaN value
    rows, cols = np.where(~np.isnan(data) & (data > 0))
    if len(rows) == 0:
        print(f"Error: No data found in {file_path}")
        return

    first_data_lat_idx = rows.min()
    first_data_lon_idx = cols.min()
    
    # Predicted coordinates for the first data point
    # Using the standard "Edge + Half-Pixel" logic
    lon_start = xmin + (first_data_lon_idx * res) + (res/2)
    lat_start = ymax - (first_data_lat_idx * res) - (res/2)
    
    print(f"File: {file_path.name}")
    print(f"  Shape: {ny}x{nx}")
    print(f"  First Data Pixel at index: [{first_data_lat_idx}, {first_data_lon_idx}]")
    print(f"  Predicted Coord: {lat_start:.4f}N, {lon_start:.4f}E")
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.imshow(data, extent=[xmin, xmin + nx*res, ymax - ny*res, ymax])
    plt.colorbar(label='Area')
    plt.title(f"Alignment Check: {file_path.name}")
    plt.show()

# Run for your primary output
from pathlib import Path
path_calc = Path("sample_output/")
check_alignment(path_calc / "conventional_annual_tillage.npy")