import xarray as xr

# Path to your file
file_path = r"D:\Users\Wilson\Downloads\Sentinel\sample_output\tillage_single_layer.nc4"

# Open dataset
ds = xr.open_dataset(file_path)

# Extract the specific variable
da = ds["tillage_class"]

# Convert to DataFrame
df = da.to_dataframe().reset_index()

# Save to CSV
df.to_csv("tillage_class.csv", index=False)

print("tillage_class saved to tillage_class.csv")