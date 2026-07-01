import xarray as xr
import rioxarray

ds = xr.open_dataset(
    r"D:\Users\Wilson\Downloads\Sentinel\sample_output\tillage_single_layer_old.nc4",
    decode_coords="all"
)

da = ds["tillage_class"]

# Set spatial dims on the DataArray itself
da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

# Set CRS
da = da.rio.write_crs("EPSG:4326")

# Export
da.rio.to_raster("Global_Tillage_Map_Old.tif", compress="LZW")

print("Conversion complete.")
