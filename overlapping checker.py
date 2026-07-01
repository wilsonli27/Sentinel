import xarray as xr
import numpy as np
from scipy import stats

# Setup paths
input_file  = "sample_output/tillage_revised.nc4"
output_file = "sample_output/tillage_single_layer_old.nc4"

# ── Notification verbosity ────────────────────────────────────────────────────
# PIXEL_NOTIFY_LIMIT : max individual pixel messages printed to console.
#                      Set to None to print ALL (warning: can be millions).
# LOG_OVERRIDES_FILE : path to write every override to disk, or None to skip.
PIXEL_NOTIFY_LIMIT = 500
LOG_OVERRIDES_FILE = "sample_output/pixel_overrides.log"
# ─────────────────────────────────────────────────────────────────────────────


def _calc_mode(arr):
    """
    Calculate mode along the last axis (crop dimension).
    nan_policy='omit' ignores NaNs so they don't become the mode.
    """
    mode_result = stats.mode(arr, axis=-1, nan_policy='omit', keepdims=False)
    return mode_result.mode


def detect_and_report_overrides(stack, lat_coords, lon_coords, crop_vars):
    """
    Iterates over every pixel and checks whether multiple crops assigned
    DIFFERENT tillage classes to it.  When they do, the mode is 'overriding'
    minority values — we print a notification for each such pixel.

    Parameters
    ----------
    stack       : numpy ndarray, shape (n_crops, n_lat, n_lon)
    lat_coords  : 1-D array of latitude  values
    lon_coords  : 1-D array of longitude values
    crop_vars   : list of crop variable names (same order as stack axis 0)

    Returns
    -------
    override_count : int   — total number of pixels with at least one conflict
    """
    print("\n── Scanning for pixel overrides / overlaps ──────────────────────────")

    n_crops, n_lat, n_lon = stack.shape
    override_count = 0
    notify_count   = 0

    log_fh = open(LOG_OVERRIDES_FILE, "w") if LOG_OVERRIDES_FILE else None
    if log_fh:
        log_fh.write(
            "lat\tlon\tn_unique_classes\tclasses_present\t"
            "mode_winner\toveridden_crops\n"
        )

    for yi in range(n_lat):
        for xi in range(n_lon):

            pixel_vals = stack[:, yi, xi]           # one value per crop
            valid_mask = ~np.isnan(pixel_vals)
            valid_vals = pixel_vals[valid_mask]

            if valid_vals.size == 0:
                continue                             # ocean / no-data pixel

            unique_classes = np.unique(valid_vals)

            if unique_classes.size <= 1:
                continue                             # all crops agree — no conflict

            # ── Conflict detected ──────────────────────────────────────────
            override_count += 1

            mode_winner = stats.mode(
                valid_vals, nan_policy='omit', keepdims=False
            ).mode

            overridden_crops = [
                crop_vars[ci]
                for ci, v in enumerate(pixel_vals)
                if not np.isnan(v) and v != mode_winner
            ]

            lat_val = float(lat_coords[yi])
            lon_val = float(lon_coords[xi])

            msg = (
                f"[OVERRIDE] lat={lat_val:+.4f}  lon={lon_val:+.4f} | "
                f"{unique_classes.size} classes present: {unique_classes.tolist()} | "
                f"MODE winner={int(mode_winner)} | "
                f"overridden crops ({len(overridden_crops)}): "
                f"{', '.join(overridden_crops)}"
            )

            # Write to log file (always, if enabled)
            if log_fh:
                log_fh.write(
                    f"{lat_val}\t{lon_val}\t{unique_classes.size}\t"
                    f"{unique_classes.tolist()}\t{int(mode_winner)}\t"
                    f"{','.join(overridden_crops)}\n"
                )

            # Print to console up to the notify limit
            if PIXEL_NOTIFY_LIMIT is None or notify_count < PIXEL_NOTIFY_LIMIT:
                print(msg)
                notify_count += 1
            elif notify_count == PIXEL_NOTIFY_LIMIT:
                print(
                    f"  ... console limit of {PIXEL_NOTIFY_LIMIT} reached. "
                    f"Further overrides written to {LOG_OVERRIDES_FILE} only."
                )
                notify_count += 1   # increment once more so we don't re-print the message

    if log_fh:
        log_fh.close()

    print(f"\n── Override scan complete ───────────────────────────────────────────")
    print(f"   Total pixels with at least one conflict : {override_count:,}")
    if LOG_OVERRIDES_FILE:
        print(f"   Full log written to                     : {LOG_OVERRIDES_FILE}")
    print(f"────────────────────────────────────────────────────────────────────\n")

    return override_count


def collapse_crops_to_map():
    print(f"Opening {input_file}...")

    # 1. Load the dataset
    ds = xr.open_dataset(input_file, chunks='auto')

    # 2. Identify crop variables
    crop_vars = [v for v in ds.data_vars if v.endswith('_till')]
    print(f"Found {len(crop_vars)} crop variables to merge.")

    # 3. Stack variables
    print("Stacking variables... (Lazy evaluation)")
    stack = ds[crop_vars].to_array(dim='crop')

    # Keep 'crop' dimension un-split so mode operates on the full vector
    stack = stack.chunk({'crop': -1})

    # ── Override detection ────────────────────────────────────────────────────
    # Compute into memory for pixel-level inspection.
    # For very large global grids you may want to do this tile-by-tile.
    print("Loading stack into memory for override detection...")
    stack_np = stack.values    # shape: (n_crops, n_lat, n_lon)

    detect_and_report_overrides(
        stack_np,
        lat_coords=ds.lat.values,
        lon_coords=ds.lon.values,
        crop_vars=crop_vars
    )
    # ─────────────────────────────────────────────────────────────────────────

    # 4. Collapse using MODE
    print("Calculating the combined map (Finding dominant tillage class)...")

    combined_map = xr.apply_ufunc(
        _calc_mode,
        stack,
        input_core_dims=[['crop']],
        dask='parallelized',
        output_dtypes=[np.float32]
    )

    # 5. Create clean output Dataset
    output_ds = xr.Dataset(
        data_vars={
            'tillage_class': (['lat', 'lon'], combined_map.data)
        },
        coords={
            'lat': ds.lat,
            'lon': ds.lon
        },
        attrs={
            'description': 'Dominant tillage map (Mode) from 42 individual crop variables',
            'classes': '1=Conventional, 2=Conservation, 3=Rotational',
            'logic_fix': 'Changed from MAX to MODE to prevent Rotational bias.'
        }
    )

    # 6. Save to disk with compression
    print(f"Saving combined map to {output_file}...")
    encoding = {'tillage_class': {'zlib': True, 'complevel': 5, '_FillValue': -9999}}
    output_ds.to_netcdf(output_file, encoding=encoding, compute=True)

    print("\n" + "=" * 30)
    print("SUCCESS: Single layer created.")
    print(f"File saved: {output_file}")
    print("=" * 30)


if __name__ == "__main__":
    collapse_crops_to_map()