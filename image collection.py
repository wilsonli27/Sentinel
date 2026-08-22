import subprocess

# The exact name of your temporary GCS bucket
gcs_bucket = "gs://gvrsf-temp-upload-bucket"

# The target Earth Engine Image Collection 
ee_collection = "projects/gvrsf-2026/assets/tillage_inputs"

# 1. Use gsutil to list all the TIFF files we just uploaded to the bucket
print("Fetching list of files from GCS...")
list_command = f"gsutil ls {gcs_bucket}/*.tif"
result = subprocess.run(list_command, capture_output=True, text=True, shell=True)

# Clean up the output to get a list of gs:// file paths
gcs_files = [line.strip() for line in result.stdout.split('\n') if line.strip()]
print(f"Found {len(gcs_files)} files in the bucket.")

# 2. Loop through the GCS files and submit them to Earth Engine
for gs_path in gcs_files:
    # Extract just the filename without the .tif to use as the asset name
    # e.g., gs://bucket/spam2020_A_VEGE_R.tif -> spam2020_A_VEGE_R
    filename = gs_path.split('/')[-1]
    asset_name = filename.replace(".tif", "")
    
    ee_asset_id = f"{ee_collection}/{asset_name}"

    # Construct the Earth Engine upload command
    command = [
        "earthengine",
        "upload",
        "image",
        "--asset_id", ee_asset_id,
        gs_path
    ]

    print(f"Submitting task for: {asset_name}...")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit {asset_name}. Error: {e}")

print("All tasks submitted! Check the Earth Engine Code Editor Tasks tab.")