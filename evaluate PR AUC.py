import os
import numpy as np
import rasterio

BASE_DIR = r"D:\Users\Wilson\Downloads\Sentinel\image_cnn\Dataset_Split"
PATCH_SIZE = 64
CLASSES = ['Conservational', 'Conventional', 'Traditional']  # must match Y_patch channel order


def compute_class_balance(data_dir, patch_size=64):
    tiff_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tif')]
    counts = np.zeros(3, dtype=np.int64)
    total_pixels = 0

    for file_path in tiff_files:
        with rasterio.open(file_path) as src:
            data = src.read()
            data = np.transpose(data, (1, 2, 0))
            data = np.nan_to_num(data, nan=0.0)

            h, w, _ = data.shape
            if h < patch_size or w < patch_size:
                continue

            start_y = (h - patch_size) // 2
            start_x = (w - patch_size) // 2
            data_cropped = data[start_y:start_y + patch_size, start_x:start_x + patch_size, :]

            X_patch = data_cropped[:, :, :13].astype(np.float32)
            Y_patch = data_cropped[:, :, 13:16].astype(np.float32)

            if np.all(X_patch[:, :, :10] == 0.0):
                continue

            y_idx = np.argmax(Y_patch, axis=-1)
            for c in range(3):
                counts[c] += np.sum(y_idx == c)
            total_pixels += y_idx.size

    return counts, total_pixels


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(BASE_DIR, split)
        counts, total = compute_class_balance(split_dir, PATCH_SIZE)
        print(f"=== {split.upper()} ===")
        for i, cls in enumerate(CLASSES):
            print(f"{cls:15s}  prevalence: {counts[i] / total:.4f}  ({counts[i]:,} px)")
        print()