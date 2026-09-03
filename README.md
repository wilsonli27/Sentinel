# SoilSecura: A Novel FCN Framework for Global Tillage Monitoring With Remote-Sensing

Code and data pipeline for a Fully Convolutional Network (ResNet-50 U-Net, dual-head,
Focal Tversky loss) that classifies global cropland into Conventional, Traditional, and
Conservation tillage from Sentinel-1/2 imagery at 250m resolution. Submitted to
*Convergence: A Journal for Young Researchers*.

## Pipeline overview

The tillage label pipeline (Modules 1-6) implements a rule-based logic tree adapted from
Porwollik et al. (2019), updated to 2020-era source datasets. See Section 2 of the paper
for full methodology.

| Step | File | Purpose |
|------|------|---------|
| Mod 1 | `Mod 1.py` | Foundational dataset importer; disaggregates SPAM2020 crop area to 250m via dasymetric allocation |
| Mod 2 | `Mod 2 - Crop Mix.py` | Computes CA-suitable crop mix ratio per pixel |
| Mod 2 (data fix) | `Mod2 Conservation Booster.py` | Standalone oversampling pass: injects synthetic high-density Conservation Agriculture patches to correct the severe (~2.8%) class imbalance in the raw labeled set |
| Mod 3 | `Mod 3 - Logit.py`* | Logit suitability scoring (aridity, erosion, field size, crop mix) calibrated against AQUASTAT CA totals |
| Mod 3b | `Mod 3b - CA Allocation for Mod 5.py`* | Produces the 22-band CA allocation map feeding Module 4 |
| Mod 4 | `Mod 4 - Tillage.py` | Applies the logic tree; assigns Conventional/Traditional/Reduced/Rotational tillage via Boolean masks |
| Mod 5 | `Mod 5 - Tillage Sums.py` | Aggregates global tillage statistics to CSV |
| Mod 6 | `Mod_6_NetCDF_Export.py` | Collapses per-crop outputs into the final 6-band master tillage raster |

*\*Repo currently also contains `Mod 3 new.py` and `Mod 3b - fast tracked without Pass1.py`,
earlier trial versions kept during development. The files listed above are the ones the
published results are based on; the alternates are retained for provenance but are not
part of the reported pipeline.*

## Spatial extrapolation (Sentinel imagery)

| File | Purpose |
|------|---------|
| `Clustering_mod1.py` | Pass 1: per-tile k=12 clustering on annual NDVI-minimum timing (Section 3.2) |
| `Clustering_mod2.py` | Pass 2: re-clusters into k=4 dominant tillage-timing regimes; assigns Cluster_ID |

## Model training and architecture

| File | Purpose |
|------|---------|
| `Final Code Dualhead Tversky.py` | Final ResNet-50 U-Net, dual-head (segmentation + fractional regression), Focal Tversky loss (α=0.3, β=0.7, γ=1.33). Produces `Global_Tillage_DualHead_Final.h5` and the results reported in Table 4 and Figure 3 of the paper. |

## Requirements

TensorFlow/Keras, rasterio, numpy, opencv-python (cv2), matplotlib, scikit-learn.
Google Earth Engine access (with a registered GEE project) required for Modules 1-3b
and the clustering scripts, which query GEE datasets directly.

## Data

Sentinel-1/2 imagery and derived indices are extracted via Google Earth Engine as
described in Section 3.3 of the paper. SPAM2020, SoilGrids250m, GloSEM 1.3, Global
Aridity Index v3, World Bank income data, and FAO AQUASTAT CA statistics are sourced
per Table 3 of the paper. Raw and intermediate rasters are not included in this repo
due to size; contact the corresponding author for access.

## Citation

If you use this code, please cite the associated paper (details added upon publication).
