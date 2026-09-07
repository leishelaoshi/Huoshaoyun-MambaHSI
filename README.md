# Huoshaoyun MambaHSI Reproducibility Repository

Reproducible MambaHSI workflow for EnMAP-based mineralization mapping of the Huoshaoyun nonsulfide Pb-Zn deposit.

## Scope

The repository starts from the **preprocessed 217-band EnMAP raster used as the model input**. It reproduces the modeling workflow:

1. input-raster validation;
2. 46 positive and 46 negative sample construction;
3. direct comparison of 1×1, 3×3, 5×5, and 7×7 spatial inputs;
4. nested buffered leave-one-out cross-validation;
5. inner 3-fold spatial-group CV with Optuna;
6. training-label proportion sensitivity analysis;
7. spatial/spectral/fusion ablation;
8. separate parameter search and whole-scene inference.

The final patch size, learning rate, batch size, weight decay, and evaluation metrics are **not hard-coded**. They are obtained by rerunning the protocol.

## Repository structure

```text
Huoshaoyun-MambaHSI-reproducibility/
├── configs/
│   └── paper_protocol.yaml
├── data/
│   └── README.md
├── comparisons/
│   └── README.md
├── scripts/
│   ├── 01_build_samples.py
│   ├── 02_patch_selection.py
│   ├── 03_nested_loocv.py
│   ├── 04_sensitivity.py
│   ├── 05_ablation.py
│   ├── 06_full_scene_prediction.py
│   ├── 07_export_outer_folds.py
│   └── run_all.py
├── src/huoshaoyun_mambahsi/
│   ├── config.py
│   ├── experiments.py
│   ├── io.py
│   ├── model.py
│   ├── sampling.py
│   ├── splits.py
│   ├── training.py
│   ├── tuning.py
│   └── workflow.py
├── tests/
├── MANUSCRIPT_ALIGNMENT.md
├── REPRODUCIBILITY.md
├── NOTICE.md
├── CITATION.cff
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

## Installation

The paper environment used Python 3.8.20, PyTorch 2.2.2, and CUDA 11.8.

```bash
conda env create -f environment.yml
conda activate huoshaoyun-mambahsi
pip install -e .
```

## Required input data

The workflow requires:

- the **already preprocessed 217-band EnMAP raster used by the study**;
- a positive ROI shapefile;
- a negative ROI shapefile.

Set their paths in `configs/paper_protocol.yaml`:

```yaml
paths:
  enmap_raster: data/processed/enmap_217_preprocessed.dat
  positive_roi: data/roi/ROI_1.shp
  negative_roi: data/roi/ROI_0.shp
  output_dir: outputs/paper_run

input:
  expected_bands: 217
```

This repository does **not** perform bad-band removal, Savitzky-Golay filtering, continuum removal, or other upstream image preprocessing. The supplied raster is treated as the finalized model-input image.

## Run the complete workflow

```bash
python scripts/run_all.py --config configs/paper_protocol.yaml
```

The full nested protocol is computationally expensive because every patch candidate and every outer fold performs an inner Optuna search.

## Run stage by stage

```bash
python scripts/01_build_samples.py --config configs/paper_protocol.yaml
python scripts/02_patch_selection.py --config configs/paper_protocol.yaml
python scripts/03_nested_loocv.py --config configs/paper_protocol.yaml
python scripts/04_sensitivity.py --config configs/paper_protocol.yaml
python scripts/05_ablation.py --config configs/paper_protocol.yaml
python scripts/06_full_scene_prediction.py --config configs/paper_protocol.yaml
python scripts/07_export_outer_folds.py --config configs/paper_protocol.yaml
```

## Main outputs

```text
outputs/paper_run/
├── 00_audit/
├── 01_dataset/
├── 02_patch_selection/
│   ├── patch_size_comparison.csv
│   └── selected_patch_size.json
├── 03_primary_nested_loocv/
│   ├── primary_predictions.csv
│   ├── primary_metrics.json
│   └── primary_learning_curves.csv
├── 04_sensitivity/
├── 05_ablation/
└── 06_full_scene/
    ├── deployment_params.json
    ├── final_mambahsi_state_dict.pt
    └── mambahsi_mineralization_score.tif
```

## Comparative models

The manuscript also reports 3D CNN, HybridSN, ViT, SpectralFormer, and SSFTT. This repository does not replace their original implementations with approximations.Use scripts/07_export_outer_folds.py to export the spatially controlled outer splits generated from the input samples for consistent model comparison.

## Important implementation note

To enable direct evaluation of the 1×1, 3×3, 5×5, and 7×7 spatial inputs used in the patch-size experiment, the implementation applies 2×2 average pooling only when the current spatial feature map is at least 2×2. The remaining MambaHSI architecture and training procedure are unchanged.
