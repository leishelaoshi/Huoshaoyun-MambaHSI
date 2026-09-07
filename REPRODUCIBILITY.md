# Reproducibility checklist

## Input boundary

Reproduction begins from the **preprocessed 217-band EnMAP raster used as the model input**. Upstream remote-sensing preprocessing is outside this repository.

## Encoded in the repository

- input raster band-count check: 217 bands
- 46 positive + 46 negative labels
- random seed 42
- center-pixel ROI rasterization (`all_touched=False`)
- patch candidates: 1×1, 3×3, 5×5, 7×7
- patch selection by OA, then Kappa/F1 for ties
- outer buffered LOOCV
- 6-pixel Chebyshev exclusion radius
- inner 3-fold spatial-group CV
- 10 Optuna trials per outer fold
- learning-rate range: 1e-5 to 1e-3, log scale
- batch candidates: 4, 8, 16
- weight-decay range: 1e-5 to 1e-3, log scale
- 20 tuning epochs
- 100 training epochs
- AdamW + CrossEntropyLoss + CosineAnnealingLR
- no geometric augmentation
- 25%, 50%, 75%, 100% sensitivity experiment
- spectral-only, spatial-only, simple-fusion, full-model ablation
- separate deployment tuning using all labels

## Comparative models

To reproduce every model in the comparison table, the exact implementations used for 3D CNN, HybridSN, ViT, SpectralFormer, and SSFTT must also be released or unambiguously referenced with their exact configuration. This repository exports the outer-fold manifests but does not replace those published architectures with approximate substitutes.

## Sensitivity and ablation hyperparameters

For each outer fold, the sensitivity and ablation experiments reuse the hyperparameters selected by the primary nested inner-CV for that same outer fold. The held-out outer sample is not used for hyperparameter selection. In the sensitivity experiment, the training-label proportion is the controlled variable; in the ablation experiment, the model module is the controlled variable.
