# Comparative models

The manuscript compares MambaHSI with 3D CNN, HybridSN, ViT, SpectralFormer, and SSFTT.
This repository does not replace those published architectures with approximate reimplementations.

Use `scripts/08_export_outer_folds.py` to export the exact outer buffered-LOOCV sample splits, then run each comparative model with the same sample IDs, patch size selected by the protocol, evaluation metrics, and independent hyperparameter optimization.

If the exact implementations used for the manuscript are available, add them here before claiming that all comparative-model code is publicly released.
