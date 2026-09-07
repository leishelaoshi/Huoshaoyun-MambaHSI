# Comparative models

The manuscript compares MambaHSI with 3D CNN, HybridSN, ViT, SpectralFormer, and SSFTT.

The exact implementations of these comparative architectures are not redistributed in this repository. This repository therefore focuses on reproducibility of the MambaHSI workflow and provides the experimental protocol and spatially controlled data splits required for consistent model comparison.

Use `scripts/07_export_outer_folds.py` to export the outer buffered-LOOCV sample splits. Comparative models should use the same sample definitions, selected patch size, evaluation metrics, and independent hyperparameter optimization protocol described in the manuscript.

The absence of the comparative-model source code should not be interpreted as a reimplementation of those published architectures.
