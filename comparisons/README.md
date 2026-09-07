# Comparative models

The manuscript compares MambaHSI with five representative hyperspectral classification models: 3D CNN, HybridSN, ViT, SpectralFormer, and SSFTT.

The comparative models are established methods with publicly available implementations or implementation resources associated with their original publications. To avoid redistributing third-party code and to preserve consistency with the original implementations, their source code is not included in this repository.

Researchers may obtain the corresponding implementations from the original authors or official/public repositories and evaluate them using the same experimental protocol adopted in this study.

For reproducible comparison, this repository provides:

- the same sample definition;
- the same selected spatial patch size;
- the same spatially controlled outer folds;
- the same evaluation metrics;
- the same independent hyperparameter-optimization strategy.

The outer buffered-LOOCV splits used for model comparison can be exported using:

`python scripts/07_export_outer_folds.py --config configs/paper_protocol.yaml`

Any comparative model can therefore be evaluated under the same data partitions and evaluation protocol used for MambaHSI.
