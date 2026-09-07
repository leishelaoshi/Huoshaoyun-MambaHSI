# Manuscript-method alignment

| Manuscript item | Repository implementation |
|---|---|
| Model input | preprocessed 217-band EnMAP raster |
| Positive labels | 46 pixels |
| Negative labels | 46 pixels |
| Class ratio | 1:1 |
| Patch candidates | 1×1, 3×3, 5×5, 7×7 |
| Final patch | selected by rerunning the comparison; not hard-coded |
| Primary validation | nested buffered LOOCV |
| Inner validation | 3-fold spatial-group CV with spatial buffer |
| Optuna trials | 10 per outer fold |
| LR search | 1e-5–1e-3, log-uniform |
| Batch size | 4, 8, 16 |
| Weight decay | 1e-5–1e-3, log-uniform |
| Tuning epochs | 20 |
| Training epochs | 100 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Scheduler | CosineAnnealingLR |
| Augmentation | none |
| Sensitivity | 25%, 50%, 75%, 100%; 3 repeats |
| Ablation | spectral-only / spatial-only / simple fusion / adaptive fusion |
| Whole-scene model | separate spatially constrained Optuna search using all labels |

## Manuscript text that should explicitly match the code

1. State that the MambaHSI workflow starts from the finalized 217-band preprocessed EnMAP image.
2. State the 6-pixel Chebyshev exclusion radius and minimum retained center separation of 7 pixels (about 210 m at 30 m GSD), if this is the protocol used for the reported results.
3. State that size-safe average pooling is used for direct 1×1/3×3/5×5/7×7 inputs.
4. State the exact rule used to select the final patch size.
