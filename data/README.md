# Data layout

Recommended local layout:

```text
data/
├── processed/
│   └── enmap_217_preprocessed.dat
└── roi/
    ├── ROI_1.shp
    ├── ROI_1.shx
    ├── ROI_1.dbf
    ├── ROI_1.prj
    ├── ROI_0.shp
    ├── ROI_0.shx
    ├── ROI_0.dbf
    └── ROI_0.prj
```

The raster must be the already preprocessed 217-band EnMAP image used as model input. This repository does not modify its spectral bands or apply image preprocessing.

`ROI_1` contains candidate positive pixels and `ROI_0` contains candidate negative pixels. The protocol selects 46 pixels from each class with seed 42 and rasterizes the ROIs using `all_touched=False`.

The repository does not include confidential mine coordinates or imagery.
