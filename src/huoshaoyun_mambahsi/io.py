from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features


def parse_wavelengths(src, n_bands: int) -> np.ndarray:
    for tags in (src.tags(ns="ENVI"), src.tags()):
        if not tags:
            continue
        for key in ("wavelength", "WAVELENGTH", "wavelengths", "WAVELENGTHS"):
            if key in tags:
                raw = str(tags[key]).replace("{", "").replace("}", "").replace(",", " ")
                vals = [float(v) for v in raw.split()]
                if len(vals) == n_bands:
                    arr = np.asarray(vals, dtype=float)
                    if np.nanmedian(arr) < 20:
                        arr *= 1000.0
                    return arr
    return np.arange(1, n_bands + 1, dtype=float)


def load_raster(path: str | Path):
    path = str(path)
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        wavelengths = parse_wavelengths(src, arr.shape[0])
        nodata = src.nodata
    return np.moveaxis(arr, 0, -1), profile, transform, crs, wavelengths, nodata


def rasterize_roi(
    path: str | Path,
    shape: Tuple[int, int],
    transform,
    crs,
    all_touched: bool = False,
) -> np.ndarray:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise RuntimeError(f"Empty ROI: {path}")
    gdf = gdf.to_crs(crs)
    mask = features.geometry_mask(
        gdf.geometry,
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=all_touched,
    )
    return np.column_stack(np.where(mask)).astype(np.int64)


def save_sample_manifest(
    path: str | Path,
    coords: np.ndarray,
    labels: np.ndarray,
    include_coordinates: bool = True,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_id": [f"S{i + 1:03d}" for i in range(len(labels))],
            "row": coords[:, 0],
            "col": coords[:, 1],
            "label": labels.astype(int),
        }
    )
    out = df if include_coordinates else df[["sample_id", "label"]]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return df


def save_wavelengths(path: str | Path, wavelengths: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "band_1based": np.arange(1, len(wavelengths) + 1),
            "wavelength_nm_or_index": wavelengths,
        }
    ).to_csv(path, index=False)
