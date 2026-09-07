from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def chebyshev_distance(a: Sequence[int], b: Sequence[int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def buffered_train_indices(
    coords: np.ndarray,
    candidate_train: Iterable[int],
    validation: Iterable[int],
    radius: int,
):
    candidate_train = np.asarray(list(candidate_train), dtype=int)
    validation = np.asarray(list(validation), dtype=int)
    keep, removed = [], []
    for i in candidate_train:
        if any(chebyshev_distance(coords[i], coords[j]) <= radius for j in validation):
            removed.append(i)
        else:
            keep.append(i)
    return np.asarray(keep, dtype=int), np.asarray(removed, dtype=int)


def outer_train_indices(coords: np.ndarray, val_idx: int, radius: int):
    candidate = [i for i in range(len(coords)) if i != val_idx]
    return buffered_train_indices(coords, candidate, [val_idx], radius)


def spatial_group_ids(coords: np.ndarray, block_size_pixels: int) -> np.ndarray:
    br = coords[:, 0] // int(block_size_pixels)
    bc = coords[:, 1] // int(block_size_pixels)
    return br.astype(np.int64) * 1_000_000 + bc.astype(np.int64)
