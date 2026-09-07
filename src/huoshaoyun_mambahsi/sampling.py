from __future__ import annotations

import numpy as np


def select_balanced_samples(
    pos_coords: np.ndarray,
    neg_coords: np.ndarray,
    n_per_class: int,
    seed: int,
):
    if len(pos_coords) < n_per_class:
        raise RuntimeError(f"Positive candidates: {len(pos_coords)} < {n_per_class}")
    if len(neg_coords) < n_per_class:
        raise RuntimeError(f"Negative candidates: {len(neg_coords)} < {n_per_class}")

    rng = np.random.default_rng(seed)
    pos = pos_coords[rng.choice(len(pos_coords), n_per_class, replace=False)]
    neg = neg_coords[rng.choice(len(neg_coords), n_per_class, replace=False)]
    coords = np.vstack([pos, neg])
    labels = np.r_[np.ones(n_per_class, dtype=np.int64), np.zeros(n_per_class, dtype=np.int64)]
    order = rng.permutation(len(labels))
    return coords[order], labels[order]


def extract_patch(img: np.ndarray, row: int, col: int, patch_size: int) -> np.ndarray:
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    h, w, _ = img.shape
    half = patch_size // 2
    r0, r1 = row - half, row + half + 1
    c0, c1 = col - half, col + half + 1
    sr0, sr1 = max(r0, 0), min(r1, h)
    sc0, sc1 = max(c0, 0), min(c1, w)
    patch = img[sr0:sr1, sc0:sc1]

    pad = ((sr0 - r0, r1 - sr1), (sc0 - c0, c1 - sc1), (0, 0))
    if any(v > 0 for pair in pad[:2] for v in pair):
        patch = np.pad(patch, pad, mode="reflect" if min(h, w) > 1 else "edge")

    return np.moveaxis(patch, -1, 0).astype(np.float32)


def build_patch_bank(img: np.ndarray, coords: np.ndarray, patch_size: int) -> np.ndarray:
    return np.stack(
        [extract_patch(img, int(r), int(c), patch_size) for r, c in coords],
        axis=0,
    )
