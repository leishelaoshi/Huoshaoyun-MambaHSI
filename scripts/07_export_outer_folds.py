#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from huoshaoyun_mambahsi.config import load_config
from huoshaoyun_mambahsi.splits import outer_train_indices
from huoshaoyun_mambahsi.workflow import output_dirs, stage_build_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper_protocol.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    out = output_dirs(cfg)
    sample_file = out["dataset"] / "samples.npz"
    if not sample_file.exists():
        stage_build_samples(cfg)
    data = np.load(sample_file, allow_pickle=True)
    coords = data["coords"]
    labels = data["labels"]
    ids = data["sample_ids"].astype(str)

    fold_dir = out["dataset"] / "outer_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for val_idx in range(len(labels)):
        train_idx, removed = outer_train_indices(
            coords, val_idx, cfg.training.outer_buffer_radius_pixels
        )
        payload = {
            "fold": val_idx,
            "validation_sample": ids[val_idx],
            "validation_label": int(labels[val_idx]),
            "train_samples": ids[train_idx].tolist(),
            "removed_by_buffer": ids[removed].tolist(),
            "buffer_radius_pixels": cfg.training.outer_buffer_radius_pixels,
        }
        (fold_dir / f"fold_{val_idx:03d}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
