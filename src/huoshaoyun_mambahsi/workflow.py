from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import PaperConfig
from .experiments import (
    aggregate_curves,
    full_scene_inference,
    run_ablation,
    run_buffered_loocv,
    run_patch_selection,
    run_sensitivity,
    train_final_model,
    tune_deployment_params,
)
from .io import load_raster, rasterize_roi, save_sample_manifest, save_wavelengths
from .sampling import build_patch_bank, select_balanced_samples
from .training import TrainParams, save_environment, seed_everything


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def output_dirs(cfg: PaperConfig):
    root = Path(cfg.paths.output_dir)
    dirs = {
        "root": root,
        "audit": root / "00_audit",
        "dataset": root / "01_dataset",
        "patch": root / "02_patch_selection",
        "primary": root / "03_primary_nested_loocv",
        "sensitivity": root / "04_sensitivity",
        "ablation": root / "05_ablation",
        "scene": root / "06_full_scene",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def stage_build_samples(cfg: PaperConfig):
    out = output_dirs(cfg)
    image_path = cfg.paths.enmap_raster
    img, _, transform, crs, wavelengths, _ = load_raster(image_path)
    h, w, c = img.shape
    if c != cfg.input.expected_bands:
        raise ValueError(f"Expected {cfg.input.expected_bands} bands, found {c}")

    pos = rasterize_roi(
        cfg.paths.positive_roi,
        (h, w),
        transform,
        crs,
        all_touched=cfg.samples.all_touched,
    )
    neg = rasterize_roi(
        cfg.paths.negative_roi,
        (h, w),
        transform,
        crs,
        all_touched=cfg.samples.all_touched,
    )
    coords, labels = select_balanced_samples(
        pos, neg, cfg.samples.samples_per_class, cfg.samples.seed
    )

    manifest = save_sample_manifest(
        out["dataset"] / "sample_manifest_internal.csv", coords, labels, True
    )
    save_sample_manifest(
        out["dataset"] / "sample_manifest_anonymized.csv", coords, labels, False
    )
    save_wavelengths(out["dataset"] / "wavelengths.csv", wavelengths)
    np.savez_compressed(
        out["dataset"] / "samples.npz",
        coords=coords,
        labels=labels,
        sample_ids=manifest["sample_id"].to_numpy(),
        wavelengths=wavelengths,
    )
    _write_json(
        {
            "image_shape_H_W_C": [h, w, c],
            "positive_candidates": int(len(pos)),
            "negative_candidates": int(len(neg)),
            "selected_positive": int((labels == 1).sum()),
            "selected_negative": int((labels == 0).sum()),
            "all_touched": cfg.samples.all_touched,
        },
        out["dataset"] / "dataset_summary.json",
    )
    return img, coords, labels, wavelengths


def load_samples(cfg: PaperConfig):
    out = output_dirs(cfg)
    npz = out["dataset"] / "samples.npz"
    if not npz.exists():
        return stage_build_samples(cfg)
    img, _, _, _, wavelengths, _ = load_raster(cfg.paths.enmap_raster)
    data = np.load(npz, allow_pickle=True)
    return img, data["coords"], data["labels"], wavelengths


def stage_patch_selection(cfg: PaperConfig):
    out = output_dirs(cfg)
    img, coords, labels, _ = load_samples(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected, df = run_patch_selection(img, labels, coords, cfg, device, out["patch"])
    return selected, df


def load_selected_patch(cfg: PaperConfig) -> int:
    out = output_dirs(cfg)
    path = out["patch"] / "selected_patch_size.json"
    if not path.exists():
        return stage_patch_selection(cfg)[0]
    return int(json.loads(path.read_text(encoding="utf-8"))["selected_patch_size"])


def stage_primary(cfg: PaperConfig):
    out = output_dirs(cfg)
    img, coords, labels, _ = load_samples(cfg)
    patch = load_selected_patch(cfg)
    patches = build_patch_bank(img, coords, patch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    met, pred, curves = run_buffered_loocv(
        patches,
        labels,
        coords,
        cfg,
        device,
        out["primary"],
        tag="primary",
        nested=True,
        record_curves=True,
    )
    aggregate_curves(curves, out["primary"] / "primary_learning_curves.csv")
    return met, pred


def load_primary_fold_params(cfg: PaperConfig) -> dict[int, TrainParams]:
    out = output_dirs(cfg)
    path = out["primary"] / "primary_predictions.csv"
    if not path.exists():
        stage_primary(cfg)
    df = pd.read_csv(path)
    return {
        int(r.fold): TrainParams(
            lr=float(r.lr),
            batch_size=int(r.batch_size),
            weight_decay=float(r.weight_decay),
            epochs=int(r.epochs),
            optimizer=str(r.optimizer),
        )
        for _, r in df.iterrows()
    }


def stage_sensitivity(cfg: PaperConfig):
    out = output_dirs(cfg)
    img, coords, labels, _ = load_samples(cfg)
    patch = load_selected_patch(cfg)
    patches = build_patch_bank(img, coords, patch)
    params_by_fold = load_primary_fold_params(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return run_sensitivity(
        patches, labels, coords, cfg, device, out["sensitivity"], params_by_fold
    )


def stage_ablation(cfg: PaperConfig):
    out = output_dirs(cfg)
    img, coords, labels, _ = load_samples(cfg)
    patch = load_selected_patch(cfg)
    patches = build_patch_bank(img, coords, patch)
    params_by_fold = load_primary_fold_params(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return run_ablation(
        patches, labels, coords, cfg, device, out["ablation"], params_by_fold
    )


def stage_full_scene(cfg: PaperConfig):
    out = output_dirs(cfg)
    img, profile, _, _, _, _ = load_raster(cfg.paths.enmap_raster)
    sample_path = out["dataset"] / "samples.npz"
    if not sample_path.exists():
        stage_build_samples(cfg)
    data = np.load(sample_path, allow_pickle=True)
    coords, labels = data["coords"], data["labels"]
    patch = load_selected_patch(cfg)
    patches = build_patch_bank(img, coords, patch)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = tune_deployment_params(
        patches, labels, coords, cfg, device, out["scene"] / "optuna"
    )
    _write_json(asdict(params), out["scene"] / "deployment_params.json")
    model = train_final_model(patches, labels, params, cfg, device)
    torch.save(model.state_dict(), out["scene"] / "final_mambahsi_state_dict.pt")
    score = full_scene_inference(
        model,
        img,
        patch,
        profile,
        cfg,
        device,
        out["scene"] / "mambahsi_mineralization_score.tif",
    )
    _write_json(
        {
            "selected_patch_size": patch,
            "score_min": float(np.nanmin(score)),
            "score_max": float(np.nanmax(score)),
            "score_mean": float(np.nanmean(score)),
        },
        out["scene"] / "deployment_summary.json",
    )
    return score


def run_all(cfg: PaperConfig):
    out = output_dirs(cfg)
    seed_everything(cfg.training.seed)
    save_environment(out["audit"] / "environment.json")
    _write_json(
        {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": cfg.training.seed,
        },
        out["audit"] / "run_info.json",
    )
    stage_build_samples(cfg)
    stage_patch_selection(cfg)
    stage_primary(cfg)
    stage_sensitivity(cfg)
    stage_ablation(cfg)
    stage_full_scene(cfg)
