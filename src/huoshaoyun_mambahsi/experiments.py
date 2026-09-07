from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from .config import PaperConfig
from .sampling import build_patch_bank, extract_patch
from .splits import outer_train_indices
from .training import (
    PatchDataset,
    TrainParams,
    metrics_dict,
    predict,
    train_model,
)
from .tuning import nested_optuna


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run_buffered_loocv(
    patches: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    cfg: PaperConfig,
    device: str,
    out_dir: str | Path,
    tag: str,
    nested: bool = True,
    fixed_params: Optional[TrainParams] = None,
    params_by_fold: Optional[dict[int, TrainParams]] = None,
    mamba_type: str = "both",
    use_att: bool = True,
    record_curves: bool = False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []

    for val_idx in range(len(labels)):
        train_idx, removed = outer_train_indices(
            coords, val_idx, cfg.training.outer_buffer_radius_pixels
        )
        if len(np.unique(labels[train_idx])) < 2:
            continue

        if nested:
            params = nested_optuna(
                patches[train_idx],
                labels[train_idx],
                coords[train_idx],
                cfg.model,
                cfg.training,
                device,
                cfg.training.seed + val_idx,
                out_dir / "optuna",
                f"{tag}_fold_{val_idx:03d}",
                cfg.training.optuna_trials_per_outer_fold,
            )
        else:
            if params_by_fold is not None:
                if val_idx not in params_by_fold:
                    raise KeyError(f"Missing parameters for outer fold {val_idx}")
                params = TrainParams(**asdict(params_by_fold[val_idx]))
            elif fixed_params is not None:
                params = TrainParams(**asdict(fixed_params))
            else:
                raise ValueError("fixed_params or params_by_fold is required when nested=False")

        model, curve = train_model(
            patches[train_idx],
            labels[train_idx],
            patches[[val_idx]],
            labels[[val_idx]],
            params,
            cfg.model,
            cfg.training,
            device,
            cfg.training.seed + val_idx,
            mamba_type=mamba_type,
            use_att=use_att,
            record_curve=record_curves,
        )
        _, prob = predict(model, patches[[val_idx]], cfg.training, device, 1)
        pred = int(prob[0] >= cfg.training.threshold)
        rows.append(
            {
                "fold": val_idx,
                "true": int(labels[val_idx]),
                "pred": pred,
                "score": float(prob[0]),
                "n_train_after_buffer": int(len(train_idx)),
                "n_removed_by_buffer": int(len(removed)),
                **asdict(params),
            }
        )
        if record_curves and not curve.empty:
            curve["fold"] = val_idx
            curves.append(curve)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise RuntimeError(f"No valid outer folds for {tag}")
    met = metrics_dict(pred_df.true, pred_df.pred, pred_df.score)
    pred_df.to_csv(out_dir / f"{tag}_predictions.csv", index=False)
    _save_json(met, out_dir / f"{tag}_metrics.json")
    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return met, pred_df, curve_df


def run_patch_selection(
    img: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    cfg: PaperConfig,
    device: str,
    out_dir: str | Path,
):
    out_dir = Path(out_dir)
    rows = []
    for p in cfg.model.patch_candidates:
        patches = build_patch_bank(img, coords, p)
        met, _, _ = run_buffered_loocv(
            patches,
            labels,
            coords,
            cfg,
            device,
            out_dir / f"patch_{p}",
            tag=f"patch_{p}",
            nested=True,
        )
        rows.append({"patch_size": p, **met})

    df = pd.DataFrame(rows)
    metric = cfg.model.patch_selection_metric
    best = df.sort_values(
        [metric, "Kappa", "F1", "patch_size"],
        ascending=[False, False, False, True],
    ).iloc[0]
    selected = int(best.patch_size)
    df.to_csv(out_dir / "patch_size_comparison.csv", index=False)
    _save_json(
        {"selected_patch_size": selected, "selection_metric": metric},
        out_dir / "selected_patch_size.json",
    )
    return selected, df


def aggregate_curves(curves: pd.DataFrame, out_path: str | Path) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()
    agg = curves.groupby("epoch").agg(
        train_loss_mean=("train_loss", "mean"),
        train_loss_std=("train_loss", "std"),
        train_acc_mean=("train_acc", "mean"),
        train_acc_std=("train_acc", "std"),
        val_loss_mean=("val_loss", "mean"),
        val_loss_std=("val_loss", "std"),
        val_acc_mean=("val_acc", "mean"),
        val_acc_std=("val_acc", "std"),
    ).reset_index()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    return agg


def _stratified_subsample(indices, labels, fraction, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for cls in (0, 1):
        cls_idx = indices[labels[indices] == cls]
        n = max(1, int(round(len(cls_idx) * fraction)))
        selected.extend(rng.choice(cls_idx, min(n, len(cls_idx)), replace=False).tolist())
    return np.asarray(selected, dtype=int)


def run_sensitivity(
    patches: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    cfg: PaperConfig,
    device: str,
    out_dir: str | Path,
    params_by_fold: dict[int, TrainParams],
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = []

    for frac in cfg.experiments.sensitivity_fractions:
        for rep in range(cfg.experiments.sensitivity_repeats):
            y_true, y_pred, y_score, n_train = [], [], [], []
            rep_seed = cfg.training.seed + 5000 + rep * 100
            for val_idx in range(len(labels)):
                train_idx, _ = outer_train_indices(
                    coords, val_idx, cfg.training.outer_buffer_radius_pixels
                )
                sub = _stratified_subsample(train_idx, labels, frac, rep_seed + val_idx)
                if len(np.unique(labels[sub])) < 2:
                    continue
                model, _ = train_model(
                    patches[sub],
                    labels[sub],
                    patches[[val_idx]],
                    labels[[val_idx]],
                    params_by_fold[val_idx],
                    cfg.model,
                    cfg.training,
                    device,
                    rep_seed + val_idx,
                    record_curve=False,
                )
                _, prob = predict(model, patches[[val_idx]], cfg.training, device, 1)
                y_true.append(int(labels[val_idx]))
                y_pred.append(int(prob[0] >= cfg.training.threshold))
                y_score.append(float(prob[0]))
                n_train.append(len(sub))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            met = metrics_dict(y_true, y_pred, y_score)
            raw.append(
                {
                    "fraction": frac,
                    "repeat": rep + 1,
                    "mean_n_train": float(np.mean(n_train)),
                    **met,
                }
            )

    raw_df = pd.DataFrame(raw)
    raw_df.to_csv(out_dir / "sensitivity_raw.csv", index=False)
    rows = []
    for frac, sub in raw_df.groupby("fraction"):
        row = {"fraction": frac, "n_repeats": len(sub)}
        for m in ("OA", "Recall", "F1", "Kappa", "mean_n_train"):
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_std"] = float(sub[m].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    return raw_df, summary


def run_ablation(
    patches: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    cfg: PaperConfig,
    device: str,
    out_dir: str | Path,
    params_by_fold: dict[int, TrainParams],
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = {
        "Spectral-only": ("spe", False),
        "Spatial-only": ("spa", False),
        "Spatial + spectral": ("both", False),
        "Full MambaHSI": ("both", True),
    }
    rows = []
    for name, (mamba_type, use_att) in variants.items():
        tag = name.lower().replace(" ", "_").replace("+", "plus")
        met, _, _ = run_buffered_loocv(
            patches,
            labels,
            coords,
            cfg,
            device,
            out_dir / tag,
            tag=tag,
            nested=False,
            params_by_fold=params_by_fold,
            mamba_type=mamba_type,
            use_att=use_att,
        )
        rows.append({"variant": name, **met})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "module_ablation.csv", index=False)
    return df


class SceneDataset(Dataset):
    def __init__(self, img: np.ndarray, patch_size: int):
        self.img = img
        self.patch_size = patch_size
        self.h, self.w, _ = img.shape

    def __len__(self):
        return self.h * self.w

    def __getitem__(self, idx: int):
        r, c = divmod(idx, self.w)
        return torch.from_numpy(extract_patch(self.img, r, c, self.patch_size)), idx


def tune_deployment_params(
    patches: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    cfg: PaperConfig,
    device: str,
    out_dir: str | Path,
) -> TrainParams:
    return nested_optuna(
        patches,
        labels,
        coords,
        cfg.model,
        cfg.training,
        device,
        cfg.training.seed + 90000,
        out_dir,
        "deployment",
        cfg.training.deployment_optuna_trials,
    )


def train_final_model(
    patches: np.ndarray,
    labels: np.ndarray,
    params: TrainParams,
    cfg: PaperConfig,
    device: str,
):
    model, _ = train_model(
        patches,
        labels,
        None,
        None,
        params,
        cfg.model,
        cfg.training,
        device,
        cfg.training.seed + 9999,
        record_curve=False,
    )
    return model


def full_scene_inference(
    model,
    img: np.ndarray,
    patch_size: int,
    profile: dict,
    cfg: PaperConfig,
    device: str,
    out_path: str | Path,
):
    ds = SceneDataset(img, patch_size)
    loader = DataLoader(
        ds,
        batch_size=cfg.training.inference_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    score = np.zeros(len(ds), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for xb, idx in loader:
            prob = torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy()
            score[idx.numpy()] = prob.astype(np.float32)
    score = score.reshape(ds.h, ds.w)

    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, count=1, nodata=None)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(score, 1)
    return score
