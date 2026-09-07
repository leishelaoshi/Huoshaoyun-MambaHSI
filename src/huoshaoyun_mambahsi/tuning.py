from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold

from .config import ModelConfig, TrainConfig
from .splits import buffered_train_indices, spatial_group_ids
from .training import TrainParams, predict, train_model


def nested_optuna(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_coords: np.ndarray,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: str,
    seed: int,
    out_dir: str | Path,
    tag: str,
    trial_budget: int,
) -> TrainParams:
    import optuna

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = spatial_group_ids(train_coords, train_cfg.inner_spatial_block_pixels)
    n_splits = min(train_cfg.inner_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError(f"{tag}: fewer than two spatial groups for inner CV")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    audit = []
    for fold, (candidate_tr, va) in enumerate(cv.split(train_x, train_y, groups=groups)):
        tr, removed = buffered_train_indices(
            train_coords,
            candidate_tr,
            va,
            train_cfg.outer_buffer_radius_pixels,
        )
        if (
            len(tr) == 0
            or len(np.unique(train_y[tr])) < 2
            or len(np.unique(train_y[va])) < 2
        ):
            continue
        folds.append((fold, tr, va))
        audit.append(
            {
                "inner_fold": fold,
                "n_train_after_buffer": len(tr),
                "n_validation": len(va),
                "n_removed_by_buffer": len(removed),
            }
        )

    if len(folds) < 2:
        raise RuntimeError(f"{tag}: fewer than two valid inner folds")
    pd.DataFrame(audit).to_csv(out_dir / f"{tag}_inner_folds.csv", index=False)

    def objective(trial):
        params = TrainParams(
            lr=trial.suggest_float(
                "lr", train_cfg.learning_rate_min, train_cfg.learning_rate_max, log=True
            ),
            batch_size=trial.suggest_categorical(
                "batch_size", list(train_cfg.batch_size_candidates)
            ),
            weight_decay=trial.suggest_float(
                "weight_decay",
                train_cfg.weight_decay_min,
                train_cfg.weight_decay_max,
                log=True,
            ),
            epochs=train_cfg.tuning_epochs,
            optimizer=train_cfg.optimizer,
        )
        scores = []
        for fold, tr, va in folds:
            model, _ = train_model(
                train_x[tr],
                train_y[tr],
                train_x[va],
                train_y[va],
                params,
                model_cfg,
                train_cfg,
                device,
                seed + fold,
                record_curve=False,
            )
            _, prob = predict(model, train_x[va], train_cfg, device, batch_size=64)
            pred = (prob >= train_cfg.threshold).astype(int)
            scores.append(cohen_kappa_score(train_y[va], pred))
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trial_budget, show_progress_bar=False)
    study.trials_dataframe().to_csv(out_dir / f"{tag}_trials.csv", index=False)

    best = study.best_params
    info = {
        "best_inner_kappa": float(study.best_value),
        "lr": float(best["lr"]),
        "batch_size": int(best["batch_size"]),
        "weight_decay": float(best["weight_decay"]),
        "trial_budget": int(trial_budget),
        "tuning_epochs": int(train_cfg.tuning_epochs),
    }
    (out_dir / f"{tag}_best.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    return TrainParams(
        lr=info["lr"],
        batch_size=info["batch_size"],
        weight_decay=info["weight_decay"],
        epochs=train_cfg.training_epochs,
        optimizer=train_cfg.optimizer,
    )
