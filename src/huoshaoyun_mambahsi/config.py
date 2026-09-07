from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class PathsConfig:
    enmap_raster: str
    positive_roi: str
    negative_roi: str
    output_dir: str


@dataclass
class InputConfig:
    expected_bands: int = 217


@dataclass
class SampleConfig:
    samples_per_class: int = 46
    all_touched: bool = False
    seed: int = 42


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    token_num: int = 4
    group_num: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    patch_candidates: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    patch_selection_metric: str = "OA"


@dataclass
class TrainConfig:
    seed: int = 42
    outer_buffer_radius_pixels: int = 6
    inner_splits: int = 3
    inner_spatial_block_pixels: int = 6
    optuna_trials_per_outer_fold: int = 10
    deployment_optuna_trials: int = 10
    learning_rate_min: float = 1e-5
    learning_rate_max: float = 1e-3
    batch_size_candidates: List[int] = field(default_factory=lambda: [4, 8, 16])
    weight_decay_min: float = 1e-5
    weight_decay_max: float = 1e-3
    tuning_epochs: int = 20
    training_epochs: int = 100
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    loss: str = "CrossEntropyLoss"
    gradient_clip_norm: float = 5.0
    threshold: float = 0.5
    num_workers: int = 0
    inference_batch_size: int = 1024


@dataclass
class ExperimentConfig:
    sensitivity_fractions: List[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.75, 1.00]
    )
    sensitivity_repeats: int = 3
    geometric_augmentation: str = "none"


@dataclass
class PaperConfig:
    paths: PathsConfig
    input: InputConfig
    samples: SampleConfig
    model: ModelConfig
    training: TrainConfig
    experiments: ExperimentConfig


def _resolve_path(value: Optional[str], root: Path) -> Optional[str]:
    if value is None or value == "":
        return value
    p = Path(value)
    return str(p if p.is_absolute() else (root / p).resolve())


def load_config(path: str | Path) -> PaperConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    root = path.parent.parent if path.parent.name == "configs" else path.parent
    paths = PathsConfig(**raw["paths"])
    paths.enmap_raster = _resolve_path(paths.enmap_raster, root)
    paths.positive_roi = _resolve_path(paths.positive_roi, root)
    paths.negative_roi = _resolve_path(paths.negative_roi, root)
    paths.output_dir = _resolve_path(paths.output_dir, root)

    cfg = PaperConfig(
        paths=paths,
        input=InputConfig(**raw.get("input", {})),
        samples=SampleConfig(**raw.get("samples", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainConfig(**raw.get("training", {})),
        experiments=ExperimentConfig(**raw.get("experiments", {})),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: PaperConfig) -> None:
    if cfg.input.expected_bands < 1:
        raise ValueError("input.expected_bands must be positive")
    if any(p % 2 == 0 or p < 1 for p in cfg.model.patch_candidates):
        raise ValueError("All patch candidates must be positive odd integers")
    if cfg.model.patch_selection_metric not in {"OA", "Kappa", "F1"}:
        raise ValueError("patch_selection_metric must be OA, Kappa, or F1")
    if cfg.experiments.geometric_augmentation != "none":
        raise ValueError("The paper protocol uses no geometric augmentation")
