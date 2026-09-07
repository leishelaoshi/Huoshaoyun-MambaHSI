from __future__ import annotations

import os
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig, TrainConfig
from .model import MambaClassifier


@dataclass
class TrainParams:
    lr: float
    batch_size: int
    weight_decay: float
    epochs: int
    optimizer: str = "AdamW"


class PatchDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx].copy()), torch.tensor(int(self.y[idx]))


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics_dict(y_true, y_pred, y_score=None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out = {
        "OA": float(accuracy_score(y_true, y_pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["AUC"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["AUC"] = float("nan")
    return out


def _build_model(in_channels: int, model_cfg: ModelConfig, mamba_type: str, use_att: bool):
    return MambaClassifier(
        in_channels=in_channels,
        mamba_type=mamba_type,
        use_att=use_att,
        hidden_dim=model_cfg.hidden_dim,
        token_num=model_cfg.token_num,
        group_num=model_cfg.group_num,
        d_state=model_cfg.d_state,
        d_conv=model_cfg.d_conv,
        expand=model_cfg.expand,
    )


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: Optional[np.ndarray],
    val_y: Optional[np.ndarray],
    params: TrainParams,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: str,
    seed: int,
    mamba_type: str = "both",
    use_att: bool = True,
    record_curve: bool = False,
):
    seed_everything(seed)
    model = _build_model(train_x.shape[1], model_cfg, mamba_type, use_att).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, params.epochs), eta_min=1e-6
    )
    loader = DataLoader(
        PatchDataset(train_x, train_y),
        batch_size=min(params.batch_size, len(train_y)),
        shuffle=True,
        num_workers=train_cfg.num_workers,
        drop_last=False,
    )

    rows = []
    for epoch in range(params.epochs):
        model.train()
        total_loss = 0.0
        total_n = 0
        total_correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total_correct += int((logits.argmax(1) == yb).sum().item())
            total_n += len(yb)
        scheduler.step()

        if record_curve:
            row = {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_n, 1),
                "train_acc": total_correct / max(total_n, 1),
            }
            if val_x is not None and len(val_x):
                model.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(np.asarray(val_x, dtype=np.float32)).to(device)
                    yv = torch.from_numpy(np.asarray(val_y, dtype=np.int64)).to(device)
                    logits = model(xv)
                    row["val_loss"] = float(criterion(logits, yv).item())
                    row["val_acc"] = float((logits.argmax(1) == yv).float().mean().item())
            rows.append(row)

    return model, pd.DataFrame(rows)


def predict(model: nn.Module, x: np.ndarray, train_cfg: TrainConfig, device: str, batch_size: int = 256):
    model.eval()
    loader = DataLoader(
        PatchDataset(x, np.zeros(len(x), dtype=np.int64)),
        batch_size=min(batch_size, max(1, len(x))),
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )
    logits = []
    with torch.no_grad():
        for xb, _ in loader:
            logits.append(model(xb.to(device)).cpu())
    logits = torch.cat(logits, dim=0)
    score = torch.softmax(logits, dim=1)[:, 1].numpy()
    return logits.numpy(), score


def save_environment(path: str | Path) -> None:
    try:
        import importlib.metadata as metadata
        version = lambda name: metadata.version(name)
    except Exception:
        version = lambda name: "unknown"

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": version("scikit-learn"),
        "optuna": version("optuna"),
        "mamba-ssm": version("mamba-ssm"),
        "rasterio": version("rasterio"),
        "geopandas": version("geopandas"),
    }
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
