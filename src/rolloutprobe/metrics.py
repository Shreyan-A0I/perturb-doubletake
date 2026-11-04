"""Accuracy metrics. DEG indices come from half 1, scoring from half 2."""
from __future__ import annotations

import numpy as np


def top_k_deg(delta_ref: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.abs(delta_ref))[:k]


def mse(pred: np.ndarray, target: np.ndarray, idx: np.ndarray | None = None) -> float:
    if idx is not None:
        pred, target = pred[idx], target[idx]
    return float(np.mean((pred - target) ** 2))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / den) if den > 0 else np.nan


def pearson_delta(pred: np.ndarray, target: np.ndarray, ctrl: np.ndarray,
                  idx: np.ndarray | None = None) -> float:
    """Correlation of predicted vs observed shifts from control. Undefined (nan)
    for the no-change null, whose predicted shift is identically zero."""
    dp, dt = pred - ctrl, target - ctrl
    if idx is not None:
        dp, dt = dp[idx], dt[idx]
    return pearson(dp, dt)


def score(pred: np.ndarray, target: np.ndarray, ctrl: np.ndarray, deg_idx: np.ndarray) -> dict:
    return {
        "mse_20deg": mse(pred, target, deg_idx),
        "pearson_delta": pearson_delta(pred, target, ctrl),
        "pearson_delta_20deg": pearson_delta(pred, target, ctrl, deg_idx),
    }
