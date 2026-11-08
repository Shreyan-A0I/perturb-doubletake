"""Stability: apply the same perturbation repeatedly, x_{k+1} = predict(x_k, (A,)).

Consistency check only (see rollout.py): Norman cannot say what a second dose of
guide A should do. Biologically the expectation is saturation (A cells already have
A knocked down), so step ratios near 0 after the first step are the plausible
behaviour, and a constant ratio of 1 means the model re-adds a fixed vector.
"""
from __future__ import annotations

import numpy as np


def repeat_trajectory(model, X0: np.ndarray, a: str, k_max: int, keep_cells: tuple[int, ...] = ()):
    """Mean profile after 0..k_max steps (cells are carried, not means), plus the
    predicted cells at the steps listed in keep_cells."""
    X = np.asarray(X0)
    means, kept = [X.mean(axis=0)], {}
    for k in range(1, k_max + 1):
        X = np.asarray(model.predict(X, (a,)))
        means.append(X.mean(axis=0))
        if k in keep_cells:
            kept[k] = X
    return np.stack(means), kept


def step_ratios(means: np.ndarray) -> np.ndarray:
    """||x_k - x_{k-1}|| / ||x_1 - x_0|| for k = 1..K (the first is 1 by definition)."""
    steps = np.linalg.norm(np.diff(means, axis=0), axis=1)
    return steps / steps[0] if steps[0] > 0 else np.full(len(steps), np.nan)


def nearest_condition(x: np.ndarray, profiles: dict) -> tuple[str, float]:
    names = list(profiles)
    d = np.array([np.linalg.norm(x - profiles[c]) for c in names])
    i = int(np.argmin(d))
    return names[i], float(d[i])
