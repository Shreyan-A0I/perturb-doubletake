"""The results contract between the two workstreams.

One row per (model, method, arm, order, pair, split_seed). Every script that
produces predictions writes this shape; every figure and test reads it. Build
figures against `fake_results()` before any model runs, then swap in real files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESULT_COLUMNS = [
    "model",                # "none" for nulls | "gears_strict" | "gears_standard" | "scgpt"
    "method",               # nochange | mean | additive | lowrank_additive | joint | rollout | rollout_measured
    "arm",                  # "carried" (label B only) | "restated" (label A+B) | "na"
    "order",                # "ab" | "ba" | "na"
    "pert_a", "pert_b",
    "support",              # seen2 | seen1 | seen0
    "split_seed",
    "eta_h1",               # stratification variable, measured on half 1 only
    "mse_20deg", "pearson_delta", "pearson_delta_20deg",
]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RESULT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Results missing columns: {missing}")
    return df[RESULT_COLUMNS]


def fake_results(n_pairs: int = 131, n_seeds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Random results with the real schema, for building figures before models run."""
    rng = np.random.default_rng(seed)
    pairs = [(f"G{i:03d}", f"G{i + 1:03d}") for i in range(0, 2 * n_pairs, 2)]
    eta = rng.gamma(2.0, 1.0, n_pairs)
    rows = []
    specs = [("none", m, "na", "na") for m in ("nochange", "mean", "additive", "lowrank_additive")]
    specs += [("gears_strict", "joint", "na", "na")]
    specs += [("gears_strict", "rollout", arm, o) for arm in ("carried", "restated") for o in ("ab", "ba")]
    for s in range(n_seeds):
        for (a, b), e in zip(pairs, eta):
            for model, method, arm, order in specs:
                rows.append({"model": model, "method": method, "arm": arm, "order": order,
                             "pert_a": a, "pert_b": b, "support": "seen2", "split_seed": s,
                             "eta_h1": e, "mse_20deg": abs(rng.normal(0.1 + 0.05 * e, 0.03)),
                             "pearson_delta": rng.uniform(0.2, 0.9),
                             "pearson_delta_20deg": rng.uniform(0.2, 0.9)})
    return pd.DataFrame(rows)[RESULT_COLUMNS]
