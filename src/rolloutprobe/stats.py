"""Uncertainty across perturbation pairs (reviewer point 3).

Metrics are first averaged per pair across split seeds, then every interval is a
bootstrap over PAIRS, never over cells. The pair is the unit of inference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"mse_20deg"}


def per_pair(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rows: pairs. Columns: methods. Averaged over split seeds."""
    key = df["pert_a"] + "+" + df["pert_b"]
    return (df.assign(pair=key)
              .groupby(["pair", "method"])[metric].mean()
              .unstack("method"))


def bootstrap_mean(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)), "n": int(values.size)}


def method_summary(df: pd.DataFrame, metric: str, n_boot: int, seed: int = 0) -> pd.DataFrame:
    table = per_pair(df, metric)
    rows = [{"method": m, "metric": metric, **bootstrap_mean(table[m].to_numpy(), n_boot, seed)}
            for m in table.columns]
    return pd.DataFrame(rows)


def advantage(df: pd.DataFrame, metric: str, method: str, ref: str = "additive") -> pd.Series:
    """Per-pair advantage of `method` over `ref`; positive always means `method` is better."""
    t = per_pair(df, metric)
    adv = t[ref] - t[method] if metric in LOWER_IS_BETTER else t[method] - t[ref]
    return adv.rename(f"adv_{method}_vs_{ref}")


def slope_vs_eta(adv: pd.Series, eta: pd.Series, n_boot: int, seed: int = 0) -> dict:
    """OLS slope of advantage on eta (half 1), bootstrap CI over pairs. A positive
    slope means the method's edge over addition grows where pairs are less additive."""
    d = pd.concat([adv, eta], axis=1).dropna()
    x, y = d.iloc[:, 1].to_numpy(), d.iloc[:, 0].to_numpy()

    def fit(xx, yy):
        xc = xx - xx.mean()
        den = xc @ xc
        return float(xc @ (yy - yy.mean()) / den) if den > 0 else np.nan

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boots.append(fit(x[i], y[i]))
    boots = np.asarray(boots)
    return {"slope": fit(x, y), "lo": float(np.nanquantile(boots, 0.025)),
            "hi": float(np.nanquantile(boots, 0.975)), "n": int(len(x))}
