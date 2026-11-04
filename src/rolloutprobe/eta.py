"""Departure from additivity (eta) and its split-half reliability.

For a pair (A, B) and cell half h, with Delta = pseudobulk shift from control:

    r_h   = Delta_AB - (Delta_A + Delta_B)       residual vector
    eta_h = ||r_h||

eta_h is biased upward by noise: even a truly additive pair has eta > 0. Because
halves 1 and 2 have independent noise, the cross product r_1 . r_2 is an
UNBIASED estimate of ||r_true||^2. Comparing it with eta_h^2 tells us what
fraction of measured non-additivity is real signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def delta(pb: dict, cond: str, half: int) -> np.ndarray:
    return pb[(cond, half)] - pb[("ctrl", half)]


def residual(pb: dict, a: str, b: str, half: int) -> np.ndarray:
    return delta(pb, f"{a}+{b}", half) - delta(pb, a, half) - delta(pb, b, half)


def eta_table(pb: dict, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        r1, r2 = residual(pb, a, b, 1), residual(pb, a, b, 2)
        n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
        d1 = np.linalg.norm(delta(pb, f"{a}+{b}", 1))
        rows.append({
            "pert_a": a, "pert_b": b,
            "eta_h1": n1, "eta_h2": n2,
            "eta2_cross": float(r1 @ r2),
            "cos_h1_h2": float(r1 @ r2 / (n1 * n2)) if n1 > 0 and n2 > 0 else np.nan,
            "double_norm_h1": d1,
            "rel_eta_h1": n1 / d1 if d1 > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def reliability(df: pd.DataFrame) -> dict:
    """Summaries that say whether eta is a property of the pair or of the noise."""
    sp = stats.spearmanr(df["eta_h1"], df["eta_h2"])
    pr = stats.pearsonr(df["eta_h1"], df["eta_h2"])
    mean_sq = ((df["eta_h1"] ** 2 + df["eta_h2"] ** 2) / 2).mean()
    return {
        "n_pairs": int(len(df)),
        "spearman_eta_h1_h2": float(sp.statistic),
        "pearson_eta_h1_h2": float(pr.statistic),
        "median_cos_residuals": float(df["cos_h1_h2"].median()),
        "frac_pairs_positive_cross": float((df["eta2_cross"] > 0).mean()),
        # share of measured eta^2 that is signal rather than noise, pooled over pairs
        "signal_fraction_eta2": float(df["eta2_cross"].mean() / mean_sq) if mean_sq > 0 else np.nan,
    }


def assign_strata(eta: pd.Series, n: int) -> pd.Series:
    labels = ["low", "mid", "high"] if n == 3 else [f"q{i + 1}" for i in range(n)]
    return pd.qcut(eta, q=n, labels=labels)
