"""The null hierarchy, all built from HALF 2 pseudobulks of training conditions.

    nochange          predict control
    mean              control + average single-perturbation shift
    additive          control + Delta_A + Delta_B
    lowrank_additive  control + low-rank denoised Delta_A + Delta_B

Why no fitted interaction term: under the strict split there are no doubles in
training, so a linear model fit on singles can only reproduce (a denoised version
of) addition. The low-rank variant is the strict-split analogue of the linear
baseline in Ahlmann-Eltze & Huber (2025): it tests whether denoising the singles
alone beats raw addition.
"""
from __future__ import annotations

import numpy as np

METHODS = ("nochange", "mean", "additive", "lowrank_additive")


def single_delta_matrix(pb: dict, singles: list[str], half: int) -> np.ndarray:
    c = pb[("ctrl", half)]
    return np.stack([pb[(s, half)] - c for s in singles])


def lowrank(D: np.ndarray, k: int) -> np.ndarray:
    k = min(k, min(D.shape))
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def predict_nulls(pb: dict, pairs: list[tuple[str, str]], train_singles: list[str],
                  half: int = 2, k: int = 10) -> dict[str, np.ndarray]:
    """Returns {method: (n_pairs, n_genes) predicted double expression}."""
    c = pb[("ctrl", half)]
    D = single_delta_matrix(pb, train_singles, half)
    Dk = lowrank(D, k)
    pos = {s: i for i, s in enumerate(train_singles)}
    mean_shift = D.mean(axis=0)

    out = {m: [] for m in METHODS}
    for a, b in pairs:
        ia, ib = pos[a], pos[b]
        out["nochange"].append(c)
        out["mean"].append(c + mean_shift)
        out["additive"].append(c + D[ia] + D[ib])
        out["lowrank_additive"].append(c + Dk[ia] + Dk[ib])
    return {m: np.stack(v) for m, v in out.items()}
