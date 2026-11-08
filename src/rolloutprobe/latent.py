"""Part B: is the measured interaction still present inside a model?

For each condition c and cell half h, embed real cells and average the embeddings
over cells: z(c, h). The latent residual for a pair mirrors eta in gene space:

    r_h = z(AB, h) - z(A, h) - z(B, h) + z(ctrl, h)      d_h = z(AB, h) - z(ctrl, h)

Raw relative residual ||r_2|| / ||d_2|| is inflated by noise, and noise levels differ
between spaces, so the main quantity is the cross-half estimate (as in eta.py):

    rel_cross = sqrt(max(r_1 . r_2, 0) / (d_1 . d_2))

r_1 . r_2 is an unbiased estimate of ||r_true||^2 and d_1 . d_2 of ||d_true||^2, so
rel_cross is a noise-corrected, scale-free "share of the double's shift that is
non-additive" in that space. Gene space is one more view (the mean expression).
pooled_rel_cross() pools over pairs, sum(r_1 . r_2) / sum(d_1 . d_2), which is stable
where single pairs have r_1 . r_2 <= 0.

An embedding function maps a block of cells (n, genes) to {view: mean over cells}.
Means of embeddings, not embeddings of means: each cell is encoded on its own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

EmbedMean = Callable[[np.ndarray], dict]


def needed_conditions(pairs) -> list[str]:
    conds = {"ctrl"}
    for a, b in pairs:
        conds |= {a, b, f"{a}+{b}"}
    return sorted(conds)


class LatentStore:
    """{(condition, half): {view: mean embedding}}, in memory or cached on disk
    (one .npz per condition and half; flattened scGPT token states are ~10 MB each)."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.mem: dict = {}

    def _path(self, cond, h):
        return self.cache / f"{cond.replace('+', '__')}_h{h}.npz"

    def has(self, cond, h) -> bool:
        return (cond, h) in self.mem or (self.cache is not None and self._path(cond, h).exists())

    def put(self, cond, h, views: dict):
        if self.cache:
            np.savez(self._path(cond, h), **views)
        else:
            self.mem[(cond, h)] = views

    def get(self, cond, h) -> dict:
        if (cond, h) in self.mem:
            return self.mem[(cond, h)]
        with np.load(self._path(cond, h)) as f:
            return {k: f[k] for k in f.files}


def fill_store(store: LatentStore, X, cond: np.ndarray, halves: np.ndarray, conds: list[str],
               embed_mean: EmbedMean, max_cells: int, rng, log=print) -> None:
    """Embed up to max_cells real cells per (condition, half) not already in the store."""
    for i, c in enumerate(conds):
        for h in (1, 2):
            if store.has(c, h):
                continue
            idx = np.flatnonzero((cond == c) & (halves == h))
            if len(idx) == 0:
                raise ValueError(f"no cells for {c} in half {h}")
            idx = np.sort(rng.choice(idx, min(max_cells, len(idx)), replace=False))
            sub = X[idx]
            sub = np.asarray(sub.toarray() if hasattr(sub, "toarray") else sub, dtype=np.float32)
            store.put(c, h, {k: np.asarray(v, dtype=np.float32) for k, v in embed_mean(sub).items()})
        if log and (i + 1) % 20 == 0:
            log(f"  embedded {i + 1}/{len(conds)} conditions")


def pair_terms(z: Callable[[str, int], dict], a: str, b: str) -> dict:
    """Per view: the cross products and norms the summaries need."""
    zs = {(c, h): z(c, h) for c in ("ctrl", a, b, f"{a}+{b}") for h in (1, 2)}
    out = {}
    for v in zs[("ctrl", 1)]:
        g = {k: np.ravel(zz[v]).astype(np.float64) for k, zz in zs.items()}
        r = {h: g[(f"{a}+{b}", h)] - g[(a, h)] - g[(b, h)] + g[("ctrl", h)] for h in (1, 2)}
        d = {h: g[(f"{a}+{b}", h)] - g[("ctrl", h)] for h in (1, 2)}
        rr, dd = float(r[1] @ r[2]), float(d[1] @ d[2])
        n1, n2 = np.linalg.norm(r[1]), np.linalg.norm(r[2])
        out[v] = {"rr_cross": rr, "dd_cross": dd,
                  "rel_h2": float(n2 / np.linalg.norm(d[2])) if np.linalg.norm(d[2]) > 0 else np.nan,
                  "rel_cross": float(np.sqrt(max(rr, 0.0) / dd)) if dd > 0 else np.nan,
                  "rep_cos": float(rr / (n1 * n2)) if n1 > 0 and n2 > 0 else np.nan}
    return out


def pooled_rel_cross(rr: np.ndarray, dd: np.ndarray) -> float:
    s_d = float(np.sum(dd))
    return float(np.sqrt(max(float(np.sum(rr)), 0.0) / s_d)) if s_d > 0 else np.nan


def pooled_ratio_boot(rr_v, dd_v, rr_ref, dd_ref, n_boot: int, seed: int = 0) -> dict:
    """pooled_rel_cross(view) / pooled_rel_cross(reference), bootstrap over pairs."""
    rr_v, dd_v, rr_ref, dd_ref = map(np.asarray, (rr_v, dd_v, rr_ref, dd_ref))

    def ratio(i):
        den = pooled_rel_cross(rr_ref[i], dd_ref[i])
        return pooled_rel_cross(rr_v[i], dd_v[i]) / den if den and den > 0 else np.nan

    n = len(rr_v)
    rng = np.random.default_rng(seed)
    boots = np.array([ratio(rng.integers(0, n, n)) for _ in range(n_boot)])
    return {"mean": ratio(np.arange(n)), "lo": float(np.nanquantile(boots, 0.025)),
            "hi": float(np.nanquantile(boots, 0.975)), "n": int(n)}
