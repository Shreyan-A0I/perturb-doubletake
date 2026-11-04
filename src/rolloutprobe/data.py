"""Loading Perturb-seq data into one canonical form.

Every cell gets a condition key:
    "ctrl"          control cells
    "A"             single perturbation of gene A
    "A+B"           double perturbation, genes sorted so the key is unique

Works with the GEARS-processed Norman file (obs["condition"] like "CBL+ctrl",
"CBL+CNN1", "ctrl"; already log-normalized) and with scPerturb-style files
(raw counts, obs["perturbation"]). Expression is returned log-normalized.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

CTRL_TOKENS = {"ctrl", "control", "non-targeting", "nontargeting", "nt", ""}
CONDITION_COLUMNS = ("condition", "perturbation", "perturbation_name", "guide_identity")


@dataclass
class PerturbData:
    X: np.ndarray | sp.csr_matrix      # cells x genes, log-normalized
    genes: list[str]
    cond: np.ndarray                   # canonical key per cell
    raw_labels: dict[str, list[str]] = field(default_factory=dict)  # key -> original labels

    @property
    def n_cells(self) -> int:
        return self.X.shape[0]

    def conditions(self) -> list[str]:
        return sorted(set(self.cond))

    def singles(self) -> list[str]:
        return [c for c in self.conditions() if c != "ctrl" and "+" not in c]

    def doubles(self) -> list[tuple[str, str]]:
        return [tuple(c.split("+")) for c in self.conditions() if c.count("+") == 1]

    def cell_counts(self) -> pd.Series:
        return pd.Series(self.cond).value_counts()


def canonical(label: str, sep: str) -> str | None:
    """Map an original condition label to a canonical key. None if > 2 genes."""
    tokens = [t.strip() for t in str(label).split(sep)]
    genes = sorted(t for t in tokens if t.lower() not in CTRL_TOKENS)
    if not genes:
        return "ctrl"
    if len(genes) > 2:
        return None
    return "+".join(genes)


def _detect_condition_column(adata: ad.AnnData, col: str | None) -> str:
    if col is not None:
        return col
    for c in CONDITION_COLUMNS:
        if c in adata.obs.columns:
            return c
    raise KeyError(f"No condition column found; looked for {CONDITION_COLUMNS}. "
                   f"Set data.condition_col in the config.")


def _detect_sep(labels: pd.Series, sep: str | None) -> str:
    if sep is not None:
        return sep
    return "+" if labels.astype(str).str.contains(r"\+", regex=True).any() else "_"


def _looks_like_counts(X) -> bool:
    vals = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if vals.size == 0:
        return False
    sample = vals[:: max(1, vals.size // 50_000)]
    return bool(np.allclose(sample, np.round(sample)) and sample.max() > 20)


def _normalize_log1p(X, target: float = 1e4):
    if sp.issparse(X):
        X = X.astype(np.float32).tocsr()
        lib = np.asarray(X.sum(axis=1)).ravel()
        lib[lib == 0] = 1.0
        X = (sp.diags((target / lib).astype(np.float32)) @ X).tocsr()
        X.data = np.log1p(X.data)
        return X
    X = np.asarray(X, dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X * (target / lib))


def _top_variable_genes(X, cell_mask: np.ndarray, n: int) -> np.ndarray:
    sub = X[cell_mask]
    if sp.issparse(sub):
        mean = np.asarray(sub.mean(axis=0)).ravel()
        mean_sq = np.asarray(sub.multiply(sub).mean(axis=0)).ravel()
        var = mean_sq - mean ** 2
    else:
        var = np.asarray(sub).var(axis=0)
    return np.sort(np.argsort(-var)[:n])


def load_h5ad(path: str, condition_col: str | None = None, sep: str | None = None,
              normalize: str = "auto", n_top_genes: int | None = None) -> PerturbData:
    adata = ad.read_h5ad(path)
    col = _detect_condition_column(adata, condition_col)
    labels = adata.obs[col].astype(str)
    sep = _detect_sep(labels, sep)

    keys = labels.map(lambda s: canonical(s, sep))
    keep = keys.notna().to_numpy()
    if (~keep).any():
        warnings.warn(f"Dropping {(~keep).sum()} cells with more than two perturbed genes.")
    adata = adata[keep].copy()
    keys = keys[keep].to_numpy()
    labels = labels[keep]

    raw: dict[str, list[str]] = {}
    for k, lab in zip(keys, labels):
        raw.setdefault(k, [])
        if lab not in raw[k]:
            raw[k].append(lab)

    X = adata.X
    if normalize == "yes" or (normalize == "auto" and _looks_like_counts(X)):
        X = _normalize_log1p(X)
    elif sp.issparse(X):
        X = X.astype(np.float32).tocsr()
    else:
        X = np.asarray(X, dtype=np.float32)

    genes = (adata.var["gene_name"].astype(str).tolist() if "gene_name" in adata.var.columns
             else adata.var_names.astype(str).tolist())

    if n_top_genes is not None and n_top_genes < len(genes):
        train_mask = np.array([("+" not in k) for k in keys])   # ctrl + singles only
        idx = _top_variable_genes(X, train_mask, n_top_genes)
        X = X[:, idx]
        genes = [genes[i] for i in idx]

    return PerturbData(X=X, genes=genes, cond=keys, raw_labels=raw)


def pseudobulk(X, groups: np.ndarray) -> dict:
    """Mean expression per group. `groups` is one hashable label per cell."""
    index = pd.Series(np.arange(len(groups))).groupby(pd.Series(list(groups))).indices
    out = {}
    for g, idx in index.items():
        m = X[idx].mean(axis=0)
        out[g] = np.asarray(m).ravel().astype(np.float64)
    return out
