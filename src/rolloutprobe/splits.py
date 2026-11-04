"""Condition splits and the split-half cell design.

The split-half rule (reviewer point 3), applied everywhere:

    Half 1  is MEASUREMENT ONLY. It defines eta strata and picks the top-k DEGs.
    Half 2  is EVERYTHING ELSE. Baseline inputs and scoring targets come from it.

So the noise that puts a pair in a high-eta stratum never also appears in the
target it is scored against.
"""
from __future__ import annotations

import pickle

import numpy as np

from .data import PerturbData, canonical, pseudobulk


def strict_split(data: PerturbData) -> dict:
    """All singles train, all doubles test. Doubles missing a single are dropped."""
    singles = set(data.singles())
    test, dropped = [], []
    for a, b in data.doubles():
        (test if (a in singles and b in singles) else dropped).append((a, b))
    return {"train": ["ctrl", *sorted(singles)], "test": sorted(test), "dropped": sorted(dropped)}


def support_tier(pair: tuple[str, str], train_singles: set[str]) -> str:
    n = sum(g in train_singles for g in pair)
    return f"seen{n}"


def half_labels(cond: np.ndarray, seed: int) -> np.ndarray:
    """Randomly assign each condition's cells to half 1 or 2, near 50/50 per condition."""
    rng = np.random.default_rng(seed)
    halves = np.zeros(len(cond), dtype=np.int8)
    for c in np.unique(cond):
        idx = np.flatnonzero(cond == c)
        idx = rng.permutation(idx)
        cut = len(idx) // 2
        halves[idx[:cut]] = 1
        halves[idx[cut:]] = 2
    return halves


def pseudobulk_by_half(data: PerturbData, halves: np.ndarray) -> dict:
    """Returns {(condition, half): mean expression vector}."""
    labels = np.array([f"{c}|{h}" for c, h in zip(data.cond, halves)])
    pb = pseudobulk(data.X, labels)
    return {(k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1])): v for k, v in pb.items()}


def gears_custom_split(data: PerturbData, split: dict, val_frac: float = 0.1,
                       seed: int = 0) -> tuple[dict, list[str]]:
    """GEARS needs a validation set of conditions. We take it from singles that
    appear in NO held-out double, so every test double stays seen2. Only if there
    are too few of those do we fall back to singles that do appear in doubles,
    and we return those so their doubles can be marked seen1."""
    rng = np.random.default_rng(seed)
    singles = [s for s in split["train"] if s != "ctrl"]
    in_doubles = {g for pair in split["test"] for g in pair}
    free = [s for s in singles if s not in in_doubles]
    n_val = max(1, int(round(val_frac * len(singles))))

    val = list(rng.permutation(free)[:n_val])
    demoted: list[str] = []
    if len(val) < n_val:
        rest = [s for s in singles if s in in_doubles]
        extra = list(rng.permutation(rest)[: n_val - len(val)])
        val += extra
        demoted = extra

    def raw(keys):
        out = []
        for k in keys:
            out += data.raw_labels.get(k, [])
        return out

    train_keys = ["ctrl"] + [s for s in singles if s not in val]
    test_keys = [f"{a}+{b}" for a, b in split["test"]]
    return {"train": raw(train_keys), "val": raw(val), "test": raw(test_keys)}, demoted


def canonical_key(label: str) -> str | None:
    """Canonical key for a GEARS-style label such as "CBL+ctrl" or "CNN1+CBL"."""
    return canonical(label, "+")


def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
