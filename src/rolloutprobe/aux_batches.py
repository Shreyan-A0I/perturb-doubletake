"""Part C: auxiliary training rows from real single-perturbed cells (singles only).

The main scGPT objective only ever starts from control cells. Two extra kinds of
rows give it a reason to read the input state, using training singles only, so
the strict split holds (no double is ever seen):

    state       input a real A cell, no flag     -> target an A cell
    saturation  input a real A cell, flag on A   -> target an A cell

Targets are a random *other* cell of the same condition, matching the main
objective, which pairs a random control input with a random perturbed target
(cells are not paired in Perturb-seq). Using the input cell itself as target
would teach copying, not "stay in this state".
"""
from __future__ import annotations

import numpy as np

KINDS = ("state", "saturation")


class AuxSampler:
    def __init__(self, cond: np.ndarray, singles: list[str]):
        if any("+" in s or s == "ctrl" for s in singles):
            raise ValueError("auxiliary rows use single perturbations only")
        self.by_cond = {s: np.flatnonzero(cond == s) for s in singles}
        self.by_cond = {s: i for s, i in self.by_cond.items() if len(i) >= 2}
        if not self.by_cond:
            raise ValueError("no single has at least 2 cells")
        self.pool = np.concatenate(list(self.by_cond.values()))
        self.cond = cond

    def sample(self, n: int, kind: str, rng) -> tuple[np.ndarray, np.ndarray, list[tuple[str, ...]]]:
        """(input cell idx, target cell idx, perturbed genes to flag per row)."""
        if kind not in KINDS:
            raise ValueError(kind)
        src = rng.choice(self.pool, n)
        tgt = np.empty(n, dtype=np.int64)
        flags = []
        for r, i in enumerate(src):
            c = self.cond[i]
            cand = self.by_cond[c]
            j = i
            while j == i:
                j = cand[rng.integers(len(cand))]
            tgt[r] = j
            flags.append((c,) if kind == "saturation" else ())
        return src, tgt, flags
