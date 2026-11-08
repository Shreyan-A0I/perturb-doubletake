"""The prediction interface: the one boundary between the two workstreams.

    predict(X, pert) -> X'
        X     (n_cells, n_genes) log-normalized expression, genes in `self.genes` order
        pert  tuple of perturbed genes; () means control / no perturbation
        X'    same shape, same space as X

Rollout, the measured-input control and the saturation ratio are all a few lines
on top of this. No experiment code should know which model is underneath.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class Predictor(Protocol):
    name: str
    genes: list[str]

    def predict(self, X: np.ndarray, pert: tuple[str, ...]) -> np.ndarray: ...


_REGISTRY: dict[str, Callable[..., Predictor]] = {}


def register(name: str):
    def deco(factory):
        _REGISTRY[name] = factory
        return factory
    return deco


def get_predictor(name: str, **kwargs) -> Predictor:
    if name == "gears":
        from .models import gears_adapter  # noqa: F401  (registers on import; needs torch)
    if name == "scgpt":
        from .models import scgpt_adapter  # noqa: F401  (registers on import; needs torch, scgpt)
    if name == "cpa":
        from .models import cpa_adapter  # noqa: F401  (registers on import; needs torch, cpa-tools)
    if name not in _REGISTRY:
        raise KeyError(f"Unknown predictor {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


@register("oracle_additive")
class OracleAdditive:
    """Test double, not a baseline. Adds known single shifts to whatever it is given.
    Under this model, rollout must equal the additive null exactly, so it checks the
    rollout plumbing on synthetic data before any real model is involved."""

    name = "oracle_additive"

    def __init__(self, genes: list[str], single_deltas: dict[str, np.ndarray]):
        self.genes = genes
        self.single_deltas = single_deltas

    def predict(self, X: np.ndarray, pert: tuple[str, ...]) -> np.ndarray:
        out = np.array(X, dtype=np.float64, copy=True)
        for g in pert:
            out = out + self.single_deltas[g]
        return out
