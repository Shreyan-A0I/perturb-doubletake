"""Experiments built on the predict() interface.

Two rollout arms (reviewer point 1). Norman's doubles keep guide A active the
whole time, and neither arm matches that exactly, so we report both:

    carried    predict(predict(X, (A,)), (B,))      A lives only in the predicted state
    restated   predict(predict(X, (A,)), (A, B))    A is named again with B

Order dependence and saturation are COMPUTATIONAL CONSISTENCY CHECKS. Norman
measures joint perturbations, so it cannot validate either against biology.
"""
from __future__ import annotations

import numpy as np

from .wrapper import Predictor


def _mean(X: np.ndarray) -> np.ndarray:
    return np.asarray(X).mean(axis=0)


def joint(model: Predictor, X0: np.ndarray, a: str, b: str) -> np.ndarray:
    return _mean(model.predict(X0, (a, b)))


def rollout(model: Predictor, X0: np.ndarray, a: str, b: str, arm: str = "carried") -> np.ndarray:
    second = (b,) if arm == "carried" else (a, b)
    return _mean(model.predict(model.predict(X0, (a,)), second))


def rollout_measured(model: Predictor, X_a_real: np.ndarray, a: str, b: str,
                     arm: str = "carried") -> np.ndarray:
    """Second step starts from REAL measured A-perturbed cells instead of predicted ones.
    If this succeeds where `rollout` fails, the failure is generated-input shift."""
    second = (b,) if arm == "carried" else (a, b)
    return _mean(model.predict(X_a_real, second))


def saturation_ratio(model: Predictor, X0: np.ndarray, a: str) -> float:
    """||x2 - x1|| / ||x1 - x0|| for repeated identical perturbation. Near 0 = saturates,
    near 1 = the model re-adds a fixed vector. Consistency check only (see module doc)."""
    x0 = _mean(X0)
    X1 = model.predict(X0, (a,))
    x1, x2 = _mean(X1), _mean(model.predict(X1, (a,)))
    den = np.linalg.norm(x1 - x0)
    return float(np.linalg.norm(x2 - x1) / den) if den > 0 else np.nan


def roundtrip_check(model: Predictor, X_ctrl: np.ndarray, X_a_real: np.ndarray, a: str) -> dict:
    """The three week-one checks. Run before a model joins the roster (reviewer point 2)."""
    x_ctrl = _mean(X_ctrl)
    X1 = model.predict(X_ctrl, (a,))
    x1 = _mean(X1)

    # 1. space: does output live where input lives?
    space = {"in_min": float(x_ctrl.min()), "in_max": float(x_ctrl.max()), "in_mean": float(x_ctrl.mean()),
             "out_min": float(x1.min()), "out_max": float(x1.max()), "out_mean": float(x1.mean())}

    # 2. pass-through of a REAL perturbed profile under no perturbation
    x_a_real = _mean(X_a_real)
    x_a_thru = _mean(model.predict(X_a_real, ()))
    # 3. pass-through of a GENERATED profile under no perturbation
    x1_thru = _mean(model.predict(X1, ()))

    def corr(u, v):
        u, v = u - u.mean(), v - v.mean()
        d = np.sqrt((u @ u) * (v @ v))
        return float(u @ v / d) if d > 0 else np.nan

    return {"gene": a, **space,
            "passthrough_real_r": corr(x_a_real, x_a_thru),
            "passthrough_real_delta_r": corr(x_a_real - x_ctrl, x_a_thru - x_ctrl),
            "passthrough_generated_r": corr(x1, x1_thru),
            "passthrough_generated_delta_r": corr(x1 - x_ctrl, x1_thru - x_ctrl)}
