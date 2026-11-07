"""Everything run_model.py computes for one held-out pair (A, B) and one split seed.

Inputs (all half 2, so nothing here touches the half-1 cells that pick DEGs and eta):
    X0    control cells          X0b   a second, disjoint control sample
    Xa    real A-perturbed cells pb    {(condition, half): pseudobulk}

Predictions are anchored to the half-2 pseudobulk of the condition their input cells
came from (see run_model.py), so each is "pseudobulk + model's mean predicted shift".

Rows returned (method / arm / order):
    single / na / a, b                 A1. the model's own single predictions from control.
                                       In-sample (models train on singles); recorded because
                                       the implied interaction below is built from them.
    joint / na / na                    f(x0, {A, B})
    rollout / carried|restated / ab|ba
    rollout_measured / carried|restated / ab   second step from real A cells
    input_sensitivity / na / na        A2. does the predicted effect of B depend on the cell?

A2, input sensitivity. With e(X) = mean predicted shift of B from input X:
    sens_corr      Pearson(e(X0), e(Xa))
    sens_rel       ||e(Xa) - e(X0)|| / ||e(X0)||
    sens_null_rel  ||e(X0b) - e(X0)|| / ||e(X0)||: the same quantity between two
                   control samples, i.e. what cell-sampling alone produces. sens_rel
                   means something only relative to this.
Note e(Xa) - e(X0) is exactly the measured-input rollout's implied interaction
(below), because the measured interaction r = (x_AB - x_A) - (x_B - x_ctrl) is itself
"effect of B on A cells minus effect of B on control cells".

A3, interaction recovery. Measured interaction r_h = D_AB - D_A - D_B (half h). Each
prediction implies one, built from the model's own predictions:
    joint, rollout      D_pred(AB) - D_pred(A) - D_pred(B)
    rollout_measured    D_pred(AB) - D_A(measured, half 2) - D_pred(B)
                        (the A part came from real cells, not from the model)
Columns:
    int_r_20deg, int_r_all   Pearson(implied, r_2) on the top-20 DEGs / all genes
    int_rel                  ||implied|| / ||D_pred(AB)||. When this is below ZERO_TOL
                             the implied interaction is zero by construction (additive
                             baseline, GEARS carried rollout) up to float error, and both
                             correlations are set to 0 rather than correlating float noise.
Reference points, per pair, on every row:
    rep_r_20deg, rep_r_all   Pearson(r_1, r_2): what an independent noisy replicate of
                             the measurement achieves. A noiseless perfect model would
                             reach about sqrt(rep_r), not rep_r, since its prediction
                             carries no noise of its own (analyze_rollout.py reports both).
Zero: the additive baseline's implied interaction is identically 0.
"""
from __future__ import annotations

import numpy as np

from . import metrics
from .eta import delta, residual

ZERO_TOL = 1e-4
NAN_SCORES = {"mse_20deg": np.nan, "pearson_delta": np.nan, "pearson_delta_20deg": np.nan}


def _mean(X):
    return np.asarray(X).mean(axis=0)


def interaction_scores(implied: np.ndarray, r2: np.ndarray, d_pred: np.ndarray, deg) -> dict:
    n = np.linalg.norm(d_pred)
    rel = float(np.linalg.norm(implied) / n) if n > 0 else np.nan
    if rel < ZERO_TOL:
        return {"int_r_20deg": 0.0, "int_r_all": 0.0, "int_rel": rel}
    return {"int_r_20deg": metrics.pearson(implied[deg], r2[deg]),
            "int_r_all": metrics.pearson(implied, r2), "int_rel": rel}


def evaluate_pair(model, a: str, b: str, pb: dict, X0, X0b, Xa, k: int,
                  joint_only: bool = False) -> list[dict]:
    ctrl2 = pb[("ctrl", 2)]
    target = pb[(f"{a}+{b}", 2)]
    deg = metrics.top_k_deg(delta(pb, f"{a}+{b}", 1), k)
    r1, r2 = residual(pb, a, b, 1), residual(pb, a, b, 2)
    ref = {"rep_r_20deg": metrics.pearson(r1[deg], r2[deg]), "rep_r_all": metrics.pearson(r1, r2)}
    x0m, xam = _mean(X0), _mean(Xa)
    d_a_meas = pb[(a, 2)] - ctrl2

    # A1: first steps, reused by every rollout below
    X1 = {a: model.predict(X0, (a,)), b: model.predict(X0, (b,))}
    s = {g: _mean(X1[g]) - x0m for g in (a, b)}           # D_pred(A), D_pred(B)

    rows = []
    for order, g in (("a", a), ("b", b)):
        deg_g = metrics.top_k_deg(delta(pb, g, 1), k)
        rows.append({"method": "single", "arm": "na", "order": order,
                     **metrics.score(ctrl2 + s[g], pb[(g, 2)], ctrl2, deg_g), **ref})

    # (method, arm, order, prediction, D_A used for the implied interaction)
    preds = [("joint", "na", "na", ctrl2 + _mean(model.predict(X0, (a, b))) - x0m, s[a])]
    e_a = _mean(model.predict(Xa, (b,))) - xam             # effect of B on real A cells
    if not joint_only:
        for arm in ("carried", "restated"):
            for order, first, second in (("ab", a, b), ("ba", b, a)):
                lab = (second,) if arm == "carried" else (first, second)   # as rollout.rollout()
                p = ctrl2 + _mean(model.predict(X1[first], lab)) - x0m
                preds.append(("rollout", arm, order, p, s[a]))
        preds.append(("rollout_measured", "carried", "ab", pb[(a, 2)] + e_a, d_a_meas))
        preds.append(("rollout_measured", "restated", "ab",
                      pb[(a, 2)] + _mean(model.predict(Xa, (a, b))) - xam, d_a_meas))

    for method, arm, order, p, d_a in preds:
        d_pred = p - ctrl2
        rows.append({"method": method, "arm": arm, "order": order,
                     **metrics.score(p, target, ctrl2, deg),
                     **interaction_scores(d_pred - d_a - s[b], r2, d_pred, deg), **ref})

    # A2
    e0, e0b = s[b], _mean(model.predict(X0b, (b,))) - _mean(X0b)
    n0 = np.linalg.norm(e0)
    rows.append({"method": "input_sensitivity", "arm": "na", "order": "na", **NAN_SCORES,
                 "sens_corr": metrics.pearson(e0, e_a),
                 "sens_rel": float(np.linalg.norm(e_a - e0) / n0) if n0 > 0 else np.nan,
                 "sens_null_rel": float(np.linalg.norm(e0b - e0) / n0) if n0 > 0 else np.nan,
                 **ref})
    return rows
