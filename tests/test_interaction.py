"""A2 and A3 on toy models with known answers (rolloutprobe/pair_eval.py).

Noise-free pseudobulks, so the measured interaction r is exact and replicates
perfectly. Three models:
    composer     effect of B depends on whether the input cell is in the A state;
                 composes perfectly
    gears_like   out = x + g(pert), g ignores x, but g(A,B) knows the interaction
                 (as GEARS's GO graph allows)
    additive     out = x + D_A + D_B
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rolloutprobe.pair_eval import evaluate_pair  # noqa: E402

G = 60
rng = np.random.default_rng(0)
CTRL = rng.uniform(0.5, 2.0, G)
D = {"A": rng.normal(0, 1, G), "B": rng.normal(0, 1, G)}
R = rng.normal(0, 0.5, G)                      # true interaction
PB = {}
for h in (1, 2):
    PB[("ctrl", h)] = CTRL
    PB[("A", h)] = CTRL + D["A"]
    PB[("B", h)] = CTRL + D["B"]
    PB[("A+B", h)] = CTRL + D["A"] + D["B"] + R


def cells(center, n=40):
    return center + rng.normal(0, 1e-3, (n, G))


X0, X0B, XA = cells(CTRL), cells(CTRL), cells(CTRL + D["A"])


class Composer:
    def a_state(self, X):          # per cell: 0 at control, 1 in the A state
        return ((X - CTRL) @ D["A"]) / (D["A"] @ D["A"])

    def predict(self, X, pert):
        out = X.copy()
        for g in pert:
            out = out + D[g]
        if "B" in pert:
            s = np.ones(len(X)) if "A" in pert else self.a_state(X)
            out = out + np.outer(s, R)
        return out


class GearsLike:
    def predict(self, X, pert):
        g = sum((D[p] for p in pert), np.zeros(G))
        return X + g + (R if set(pert) == {"A", "B"} else 0)


class Additive:
    def predict(self, X, pert):
        return X + sum((D[p] for p in pert), np.zeros(G))


def run(model):
    rows = evaluate_pair(model, "A", "B", PB, X0, X0B, XA, k=20)
    return {(r["method"], r["arm"], r["order"]): r for r in rows}


def test_reference_points():
    r = run(Additive())
    assert r[("joint", "na", "na")]["rep_r_all"] == pytest.approx(1.0)     # noise-free replicate
    for key in [("joint", "na", "na"), ("rollout", "carried", "ab"), ("rollout_measured", "carried", "ab")]:
        assert r[key]["int_r_all"] == 0.0 and r[key]["int_rel"] < 1e-4, key


def test_composer_recovers_interaction_and_is_input_sensitive():
    r = run(Composer())
    for key in [("joint", "na", "na"), ("rollout", "carried", "ab"), ("rollout_measured", "carried", "ab")]:
        assert r[key]["int_r_all"] > 0.99 and r[key]["int_r_20deg"] > 0.99, key
    s = r[("input_sensitivity", "na", "na")]
    assert s["sens_rel"] == pytest.approx(np.linalg.norm(R) / np.linalg.norm(D["B"]), rel=1e-2)
    assert s["sens_null_rel"] < 1e-2 < s["sens_rel"]


def test_gears_like_joint_knows_interaction_but_rollout_cannot_carry_it():
    r = run(GearsLike())
    assert r[("joint", "na", "na")]["int_r_all"] > 0.99
    assert r[("rollout", "carried", "ab")]["int_r_all"] == 0.0
    assert r[("rollout", "carried", "ba")]["int_r_all"] == 0.0
    assert r[("rollout_measured", "carried", "ab")]["int_r_all"] == 0.0
    s = r[("input_sensitivity", "na", "na")]
    assert s["sens_rel"] < 1e-6 and s["sens_corr"] == pytest.approx(1.0)


def test_singles_rows_recorded():
    r = run(Composer())
    for order in ("a", "b"):
        assert r[("single", "na", order)]["pearson_delta"] == pytest.approx(1.0, abs=1e-6)
