"""Carry, erasure, variance ratios and the interaction-blind references
(rolloutprobe/diagnostics.py, scripts/diagnostics.py) on toy models with known answers."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rolloutprobe import diagnostics as dg              # noqa: E402
from rolloutprobe.pair_eval import evaluate_pair         # noqa: E402

G = 60
rng = np.random.default_rng(3)
CTRL = rng.uniform(0.5, 2.0, G)
D = {"A": rng.normal(0, 1, G), "B": rng.normal(0, 1, G)}
R = rng.normal(0, 0.5, G)
PB = {}
for h in (1, 2):
    PB[("ctrl", h)] = CTRL
    PB[("A", h)] = CTRL + D["A"]
    PB[("B", h)] = CTRL + D["B"]
    PB[("A+B", h)] = CTRL + D["A"] + D["B"] + R


def cells(center, n=50, sd=0.1):
    return center + rng.normal(0, sd, (n, G))


X0, X0B, XA = cells(CTRL), cells(CTRL), cells(CTRL + D["A"])
XAB, XB = cells(CTRL + D["A"] + D["B"] + R), cells(CTRL + D["B"])


class GearsLike:                       # out = x + g(pert): carries the input in full
    def predict(self, X, pert):
        return X + sum((D[p] for p in pert), np.zeros(G))


class Eraser:                          # ignores the input: every cell -> ctrl + effect
    def predict(self, X, pert):
        return np.tile(CTRL + sum((D[p] for p in pert), np.zeros(G)), (len(X), 1))


class Partial:                         # carries a fraction c of the input's offset, shrinks variance
    c = 0.25

    def predict(self, X, pert):
        return CTRL + self.c * (X - CTRL) + sum((D[p] for p in pert), np.zeros(G))


def diag(model):
    return dg.pair_diagnostics(dg.PredCache(model), "A", "B", PB, X0, X0B, XA, XAB, XB, k=20)


def a2_row(model):
    rows = evaluate_pair(model, "A", "B", PB, X0, X0B, XA, k=20)
    return next(r for r in rows if r["method"] == "input_sensitivity")


def test_gears_like_carries_everything():
    r = diag(GearsLike())
    assert r["carry"] == pytest.approx(1.0) and r["carry_20deg"] == pytest.approx(1.0)
    assert r["carry_cos"] == pytest.approx(1.0)
    assert r["eras_rel"] < 1e-9 and r["eras_resid"] == pytest.approx(1.0)   # e = 0
    assert r["vr_ctrl"] == pytest.approx(1.0) and r["vr_input_a"] == pytest.approx(1.0)


def test_eraser_erases_the_input_state():
    r = diag(Eraser())
    assert r["carry"] == pytest.approx(0.0, abs=1e-12)
    # e = -(mean Xa - mean X0) exactly; ~ -D_A up to cell sampling
    assert r["eras_corr_sample"] == pytest.approx(1.0)
    assert r["eras_rel_sample"] == pytest.approx(1.0)
    assert r["eras_resid_sample"] == pytest.approx(0.0, abs=1e-12)
    assert r["eras_corr"] > 0.99 and r["eras_rel"] == pytest.approx(1.0, abs=0.05)
    assert r["eras_resid"] < 0.05
    assert r["vr_ctrl"] == pytest.approx(0.0, abs=1e-12) and r["vr_target"] == pytest.approx(0.0, abs=1e-12)


def test_partial_carry_and_identities():
    m = Partial()
    r = diag(m)
    assert r["carry"] == pytest.approx(m.c) and r["carry_20deg"] == pytest.approx(m.c)
    assert r["eras_rel_sample"] == pytest.approx(1 - m.c)
    assert r["eras_resid_sample"] == pytest.approx(r["carry"])           # identity: e + d_in = d_out
    assert r["vr_ctrl"] == pytest.approx(m.c ** 2)                       # variance scales by c^2
    # the A2 numbers agree with pair_eval's on the same cells
    a2 = a2_row(m)
    for c in ("sens_rel", "sens_null_rel", "sens_corr"):
        assert r[c] == pytest.approx(a2[c])


def test_cache_predicts_control_once_per_b():
    pred = dg.PredCache(GearsLike())
    for _ in range(3):
        dg.pair_diagnostics(pred, "A", "B", PB, X0, X0B, XA, XAB, XB, k=20)
    assert pred.calls == 2 + 3          # f(X0,B), f(X0b,B) once; f(Xa,B) per call


def test_references_recover_planted_structure():
    pb = dict(PB)
    for h in (1, 2):                   # interaction = -D_A: a "forget A" world
        pb[("A+B", h)] = CTRL + D["B"]
    r = dg.reference_row(pb, "A", "B", k=20)
    assert r["forget_a_r_all"] == pytest.approx(1.0) and r["forget_a_r_20deg"] == pytest.approx(1.0)
    assert r["forget_a_r_all_h2"] == pytest.approx(1.0)
    r0 = dg.reference_row(PB, "A", "B", k=20)  # random interaction: references uncorrelated by design
    assert abs(r0["forget_a_r_all"]) < 0.5 and r0["rep_r_all"] == pytest.approx(1.0)


def test_half1_references_share_no_noise_with_half2():
    """Additive truth, independent noise per half: the half-2 reference is inflated by
    shared noise, the half-1 reference is not (on average)."""
    g = 400
    h1, h2 = [], []
    for s in range(30):
        rr = np.random.default_rng(100 + s)
        ctrl, da, db = rr.uniform(1, 2, g), rr.normal(0, 1, g), rr.normal(0, 1, g)
        pb = {}
        for h in (1, 2):
            noise = lambda: rr.normal(0, 0.3, g)   # noqa: E731
            pb.update({("ctrl", h): ctrl + noise(), ("A", h): ctrl + da + noise(),
                       ("B", h): ctrl + db + noise(), ("A+B", h): ctrl + da + db + noise()})
        r = dg.reference_row(pb, "A", "B", k=20)
        h1.append(r["forget_a_r_all"]), h2.append(r["forget_a_r_all_h2"])
    assert abs(np.mean(h1)) < 0.05 and np.mean(h2) > 0.1


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("diag")
    h5 = d / "synth.h5ad"
    subprocess.run([sys.executable, "scripts/make_synthetic.py", "--out", str(h5), "--genes", "120",
                    "--pairs", "8"], cwd=ROOT, check=True, capture_output=True, text=True)
    return d, h5


def test_script_end_to_end(synth):
    d, h5 = synth
    common = ["--data", str(h5), "--out", str(d), "--n-boot", "50"]
    for extra in (["--model", "oracle_additive", "--tag", "oracle", "--check", str(d / "none.csv")],
                  ["--model", "none", "--seeds", "0", "1"]):
        subprocess.run([sys.executable, "scripts/diagnostics.py", *common, *extra], cwd=ROOT,
                       check=True, capture_output=True, text=True)
    m = pd.read_csv(d / "diagnostics_oracle.csv")
    assert len(m) > 0 and np.allclose(m["carry"], 1.0) and np.allclose(m["vr_ctrl"], 1.0)
    ref = pd.read_csv(d / "diagnostics_references.csv")
    assert set(ref["split_seed"]) == {0, 1} and ref["forget_a_r_20deg"].between(-1, 1).all()
