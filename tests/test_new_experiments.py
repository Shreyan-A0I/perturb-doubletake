"""Part B (latent additivity), Part C (auxiliary rows), stability and discriminator
drift on toy inputs with known answers."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rolloutprobe import latent, stability                    # noqa: E402
from rolloutprobe.aux_batches import AuxSampler               # noqa: E402
from rolloutprobe.discriminator import cv_accuracy            # noqa: E402
from rolloutprobe.wrapper import OracleAdditive               # noqa: E402

# ---- Part B --------------------------------------------------------------
G = 40
rng = np.random.default_rng(1)
CTRL = rng.uniform(1, 2, G)
DA, DB = rng.normal(0, 1, G), rng.normal(0, 1, G)
R = np.zeros(G)
R[:5] = 2.0                                     # interaction lives on genes 0-4
TRUE = {"ctrl": CTRL, "A": CTRL + DA, "B": CTRL + DB, "A+B": CTRL + DA + DB + R}


def store_for(embed, noise=0.0):
    """Exact (or noisy, independently per half) condition means through an embedding."""
    s = latent.LatentStore()
    for c, x in TRUE.items():
        for h in (1, 2):
            xx = x + noise * np.random.default_rng(10 * h + len(c)).normal(0, 1, G)
            s.put(c, h, embed(xx[None, :]))
    return s


def test_linear_embedding_keeps_interaction_and_projection_drops_it():
    keep = lambda X: {"genes": X.mean(0), "linear": (X @ np.diag(np.linspace(1, 3, G))).mean(0)}  # noqa: E731
    t = latent.pair_terms(store_for(keep).get, "A", "B")
    assert t["genes"]["rel_cross"] > 0.1
    # a per-gene rescaling is linear: the residual is still exactly there
    assert t["linear"]["rel_cross"] > 0.1 and t["linear"]["rep_cos"] == pytest.approx(1.0)
    drop = lambda X: {"genes": X.mean(0), "proj": X[:, 5:].mean(0)}   # noqa: E731  (discards genes 0-4)
    t = latent.pair_terms(store_for(drop).get, "A", "B")
    assert t["proj"]["rel_cross"] == pytest.approx(0.0, abs=1e-9)
    assert t["genes"]["rel_cross"] == pytest.approx(np.linalg.norm(R) / np.linalg.norm(DA + DB + R))


def test_cross_half_estimate_ignores_independent_noise():
    """A truly additive pair measured with noise: raw ||r||/||d|| is inflated,
    the cross-half estimate stays near 0 on average."""
    global R
    saved, R = R, np.zeros(G)
    TRUE["A+B"] = CTRL + DA + DB
    try:
        vals_raw, vals_cross = [], []
        for k in range(40):
            s = latent.LatentStore()
            for c, x in TRUE.items():
                for h in (1, 2):
                    s.put(c, h, {"genes": x + 0.3 * np.random.default_rng(1000 * k + 10 * h + len(c)).normal(0, 1, G)})
            t = latent.pair_terms(s.get, "A", "B")["genes"]
            vals_raw.append(t["rel_h2"])
            vals_cross.append(t["rr_cross"] / t["dd_cross"])
        assert np.mean(vals_raw) > 0.2
        assert abs(np.mean(vals_cross)) < 0.05
    finally:
        R = saved
        TRUE["A+B"] = CTRL + DA + DB + R


def test_fill_store_disk_cache_and_ratio(tmp_path):
    cond = np.array(["ctrl"] * 20 + ["A"] * 20 + ["B"] * 20 + ["A+B"] * 20)
    X = np.vstack([np.tile(TRUE[c], (20, 1)) for c in ("ctrl", "A", "B", "A+B")]).astype(np.float32)
    halves = np.tile([1, 2], 40)
    calls = []

    def embed(Xb):
        calls.append(len(Xb))
        return {"genes": Xb.mean(0), "half": Xb[:, 5:].mean(0)}

    store = latent.LatentStore(tmp_path / "cache")
    latent.fill_store(store, X, cond, halves, latent.needed_conditions([("A", "B")]), embed, 5,
                      np.random.default_rng(0), log=None)
    assert calls == [5] * 8
    latent.fill_store(store, X, cond, halves, ["ctrl", "A"], embed, 5, np.random.default_rng(0), log=None)
    assert len(calls) == 8                                  # cached: nothing recomputed
    t = latent.pair_terms(store.get, "A", "B")
    r = latent.pooled_ratio_boot([t["half"]["rr_cross"]] * 3, [t["half"]["dd_cross"]] * 3,
                                 [t["genes"]["rr_cross"]] * 3, [t["genes"]["dd_cross"]] * 3, 50)
    assert r["mean"] == pytest.approx(0.0, abs=1e-6)


# ---- Part C ---------------------------------------------------------------
def test_aux_sampler_singles_only_same_condition():
    cond = np.array(["ctrl"] * 5 + ["A"] * 4 + ["B"] * 3 + ["A+B"] * 3 + ["C"] * 1)
    s = AuxSampler(cond, ["A", "B", "C"])
    assert set(s.by_cond) == {"A", "B"}                     # C has one cell: no distinct target
    g = np.random.default_rng(0)
    for kind in ("state", "saturation"):
        src, tgt, flags = s.sample(200, kind, g)
        assert (cond[src] == cond[tgt]).all() and (src != tgt).all()
        assert set(cond[src]) <= {"A", "B"}
        assert all(f == (() if kind == "state" else (cond[i],)) for f, i in zip(flags, src))
    with pytest.raises(ValueError):
        AuxSampler(cond, ["A", "A+B"])


# ---- stability -----------------------------------------------------------
def test_stability_additive_never_saturates():
    m = OracleAdditive([f"g{i}" for i in range(G)], {"A": DA})
    X0 = np.tile(CTRL, (10, 1))
    means, kept = stability.repeat_trajectory(m, X0, "A", 4, keep_cells=(2,))
    np.testing.assert_allclose(stability.step_ratios(means), 1.0)
    np.testing.assert_allclose(kept[2].mean(0), CTRL + 2 * DA)
    near, d = stability.nearest_condition(means[1], TRUE)
    assert near == "A" and d == pytest.approx(0.0, abs=1e-9)


class Saturating:
    def predict(self, X, pert):                             # moves cells halfway to the A state
        return X + 0.5 * ((CTRL + DA) - X)


def test_stability_saturating_model():
    means, _ = stability.repeat_trajectory(Saturating(), np.tile(CTRL, (4, 1)), "A", 3)
    np.testing.assert_allclose(stability.step_ratios(means), [1, 0.5, 0.25])


# ---- discriminator -------------------------------------------------------
def test_discriminator_chance_and_separable():
    g = np.random.default_rng(0)
    A, B = g.normal(0, 1, (150, 30)), g.normal(0, 1, (150, 30))
    assert 0.35 < cv_accuracy(A, B, n_comp=10) < 0.65
    assert cv_accuracy(A, B + 1.5, n_comp=10) > 0.95


# ---- scripts on synthetic data -------------------------------------------
def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth_new")
    h5 = d / "norman_like.h5ad"
    run("scripts/make_synthetic.py", "--out", str(h5), "--genes", "120", "--pairs", "12")
    return {"h5": h5, "res": d / "results", "truth": pd.read_csv(h5.with_suffix(".truth.csv"))}


def test_latent_script_gene_space(synth):
    run("scripts/latent_additivity.py", "--model", "genes", "--tag", "genes", "--data", str(synth["h5"]),
        "--out", str(synth["res"]), "--n-boot", "50")
    pairs = pd.read_csv(synth["res"] / "latent_genes_pairs.csv")
    m = pairs.merge(synth["truth"], on=["pert_a", "pert_b"])
    # planted interactions show up as a larger noise-corrected residual
    assert m[m.interacting]["rel_cross"].min() > m[~m.interacting]["rel_cross"].median()
    summ = pd.read_csv(synth["res"] / "latent_genes.csv").set_index("quantity")
    assert summ.loc["cross_half_signal", "lo"] > 0          # planted interactions: signal above the null


def test_stability_and_drift_scripts_on_oracle(synth):
    common = ["--model", "oracle_additive", "--tag", "oracle", "--data", str(synth["h5"]),
              "--out", str(synth["res"]), "--n-genes", "2"]
    run("scripts/stability.py", *common, "--steps", "3")
    st = pd.read_csv(synth["res"] / "stability_oracle.csv")
    np.testing.assert_allclose(st["step_ratio"], 1.0, atol=1e-5)
    run("scripts/drift_discriminator.py", *common, "--steps", "1", "--n-pairs", "3", "--n-comp", "10")
    dr = pd.read_csv(synth["res"] / "drift_oracle.csv")
    assert set(dr["variant"]) >= {"floor", "scale", "step_1", "joint", "rollout_carried_ab"}
    assert dr["accuracy"].between(0, 1).all()
