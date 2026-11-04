"""End-to-end checks on synthetic data with known answers. Run: pytest -q"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rolloutprobe import schema, splits          # noqa: E402
from rolloutprobe.data import canonical          # noqa: E402


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    h5 = d / "norman_like.h5ad"
    run("scripts/make_synthetic.py", "--out", str(h5), "--genes", "250", "--pairs", "30")
    res = d / "results"
    run("scripts/run_baselines.py", "--data", str(h5), "--out", str(res))
    run("scripts/run_model.py", "--model", "oracle_additive", "--tag", "oracle",
        "--data", str(h5), "--out", str(res), "--seeds", "0", "1")
    return {"h5": h5, "res": res, "truth": pd.read_csv(h5.with_suffix(".truth.csv"))}


def test_canonical_labels():
    assert canonical("CBL+ctrl", "+") == "CBL"
    assert canonical("ctrl+CBL", "+") == "CBL"
    assert canonical("CNN1+CBL", "+") == "CBL+CNN1"
    assert canonical("ctrl", "+") == "ctrl"
    assert canonical("control", "_") == "ctrl"
    assert canonical("CBL_CNN1", "_") == "CBL+CNN1"
    assert canonical("A+B+C", "+") is None


def test_half_labels_disjoint_and_balanced():
    cond = np.array(["ctrl"] * 101 + ["A"] * 40 + ["A+B"] * 7)
    h = splits.half_labels(cond, seed=3)
    assert set(np.unique(h)) == {1, 2}
    for c in np.unique(cond):
        n1, n2 = (h[cond == c] == 1).sum(), (h[cond == c] == 2).sum()
        assert abs(n1 - n2) <= 1


def test_results_follow_schema(synth):
    for f in ("baselines.csv", "model_oracle.csv"):
        schema.validate(pd.read_csv(synth["res"] / f))
    schema.validate(schema.fake_results(n_pairs=5, n_seeds=1))


def test_eta_recovers_planted_interactions(synth):
    eta = pd.read_csv(synth["res"] / "eta.csv").groupby(["pert_a", "pert_b"])[["eta_h1", "eta2_cross"]].mean()
    m = eta.reset_index().merge(synth["truth"], on=["pert_a", "pert_b"])
    hi, lo = m[m.interacting], m[~m.interacting]
    assert hi["eta_h1"].min() > lo["eta_h1"].max(), "eta should separate interacting pairs"
    # cross-half estimate is unbiased: near zero for truly additive pairs
    assert abs(lo["eta2_cross"].mean()) < 0.1 * hi["eta2_cross"].mean()


def test_null_hierarchy_sane(synth):
    s = json.loads((synth["res"] / "summary.json").read_text())
    mse = {r["method"]: r["mean"] for r in s["null_hierarchy"] if r["metric"] == "mse_20deg"}
    assert mse["additive"] < mse["mean"] < mse["nochange"] * 1.01


def test_oracle_rollout_equals_additive_null(synth):
    """Under an exactly additive model, carried rollout, joint and measured-input
    rollout must reproduce the additive null. Checks the rollout plumbing and the
    input-sampling anchor in run_model.py."""
    m = pd.read_csv(synth["res"] / "model_oracle.csv")
    b = pd.read_csv(synth["res"] / "baselines.csv")
    add = b[b.method == "additive"].set_index(["pert_a", "pert_b", "split_seed"])["mse_20deg"]
    for method, arm in [("joint", "na"), ("rollout", "carried"), ("rollout_measured", "carried")]:
        g = m[(m.method == method) & (m.arm == arm)].set_index(["pert_a", "pert_b", "split_seed"])["mse_20deg"]
        assert np.allclose(g.to_numpy(), add.loc[g.index].to_numpy(), atol=1e-5), (method, arm)
    # restated arm re-applies A, so it must NOT equal addition under this model
    r = m[(m.method == "rollout") & (m.arm == "restated")].set_index(["pert_a", "pert_b", "split_seed"])["mse_20deg"]
    assert (r.to_numpy() > add.loc[r.index].to_numpy() + 1e-3).mean() > 0.9
