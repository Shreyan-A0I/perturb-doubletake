"""Discriminator drift: can a classifier tell real cells from predicted cells?

    python scripts/drift_discriminator.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict
    python scripts/drift_discriminator.py --model oracle_additive --data data/synthetic/norman_like.h5ad --tag oracle

Cross-validated accuracy of a PCA + logistic-regression classifier
(rolloutprobe/discriminator.py), classes balanced: 0.5 = indistinguishable.

Two comparisons:
    repeat    real A cells vs cells after k applications of A to control cells
              (k in --steps), for the n singles with the largest half-1 shift
    pairs     real AB cells vs predicted AB cells (joint, carried rollout ab,
              measured-input carried rollout), for the --n-pairs pairs with the
              largest half-1 eta (all pairs if 0)
Every comparison comes with two reference points on real cells only:
    floor     real target cells, half 1 vs half 2 (two samples of one condition)
    scale     real control cells vs real target cells
Predicted cells use half-2 inputs; real target cells are half 2.

Writes results/drift_<tag>.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rolloutprobe import eta as eta_mod, splits, stability   # noqa: E402
from rolloutprobe.config import load_config                  # noqa: E402
from rolloutprobe.data import load_h5ad                      # noqa: E402
from rolloutprobe.discriminator import cv_accuracy           # noqa: E402
from run_model import build_model                            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", required=True, choices=["gears", "scgpt", "cpa", "oracle_additive"])
    p.add_argument("--tag", required=True)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--gears-data-dir", default="data/gears")
    p.add_argument("--data", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-cells", type=int, default=300)
    p.add_argument("--n-genes", type=int, default=5)
    p.add_argument("--steps", type=int, nargs="*", default=[1, 2, 3, 5])
    p.add_argument("--n-pairs", type=int, default=20)
    p.add_argument("--n-comp", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="split seed (halves)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    out_dir = Path(args.out or cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_h5ad(args.data or dcfg["path"], condition_col=dcfg.get("condition_col"),
                     sep=dcfg.get("sep"), normalize=dcfg.get("normalize", "auto"),
                     n_top_genes=dcfg.get("n_top_genes"))
    split = splits.strict_split(data)
    train_singles = [s for s in split["train"] if s != "ctrl"]
    halves = splits.half_labels(data.cond, args.seed)
    pb = splits.pseudobulk_by_half(data, halves)
    model = build_model(args, data, pb, train_singles)
    can = (lambda g: model.can_perturb(g)) if hasattr(model, "can_perturb") else (lambda g: True)
    rng = np.random.default_rng(args.seed)

    def cells(cond, half=2):
        idx = np.flatnonzero((data.cond == cond) & (halves == half))
        idx = rng.choice(idx, min(args.max_cells, len(idx)), replace=False)
        sub = data.X[idx]
        return np.asarray(sub.toarray() if hasattr(sub, "toarray") else sub, dtype=np.float32)

    acc = lambda A, B: cv_accuracy(A, B, n_comp=args.n_comp, seed=args.seed)  # noqa: E731
    X0 = cells("ctrl")
    rows = []

    def put(kind, target, variant, a):
        rows.append({"model": args.tag, "split_seed": args.seed, "comparison": kind,
                     "target": target, "variant": variant, "accuracy": a})

    ok = [s for s in train_singles if can((s,))]
    size = {s: np.linalg.norm(pb[(s, 1)] - pb[("ctrl", 1)]) for s in ok}
    for a in sorted(ok, key=lambda s: -size[s])[:args.n_genes]:
        real = cells(a)
        put("repeat", a, "floor", acc(cells(a, half=1), real))
        put("repeat", a, "scale", acc(X0, real))
        _, kept = stability.repeat_trajectory(model, X0, a, max(args.steps), keep_cells=tuple(args.steps))
        for k in args.steps:
            put("repeat", a, f"step_{k}", acc(kept[k], real))
        print(f"  repeat {a} done")

    pairs = [pr for pr in split["test"] if can(pr)]
    if args.n_pairs:
        et = eta_mod.eta_table(pb, pairs).sort_values("eta_h1", ascending=False)
        pairs = list(zip(et["pert_a"], et["pert_b"]))[:args.n_pairs]
    for a, b in pairs:
        ab = f"{a}+{b}"
        real = cells(ab)
        put("pairs", ab, "floor", acc(cells(ab, half=1), real))
        put("pairs", ab, "scale", acc(X0, real))
        put("pairs", ab, "joint", acc(model.predict(X0, (a, b)), real))
        put("pairs", ab, "rollout_carried_ab", acc(model.predict(model.predict(X0, (a,)), (b,)), real))
        put("pairs", ab, "rollout_measured_carried_ab", acc(model.predict(cells(a), (b,)), real))
    print(f"  {len(pairs)} pairs done")

    df = pd.DataFrame(rows)
    path = out_dir / f"drift_{args.tag}.csv"
    df.to_csv(path, index=False)
    pd.set_option("display.width", 160)
    print("accuracy 0.5 = indistinguishable from real; floor = real vs real; scale = control vs target")
    print(df.groupby(["comparison", "variant"])["accuracy"].agg(["mean", "min", "max", "count"])
            .round(3).to_string())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
