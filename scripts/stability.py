"""Stability: repeat the same perturbation 1..K times from control cells.

    python scripts/stability.py --model gears --model-dir models/gears_strict_s1 --tag gears_strict
    python scripts/stability.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict
    python scripts/stability.py --model oracle_additive --data data/synthetic/norman_like.h5ad --tag oracle

For the n singles with the largest half-1 shift (among those the model can perturb):
    step_ratio      ||x_k - x_{k-1}|| / ||x_1 - x_0||. Near 0 after step 1 = saturates;
                    constant 1 = re-adds a fixed vector (GEARS, by construction).
    dist_target     distance of the step-k profile from the measured A profile (half 2),
                    relative to the distance between control and A (so 1 = as far as control)
    nearest         nearest real condition (half-2 pseudobulk) to the step-k profile, and
                    its distance relative to ||A - ctrl||
Profiles are anchored like run_model.py: ctrl pseudobulk + mean predicted shift.
Consistency check only: Norman cannot say what a second dose should do.

Writes results/stability_<tag>.csv.
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

from rolloutprobe import splits, stability          # noqa: E402
from rolloutprobe.config import load_config        # noqa: E402
from rolloutprobe.data import load_h5ad            # noqa: E402
from run_model import build_model                  # noqa: E402


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
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    out_dir = Path(args.out or cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_h5ad(args.data or dcfg["path"], condition_col=dcfg.get("condition_col"),
                     sep=dcfg.get("sep"), normalize=dcfg.get("normalize", "auto"),
                     n_top_genes=dcfg.get("n_top_genes"))
    train_singles = [s for s in splits.strict_split(data)["train"] if s != "ctrl"]

    rows, model = [], None
    for seed in args.seeds:
        halves = splits.half_labels(data.cond, seed)
        pb = splits.pseudobulk_by_half(data, halves)
        if model is None or args.model == "oracle_additive":
            model = build_model(args, data, pb, train_singles)
        ok = [s for s in train_singles if not hasattr(model, "can_perturb") or model.can_perturb((s,))]
        size = {s: np.linalg.norm(pb[(s, 1)] - pb[("ctrl", 1)]) for s in ok}
        genes = sorted(ok, key=lambda s: -size[s])[:args.n_genes]
        profiles = {c: v for (c, h), v in pb.items() if h == 2}
        ctrl2 = profiles["ctrl"]

        rng = np.random.default_rng(seed)
        idx = np.flatnonzero((data.cond == "ctrl") & (halves == 2))
        idx = rng.choice(idx, min(args.max_cells, len(idx)), replace=False)
        X0 = data.X[idx]
        X0 = np.asarray(X0.toarray() if hasattr(X0, "toarray") else X0, dtype=np.float32)

        for a in genes:
            means, _ = stability.repeat_trajectory(model, X0, a, args.steps)
            ratios = stability.step_ratios(means)
            scale = np.linalg.norm(profiles[a] - ctrl2)
            for k in range(1, args.steps + 1):
                x = ctrl2 + means[k] - means[0]
                near, dist = stability.nearest_condition(x, profiles)
                rows.append({"model": args.tag, "gene": a, "split_seed": seed, "step": k,
                             "step_ratio": float(ratios[k - 1]),
                             "dist_target": float(np.linalg.norm(x - profiles[a]) / scale),
                             "nearest": near, "nearest_dist": dist / scale})
        print(f"  split seed {seed} done ({genes})")

    df = pd.DataFrame(rows)
    path = out_dir / f"stability_{args.tag}.csv"
    df.to_csv(path, index=False)
    pd.set_option("display.width", 160)
    print(df.groupby("step")[["step_ratio", "dist_target", "nearest_dist"]].mean().round(3).to_string())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
