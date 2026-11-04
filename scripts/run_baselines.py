"""Midterm items 1 and 2: the null hierarchy and the split-half reliability of eta.

CPU only. For each split seed:
    1. split every condition's cells into halves 1 and 2
    2. eta and top-k DEGs from half 1
    3. null predictions from half-2 training conditions, scored on half-2 doubles
Then bootstrap over pairs.

    python scripts/run_baselines.py --config configs/default.yaml
    python scripts/run_baselines.py --config configs/default.yaml --data data/synthetic/norman_like.h5ad

Writes results/baselines.csv (schema rows), results/eta.csv, results/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rolloutprobe import eta as eta_mod                       # noqa: E402
from rolloutprobe import metrics, nulls, schema, splits, stats  # noqa: E402
from rolloutprobe.config import load_config                   # noqa: E402
from rolloutprobe.data import load_h5ad                       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data", default=None, help="override data.path")
    p.add_argument("--out", default=None, help="override output.results_dir")
    args = p.parse_args()

    cfg = load_config(args.config)
    dcfg, acfg = cfg["data"], cfg["analysis"]
    out_dir = Path(args.out or cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_h5ad(args.data or dcfg["path"], condition_col=dcfg.get("condition_col"),
                     sep=dcfg.get("sep"), normalize=dcfg.get("normalize", "auto"),
                     n_top_genes=dcfg.get("n_top_genes"))
    split = splits.strict_split(data)
    train_singles = [s for s in split["train"] if s != "ctrl"]
    pairs = split["test"]
    counts = data.cell_counts()
    print(f"loaded {data.n_cells} cells x {len(data.genes)} genes | "
          f"{len(train_singles)} singles, {len(pairs)} doubles, {len(split['dropped'])} doubles dropped")
    print(f"cells per double: median {int(counts[[f'{a}+{b}' for a, b in pairs]].median())}, "
          f"min {int(counts[[f'{a}+{b}' for a, b in pairs]].min())}")

    k, top_k = acfg["lowrank_k"], acfg["top_k_deg"]
    rows, eta_tables = [], []
    for seed in acfg["split_seeds"]:
        halves = splits.half_labels(data.cond, seed)
        pb = splits.pseudobulk_by_half(data, halves)

        et = eta_mod.eta_table(pb, pairs).assign(split_seed=seed)
        eta_tables.append(et)
        eta_h1 = dict(zip(zip(et["pert_a"], et["pert_b"]), et["eta_h1"]))

        preds = nulls.predict_nulls(pb, pairs, train_singles, half=2, k=k)
        ctrl2 = pb[("ctrl", 2)]
        for i, (a, b) in enumerate(pairs):
            target = pb[(f"{a}+{b}", 2)]
            deg = metrics.top_k_deg(eta_mod.delta(pb, f"{a}+{b}", 1), top_k)   # half 1
            for method, P in preds.items():
                rows.append({"model": "none", "method": method, "arm": "na", "order": "na",
                             "pert_a": a, "pert_b": b,
                             "support": splits.support_tier((a, b), set(train_singles)),
                             "split_seed": seed, "eta_h1": eta_h1[(a, b)],
                             **metrics.score(P[i], target, ctrl2, deg)})
        print(f"  split seed {seed} done")

    res = schema.validate(pd.DataFrame(rows))
    res.to_csv(out_dir / "baselines.csv", index=False)
    eta_all = pd.concat(eta_tables, ignore_index=True)
    eta_all.to_csv(out_dir / "eta.csv", index=False)

    # Reliability: per seed, then averaged, so one lucky split cannot carry it.
    rel = pd.DataFrame([eta_mod.reliability(g) for _, g in eta_all.groupby("split_seed")])
    reliability = {c: float(rel[c].mean()) for c in rel.columns}
    reliability["per_seed_spearman"] = rel["spearman_eta_h1_h2"].round(3).tolist()

    nb = acfg["n_boot"]
    summary_tables = pd.concat([stats.method_summary(res, m, nb) for m in
                                ("mse_20deg", "pearson_delta", "pearson_delta_20deg")])

    # Does the additive null get worse where half-1 eta is high? Independent halves,
    # so a positive relationship here is evidence that eta strata mean something.
    eta_pair = eta_all.groupby(["pert_a", "pert_b"])["eta_h1"].mean()
    eta_pair.index = [f"{a}+{b}" for a, b in eta_pair.index]
    add_mse = stats.per_pair(res, "mse_20deg")["additive"]
    strata = eta_mod.assign_strata(eta_pair, acfg["n_strata"])
    by_stratum = (pd.DataFrame({"mse": add_mse, "stratum": strata})
                    .groupby("stratum", observed=True)["mse"]
                    .apply(lambda v: stats.bootstrap_mean(v.to_numpy(), nb, 0)).unstack())
    lowrank_slope = stats.slope_vs_eta(stats.advantage(res, "mse_20deg", "lowrank_additive"),
                                       eta_pair, nb)

    summary = {
        "n_pairs": len(pairs), "n_singles": len(train_singles), "n_genes": len(data.genes),
        "split_seeds": acfg["split_seeds"],
        "eta_reliability": reliability,
        "null_hierarchy": summary_tables.to_dict(orient="records"),
        "additive_mse_by_eta_stratum": by_stratum.reset_index().to_dict(orient="records"),
        "lowrank_vs_additive_slope_on_eta": lowrank_slope,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\nEta split-half reliability (mean over seeds)")
    for key in ("spearman_eta_h1_h2", "median_cos_residuals", "signal_fraction_eta2"):
        print(f"  {key:28s} {reliability[key]:.3f}")
    print("\nNull hierarchy, mean over pairs [95% bootstrap CI]")
    for _, r in summary_tables.iterrows():
        print(f"  {r['metric']:20s} {r['method']:18s} {r['mean']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]")
    print("\nAdditive MSE@20DEG by half-1 eta stratum")
    for s, r in by_stratum.iterrows():
        print(f"  {s:6s} {r['mean']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]")
    print(f"\nwrote {out_dir}/baselines.csv, eta.csv, summary.json")


if __name__ == "__main__":
    main()
