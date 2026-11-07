"""RQ1/RQ2 analysis of model results against the additive baseline.

    python scripts/analyze_rollout.py                       # every results/model_*.csv
    python scripts/analyze_rollout.py --models scgpt_strict

For each model and each prediction variant (joint; rollout per arm and order;
measured-input rollout per arm), on the pairs that model was scored on:

    advantage      per-pair (additive MSE - model MSE) on the top-20 DEGs, and
                   (model - additive) Pearson delta; positive = model better.
                   Mean, 95% bootstrap CI over pairs, share of pairs better.
    slope_eta      OLS slope of advantage on eta (half 1), bootstrap CI.
    slope_rel_eta  same on eta / |double shift| (half 1), the effect-size check
                   promised in the midterm report.
    tertile_<t>    mean advantage within each eta tertile (tertiles fixed on all
                   131 pairs), which cancels the scale of the perturbation.

Plus two consistency checks on the rollout rows:
    order_gap      per-pair |score(A then B) - score(B then A)|. A score-level
                   proxy for order dependence (CSV rows hold scores, not profiles).
    arm_gap        per-pair score(carried) - score(restated), same order "ab".

Writes results/rollout_analysis.csv (long format) and prints a summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rolloutprobe import eta as eta_mod, stats  # noqa: E402

METRICS = ["mse_20deg", "pearson_delta"]


def variant(r) -> str:
    parts = [r["method"]] + [v for v in (r["arm"], r["order"]) if v != "na"]
    return "_".join(parts)


def pair_key(df):
    return df["pert_a"] + "+" + df["pert_b"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--models", nargs="*", default=None, help="model tags; default: all model_*.csv")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", default=None, help="default: <results>/rollout_analysis.csv")
    args = p.parse_args()
    rdir = Path(args.results)

    base = pd.read_csv(rdir / "baselines.csv")
    add = base[base["method"] == "additive"].copy()
    files = sorted(rdir.glob("model_*.csv"))
    res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if res.empty:
        raise SystemExit(f"no model_*.csv in {rdir}")
    if args.models:
        res = res[res["model"].isin(args.models)]
    res["method"] = res.apply(variant, axis=1)

    eta = pd.read_csv(rdir / "eta.csv").assign(pair=lambda d: pair_key(d))
    eta_pair = eta.groupby("pair")[["eta_h1", "rel_eta_h1"]].mean()
    tert = eta_mod.assign_strata(eta_pair["eta_h1"], 3)

    out = []

    def put(model, var, metric, qty, d):
        out.append({"model": model, "variant": var, "metric": metric, "quantity": qty, **d})

    for model, mres in res.groupby("model"):
        pairs = set(pair_key(mres))
        # same pairs and same split seeds as the model, so averages are comparable
        ref = add[pair_key(add).isin(pairs) & add["split_seed"].isin(mres["split_seed"].unique())]
        ref = ref.assign(method="additive")
        both = pd.concat([mres, ref], ignore_index=True)
        variants = sorted(mres["method"].unique())
        for metric in METRICS:
            for v in variants:
                adv = stats.advantage(both, metric, v)
                put(model, v, metric, "advantage",
                    {**stats.bootstrap_mean(adv.to_numpy(), args.n_boot, 0),
                     "share_better": float((adv.dropna() > 0).mean())})
                put(model, v, metric, "slope_eta",
                    stats.slope_vs_eta(adv, eta_pair["eta_h1"], args.n_boot))
                put(model, v, metric, "slope_rel_eta",
                    stats.slope_vs_eta(adv, eta_pair["rel_eta_h1"], args.n_boot))
                for t in ("low", "mid", "high"):
                    a = adv.reindex(tert.index[tert == t]).dropna()
                    put(model, v, metric, f"tertile_{t}", stats.bootstrap_mean(a.to_numpy(), args.n_boot, 0))

            t = stats.per_pair(mres, metric)
            for arm in ("carried", "restated"):
                ab, ba = f"rollout_{arm}_ab", f"rollout_{arm}_ba"
                if ab in t and ba in t:
                    gap = (t[ab] - t[ba]).abs()
                    put(model, f"rollout_{arm}", metric, "order_gap",
                        stats.bootstrap_mean(gap.to_numpy(), args.n_boot, 0))
            if "rollout_carried_ab" in t and "rollout_restated_ab" in t:
                gap = t["rollout_carried_ab"] - t["rollout_restated_ab"]
                put(model, "carried_minus_restated", metric, "arm_gap",
                    stats.bootstrap_mean(gap.to_numpy(), args.n_boot, 0))

    df = pd.DataFrame(out)
    path = Path(args.out) if args.out else rdir / "rollout_analysis.csv"
    df.to_csv(path, index=False)

    pd.set_option("display.width", 160)
    show = df[(df["metric"] == "mse_20deg") & df["quantity"].isin(["advantage", "slope_eta", "slope_rel_eta"])]
    for model, d in show.groupby("model"):
        print(f"\n{model}: advantage over addition, MSE top-20 DEGs (positive = model better)")
        wide = d.assign(val=lambda x: x.apply(
            lambda r: f"{r['mean'] if pd.notna(r.get('mean')) else r['slope']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]",
            axis=1))
        print(wide.pivot(index="variant", columns="quantity", values="val").to_string())
        sb = d[d["quantity"] == "advantage"].set_index("variant")["share_better"]
        print("share of pairs better than addition:", ", ".join(f"{k} {v:.0%}" for k, v in sb.items()))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
