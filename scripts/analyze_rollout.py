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

A1/A2/A3 (when the model file has the columns; see rolloutprobe/pair_eval.py):
    single_fit_in_sample   the model's own singles vs measured singles, Pearson delta
    interaction_r          A3: Pearson(implied, measured half-2 interaction), top-20
                           DEGs and all genes, printed next to its zero (additive
                           baseline, 0 by construction), the replicate correlation
                           (half 1 vs half 2) and the noiseless ceiling ~ sqrt(replicate)
    interaction_size       ||implied interaction|| / ||predicted double shift||
    sensitivity            A2: sens_corr, sens_rel, sens_null_rel, and the paired
                           excess sens_rel - sens_null_rel

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
NOT_PREDICTIONS = ("single", "input_sensitivity")   # rows that are not predictions of the double


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
        variants = sorted(v for v in mres["method"].unique() if not v.startswith(NOT_PREDICTIONS))
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

        boot = lambda x: stats.bootstrap_mean(np.asarray(x, dtype=float), args.n_boot, 0)  # noqa: E731
        mp = mres.assign(pair=pair_key(mres))

        # A1: the model's own singles against the measured singles. In-sample: models
        # train on singles. Bad fit here contaminates every implied interaction below.
        sing = mp[mp["method"].str.startswith("single")]
        if len(sing):
            gene = np.where(sing["order"] == "a", sing["pert_a"], sing["pert_b"])
            fit = sing.assign(gene=gene).groupby("gene")["pearson_delta"].mean()
            put(model, "single", "pearson_delta", "single_fit_in_sample", boot(fit))

        # A3: interaction recovery, always next to its zero and its ceiling
        if "int_r_all" in mp.columns and mp["int_r_all"].notna().any():
            pp = mp.groupby(["pair", "method"])[["int_r_20deg", "int_r_all", "int_rel"]].mean()
            rep = mp.groupby("pair")[["rep_r_20deg", "rep_r_all"]].mean()
            for g in ("20deg", "all"):
                col = f"int_r_{g}"
                r = rep[f"rep_r_{g}"]
                put(model, "additive (zero)", col, "interaction_r",
                    {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": int(len(r))})
                put(model, "replicate (half 1 vs 2)", col, "interaction_r", boot(r))
                put(model, "noiseless ceiling", col, "interaction_r", boot(np.sqrt(r.clip(lower=0))))
                for v in variants:
                    x = pp.xs(v, level="method")[col]
                    put(model, v, col, "interaction_r", {**boot(x), "share_positive": float((x.dropna() > 0).mean())})
            for v in variants:
                put(model, v, "int_rel", "interaction_size", boot(pp.xs(v, level="method")["int_rel"]))

        # A2: input sensitivity, against the control-vs-control sampling null
        sens = mp[mp["method"] == "input_sensitivity"]
        if len(sens):
            sp = sens.groupby("pair")[["sens_corr", "sens_rel", "sens_null_rel"]].mean()
            for c in sp.columns:
                put(model, "input_sensitivity", c, "sensitivity", boot(sp[c]))
            put(model, "input_sensitivity", "sens_rel_minus_null", "sensitivity",
                {**boot(sp["sens_rel"] - sp["sens_null_rel"]),
                 "share_positive": float(((sp["sens_rel"] - sp["sens_null_rel"]).dropna() > 0).mean())})

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
    fmt = lambda r: f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"  # noqa: E731
    a3 = df[df["quantity"] == "interaction_r"]
    for model, d in a3.groupby("model"):
        print(f"\n{model}: A3 interaction recovery, Pearson(implied, measured half-2 interaction)")
        print("  zero = additive baseline; replicate = what a second noisy measurement reaches;"
              " noiseless ceiling ~ sqrt(replicate)")
        print(d.assign(val=d.apply(fmt, axis=1)).pivot(index="variant", columns="metric", values="val")
               .to_string())
    a2 = df[df["quantity"] == "sensitivity"]
    for model, d in a2.groupby("model"):
        print(f"\n{model}: A2 input sensitivity (effect of B from real A cells vs from control cells)")
        print("  sens_rel = ||e(A cells) - e(ctrl)|| / ||e(ctrl)||; sens_null_rel = same between two control"
              " samples; 0 = prediction ignores the input cell")
        for _, r in d.iterrows():
            print(f"  {r['metric']:<20} {fmt(r)}" + (f"  share of pairs > 0: {r['share_positive']:.0%}"
                                                    if pd.notna(r.get("share_positive")) else ""))
    sf = df[df["quantity"] == "single_fit_in_sample"]
    for _, r in sf.iterrows():
        print(f"\n{r['model']}: own singles vs measured singles (in-sample), Pearson delta {fmt(r)}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
