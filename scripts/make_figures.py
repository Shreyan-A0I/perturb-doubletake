"""Midterm figures from results/. Works on real results or on schema.fake_results().

    python scripts/make_figures.py --results results --out figures
    python scripts/make_figures.py --fake --out figures/fake      # before anything has run

Fig 1  eta reliability: half-1 vs half-2 eta, and the cross-half signal estimate
Fig 2  null hierarchy: mean per method with 95% bootstrap CI over pairs
Fig 3  additive error vs half-1 eta (independent halves, so no circularity)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
from scipy import stats as sstats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rolloutprobe import schema, stats  # noqa: E402

ORDER = ["nochange", "mean", "additive", "lowrank_additive", "joint", "rollout", "rollout_measured"]
LABEL = {"nochange": "No change", "mean": "Train mean", "additive": "Additive",
         "lowrank_additive": "Low-rank additive", "joint": "Joint", "rollout": "Rollout",
         "rollout_measured": "Rollout (measured input)"}
MODEL_LABEL = {"gears_strict": "GEARS, strict split", "gears_standard": "GEARS, standard split"}
MODEL_ORDER = ["gears_strict", "gears_standard"]
GREY, ACCENT = "#6b6b6b", "#2f6db5"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "figure.dpi": 150})


def fig_eta(eta: pd.DataFrame, out: Path):
    s0 = eta[eta["split_seed"] == eta["split_seed"].min()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    ax1.scatter(s0["eta_h1"], s0["eta_h2"], s=12, color=ACCENT, alpha=0.75, lw=0)
    lim = [0, max(s0["eta_h1"].max(), s0["eta_h2"].max()) * 1.05]
    ax1.plot(lim, lim, color=GREY, lw=0.8, ls="--")
    ax1.set(xlim=lim, ylim=lim, xlabel="eta, cell half 1", ylabel="eta, cell half 2")
    rho = sstats.spearmanr(s0["eta_h1"], s0["eta_h2"]).statistic
    ax1.set_title(f"Eta replicates across cell halves (Spearman {rho:.2f})")

    pair = eta.groupby(["pert_a", "pert_b"])[["eta_h1", "eta2_cross"]].mean()
    ax2.scatter(pair["eta_h1"] ** 2, pair["eta2_cross"], s=12, color=ACCENT, alpha=0.75, lw=0)
    hi = (pair["eta_h1"] ** 2).max() * 1.05
    ax2.plot([0, hi], [0, hi], color=GREY, lw=0.8, ls="--", label="all signal")
    ax2.axhline(0, color=GREY, lw=0.8)
    frac = pair["eta2_cross"].mean() / (pair["eta_h1"] ** 2).mean()
    ax2.set(xlabel="measured eta squared (half 1)", ylabel="cross-half estimate of true eta squared")
    ax2.set_title(f"{frac:.0%} of measured non-additivity is signal")
    ax2.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, out / "fig1_eta_reliability")


def fig_hierarchy(res: pd.DataFrame, out: Path, n_boot: int):
    metrics_ = [("mse_20deg", "MSE, top-20 DEGs (lower is better)"),
                ("pearson_delta", "Pearson delta, all genes (higher is better)")]
    n_pairs = res.drop_duplicates(["model", "pert_a", "pert_b"]).groupby("model").size()
    n_all = n_pairs.get("none", n_pairs.max())

    def display(r):
        if r["model"] == "none":
            return LABEL.get(r["method"], r["method"])
        name = MODEL_LABEL.get(r["model"], r["model"])
        method = "" if r["method"] == "joint" else f": {LABEL.get(r['method'], r['method'])}"
        arm = "" if r["arm"] == "na" else f", {r['arm']}"
        n = n_pairs[r["model"]]
        return f"{name}{method}{arm}" + (f" ({n} pairs)" if n < n_all else "")

    def rank(r):
        m = r["model"]
        return (0 if m == "none" else 1,
                MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m,
                ORDER.index(r["method"]) if r["method"] in ORDER else 99, r["arm"])

    res = res.copy()
    keys = res[["model", "method", "arm"]].drop_duplicates()
    keys = keys.assign(label=keys.apply(display, axis=1), rank=keys.apply(rank, axis=1))
    res = res.merge(keys, on=["model", "method", "arm"])
    order = keys.sort_values("rank")["label"].tolist()
    res["method"] = res["label"]                      # stats groups by "method"

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.9 + 0.32 * len(order)))
    for ax, (m, title) in zip(axes, metrics_):
        tab = (stats.method_summary(res, m, n_boot).dropna(subset=["mean"])
                 .set_index("method").reindex([o for o in order]).dropna(subset=["mean"]))
        y = np.arange(len(tab))
        colors = [ACCENT if s == "Additive" else GREY for s in tab.index]
        ax.barh(y, tab["mean"], color=colors, alpha=0.85, height=0.6)
        ax.errorbar(tab["mean"], y, xerr=[tab["mean"] - tab["lo"], tab["hi"] - tab["mean"]],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
        ax.set_yticks(y, tab.index)
        ax.invert_yaxis()
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, out / "fig2_null_hierarchy")


def fig_additive_vs_eta(res: pd.DataFrame, eta: pd.DataFrame, out: Path):
    add = stats.per_pair(res, "mse_20deg")["additive"]
    e = eta.groupby(["pert_a", "pert_b"])["eta_h1"].mean()
    e.index = [f"{a}+{b}" for a, b in e.index]
    d = pd.concat([add.rename("mse"), e.rename("eta")], axis=1).dropna()
    rho = sstats.spearmanr(d["eta"], d["mse"]).statistic
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.scatter(d["eta"], d["mse"], s=12, color=ACCENT, alpha=0.75, lw=0)
    ax.set(xlabel="eta, cell half 1", ylabel="additive MSE, top-20 DEGs (half 2)", yscale="log")
    ax.set_title(f"Addition fails where eta is high\n(Spearman {rho:.2f}, independent halves)")
    fig.tight_layout()
    _save(fig, out / "fig3_additive_vs_eta")


def _save(fig, stem: Path):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--fake", action="store_true", help="use schema.fake_results()")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.fake:
        res = schema.fake_results()
        fig_hierarchy(res, out, args.n_boot)
        return
    res = pd.read_csv(Path(args.results) / "baselines.csv")
    extra = sorted(Path(args.results).glob("model_*.csv"))     # model results, same schema
    if extra:
        res = pd.concat([res, *[pd.read_csv(f) for f in extra]], ignore_index=True)
    eta = pd.read_csv(Path(args.results) / "eta.csv")
    fig_eta(eta, out)
    fig_hierarchy(res, out, args.n_boot)
    fig_additive_vs_eta(res, eta, out)


if __name__ == "__main__":
    main()
