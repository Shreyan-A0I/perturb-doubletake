"""Reproducible versions of the A2/A3 scratch diagnostics (definitions in
src/rolloutprobe/diagnostics.py): carry, erasure decomposition and output-variance
ratios per model, and the interaction-blind A3 references (no model).

results/model_<tag>.csv holds only scores, not prediction vectors, so the model
diagnostics need inference. They reuse run_model.py's model constructors, pair list
(including the can_perturb exclusions) and cell draws: for a given split seed the
input cells are exactly those run_model.py used, so the recomputed sens_rel and
sens_null_rel must match model_<tag>.csv (checked and printed at the end).
Predictions needed per pair: f(Xa, B), plus f(X0, B) and f(X0b, B) once per B.

    python scripts/diagnostics.py --model gears --model-dir models/gears_strict_s1 --tag gears_strict
    python scripts/diagnostics.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict --batch-size 32
    python scripts/diagnostics.py --model cpa --model-dir models/cpa_strict_s1 --tag cpa_strict
    python scripts/diagnostics.py --model none            # references -> diagnostics_references.csv

Model runs default to split seed 0; the references default to all split seeds.
Writes results/diagnostics_<tag>.csv (one row per pair and split seed).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rolloutprobe import diagnostics as dg, splits, stats   # noqa: E402
from rolloutprobe.config import load_config                 # noqa: E402
from rolloutprobe.data import load_h5ad                     # noqa: E402

MODEL_COLS = ["carry", "carry_20deg", "carry_cos",
              "eras_corr", "eras_rel", "eras_resid",
              "eras_corr_20deg", "eras_rel_20deg", "eras_resid_20deg",
              "eras_corr_sample", "eras_rel_sample", "eras_resid_sample",
              "vr_ctrl", "vr_input_a", "vr_target", "vr_target_b"]
REF_COLS = ["forget_a_r_20deg", "shrink_sum_r_20deg", "forget_a_r_all", "shrink_sum_r_all",
            "forget_a_r_20deg_h2", "shrink_sum_r_20deg_h2", "forget_a_r_all_h2",
            "shrink_sum_r_all_h2", "rep_r_20deg", "rep_r_all"]


def dense(sub):
    return sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)


def summarize(df: pd.DataFrame, cols: list[str], n_boot: int) -> pd.DataFrame:
    """Per pair averaged over split seeds, then mean and 95% bootstrap CI over pairs."""
    pp = df.assign(pair=df["pert_a"] + "+" + df["pert_b"]).groupby("pair")[cols].mean()
    return pd.DataFrame([{"quantity": c, **stats.bootstrap_mean(pp[c].to_numpy(), n_boot, 0)}
                         for c in cols])


def fmt(r) -> str:
    return f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  n={r['n']}"


def run_references(args, data, pairs, seeds, k) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        pb = splits.pseudobulk_by_half(data, splits.half_labels(data.cond, seed))
        for a, b in pairs:
            rows.append({"pert_a": a, "pert_b": b, "split_seed": seed, **dg.reference_row(pb, a, b, k)})
    df = pd.DataFrame(rows)
    print(f"\nInteraction-blind A3 references, Pearson(reference, half-2 interaction r2); "
          f"{len(pairs)} pairs; references from half 1 unless *_h2")
    for seed in seeds:
        s = summarize(df[df["split_seed"] == seed], REF_COLS, args.n_boot)
        print(f"  split seed {seed}: " + "  ".join(f"{q} {m:+.3f}" for q, m in zip(s["quantity"], s["mean"])
                                                 if q in REF_COLS[:4]))
    print("  averaged over split seeds (per pair), 95% bootstrap CI over pairs:")
    for _, r in summarize(df, REF_COLS, args.n_boot).iterrows():
        print(f"    {r['quantity']:<24} {fmt(r)}")
    return df


def run_models(args, cfg, data, split, pairs, seeds, k) -> pd.DataFrame:
    import run_model    # same constructors and training view as the main experiment

    train_singles = [s for s in split["train"] if s != "ctrl"]
    skip, model_singles = run_model.model_training_view(args, train_singles)
    pairs = [pr for pr in pairs if f"{pr[0]}+{pr[1]}" not in skip]
    rng = np.random.default_rng(0)      # run_model.py's generator: same cell draws
    X = data.X
    rows, model = [], None
    for seed in seeds:
        halves = splits.half_labels(data.cond, seed)
        pb = splits.pseudobulk_by_half(data, halves)
        if model is None or args.model == "oracle_additive":
            t = time.time()
            model = run_model.build_model(args, data, pb, train_singles)
            print(f"model loaded in {time.time() - t:.0f} s")
            if hasattr(model, "can_perturb"):
                bad = [pr for pr in pairs if not model.can_perturb(pr)]
                if bad:
                    pairs = [pr for pr in pairs if pr not in bad]
                    print(f"excluded {len(bad)} doubles the model cannot perturb: {['+'.join(p) for p in bad]}")
        pred = dg.PredCache(model)

        def cells(cond, gen=rng, exclude=None):        # copy of run_model.py's sampler
            idx = np.flatnonzero((data.cond == cond) & (halves == 2))
            if exclude is not None:
                idx = np.setdiff1d(idx, exclude)
            idx = gen.choice(idx, min(args.max_cells, len(idx)), replace=False)
            return idx, dense(X[idx])

        def all_cells(cond):                            # variance reference only; no rng
            return dense(X[np.flatnonzero((data.cond == cond) & (halves == 2))])

        i0, X0 = cells("ctrl")
        _, X0b = cells("ctrl", gen=np.random.default_rng(10_000 + seed), exclude=i0)
        t = time.time()
        for i, (a, b) in enumerate(pairs):
            _, Xa = cells(a)
            r = dg.pair_diagnostics(pred, a, b, pb, X0, X0b, Xa, all_cells(f"{a}+{b}"), all_cells(b), k)
            rows.append({"model": args.tag, "pert_a": a, "pert_b": b,
                         "support": splits.support_tier((a, b), model_singles), "split_seed": seed, **r})
            if (i + 1) % 20 == 0:
                print(f"  seed {seed}: {i + 1}/{len(pairs)} pairs, {time.time() - t:.0f} s, "
                      f"{pred.calls} predict calls", flush=True)
        print(f"  split seed {seed} done: {len(pairs)} pairs, {pred.calls} predict calls, {time.time() - t:.0f} s")
    df = pd.DataFrame(rows)

    print(f"\n{args.tag}: diagnostics, split seeds {seeds}; mean and 95% bootstrap CI over pairs")
    for _, r in summarize(df, MODEL_COLS, args.n_boot).iterrows():
        print(f"  {r['quantity']:<20} {fmt(r)}")
    print(f"  per-gene variance (means over pairs): ctrl {df['var_ctrl'].mean():.4f}  "
          f"pred from ctrl {df['var_pred_ctrl'].mean():.5f}  pred from A {df['var_pred_a'].mean():.5f}  "
          f"real A+B {df['var_target_ab'].mean():.4f}")
    check_a2(df, Path(args.check or ROOT / "results" / f"model_{args.tag}.csv"))
    return df


def check_a2(df: pd.DataFrame, path: Path) -> None:
    """The A2 numbers recomputed here should equal run_model.py's (same cells, same model)."""
    if not path.exists():
        print(f"A2 consistency check skipped: {path} not found")
        return
    m = pd.read_csv(path)
    m = m[m["method"] == "input_sensitivity"][["pert_a", "pert_b", "split_seed", "sens_rel", "sens_null_rel"]]
    j = df.merge(m, on=["pert_a", "pert_b", "split_seed"], suffixes=("", "_run_model"))
    if j.empty:
        print(f"A2 consistency check: no matching rows in {path}")
        return
    print(f"A2 consistency check vs {path.name} ({len(j)} pair-seeds):")
    for c in ("sens_rel", "sens_null_rel"):
        d = (j[c] - j[f"{c}_run_model"]).abs()
        rel = d / j[f"{c}_run_model"].abs()
        print(f"  {c}: max |diff| {d.max():.2e}, max rel diff {rel.max():.2e}, "
              f"mean here {j[c].mean():.4f} vs run_model {j[f'{c}_run_model'].mean():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", required=True, choices=["gears", "scgpt", "cpa", "oracle_additive", "none"],
                   help="'none' computes the interaction-blind references (no model)")
    p.add_argument("--tag", default=None, help="model tag; default 'references' for --model none")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--gears-data-dir", default="data/gears")
    p.add_argument("--data", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8, help="scGPT cells per forward pass")
    p.add_argument("--max-cells", type=int, default=300, help="input cells per prediction (as run_model)")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="split seeds; default 0 for models, all config seeds for references")
    p.add_argument("--n-boot", type=int, default=None, help="default: analysis.n_boot from the config")
    p.add_argument("--check", default=None, help="model CSV for the A2 check; default results/model_<tag>.csv")
    p.add_argument("--out", default=None, help="override output.results_dir")
    args = p.parse_args()
    if args.model == "none":
        args.tag = args.tag or "references"
    elif not args.tag:
        p.error("--tag is required with a model")

    cfg = load_config(args.config)
    dcfg, acfg = cfg["data"], cfg["analysis"]
    args.n_boot = args.n_boot or acfg["n_boot"]
    out_dir = Path(args.out or cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = load_h5ad(args.data or dcfg["path"], condition_col=dcfg.get("condition_col"),
                     sep=dcfg.get("sep"), normalize=dcfg.get("normalize", "auto"),
                     n_top_genes=dcfg.get("n_top_genes"))
    split = splits.strict_split(data)
    pairs, k = split["test"], acfg["top_k_deg"]

    if args.model == "none":
        seeds = args.seeds if args.seeds is not None else acfg["split_seeds"]
        df = run_references(args, data, pairs, seeds, k)
    else:
        seeds = args.seeds if args.seeds is not None else [0]
        df = run_models(args, cfg, data, split, pairs, seeds, k)

    path = out_dir / f"diagnostics_{args.tag}.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"\nwrote {path} ({path.stat().st_size / 1e6:.2f} MB), total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
