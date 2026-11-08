"""Part B: is the measured interaction still present inside the model?

    python scripts/latent_additivity.py --model genes --tag genes                 # gene space only, CPU
    python scripts/latent_additivity.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict
    python scripts/latent_additivity.py --model cpa --model-dir models/cpa_strict_s1 --tag cpa_strict

Embeds real half-1 and half-2 cells of ctrl, A, B and AB for every strict test
pair, with no perturbation flags, and compares the noise-corrected relative
residual (rolloutprobe/latent.py) in each latent view with the same quantity in
gene space, computed from the same cells.

Views:
    genes                       mean expression (always computed)
    input_tokens, last_tokens   scGPT per-gene token states entering / leaving the
                                transformer, flattened (genes x d)
    input_pooled, last_pooled   the same, averaged over genes (d)
    basal                       CPA's basal latent z_basal = encoder(x), before the
                                perturbation embedding is added (n_latent)

GEARS is not supported on purpose: its input cell never enters a latent
(out = x + g(pert)), so there is nothing to probe.

Diagnostic for the write-up, not a gate. Two summaries matter:
    cross_half_signal   per view, mean over pairs of (r_1 . r_2) / (d_1 . d_2) with a
                        bootstrap CI and the share of pairs above 0. r_1 . r_2 has
                        expectation 0 when there is no interaction signal (independent
                        halves), so "CI excludes 0" is a test with a built-in null,
                        the same split-half logic as eta in gene space.
    ratio_to_input      last layer vs transformer input. scGPT's per-gene residual
                        stream carries each gene's input value, so the absolute
                        latent residual mostly reflects the input expression; the
                        change across layers is the informative part.

Writes results/latent_<tag>_pairs.csv (per pair, view, split seed) and
results/latent_<tag>.csv (pooled summaries with bootstrap CIs over pairs).
Embeddings are cached in --cache (default models/latent_cache/<tag>, ignored by
git), so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rolloutprobe import latent, splits, stats      # noqa: E402
from rolloutprobe.config import load_config        # noqa: E402
from rolloutprobe.data import load_h5ad            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", required=True, choices=["genes", "scgpt", "cpa"])
    p.add_argument("--tag", required=True)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-cells", type=int, default=300, help="cells per condition and half")
    p.add_argument("--seeds", type=int, nargs="*", default=[0], help="split seeds (halves)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--cache", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    out_dir = Path(args.out or cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_h5ad(args.data or dcfg["path"], condition_col=dcfg.get("condition_col"),
                     sep=dcfg.get("sep"), normalize=dcfg.get("normalize", "auto"),
                     n_top_genes=dcfg.get("n_top_genes"))
    pairs = splits.strict_split(data)["test"]
    conds = latent.needed_conditions(pairs)

    if args.model in ("scgpt", "cpa"):
        from rolloutprobe.wrapper import get_predictor
        kw = {"batch_size": args.batch_size} if args.model == "scgpt" else {}
        m = get_predictor(args.model, model_dir=args.model_dir, genes=data.genes, device=args.device, **kw)

        def embed(X):
            return {"genes": X.mean(axis=0), **m.embed_means(X)}
    else:
        def embed(X):
            return {"genes": X.mean(axis=0)}

    rows = []
    for seed in args.seeds:
        halves = splits.half_labels(data.cond, seed)
        cache = args.cache or f"models/latent_cache/{args.tag}"
        store = latent.LatentStore(f"{cache}/seed{seed}" if args.model != "genes" or args.cache else None)
        latent.fill_store(store, data.X, data.cond, halves, conds, embed, args.max_cells,
                          np.random.default_rng(seed))
        for a, b in pairs:
            for view, t in latent.pair_terms(store.get, a, b).items():
                rows.append({"model": args.tag, "pert_a": a, "pert_b": b, "split_seed": seed,
                             "view": view, **t})
        print(f"  split seed {seed} done ({len(pairs)} pairs, {len(conds)} conditions)")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"latent_{args.tag}_pairs.csv", index=False)

    # pooled over pairs (cross products averaged over split seeds first)
    pp = df.groupby(["view", "pert_a", "pert_b"])[["rr_cross", "dd_cross", "rep_cos"]].mean()
    views = list(dict.fromkeys(df["view"]))
    ref = pp.xs("genes", level="view")
    summ = []
    for v in views:
        t = pp.xs(v, level="view").reindex(ref.index)
        summ.append({"model": args.tag, "view": v, "quantity": "pooled_rel_cross",
                     "mean": latent.pooled_rel_cross(t["rr_cross"], t["dd_cross"]), "n": len(t)})
        frac = (t["rr_cross"] / t["dd_cross"]).replace([np.inf, -np.inf], np.nan).dropna()
        summ.append({"model": args.tag, "view": v, "quantity": "cross_half_signal",
                     **stats.bootstrap_mean(frac.to_numpy(), args.n_boot, 0),
                     "share_positive": float((frac > 0).mean())})
        summ.append({"model": args.tag, "view": v, "quantity": "median_rep_cos",
                     "mean": float(t["rep_cos"].median()), "n": len(t)})
        if v != "genes":
            summ.append({"model": args.tag, "view": v, "quantity": "ratio_to_genes",
                         **latent.pooled_ratio_boot(t["rr_cross"], t["dd_cross"], ref["rr_cross"],
                                                    ref["dd_cross"], args.n_boot)})
    for kind in ("tokens", "pooled"):           # does the transformer itself drop the interaction?
        if f"last_{kind}" in views:
            a_, b_ = pp.xs(f"last_{kind}", level="view"), pp.xs(f"input_{kind}", level="view")
            summ.append({"model": args.tag, "view": f"last_{kind}", "quantity": "ratio_to_input",
                         **latent.pooled_ratio_boot(a_["rr_cross"], a_["dd_cross"], b_["rr_cross"],
                                                    b_["dd_cross"], args.n_boot)})
    s = pd.DataFrame(summ)
    path = out_dir / f"latent_{args.tag}.csv"
    s.to_csv(path, index=False)
    pd.set_option("display.width", 160)
    print("\ncross_half_signal: mean (r1.r2)/(d1.d2) over pairs; 0 = no interaction signal (built-in null)."
          "\nratio_to_input: last layer vs transformer input (the change across layers is what is informative)")
    print(s.to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
