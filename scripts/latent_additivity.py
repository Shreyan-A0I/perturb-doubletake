"""Part B: is the measured interaction still present inside the model?

    python scripts/latent_additivity.py --model genes --tag genes                 # gene space only, CPU
    python scripts/latent_additivity.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict

Embeds real half-1 and half-2 cells of ctrl, A, B and AB for every strict test
pair, with no perturbation flags, and compares the noise-corrected relative
residual (rolloutprobe/latent.py) in each latent view with the same quantity in
gene space, computed from the same cells.

Views:
    genes                       mean expression (always computed)
    input_tokens, last_tokens   scGPT per-gene token states entering / leaving the
                                transformer, flattened (genes x d)
    input_pooled, last_pooled   the same, averaged over genes (d)

GEARS is not supported on purpose: its input cell never enters a latent
(out = x + g(pert)), so there is nothing to probe.

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

from rolloutprobe import latent, splits             # noqa: E402
from rolloutprobe.config import load_config        # noqa: E402
from rolloutprobe.data import load_h5ad            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", required=True, choices=["genes", "scgpt"])
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

    if args.model == "scgpt":
        from rolloutprobe.wrapper import get_predictor
        m = get_predictor("scgpt", model_dir=args.model_dir, genes=data.genes, device=args.device,
                          batch_size=args.batch_size)

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
    print("\nrelative residual = share of the double's shift that is non-additive, noise-corrected "
          "(cross-half); ratio < 1 = the view keeps less of the interaction than gene space")
    print(s.to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
