"""Experiment 1 for any registered model: joint, rollout (both arms, both orders),
rollout from measured input, the model's own singles, input sensitivity (A2) and
interaction recovery (A3); see rolloutprobe/pair_eval.py. Writes schema rows to results/model_<tag>.csv,
which make_figures.py picks up automatically next to the nulls.

Uses the same split-half design as the nulls:
    input cells     control cells from half 2 (and A-perturbed cells from half 2
                    for the measured-input arm)
    target          half-2 pseudobulk of the double
    DEGs, eta       half 1

    python scripts/run_model.py --model gears --model-dir models/gears_strict_s1 --tag gears_strict
    python scripts/run_model.py --model oracle_additive --data data/synthetic/norman_like.h5ad --tag oracle
    python scripts/run_model.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict
    python scripts/run_model.py --model cpa --model-dir models/cpa_strict_s1 --tag cpa_strict
    python scripts/run_model.py ... --joint-only        # midterm: skip rollout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rolloutprobe import eta as eta_mod                         # noqa: E402
from rolloutprobe import pair_eval, schema, splits               # noqa: E402
from rolloutprobe.config import load_config                     # noqa: E402
from rolloutprobe.data import load_h5ad                         # noqa: E402
from rolloutprobe.wrapper import get_predictor                  # noqa: E402


def build_model(args, data, pb, train_singles):
    if args.model == "oracle_additive":
        c = pb[("ctrl", 2)]
        return get_predictor("oracle_additive", genes=data.genes,
                             single_deltas={s: pb[(s, 2)] - c for s in train_singles})
    if args.model == "gears":
        from rolloutprobe.models.gears_adapter import load_pert_data
        info = json.loads((Path(args.model_dir) / "rolloutprobe_split.json").read_text())
        pert_data = load_pert_data(args.gears_data_dir, split=info["gears_split"], seed=info["seed"],
                                   split_dict_path=info.get("split_dict_path"))
        m = get_predictor("gears", pert_data=pert_data, model_dir=args.model_dir, device=args.device)
        if m.genes != data.genes:
            raise ValueError("GEARS gene order differs from the analysis data. Point --data at the "
                             "same perturb_processed.h5ad that GEARS trained on.")
        return m
    if args.model == "scgpt":
        return get_predictor("scgpt", model_dir=args.model_dir, genes=data.genes, device=args.device,
                             batch_size=args.batch_size)
    if args.model == "cpa":          # own batch size (small MLP); --batch-size is scGPT's
        return get_predictor("cpa", model_dir=args.model_dir, genes=data.genes, device=args.device)
    raise ValueError(args.model)


def model_training_view(args, all_singles: list[str]) -> tuple[set[str], set[str]]:
    """(doubles the model trained on, singles the model trained on).
    Doubles it saw are excluded from scoring; singles it saw set the support tier."""
    if args.model not in ("gears", "scgpt", "cpa"):
        return set(), set(all_singles)
    info = json.loads((Path(args.model_dir) / "rolloutprobe_split.json").read_text())
    seen = {splits.canonical_key(c) for c in info.get("seen_conditions", [])}
    seen.discard(None)
    return {k for k in seen if "+" in k}, {k for k in seen if "+" not in k and k != "ctrl"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", required=True, choices=["gears", "scgpt", "cpa", "oracle_additive"])
    p.add_argument("--tag", required=True, help="model column value, e.g. gears_strict")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--gears-data-dir", default="data/gears")
    p.add_argument("--data", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8, help="scGPT cells per forward pass")
    p.add_argument("--max-cells", type=int, default=300, help="input cells per prediction")
    p.add_argument("--joint-only", action="store_true")
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="override split seeds")
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
    skip, model_singles = model_training_view(args, train_singles)
    pairs = [pr for pr in split["test"] if f"{pr[0]}+{pr[1]}" not in skip]
    if skip:
        print(f"scoring {len(pairs)} doubles; {len(split['test']) - len(pairs)} excluded as seen in training")

    rng = np.random.default_rng(0)
    X = data.X
    rows, model = [], None
    for seed in (args.seeds if args.seeds is not None else acfg["split_seeds"]):
        halves = splits.half_labels(data.cond, seed)
        pb = splits.pseudobulk_by_half(data, halves)
        if model is None or args.model == "oracle_additive":   # oracle is rebuilt per half-split
            model = build_model(args, data, pb, train_singles)
            if hasattr(model, "can_perturb"):
                unmodelable = [pr for pr in pairs if not model.can_perturb(pr)]
                if unmodelable:
                    pairs = [pr for pr in pairs if pr not in unmodelable]
                    print(f"excluded {len(unmodelable)} doubles the model cannot perturb: "
                          f"{['+'.join(pr) for pr in unmodelable]}")

        def cells(cond, gen=rng, exclude=None):
            idx = np.flatnonzero((data.cond == cond) & (halves == 2))
            if exclude is not None:
                idx = np.setdiff1d(idx, exclude)
            idx = gen.choice(idx, min(args.max_cells, len(idx)), replace=False)
            sub = X[idx]
            return idx, (sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub))

        i0, X0 = cells("ctrl")
        # second control sample for the A2 null; own generator so the draws above and
        # below (and hence every pre-existing result) are unchanged
        _, X0b = cells("ctrl", gen=np.random.default_rng(10_000 + seed), exclude=i0)
        et = eta_mod.eta_table(pb, pairs)
        eta_h1 = dict(zip(zip(et["pert_a"], et["pert_b"]), et["eta_h1"]))

        for a, b in pairs:
            base = {"model": args.tag, "pert_a": a, "pert_b": b,
                    "support": splits.support_tier((a, b), model_singles),
                    "split_seed": seed, "eta_h1": eta_h1[(a, b)]}
            # Predictions are anchored to the half-2 pseudobulk of the condition their
            # input cells came from. Removes the sampling noise of the ~300 input cells,
            # which the nulls (built on full pseudobulks) do not carry, so models are not
            # penalized for it. Under an exactly additive model, carried rollout then
            # equals the additive null to floating-point precision (tests check this).
            _, Xa = cells(a)
            for r in pair_eval.evaluate_pair(model, a, b, pb, X0, X0b, Xa, acfg["top_k_deg"],
                                             joint_only=args.joint_only):
                rows.append({**base, **r})
        print(f"  split seed {seed} done ({len(pairs)} pairs)")

    res = schema.validate(pd.DataFrame(rows))
    path = out_dir / f"model_{args.tag}.csv"
    res.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
