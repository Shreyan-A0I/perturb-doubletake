"""Gate 1 (reviewer point 2): a full input -> output -> input example per model.
No model joins the roster until this passes.

    python scripts/roundtrip_check.py --model gears --model-dir models/gears_strict_s1 --tag gears_strict
    python scripts/roundtrip_check.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict
    python scripts/roundtrip_check.py --model oracle_additive --data data/synthetic/norman_like.h5ad --tag oracle

For each test gene A, reports:
    space          ranges of control input vs predicted output (same space?)
    real r         real A-perturbed cells passed through with no perturbation;
                   does the model return them roughly unchanged? (want > 0.95)
    real delta r   the same, on shifts from control: the harder version
    gen r / delta  the same, for the model's own predicted A state
    saturation     ||x2 - x1|| / ||x1 - x0|| applying A twice (consistency check only;
                   Norman cannot validate it biologically)

Reading the result: fail on "real" = the model cannot take non-control input at
all, so composition cannot be tested and that is the finding. Pass on "real" but
fail on "gen" = generated-input distribution shift.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rolloutprobe import rollout, splits        # noqa: E402
from rolloutprobe.config import load_config     # noqa: E402
from rolloutprobe.data import load_h5ad         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_model import build_model               # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", required=True, choices=["gears", "scgpt", "oracle_additive"])
    p.add_argument("--tag", required=True)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--gears-data-dir", default="data/gears")
    p.add_argument("--data", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8, help="scGPT cells per forward pass")
    p.add_argument("--genes", nargs="*", default=None, help="default: 5 singles with most cells")
    p.add_argument("--max-cells", type=int, default=300)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    out_dir = Path(args.out or cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_h5ad(args.data or dcfg["path"], condition_col=dcfg.get("condition_col"),
                     sep=dcfg.get("sep"), normalize=dcfg.get("normalize", "auto"),
                     n_top_genes=dcfg.get("n_top_genes"))
    singles = data.singles()
    genes = args.genes or data.cell_counts()[singles].sort_values(ascending=False).index[:5].tolist()

    halves = splits.half_labels(data.cond, 0)
    pb = splits.pseudobulk_by_half(data, halves)
    model = build_model(args, data, pb, singles)
    rng = np.random.default_rng(0)

    def cells(cond):
        idx = np.flatnonzero((data.cond == cond) & (halves == 2))
        idx = rng.choice(idx, min(args.max_cells, len(idx)), replace=False)
        sub = data.X[idx]
        return sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)

    X0 = cells("ctrl")
    rows = []
    for g in genes:
        r = rollout.roundtrip_check(model, X0, cells(g), g)
        r["saturation_ratio"] = rollout.saturation_ratio(model, X0, g)
        rows.append(r)
    df = pd.DataFrame(rows)

    cols = ["gene", "in_mean", "out_mean", "passthrough_real_r", "passthrough_real_delta_r",
            "passthrough_generated_r", "passthrough_generated_delta_r", "saturation_ratio"]
    print(df[cols].round(3).to_string(index=False))
    verdict = ("PASS" if (df["passthrough_real_r"] > 0.95).all() and (df["passthrough_generated_r"] > 0.95).all()
               else "REAL-INPUT FAIL" if (df["passthrough_real_r"] <= 0.95).any() else "GENERATED-INPUT FAIL")
    print(f"\nverdict: {verdict}")
    path = out_dir / f"roundtrip_{args.tag}.json"
    path.write_text(json.dumps({"verdict": verdict, "rows": rows}, indent=2, default=float))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
