"""Train GEARS on Norman under either regime (reviewer point 4).

    standard   GEARS's own "simulation" split, which includes some doubles in
               training. The diagnostic arm: how good is GEARS as a joint
               predictor when trained the way its authors intended?
    strict     every single in training, every double held out (our design)

    python scripts/train_gears.py --split standard --seed 1
    python scripts/train_gears.py --split strict   --seed 1

First run downloads Norman to <data-dir>/norman/ (a few hundred MB). Writes the
model to models/gears_<split>_s<seed>/ plus rolloutprobe_split.json, which
run_model.py reads to exclude pairs the model saw in training.

STATUS: follows the documented cell-gears API but was not executed where this
scaffold was written. If a call signature differs in your installed version,
the GEARS README is the reference: https://github.com/snap-stanford/GEARS
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rolloutprobe import splits             # noqa: E402
from rolloutprobe.data import load_h5ad     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["standard", "strict"], required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data-dir", default="data/gears")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="models")
    args = p.parse_args()

    from gears import GEARS, PertData

    out = Path(args.out) / f"gears_{args.split}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    pert_data = PertData(args.data_dir)
    pert_data.load(data_name="norman")
    h5ad = Path(args.data_dir) / "norman" / "perturb_processed.h5ad"
    print(f"Norman at {h5ad}  (use this path as data.path in configs/default.yaml)")

    info = {"seed": args.seed}
    if args.split == "strict":
        data = load_h5ad(str(h5ad))
        strict = splits.strict_split(data)
        split_dict, demoted = splits.gears_custom_split(data, strict, val_frac=0.1, seed=args.seed)
        split_path = out / "split.pkl"
        splits.save_pickle(split_dict, str(split_path))
        pert_data.prepare_split(split="custom", split_dict_path=str(split_path))
        info.update(gears_split="custom", split_dict_path=str(split_path),
                    val_singles_in_doubles=demoted)
        print(f"strict split: {len(split_dict['train'])} train labels, {len(split_dict['val'])} val, "
              f"{len(split_dict['test'])} test doubles; {len(demoted)} val singles also appear in doubles")
    else:
        pert_data.prepare_split(split="simulation", seed=args.seed)
        info.update(gears_split="simulation")

    s2c = getattr(pert_data, "set2conditions", {}) or {}
    info["seen_conditions"] = list(s2c.get("train", [])) + list(s2c.get("val", []))
    info["test_conditions"] = list(s2c.get("test", []))

    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=128)
    model = GEARS(pert_data, device=args.device)
    model.model_initialize(hidden_size=args.hidden)
    model.train(epochs=args.epochs)
    model.save_model(str(out))
    (out / "rolloutprobe_split.json").write_text(json.dumps(info, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
