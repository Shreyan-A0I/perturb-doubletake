"""Train CPA (cpa-tools 0.8.8) on Norman, strict split.

    python scripts/train_cpa.py --seed 1

Follows CPA's Norman tutorial (docs/tutorials/Norman.ipynb at commit fbd7c02):
the same model and trainer hyperparameters, max_epochs 2000, batch 2048, early
stopping patience 5 checked every 5 epochs. Deliberate differences, recorded in
rolloutprobe_split.json:

  data      the GEARS-processed file everything else uses (same 5,045 genes, same
            cells), which is log-normalized with no counts layer. The tutorial trains
            on raw counts with recon_loss="nb". Here recon_loss="gauss" on the
            log-normalized X (is_count_data=False), which CPA supports. Output is then
            in the same space as every other model's.
  split     train = control + the 95 training singles; ood (never used in training)
            = every double and the 10 validation singles held out from the other
            models (same gears_custom_split rule and seed: singles in no double).
            valid (early stopping only): by default a random 10% of the training
            conditions' cells. CPA cannot embed a gene it has never seen, so held-out
            singles would only measure an untrained embedding. --valid-mode singles
            uses the 10 held-out singles instead, as GEARS and scGPT do.
  covariate the tutorial's cell_type covariate is dropped (one cell line, K562).
  seed      CPA's model seed = --seed (the tutorial fixes 8206).

After training: checks that the adapter's direct path (module.get_expression) gives
the same predictions as the public CPA.predict on real control cells (counterfactual
setup from the tutorial). The result is written to rolloutprobe_split.json.

Writes models/<name>_s<seed>/: CPA's saved model with its AnnData (model.pt,
CPA_info.json, adata.h5ad, history.csv) and rolloutprobe_split.json.

STATUS: written against the cpa-tools source, never executed. Read docs/TASKS.md
(CPA steps) before running.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rolloutprobe import splits                 # noqa: E402
from rolloutprobe.config import load_config     # noqa: E402
from rolloutprobe.data import load_h5ad         # noqa: E402

# docs/tutorials/Norman.ipynb, cell 13, except recon_loss (counts unavailable) and seed
MODEL_PARAMS = {
    "n_latent": 32, "recon_loss": "gauss", "doser_type": "linear",
    "n_hidden_encoder": 256, "n_layers_encoder": 4, "n_hidden_decoder": 256, "n_layers_decoder": 2,
    "use_batch_norm_encoder": True, "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False, "use_layer_norm_decoder": False,
    "dropout_rate_encoder": 0.2, "dropout_rate_decoder": 0.0, "variational": False,
}
TRAINER_PARAMS = {
    "n_epochs_kl_warmup": None, "n_epochs_adv_warmup": 50, "n_epochs_mixup_warmup": 10,
    "n_epochs_pretrain_ae": 10, "mixup_alpha": 0.1, "lr": 0.0001, "wd": 3.2170178270865573e-06,
    "adv_steps": 3, "reg_adv": 10.0, "pen_adv": 20.0, "adv_lr": 0.0001, "adv_wd": 7.051355554517135e-06,
    "n_layers_adv": 2, "n_hidden_adv": 128, "use_batch_norm_adv": True, "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3, "step_size_lr": 25, "do_clip_grad": False, "adv_loss": "cce",
    "gradient_clip_value": 5.0,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--check-val-every", type=int, default=5)
    p.add_argument("--valid-mode", choices=["cells", "singles"], default="cells")
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--name", default="cpa_strict")
    p.add_argument("--out", default="models")
    args = p.parse_args()

    import anndata as ad
    import cpa
    from rolloutprobe.models.cpa_adapter import CPAPredictor
    from cpa._utils import CPA_REGISTRY_KEYS

    out = Path(args.out) / f"{args.name}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    dcfg = load_config(args.config)["data"]
    data = load_h5ad(dcfg["path"], condition_col=dcfg.get("condition_col"), sep=dcfg.get("sep"),
                     normalize=dcfg.get("normalize", "auto"), n_top_genes=dcfg.get("n_top_genes"))
    if len(set(data.genes)) != len(data.genes):
        raise ValueError("gene names are not unique")
    strict = splits.strict_split(data)
    split_dict, demoted = splits.gears_custom_split(data, strict, val_frac=0.1, seed=args.seed)
    val = sorted({splits.canonical_key(l) for l in split_dict["val"]})
    train = ["ctrl"] + [s for s in strict["train"] if s != "ctrl" and s not in val]

    in_train = np.isin(data.cond, train)
    split = np.where(in_train, "train", "ood").astype(object)
    if args.valid_mode == "cells":
        idx = np.flatnonzero(in_train)
        split[rng.choice(idx, int(round(args.valid_frac * len(idx))), replace=False)] = "valid"
    else:
        split[np.isin(data.cond, val)] = "valid"
    is_double = np.array(["+" in c for c in data.cond])
    assert not (is_double & (split != "ood")).any(), "a double reached train or valid"
    print(f"strict split: {len(train) - 1} train singles + ctrl, {len(val)} held-out singles {val}, "
          f"{len(strict['test'])} test doubles; cells: {pd.Series(split).value_counts().to_dict()}")

    adata = ad.AnnData(X=data.X, obs=pd.DataFrame({"condition": data.cond, "split": split},
                                                  index=[f"c{i}" for i in range(data.n_cells)]),
                       var=pd.DataFrame(index=data.genes))
    cpa.CPA.setup_anndata(adata, perturbation_key="condition", control_group="ctrl", dosage_key=None,
                          is_count_data=False, categorical_covariate_keys=[], max_comb_len=2)
    model = cpa.CPA(adata=adata, split_key="split", train_split="train", valid_split="valid",
                    test_split="ood", seed=args.seed, **MODEL_PARAMS)
    model.train(max_epochs=args.max_epochs, use_gpu=args.device.startswith("cuda"),
                batch_size=args.batch_size, plan_kwargs=TRAINER_PARAMS,
                early_stopping_patience=args.patience, check_val_every_n_epoch=args.check_val_every,
                save_path=str(out))
    model.save(str(out), overwrite=True, save_anndata=True)

    info = {"seed": args.seed, "split": "strict", "model": "cpa", "cpa_version": cpa.__version__,
            "source_commit": "fbd7c0250edc23eff003a10c99655579c53afd63",
            "recon_loss": "gauss", "input": "log-normalized X from the GEARS file (no counts layer)",
            "valid_mode": args.valid_mode, "val_singles": val, "val_singles_in_doubles": demoted,
            "seen_conditions": train, "test_conditions": [f"{a}+{b}" for a, b in strict["test"]],
            "model_params": {**MODEL_PARAMS, "seed": args.seed}, "trainer_params": TRAINER_PARAMS,
            "epochs_run": int(len(model.epoch_history)) if model.epoch_history is not None else None}

    # direct path (adapter) vs public CPA.predict, counterfactual setup as in the tutorial:
    # cells of a test double, X replaced by control cells, labels kept
    a, b = strict["test"][0]
    target = np.flatnonzero(data.cond == f"{a}+{b}")[:64]
    ctrl = rng.choice(np.flatnonzero(data.cond == "ctrl"), len(target), replace=False)
    Xc = data.X[ctrl]
    Xc = np.asarray(Xc.toarray() if hasattr(Xc, "toarray") else Xc, dtype=np.float32)
    sub = adata[target].copy()
    sub.X = Xc
    model.predict(sub, batch_size=len(target), n_samples=1)
    public = np.asarray(sub.obsm["CPA_pred"], dtype=np.float32)
    adapter = CPAPredictor(str(out), data.genes, device=args.device, module=model.module,
                           pert_encoder=dict(cpa.CPA.pert_encoder), keys=CPA_REGISTRY_KEYS,
                           max_comb_len=CPA_REGISTRY_KEYS.MAX_COMB_LENGTH,
                           trained_singles=[s for s in train if s != "ctrl"])
    direct = adapter.predict(Xc, (a, b))
    diff = float(np.abs(public - direct).max())
    info["public_predict_check"] = {"pair": f"{a}+{b}", "cells": int(len(target)), "max_abs_diff": diff}
    print(f"adapter vs CPA.predict on {a}+{b}: max |diff| = {diff:.2e}")
    (out / "rolloutprobe_split.json").write_text(json.dumps(info, indent=2))
    if not diff < 1e-4:
        raise SystemExit("adapter and CPA.predict disagree; do not use this model, report it")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
