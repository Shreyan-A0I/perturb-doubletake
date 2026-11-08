"""Fine-tune pretrained scGPT (whole-human) on Norman singles, strict split.

    python scripts/train_scgpt.py --pretrained models/scgpt_human --seed 1

Training follows scGPT's Tutorial_Perturbation, with our own data loading instead
of GEARS's dataloader (whose batch format changed between cell-gears versions):

    input   a random control cell (its expression values)
    flags   1 on the perturbed gene(s) of the target cell's condition
    target  the perturbed cell's expression; MSE over the genes in the batch

Conditions: control + every single except 10 validation singles. The validation
singles are the ones GEARS's strict run held out (same function, same seed), all
from singles that appear in no double. No double is ever seen. The whole-human
weights are loaded for the encoder, value encoder and transformer; the
perturbation embedding and decoder start fresh, as in the tutorial.

Each step uses max_seq_len genes: every flagged gene in the batch plus a random
subset of the rest (the tutorial samples randomly and can drop the flagged gene;
we always keep it). Prediction uses all genes. Model selection: Pearson delta on
the validation singles, predicted from control cells.

Writes models/<name>_s<seed>/ (default name scgpt_strict): best_model.pt,
args.json, vocab.json, rolloutprobe_split.json, train_log.csv.

Part C (off by default; docs/TASKS.md says when to use it): --w-state and
--w-saturation add rows that start from real cells of training singles
(rolloutprobe/aux_batches.py):
    state       real A cell, no flag     -> an A cell
    saturation  real A cell, flag on A   -> an A cell
    loss = main MSE + w_state * state MSE + w_saturation * saturation MSE
Still singles only, so the strict split holds. Use a different --name, e.g.
    python scripts/train_scgpt.py --seed 1 --name scgpt_carry --w-state 1 --w-saturation 1

STATUS: written against the scGPT API but not executed where it was written. Read
docs/TASKS.md before running.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rolloutprobe import splits                 # noqa: E402
from rolloutprobe.config import load_config     # noqa: E402
from rolloutprobe.data import load_h5ad         # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--pretrained", default="models/scgpt_human",
                   help="folder with best_model.pt, args.json, vocab.json")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5, help="epochs without val improvement")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--val-cells", type=int, default=64, help="control cells per val prediction")
    p.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch (smoke test)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out", default="models")
    p.add_argument("--name", default="scgpt_strict", help="output folder is <out>/<name>_s<seed>")
    p.add_argument("--w-state", type=float, default=0.0, help="Part C: state-preservation loss weight")
    p.add_argument("--w-saturation", type=float, default=0.0, help="Part C: saturation loss weight")
    p.add_argument("--aux-batch-size", type=int, default=16, help="Part C: rows per auxiliary kind per step")
    args = p.parse_args()

    import torch
    from rolloutprobe.aux_batches import AuxSampler
    from rolloutprobe.models import scgpt_adapter as sa

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    amp = not args.no_amp and args.device.startswith("cuda")
    out = Path(args.out) / f"{args.name}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    # ---- data and split
    dcfg = load_config(args.config)["data"]
    data = load_h5ad(dcfg["path"], condition_col=dcfg.get("condition_col"), sep=dcfg.get("sep"),
                     normalize=dcfg.get("normalize", "auto"), n_top_genes=dcfg.get("n_top_genes"))
    X = data.X.tocsr() if hasattr(data.X, "tocsr") else np.asarray(data.X, dtype=np.float32)
    genes, n_genes = data.genes, len(data.genes)
    strict = splits.strict_split(data)
    split_dict, demoted = splits.gears_custom_split(data, strict, val_frac=0.1, seed=args.seed)
    val = sorted({splits.canonical_key(l) for l in split_dict["val"]})
    train = ["ctrl"] + [s for s in strict["train"] if s != "ctrl" and s not in val]
    print(f"strict split: {len(train) - 1} train singles + ctrl, {len(val)} val singles {val}, "
          f"{len(strict['test'])} test doubles (never used); {len(demoted)} val singles also in doubles")

    def rows(idx):
        sub = X[idx]
        return np.asarray(sub.toarray() if hasattr(sub, "toarray") else sub, dtype=np.float32)

    ctrl_idx = np.flatnonzero(data.cond == "ctrl")
    train_idx = np.flatnonzero(np.isin(data.cond, train))
    gpos = {g: i for i, g in enumerate(genes)}
    cell_flags = [[gpos[g] for g in c.split("+")] if c != "ctrl" else [] for c in data.cond]
    print(f"{len(train_idx)} training cells ({len(ctrl_idx)} control)")
    aux_w = {"state": args.w_state, "saturation": args.w_saturation}
    aux_w = {k: w for k, w in aux_w.items() if w > 0}
    if aux_w:
        aux = AuxSampler(data.cond, [s for s in train if s != "ctrl"])     # training singles only
        print(f"Part C auxiliary losses {aux_w}, {args.aux_batch_size} rows each per step, "
              f"from {len(aux.by_cond)} training singles")

    # ---- model
    vocab = sa.load_vocab(args.pretrained)
    token_ids_np, unknown = sa.gene_token_ids(genes, vocab)
    print(f"{n_genes - len(unknown)} of {n_genes} genes in the scGPT vocab; "
          f"{len(unknown)} get the pad token")
    cfg = sa.read_config(args.pretrained)
    model = sa.build_model(cfg, vocab)
    loaded = sa.load_weights(model, Path(args.pretrained) / "best_model.pt", prefixes=sa.LOAD_PREFIXES)
    n_pre = {pre: sum(k.startswith(pre) for k in loaded) for pre in sa.LOAD_PREFIXES}
    print(f"loaded {len(loaded)} pretrained tensors: {n_pre}")
    model.to(args.device)
    token_ids = torch.from_numpy(token_ids_np).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ---- validation: Pearson delta of predicted vs measured pseudobulk shift
    ctrl_mean = rows(ctrl_idx).mean(axis=0)
    val_X0 = rows(rng.choice(ctrl_idx, min(args.val_cells, len(ctrl_idx)), replace=False))
    val_target = {s: rows(np.flatnonzero(data.cond == s)).mean(axis=0) for s in val}

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        pred = sa.ScGPTPredictor(str(out), genes, device=args.device, batch_size=args.eval_batch_size,
                                 amp=amp, model=model, vocab=vocab)
        scores = []
        for s in val:
            d_pred = pred.predict(val_X0, (s,)).mean(axis=0) - val_X0.mean(axis=0)
            d_true = val_target[s] - ctrl_mean
            scores.append(np.corrcoef(d_pred, d_true)[0, 1])
        model.train()
        return float(np.nanmean(scores))

    # ---- training
    log, best, bad = [], -np.inf, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        order = rng.permutation(train_idx)
        n_steps = len(order) // args.batch_size
        if args.max_steps:
            n_steps = min(n_steps, args.max_steps)
        for step in range(n_steps):
            tgt = order[step * args.batch_size:(step + 1) * args.batch_size]
            src = rng.choice(ctrl_idx, len(tgt))
            row_flags = [cell_flags[i] for i in tgt]           # main rows: flag the target's genes
            group = [("main", len(tgt))]
            for kind in aux_w:                                   # Part C rows, appended
                s_i, t_i, fl = aux.sample(args.aux_batch_size, kind, rng)
                src, tgt = np.concatenate([src, s_i]), np.concatenate([tgt, t_i])
                row_flags += [[gpos[g] for g in genes_] for genes_ in fl]
                group.append((kind, len(s_i)))
            flagged = sorted({g for fl in row_flags for g in fl})
            rest = np.setdiff1d(np.arange(n_genes), flagged)
            k = max(0, min(args.max_seq_len, n_genes) - len(flagged))
            sel = np.sort(np.concatenate([flagged, rng.choice(rest, k, replace=False)])).astype(np.int64)

            flags = np.zeros((len(tgt), n_genes), dtype=np.int64)
            for r, fl in enumerate(row_flags):
                flags[r, fl] = 1
            v_in = torch.from_numpy(rows(src)[:, sel]).to(args.device)
            v_out = torch.from_numpy(rows(tgt)[:, sel]).to(args.device)
            f = torch.from_numpy(flags[:, sel]).to(args.device)
            sel_t = torch.from_numpy(sel).to(args.device)

            pred = sa.forward(model, token_ids[sel_t], v_in, f, amp)
            per_row = ((pred - v_out) ** 2).mean(dim=1)
            parts, start = {}, 0
            for kind, n in group:
                parts[kind] = per_row[start:start + n].mean()
                start += n
            loss = parts["main"] + sum(w * parts[kind] for kind, w in aux_w.items())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(parts["main"].item())
            if step % 200 == 0:
                extra = "".join(f", {kind} {parts[kind].item():.4f}" for kind in aux_w)
                print(f"  epoch {epoch} step {step}/{n_steps} main loss {parts['main'].item():.4f}{extra}",
                      flush=True)
        sched.step()

        score = validate()
        log.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_pearson_delta": score,
                    "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch}: train MSE {log[-1]['train_mse']:.4f}, val Pearson delta {score:.3f}, "
              f"{log[-1]['minutes']:.1f} min", flush=True)
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        if score > best:
            best, bad = score, 0
            torch.save(model.state_dict(), out / "best_model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs")
                break

    for f in ("args.json", "vocab.json"):
        shutil.copy(Path(args.pretrained) / f, out / f)
    info = {"seed": args.seed, "split": "strict", "pretrained": str(args.pretrained), "name": args.name,
            "aux_loss_weights": aux_w, "aux_batch_size": args.aux_batch_size if aux_w else 0,
            "val_singles": val, "val_singles_in_doubles": demoted, "best_val_pearson_delta": best,
            "seen_conditions": train + val, "test_conditions": [f"{a}+{b}" for a, b in strict["test"]],
            "genes_not_in_vocab": unknown}
    (out / "rolloutprobe_split.json").write_text(json.dumps(info, indent=2))
    print(f"best val Pearson delta {best:.3f}\nsaved {out}")


if __name__ == "__main__":
    main()
