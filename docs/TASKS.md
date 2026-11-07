# Tasks, step by step: rollout and scGPT

Post-midterm work, from the Nov 15 checkpoint to the final report. Steps 1–9 run on the GPU machine (the oc3 H100 environment used for GEARS). Run every command from the repo root, in order, and stop at any check that fails. Background and design rules: [README](../README.md) and [RUNBOOK](RUNBOOK.md). The midterm tasks (setup, baselines, figures, GEARS training, joint predictions, GEARS round-trip check) are done and recorded in the README log.

> [!WARNING]
> **Review the code before running anything (step 0).** `scripts/train_scgpt.py`, `src/rolloutprobe/models/scgpt_adapter.py` and `scripts/analyze_rollout.py` are new. The scGPT code was written against scGPT's API on a machine without torch or a GPU, so it has **never been executed**. Only its plumbing is tested, with a stand-in model. If you are an agent running this file: read every file listed in step 0 against the installed `scgpt` source before the first GPU run. Assume something was missed. If you find a bug, fix it in its own commit on your branch and say what you changed in the report. Never change the experimental design (split, validation singles, seeds, metrics, gate threshold) to make a step pass.

## Rules

1. **Work on a branch** named `<shreyan|datta>/<topic>`, for example `datta/scgpt`. Do not commit to `main` directly. Never force-push.
2. **Never commit data or model checkpoints.** `data/`, `models/`, `*.h5ad`, `*.pkl`, `*.pt` and `*.zip` stay ignored. `results/` and `figures/` are committed; keep each file under 5 MB. Run `git status --short` before every commit and check that only intended files are staged.
3. **Code changes happen in step 0 only.** After the first real run starts, do not edit `src/`, `scripts/` or `tests/`. If something fails, stop and report the full error.
4. **Commit messages are one line**: no body, no trailers. Use the existing git identity; do not change `user.name` or `user.email`.
5. **Only download what step 3 names** (the scGPT checkpoint, under 200 MB) and the packages in step 2. Ask before downloading anything else.

## Prerequisites (already done; verify only)

- The `doubletake` env with GEARS (torch, torch_geometric, cell-gears) works, and `pytest -q` passes.
- `data/gears/norman/perturb_processed.h5ad` exists, and the count command in the README prints `91205 cells 5045 genes 105 singles 131 doubles`.
- `models/gears_strict_s1/` and `models/gears_standard_s1/` exist, each with `rolloutprobe_split.json`.

If any of these is missing, stop and report. Do not retrain GEARS without asking.

## 0. Review the new code (before any GPU run)

Files: `src/rolloutprobe/models/scgpt_adapter.py`, `scripts/train_scgpt.py`, `scripts/analyze_rollout.py`, and the small `scgpt` additions to `scripts/run_model.py`, `scripts/roundtrip_check.py` and `src/rolloutprobe/wrapper.py`.

Check these against the installed package (after step 2), mainly `scgpt/model/generation_model.py` and `scgpt/tokenizer/gene_tokenizer.py`:

- [ ] `TransformerGenerator.__init__` accepts the arguments `build_model()` passes: `nlayers_cls`, `n_cls`, `vocab`, `dropout`, `pad_token`, `pad_value`, `pert_pad_id`, `use_fast_transformer`.
- [ ] `TransformerGenerator.forward(src, values, input_pert_flags, src_key_padding_mask, CLS, CCE, MVC, ECS)` returns a dict with `"mlm_output"` of shape (batch, genes).
- [ ] The perturbation-flag embedding has 3 rows (0 = not perturbed, 1 = perturbed, 2 = padding), so `PERT_PAD_ID = 2` is valid.
- [ ] `GeneVocab.from_file`, `append_token`, `set_default_index`, `in` and `[]` exist as used in `load_vocab()`.
- [ ] The checkpoint's `args.json` has the keys `read_config()` reads (`embsize`, `nheads`, `d_hid`, `nlayers`, `n_layers_cls`).
- [ ] `load_weights()` renames flash-attention keys (`Wqkv.*` → `in_proj_*`) because we run without flash-attn. Compare with `scgpt/utils/util.py` `load_pretrained`. Training must print a nonzero count for each of `encoder`, `value_encoder` and `transformer`, otherwise the model is not starting from pretrained weights.
- [ ] Training matches the scGPT perturbation tutorial: input is a random control cell, target is a perturbed cell, flags mark the target's genes, and the loss is MSE over the selected genes. Differences that are deliberate: our own data loading (not GEARS's dataloader), our strict split, and always keeping flagged genes in the gene subset.

## 1. Full GEARS rollout (`doubletake` env)

```bash
python scripts/run_model.py --model gears --model-dir models/gears_strict_s1   --tag gears_strict
python scripts/run_model.py --model gears --model-dir models/gears_standard_s1 --tag gears_standard
python scripts/analyze_rollout.py
python scripts/make_figures.py
```

This overwrites `results/model_gears_*.csv` with joint + rollout rows (the joint rows come out the same as before).

- **Check:** strict prints `excluded 3 doubles the model cannot perturb`. Standard also prints `scoring N doubles; M excluded as seen in training`.
- **Expected, from the midterm analysis** (GEARS computes out = x + g(pert), so its rollouts are fixed sums):
  - Carried rollout: close to the additive baseline, and `order_gap` near 0 for `rollout_carried`.
  - Restated rollout: worse than joint, with a clear `order_gap`.
  - Measured-input carried rollout behaves like carried.
- If the carried order gap is not near 0, stop and report: it would contradict the residual structure the round-trip check found.

## 2. scGPT environment (new env, GPU machine)

Keep scGPT out of `doubletake`: the `scgpt` package pins its own versions, which conflict with cell-gears.

```bash
conda create -n scgpt python=3.10 -y && conda activate scgpt
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torchtext==0.18.0 "numpy<2" -r requirements.txt gdown
pip install scgpt
```

- `scgpt` needs torchtext (its vocab class is built on it), and torchtext 0.18 needs torch 2.3. If `pip install scgpt` tries to replace torch, use `pip install scgpt --no-deps` and install whatever modules the check below reports missing. Do not install `flash-attn`. It is optional and hard to build, and the code runs without it (`use_fast_transformer=False`).
- **Check:**
  ```bash
  python -c "import torch; from scgpt.model import TransformerGenerator; from scgpt.tokenizer.gene_tokenizer import GeneVocab; print(torch.__version__, torch.cuda.is_available())"
  ```
  It must print `2.3.0+cu121 True`. A warning about flash-attn is fine.

## 3. Download the pretrained checkpoint

Whole-human checkpoint, under 200 MB: https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y

```bash
gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y -O models/scgpt_human
ls models/scgpt_human
```

- **Check:** `best_model.pt`, `args.json` and `vocab.json` sit directly in `models/scgpt_human/`. If gdown made a subfolder, move the three files up.
- If gdown is blocked (a Drive quota or permission error), download the folder in a browser and copy the three files to the same place.

## 4. Tests in the scGPT env

```bash
pytest -q
```

- **Check:** `10 passed`. That is the 7 pipeline tests plus 3 scGPT adapter tests, which only run where torch is installed. The CPU env without torch shows `7 passed, 1 skipped`.

## 5. Smoke test (a few minutes)

```bash
python scripts/train_scgpt.py --pretrained models/scgpt_human --epochs 1 --max-steps 20 --out models/smoke
python scripts/roundtrip_check.py --model scgpt --model-dir models/smoke/scgpt_strict_s1 --tag smoke --out /tmp/smoke_results
rm -rf models/smoke /tmp/smoke_results
```

- **Check, from the training log:**
  - `strict split: 95 train singles + ctrl, 10 val singles [...], 131 test doubles`
  - the number of genes found in the scGPT vocab (report it)
  - `loaded N pretrained tensors` with all three counts nonzero
  - a finite loss
- The round-trip check only has to finish without errors here; the verdict of a 20-step model means nothing.
- Report peak GPU memory (`nvidia-smi`). If inference runs out of memory, lower `--batch-size` (default 8) in the commands below.

## 6. Fine-tune scGPT

```bash
python scripts/train_scgpt.py --pretrained models/scgpt_human --seed 1
```

Defaults follow the scGPT tutorial: 15 epochs, lr 1e-4 decaying by 0.9 per epoch, batch 32, 1,536 genes per step. Training stops early after 5 epochs without improvement.

- **Check:**
  - It ends with `saved models/scgpt_strict_s1`, and that folder contains `best_model.pt`, `args.json`, `vocab.json`, `rolloutprobe_split.json` and `train_log.csv`.
  - The validation Pearson delta rises over the first epochs. For scale, GEARS's joint prediction on the test doubles reached 0.67.
- Copy `train_log.csv` into `results/scgpt_train_log.csv` so it gets committed.

## 7. Round-trip gate on scGPT

```bash
python scripts/roundtrip_check.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict
```

The verdict uses correlation on raw expression (> 0.95). GEARS passed that trivially (0.999), because mean expression levels dominate it. **The informative columns are `passthrough_real_delta_r` and `passthrough_generated_delta_r`** (shifts from control) **and `saturation_ratio`.** Report all of them.

A likely outcome to watch for: scGPT was trained to map unflagged input to a control cell. So real A-cells passed through with no flag may come back as control (raw r high, delta r near 0). That means the model "forgets" an unflagged state. It is a real finding for RQ3, not a bug.

- `PASS` or `GENERATED-INPUT FAIL`: continue to step 8. The measured-input arm is exactly what diagnoses generated-input failure.
- `REAL-INPUT FAIL`: stop and report. Composition cannot be tested for this model, and that is the result.
- Any verdict is a result. Do not change code or thresholds to make it pass.

## 8. scGPT joint and rollout predictions

Time one split seed first; scGPT reads all 5,045 genes per cell, so it is much slower than GEARS:

```bash
python scripts/run_model.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict --seeds 0
```

- **Check:**
  - it prints `split seed 0 done (131 pairs)`
  - no doubles are excluded (every perturbed gene is an expression gene)
  - it writes `results/model_scgpt_strict.csv`
- If one seed took under about 3 hours, rerun without `--seeds 0` to get all 5 (this overwrites the file). Otherwise keep seed 0, say so in the report, and note it in the README log. `analyze_rollout.py` compares against addition on the same seeds, so a 1-seed run is still a fair comparison.

Then:

```bash
python scripts/analyze_rollout.py
python scripts/make_figures.py
```

## 9. Record and push

1. In the README:
   - Tick finished checklist items, with initials and date
   - Add a Log line with the key numbers: carried vs restated advantage for GEARS, the scGPT gate verdict and delta r values, and scGPT joint and carried advantage with share of pairs better
   - Update "Last updated"
2. Commit and push:
   ```bash
   git add README.md results/ figures/
   git status --short        # only README.md, results/ and figures/ (plus any step-0 code fixes, committed separately)
   git commit -m "<one-line summary>"
   git push -u origin <your-branch>
   ```
3. Do not merge into `main`; Shreyan reviews and merges.

## Report back

Report in this order:

- Step 0: what you checked, and any bug fixed (file, line, what changed, commit hash)
- Envs and tests: versions printed in step 2, the `pytest -q` result
- Steps 1, 5–8: the printed output (split line, vocab coverage, pretrained tensor counts, the per-epoch log, the gate table and verdict, per-seed timing, the `analyze_rollout.py` summary)
- Git: branch, commit hash, push result
- Anything skipped or failed, and why

## Not written yet (development, after Nov 15)

These have no code yet. Each needs its own branch and a test on synthetic data before it runs on real data.

- **Stability experiment** (cut list, third): repeat the same perturbation for 1 to 10 steps from control cells, `x_{k+1} = predict(x_k, (A,))`, for the 5 largest singles. Report the saturation ratio per step, and the distance of the step-k pseudobulk from the nearest real condition's pseudobulk. Builds on `rollout.saturation_ratio`; the new script would be `scripts/stability.py`.
- **Discriminator drift metric** (cut list, first to go): train a classifier to tell real cells from predicted cells after k steps; accuracy near 0.5 means no drift. Skip it unless everything above is done.
- **Final report** (due Dec 3): copy `report/midterm/` to `report/final/`. Add the rollout results (from `results/rollout_analysis.csv`) to Experiments and Analysis, plus a Conclusion and Team Contributions, which the final rubric requires. The figure of advantage against eta per variant is still to be added to `scripts/make_figures.py`.
