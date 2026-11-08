# Tasks, step by step: rollout and scGPT

Post-midterm work, from the Nov 15 checkpoint to the final report. Steps 1–12 run on the GPU machine (the oc3 H100 environment used for GEARS). Run every command from the repo root, in order, and stop at any check that fails. Background and design rules: [README](../README.md) and [RUNBOOK](RUNBOOK.md). The midterm tasks (setup, baselines, figures, GEARS training, joint predictions, GEARS round-trip check) are done and recorded in the README log.

`run_model.py` now also records three things per pair (definitions in `src/rolloutprobe/pair_eval.py`), so the GEARS rerun in step 1 and the first scGPT run in step 8 produce them in the same pass:

- **A1. The model's own singles** from control cells. Both analyses below are built from them.
- **A2. Input sensitivity: does the predicted effect of B depend on which cell it starts from?** It compares the predicted shift of B from real A-cells with the shift from control cells (`sens_rel`), against the same comparison between two control samples (`sens_null_rel`, what cell sampling alone produces). **A2 is the gate for whether composition can be tested at all** (step 8).
- **A3. Interaction recovery: does the model get the non-additive part right?** It is the Pearson correlation between the interaction a prediction implies, D_pred(AB) − D_pred(A) − D_pred(B), and the measured half-2 interaction, on the top-20 DEGs and on all genes. Its zero is the additive baseline, which is 0 by construction. Its references are the half-1 vs half-2 replicate correlation and the noiseless ceiling, about √replicate.

## GPU budget

There are about 1–2 days of H100 hours, spread across sessions. The plan below uses roughly 10–25 of them. These timings are estimates from compute counts, not measurements. Replace them with the real numbers from step 5 and from seed 0 in step 8.

| Run | Estimate per run | Runs | Total |
|---|---|---|---|
| GEARS full rollout (step 1) | under 1 h | 2 | 1–2 h |
| scGPT fine-tune (step 6), at most 15 epochs of about 1,630 steps | 1–2 h | 3 training seeds | 3–6 h |
| scGPT `run_model.py`, one split seed: 131 pairs × about 3,000 cell predictions on all 5,045 genes | 20–60 min | 3 training seeds × 5 split seeds | 5–15 h |
| CPA (step 8b), in the idle window while the step 8 report waits: env, training, 5 split seeds, Part B | about 1 h training; the rest minutes (small MLP, not a transformer) | 1 | 1–2 h |
| Stability and drift (step 10), scGPT and GEARS | under 15 min | 2 models | under 0.5 h |
| Part B diagnostic (step 11): about 237 conditions × 2 halves × up to 300 cells | under 1 h | 1 | under 1 h |
| Part C (step 12), **not planned by default**: one fine-tune plus 5 split seeds | 3–7 h | 0 or 1 | 0–7 h |

Two additions use the spare hours, and neither changes the design:

1. **Three scGPT fine-tuning runs instead of one** (training seeds 1, 2, 3). A2 and A3 come from a single trained network. Without replicate runs, there is no way to tell a property of scGPT on Norman apart from the outcome of one training run. A second and third run are needed whatever A2 shows: a null result needs replication as much as a positive one.
   - **Caveat:** the seed changes both the training run (initialisation, batch order) and which 10 singles are held out for validation. If the runs disagree, the two causes cannot be separated, so report the disagreement without attributing it to initialisation. For A2 this is unlikely to matter: whether the output depends on the input is structural, and ten validation singles should not move it.
2. **All 5 split seeds for every scGPT run**, by default rather than only when time allows. This puts scGPT on the same footing as the baselines and GEARS.

**The inference estimate is the uncertain line.** Prediction reads all 5,045 genes while training reads 1,536. Attention cost grows with the square of sequence length, so a prediction costs about 11× a training cell in attention. The estimate assumes PyTorch's own fused attention kernel handles this in fp16; it could fall back to the plain one, since flash-attn is not installed.
- **Get an early timing signal from step 7.** The round-trip check predicts on about 1.7% of a split seed's cells, so its run time × 60 approximates one split seed. Report it. If it implies more than 2 hours per split seed, say so before step 8.

**If time runs short, cut split seeds, not cells.** Lowering `--max-cells` looks like free savings, but `sens_null_rel` measures exactly the noise from sampling different cells. Fewer cells raise the A2 null floor, which makes the gate less able to detect a real effect. The fallback is: training seed 1 on all 5 split seeds, and training seeds 2 and 3 on split seed 0 only. Report that choice.

## How to report results

The study's value is that each claim is grounded. For every result:

- **State the question first** and what each possible outcome would mean. If two outcomes would lead to the same next step, the experiment is not worth running.
- **Never give a number without its reference point.** That means its zero (usually the additive baseline, by construction) and its ceiling where one exists (measurement noise caps most metrics well below 1.0).
- **Compare with the prediction written below.** If the result differs, that is the interesting part. Do not adjust the framing to fit it.
- **Say which of three failures a bad result supports**, and what evidence separates them:
  1. **Architecture:** the model cannot do it. GEARS's predicted shift does not read the input cell.
  2. **Data:** the training data cannot teach it. In Norman, every training input is a control cell, so nothing teaches how a perturbation acts on an already-perturbed cell.
  3. **Unfair test:** the test penalises the model for something else. See step 7.
- **Report negative and null results plainly.** "No, and here is the mechanism" is the expected outcome of this project.

> [!WARNING]
> **Review the code before running anything (step 0).** `scripts/train_scgpt.py`, `src/rolloutprobe/models/scgpt_adapter.py`, `src/rolloutprobe/pair_eval.py`, `scripts/analyze_rollout.py`, the step 10–12 code (listed in step 0) and the CPA code (step 8b) are new. The scGPT code was written against scGPT's API on a machine without the `scgpt` package or a GPU, so it has **never run with the real scGPT model**. The CPA code was written against the cpa-tools source and has never run with cpa-tools either. Only its plumbing is tested, with a stand-in network. If you are an agent running this file: read every file listed in step 0 against the installed `scgpt` source before the first GPU run. Assume something was missed. If you find a bug, fix it in its own commit on your branch and say what you changed in the report. Never change the experimental design (split, validation singles, seeds, metrics, gate threshold) to make a step pass.

## Rules

1. **Work on a branch** named `<shreyan|datta>/<topic>`, for example `datta/scgpt`. Do not commit to `main` directly. Never force-push.
2. **Never commit data or model checkpoints.** `data/`, `models/`, `*.h5ad`, `*.pkl`, `*.pt` and `*.zip` stay ignored. `results/` and `figures/` are committed; keep each file under 5 MB. Run `git status --short` before every commit and check that only intended files are staged.
3. **Code changes happen in step 0 only.** After the first real run starts, do not edit `src/`, `scripts/` or `tests/`. If something fails, stop and report the full error.
4. **Commit messages are one line**: no body, no trailers. Use the existing git identity; do not change `user.name` or `user.email`.
5. **Only download what step 3 names** (the scGPT checkpoint, under 200 MB) and the packages in steps 2 and 8b (C1). Ask before downloading anything else.

## Prerequisites (already done; verify only)

- The `doubletake` env with GEARS (torch, torch_geometric, cell-gears) works, and `pytest -q` passes.
- `data/gears/norman/perturb_processed.h5ad` exists, and the count command in the README prints `91205 cells 5045 genes 105 singles 131 doubles`.
- `models/gears_strict_s1/` and `models/gears_standard_s1/` exist, each with `rolloutprobe_split.json`.

If any of these is missing, stop and report. Do not retrain GEARS without asking.

## 0. Review the new code (before any GPU run)

Files: `src/rolloutprobe/models/scgpt_adapter.py`, `scripts/train_scgpt.py`, `src/rolloutprobe/pair_eval.py`, `scripts/analyze_rollout.py`, and the small `scgpt` additions to `scripts/run_model.py`, `scripts/roundtrip_check.py` and `src/rolloutprobe/wrapper.py`. For steps 10–12: `src/rolloutprobe/latent.py`, `aux_batches.py`, `stability.py`, `discriminator.py`, `scripts/latent_additivity.py`, `scripts/stability.py`, `scripts/drift_discriminator.py` and figure 4 in `scripts/make_figures.py`.

Check these against the installed package (after step 2), mainly `scgpt/model/generation_model.py` and `scgpt/tokenizer/gene_tokenizer.py`:

- [ ] `TransformerGenerator.__init__` accepts the arguments `build_model()` passes: `nlayers_cls`, `n_cls`, `vocab`, `dropout`, `pad_token`, `pad_value`, `pert_pad_id`, `use_fast_transformer`.
- [ ] `TransformerGenerator.forward(src, values, input_pert_flags, src_key_padding_mask, CLS, CCE, MVC, ECS)` returns a dict with `"mlm_output"` of shape (batch, genes).
- [ ] The perturbation-flag embedding has 3 rows (0 = not perturbed, 1 = perturbed, 2 = padding), so `PERT_PAD_ID = 2` is valid.
- [ ] `GeneVocab.from_file`, `append_token`, `set_default_index`, `in` and `[]` exist as used in `load_vocab()`.
- [ ] The checkpoint's `args.json` has the keys `read_config()` reads (`embsize`, `nheads`, `d_hid`, `nlayers`, `n_layers_cls`).
- [ ] `load_weights()` renames flash-attention keys (`Wqkv.*` → `in_proj_*`) because we run without flash-attn. Compare with `scgpt/utils/util.py` `load_pretrained`. Training must print a nonzero count for each of `encoder`, `value_encoder` and `transformer`, otherwise the model is not starting from pretrained weights.
- [ ] `ScGPTPredictor.embed_means` (Part B) hooks `model.transformer_encoder` and reads its first input as the summed embedding (gene + value + flag) and its output as the last layer's states, each (batch, genes, d). Check this against `TransformerGenerator._encode`. If `_encode` normalises or transforms the sum before the encoder, the "input" view is that transformed tensor; say so in the report.
- [ ] Part C in `train_scgpt.py` (only runs with `--w-state` or `--w-saturation` above 0): auxiliary rows come only from training singles (never validation singles or doubles), and the main loss is unchanged when both weights are 0.
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
- **A1–A3 for GEARS** (the `analyze_rollout.py` printout):
  - **A2, code check.** Prediction: `sens_rel` below 1e-4 and `sens_corr` = 1.000, because GEARS's predicted shift does not read the input cell. Anything else means a bug in the new code, or that GEARS reads its input after all. Stop and report either way.
  - **A3.** Prediction: the carried rollout and measured-input carried rollout are exactly 0 by construction. The joint prediction *can* be positive, because the GO graph allows g(A,B) ≠ g(A) + g(B). If joint is positive and carried is 0, the finding is: GEARS has some interaction knowledge, but its rollout cannot carry it. Report joint's value next to the replicate correlation R and the noiseless ceiling √R, as in "joint X, replicate R, ceiling √R". The ceiling is √R, not R. For example, R = 0.6 gives a ceiling of about 0.77.
  - **Singles fit:** in-sample, so expect high values. If it is low, the implied interactions are contaminated by single-prediction error; say so.

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

- **Check:** `28 passed`. That is 21 CPU tests plus 4 scGPT and 3 CPA adapter tests, which only run where torch is installed (they use stand-in networks, so neither `scgpt` nor `cpa-tools` is needed). The CPU env without torch shows `21 passed, 2 skipped`.

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
- Copy `train_log.csv` into `results/scgpt_train_log_s1.csv` so it gets committed.

**Training seeds 2 and 3.** Train these after seed 1, using the same command with `--seed 2` and `--seed 3`. They write `models/scgpt_strict_s2` and `models/scgpt_strict_s3`, and their logs go to `results/scgpt_train_log_s2.csv` and `results/scgpt_train_log_s3.csv`. They may train while step 8's report is waiting for a decision, because both A2 outcomes need them.
- **The seed also changes the 10 validation singles.** `gears_custom_split` draws them with that seed, but always from singles that appear in no double. The training singles still include every gene in a test double, and the 131 test doubles stay the same. This is intended: the replicate varies the training run, not the test.
- Each run must pass the same checks as seed 1, including the `95 train singles` line.

## 7. Round-trip check on scGPT (diagnostic, not the gate)

```bash
python scripts/roundtrip_check.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict
```

**This no longer decides whether scGPT continues; always go on to step 8.** It passes real A-cells through with *no* perturbation flag. scGPT was trained so that no flags means "output a control cell": that is literally its control condition. So scGPT can fail this check because of its training objective alone, while still letting the input state change a *flagged* perturbation. The check mixes up "ignores its input" (GEARS) with "returns to control when unflagged" (scGPT), and those mean opposite things for composition. That is failure type 3, an unfair test. A2 in step 8 asks the right question.

**Time this run** (`time python ...`). It is the early estimate of step 8's cost (see GPU budget).

It is still the right diagnostic for drift on generated input. The verdict uses correlation on raw expression (> 0.95), which GEARS passed trivially (0.999) because mean expression levels dominate it. Report every column; the informative ones are `passthrough_real_delta_r`, `passthrough_generated_delta_r` and `saturation_ratio`. Do not change code or thresholds.

## 8. scGPT joint and rollout predictions

Time one split seed first; scGPT reads all 5,045 genes per cell, so it is much slower than GEARS:

```bash
python scripts/run_model.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict --seeds 0
```

- **Check:**
  - it prints `split seed 0 done (131 pairs)`
  - no doubles are excluded (every perturbed gene is an expression gene)
  - it writes `results/model_scgpt_strict.csv`
Then:

```bash
python scripts/analyze_rollout.py
python scripts/make_figures.py
```

**Stop here and report A2 and A3 before running anything more.** A2 may end the study.

- **A2 question:** does scGPT's predicted effect of B depend on the cell it starts from? Before running, the answer is unknown, and this is the result that matters most. Its zero is GEARS, at exactly 0. Read `sens_rel` against `sens_null_rel`; `sens_rel_minus_null` is the paired excess, with CI and the share of pairs above 0.
  - **No excess** (CI includes 0, or the excess is negligible next to `sens_null_rel`): no tested model's prediction depends on cell state. GEARS fails for architectural reasons; scGPT fails despite pretraining, which points at the training data (type 2). The study's conclusion is then clean. Do not try to rescue it; write it up.
  - **Clear excess:** composition is testable for scGPT. A3 then says whether that state dependence carries the *right* interaction. Note that `rollout_measured_carried_ab` in A3 is exactly "the effect of B on real A-cells minus its effect on control cells", correlated with the measured interaction, so A2 and A3 meet there.
- **A3 question:** does any scGPT prediction recover the measured interaction? Report each variant next to its zero, the replicate correlation and the noiseless ceiling.
- **Part C gate (read from the same A3 printout, no extra run):** does scGPT have interaction knowledge that rollout cannot carry? Part C's premise is exactly that, so this decides whether Part C has anything to work with.
  - **Joint A3 CI above 0 while `rollout_carried_ab` sits near 0:** the joint prediction knows something about the interaction that the rollout path loses. Part C is possible (step 12).
  - **Joint A3 itself near 0** (CI includes 0): the model has no interaction knowledge, and Part C cannot create any. Its losses teach state preservation and saturation, not how A modulates B. Part C is closed; say so in the report.
- Also report seed 0's run time. `analyze_rollout.py` compares against addition on the same seeds, so a 1-seed run is still a fair comparison.

**After the report: all split seeds and all training seeds.** Unless Shreyan says otherwise, run all 5 split seeds for each training seed:

```bash
python scripts/run_model.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict
python scripts/run_model.py --model scgpt --model-dir models/scgpt_strict_s2 --tag scgpt_strict_s2
python scripts/run_model.py --model scgpt --model-dir models/scgpt_strict_s3 --tag scgpt_strict_s3
python scripts/analyze_rollout.py
python scripts/make_figures.py
```

- The first command overwrites the seed-0-only `results/model_scgpt_strict.csv`.
- Each tag writes its own file and gets its own block in the `analyze_rollout.py` printout.
- For A2 and A3, report whether the three training seeds agree. Agreement means the CI of `sens_rel_minus_null` points the same way in all three, and the A3 values sit in the same range relative to the ceiling. If they disagree, that is the result: the property depends on the training run, not on scGPT. Report it as such, and do not pick one run. Training run and validation singles change together, so do not attribute a disagreement to initialisation alone.

## 8b. CPA (in the idle window while the step 8 report waits)

**Not on the critical path.** The order stays: GEARS rerun (1) → scGPT seed 1 training (6) → scGPT split seed 0 (8) → stop and report A2. CPA runs while that report waits for Shreyan's decision, alongside scGPT training seeds 2 and 3. If step 8 is not done yet, CPA waits. It never displaces a step 1–8 run.

**Why CPA is worth the time.** It is the third distinct mechanism:
- GEARS cannot read its input.
- scGPT architecturally can, but every training input was a control cell.
- CPA was designed to compose in latent space by construction: encode the cell to a basal state, add perturbation embeddings, decode.

If CPA also fails, the finding becomes "composition does not survive any of the three routes", not "these two models happen not to compose". CPA is also the one model where Part B is a sharp test, because its basal latent is a real bottleneck with adversarial pressure on it.

**Can CPA be tested at all? Yes.** Checked in the source before writing anything: github.com/theislab/cpa, commit `fbd7c02`, cpa-tools 0.8.8.
- `CPAModule.inference` encodes the given input cell x to `z_basal`, adds the perturbation embedding, and decodes. It does not sample its own cells.
- The public `CPA.predict` takes any AnnData's X as input. CPA's Norman tutorial makes its counterfactuals exactly this way: set `adata.X` to control cells and keep the target labels.

So A2 is meaningful for CPA. The adapter docstring (`src/rolloutprobe/models/cpa_adapter.py`) lists what was verified against source and what was not.

**Decisions recorded in advance** (all in `scripts/train_cpa.py` and written to `rolloutprobe_split.json`):
- **Counts vs normalised data.** CPA's tutorial trains on raw counts with a negative binomial loss. The GEARS-processed file everything else uses is log-normalized and has no counts layer. CPA is trained with `recon_loss="gauss"` on that log-normalized X (`is_count_data=False`), which CPA supports. This keeps the same cells, the same 5,045 genes and the same expression space as every other model. Fetching Norman counts from elsewhere would need a download over 10 MB and a different gene set, so it was not done.
- **Strict split.** Training sees control plus the 95 training singles. The 10 held-out singles (same `gears_custom_split` rule and seed: singles in no double) and every double are `ood`, so CPA's training loop never sees them.
- **CPA's own validation set** (early stopping only). By default this is a random 10% of the training conditions' cells. CPA cannot embed a gene it has never seen, so validating on held-out singles would only measure untrained embeddings. `--valid-mode singles` gives the GEARS/scGPT-style alternative.
- **Hyperparameters.** These are CPA's Norman tutorial values, unchanged except `recon_loss`, `seed` (= `--seed`) and dropping the single-valued `cell_type` covariate.
- **Adapter path.** The adapter calls `module.get_expression`, which is the per-batch call inside `CPA.predict`. It avoids `CPA.predict` itself because scvi-tools registers and keeps every new AnnData, and run_model makes about 6,500 calls per run. `train_cpa.py` checks the two paths agree on real cells and refuses to save a model if they differ.

### C1. CPA environment (timebox: one hour)

`cpa-tools` pins conflict with both `doubletake` and `scgpt` (python <3.11, torch ≤2.0.1, numpy <1.24, anndata <0.10, scvi-tools <1.0), so it gets a third env. **If `import cpa` does not work within one hour, stop, report the error and the versions, and skip CPA.** Dependency resolution must not eat the window.

```bash
conda create -n cpa python=3.10 -y && conda activate cpa
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install cpa-tools
pip install pyyaml pytest
python -c "import cpa, scvi, torch, anndata, numpy; print(cpa.__version__, scvi.__version__, torch.__version__, anndata.__version__, numpy.__version__, torch.cuda.is_available())"
```

- Do not install our `requirements.txt` here. It asks for numpy ≥1.24 and anndata ≥0.10, which cpa-tools forbids. Our code runs on what cpa-tools brings, and C2 checks this.
- **Known risks, both from the source:**
  - `cpa/_model.py` has a stray `from tkinter import N` at the top. If the import fails on tkinter, run `conda install tk`.
  - CPA's training plan uses Lightning's `validation_epoch_end` hook, which Lightning 2 removed. An error there is a version mismatch; report the versions. It counts against the timebox.

### C2. Review the CPA code (before training)

Read `src/rolloutprobe/models/cpa_adapter.py` and `scripts/train_cpa.py` against the installed `cpa` package (`python -c "import cpa, os; print(os.path.dirname(cpa.__file__))"`):
- [ ] `CPA.setup_anndata(adata, perturbation_key, control_group, dosage_key=None, is_count_data, categorical_covariate_keys, max_comb_len)` has this signature, and builds `obsm["perts"]` and `obsm["perts_doses"]` as the adapter's `_perts()` does: indices in vocabulary order, padding 0, dose 1.0 per perturbation, control as its own entry `"ctrl"`.
- [ ] `CPA(adata, split_key, train_split, valid_split, test_split, seed, **model_params)` and `train(max_epochs, use_gpu, batch_size, plan_kwargs, early_stopping_patience, check_val_every_n_epoch, save_path)` have these signatures.
- [ ] `CPAModule.get_expression(tensors, n_samples)` returns a dict with `"px"` (the decoded mean for `gauss`) and `"z_basal"`, and reads only the four tensor keys the adapter builds (`X`, the perturbation label column, `perts`, `perts_doses`) when there are no covariates.
- [ ] `CPA.load(dir_path, use_gpu)` works without passing an AnnData when the model was saved with `save_anndata=True`, and sets `CPA.pert_encoder` and `CPA_REGISTRY_KEYS.MAX_COMB_LENGTH`.
- [ ] The installed version is 0.8.x. If it is not, compare `_module.py` and `_model.py` with the commit above before going on.

### C3. Tests in the CPA env

```bash
pytest -q
```

**Check:** `28 passed`. If tests fail only because of package versions (numpy, anndata), report which. CPA can still run if `pytest -q tests/test_cpa_adapter.py` passes.

### C4. Train CPA (about an hour)

```bash
python scripts/train_cpa.py --seed 1
```

- **Checks:**
  - The split line: `95 train singles + ctrl, 10 held-out singles [...], 131 test doubles`, with no double in train or valid (the script asserts this).
  - It ends with `adapter vs CPA.predict on ...: max |diff| = ...` below 1e-4, then `saved models/cpa_strict_s1`. If the diff check fails, the model is not saved; report the number.
- Copy `models/cpa_strict_s1/history.csv` to `results/cpa_train_history_s1.csv`.

### C5. Predictions, then CPA's A2, A3 and Part B

```bash
python scripts/roundtrip_check.py --model cpa --model-dir models/cpa_strict_s1 --tag cpa_strict
python scripts/run_model.py --model cpa --model-dir models/cpa_strict_s1 --tag cpa_strict
python scripts/latent_additivity.py --model cpa --model-dir models/cpa_strict_s1 --tag cpa_strict
python scripts/analyze_rollout.py
python scripts/make_figures.py
```

- All 5 split seeds, since CPA is a small MLP. `run_model.py` should score all 131 doubles: every test double's genes are training singles, which have trained embeddings.
- **Round-trip:** CPA passes real cells through with the `"ctrl"` label, which is how it reconstructs control cells. Like scGPT's, this is a diagnostic, not a gate.

**Predictions, written before running.** Compare against these; do not rationalise afterwards.

- **A2, input sensitivity.** Expect `sens_rel` to be small but *not* exactly zero, unlike GEARS. The basal encoder is trained adversarially so that a classifier cannot recover the perturbation from it. Encoding a real A cell should therefore give a basal state close to a control cell's. The stripping is imperfect, so some perturbation information leaks through. Prediction: small, and plausibly not clearly above `sens_null_rel`.
- **Rollout, a consequence of the same design, also predicted.** If stripping works, re-encoding the step-1 output removes A. The carried rollout A→B then predicts roughly CPA's single-B prediction: the first step is forgotten.
  - Expect `rollout_carried` to do worse than addition, with a large order gap (A→B ≈ B and B→A ≈ A).
  - Expect `rollout_measured_carried_ab` to look like the single B prediction.
  - In that case the implied interaction is about −D_pred(A), not 0. Its A3 can then be nonzero, with either sign, without any composition. Read rollout A3 only together with the order gap and `int_rel`.
- **Joint.** `predict(x, (A, B))` decodes z_basal + e_A + e_B, which is additive in the latent. The decoder is nonlinear, so the gene-space prediction can be non-additive even though no double was ever trained on. A positive joint A3 would then be decoder nonlinearity that happens to align with the measured interaction. It would not be learned interaction knowledge.
- **A2 and A3 together.** Both metrics are needed to tell leakage from composition. Worth a paragraph in the final report:

| A2 excess over null | A3 (joint or rollout) | Reading |
|---|---|---|
| Not above null | Near zero | The basal encoder strips the input state, as designed. Composition fails for a third, nameable reason |
| Above null | Near zero | Leakage through imperfect adversarial stripping, not composition |
| Above null | Clearly above zero, with a small order gap | CPA composes. The most surprising outcome; it would change the study's conclusion |

- **Failure types for CPA differ from the other two.** CPA *does* train on perturbed inputs: each cell is reconstructed from itself with its own label. What it never sees is a perturbed cell given a *different* label. A failure is then architectural by design (the adversary removes the state), not "only control inputs in training" (type 2).
- **Part B on CPA (the `basal` view).** Same split-half test as the other views: `cross_half_signal` = mean over pairs of (r₁·r₂)/(d₁·d₂), with its bootstrap interval and the share of pairs above 0. Prediction: the basal latent carries little or no interaction signal (interval includes 0, or the value is far below the `genes` view). The encoder is trained to remove exactly the information that distinguishes perturbed states. This is a predicted failure with a nameable cause, which is why Part B means something for CPA.

**Report for 8b:**
- C1: the printed versions, and how long the env took
- C2: what you checked, and any fix (in its own commit)
- C3: the test result
- C4: the split line, the epochs run, and the `public_predict_check` diff
- The round-trip table
- For A2, A3 and Part B: the question, the prediction above, the number with its zero, null and ceiling, and the row of the table it falls in
- Then the branch and commit

## 9. Record and push

1. In the README:
   - Tick finished checklist items, with initials and date
   - Add a Log line with the key numbers: carried vs restated advantage for GEARS, the scGPT A2 result (`sens_rel_minus_null` with CI, per training seed), the scGPT round-trip diagnostic verdict and delta r values, and scGPT joint and carried advantage with share of pairs better
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
- Steps 1, 5–8: the printed output (split line, vocab coverage, pretrained tensor counts, the per-epoch log, the round-trip table, per-seed timing, the full `analyze_rollout.py` printout)
- For A2 and A3 (GEARS and scGPT): the question, the prediction written above, the number with its zero and its ceiling, and which failure type (architecture, data, unfair test) it supports
- Whether the three scGPT training seeds agree on A2 and A3, and the best validation Pearson delta of each
- Git: branch, commit hash, push result
- Anything skipped or failed, and why

## 10. Stability and drift (after the step 8 report; minutes per model)

These are consistency checks: Norman cannot say what a second dose of guide A should do. Use the `scgpt` env for scGPT and `doubletake` for GEARS.

```bash
python scripts/stability.py          --model scgpt --model-dir models/scgpt_strict_s1   --tag scgpt_strict
python scripts/drift_discriminator.py --model scgpt --model-dir models/scgpt_strict_s1   --tag scgpt_strict
python scripts/stability.py          --model gears --model-dir models/gears_strict_s1   --tag gears_strict
python scripts/drift_discriminator.py --model gears --model-dir models/gears_strict_s1   --tag gears_strict
```

- **Stability question:** what does a repeated perturbation do? `step_ratio` near 0 after step 1 means it saturates, which is the biologically plausible behaviour. A constant 1 means it re-adds a fixed vector.
  - **Prediction for GEARS:** exactly 1 at every step, by construction. Anything else is a bug.
  - **scGPT:** unknown. Also report `dist_target`, the distance from the measured A profile in units of ‖A − ctrl‖ (step 1 should be well below 1), and whether `nearest` stays on A or drifts to other conditions.
- **Drift question:** do predicted cells look like real cells? Accuracy 0.5 means indistinguishable. Always report it next to `floor` (real vs real, half 1 vs half 2) and `scale` (control vs target).
  - Accuracy near `scale` means the predictions are as distinguishable from the target as control cells are.
  - **Expectation for GEARS:** its predicted cells are control cells plus one fixed vector each, so they keep the control cells' spread. Its accuracy should mostly reflect the error in the mean.
  - **scGPT** outputs may be smoother than real cells, which a classifier detects even when the mean is right. If so, say that this measures realism of single cells, not the composition question.

## 11. Part B diagnostic: is the interaction present inside scGPT? (after the step 8 report; under 1 GPU hour)

**A diagnostic for the write-up, not a gate.** The Part C gate is joint A3 (step 8). For scGPT, Part B's absolute numbers are expected to say little. scGPT's per-gene residual stream is close to lossless by design, so encoding real AB cells reflects the interaction simply because the AB cells' input expression contains it. That does not show that any layer uses it. Part B is sharp only for an architecture with a real bottleneck, such as CPA (step 8b, C5).

```bash
python scripts/latent_additivity.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict
```

- **What it does:** embeds real cells of ctrl, A, B and AB (half 1 and half 2, up to 300 per condition and half) with no perturbation flags, for all 131 test pairs. Definitions are in `src/rolloutprobe/latent.py`.
- **Views:**
  - `genes`
  - `input_tokens` and `last_tokens`: per-gene states entering and leaving the transformer, flattened
  - `input_pooled` and `last_pooled`: averaged over genes
- **Disk use:** embeddings are cached in `models/latent_cache/` (ignored by git), about 20 MB per condition and half, roughly 10 GB in all. A rerun resumes from the cache. Delete the cache after the results are committed.
- **Output:** `results/latent_scgpt_strict.csv` (summaries) and `results/latent_scgpt_strict_pairs.csv`.
- **GEARS is not run:** its input cell never enters a latent (out = x + g(pert)), so there is nothing to probe. Say so in the report.

**What to report. There are no invented thresholds; each test has a built-in null.**
- **`cross_half_signal` per view:** the mean over pairs of (r_1 · r_2) / (d_1 · d_2), with a bootstrap CI and the share of pairs above 0. The halves are independent, so r_1 · r_2 has expectation 0 when a view carries no interaction signal. "CI excludes 0" is the same split-half test that eta uses in gene space.
- **`ratio_to_input` for `last_tokens` and `last_pooled`:** the last layer against what entered the transformer. **This change across layers is the informative quantity.**
  - Near 1: the layers pass the interaction through unchanged.
  - Below 1: the layers shrink it.
  - It still does not show that the decoder *uses* the interaction; A3 answers that.
- `ratio_to_genes` and `pooled_rel_cross` are in the CSV for completeness. Do not interpret the absolute ratio to gene space, for the pass-through reason above.

## 12. Part C: can scGPT be made to carry state? (default: future work)

**Not planned by default.** The default is to describe Part C as future work in the final report. Run it only if both of these hold:
- the step 8 gate showed joint A3 above 0 with rollout near 0
- Shreyan approves it after all step 8 runs are done, given the time left before the Dec 3 report

The scGPT version needs no new code: the losses are already in `train_scgpt.py`, off by default. The GEARS version needs a fork and is not written (see the end of this file).

An architecture change alone cannot work. Norman gives no gradient for state-conditioning, because every training input is a control cell, so a new pathway would be ignored. scGPT already reads its input, so it needs no architecture change, only a training signal from singles (`src/rolloutprobe/aux_batches.py`):
- **State:** input a real A cell with no flag; the target is another A cell. This contradicts scGPT's current "no flags → control" behaviour, which is the point.
- **Saturation:** input a real A cell with A flagged; the target is another A cell, since A cells already have guide A active.

Both use training singles only, so the strict split holds.

```bash
python scripts/train_scgpt.py --pretrained models/scgpt_human --seed 1 --name scgpt_carry --w-state 1 --w-saturation 1
python scripts/run_model.py --model scgpt --model-dir models/scgpt_carry_s1 --tag scgpt_carry
python scripts/stability.py --model scgpt --model-dir models/scgpt_carry_s1 --tag scgpt_carry
python scripts/analyze_rollout.py
python scripts/make_figures.py
```

- **Weights:** 1 and 1 is the starting point and is not tuned. Do not tune them on the test doubles. If the run is unstable, report it; do not search over weights.
- **Check:** the training log prints the main loss and each auxiliary loss. The main validation Pearson delta should stay close to `scgpt_strict_s1`'s. If it drops a lot, the auxiliary losses are fighting the main task; report it.
- **Question:** does carrying state let rollout express the interaction knowledge the joint prediction has? Compare `scgpt_carry` with `scgpt_strict` on:
  - A2 (`sens_rel_minus_null` should rise)
  - A3 for `rollout_carried_ab` and `rollout_measured_carried_ab`, against joint A3, the replicate and the ceiling
  - carried-rollout advantage over addition
  - stability `step_ratio`, which should fall toward 0 if saturation was learned
- **Expected best case:** rollout A3 moves toward joint A3. No singles-only signal can teach how A modulates B, so rollout A3 may stay near 0 even if A2 rises. That outcome isolates the limitation to the training data (failure type 2).

## Report back for steps 10–12

The same format as above: for each step, the question, the prediction written here, the numbers with their zero, floor or ceiling, and which failure type they support. For Part B, give `cross_half_signal` per view with its CI, and `ratio_to_input`. For Part C, either the results or "not run", with the step 8 gate result that decided it.

## Not written (and why)

- **GEARS with state input (Part C for GEARS):** out = x + g(pert, x). It needs a fork of cell-gears' `GEARS_Model.forward`, so that the perturbation module also reads an embedding of the input. It also needs a training loop that feeds real single-perturbed cells as inputs, which GEARS's dataloader cannot do (it pairs control inputs with perturbed targets itself). This was not written because neither could be checked against the cell-gears source here. It is future work for the final report, not a planned result.
- **Discrete or quantised latent "state islands":** deliberately not built. Learning transitions between states needs (state_i, action) → state_j across many starting states. Norman only has (control, action) → state_j, which is one row of that table. A discrete latent changes the representation without adding the missing observations.
- **Final report** (due Dec 3): copy `report/midterm/` to `report/final/`. Add the rollout results (from `results/rollout_analysis.csv`, figure 4 `figures/fig4_advantage_vs_eta`) to Experiments and Analysis, plus a Conclusion and Team Contributions, which the final rubric requires.
