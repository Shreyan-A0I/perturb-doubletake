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
| Stability and drift (step 10), scGPT and GEARS | under 15 min | 2 models | under 0.5 h |
| Part B, latent additivity (step 11): about 237 conditions × 2 halves × up to 300 cells | under 1 h | 1 | under 1 h |
| Part C, only if Part B is favourable (step 12): one fine-tune plus 5 split seeds | 3–7 h | 1 | 3–7 h |

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
> **Review the code before running anything (step 0).** `scripts/train_scgpt.py`, `src/rolloutprobe/models/scgpt_adapter.py`, `src/rolloutprobe/pair_eval.py`, `scripts/analyze_rollout.py` and the step 10–12 code (listed in step 0) are new. The scGPT code was written against scGPT's API on a machine without the `scgpt` package or a GPU, so it has **never run with the real scGPT model**. Only its plumbing is tested, with a stand-in network. If you are an agent running this file: read every file listed in step 0 against the installed `scgpt` source before the first GPU run. Assume something was missed. If you find a bug, fix it in its own commit on your branch and say what you changed in the report. Never change the experimental design (split, validation singles, seeds, metrics, gate threshold) to make a step pass.

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

- **Check:** `25 passed`. That is 21 CPU tests plus 4 scGPT adapter tests, which only run where torch is installed. The CPU env without torch shows `21 passed, 1 skipped`.

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
- Also report seed 0's run time. `analyze_rollout.py` compares against addition on the same seeds, so a 1-seed run is still a fair comparison. Shreyan decides whether to start Part B.

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

## 11. Part B: where is the interaction lost? (after A2 is reported; under 1 GPU hour)

Question: is the measured interaction still present inside scGPT? Part B decides whether Part C is worth its GPU time.

```bash
python scripts/latent_additivity.py --model scgpt --model-dir models/scgpt_strict_s1 --tag scgpt_strict
```

- **What it does:** embeds real cells of ctrl, A, B and AB (half 1 and half 2, up to 300 per condition and half) with no perturbation flags, for all 131 test pairs.
- **The quantity:** the noise-corrected share of the double's shift that is non-additive (`rel_cross`, defined in `src/rolloutprobe/latent.py`). It is computed per view and compared with gene space on the same cells.
- **Views:**
  - `genes`
  - `input_tokens` and `last_tokens`: per-gene states entering and leaving the transformer, flattened
  - `input_pooled` and `last_pooled`: averaged over genes
- **Disk use:** embeddings are cached in `models/latent_cache/` (ignored by git), about 20 MB per condition and half, roughly 10 GB in all. A rerun resumes from the cache. Delete the cache after the results are committed.
- **Output:** `results/latent_scgpt_strict.csv` (summaries) and `results/latent_scgpt_strict_pairs.csv`.
- **GEARS is not run:** its input cell never enters a latent (out = x + g(pert)), so there is nothing to probe. Say so in the report.

**Decision.** Read `ratio_to_genes` for `last_tokens`, with its bootstrap CI over pairs. The thresholds are fixed before the run (Shreyan: confirm or change them before Datta runs this step):

| Result | Meaning | Next |
|---|---|---|
| CI upper bound below 0.5 | The encoder discards most of the interaction, so no latent-space method can recover it | Skip Part C |
| CI lower bound above 1.5 | Something is wrong | Investigate first |
| Otherwise (comparable) | The interaction survives into the latent, so the perturbation path fails to use it | Part C is worth attempting |

**Read alongside: `ratio_to_input` for `last_tokens`.** Per-gene token states carry each gene's own input value through the transformer's residual connections. So `last_tokens` can look "comparable" to gene space simply because the input is copied through, even if no layer uses the interaction. `ratio_to_input` compares the last layer with what entered the transformer.
- Near 1: the transformer passes the interaction through.
- Well below 1: the transformer shrinks it.

The pooled views have no per-gene copy, so they are the less trivial test. Report all four views.

## 12. Part C: can scGPT be made to carry state? (only if step 11 says "comparable")

An architecture change alone cannot work. Norman gives no gradient for state-conditioning, because every training input is a control cell, so a new pathway would be ignored. scGPT already reads its input, so Part C needs no architecture change, only a training signal from singles. `train_scgpt.py` has two auxiliary losses, off by default (`src/rolloutprobe/aux_batches.py`):
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
- **Question:** does carrying state make rollout compose? Compare `scgpt_carry` with `scgpt_strict` on:
  - A2 (`sens_rel_minus_null` should rise)
  - A3 for `rollout_carried_ab` and `rollout_measured_carried_ab`, against the replicate and the ceiling
  - carried-rollout advantage over addition
  - stability `step_ratio`, which should fall toward 0 if saturation was learned
- **Expected best case:** rollout stops being a fixed sum and moves toward the joint prediction. No singles-only signal can teach how A modulates B, so A3 may stay near 0 even if A2 rises. That outcome isolates the limitation to the training data (failure type 2), and is a strong result.

## Report back for steps 10–12

The same format as above: for each step, the question, the prediction written here, the numbers with their zero, floor or ceiling, and which failure type they support. Include the Part B decision row and which threshold decided it.

## Not written (and why)

- **GEARS with state input (Part C for GEARS):** out = x + g(pert, x). It needs a fork of cell-gears' `GEARS_Model.forward`, so that the perturbation module also reads an embedding of the input. It also needs a training loop that feeds real single-perturbed cells as inputs, which GEARS's dataloader cannot do (it pairs control inputs with perturbed targets itself). This was not written because neither could be checked against the cell-gears source here. Only attempt it if scGPT Part C shows that state-carrying helps; otherwise there is no reason to expect it to.
- **CPA** (after Part C, if at all): `cpa-tools`, strict split, behind `predict()`. It is interesting as a contrast because it composes in latent space by construction. Part B prediction for CPA: the latent residual is much smaller than in gene space, because its basal encoder is trained adversarially so that perturbation identity cannot be recovered from it. Not written: it needs a new environment (scvi-tools), and its API could not be checked here.
- **Discrete or quantised latent "state islands":** deliberately not built. Learning transitions between states needs (state_i, action) → state_j across many starting states. Norman only has (control, action) → state_j, which is one row of that table. A discrete latent changes the representation without adding the missing observations.
- **Final report** (due Dec 3): copy `report/midterm/` to `report/final/`. Add the rollout results (from `results/rollout_analysis.csv`, figure 4 `figures/fig4_advantage_vs_eta`) to Experiments and Analysis, plus a Conclusion and Team Contributions, which the final rubric requires.
