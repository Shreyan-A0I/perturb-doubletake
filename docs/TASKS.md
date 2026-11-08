# Tasks, step by step: rollout and scGPT

Post-midterm work, from the Nov 15 checkpoint to the final report. Steps 1–9 run on the GPU machine (the oc3 H100 environment used for GEARS). Run every command from the repo root, in order, and stop at any check that fails. Background and design rules: [README](../README.md) and [RUNBOOK](RUNBOOK.md). The midterm tasks (setup, baselines, figures, GEARS training, joint predictions, GEARS round-trip check) are done and recorded in the README log.

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

Two additions use the spare hours, and neither changes the design:

1. **Three scGPT fine-tuning runs instead of one** (training seeds 1, 2, 3). A2 and A3 come from a single trained network. Without replicate runs, there is no way to tell a property of scGPT on Norman apart from the outcome of one training run. A second and third run are needed whatever A2 shows: a null result needs replication as much as a positive one.
2. **All 5 split seeds for every scGPT run**, by default rather than only when time allows. This puts scGPT on the same footing as the baselines and GEARS.

If timings come out far above these estimates, keep training seed 1 on all 5 split seeds, and run seeds 2 and 3 on split seed 0 only. Report that choice.

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
> **Review the code before running anything (step 0).** `scripts/train_scgpt.py`, `src/rolloutprobe/models/scgpt_adapter.py`, `src/rolloutprobe/pair_eval.py` and `scripts/analyze_rollout.py` are new. The scGPT code was written against scGPT's API on a machine without torch or a GPU, so it has **never been executed**. Only its plumbing is tested, with a stand-in model. If you are an agent running this file: read every file listed in step 0 against the installed `scgpt` source before the first GPU run. Assume something was missed. If you find a bug, fix it in its own commit on your branch and say what you changed in the report. Never change the experimental design (split, validation singles, seeds, metrics, gate threshold) to make a step pass.

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

Files: `src/rolloutprobe/models/scgpt_adapter.py`, `scripts/train_scgpt.py`, `src/rolloutprobe/pair_eval.py`, `scripts/analyze_rollout.py`, and the small `scgpt` additions to `scripts/run_model.py`, `scripts/roundtrip_check.py` and `src/rolloutprobe/wrapper.py`.

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
- **A1–A3 for GEARS** (the `analyze_rollout.py` printout):
  - **A2, code check.** Prediction: `sens_rel` below 1e-4 and `sens_corr` = 1.000, because GEARS's predicted shift does not read the input cell. Anything else means a bug in the new code, or that GEARS reads its input after all. Stop and report either way.
  - **A3.** Prediction: the carried rollout and measured-input carried rollout are exactly 0 by construction. The joint prediction *can* be positive, because the GO graph allows g(A,B) ≠ g(A) + g(B). If joint is positive and carried is 0, the finding is: GEARS has some interaction knowledge, but its rollout cannot carry it. Report joint's value next to the replicate correlation and the noiseless ceiling, for example "0.15 of an achievable ~0.6".
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

- **Check:** `15 passed`. That is 12 CPU tests plus 3 scGPT adapter tests, which only run where torch is installed. The CPU env without torch shows `12 passed, 1 skipped`.

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
- For A2 and A3, report whether the three training seeds agree. Agreement means the CI of `sens_rel_minus_null` points the same way in all three, and the A3 values sit in the same range relative to the ceiling. If they disagree, that is the result: the property depends on the training run, not on scGPT. Report it as such, and do not pick one run.

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
- Steps 1, 5–8: the printed output (split line, vocab coverage, pretrained tensor counts, the per-epoch log, the round-trip table, per-seed timing, the full `analyze_rollout.py` printout)
- For A2 and A3 (GEARS and scGPT): the question, the prediction written above, the number with its zero and its ceiling, and which failure type (architecture, data, unfair test) it supports
- Whether the three scGPT training seeds agree on A2 and A3, and the best validation Pearson delta of each
- Git: branch, commit hash, push result
- Anything skipped or failed, and why

## Not written yet (development, after Nov 15)

These have no code yet. Each needs its own branch and a test on synthetic data before it runs on real data. Order: Part B only after A2 is reported, then Part C only if Part B is favourable, then CPA if at all.

- **Part B: where is the interaction lost?** It is cheap and needs no training, and it decides whether Part C is worth attempting. Question: is the interaction represented anywhere inside the model?
  - **Method:** embed real cells from ctrl, A, B and AB, with no perturbation flags so the embedding is the input state. Test whether the latent is additive: z_AB − z_ctrl against (z_A − z_ctrl) + (z_B − z_ctrl). Compare the relative residual ‖r‖ / ‖Δ_AB‖ in latent space with the same quantity in gene space, so the comparison is scale-free. Use embeddings from half 2, and half 1 vs half 2 for the noise ceiling.
  - **Decided in advance:**
    - Latent residual much smaller than in gene space: the encoder discards the interaction, so no latent-space method can recover it, and Part C is not worth it.
    - Comparable: the interaction survives into the latent, so the perturbation head or decoder fails to use it, and Part C is worth attempting.
    - Larger: something is wrong; investigate first.
  - **scGPT caveat:** the fine-tuned perturbation model (`TransformerGenerator`) has no CLS token in its inputs (gene tokens only), so "the CLS embedding" does not exist for it. Its decoder reads the per-gene token states of the last transformer layer. Use those, flattened, as the latent, and mean-pooled as a secondary view. The pretrained whole-human model does have a CLS token, but it is a different network from the one making the predictions.
  - **CPA:** expect "much smaller", because its basal encoder is trained adversarially so that perturbation identity cannot be recovered from it. That is a predicted failure with a nameable cause.
- **Part C: can a model be made to carry state?** Only if Part B says yes, and only after A2, A3 and B are reported. An architecture change alone cannot work: Norman gives no gradient for state-conditioning, since every training input is a control cell, so a new pathway would be ignored. It needs an architecture change *plus* a training signal from singles only.
  - **Architecture:** for GEARS, out = x + g(pert, x), meaning the perturbation module also reads an embedding of the input. For scGPT nothing changes; it already reads its input.
  - **Two auxiliary losses on real cells:**
    - State preservation: predict(x_A, ∅) ≈ x_A. This contradicts scGPT's current "no flags → control" behaviour, which is the point.
    - Saturation: predict(x_A, (A,)) ≈ x_A, since A-cells already have guide A active.
  - **Expected best case:** rollout stops being a fixed sum and approaches the joint prediction. No singles-only signal can teach how A modulates B. A failure is still a strong result, because it isolates the limitation to the training distribution.
- **Not building: discrete or quantised latent "state islands".** Learning transitions between states needs (state_i, action) → state_j across many starting states. Norman only has (control, action) → state_j, which is one row of that table. A discrete latent changes the representation without adding the missing observations.
- **CPA** (after Part C, if at all): `cpa-tools`, strict split, behind `predict()`. Interesting as a contrast because it composes in latent space by construction.

- **Stability experiment** (cut list, third): repeat the same perturbation for 1 to 10 steps from control cells, `x_{k+1} = predict(x_k, (A,))`, for the 5 largest singles. Report the saturation ratio per step, and the distance of the step-k pseudobulk from the nearest real condition's pseudobulk. Builds on `rollout.saturation_ratio`; the new script would be `scripts/stability.py`.
- **Discriminator drift metric** (cut list, first to go): train a classifier to tell real cells from predicted cells after k steps; accuracy near 0.5 means no drift. Skip it unless everything above is done.
- **Final report** (due Dec 3): copy `report/midterm/` to `report/final/`. Add the rollout results (from `results/rollout_analysis.csv`) to Experiments and Analysis, plus a Conclusion and Team Contributions, which the final rubric requires. The figure of advantage against eta per variant is still to be added to `scripts/make_figures.py`.
