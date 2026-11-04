# Runbook

Technical reference: setup, commands, design rules, code layout. Run every command
from the repo root. Project context, tasks and status live in the top-level
[README](../README.md).

## Setup

```bash
pip install -r requirements.txt
pytest -q                     # 6 tests on synthetic data, about 10 seconds
```

No package install is needed; scripts add `src/` to the path themselves.

On the GPU machine, also install torch, torch_geometric and `cell-gears` following
the [GEARS README](https://github.com/snap-stanford/GEARS). Their builds depend on
your CUDA version, so they are not pinned here.

## Midterm runbook

**Step 0: get the data.** Use the file GEARS downloads, so the baselines and GEARS
share one gene set and gene order:

```bash
python -c "from gears import PertData; PertData('data/gears').load(data_name='norman')"
# -> data/gears/norman/perturb_processed.h5ad, the default path in configs/default.yaml
```

**Step 1: baselines and eta reliability (Sri Datta, CPU, minutes).**

```bash
python scripts/run_baselines.py
python scripts/make_figures.py
```

The first line it prints is the dataset count. Expect roughly 105 singles and 131
doubles. If those numbers are far off, the condition labels were parsed wrong: set
`data.condition_col` and `data.sep` in the config.

**Step 2: GEARS in both regimes (Shreyan, GPU, about an hour each).**

```bash
python scripts/train_gears.py --split standard --seed 1
python scripts/train_gears.py --split strict   --seed 1
python scripts/run_model.py --model gears --model-dir models/gears_standard_s1 --tag gears_standard --joint-only
python scripts/run_model.py --model gears --model-dir models/gears_strict_s1   --tag gears_strict   --joint-only
python scripts/make_figures.py      # picks up results/model_*.csv automatically
```

**Step 3, stretch: one full round-trip for GEARS.**

```bash
python scripts/roundtrip_check.py --model gears --model-dir models/gears_strict_s1 --tag gears_strict
```

Drop `--joint-only` from step 2 to run the full rollout experiment after the midterm.

## Design rules the code enforces

| Rule | Where | Why |
| --- | --- | --- |
| Cell half 1 is measurement only (eta, DEG choice); half 2 is everything else (baseline inputs, model inputs, scoring targets) | `splits.py` | The noise that places a pair in a stratum never appears in the target it is scored on (reviewer point 3) |
| All intervals bootstrap over perturbation pairs, after averaging over split seeds | `stats.py` | The pair is the unit of inference |
| Model predictions are anchored to the pseudobulk of the input condition | `run_model.py` | Removes sampling noise from the ~300 input cells that the nulls do not carry. Without it, every model is penalized relative to the baselines |
| Rollout runs in two arms: `carried` (label B only) and `restated` (label A+B) | `rollout.py` | Norman keeps guide A active throughout; neither arm matches that exactly (reviewer point 1) |
| Order dependence and saturation are reported as computational consistency checks | `rollout.py` | Norman measures joint perturbations and cannot validate either |
| No model joins the roster without passing `roundtrip_check.py` | `rollout.py` | Reviewer point 2 |

## Layout

```
configs/default.yaml        all settings
src/rolloutprobe/
  data.py                   load h5ad (GEARS or scPerturb format), canonical condition keys, pseudobulk
  splits.py                 strict split, half split, GEARS custom split export
  eta.py                    non-additivity, cross-half signal estimate, reliability
  nulls.py                  nochange, mean, additive, lowrank_additive
  metrics.py                MSE on top-20 DEGs, Pearson delta
  stats.py                  bootstrap over pairs, advantage vs eta slope
  schema.py                 results contract + fake_results() for building figures early
  wrapper.py                predict(X, pert) -> X interface; oracle test model
  rollout.py                joint, rollout arms, measured-input arm, saturation, round-trip
  models/gears_adapter.py   GEARS behind predict()
scripts/
  make_synthetic.py         synthetic Norman-shaped data with planted interactions
  run_baselines.py          midterm items 1-2
  train_gears.py            GEARS, standard or strict split
  run_model.py              joint + rollout for any registered model
  roundtrip_check.py        gate 1
  make_figures.py           fig 1 eta reliability, fig 2 hierarchy, fig 3 additive error vs eta
tests/test_pipeline.py
```

## What is verified and what is not

**Verified on synthetic data with known answers** (`pytest`): label parsing, half
splits, eta (separates every planted interacting pair; the cross-half estimate sits
near zero for truly additive pairs), the null hierarchy, the results schema, and the
rollout plumbing. Under an exactly additive model, joint prediction, carried rollout and
measured-input rollout all reproduce the additive null to 1e-6. That is the check
that caught the input-sampling bias the anchoring rule fixes.

**Checked against source, not executed:** `models/gears_adapter.py` and
`scripts/train_gears.py`. Every GEARS call was checked against GEARS master
(attribute names, the custom-split pickle, output shape, control handling), but
none ran here (no GPU, no data access). Run `roundtrip_check.py` before trusting
any GEARS number.

**Why GEARS needs a custom adapter at all:** `GEARS.predict()` accepts only a
perturbation list and samples its own control cells internally. It has no way to
continue from a state you hand it. The adapter builds the graph batch directly from
the expression you pass in, mirroring GEARS's own `create_cell_graph()`.

## Results contract

Every prediction is one row, defined in `schema.py`:

`model, method, arm, order, pert_a, pert_b, support, split_seed, eta_h1, mse_20deg, pearson_delta, pearson_delta_20deg`

Nulls write `results/baselines.csv`; each model writes `results/model_<tag>.csv`.
Figures read whatever is there. To build or restyle figures before any model runs:
`python scripts/make_figures.py --fake --out figures/fake`.
