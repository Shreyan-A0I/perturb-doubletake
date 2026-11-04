# perturb-doubletake

**Do perturbation models compose under their own output?**

GenAI in Biomedicine course project, CMU, Fall 2026
Shreyan Nalwad (`snalwad`) · Sri Datta Bandreddi (`sbandred`)

This README is the shared source of truth for the project: context, decisions, data, and the task list. Both of us update it. Technical setup and commands live in [docs/RUNBOOK.md](docs/RUNBOOK.md).

---

## Status

*Update this block whenever something changes.*

- **Last updated:** 2026-11-03 by Shreyan
- **Next deadline:** Midterm report, Nov 6 (confirm exact time)
- **Phase:** Midterm sprint. Nothing has run on real data yet
- **Blockers:** None

---

## The project in brief

- Perturbation models (GEARS, scGPT, State) predict the expression profile after a genetic perturbation: control cell in, perturbed profile out
- Their input and output live in the same space, so a prediction can be fed back in and perturbed again: `predict(predict(x, A), B)`
- No published work checks whether doing that gives anything coherent
- **Our question:** do these models carry a reusable cell state, or are they one-step predictors?
- **Test:** train on single perturbations only, hold out every double, and compare
  - the model's joint prediction for A+B
  - its sequential rollout, A then B
  - simple baselines, mainly addition: control + effect of A + effect of B
- **Motivation, not the goal:** a virtual cell world model (Xing & Song, *Cell* 2026) would need exactly this property. We test whether a precondition holds; we are not building a virtual cell
- **Known starting point:** Ahlmann-Eltze & Huber (*Nat Methods* 2025) showed deep models lose to the additive baseline on Norman doubles. They only tested the joint prediction. Nobody has fed outputs back in

---

## Key decisions

*Add new decisions at the bottom with a date.*

- **Dataset:** Norman et al. 2019, the GEARS-processed version. It is the only public set with double perturbations plus all their singles
- **Strict split:** all singles train, all doubles test. Not GEARS's default split, which puts some doubles in training
- **Split-half cells (reviewer point 3):** each condition's cells are split randomly into two halves
  - Half 1 is for measurement only: non-additivity (eta) and DEG selection
  - Half 2 is for everything else: baseline inputs, model inputs, scoring targets
  - Reason: the noise that puts a pair in a "highly non-additive" group must not also be in the target it is scored on
- **Uncertainty:** bootstrap over perturbation pairs, never over cells
- **Two rollout arms (reviewer point 1):** Norman keeps guide A active the whole time, and neither arm matches that exactly, so we report both
  - `carried`: second step gets label B only; A survives only through the predicted state
  - `restated`: second step gets label A+B
- **Order and saturation are consistency checks only.** Norman measures joint perturbations, so it cannot validate application order or repeated activation biologically
- **Gate before adding any model (reviewer point 2):** it must pass a full input → output → input example (`scripts/roundtrip_check.py`)
- **Models:**
  - GEARS is the anchor, trained two ways (reviewer point 4)
    - standard split: diagnostic, how good is it as intended?
    - strict split: our experiment
  - scGPT is second, only after it passes the round-trip gate
  - Geneformer is out: its perturbation output is an embedding shift with no expression decoder
  - State and CPA are out for time
- **2026-11-02:** Calendar is 4.5 weeks, not 8. Experiment 3 (ComboSciPlex, Replogle) is cut
- **2026-11-03:** Midterm scope set to baselines, eta reliability, and GEARS joint prediction

---

## Data and resources

**Data** (none of it goes in git; everything under `data/` is ignored)

- **Norman, GEARS-processed** (use this one):
  - https://dataverse.harvard.edu/api/access/datafile/6154020
  - Zip; unpack so you get `data/gears/norman/perturb_processed.h5ad`
- **GEARS gene–GO graph** (required by GEARS):
  - https://dataverse.harvard.edu/api/access/datafile/6153417
  - Save as `data/gears/gene2go_all.pkl`
- **GEARS perturbation gene list** (required by GEARS):
  - https://dataverse.harvard.edu/api/access/datafile/6934320
  - Save as `data/gears/essential_all_data_pert_genes.pkl`
- **Easiest route:** GEARS fetches all three itself:
  ```bash
  python -c "from gears import PertData; PertData('data/gears').load(data_name='norman')"
  ```
- **Backup only:** scPerturb Norman, https://zenodo.org/records/13350497, file `NormanWeissman2019_filtered.h5ad` (699 MB). It has a different gene set, so do not mix it with the GEARS file
- **Raw, not needed:** GEO GSE133344
- **Sanity check after download:** `run_baselines.py` prints the dataset size first. It should be close to 105 singles and 131 doubles. If not, the condition labels were parsed wrong
- **Check we have the same data:** both machines should give exactly 91,205 cells, 5,045 genes, 105 singles, 131 doubles:
  ```bash
  python -c "import sys; sys.path.insert(0,'src'); from rolloutprobe.data import load_h5ad; d=load_h5ad('data/gears/norman/perturb_processed.h5ad'); print(d.n_cells,'cells', len(d.genes),'genes', len(d.singles()),'singles', len(d.doubles()),'doubles')"
  ```
  And the files should match these SHA-256 checksums (`shasum -a 256 <file>`):
  ```
  23ffb0fac6a847ff927cf7509d80d85052bfefbfb97610786a2dafaaefa0b6a0  data/gears/norman/perturb_processed.h5ad
  e0cdc85330b468c0b99fa4929f18c411124bca60a88153566ad1dd858d73a806  data/gears/norman/go.csv
  f145c5e84a53048d87942a417d870a4f2d8db50200b96e492b358c13aba8c771  data/gears/gene2go_all.pkl
  46c3dfe354d8ad5c0da22c69f3d0ca451987b1a61ed9d984279b22b9565ff8d7  data/gears/essential_all_data_pert_genes.pkl
  ```
  The two `.pkl` files are only needed for GEARS training; the baselines need just the `.h5ad`

**Software**

- GEARS: https://github.com/snap-stanford/GEARS (`pip install cell-gears`)
- PyTorch: https://pytorch.org/get-started/locally/ (match the cluster's CUDA)
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html (install after torch)
- scGPT, after the midterm: https://github.com/bowang-lab/scGPT (whole-human checkpoint, see "Pretrained scGPT Model Zoo" in its README)
- Use a fresh conda env with Python 3.10 or 3.11. GEARS pins numpy 1.26.4, scanpy 1.10.2, scipy 1.14.1

**Papers**

- Norman et al. 2019, *Science*: https://doi.org/10.1126/science.aax4438
- Ahlmann-Eltze & Huber 2025, *Nature Methods*: https://doi.org/10.1038/s41592-025-02772-6
- GEARS, Roohani et al. 2024, *Nature Biotechnology*: https://doi.org/10.1038/s41587-023-01905-6
- scGPT, Cui et al. 2024, *Nature Methods*: https://doi.org/10.1038/s41592-024-02201-0
- Virtual cell world model, Xing & Song 2026, *Cell* (motivation)

---

## Getting started (for either of us, on a new machine)

```bash
git clone https://github.com/Shreyan-A0I/perturb-doubletake.git
cd perturb-doubletake
conda create -n doubletake python=3.11 -y && conda activate doubletake
pip install -r requirements.txt
pytest -q                          # 6 tests on synthetic data, ~10 s, no download needed
# GPU work only: install torch, torch_geometric, cell-gears (links above)
# then get the data (see "Data and resources")
```

- No data yet? Run the full pipeline on synthetic data:
  ```bash
  python scripts/make_synthetic.py
  python scripts/run_baselines.py --data data/synthetic/norman_like.h5ad --out results/synthetic
  python scripts/make_figures.py --results results/synthetic --out figures/synthetic
  ```
- Figures can be built before any model runs: `python scripts/make_figures.py --fake --out figures/fake`

---

## Task checklist

Owner tags: **[S]** Shreyan, **[D]** Datta, **[S+D]** both. Tick items as you finish them and put your initials and the date beside each tick.

### Setup

- [ ] [S] Repo created and scaffold pushed
- [ ] [S] Add Datta as a collaborator on GitHub (needs his GitHub username)
- [ ] [S+D] Each of us clones the repo, runs `pytest -q`, and gets 6 passed
- [ ] [S] GPU environment: torch + torch_geometric + cell-gears installed
- [ ] [S] Norman downloaded to `data/gears/`; file path shared with Datta
- [ ] [D] Norman copied to Datta's machine; counts check out (~105 singles, ~131 doubles)

### Midterm sprint (due Nov 6)

**Results**

- [ ] [D] `run_baselines.py` on Norman: null hierarchy + eta reliability
- [ ] [D] `make_figures.py`: Fig 1 (eta reliability), Fig 2 (null hierarchy), Fig 3 (additive error vs eta)
- [ ] [S] Train GEARS, standard split (`train_gears.py --split standard`)
- [ ] [S] Train GEARS, strict split (`train_gears.py --split strict`)
- [ ] [S] `run_model.py --joint-only` for both GEARS models; rerun `make_figures.py`
- [ ] [S] Stretch: `roundtrip_check.py` on strict GEARS, a single example is enough

**Report sections**

- [ ] [S] Introduction / motivation: from the proposal, plus the reviewer changes
- [ ] [D] Related work: the papers that differ least from ours, and exactly how
- [ ] [S] Methodology: research questions, strict split, split-half design, rollout arms, round-trip gate
- [ ] [D] Experiments: preliminary results (Figs 1–3, null table, GEARS joint numbers)
- [ ] [D] Analysis: what the eta reliability and the additive-error trend tell us
- [ ] [S+D] Timeline for remaining work (see below)
- [ ] [S+D] Limitations
- [ ] [S+D] Proofread; submit

### After the midterm

- [ ] [S] Full rollout on GEARS strict: both arms, both orders, measured-input control (`run_model.py` without `--joint-only`)
- [ ] [S] GEARS standard-split rollout as the diagnostic comparison
- [ ] [S] scGPT: wrapper adapter, then round-trip gate. Only if it passes: joint + rollout
- [ ] [D] Rollout analysis: advantage over additive vs eta (slope + CI), order dependence vs eta, arm comparison
- [ ] [D] Stability: saturation ratio, drift from real cells over k = 1..10 steps
- [ ] [S+D] Decide by Nov 15: keep scGPT, or cut to GEARS only
- [ ] [S+D] Draft figures for final report by Nov 22
- [ ] [S+D] Final report + presentation (last class Dec 3)

### Cut order if time runs short

1. Discriminator-based drift metric
2. Five split seeds down to three
3. scGPT (GEARS alone is a complete result)
4. Stability experiment (rollout composition is the core question)

---

## Working conventions

- **Branches:** `main` must always pass `pytest -q`. Do the work on `shreyan/<topic>` or `datta/<topic>` and merge when it works
- **Never commit** data, models, results or figures. `.gitignore` covers `data/`, `models/`, `results/`, `figures/`, `*.h5ad`, `*.pkl`, `*.pt`. Share results through the drive folder instead
- **Results contract:** every prediction is one row in the schema in `src/rolloutprobe/schema.py`
  - Nulls write `results/baselines.csv`
  - Each model writes `results/model_<tag>.csv`
  - Figures read whatever is there
- **New model:** add an adapter under `src/rolloutprobe/models/` behind `predict(X, pert) -> X`, then pass `roundtrip_check.py` before anything else
- **Changing a design rule** (split, metrics, schema): add a dated line to "Key decisions" first and tell the other person
- **Shared drive for results/figures:** *(add link)*
- **Execution plan doc:** *(add link once shared)*

---

## Log

*One line per meaningful result or change, newest first. This doubles as the record for the final report's team contributions section.*

- 2026-11-03 [S] Scaffold built and tested on synthetic data; repo initialized
