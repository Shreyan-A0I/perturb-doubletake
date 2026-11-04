# Tasks, step by step

Executable version of the midterm items in the [README](../README.md) checklist. Each task has its owner tag (**[S]** Shreyan, **[D]** Datta), the commands to run, and a check for the output. Do the shared setup first, then your own tasks in order. Run every command from the repo root, and stop at any check that fails. Background and design rules: [README](../README.md) and [RUNBOOK](RUNBOOK.md).

## Rules

1. **Work on a branch** named `<shreyan|datta>/<topic>`, for example `datta/midterm-baselines`. Never commit to `main` directly. Never force-push.
2. **Never commit data or outputs.** `data/`, `results/`, `figures/`, `models/`, `*.h5ad`, `*.pkl`, `*.pt`, `*.zip` stay ignored. Run `git status --short` before every commit and check that only intended files are staged.
3. **Do not edit `src/`, `scripts/` or `tests/` while running these tasks.** If a test or script fails, stop and report the full error. Code fixes are a separate change on their own branch.
4. **Commit messages are one line**: no body, no trailers. Use the existing git identity; do not change `user.name` or `user.email`.
5. **Only download the files named in shared step 2.** Ask before downloading anything else.
6. **Only install the GPU stack (torch, torch_geometric, cell-gears) in task S1.** Every other task runs on CPU.

## Shared setup (both)

### 1. Environment and tests

```bash
git checkout -b <shreyan|datta>/<topic>
conda create -n doubletake python=3.11 -y && conda activate doubletake
pip install -r requirements.txt
pytest -q
```

- **Check:** `6 passed`. Then delete `.pytest_cache/` and any `__pycache__/`.

### 2. Data

Skip whatever is already in place. The Norman dataset needs about 2.5 GB of free disk.

```bash
mkdir -p data/gears
curl -L -o /tmp/norman.zip https://dataverse.harvard.edu/api/access/datafile/6154020
unzip -q /tmp/norman.zip -d data/gears/
rm /tmp/norman.zip
```

Only on the machine that will train GEARS, also fetch the two GEARS files (about 10 MB together):

```bash
curl -L -o data/gears/gene2go_all.pkl https://dataverse.harvard.edu/api/access/datafile/6153417
curl -L -o data/gears/essential_all_data_pert_genes.pkl https://dataverse.harvard.edu/api/access/datafile/6934320
```

- **Check the layout:** `data/gears/norman/perturb_processed.h5ad` (about 2.1 GB) and `data/gears/norman/go.csv` exist.
- **Check the counts:** run the command under "Check we have the same data" in the README. It must print exactly `91205 cells 5045 genes 105 singles 131 doubles`.
- **Check the checksums:** `shasum -a 256 <file>` (on Linux: `sha256sum`) for each downloaded file must match the README values.
- If any check fails, stop and report the output. Do not continue on different data.

## D1. Baselines and eta reliability [D]

CPU, about 10 seconds.

```bash
python scripts/run_baselines.py
```

- **Check:** it writes `results/baselines.csv`, `results/eta.csv` and `results/summary.json`.
- **Expected numbers** (seeds are fixed in `configs/default.yaml`, so these should match closely):
  - `pearson_delta`, additive: about 0.873, the best of the four nulls
  - Additive MSE@20DEG by eta stratum: low about 0.028, mid about 0.063, high about 0.152 (rising with eta)
  - `nochange` shows `nan` for Pearson delta. This is expected: it predicts zero change, so the correlation is undefined
- If the numbers are far off, stop and report the full printed output.

## D2. Figures 1–3 [D]

```bash
python scripts/make_figures.py
```

- **Check:** `figures/` contains `fig1_eta_reliability`, `fig2_null_hierarchy` and `fig3_additive_vs_eta`, each as `.png` and `.pdf`.

## S1. GPU environment [S]

On the GPU machine, in the `doubletake` env from shared step 1:

1. Install torch for the machine's CUDA version, following https://pytorch.org/get-started/locally/
2. Install torch_geometric, following https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
3. `pip install cell-gears`. GEARS pins numpy 1.26.4, scanpy 1.10.2 and scipy 1.14.1, so this may downgrade packages

- **Check:**
  ```bash
  python -c "import torch, torch_geometric, gears; print(torch.__version__, torch.cuda.is_available())"
  pytest -q
  ```
  It must print `True` for CUDA, and the tests must still give `6 passed` after the downgrades.

## S2. Train GEARS, both splits [S]

GPU, about an hour each. The GEARS code was checked against GEARS master but has never been run, so read errors carefully and report them in full.

```bash
python scripts/train_gears.py --split standard --seed 1
python scripts/train_gears.py --split strict   --seed 1
```

- **Check:** each run ends with `saved models/gears_<split>_s1`, and that folder contains `rolloutprobe_split.json`.
- The strict run prints its train, val and test label counts. Test should be the 131 doubles.

## S3. GEARS joint predictions [S]

```bash
python scripts/run_model.py --model gears --model-dir models/gears_standard_s1 --tag gears_standard --joint-only
python scripts/run_model.py --model gears --model-dir models/gears_strict_s1   --tag gears_strict   --joint-only
python scripts/make_figures.py
```

- **Check:** each run prints `scoring N doubles; M excluded as seen in training` and writes `results/model_<tag>.csv`.
  - Strict: 0 excluded, since no double was in training
  - Standard: some excluded. This is expected, because GEARS's own split trains on some doubles
- The figures now include both GEARS models.

## S4. Round-trip check, stretch [S]

```bash
python scripts/roundtrip_check.py --model gears --model-dir models/gears_strict_s1 --tag gears_strict
```

- Prints a per-gene table and a verdict: `PASS`, `REAL-INPUT FAIL` or `GENERATED-INPUT FAIL`. Writes `results/roundtrip_gears_strict.json`.
- Any verdict is a result. Report it; do not change code to make it pass.

## Record and push (both)

1. In the README:
   - Tick the items you finished, with initials and date beside each tick
   - Add a line at the top of the Log: date, owner tag, and the key numbers
   - Update "Last updated" in the Status block
2. Commit and push your branch:
   ```bash
   git add README.md
   git status --short        # only README.md should be staged
   git commit -m "<one-line summary>"
   git push -u origin <your-branch>
   ```
3. Do not merge into `main`; the branch owner reviews and merges. If both of you edited the README checklist, keep both sets of ticks when resolving the merge.
4. `results/` and `figures/` are not in git. Upload them to the shared drive (link in the README's Working conventions).

## Report back

Report in this order:

- Tests: result
- Data: counts, and whether the checksums matched
- Each task you ran: its printed output (summary table, verdict, file paths)
- Git: branch, commit hash, push result
- Anything skipped or failed, and why

## Not covered here

The report sections are writing tasks; see the README checklist. Figures 1–3, the D1 numbers and the S3 results are the inputs to Experiments and Analysis. Post-midterm work (full rollout, scGPT) gets added here when it starts.
