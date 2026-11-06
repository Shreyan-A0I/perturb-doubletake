# Midterm report: sections owned by [S]

Draft written 2026-11-04 from the README, the RUNBOOK and the code. The proposal itself is
not in the repo, so the Introduction should be reconciled with its wording before submission.

---

## Introduction

Perturbation response models such as GEARS (Roohani et al., 2024), scGPT (Cui et al.,
2024) and State predict how a cell's expression profile changes after a genetic
perturbation: a control cell goes in, a predicted perturbed profile comes out. Because
input and output live in the same expression space, a prediction can in principle be
fed back in and perturbed again, `predict(predict(x, A), B)`. If these models have
learned a reusable representation of cell state, that rollout should approximate the
measured double perturbation A+B. If they are only one-step maps from control cells,
it should not.

The question matters because of where the field wants to go. Proposals for virtual cell
"world models" (Xing & Song, *Cell* 2026) envision simulating cells under sequences of
interventions, which requires exactly this property: a model's output must be a valid
input. As far as we know, no published work tests whether current models have it.

The starting point is sobering. Ahlmann-Eltze & Huber (*Nat Methods* 2025) showed that
on Norman et al.'s (2019) double perturbations, deep models do not beat a simple additive
baseline that sums the two single-perturbation effects. They evaluated only the joint
prediction, however. We ask the complementary question: whether models compose under
their own output.

We train on single perturbations only, hold out every double, and compare three
predictions for each pair: the model's joint prediction of A+B, its sequential rollout
A then B, and simple baselines, mainly addition. Our motivation is the world-model use
case, but our goal is narrower: we test whether a precondition for it holds, not build a
virtual cell.

**Changes since the proposal, in response to reviewers.**

1. *Rollout arms (point 1).* Norman's double perturbations keep guide A active the whole
   time, which neither rollout exactly reproduces. We report two arms: `carried`, where
   the second step receives label B only, and `restated`, where it receives A+B.
2. *Round-trip gate (point 2).* No model enters the comparison until it passes a full
   input → output → input check on real and generated inputs.
3. *Split-half cells (point 3).* Each condition's cells are split into two random
   halves, one used only for measuring non-additivity and the other for everything
   else, so that noise cannot drive both stratification and scoring.
4. *Two training regimes for GEARS (point 4).* GEARS is trained on its own standard
   split as a diagnostic, and on our strict singles-only split for the experiment.

We also narrowed scope for a 4.5-week calendar: the ComboSciPlex/Replogle experiment is
cut, Geneformer is excluded (it outputs an embedding shift with no expression decoder),
and State and CPA are out for time.

---

## Methodology

**Research questions.**

- **RQ1 (joint).** Trained on singles only, does a model predict held-out doubles better
  than the additive null, and does any advantage concentrate in pairs that are
  measurably non-additive?
- **RQ2 (composition).** Does the sequential rollout `f(f(x, A), B)` approach the
  measured double as well as the joint prediction, and how does it compare with addition?
- **RQ3 (diagnosis).** When rollout fails, is the cause that the model cannot accept
  non-control input at all, or that its own predictions drift off the distribution of
  real cells?

**Data.** Norman et al. (2019) CRISPRa Perturb-seq in K562 cells, as processed by GEARS:
91,205 cells × 5,045 genes, 105 single and 131 double perturbations, every double with
both singles measured. It is the only public dataset with double perturbations and all
of their singles.

**Strict split.** All single perturbations (and control) are training conditions; all
131 doubles are held out as test. This differs from GEARS's default "simulation" split,
which trains on some doubles. Under the strict split the model and the additive null see
the same information. For GEARS training, 10% of the singles are held out as validation.

**Split-half design.** Each condition's cells are divided at random into two halves,
repeated over five split seeds.

- *Half 1 is measurement only:* the non-additivity score eta and the choice of the
  top-20 differentially expressed genes (DEGs) used for scoring.
- *Half 2 is everything else:* inputs to the nulls and models, and scoring targets.

Non-additivity of a pair is eta = ‖Δ_AB − Δ_A − Δ_B‖, with Δ the pseudobulk shift from
control. Because the halves have independent noise, the cross-half product r₁·r₂ is an
unbiased estimate of the true squared residual, which tells us how much of eta is signal.
Pairs are stratified into eta tertiles on half 1.

**Baselines.** All built from half-2 pseudobulks of training conditions: no change
(control), train mean (control plus average single shift), additive (control + Δ_A +
Δ_B), and low-rank additive (singles denoised with a rank-10 SVD before adding, the
strict-split analogue of Ahlmann-Eltze & Huber's linear baseline).

**Model predictions.** Each model sits behind a common interface `predict(X, pert) → X`
that takes a matrix of cells and a perturbation and returns cells in the same gene space.
GEARS's own `predict()` samples control cells internally and cannot continue from a given
state, so our adapter builds GEARS's input graph directly from the expression we pass in.
Model predictions are anchored to the pseudobulk of their input condition, which removes
sampling noise from the ~300 input cells that the nulls do not carry; without this,
models would be penalized relative to baselines. For each held-out pair we compute:

- *joint:* `f(x₀, {A, B})`
- *rollout, carried:* `f(f(x₀, A), B)`
- *rollout, restated:* `f(f(x₀, A), {A, B})`
- *measured-input rollout:* `f(x_A, B)` from real A-perturbed cells, which separates
  failure to compose from failure on generated input (RQ3)
- both orders, A then B and B then A

For GEARS in the standard regime, doubles seen in training are excluded from scoring.

**Round-trip gate.** Before any model joins the comparison, it must pass a check on
single perturbations: real A-perturbed cells passed through with no perturbation should
come back roughly unchanged (r > 0.95 on expression and on shifts from control), and the
same should hold for the model's own predicted A state. Failure on real input means
composition cannot be tested for that model, which is itself a finding; failure only on
generated input means distribution shift.

**Consistency checks.** Order dependence (A→B vs. B→A) and saturation (applying A twice)
are reported as computational consistency checks only. Norman measures joint
perturbations and cannot validate either biologically.

**Metrics and inference.** MSE on the top-20 DEGs (selected on half 1) and Pearson
correlation of predicted vs. observed shift from control, over all genes and over the
top-20 DEGs. Per-pair results are averaged over split seeds; all 95% intervals are from
2,000 bootstrap resamples over perturbation pairs, never over cells, since the pair is
the unit of inference. The main rollout analysis is a model's advantage over additive as
a function of eta, summarized by a slope with a bootstrap CI.

**Models.** GEARS is the anchor, trained in both regimes. scGPT follows after the midterm
only if it passes the round-trip gate.
