# Midterm report: sections owned by [D]

Draft written 2026-11-04 from the `run_baselines.py` / `make_figures.py` run on the
GEARS-processed Norman file (91,205 cells, 5,045 genes, 105 singles, 131 doubles;
checksums match the README). GEARS numbers added 2026-11-04 after S2/S3 (seed 1, 20 epochs).
Citations marked *(verify)* need checking of venue/year before submission.

---

## Related work

**Predicting double perturbations from singles.** Norman et al. (2019) measured 131
pairwise CRISPRa perturbations together with all of their constituent singles in K562
cells and showed that most pairs combine roughly additively in expression space, with a
minority showing synergy, buffering or neomorphic behavior. This dataset has become the
standard benchmark for combinatorial prediction, and it is the one we use.

**Deep perturbation models.** GEARS (Roohani et al., 2024) encodes each perturbed gene
with a gene-ontology graph and predicts the post-perturbation expression of a control
cell; it was designed for unseen combinations and is evaluated on Norman doubles.
scGPT (Cui et al., 2024) fine-tunes a pretrained transformer on the same task. CPA
(Lotfollahi et al., 2023) and scGen (Lotfollahi et al., 2019) represent perturbations as
vectors in a learned latent space and compose them by addition there; State (Arc
Institute, 2025 *(verify)*) is a recent large transition model. All of these are trained
and evaluated as *one-step* maps, control → perturbed.

**Closest work: simple baselines beat deep models.** Ahlmann-Eltze & Huber (2025)
showed that on Norman doubles, adding the two single-perturbation shifts beats GEARS,
scGPT and other foundation models, and that a linear model on learned embeddings matches
them on unseen singles. Several benchmarks reached related conclusions (e.g.
PertEval-scFM, Wenteler et al.; Csendes et al.; Systema, Viñas Torné et al. *(verify
all three)*), including that much apparent accuracy comes from systematic shifts shared
by all perturbations rather than perturbation-specific effects. Our study differs from
these in three concrete ways:

1. **What is evaluated.** They score only the joint prediction `f(x, A+B)`. We also
   feed the model's own output back in, `f(f(x, A), B)`, and ask whether the model
   carries a reusable cell state. To our knowledge no benchmark has tested this.
2. **Split.** GEARS's default split, used in most comparisons, places some doubles in
   training. We train on singles only and hold out all 131 doubles, so the additive null
   and the models see exactly the same information.
3. **Noise control.** Prior comparisons stratify pairs by non-additivity measured on the
   same cells that are then scored. We split each condition's cells into two halves:
   non-additivity (eta) and DEG selection use half 1, every input and target uses
   half 2. This removes the selection bias where noise that makes a pair look
   "non-additive" also inflates its error.

**Latent-space composition.** scGen and CPA compose effects by vector addition inside a
latent space, which is the closest prior idea to rollout. But composition there is a
modelling assumption built into the architecture, not a property that is tested on
measured doubles, and it is done in latent space rather than by re-entering the model
through its expression-space input.

**Virtual cells and world models.** Xing & Song (2026) argue for virtual cell models that
can be rolled forward under sequences of interventions. Such a model needs its outputs to
be valid inputs. We do not build one; we test whether current perturbation models
already satisfy that precondition on the one dataset where it can be checked against
measured combinations.

---

## Experiments (preliminary results)

**Setup.** GEARS-processed Norman data (91,205 cells × 5,045 genes; 105 singles,
131 doubles; median 257 cells per double, minimum 49). All singles are training
conditions; all doubles are test. Each condition's cells are split at random into two
halves, repeated for 5 split seeds; per-pair results are averaged over seeds, and 95%
confidence intervals come from 2,000 bootstrap resamples over the 131 pairs. Metrics:
MSE on the top-20 differentially expressed genes (chosen on half 1, scored on half 2),
and Pearson correlation of predicted vs. observed shift from control ("Pearson δ"), over
all genes and over the top-20 DEGs.

**Null hierarchy (Fig. 2, Table 1).** The additive null, control + Δ_A + Δ_B, is the
strongest by every metric. Denoising the single shifts with a rank-10 SVD makes it worse,
not better: low-rank addition beats raw addition on only 18% of pairs (23/131).

*Table 1. Nulls on 131 held-out doubles, mean [95% bootstrap CI].*

| Null              | MSE@20 DEG ↓           | Pearson δ (all) ↑     | Pearson δ (20 DEG) ↑  |
| ----------------- | ---------------------- | --------------------- | --------------------- |
| No change         | 0.734 [0.643, 0.824]   | undefined             | undefined             |
| Train mean        | 0.575 [0.494, 0.656]   | 0.533 [0.488, 0.575]  | 0.644 [0.582, 0.700]  |
| Low-rank additive | 0.140 [0.123, 0.158]   | 0.789 [0.766, 0.810]  | 0.870 [0.847, 0.891]  |
| **Additive**      | **0.081 [0.067, 0.096]** | **0.873 [0.857, 0.886]** | **0.959 [0.952, 0.964]** |
| GEARS, standard † | 0.232 [0.194, 0.271]   | 0.605 [0.558, 0.653]  | 0.798 [0.740, 0.851]  |
| GEARS, strict ‡   | 0.181 [0.161, 0.204]   | 0.673 [0.645, 0.700]  | 0.862 [0.833, 0.888]  |

(No-change predicts zero shift, so its correlation is undefined.) † 71 doubles: GEARS's
own split trains on 57 doubles, which are excluded; additive on the same 71 pairs: MSE
0.085, Pearson δ 0.869. ‡ 128 doubles: three doubles containing a gene outside GEARS's
GO graph (IER5L+LYL1, RHOXF2BB+SET, RHOXF2BB+ZBTB25) cannot be predicted; additive on
the same 128 pairs: MSE 0.082, Pearson δ 0.872.

**Non-additivity is a reliable property of the pair (Fig. 1).** For each pair, eta is the
norm of the residual Δ_AB − Δ_A − Δ_B. Measured independently on the two cell halves,
eta agrees with Spearman ρ = 0.944 (Pearson 0.941; per seed 0.932–0.955). The residual
vectors themselves point the same way across halves (median cosine 0.55). Because the
two halves have independent noise, r₁·r₂ is an unbiased estimate of the true squared
residual; it is positive for 99.5% of pairs, and it accounts for 61% of the measured
eta² on average. So eta is mostly signal, with a noise floor of roughly 40%.

**Additive error rises with eta (Fig. 3, Table 2).** Splitting pairs into eta tertiles
(eta from half 1, error on half 2), the additive MSE rises about fivefold from the low to
the high tertile, with non-overlapping CIs. Across pairs, half-1 eta and half-2 additive
MSE correlate with Spearman ρ = 0.82.

*Table 2. Additive null by half-1 eta tertile (~44 pairs each).*

| Eta tertile | Additive MSE@20 DEG [95% CI] | Pearson δ (all) |
| ----------- | ---------------------------- | --------------- |
| Low         | 0.028 [0.024, 0.033]         | 0.880           |
| Mid         | 0.063 [0.052, 0.075]         | 0.878           |
| High        | 0.152 [0.121, 0.186]         | 0.862           |

The most non-additive pairs are dominated by CEBP family combinations (CEBPA+CEBPB,
CEBPA+CEBPE, CEBPE+SET, CEBPA+KLF1), consistent with the redundant-differentiation
programs that Norman et al. described.

**GEARS joint prediction (Fig. 2, Table 1).** GEARS was trained for 20 epochs under both
regimes and scored on the same cells and metrics as the nulls. In both, it is clearly worse
than addition. Strict split, 128 doubles: MSE@20 DEG 0.181 vs. 0.082 for addition
(difference 0.100 [0.077, 0.122]); GEARS beats addition on only 14% of pairs. Standard
split, 71 doubles: 0.232 vs. 0.085 (difference 0.148 [0.106, 0.187]); GEARS is better on
13% of pairs. GEARS also loses to the low-rank additive null (0.141) in both regimes and
beats only the train-mean and no-change nulls. By eta tertile (strict), GEARS's MSE is
0.140 / 0.210 / 0.194 against additive 0.028 / 0.064 / 0.153: its deficit shrinks in the
high-eta tertile but does not reverse. This reproduces Ahlmann-Eltze & Huber's finding
under a stricter split. The standard and strict rows are not directly comparable, as they
are scored on different pair sets (the standard set includes pairs with an unseen single).

**Round-trip gate (stretch).** The strict GEARS model passes `roundtrip_check.py` on the 5
singles with the most cells (KLF1, BAK1, CEBPE, ETS2, UBASH3B): passing real A-perturbed
cells through with no perturbation returns them almost unchanged (r = 0.999 on expression,
0.80–0.99 on shifts from control), and the same holds for its own predicted states. So
GEARS can accept its own output, and the rollout experiment is well posed for it. But the
pass is structural rather than learned. The saturation ratio, ‖x₂ − x₁‖ / ‖x₁ − x₀‖ when A
is applied twice, is exactly 1.0 for all five genes. The GEARS source explains why
(`gears/model.py`, forward pass): the input expression enters only as the final residual,
`out = x + g(pert)`, and the shift g depends on the perturbation and gene embeddings but not
on the input cell. GEARS's prediction is therefore state-independent. Consequences for the
experiment: the carried rollout is exactly x + g(A) + g(B), an additive combination of
GEARS's own single-perturbation predictions; the order of application cannot matter; and
the restated rollout is x + g(A) + g(A,B). So for GEARS, the composition question has an
analytic answer, and the empirical question becomes whether g(A) + g(B) is closer to the
measured double than GEARS's joint g(A,B). That comparison is still informative, and
`run_model.py` without `--joint-only` produces it, but it should be framed this way in the
report. scGPT, whose output does depend on input expression, is the model where
composition is a genuinely open question.

---

## Analysis

**Eta is usable as a stratifying variable.** The central worry about stratifying by
non-additivity was that it might mostly measure noise, so that "highly non-additive"
pairs would just be the noisiest ones, and their high error would be a regression-to-
the-mean artifact. The split-half design rules this out: eta replicates across disjoint
cells (ρ = 0.94), and about 61% of eta² survives the cross-half estimate. Since the
strata are defined on half 1 and every error is computed on half 2, the rising error in
Table 2 cannot come from shared noise. The strata therefore pick out pairs where the
biology really departs from addition, and these are where a model with real interaction
knowledge has room to beat the additive null.

**Addition is a hard bar, and the room above it is concentrated.** Additive prediction
reaches Pearson δ 0.87 over all genes and 0.96 over the top-20 DEGs. Most of its error
sits in the top eta tertile, where MSE is five times that of the low tertile. This means
an overall average will mostly reflect the easy, near-additive pairs; the comparisons
that matter for both the joint and the rollout question are those in the high-eta
stratum. That is why the rollout analysis after the midterm is planned as advantage over
additive as a function of eta, rather than as a single average.

**Denoising singles does not help.** The low-rank additive null, the strict-split
analogue of Ahlmann-Eltze & Huber's linear baseline, is worse than raw addition on 82%
of pairs. With single-cell pseudobulks from ~100–300 cells, the singles in this dataset
are already precise enough that the rank-10 projection removes more real
perturbation-specific signal than noise. A model trained only on singles gains nothing
from smoothing them; it has to know something about how effects interact.

**Caveat: eta partly tracks effect size.** Absolute eta correlates with the size of the
double's shift (ρ = 0.54), and the double's shift size alone correlates with additive
MSE about as well (ρ = 0.58). Eta relative to the double's shift is almost unrelated to
additive MSE (ρ = 0.15). Part of the eta → error trend is therefore "larger
perturbations have larger errors in absolute terms", and not only "interacting pairs are
harder". Pearson δ, which is scale-free, barely falls across strata (0.880 → 0.862).
For the final report we should (a) report model advantage over additive within strata,
which cancels the scale, and (b) check that the conclusions hold with relative eta or
with effect size as a covariate.

**GEARS cannot carry state.** GEARS's predicted shift does not depend on the input cell
(see Experiments), so it is a one-step predictor by architecture: rolling it out reduces to
adding its own single-perturbation predictions. Its joint prediction is also worse than
adding the measured single shifts (MSE 0.181 vs. 0.082 on the strict split). Together, the
midterm results already answer RQ1 for GEARS negatively and make RQ2 for GEARS a comparison
of two additive constructions. This strengthens the case for getting scGPT through the
round-trip gate after the midterm.

**What this means for the next steps.** The additive null sets a high bar on Norman,
and the high-eta tertile gives a reliable, noise-independent set of pairs where a model
could beat it. If GEARS on the strict split cannot beat addition there in joint
prediction, a failure of its rollout would be hard to interpret; if it does, the
rollout test asks whether that advantage survives when the model must re-use its own
predicted state.
