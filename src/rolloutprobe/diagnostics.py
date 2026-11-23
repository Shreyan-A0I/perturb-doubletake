"""Why is A2 large? Carry, erasure and output-variance diagnostics, plus the
interaction-blind A3 references. Used by scripts/diagnostics.py.

Notation follows pair_eval.py. For a held-out pair (A, B) and one split seed:
    X0, X0b   two disjoint control samples (half 2)        Xa   real A cells (half 2)
    D_A       measured A shift, half-2 pseudobulk: pb[(A, 2)] - pb[("ctrl", 2)]
    deg       top-k DEGs of the double, chosen on half 1 (as pair_eval)
    e(X)      mean f(X, B) - mean X  (predicted shift of B from input X)

Model diagnostics, per pair (pair_diagnostics):
    carry       ||mean f(Xa,B) - mean f(X0,B)|| / ||mean Xa - mean X0||
                How much of the input difference survives into the output.
                1 = carried in full (out = x + g(pert), GEARS); 0 = input ignored.
                carry_20deg: same on the DEG coordinates. carry_cos: cosine between the
                two difference vectors (direction of what is carried).
    erasure     e = e(Xa) - e(X0), the A2 difference vector (sens_rel = ||e|| / ||e(X0)||).
                A model that ignores its input gives e = -(mean Xa - mean X0) ~ -D_A.
                eras_corr      Pearson(e, -D_A)
                eras_rel       ||e|| / ||D_A||
                eras_resid     ||e + D_A|| / ||D_A||   (what is left once erasure is removed)
                eras_*_sample  the same with D_A replaced by the sampled-cell difference
                               mean Xa - mean X0; then eras_resid_sample == carry exactly.
                *_20deg        on the DEG coordinates.
    variance    mean over genes of the across-cell variance (ddof=1):
                vr_ctrl    var f(X0,B) / var X0           (prediction vs its real input)
                vr_input_a var f(Xa,B) / var Xa
                vr_target  var f(Xa,B) / var(real A+B cells, all of half 2)
                vr_target_b var f(X0,B) / var(real B cells, all of half 2)
    A2 check    sens_rel, sens_null_rel recomputed exactly as pair_eval (with the same
                cell draws they match results/model_<tag>.csv).

Interaction-blind references, per pair and split seed (reference_row). No model.
r2 = D_AB - D_A - D_B on half 2 (the A3 target). Reference vectors from HALF 1, so
they share no sampling noise with r2:
    forget_a     -D_A(h1)            what an input-erasing rollout implies
    shrink_sum   -(D_A + D_B)(h1)    what a prediction that under-shoots the sum implies
Pearson with r2 on the top-k DEGs (half 1) and all genes. The *_h2 columns use the
half-2 shifts instead; they share noise with r2 (r2 contains -D_A(h2)), so they are
inflated and only recorded to explain earlier scratch numbers.
"""
from __future__ import annotations

import numpy as np

from . import metrics
from .eta import delta, residual


def _mean(X):
    return np.asarray(X).mean(axis=0)


def mean_gene_var(X) -> float:
    X = np.asarray(X, dtype=np.float64)
    return float(X.var(axis=0, ddof=1).mean()) if len(X) > 1 else np.nan


def _ratio(num, den) -> float:
    return float(num / den) if den > 0 else np.nan


def _norm(x) -> float:
    return float(np.linalg.norm(x))


def _cos(x, y) -> float:
    return _ratio(float(x @ y), _norm(x) * _norm(y))


def erasure_terms(e: np.ndarray, d_a: np.ndarray, suffix: str = "") -> dict:
    nd = _norm(d_a)
    return {f"eras_corr{suffix}": metrics.pearson(e, -d_a),
            f"eras_rel{suffix}": _ratio(_norm(e), nd),
            f"eras_resid{suffix}": _ratio(_norm(e + d_a), nd)}


class PredCache:
    """f(X, pert) for inputs that repeat across pairs (the control samples), keyed by
    a caller-supplied name. Valid because all adapters predict deterministically."""

    def __init__(self, model):
        self.model, self.store, self.calls = model, {}, 0

    def __call__(self, X, pert, key=None):
        if key is not None and (key, pert) in self.store:
            return self.store[(key, pert)]
        self.calls += 1
        out = np.asarray(self.model.predict(X, pert))
        if key is not None:
            self.store[(key, pert)] = out
        return out


def pair_diagnostics(pred: PredCache, a: str, b: str, pb: dict, X0, X0b, Xa, X_ab, X_b,
                     k: int) -> dict:
    """X_ab, X_b: real target cells (half 2) for the variance ratios only."""
    deg = metrics.top_k_deg(delta(pb, f"{a}+{b}", 1), k)
    x0m, x0bm, xam = _mean(X0), _mean(X0b), _mean(Xa)
    f0 = pred(X0, (b,), key="X0")
    f0b = pred(X0b, (b,), key="X0b")
    fa = pred(Xa, (b,))
    f0m, fam = _mean(f0), _mean(fa)

    d_in = xam - x0m                     # input difference (sampled cells)
    d_out = fam - f0m                    # output difference
    d_a = pb[(a, 2)] - pb[("ctrl", 2)]   # measured A shift, as pair_eval's d_a_meas
    e0, ea, e0b = f0m - x0m, fam - xam, _mean(f0b) - x0bm
    e = ea - e0                          # == d_out - d_in

    n0 = _norm(e0)
    row = {
        "carry": _ratio(_norm(d_out), _norm(d_in)),
        "carry_20deg": _ratio(_norm(d_out[deg]), _norm(d_in[deg])),
        "carry_cos": _cos(d_out, d_in),
        **erasure_terms(e, d_a),
        **erasure_terms(e[deg], d_a[deg], "_20deg"),
        **erasure_terms(e, d_in, "_sample"),
        "norm_d_a": _norm(d_a), "norm_d_in": _norm(d_in), "norm_e": _norm(e),
        "sens_rel": _ratio(_norm(e), n0),
        "sens_null_rel": _ratio(_norm(e0b - e0), n0),
        "sens_corr": metrics.pearson(e0, ea),
    }
    v0, va, vf0, vfa = (mean_gene_var(x) for x in (X0, Xa, f0, fa))
    vab, vb = mean_gene_var(X_ab), mean_gene_var(X_b)
    row.update({"var_ctrl": v0, "var_a": va, "var_target_ab": vab, "var_target_b": vb,
                "var_pred_ctrl": vf0, "var_pred_a": vfa,
                "vr_ctrl": _ratio(vf0, v0), "vr_input_a": _ratio(vfa, va),
                "vr_target": _ratio(vfa, vab), "vr_target_b": _ratio(vf0, vb)})
    return row


def reference_row(pb: dict, a: str, b: str, k: int) -> dict:
    deg = metrics.top_k_deg(delta(pb, f"{a}+{b}", 1), k)
    r2 = residual(pb, a, b, 2)
    out = {}
    for h, suf in ((1, ""), (2, "_h2")):
        refs = {"forget_a": -delta(pb, a, h), "shrink_sum": -(delta(pb, a, h) + delta(pb, b, h))}
        for name, v in refs.items():
            out[f"{name}_r_20deg{suf}"] = metrics.pearson(v[deg], r2[deg])
            out[f"{name}_r_all{suf}"] = metrics.pearson(v, r2)
    r1 = residual(pb, a, b, 1)
    out["rep_r_20deg"] = metrics.pearson(r1[deg], r2[deg])
    out["rep_r_all"] = metrics.pearson(r1, r2)
    return out
