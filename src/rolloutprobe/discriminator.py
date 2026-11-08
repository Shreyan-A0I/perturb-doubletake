"""Can a classifier tell real cells from predicted cells? Cross-validated accuracy;
0.5 means indistinguishable.

Per fold: PCA fitted on the training fold only, standardized components, L2
logistic regression (scipy L-BFGS, no extra dependency). Classes are balanced by
subsampling the larger one, so 0.5 is the chance level.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _fit_logreg(Z: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    Zb = np.hstack([Z, np.ones((len(Z), 1))])

    def loss(w):
        s = Zb @ w
        # log(1 + exp(-y s)) with y in {-1, 1}, stable
        m = -y * s
        nll = np.logaddexp(0, m).mean()
        p = 1 / (1 + np.exp(-m))
        grad = (Zb * (-y * p)[:, None]).mean(axis=0)
        reg = w.copy()
        reg[-1] = 0
        return nll + 0.5 * l2 * (w[:-1] @ w[:-1]) / len(Z), grad + l2 * reg / len(Z)

    res = minimize(loss, np.zeros(Zb.shape[1]), jac=True, method="L-BFGS-B")
    return res.x


def cv_accuracy(A: np.ndarray, B: np.ndarray, n_comp: int = 50, folds: int = 5,
                l2: float = 1.0, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    n = min(len(A), len(B))
    if n < folds:
        return np.nan
    A, B = A[rng.choice(len(A), n, replace=False)], B[rng.choice(len(B), n, replace=False)]
    X = np.vstack([A, B])
    y = np.r_[np.ones(n), -np.ones(n)]
    fold = rng.permutation(np.arange(2 * n) % folds)
    correct = 0
    for f in range(folds):
        tr, te = fold != f, fold == f
        mu = X[tr].mean(axis=0)
        _, _, vt = np.linalg.svd(X[tr] - mu, full_matrices=False)
        comps = vt[:min(n_comp, vt.shape[0] - 1)]
        Ztr, Zte = (X[tr] - mu) @ comps.T, (X[te] - mu) @ comps.T
        sd = Ztr.std(axis=0)
        sd[sd == 0] = 1
        Ztr, Zte = Ztr / sd, Zte / sd
        w = _fit_logreg(Ztr, y[tr], l2)
        pred = np.sign(np.hstack([Zte, np.ones((len(Zte), 1))]) @ w)
        correct += int((pred == y[te]).sum())
    return correct / (2 * n)
