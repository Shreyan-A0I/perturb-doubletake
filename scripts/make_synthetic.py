"""Write a small Norman-shaped synthetic dataset with KNOWN interactions.

Used to test the pipeline end to end without the real data. A fixed fraction of
pairs get a genuine interaction term; the rest are exactly additive in log-rate
space. A working pipeline should give those pairs higher eta, and the additive
null should do worse on them.

    python scripts/make_synthetic.py --out data/synthetic/norman_like.h5ad
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic/norman_like.h5ad")
    p.add_argument("--genes", type=int, default=400)
    p.add_argument("--singles", type=int, default=24)
    p.add_argument("--pairs", type=int, default=40)
    p.add_argument("--interacting-frac", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    G = args.genes
    genes = [f"GENE{i:04d}" for i in range(G)]
    perturbed = genes[: args.singles]            # perturbed genes are also measured genes
    mu = rng.normal(0.0, 1.0, G)                 # baseline log rate

    def sparse_effect(n_hit, scale):
        v = np.zeros(G)
        idx = rng.choice(G, n_hit, replace=False)
        v[idx] = rng.normal(0, scale, n_hit)
        return v

    single_eff = {g: sparse_effect(25, 1.0) for g in perturbed}
    for i, g in enumerate(perturbed):            # CRISPRa: the target goes up
        single_eff[g][i] += 2.0

    all_pairs = list(itertools.combinations(perturbed, 2))
    pair_idx = rng.choice(len(all_pairs), args.pairs, replace=False)
    pairs = [all_pairs[i] for i in pair_idx]
    n_int = int(round(args.interacting_frac * len(pairs)))
    interacting = set(pairs[:n_int])

    def cells(log_rate, n):
        lib = rng.gamma(4.0, 2500.0 / 4.0, n)                       # library size
        rate = np.exp(log_rate)
        rate = rate / rate.sum()
        return rng.poisson(np.outer(lib, rate)).astype(np.float32)

    blocks, labels = [], []
    blocks.append(cells(mu, 3000)); labels += ["ctrl"] * 3000
    for g in perturbed:
        n = int(rng.integers(200, 500))
        blocks.append(cells(mu + single_eff[g], n)); labels += [f"{g}+ctrl"] * n
    truth = []
    for a, b in pairs:
        inter = sparse_effect(20, 1.2) if (a, b) in interacting else np.zeros(G)
        n = int(rng.integers(200, 500))
        blocks.append(cells(mu + single_eff[a] + single_eff[b] + inter, n))
        labels += [f"{a}+{b}"] * n
        a_, b_ = sorted((a, b))
        truth.append({"pert_a": a_, "pert_b": b_, "interacting": (a, b) in interacting,
                      "true_eta_lograte": float(np.linalg.norm(inter))})

    X = sp.csr_matrix(np.vstack(blocks))
    adata = ad.AnnData(X=X, obs=pd.DataFrame({"condition": labels}),
                       var=pd.DataFrame({"gene_name": genes}, index=genes))
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    pd.DataFrame(truth).to_csv(out.with_suffix(".truth.csv"), index=False)
    print(f"wrote {out}: {adata.n_obs} cells x {G} genes, {len(perturbed)} singles, "
          f"{len(pairs)} doubles ({n_int} interacting)")


if __name__ == "__main__":
    main()
