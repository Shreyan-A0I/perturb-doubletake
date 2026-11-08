"""CPA (Compositional Perturbation Autoencoder, cpa-tools 0.8.8) behind predict().

Source checked: github.com/theislab/cpa at commit fbd7c0250edc23eff003a10c99655579c53afd63
(cpa/_module.py, _model.py, _utils.py, _data.py, _task.py, docs/tutorials/Norman.ipynb).

THE QUESTION THAT DECIDES WHETHER CPA CAN BE TESTED: does prediction read the input
cell, or sample its own cells like GEARS.predict()? It reads the input. Verified:
    CPAModule.inference(x, perts, perts_doses, covars_dict) encodes the given x:
        z_basal = encoder(x)                     ('gauss' loss: x used as is)
        z       = z_basal + pert_network(perts, doses) + covariate embeddings
    generative(z) decodes; get_expression() returns px.loc (gauss) as the prediction.
    CPA.predict(adata) runs get_expression over adata's cells, with X as input and the
    perturbation labels from adata.obs. Its docstring and the Norman tutorial produce
    counterfactuals exactly so: set adata.X to control cells, keep the target labels.
So the public API takes arbitrary input cells, and A2 is meaningful for CPA.

Why this adapter calls module.get_expression() directly instead of CPA.predict():
CPA.predict needs a registered AnnData per call; scvi-tools keeps a manager (holding
the AnnData) for every new object it sees, and run_model.py makes ~6,500 predict
calls per run. get_expression() is the same per-batch call CPA.predict makes.
scripts/train_cpa.py checks that the two give the same output on real cells.

Also verified against source:
    perturbation encoding  setup_anndata: obs[perturbation_key] strings, combinations
                           joined by "+", up to max_comb_len (2) indices per cell, padded
                           with index 0; control is itself an entry ("ctrl") with dose 1.0
    dosages                dosage_key=None -> "1.0" per perturbation token
    PerturbationNetwork    sum over the combination of dose-scaled embeddings, padding
                           masked; so an unseen double A+B is embedded as e_A + e_B
    covariates             none here (one cell type); the adapter refuses a model with any
Not verified (needs the installed package; see docs/TASKS.md step 8b, C2):
    CPA.load() restoring the module and CPA.pert_encoder from a saved directory
    that get_expression() runs on tensors built here (key names from CPA_REGISTRY_KEYS)

STATUS: written against the source above, never executed (no cpa-tools where it was
written). tests/test_cpa_adapter.py checks the plumbing with a stand-in module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..wrapper import register

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("CPA adapter needs torch (the cpa env; docs/TASKS.md step 8b, C1).") from e

CONTROL = "ctrl"


@register("cpa")
class CPAPredictor:
    """Loads a directory written by scripts/train_cpa.py: CPA's own saved model
    (model.pt, CPA_info.json, adata.h5ad) plus rolloutprobe_split.json."""

    name = "cpa"

    def __init__(self, model_dir: str, genes: list[str], device: str = "cuda", batch_size: int = 512,
                 module=None, pert_encoder: dict | None = None, keys=None, max_comb_len: int = 2,
                 trained_singles: list[str] | None = None):
        self.genes = list(genes)
        self.device = device
        self.batch_size = batch_size
        if module is None:                      # normal path; tests inject a stand-in
            import cpa
            from cpa._utils import CPA_REGISTRY_KEYS
            model = cpa.CPA.load(model_dir, use_gpu=device.startswith("cuda"))
            if list(model.adata.var_names) != self.genes:
                raise ValueError("CPA gene order differs from the analysis data")
            module, pert_encoder, keys = model.module, dict(cpa.CPA.pert_encoder), CPA_REGISTRY_KEYS
            max_comb_len = keys.MAX_COMB_LENGTH
            info = json.loads((Path(model_dir) / "rolloutprobe_split.json").read_text())
            trained_singles = [c for c in info["seen_conditions"] if c != CONTROL and "+" not in c]
        if getattr(module, "covars_encoder", {}):
            raise ValueError("this adapter assumes a CPA model without covariates")
        if getattr(module, "recon_loss", "gauss") != "gauss":
            raise ValueError("expected recon_loss='gauss' (log-normalized input, see train_cpa.py)")
        self.module = module.to(device).eval()
        self.pert_encoder = pert_encoder
        self.keys = keys
        self.max_comb_len = max_comb_len
        # genes with a trained embedding; others exist in pert_encoder (every registered
        # condition does) but their embedding never received a gradient
        self.trained = set(trained_singles if trained_singles is not None else
                           [g for g in pert_encoder if g not in ("<PAD>", CONTROL)])

    def can_perturb(self, genes) -> bool:
        return all(g in self.trained for g in genes)

    def _perts(self, pert: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Index and dose rows exactly as setup_anndata builds them."""
        names = list(pert) if pert else [CONTROL]
        if len(names) > self.max_comb_len:
            raise ValueError(f"CPA was set up for at most {self.max_comb_len} perturbations")
        missing = [g for g in names if g not in self.pert_encoder]
        if missing:
            raise KeyError(f"CPA has no embedding for {missing}")
        idx = sorted(self.pert_encoder[g] for g in names)      # setup_anndata order: vocab order
        pad = self.max_comb_len - len(idx)
        return (np.array(idx + [self.keys.PADDING_IDX] * pad, dtype=np.int64),
                np.array([1.0] * len(idx) + [0.0] * pad, dtype=np.float32))

    def _tensors(self, x: "torch.Tensor", pert: tuple[str, ...]) -> dict:
        idx, dose = self._perts(pert)
        n = x.shape[0]
        k = self.keys
        return {k.X_KEY: x,
                k.PERTURBATION_KEY: torch.zeros(n, 1, dtype=torch.long, device=x.device),  # label code; only mixup reads it
                k.PERTURBATIONS: torch.from_numpy(np.tile(idx, (n, 1))).to(x.device),
                k.PERTURBATIONS_DOSAGES: torch.from_numpy(np.tile(dose, (n, 1))).to(x.device)}

    @torch.no_grad()
    def _run(self, X, pert, key: str) -> np.ndarray:
        X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
        if X.shape[1] != len(self.genes):
            raise ValueError(f"X has {X.shape[1]} genes, model expects {len(self.genes)}")
        outs = []
        for i in range(0, len(X), self.batch_size):
            x = torch.from_numpy(X[i:i + self.batch_size]).to(self.device)
            out = self.module.get_expression(self._tensors(x, pert), n_samples=1)
            outs.append(out[key].float().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def predict(self, X: np.ndarray, pert: tuple[str, ...]) -> np.ndarray:
        """Encode each input cell, add the perturbation embedding, decode. () = control."""
        return self._run(X, pert, "px")

    def embed_means(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Part B (scripts/latent_additivity.py): mean basal latent of real cells,
        z_basal = encoder(x). It does not depend on the label, which is added after."""
        return {"basal": self._run(X, (), "z_basal").mean(axis=0)}
