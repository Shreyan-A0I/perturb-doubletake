"""GEARS behind the predict() interface.

STATUS: checked line by line against GEARS master (gears/gears.py, model.py,
utils.py) but not executed, since the build environment had no GPU or data access.
Confirmed in the source: `GEARS.pert_list` holds the perturbation gene list;
the model returns (n_cells, n_genes) and is residual (it adds its predicted change
to the input x, model.py line ~206), so expression in = expression out; a batch of
all-control cells (pert_idx [-1]) is handled. Run scripts/roundtrip_check.py first.

Why this cannot just call GEARS.predict(): that method takes only a perturbation
list and samples its own CONTROL cells internally, so there is no way to hand it a
state to continue from. Re-entry therefore builds the PyG batch directly from the
expression we pass in, mirroring GEARS's own create_cell_graph_for_prediction().
"""
from __future__ import annotations

import numpy as np

from ..wrapper import register

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ImportError as e:  # pragma: no cover
    raise ImportError("GEARS adapter needs torch and torch_geometric installed.") from e


def load_pert_data(data_dir: str, split: str = "simulation", seed: int = 1,
                   split_dict_path: str | None = None, batch_size: int = 32):
    from gears import PertData
    pert_data = PertData(data_dir)
    pert_data.load(data_name="norman")   # downloads to <data_dir>/norman on first run
    if split == "custom":
        pert_data.prepare_split(split="custom", split_dict_path=split_dict_path)
    else:
        pert_data.prepare_split(split=split, seed=seed)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=128)
    return pert_data


@register("gears")
class GearsPredictor:
    name = "gears"

    def __init__(self, pert_data, model_dir: str, device: str = "cuda", batch_size: int = 256):
        from gears import GEARS
        self.gears = GEARS(pert_data, device=device)
        self.gears.load_pretrained(model_dir)
        self.model = self.gears.best_model.eval()
        self.device = device
        self.batch_size = batch_size
        self.genes = list(pert_data.adata.var["gene_name"].astype(str))
        # GEARS indexes perturbations into its GO-graph gene list, not the
        # expression gene list (confirmed: gears.py sets self.pert_list).
        self.pert_list = list(self.gears.pert_list)
        self._pert_pos = {g: i for i, g in enumerate(self.pert_list)}

    def can_perturb(self, genes) -> bool:
        return all(g in self._pert_pos for g in genes)

    def _pert_idx(self, pert: tuple[str, ...]) -> list[int]:
        if not pert:
            return [-1]                     # GEARS's encoding for control
        missing = [g for g in pert if g not in self._pert_pos]
        if missing:
            raise KeyError(f"GEARS cannot perturb {missing}: not in its GO gene graph.")
        return [self._pert_pos[g] for g in pert]

    @torch.no_grad()
    def predict(self, X: np.ndarray, pert: tuple[str, ...]) -> np.ndarray:
        X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
        pidx = self._pert_idx(pert)
        graphs = [Data(x=torch.from_numpy(row).reshape(-1, 1), pert_idx=pidx) for row in X]
        outs = []
        for batch in DataLoader(graphs, batch_size=self.batch_size, shuffle=False):
            batch = batch.to(self.device)
            out = self.model(batch)
            if isinstance(out, tuple):      # uncertainty mode returns (pred, logvar)
                out = out[0]
            # Model returns torch.stack of per-cell vectors: (n_cells_in_batch, n_genes).
            outs.append(out.reshape(batch.num_graphs, -1).cpu().numpy())
        return np.concatenate(outs, axis=0)
