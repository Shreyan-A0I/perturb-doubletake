"""CPA adapter plumbing, with a stand-in for CPAModule. Needs torch (cpa env, or any
env with torch); skipped elsewhere. cpa-tools itself is not needed."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rolloutprobe import rollout                          # noqa: E402
from rolloutprobe.models.cpa_adapter import CPAPredictor  # noqa: E402

KEYS = SimpleNamespace(X_KEY="X", PERTURBATION_KEY="condition", PERTURBATIONS="perts",
                       PERTURBATIONS_DOSAGES="perts_doses", PADDING_IDX=0)
ENC = {"<PAD>": 0, "ctrl": 1, "A": 2, "B": 3, "V": 4}     # V: held-out single, untrained
G = 4


class StandIn(torch.nn.Module):
    """get_expression(tensors, n_samples) like CPAModule: px = x + sum of dose-scaled
    embeddings (padding masked), z_basal = first two genes of x. Records the rows it saw."""

    covars_encoder: dict = {}
    recon_loss = "gauss"

    def __init__(self):
        super().__init__()
        emb = torch.zeros(len(ENC), G)
        emb[ENC["ctrl"]] = 0.0
        emb[ENC["A"], 0] = 1.0
        emb[ENC["B"], 1] = 10.0
        self.emb = torch.nn.Parameter(emb, requires_grad=False)
        self.seen = []

    def get_expression(self, tensors, n_samples=1):
        x, p, d = tensors[KEYS.X_KEY], tensors[KEYS.PERTURBATIONS], tensors[KEYS.PERTURBATIONS_DOSAGES]
        assert tensors[KEYS.PERTURBATION_KEY].shape == (x.shape[0], 1)
        self.seen.append((x.shape[0], p[0].tolist(), d[0].tolist()))
        z = (self.emb[p] * (d * (p != KEYS.PADDING_IDX)).unsqueeze(-1)).sum(1)
        return {"px": x + z, "z_basal": x[:, :2]}


def make(batch_size=3):
    return CPAPredictor("unused", [f"g{i}" for i in range(G)], device="cpu", batch_size=batch_size,
                        module=StandIn(), pert_encoder=ENC, keys=KEYS, max_comb_len=2,
                        trained_singles=["A", "B"])


def test_predict_encodes_like_setup_anndata():
    m = make()
    X = np.random.default_rng(0).random((7, G)).astype(np.float32)
    np.testing.assert_allclose(m.predict(X, ("B", "A")) - X, np.tile([1, 10, 0, 0], (7, 1)), atol=1e-6)
    assert [s[0] for s in m.module.seen] == [3, 3, 1]                 # batched, no cell lost
    assert m.module.seen[0][1:] == ([2, 3], [1.0, 1.0])               # vocab order, both doses 1
    m.predict(X, ())
    assert m.module.seen[-1][1:] == ([1, 0], [1.0, 0.0])              # control = "ctrl" + padding
    m.predict(X, ("A",))
    assert m.module.seen[-1][1:] == ([2, 0], [1.0, 0.0])


def test_can_perturb_and_errors():
    m = make()
    assert m.can_perturb(("A", "B")) and not m.can_perturb(("A", "V"))   # V has no trained embedding
    with pytest.raises(KeyError):
        m.predict(np.zeros((1, G), np.float32), ("NOPE",))
    with pytest.raises(ValueError):
        m.predict(np.zeros((1, G), np.float32), ("A", "B", "V"))


def test_embed_means_and_rollout():
    m = make()
    X = np.random.default_rng(1).random((5, G)).astype(np.float32)
    np.testing.assert_allclose(m.embed_means(X)["basal"], X[:, :2].mean(0), atol=1e-6)
    np.testing.assert_allclose(rollout.rollout(m, np.zeros((2, G), np.float32), "A", "B"), [1, 10, 0, 0], atol=1e-6)
