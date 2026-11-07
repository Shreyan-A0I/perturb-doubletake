"""scGPT adapter plumbing, with a stand-in for TransformerGenerator.
Needs torch (scGPT environment only); skipped elsewhere. scgpt itself is not needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rolloutprobe import rollout                         # noqa: E402
from rolloutprobe.models.scgpt_adapter import ScGPTPredictor  # noqa: E402


class StandIn(torch.nn.Module):
    """Same call signature as TransformerGenerator.forward. Output = input + flag,
    so a perturbed gene goes up by exactly 1 and nothing else changes."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, src, values, input_pert_flags, src_key_padding_mask, CLS, CCE, MVC, ECS):
        assert src.shape == values.shape == input_pert_flags.shape == src_key_padding_mask.shape
        self.calls.append(values.shape[0])
        return {"mlm_output": values + input_pert_flags.float() + self.dummy}


GENES = ["G0", "G1", "G2", "G3", "G4"]
VOCAB = {"<pad>": 0, "G0": 10, "G1": 11, "G2": 12, "G3": 13}      # G4 unknown to scGPT


def make(batch_size=3):
    return ScGPTPredictor("unused", GENES, device="cpu", batch_size=batch_size, amp=False,
                          model=StandIn(), vocab=VOCAB)


def test_predict_shape_batches_and_flags():
    m = make()
    X = np.random.default_rng(0).random((7, 5)).astype(np.float32)
    out = m.predict(X, ("G1", "G4"))
    assert out.shape == X.shape
    assert m.model.calls == [3, 3, 1]                           # batched, no cell lost
    np.testing.assert_allclose(out - X, np.tile([0, 1, 0, 0, 1], (7, 1)), atol=1e-6)
    np.testing.assert_allclose(m.predict(X, ()), X, atol=1e-6)  # control: no flag
    assert m.unknown_genes == ["G4"]
    assert m.token_ids.tolist() == [10, 11, 12, 13, 0]


def test_can_perturb_and_unknown_gene():
    m = make()
    assert m.can_perturb(("G0", "G4")) and not m.can_perturb(("G0", "NOPE"))
    with pytest.raises(KeyError):
        m.predict(np.zeros((1, 5), np.float32), ("NOPE",))


def test_rollout_arms_through_adapter():
    m = make()
    X = np.zeros((4, 5), np.float32)
    np.testing.assert_allclose(rollout.rollout(m, X, "G0", "G2", "carried"), [1, 0, 1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(rollout.rollout(m, X, "G0", "G2", "restated"), [2, 0, 1, 0, 0], atol=1e-6)
    assert rollout.saturation_ratio(m, X, "G0") == pytest.approx(1.0)
