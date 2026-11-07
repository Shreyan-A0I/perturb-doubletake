"""scGPT behind the predict() interface, plus the helpers train_scgpt.py shares.

Model: scgpt.model.TransformerGenerator, the perturbation model from scGPT's
Tutorial_Perturbation (https://github.com/bowang-lab/scGPT). Each gene is one
token: gene-id embedding + embedding of its input expression value + embedding of
a perturbation flag (1 = this gene is perturbed). The decoder predicts the
post-perturbation expression of every input gene. Output is absolute expression
in the same log-normalized space as the input.

Why this cannot just call model.pred_perturb(): that method reads a GEARS batch,
whose input cells are control cells GEARS samples itself, so there is no way to
hand it a state to continue from. predict() below builds the same forward pass
from the expression we pass in.

STATUS: written against scGPT's TransformerGenerator API (scgpt 0.2.x) but not
executed where it was written (no GPU, no torch). The torch-only test in
tests/test_scgpt_adapter.py checks the plumbing with a stand-in model. If a call
signature differs in your installed scgpt, scgpt/model/generation_model.py is the
reference. Run scripts/roundtrip_check.py before anything else.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..wrapper import register

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("scGPT adapter needs torch installed (see docs/TASKS.md, scGPT environment).") from e

PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
PAD_VALUE = 0
PERT_PAD_ID = 2          # flag values: 0 not perturbed, 1 perturbed, 2 padding (unused: no padding here)
LOAD_PREFIXES = ("encoder", "value_encoder", "transformer")   # pretrained parts, as in the tutorial


def load_vocab(model_dir: str | Path):
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    vocab = GeneVocab.from_file(Path(model_dir) / "vocab.json")
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[PAD_TOKEN])
    return vocab


def gene_token_ids(genes: list[str], vocab) -> tuple[np.ndarray, list[str]]:
    """Vocab id per expression gene. Genes scGPT does not know get the pad token,
    as in the tutorial; they still carry their value and flag."""
    ids = np.array([vocab[g] if g in vocab else vocab[PAD_TOKEN] for g in genes], dtype=np.int64)
    missing = [g for g in genes if g not in vocab]
    return ids, missing


def read_config(model_dir: str | Path) -> dict:
    cfg = json.loads((Path(model_dir) / "args.json").read_text())
    return {"embsize": cfg.get("embsize", 512), "nheads": cfg.get("nheads", 8),
            "d_hid": cfg.get("d_hid", 512), "nlayers": cfg.get("nlayers", 12),
            "n_layers_cls": cfg.get("n_layers_cls", 3)}


def build_model(cfg: dict, vocab, dropout: float = 0.0, use_fast_transformer: bool = False):
    from scgpt.model import TransformerGenerator
    return TransformerGenerator(
        len(vocab), cfg["embsize"], cfg["nheads"], cfg["d_hid"], cfg["nlayers"],
        nlayers_cls=cfg["n_layers_cls"], n_cls=1, vocab=vocab, dropout=dropout,
        pad_token=PAD_TOKEN, pad_value=PAD_VALUE, pert_pad_id=PERT_PAD_ID,
        use_fast_transformer=use_fast_transformer,
    )


def load_weights(model, weights: str | Path, prefixes: tuple[str, ...] | None = None) -> list[str]:
    """Load a state dict. With `prefixes`, only keys starting with one of them
    (pretrained checkpoint -> new perturbation model). Returns the loaded keys.

    The whole-human checkpoint was trained with flash-attn, whose attention weights
    are named "Wqkv.*"; PyTorch's own layers call them "in_proj_*". scGPT's
    utils.load_pretrained renames them the same way. Without this, every
    transformer layer would be silently skipped, so we also fail loudly if any
    requested prefix loads nothing."""
    state = torch.load(weights, map_location="cpu")
    own = model.state_dict()
    if not any("Wqkv." in k for k in own):
        state = {k.replace("Wqkv.", "in_proj_"): v for k, v in state.items()}
    wanted = {k: v for k, v in state.items() if prefixes is None or k.startswith(prefixes)}
    keep = {k: v for k, v in wanted.items() if k in own and v.shape == own[k].shape}
    skipped = sorted(set(wanted) - set(keep))
    if skipped:
        print(f"load_weights: skipped {len(skipped)} tensors (name or shape mismatch), e.g. {skipped[:5]}")
    for pre in prefixes or ():
        if not any(k.startswith(pre) for k in keep):
            raise RuntimeError(f"no pretrained tensors loaded for '{pre}'; check checkpoint and scgpt version")
    own.update(keep)
    model.load_state_dict(own)
    return sorted(keep)


def pert_flags(genes: list[str], pert: tuple[str, ...]) -> np.ndarray:
    pos = {g: i for i, g in enumerate(genes)}
    missing = [g for g in pert if g not in pos]
    if missing:
        raise KeyError(f"scGPT cannot flag {missing}: not among the expression genes.")
    flags = np.zeros(len(genes), dtype=np.int64)
    flags[[pos[g] for g in pert]] = 1
    return flags


def forward(model, token_ids, values, flags, amp: bool):
    """One forward pass. token_ids (L,), values (B, L) float, flags (B, L) long;
    all on the model's device. Returns (B, L) predicted expression."""
    b = values.shape[0]
    src = token_ids.unsqueeze(0).expand(b, -1)
    mask = torch.zeros_like(values, dtype=torch.bool)
    with torch.autocast(device_type=values.device.type, dtype=torch.float16,
                        enabled=amp and values.device.type == "cuda"):
        out = model(src, values, flags, src_key_padding_mask=mask,
                    CLS=False, CCE=False, MVC=False, ECS=False)
    return out["mlm_output"].float()


@register("scgpt")
class ScGPTPredictor:
    """Loads a model dir written by scripts/train_scgpt.py: best_model.pt (full
    fine-tuned state), args.json and vocab.json (copied from the pretrained
    checkpoint), rolloutprobe_split.json."""

    name = "scgpt"

    def __init__(self, model_dir: str, genes: list[str], device: str = "cuda",
                 batch_size: int = 8, amp: bool = True, model=None, vocab=None):
        self.genes = list(genes)
        self.device = device
        self.batch_size = batch_size
        self.amp = amp
        if model is None:                     # normal path; tests inject a stand-in model
            vocab = load_vocab(model_dir)
            model = build_model(read_config(model_dir), vocab)
            load_weights(model, Path(model_dir) / "best_model.pt")
        self.model = model.to(device).eval()
        ids, self.unknown_genes = gene_token_ids(self.genes, vocab)
        self.token_ids = torch.from_numpy(ids).to(device)
        self._gene_set = set(self.genes)

    def can_perturb(self, genes) -> bool:
        return all(g in self._gene_set for g in genes)

    @torch.no_grad()
    def predict(self, X: np.ndarray, pert: tuple[str, ...]) -> np.ndarray:
        X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
        if X.shape[1] != len(self.genes):
            raise ValueError(f"X has {X.shape[1]} genes, model expects {len(self.genes)}")
        flags = torch.from_numpy(pert_flags(self.genes, pert)).to(self.device)
        outs = []
        for i in range(0, len(X), self.batch_size):        # all genes: no subsampling at prediction
            v = torch.from_numpy(X[i:i + self.batch_size]).to(self.device)
            f = flags.unsqueeze(0).expand(v.shape[0], -1)
            outs.append(forward(self.model, self.token_ids, v, f, self.amp).cpu().numpy())
        return np.concatenate(outs, axis=0)
