"""Shared constants and helpers for the ESR-on-values project."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

MODEL_ID = os.environ.get("ESR_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / MODEL_ID.split("/")[-1]  # per-model outputs
OUT.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def chat_prompt(tok, user: str) -> str:
    """User turn rendered with the chat template, ending at the assistant header."""
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def device() -> str:
    """Device map: shard across GPUs when there are several (70B), else the single available device."""
    import torch
    if torch.cuda.is_available():
        return "auto" if torch.cuda.device_count() > 1 else "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device()).eval()
    return tok, model


@contextmanager
def steer(model, layer: int, vec: torch.Tensor | None):
    """Add `vec` to the residual stream at hidden_states[layer] (= output of block layer-1), all positions.

    Args:
        model: HF causal LM with model.model.layers.
        layer: index into output_hidden_states (1..L); 0 (embeddings) is not supported.
        vec: [D] vector already scaled; None or zero means no-op.
    """
    if vec is None or layer == 0:
        yield
        return
    block = model.model.layers[layer - 1]
    v = vec.to(next(block.parameters()).device, model.dtype)  # the block's own GPU when sharded

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + v,) + tuple(out[1:])
        return out + v

    h = block.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()
