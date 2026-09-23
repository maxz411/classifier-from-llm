"""Loading a causal LM and reading its LM head.

Everything else in the package works on two pieces of a Hugging Face causal LM: the
transformer body (embeddings, blocks, final norm), which turns token ids into hidden
states, and the LM head, which turns a hidden state into next-token logits.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    name: str = "Qwen/Qwen3-0.6B-Base",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM in eval mode, with SDPA attention (the block mask needs it):
    bf16 on CUDA, fp32 elsewhere. ``device="auto"`` shards the layers over all GPUs."""
    sharded = device == "auto"  # spread the layers over every GPU, for models too big for one
    if sharded:
        device = torch.device("cuda")
    elif device is None:
        device = pick_device()
    else:
        device = torch.device(device)
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation="sdpa", device_map="auto" if sharded else None
    )
    if not sharded:
        model.to(device)
    model.eval()
    return model, tok


def unwrap(model):
    """The underlying causal LM, whether or not a peft adapter wraps it."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def decoder_of(model):
    """The transformer body (embeddings + blocks + final norm), without the LM head."""
    base = unwrap(model)
    return getattr(base, base.base_model_prefix)


def logprobs_of_targets(
    model: PreTrainedModel, hidden: torch.Tensor, targets: torch.Tensor, rows_per_chunk: int = 2048
) -> torch.Tensor:
    """``log P(targets[r] | hidden[r])`` for each hidden row: ``hidden`` is ``(R, H)``,
    ``targets`` is ``(R,)``, the result is ``(R,)`` in float32.

    Conceptually::

        logits = lm_head(hidden).float()                        # (R, vocabulary)
        return log_softmax(logits)[arange(R), targets]

    Performance only: the rows are processed ``rows_per_chunk`` at a time so that the
    full-vocabulary logits never exist for more than one chunk (2048 rows of a 150k
    vocabulary in fp32 is about 1.2 GB).
    """
    lm_head = unwrap(model).get_output_embeddings()
    out = []
    for start in range(0, hidden.shape[0], rows_per_chunk):
        logits = lm_head(hidden[start : start + rows_per_chunk]).float()
        tgt = targets[start : start + rows_per_chunk]
        out.append(logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1))
    return torch.cat(out)
