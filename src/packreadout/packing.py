"""Packed scoring: all options of one prompt in a single forward pass.

The packed sequence is ``prompt ++ opt_1 ++ opt_2 ++ ... ++ opt_n``. Two things make
the options independent of each other, so the result equals scoring each option on
its own:

* a block attention mask: a token may attend to the prompt and to earlier tokens of
  its own option, never to another option;
* position ids restart at ``len(prompt)`` for every option.

Cost is one forward over ``P + sum(L_i)`` tokens. The prompt is computed once, no
KV cache is copied, and options are not padded to a common length.

:func:`packed_forward` runs the transformer body and returns hidden states. The two
readouts, :func:`option_token_logprobs` (LM head) and :func:`option_last_hidden`
(scalar head), read the option rows of those hidden states, and
:func:`readout_logits` turns either into one logit per option. Gradients flow
through all of it, so training and evaluation use the same functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from packreadout.model import decoder_of, logprobs_of_targets

Item = tuple[list[int], list[list[int]]]  # (prompt_ids, option_ids per option)


@dataclass
class PackedSequence:
    ids: list[int]
    position_ids: list[int]
    segment_ids: list[int]  # 0 on the prompt, i+1 on option i
    read_rows: list[int]  # row whose next-token distribution scores targets[j]
    targets: list[int]  # option tokens, option by option
    last_rows: list[int]  # row of the last token of each option


def build_packed(prompt_ids: list[int], opt_ids: list[list[int]]) -> PackedSequence:
    P = len(prompt_ids)
    ids, pos, seg = list(prompt_ids), list(range(P)), [0] * P
    read_rows, last_rows = [], []
    for i, opt in enumerate(opt_ids):
        if not opt:
            raise ValueError("an option tokenized to nothing; every option needs at least one token")
        start = len(ids)
        ids.extend(opt)
        pos.extend(range(P, P + len(opt)))
        seg.extend([i + 1] * len(opt))
        # option token j is predicted by row start+j-1; the first one by the prompt's last row
        read_rows.extend(P - 1 if j == 0 else start + j - 1 for j in range(len(opt)))
        last_rows.append(start + len(opt) - 1)
    targets = [t for opt in opt_ids for t in opt]
    return PackedSequence(ids, pos, seg, read_rows, targets, last_rows)


def block_attention_mask(segment_ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Boolean ``(B, 1, T, T)`` mask, True where the query may attend to the key:
    causal, and the key is in the prompt or in the query's own option.

    Padded query rows attend to key 0 so that no row is fully masked (SDPA would
    return NaN for it)."""
    B, T = segment_ids.shape
    causal = torch.tril(torch.ones((T, T), dtype=torch.bool, device=segment_ids.device))
    q, k = segment_ids[:, :, None], segment_ids[:, None, :]
    allowed = causal[None] & ((k == 0) | (k == q)) & valid[:, None, :]
    first_key = (torch.arange(T, device=segment_ids.device) == 0)[None, None, :]
    return (allowed | ((~valid)[:, :, None] & first_key))[:, None]


def packed_forward(
    model: PreTrainedModel, tok: PreTrainedTokenizerBase, items: list[Item]
) -> tuple[torch.Tensor, list[PackedSequence]]:
    """One forward over the packed, right-padded batch. Returns ``(hidden (B, T, H), packed)``."""
    device = model.device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    packed = [build_packed(*item) for item in items]
    B, T = len(items), max(len(p.ids) for p in packed)
    # performance only: the padded length is conceptually just max(len(p.ids)); rounding it up
    # to a multiple of 8 lets PyTorch pick its memory-efficient attention kernel for the mask
    T = (T + 7) // 8 * 8
    ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    pos = torch.zeros((B, T), dtype=torch.long, device=device)
    seg = torch.full((B, T), -1, dtype=torch.long, device=device)
    valid = torch.zeros((B, T), dtype=torch.bool, device=device)
    for r, p in enumerate(packed):
        n = len(p.ids)
        ids[r, :n] = torch.tensor(p.ids, device=device)
        pos[r, :n] = torch.tensor(p.position_ids, device=device)
        seg[r, :n] = torch.tensor(p.segment_ids, device=device)
        valid[r, :n] = True
    mask = block_attention_mask(seg, valid)
    hidden = decoder_of(model)(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
    return hidden, packed


def option_token_logprobs(
    model: PreTrainedModel, hidden: torch.Tensor, packed: list[PackedSequence], items: list[Item]
) -> list[list[torch.Tensor]]:
    """LM-head readout: per item, per option, the log-prob of each of its tokens,
    ``log P(opt_i[t] | prompt, opt_i[:t])``, read at the row that predicts the token."""
    device = hidden.device
    rows_b = [r for r, p in enumerate(packed) for _ in p.read_rows]
    rows_t = [t for p in packed for t in p.read_rows]
    targets = [t for p in packed for t in p.targets]
    selected = hidden[torch.tensor(rows_b, device=device), torch.tensor(rows_t, device=device)]
    lps = logprobs_of_targets(model, selected, torch.tensor(targets, device=device))
    out, cursor = [], 0
    for _, opt_ids in items:
        per_option = []
        for opt in opt_ids:
            per_option.append(lps[cursor : cursor + len(opt)])
            cursor += len(opt)
        out.append(per_option)
    return out


def option_last_hidden(hidden: torch.Tensor, packed: list[PackedSequence]) -> list[torch.Tensor]:
    """Scalar-head readout input: per item, the hidden state at each option's last token, ``(n_i, H)``."""
    return [hidden[r, torch.tensor(p.last_rows, device=hidden.device)] for r, p in enumerate(packed)]


def readout_logits(
    model: PreTrainedModel,
    hidden: torch.Tensor,
    packed: list[PackedSequence],
    items: list[Item],
    readout: str,
    head: torch.nn.Linear | None = None,
) -> list[torch.Tensor]:
    """One logit per option, per item, from the packed forward (differentiable).

    ``lm_head``: the sum of the option's token log-probs, ``s_i = log P(opt_i | prompt)``;
    ``scalar``: ``head(h_i)`` with ``h_i`` the hidden state at the option's last token;
    ``both``: their sum. The same function serves training and evaluation.
    """
    if readout == "lm_head":
        return [
            torch.stack([lp.sum() for lp in per_option])
            for per_option in option_token_logprobs(model, hidden, packed, items)
        ]
    scalar = [  # the head is fp32, and on the first GPU when the model is sharded over several
        head(h.to(head.weight.device, head.weight.dtype)).squeeze(-1) for h in option_last_hidden(hidden, packed)
    ]
    if readout == "scalar":
        return scalar
    if readout == "both":
        lm = readout_logits(model, hidden, packed, items, "lm_head")
        return [s + a.to(s.device, s.dtype) for s, a in zip(lm, scalar)]
    raise ValueError(f"unknown readout {readout!r}; expected lm_head, scalar or both")
