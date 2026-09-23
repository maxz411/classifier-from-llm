"""Latency baseline: prefill the prompt once, replicate its KV cache across the
options, score the padded option batch. Same log-likelihoods as the packed path, but
the cache copy is ``O(n * P)`` and dominates once there are many options."""

from __future__ import annotations

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from packreadout.formatting import DEFAULT_FORMAT, OptionFormat
from packreadout.model import decoder_of, logprobs_of_targets
from packreadout.scoring import ScoreResult, encode_options, make_result


@torch.no_grad()
def score_options_cached(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    options: list[str],
    fmt: OptionFormat = DEFAULT_FORMAT,
) -> ScoreResult:
    """Prefill the prompt once, replicate its KV cache, score the padded option batch."""
    prompt_ids, opt_ids = encode_options(tok, prompt, options, fmt)
    device = model.device
    n, P, L = len(options), len(prompt_ids), max(len(ids) for ids in opt_ids)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    decoder = decoder_of(model)

    # prompt prefill: the last hidden row predicts the first token of every option
    out = decoder(input_ids=torch.tensor([prompt_ids], device=device), use_cache=True)
    last_hidden = out.last_hidden_state[0, -1]
    cache = out.past_key_values
    cache.batch_repeat_interleave(n)  # the O(n * P) copy that the packed path avoids

    # option batch, right padded, positions continuing from P
    opt_tensor = torch.full((n, L), pad_id, dtype=torch.long, device=device)
    opt_mask = torch.zeros((n, L), dtype=torch.long, device=device)
    for i, ids in enumerate(opt_ids):
        opt_tensor[i, : len(ids)] = torch.tensor(ids, device=device)
        opt_mask[i, : len(ids)] = 1
    attention_mask = torch.cat([torch.ones((n, P), dtype=torch.long, device=device), opt_mask], dim=1)
    position_ids = torch.arange(P, P + L, device=device).unsqueeze(0).expand(n, L)
    hidden = decoder(
        input_ids=opt_tensor,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    ).last_hidden_state  # (n, L, H); row t predicts option token t+1

    # LM head only on rows that predict a real option token
    rows = [last_hidden.unsqueeze(0).expand(n, -1)]
    targets = [opt_tensor[:, 0]]
    if L > 1:
        real = opt_mask[:, 1:].bool()
        rows.append(hidden[:, :-1][real])
        targets.append(opt_tensor[:, 1:][real])
    lps = logprobs_of_targets(model, torch.cat(rows), torch.cat(targets)).cpu().numpy()
    first, rest = lps[:n], lps[n:]
    token_lps, cursor = [], 0
    for i, ids in enumerate(opt_ids):
        k = len(ids) - 1
        token_lps.append(np.concatenate([[first[i]], rest[cursor : cursor + k]]))
        cursor += k
    return make_result(options, opt_ids, token_lps, fmt)
