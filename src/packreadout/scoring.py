"""Score a fixed set of options under a causal language model.

The quantity computed everywhere in this package is the teacher-forced log-likelihood
of each option as a continuation of the prompt::

    s_i = sum_t log P(opt_i[t] | prompt, opt_i[:t])

Because the options are known in advance, every option token can be scored in a
single forward pass; nothing is decoded autoregressively. Two implementations of
the same quantity live here:

* :func:`score_examples` and :func:`score_options` (the default): one sequence
  ``prompt ++ opt_1 ++ ... ++ opt_n`` with a block attention mask, see
  :mod:`packreadout.packing`. One forward of ``P + sum(L_i)`` tokens. A model trained
  with a scalar head is scored by the same functions (``readout="scalar"``).
* :func:`score_options_naive`: every ``prompt + option`` is its own full sequence.
  The reference implementation the packed path is tested against.

Log-probabilities are always computed in float32, and the LM head is only ever
applied to the rows that predict an option token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from packreadout.formatting import DEFAULT_FORMAT, OptionFormat
from packreadout.model import decoder_of, logprobs_of_targets
from packreadout.normalize import normalized_scores, to_probs
from packreadout.packing import option_token_logprobs, packed_forward, readout_logits


@dataclass
class ScoreResult:
    """Per-option scores for one prompt, plus what the normalizations need."""

    options: list[str]
    sum_logprob: np.ndarray  # (n,) s_i: the option's log-likelihood, or its head logit for a trained head
    n_tokens: np.ndarray  # (n,)
    n_bytes: np.ndarray  # (n,) UTF-8 bytes of the rendered option
    n_chars: np.ndarray  # (n,) characters of the bare option text (lm-eval acc_norm)
    token_logprobs: list[np.ndarray] = field(default_factory=list)
    uncond_logprob: np.ndarray | None = None  # (n,) log P(opt | cue alone), for PMI

    def scores(self, norm: str = "raw") -> np.ndarray:
        return normalized_scores(self.sum_logprob, self.n_tokens, self.n_bytes, norm, self.uncond_logprob, self.n_chars)

    def probs(self, norm: str = "raw", temperature: float = 1.0) -> np.ndarray:
        return to_probs(self.scores(norm), temperature)

    def argmax(self, norm: str = "raw") -> int:
        return int(np.argmax(self.scores(norm)))


# -------------------------------------------------------------------- encoding


def cue_of(prompt: str) -> str:
    """The prompt's last line (``"Label:"``, ``"Answer:"``): where the option continues it.
    Alone, it is the content-free prompt that PMI normalization conditions on."""
    return prompt[prompt.rfind("\n") + 1 :]


def tokenize_pieces(tok: PreTrainedTokenizerBase, pieces: list[str]) -> list[list[int]]:
    """Tokenize ``pieces[0] + pieces[1] + ...`` and split the ids at the piece boundaries.

    Tokenizing the whole concatenation is what the model would see if the text were
    generated; we then require that each prefix of the
    concatenation tokenizes to a prefix of the ids, so that later pieces can be
    scored as continuations of earlier ones.
    """
    ids_so_far: list[int] = []
    out: list[list[int]] = []
    text = ""
    for piece in pieces:
        text += piece
        ids = tok(text, add_special_tokens=True).input_ids
        if ids[: len(ids_so_far)] != ids_so_far:
            raise ValueError(f"tokenizer merged across a piece boundary before {piece!r}; adjust the prompt cue")
        out.append(ids[len(ids_so_far) :])
        ids_so_far = ids
    return out


def encode_options(
    tok: PreTrainedTokenizerBase,
    prompt: str,
    options: list[str],
    fmt: OptionFormat = DEFAULT_FORMAT,
) -> tuple[list[int], list[list[int]]]:
    """``(prompt_ids, [option_ids...])`` such that ``prompt_ids + option_ids[i]`` is the
    tokenization of ``prompt + fmt.render(options[i])``.

    Conceptually::

        prompt_ids = tok(prompt).input_ids
        option_ids = [tokenize_pieces(tok, [prompt, fmt.render(o)])[1] for o in options]

    Performance only: tokenizer merges can only happen at the junction, so each option is
    tokenized together with the prompt's last line (the cue) instead of the whole prompt,
    which would cost ``options x prompt length``. If the cue's tokens are not a suffix of
    the prompt's tokens, the conceptual version above is used.
    """
    prompt_ids = tok(prompt, add_special_tokens=True).input_ids
    cue = cue_of(prompt)
    cue_ids = tok(cue, add_special_tokens=False).input_ids
    if not cue_ids or prompt_ids[-len(cue_ids) :] != cue_ids:
        return prompt_ids, [tokenize_pieces(tok, [prompt, fmt.render(opt)])[1] for opt in options]
    opt_ids = []
    for opt in options:
        joint = tok(cue + fmt.render(opt), add_special_tokens=False).input_ids
        if joint[: len(cue_ids)] != cue_ids:
            raise ValueError(
                f"tokenizer merged across the cue/option boundary for option {opt!r}; adjust the prompt cue"
            )
        if len(joint) == len(cue_ids):
            raise ValueError(f"option {opt!r} tokenized to nothing")
        opt_ids.append(joint[len(cue_ids) :])
    return prompt_ids, opt_ids


def make_result(
    options: list[str], opt_ids: list[list[int]], token_lps: list[np.ndarray], fmt: OptionFormat
) -> ScoreResult:
    """Bundle per-option, per-token log-probs with the option lengths the normalizations use."""
    return ScoreResult(
        options=list(options),
        sum_logprob=np.array([float(lp.sum()) for lp in token_lps]),
        n_tokens=np.array([len(ids) for ids in opt_ids]),
        n_bytes=np.array([len(fmt.render(o).encode("utf-8")) for o in options]),
        n_chars=np.array([len(o) for o in options]),
        token_logprobs=[np.asarray(lp, dtype=np.float64) for lp in token_lps],
    )


# ------------------------------------------------------------- packed (default)


@torch.no_grad()
def score_examples(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: list[str],
    option_sets: list[list[str]],
    fmt: OptionFormat = DEFAULT_FORMAT,
    readout: str = "lm_head",
    head: torch.nn.Linear | None = None,
    uncond: bool = False,
    batch_size: int = 16,
) -> list[ScoreResult]:
    """Score many examples, ``batch_size`` packed sequences per forward.

    ``readout="lm_head"`` gives every option token's log-prob, so all normalizations
    apply; a trained ``scalar`` or ``both`` readout gives one logit per option, stored
    as a single "token". ``uncond`` also scores every option after the bare cue, for PMI.
    """
    encoded = [encode_options(tok, p, o, fmt) for p, o in zip(prompts, option_sets)]
    results: list[ScoreResult] = [None] * len(encoded)  # type: ignore[list-item]
    # Conceptually the batches are encoded[0:B], encoded[B:2B], ... Performance only: taking
    # the examples longest first puts sequences of similar length together, so less padding.
    order = sorted(range(len(encoded)), key=lambda i: -(len(encoded[i][0]) + sum(map(len, encoded[i][1]))))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = [encoded[i] for i in idx]
        hidden, packed = packed_forward(model, tok, batch)
        if readout == "lm_head":
            per_example = option_token_logprobs(model, hidden, packed, batch)
        else:
            per_example = [
                [s[None] for s in logits] for logits in readout_logits(model, hidden, packed, batch, readout, head)
            ]
        for i, per_option in zip(idx, per_example):
            results[i] = make_result(
                option_sets[i], encoded[i][1], [lp.float().cpu().numpy() for lp in per_option], fmt
            )
    if uncond:
        # Conceptually, for every example: uncond_logprob = score_options(model, tok, cue, options).sum_logprob
        # with cue the prompt's last line. Performance only: the cue is shared by every example of a
        # task, so each distinct option is scored once after it, 64 options per packed sequence.
        if readout != "lm_head":
            raise ValueError("PMI normalization needs the LM-head readout")
        by_cue: dict[str, set[str]] = {}
        for prompt, options in zip(prompts, option_sets):
            by_cue.setdefault(cue_of(prompt), set()).update(options)
        lookup = {}
        for cue, options in by_cue.items():
            options = sorted(options)
            for start in range(0, len(options), 64):
                chunk = options[start : start + 64]
                lookup.update(
                    ((cue, o), lp) for o, lp in zip(chunk, score_options(model, tok, cue, chunk, fmt).sum_logprob)
                )
        for prompt, r in zip(prompts, results):
            r.uncond_logprob = np.array([lookup[(cue_of(prompt), o)] for o in r.options])
    return results


def score_options(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    options: list[str],
    fmt: OptionFormat = DEFAULT_FORMAT,
    uncond: bool = False,
) -> ScoreResult:
    """Score the options of one prompt in one forward of ``P + sum(L_i)`` tokens."""
    return score_examples(model, tok, [prompt], [options], fmt, uncond=uncond)[0]


# ------------------------------------------------------------- naive reference


@torch.no_grad()
def score_sequences(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    pairs: list[tuple[list[int], list[int]]],
    batch_size: int = 16,
) -> list[np.ndarray]:
    """Per-token log-probs of ``cont_ids`` after ``prefix_ids`` for each pair, every pair
    being one ordinary sequence, so nothing is shared between pairs.

    Conceptually, for each pair::

        hidden = decoder(prefix_ids + cont_ids).last_hidden_state[0]
        rows = hidden[len(prefix_ids) - 1 : -1]         # position p predicts token p + 1
        logprobs = logprobs_of_targets(model, rows, cont_ids)

    Performance only: ``batch_size`` pairs go through the model together as one
    right-padded batch, longest first, which is what lm-evaluation-harness does too.
    """
    device = model.device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    decoder = decoder_of(model)
    results: list[np.ndarray] = [np.empty(0)] * len(pairs)
    order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        seqs = [pairs[i][0] + pairs[i][1] for i in idx]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(seqs), T), dtype=torch.long, device=device)
        for r, s in enumerate(seqs):
            ids[r, : len(s)] = torch.tensor(s, device=device)
            mask[r, : len(s)] = 1
        hidden = decoder(input_ids=ids, attention_mask=mask).last_hidden_state
        rows, cols, tgts = [], [], []
        for r, i in enumerate(idx):
            P, Lc = len(pairs[i][0]), len(pairs[i][1])
            rows.extend([r] * Lc)
            cols.extend(range(P - 1, P - 1 + Lc))
            tgts.extend(pairs[i][1])
        sel = hidden[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]
        lps = logprobs_of_targets(model, sel, torch.tensor(tgts, device=device)).cpu().numpy()
        cursor = 0
        for i in idx:
            Lc = len(pairs[i][1])
            results[i] = lps[cursor : cursor + Lc].astype(np.float64)
            cursor += Lc
    return results


@torch.no_grad()
def score_options_naive(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    options: list[str],
    fmt: OptionFormat = DEFAULT_FORMAT,
    uncond: bool = False,
    batch_size: int = 16,
) -> ScoreResult:
    """The reference: one full sequence ``prompt + option`` per option."""
    prompt_ids, opt_ids = encode_options(tok, prompt, options, fmt)
    token_lps = score_sequences(model, tok, [(prompt_ids, ids) for ids in opt_ids], batch_size)
    result = make_result(options, opt_ids, token_lps, fmt)
    if uncond:
        result.uncond_logprob = score_options_naive(
            model, tok, cue_of(prompt), options, fmt, batch_size=batch_size
        ).sum_logprob
    return result
