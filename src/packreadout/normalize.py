"""Turn per-option log-likelihoods into a distribution over the option set.

Scores are combined as ``softmax(score / T)``. ``score`` is one of:

* ``raw``: summed token log-probs (lm-eval ``acc``). Biased toward short options.
* ``token``: raw / number of tokens. Tokenizer dependent.
* ``byte``: raw / number of UTF-8 bytes of the rendered option (space + text + terminator).
* ``char``: raw / number of characters of the bare option text. This is exactly what
  lm-evaluation-harness reports as ``acc_norm``.
* ``pmi``: raw - log P(option | null prompt). Removes the option's prior
  (Holtzman et al. 2021). Needs an extra scoring pass with a null prompt.

Shared token prefixes contribute equally to every option that shares them and
cancel in the softmax; only divergent suffixes matter.
"""

from __future__ import annotations

import numpy as np

NORMS = ("raw", "token", "byte", "char", "pmi")


def normalized_scores(
    sum_logprob: np.ndarray,
    n_tokens: np.ndarray,
    n_bytes: np.ndarray,
    norm: str,
    uncond_logprob: np.ndarray | None = None,
    n_chars: np.ndarray | None = None,
) -> np.ndarray:
    if norm == "raw":
        return sum_logprob
    if norm == "token":
        return sum_logprob / np.maximum(n_tokens, 1)
    if norm == "byte":
        return sum_logprob / np.maximum(n_bytes, 1)
    if norm == "char":
        if n_chars is None:
            raise ValueError("char normalization needs option character counts")
        return sum_logprob / np.maximum(n_chars, 1)
    if norm == "pmi":
        if uncond_logprob is None:
            raise ValueError("pmi normalization needs unconditional option log-probs")
        return sum_logprob - uncond_logprob
    raise ValueError(f"unknown norm {norm!r}; expected one of {NORMS}")


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def to_probs(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """softmax(scores / T) along the last axis."""
    return np.exp(log_softmax(np.asarray(scores, dtype=np.float64) / temperature))


def fit_temperature(
    scores_list: list[np.ndarray],
    gold: list[int],
    grid: np.ndarray | None = None,
) -> float:
    """1-D grid search for the temperature minimizing NLL of the gold option.

    Option sets may differ in size per example, hence a list of score vectors.
    """
    if not gold:
        return 1.0
    if grid is None:
        grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 121))
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        nll = 0.0
        for s, g in zip(scores_list, gold):
            nll -= log_softmax(np.asarray(s, dtype=np.float64) / t)[g]
        nll /= max(len(gold), 1)
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t
