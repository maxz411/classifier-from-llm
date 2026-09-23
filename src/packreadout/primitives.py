"""The three decision primitives, each a thin wrapper over :func:`score_options`.

* :func:`choice`     - probability of each option
* :func:`score`      - expected position on an ordered scale of levels
* :func:`true_false` - probability that the answer is yes

``state`` is the text being judged, ``question`` is what to decide about it.
"""

from __future__ import annotations

import numpy as np

from packreadout.scoring import score_options


def choice(model, tok, state: str, question: str, options: list[str], temperature: float = 1.0) -> dict[str, float]:
    prompt = f"{state.strip()}\n\nQuestion: {question.strip()}\nOptions: {', '.join(options)}\nAnswer:"
    probs = score_options(model, tok, prompt, options).probs(temperature=temperature)
    return dict(zip(options, probs.tolist()))


def score(
    model, tok, state: str, question: str, levels: list[str], temperature: float = 1.0
) -> tuple[float, list[float]]:
    """Levels are ordered from low to high. Returns ``(expected level index, probability per level)``.

    The model answers with the level *number*, so every option is one short token
    and length bias does not enter."""
    legend = "\n".join(f"{i}: {level}" for i, level in enumerate(levels))
    prompt = f"{state.strip()}\n\nQuestion: {question.strip()}\nLevels:\n{legend}\nAnswer (level number):"
    probs = score_options(model, tok, prompt, [str(i) for i in range(len(levels))]).probs(temperature=temperature)
    return float(np.dot(np.arange(len(levels)), probs)), probs.tolist()


def true_false(model, tok, state: str, question: str, temperature: float = 1.0) -> float:
    prompt = f"{state.strip()}\n\nQuestion: {question.strip()}\nAnswer (yes or no):"
    probs = score_options(model, tok, prompt, ["yes", "no"]).probs(temperature=temperature)
    return float(probs[0])
