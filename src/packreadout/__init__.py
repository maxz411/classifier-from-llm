"""packreadout: score a fixed option set under a causal LM in one pass, then normalize."""

from packreadout import (  # noqa: F401  (the method's modules; packreadout.gptoss has the same four)
    model,
    packing,
    scoring,
    train,
)
from packreadout.model import load_model
from packreadout.normalize import NORMS, to_probs
from packreadout.primitives import choice, score, true_false
from packreadout.scoring import ScoreResult, score_examples, score_options, score_options_naive

__all__ = [
    "NORMS",
    "ScoreResult",
    "choice",
    "load_model",
    "score",
    "score_examples",
    "score_options",
    "score_options_naive",
    "to_probs",
    "true_false",
]
