"""optscore: score a fixed option set under a causal LM in one pass, then normalize."""

from optscore import (  # noqa: F401  (the method's modules; optscore.gptoss has the same four)
    model,
    packing,
    scoring,
    train,
)
from optscore.model import load_model
from optscore.normalize import NORMS, to_probs
from optscore.primitives import choice, score, true_false
from optscore.scoring import ScoreResult, score_examples, score_options, score_options_naive

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
