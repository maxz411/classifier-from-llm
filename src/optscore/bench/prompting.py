"""Shared pieces of the prompting baselines: asking an instruction-tuned model to name the label,
and reading its reply. A reply that names no option is an invalid decision, which option
scoring cannot produce; the baselines count and keep every such reply."""

from __future__ import annotations

import re

import numpy as np
from sklearn.metrics import f1_score

INSTRUCTION = "Reply with one of the listed answers, exactly as written, and nothing else."


def user_message(prompt: str) -> str:
    """The scorer's prompt without its cue line, followed by the instruction to name one answer."""
    body = prompt.rstrip()
    if body.endswith("Answer:") or body.endswith("Label:"):
        body = body.rsplit("\n", 1)[0]
    return body + "\n" + INSTRUCTION


def normalize(text: str) -> str:
    """Lower-case, surrounding quotes and punctuation removed, whitespace collapsed."""
    return re.sub(r"\s+", " ", text.strip().strip("\"'`*").strip(".:;,!").strip().lower())


def match_option(text: str, options: list[str]) -> int | None:
    """The option a reply names: an exact match after normalization, else the longest option that
    appears in the reply as whole words, else the single option the reply is a part of, else None
    (an invalid reply)."""
    reply = normalize(text)
    normalized = [normalize(o) for o in options]
    if reply in normalized:
        return normalized.index(reply)
    contained = [i for i, o in enumerate(normalized) if o and re.search(rf"(?<!\w){re.escape(o)}(?!\w)", reply)]
    if contained:
        return max(contained, key=lambda i: len(normalized[i]))
    within = [i for i, o in enumerate(normalized) if reply and re.search(rf"(?<!\w){re.escape(reply)}(?!\w)", o)]
    return within[0] if len(within) == 1 else None


def score_replies(replies: list[str], examples) -> dict:
    """Accuracy, macro-F1 over option indices, the invalid-reply rate, and every invalid reply."""
    pred = np.array([-1 if (m := match_option(r, e.options)) is None else m for r, e in zip(replies, examples)])
    gold = np.array([e.gold for e in examples])
    n_options = max(len(e.options) for e in examples)
    invalid = [{"reply": r, "options": e.options, "gold": e.gold} for r, p, e in zip(replies, pred, examples) if p < 0]
    return {
        "acc": float((pred == gold).mean()),
        "macro_f1": float(f1_score(gold, pred, labels=list(range(n_options)), average="macro", zero_division=0)),
        "unmatched": float((pred < 0).mean()),
        "unmatched_empty": float(np.mean([not r.strip() for r in replies])),  # the model produced nothing
        "replies": replies,
        "unmatched_replies": invalid,
    }
