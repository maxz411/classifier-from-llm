"""Multiple-choice and yes/no tasks: options differ per example."""

from __future__ import annotations

import random
import re

from datasets import load_dataset

from packreadout.formatting import LM_EVAL_FORMAT, mcq_prompt
from packreadout.tasks.base import Example, register, take

YES_NO = ["yes", "no"]


def _split(split: str, train: str, test: str) -> str:
    return train if split == "train" else test


def _choices_example(r) -> Example:
    """The layout shared by the AI2 datasets: ``choices.text``, ``choices.label`` and ``answerKey``."""
    options = r["choices"]["text"]
    return Example(mcq_prompt(r["question"], options), options, r["choices"]["label"].index(r["answerKey"]))


# ---- training mixture


@register("sciq", "train")
def sciq(split, limit):
    ds = take(load_dataset("allenai/sciq", split=_split(split, "train", "test")), limit)
    out = []
    for i, row in enumerate(ds):
        options = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
        order = [0, 1, 2, 3]
        random.Random(i).shuffle(order)  # shuffle positions, so a repeated string cannot shadow the answer
        options = [options[j] for j in order]
        out.append(Example(mcq_prompt(row["question"], options), options, order.index(0)))
    return out


@register("openbookqa", "train")
def openbookqa(split, limit):
    ds = take(load_dataset("allenai/openbookqa", "main", split=_split(split, "train", "test")), limit)
    return [_choices_example({**r, "question": r["question_stem"]}) for r in ds]


@register("commonsense_qa", "train")
def commonsense_qa(split, limit):
    return [
        _choices_example(r)
        for r in take(load_dataset("tau/commonsense_qa", split=_split(split, "train", "validation")), limit)
    ]


@register("qasc", "train")
def qasc(split, limit):
    return [
        _choices_example(r)
        for r in take(load_dataset("allenai/qasc", split=_split(split, "train", "validation")), limit)
    ]


@register("arc_easy", "train")
def arc_easy(split, limit):
    return [
        _choices_example(r)
        for r in take(load_dataset("allenai/ai2_arc", "ARC-Easy", split=_split(split, "train", "test")), limit)
    ]


@register("hellaswag", "train")
def hellaswag(split, limit):
    ds = take(load_dataset("Rowan/hellaswag", split=_split(split, "train", "validation")), limit)
    out = []
    for r in ds:
        context = _hellaswag_clean(r["activity_label"] + ": " + r["ctx"])
        endings = [_hellaswag_clean(e) for e in r["endings"]]
        out.append(Example(mcq_prompt("How does the text continue?", endings, context), endings, int(r["label"])))
    return out


@register("winogrande", "train")
def winogrande(split, limit):
    ds = take(load_dataset("allenai/winogrande", "winogrande_xl", split=_split(split, "train", "validation")), limit)
    return [
        Example(
            mcq_prompt("What fills the blank (_)?", [r["option1"], r["option2"]], r["sentence"]),
            [r["option1"], r["option2"]],
            int(r["answer"]) - 1,
        )
        for r in ds
    ]


@register("boolq", "train")
def boolq(split, limit):
    ds = take(load_dataset("google/boolq", split=_split(split, "train", "validation")), limit)
    return [Example(mcq_prompt(r["question"] + "?", YES_NO, r["passage"]), YES_NO, 0 if r["answer"] else 1) for r in ds]


# ---- held out


@register("arc_challenge", "heldout")
def arc_challenge(split, limit):
    return [
        _choices_example(r)
        for r in take(load_dataset("allenai/ai2_arc", "ARC-Challenge", split=_split(split, "train", "test")), limit)
    ]


@register("mmlu", "heldout")
def mmlu(split, limit):
    ds = take(load_dataset("cais/mmlu", "all", split=_split(split, "auxiliary_train", "test")), limit)
    return [Example(mcq_prompt(r["question"], r["choices"]), list(r["choices"]), int(r["answer"])) for r in ds]


@register("strategyqa", "heldout")
def strategyqa(split, limit):
    ds = take(load_dataset("ChilleD/StrategyQA", split=_split(split, "train", "test")), limit)
    return [Example(mcq_prompt(r["question"], YES_NO), YES_NO, 0 if r["answer"] else 1) for r in ds]


# ---- sanity checks in the exact lm-evaluation-harness format (no terminator, no options listed)


@register("arc_easy_lmeval", "sanity", fmt=LM_EVAL_FORMAT)
def arc_easy_lmeval(split, limit):
    ds = take(load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"), limit, first=True)
    return [
        Example(
            f"Question: {r['question']}\nAnswer:", r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])
        )
        for r in ds
    ]


@register("hellaswag_lmeval", "sanity", fmt=LM_EVAL_FORMAT)
def hellaswag_lmeval(split, limit):
    ds = take(load_dataset("Rowan/hellaswag", split="validation"), limit, first=True)
    out = []
    for r in ds:
        query = _hellaswag_clean(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
        out.append(Example(query, [_hellaswag_clean(e) for e in r["endings"]], int(r["label"])))
    return out


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")
