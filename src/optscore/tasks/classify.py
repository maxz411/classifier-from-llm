"""Text classification tasks, table driven: one spec per dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset

from optscore.formatting import classify_prompt
from optscore.tasks.base import Example, register, take

MAX_TEXT_CHARS = 1500  # long reviews are cut; enough for the label, keeps prompts short


@dataclass(frozen=True)
class ClassSpec:
    dataset: str
    instruction: str
    config: str | None = None
    text: str = "text"
    label: str = "label"
    labels: tuple[str, ...] | None = None  # None: read from the ClassLabel feature
    splits: tuple[str, str] = ("train", "test")  # (train split, test split)


SPECS: dict[str, tuple[str, ClassSpec]] = {  # name -> (role, spec)
    # ---- training mixture
    "ag_news": ("train", ClassSpec("fancyzhx/ag_news", "Classify the news article into one topic.")),
    "dbpedia": (
        "train",
        ClassSpec("fancyzhx/dbpedia_14", "Classify the Wikipedia article into one category.", text="content"),
    ),
    "yelp": ("train", ClassSpec("Yelp/yelp_review_full", "Rate the review from 1 star to 5 stars.")),
    "imdb": (
        "train",
        ClassSpec("stanfordnlp/imdb", "Classify the sentiment of the movie review.", labels=("negative", "positive")),
    ),
    "emotion": (
        "train",
        ClassSpec("dair-ai/emotion", "Classify the emotion expressed in the message.", config="split"),
    ),
    "tweet_sentiment": (
        "train",
        ClassSpec("cardiffnlp/tweet_eval", "Classify the sentiment of the tweet.", config="sentiment"),
    ),
    "tweet_offensive": ("train", ClassSpec("cardiffnlp/tweet_eval", "Is the tweet offensive?", config="offensive")),
    "tweet_irony": ("train", ClassSpec("cardiffnlp/tweet_eval", "Is the tweet ironic?", config="irony")),
    "clinc150": (
        "train",
        ClassSpec("clinc/clinc_oos", "Classify the user's request into one intent.", config="plus", label="intent"),
    ),
    "yahoo": (
        "train",
        ClassSpec(
            "community-datasets/yahoo_answers_topics",
            "Classify the question into one topic.",
            text="question_title",
            label="topic",
        ),
    ),
    # ---- held out: never trained on
    "banking77": (
        "heldout",
        ClassSpec("mteb/banking77", "Classify the customer's banking query into exactly one intent."),
    ),
    "massive": (
        "heldout",
        ClassSpec("mteb/amazon_massive_intent", "Classify the voice assistant request into one intent.", config="en"),
    ),
    "sst5": ("heldout", ClassSpec("SetFit/sst5", "Classify the sentiment of the movie review sentence.")),
    "tweet_emotion": (
        "heldout",
        ClassSpec("cardiffnlp/tweet_eval", "Classify the emotion expressed in the tweet.", config="emotion"),
    ),
    "tweet_hate": ("heldout", ClassSpec("cardiffnlp/tweet_eval", "Is the tweet hateful?", config="hate")),
    "rotten_tomatoes": (
        "heldout",
        ClassSpec(
            "cornell-movie-review-data/rotten_tomatoes",
            "Classify the sentiment of the movie review.",
            labels=("negative", "positive"),
        ),
    ),
    "sms_spam": (
        "heldout",
        ClassSpec(
            "ucirvine/sms_spam", "Is the SMS message spam?", config="plain_text", text="sms", splits=("train", "train")
        ),
    ),
    "counterfactual": (
        "heldout",
        ClassSpec(
            "mteb/amazon_counterfactual",
            "Does the review sentence describe a counterfactual (something that did not happen)?",
            config="en",
        ),
    ),
    "go_emotions": (
        "heldout",
        ClassSpec(
            "google-research-datasets/go_emotions",
            "Classify the emotion expressed in the comment.",
            config="simplified",
            label="labels",
        ),
    ),
}


def pretty(name: str) -> str:
    """'card_arrival' -> 'card arrival', 'EducationalInstitution' -> 'Educational Institution'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).replace("_", " ")


def label_names(ds, spec: ClassSpec) -> list[str]:
    """Label names in id order, from the spec, the ClassLabel feature, or the label values.
    Derived from the split being loaded; every dataset used here has the same ids in train and test."""
    feature = ds.features[spec.label]
    feature = getattr(feature, "feature", feature)  # go_emotions: a list of ClassLabel
    if spec.labels:
        names = list(spec.labels)
    elif hasattr(feature, "names"):
        names = list(feature.names)
    elif isinstance(ds[0][spec.label], str):  # string labels: ids are their sorted order
        names = sorted(set(ds[spec.label]))
    else:  # mteb datasets: integer label + label_text
        names = [t for _, t in sorted({(int(r[spec.label]), r["label_text"]) for r in ds})]
    return [pretty(n) for n in names]


def label_id(row, spec: ClassSpec, names: list[str]) -> int:
    value = row[spec.label]
    if isinstance(value, list):  # single-label rows of a multi-label dataset
        value = value[0]
    return names.index(pretty(value)) if isinstance(value, str) else int(value)


def load_classification(spec: ClassSpec, split: str, limit: int | None) -> list[Example]:
    ds = load_dataset(spec.dataset, spec.config, split=spec.splits[0] if split == "train" else spec.splits[1])
    names = label_names(ds, spec)
    if spec.label == "labels":  # multi-label source: keep single-label rows only
        ds = ds.filter(lambda r: len(r["labels"]) == 1)
    ds = take(ds, limit)
    return [
        Example(
            classify_prompt(row[spec.text][:MAX_TEXT_CHARS], names, spec.instruction), names, label_id(row, spec, names)
        )
        for row in ds
    ]


for _name, (_role, _spec) in SPECS.items():
    register(_name, _role)(lambda split, limit, spec=_spec: load_classification(spec, split, limit))
