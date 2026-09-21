"""Plug option scoring into the BTZSC zero-shot classification benchmark (ICLR 2026).

BTZSC hands a model the texts and the verbalized label descriptions of a dataset
(e.g. "This example news text is about business news") and expects a score per label.
We list the descriptions in the prompt and score each one as an option.
"""

from __future__ import annotations

import numpy as np
import torch
from btzsc import BaseModel
from datasets import load_dataset

from optscore.formatting import classify_prompt
from optscore.scoring import score_examples

INSTRUCTION = "Pick the label that describes the text."


class OptionScoringModel(BaseModel):
    """The benchmark's model interface, over a base model or a trained adapter."""

    model_type = "llm"

    def __init__(
        self, model, tok, name: str, readout: str = "lm_head", head: torch.nn.Linear | None = None, score=score_examples
    ):
        self.model, self.tok, self.model_name, self.readout, self.head, self.score = (
            model,
            tok,
            name,
            readout,
            head,
            score,
        )

    def predict_scores(self, texts: list[str], labels: list[str], batch_size: int = 8) -> np.ndarray:
        prompts = [classify_prompt(t, labels, INSTRUCTION) for t in texts]
        results = self.score(
            self.model,
            self.tok,
            prompts,
            [labels] * len(texts),
            readout=self.readout,
            head=self.head,
            batch_size=batch_size,
        )
        return np.stack([r.probs() for r in results])

    def predict(self, texts: list[str], labels: list[str], batch_size: int = 8) -> np.ndarray:
        return self.predict_scores(texts, labels, batch_size).argmax(axis=1)


def n_classes_of(labels: list[int]) -> int:
    """The benchmark stores one row per (text, label) pair, grouped by example, with a binary
    entailment label; the number of classes is where that binary pattern first repeats."""
    for i in range(1, len(labels)):
        if labels[i] == labels[0]:
            return i
    return len(labels)


def load_examples(name: str, max_examples: int | None = None, seed: int = 0) -> tuple[list[str], list[str], np.ndarray]:
    """Texts, label descriptions and gold indices of one BTZSC dataset, for a seeded random
    subset of examples. We pick the examples first and read only their rows: the benchmark's
    own loader reads whole columns element by element, which takes hours on its million-row
    datasets, and its ``max_samples`` keeps the first rows, which are grouped by class."""
    from btzsc.data import REPO_ID

    rows = load_dataset(REPO_ID, name=name, split="test")
    head = rows[:1000]["labels"]
    n_classes = n_classes_of(head)
    if n_classes >= len(head):
        raise ValueError(f"{name}: could not find the label pattern in the first rows")
    n_examples = len(rows) // n_classes
    keep = np.arange(n_examples)
    if max_examples and n_examples > max_examples:
        keep = np.sort(np.random.default_rng(seed).choice(n_examples, max_examples, replace=False))
    sub = rows.select([i * n_classes + j for i in keep for j in range(n_classes)]).to_dict()
    texts = sub["text"][::n_classes]
    labels = sub["hypothesis"][:n_classes]
    gold = np.asarray(sub["labels"]).reshape(-1, n_classes).argmax(axis=1)
    return texts, labels, gold


def evaluate_btzsc(
    adapter: OptionScoringModel, max_examples: int | None = None, seed: int = 0, datasets: list[str] | None = None
) -> dict:
    """Run the benchmark with the same metrics as ``BTZSCBenchmark.evaluate`` on a seeded random
    subset of ``max_examples`` examples per dataset."""
    from btzsc.data import BTZSC_DATASETS, TASK_GROUPS
    from btzsc.metrics import compute_metrics, compute_task_summary

    per_dataset = {}
    for name in datasets or BTZSC_DATASETS:
        texts, labels, gold = load_examples(name, max_examples, seed)
        predictions = adapter.predict(texts, labels)
        per_dataset[name] = compute_metrics(predictions=predictions, references=gold)
        print(f"{name:28s} macro-F1 {per_dataset[name]['macro_f1']:.3f}  n={len(texts)}", flush=True)
    return {
        "per_dataset_results": per_dataset,
        "task_summary": compute_task_summary(per_dataset, TASK_GROUPS),
        "model_name": adapter.model_name,
        "max_examples_per_dataset": max_examples,
    }
