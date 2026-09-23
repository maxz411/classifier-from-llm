"""Evaluate option scoring on a task: accuracy, calibration and selective prediction.

A report (one JSON file per task) has ``by_norm[norm]["t1"]`` and
``by_norm[norm]["temp_scaled"]``, each a dict of the metrics below at temperature 1 and
after cross-fitted temperature scaling, plus ``per_example`` with every example's gold
index and per-option scores. ``norm`` is one of :data:`packreadout.normalize.NORMS`, or
``raw+bc`` for batch calibration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from packreadout.formatting import symbol_prompt
from packreadout.normalize import fit_temperature, log_softmax, to_probs
from packreadout.scoring import score_examples
from packreadout.tasks.base import Example, Task
from packreadout.tasks.paraphrase import paraphrase_examples


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top-1 confidence."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def selective_accuracy(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Accuracy on the most confident ``coverage`` fraction of examples."""
    k = max(1, int(round(coverage * len(conf))))
    return float(correct[np.argsort(-conf)[:k]].mean())


def metrics_for(scores_list: list[np.ndarray], gold: list[int], temperature: float = 1.0) -> dict:
    probs = [to_probs(s, temperature) for s in scores_list]
    pred = np.array([int(np.argmax(p)) for p in probs])
    g = np.array(gold)
    correct = (pred == g).astype(float)
    conf = np.array([p.max() for p in probs])
    nll = -np.mean([log_softmax(s / temperature)[k] for s, k in zip(scores_list, gold)])
    brier = np.mean([np.sum((p - np.eye(len(p))[k]) ** 2) for p, k in zip(probs, gold)])
    # macro-F1 is over option indices: a class F1 only when all examples share the option set
    out = {
        "n": len(gold),
        "acc": float(correct.mean()),
        "macro_f1": float(f1_score(g, pred, average="macro")),
        "nll": float(nll),
        "brier": float(brier),
        "ece": ece(conf, correct),
        "mean_conf": float(conf.mean()),
        "sel_acc@50": selective_accuracy(conf, correct, 0.5),
        "sel_acc@80": selective_accuracy(conf, correct, 0.8),
        "temperature": float(temperature),
    }
    if 0 < correct.sum() < len(correct):
        out["auroc"] = float(roc_auc_score(correct, conf))  # does confidence separate right from wrong
    return out


def crossfit_temperature(scores_list: list[np.ndarray], gold: list[int]) -> dict:
    """2-fold: fit T on the even examples, evaluate on the odd ones and vice versa, pool."""
    n = len(gold)
    if n < 4:
        return metrics_for(scores_list, gold)
    folds = [np.arange(n)[0::2], np.arange(n)[1::2]]
    held_scores, held_gold, temps = [], [], []
    for fit_idx, eval_idx in [(folds[0], folds[1]), (folds[1], folds[0])]:
        t = fit_temperature([scores_list[i] for i in fit_idx], [gold[i] for i in fit_idx])
        temps.append(t)
        held_scores.extend(scores_list[i] / t for i in eval_idx)
        held_gold.extend(gold[i] for i in eval_idx)
    return {**metrics_for(held_scores, held_gold), "temperature": float(np.mean(temps))}


def batch_calibrate(scores_list: list[np.ndarray]) -> list[np.ndarray]:
    """Batch Calibration (Zhou et al., ICLR 2024): subtract the mean score of each option over
    the test inputs, removing the prompt's label prior. Label free; needs a shared option set."""
    mean = np.mean(np.stack(scores_list), axis=0)
    return [s - mean for s in scores_list]


def as_symbol_examples(examples: list[Example]) -> list[Example]:
    """The first-token baseline: score the letter of each option instead of its text."""
    out = []
    for e in examples:
        prompt, letters = symbol_prompt(e.prompt, e.options)
        out.append(Example(prompt, letters, e.gold))
    return out


def as_fewshot_examples(task: Task, examples: list[Example], shots: int) -> list[Example]:
    """In-context baseline: ``shots`` training examples, each followed by its gold option, precede every prompt."""
    demos = task.load("train", shots)
    prefix = "".join(f"{d.prompt} {d.options[d.gold]}\n\n" for d in demos)
    return [Example(prefix + e.prompt, e.options, e.gold) for e in examples]


def evaluate(
    model,
    tok,
    task: Task,
    limit: int | None = None,
    batch_size: int = 8,
    method: str = "text",  # text: score option strings; symbol: score option letters; paraphrase: renamed labels; fewshot<k>
    readout: str = "lm_head",
    head: torch.nn.Linear | None = None,
    uncond: bool = False,
    score=score_examples,  # packreadout.gptoss.scoring.score_examples for a gpt-oss model
) -> dict:
    """Score the task's test examples and compute every metric for every applicable normalization."""
    examples = task.load("test", limit)
    if method == "paraphrase":
        examples = paraphrase_examples(task.name, examples)
        if examples is None:
            raise ValueError("no paraphrases for this task")
    if method == "symbol":
        examples = as_symbol_examples(examples)
    if method.startswith("fewshot"):
        examples = as_fewshot_examples(task, examples, int(method.removeprefix("fewshot")))

    if readout != "lm_head":
        norms = ("raw",)  # a trained head gives one logit per option: nothing to normalize by length
    elif method == "symbol":
        norms = ("raw",)  # every option is a letter of the same length
    else:
        norms = ("raw", "token", "byte", "char") + (("pmi",) if uncond else ())

    t0 = time.perf_counter()
    results = score(
        model,
        tok,
        [e.prompt for e in examples],
        [e.options for e in examples],
        task.fmt,
        readout,
        head,
        uncond,
        batch_size,
    )
    gold = [e.gold for e in examples]
    report = {
        "task": task.name,
        "role": task.role,
        "method": method,
        "n": len(examples),
        "n_options": float(np.mean([len(e.options) for e in examples])),
        "seconds": time.perf_counter() - t0,
        "by_norm": {},
    }
    shared_options = all(e.options == examples[0].options for e in examples)
    for norm in norms:
        scores_list = [r.scores(norm) for r in results]
        report["by_norm"][norm] = {
            "t1": metrics_for(scores_list, gold),
            "temp_scaled": crossfit_temperature(scores_list, gold),
        }
        if shared_options and norm == "raw":
            bc = batch_calibrate(scores_list)
            report["by_norm"]["raw+bc"] = {"t1": metrics_for(bc, gold), "temp_scaled": crossfit_temperature(bc, gold)}
    report["per_example"] = [
        {
            "gold": e.gold,
            "sum_logprob": r.sum_logprob.tolist(),
            "n_tokens": r.n_tokens.tolist(),
            "n_bytes": r.n_bytes.tolist(),
            "n_chars": r.n_chars.tolist(),
            "uncond_logprob": None if r.uncond_logprob is None else r.uncond_logprob.tolist(),
        }
        for e, r in zip(examples, results)
    ]
    return report


def summary_table(report: dict) -> str:
    rows = [
        "| norm | acc | macro-F1 | nll | brier | ece | auroc | sel@50 | ece (T fit) | T |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for norm, m in report["by_norm"].items():
        a, b = m["t1"], m["temp_scaled"]
        rows.append(
            f"| {norm} | {a['acc']:.3f} | {a['macro_f1']:.3f} | {a['nll']:.3f} | {a['brier']:.3f} | {a['ece']:.3f} "
            f"| {a.get('auroc', float('nan')):.3f} | {a['sel_acc@50']:.3f} | {b['ece']:.3f} | {b['temperature']:.2f} |"
        )
    return "\n".join(rows)


def save_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
