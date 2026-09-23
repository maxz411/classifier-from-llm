"""Latency of option scoring vs. constrained generation vs. JSON generation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from packreadout.bench.cached_baseline import score_options_cached
from packreadout.bench.generate_baseline import constrained_generate, json_generate
from packreadout.formatting import DEFAULT_FORMAT, OptionFormat, classify_prompt
from packreadout.scoring import score_options, score_options_naive


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass
class Timing:
    method: str
    n_options: int
    max_opt_tokens: int
    prompt_tokens: float  # mean over the inputs
    p50_ms: float
    p90_ms: float
    mean_ms: float
    tokens_forwarded: float  # tokens pushed through the model per call, mean over the inputs
    decode_steps: float  # sequential decode steps per call, mean over the inputs; 0 for scoring
    times_ms: list[float]  # one wall time per input, after warm-up


def _time_each(fns, device, warmup: int) -> list[float]:
    """One wall time in ms per callable, after ``warmup`` untimed calls of the first; NaN throughout if
    the method does not fit in memory."""
    try:
        for _ in range(warmup):
            fns[0]()
            _sync(device)
        times = []
        for fn in fns:
            _sync(device)
            t0 = time.perf_counter()
            fn()
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)
    except torch.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return [float("nan")] * len(fns)
    return times


def _timing(method: str, n: int, L: int, P: list[int], times: list[float], tokens: list, steps: list) -> Timing:
    arr = np.array(times)
    return Timing(
        method,
        n,
        L,
        float(np.mean(P)),
        float(np.nanpercentile(arr, 50)),
        float(np.nanpercentile(arr, 90)),
        float(np.nanmean(arr)),
        float(np.mean(tokens)),
        float(np.mean(steps)),
        [float(t) for t in times],
    )


METHODS = ("score_packed", "score_cached", "score_naive", "constrained_gen", "json_gen")


def bench_case(
    model,
    tok,
    prompts: list[str],
    options: list[str],
    fmt: OptionFormat = DEFAULT_FORMAT,
    warmup: int = 3,
    methods: tuple[str, ...] = METHODS,
) -> list[Timing]:
    """Time every method once per input, so that each Timing holds the spread over inputs: scoring
    depends only on token counts, generation also on how many tokens the model decides to produce."""
    device = model.device
    P = [len(tok(p, add_special_tokens=True).input_ids) for p in prompts]
    opt_lens = [len(tok(fmt.render(o), add_special_tokens=False).input_ids) for o in options]
    L, n, zeros = max(opt_lens), len(options), [0] * len(prompts)
    out = []
    if "score_packed" in methods:
        times = _time_each([lambda p=p: score_options(model, tok, p, options, fmt) for p in prompts], device, warmup)
        out.append(_timing("score_packed", n, L, P, times, [p + sum(opt_lens) for p in P], zeros))
    if "score_cached" in methods:
        times = _time_each(
            [lambda p=p: score_options_cached(model, tok, p, options, fmt) for p in prompts], device, warmup
        )
        out.append(_timing("score_cached", n, L, P, times, [p + n * L for p in P], zeros))
    if "score_naive" in methods:
        times = _time_each(
            [lambda p=p: score_options_naive(model, tok, p, options, fmt, batch_size=16) for p in prompts],
            device,
            warmup,
        )
        out.append(_timing("score_naive", n, L, P, times, [n * (p + L) for p in P], zeros))
    if "constrained_gen" in methods:  # one decode step per token of the label the model chooses for each input
        steps = [constrained_generate(model, tok, p, options, fmt).n_steps - 1 for p in prompts]  # first token: prefill
        times = _time_each(
            [lambda p=p: constrained_generate(model, tok, p, options, fmt) for p in prompts], device, warmup
        )
        out.append(_timing("constrained_gen", n, L, P, times, [p + s for p, s in zip(P, steps)], steps))
    if "json_gen" in methods:  # one decode step per token of the rationale and label the model writes
        steps = [json_generate(model, tok, p, options).n_steps - 1 for p in prompts]
        times = _time_each([lambda p=p: json_generate(model, tok, p, options) for p in prompts], device, warmup)
        out.append(_timing("json_gen", n, L, P, times, [p + s for p, s in zip(P, steps)], steps))
    return out


def label_pool() -> list[str]:
    """Real label names from the intent datasets, enough for a 255-option sweep."""
    from packreadout.tasks import get_task

    names: list[str] = []
    for task in ("banking77", "clinc150", "massive"):
        for name in get_task(task).load("test", 1)[0].options:
            if name not in names:
                names.append(name)
    return names


def sample_texts(k: int = 20) -> list[str]:
    """The benchmark's inputs: k customer queries from the Banking77 test set."""
    from datasets import load_dataset

    from packreadout.tasks.classify import SPECS

    spec = SPECS["banking77"][1]
    ds = load_dataset(spec.dataset, spec.config, split=spec.splits[1])
    return [row[spec.text] for row in ds.select(range(k))]


def merged(timings: list[Timing]) -> Timing:
    """One record for several draws of the same method and option count: every timed call pooled, the
    per-call means averaged, the option length the longest drawn."""
    times = [t for x in timings for t in x.times_ms]
    return Timing(
        timings[0].method,
        timings[0].n_options,
        max(x.max_opt_tokens for x in timings),
        float(np.mean([x.prompt_tokens for x in timings])),
        float(np.nanpercentile(times, 50)),
        float(np.nanpercentile(times, 90)),
        float(np.nanmean(times)),
        float(np.mean([x.tokens_forwarded for x in timings])),
        float(np.mean([x.decode_steps for x in timings])),
        times,
    )


def sweep_n_options(model, tok, texts: list[str], ns=(2, 4, 8, 16, 77, 255), draws=3, seed=0, **kw) -> list[Timing]:
    """At each option count, ``draws`` random label sets from the pool, each timed on every input, so that a
    generation method's cost reflects typical label lengths rather than one particular set."""
    pool = label_pool()
    out = []
    for n in ns:
        by_method: dict[str, list[Timing]] = {}
        for d in range(draws):
            labels = random.Random(seed + d).sample(pool, n)
            prompts = [
                classify_prompt(t, labels, "Classify the customer's banking query into exactly one intent.")
                for t in texts
            ]
            for timing in bench_case(model, tok, prompts, labels, **kw):
                by_method.setdefault(timing.method, []).append(timing)
        out.extend(merged(ts) for ts in by_method.values())
    return out


def sweep_option_length(model, tok, texts: list[str], words=(1, 3, 6, 12), n_options=8, **kw) -> list[Timing]:
    """Options of 1 to 12 words (code words repeated), to isolate the cost of option length."""
    base = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"][:n_options]
    out = []
    for w in words:
        options = [" ".join([b] * w) for b in base]
        prompts = [classify_prompt(t, options, "Pick the code word that best matches the text.") for t in texts]
        out.extend(bench_case(model, tok, prompts, options, **kw))
    return out


def table(timings: list[Timing]) -> str:
    lines = [
        "| method | n_opt | max_opt_tok | prompt_tok | p50 ms | p90 ms | tokens fwd | decode steps |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in timings:
        lines.append(
            f"| {t.method} | {t.n_options} | {t.max_opt_tokens} | {t.prompt_tokens:.0f} | {t.p50_ms:.1f} | {t.p90_ms:.1f} | {t.tokens_forwarded:.0f} | {t.decode_steps:.1f} |"
        )
    return "\n".join(lines)
