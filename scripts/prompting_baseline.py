"""Prompt an instruction-tuned model to name the label: the usual way an LLM is used as a classifier.

uv run scripts/prompting_baseline.py --model Qwen/Qwen3-1.7B --tasks heldout --limit 500

Every example gets the same prompt as our scorer, without the cue line, plus the instruction to
reply with one of the listed answers, in the model's chat format (thinking off). Its reply is read
two ways: ``free`` generates greedily and matches the text to an option (an unmatched reply counts
as wrong); ``constrained`` restricts greedy decoding to the option strings. Both are timed one
decision at a time, like the latency benchmark. Writes results/prompting/<model>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from optscore.bench.generate_baseline import Trie, _prefill, _step
from optscore.bench.prompting import score_replies, user_message
from optscore.model import load_model
from optscore.tasks import get_task, tasks_with_role


def chat_prompt(tok, prompt: str) -> str:
    """The user message in the model's chat format, with the assistant turn opened and thinking off."""
    messages = [{"role": "user", "content": user_message(prompt)}]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # a chat template without a thinking switch
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def free_generate(model, tok, prompt_ids: list[int], max_new_tokens: int = 24) -> tuple[str, int]:
    """Greedy generation up to the first newline or end-of-turn token."""
    logits, cache = _prefill(model, prompt_ids)
    produced: list[int] = []
    for _ in range(max_new_tokens):
        t = int(torch.argmax(logits))
        if t in (tok.eos_token_id, tok.pad_token_id) or "\n" in tok.decode([t]):
            break
        produced.append(t)
        logits, cache = _step(model, t, cache)
    return tok.decode(produced), len(produced)


@torch.no_grad()
def constrained_choice(model, tok, prompt_ids: list[int], options: list[str]) -> tuple[int, int]:
    """Greedy decoding restricted to the option strings (a trie over their tokens, as the reply's first tokens)."""
    trie = Trie()
    for i, o in enumerate(options):
        trie.insert(tok(o, add_special_tokens=False).input_ids, i)
    logits, cache = _prefill(model, prompt_ids)
    node, steps = trie, 0
    while node.leaf is None:
        allowed = torch.tensor(list(node.children), device=logits.device)
        masked = torch.full_like(logits, float("-inf"))
        masked[allowed] = logits[allowed]
        t = int(torch.argmax(masked))
        node, steps = node.children[t], steps + 1
        if node.leaf is None:
            logits, cache = _step(model, t, cache)
    return node.leaf, steps


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def evaluate_task(model, tok, task, limit: int) -> dict:
    examples = task.load("test", limit)
    out = {"n": len(examples)}
    for mode in ("free", "constrained"):
        replies, preds, times, tokens = [], [], [], []
        for e in examples:
            ids = tok(chat_prompt(tok, e.prompt), add_special_tokens=False).input_ids
            sync(model.device)
            t0 = time.perf_counter()
            if mode == "free":
                text, n = free_generate(model, tok, ids)
                replies.append(text)
            else:
                pred, n = constrained_choice(model, tok, ids, e.options)
                preds.append(pred)
            sync(model.device)
            times.append((time.perf_counter() - t0) * 1000)
            tokens.append(n)
        if mode == "free":
            out[mode] = score_replies(replies, examples)
        else:  # constrained decoding always names an option
            gold, pred = np.array([e.gold for e in examples]), np.array(preds)
            n_options = max(len(e.options) for e in examples)
            out[mode] = {
                "acc": float((pred == gold).mean()),
                "macro_f1": float(
                    f1_score(gold, pred, labels=list(range(n_options)), average="macro", zero_division=0)
                ),
                "unmatched": 0.0,
            }
        out[mode].update({"p50_ms": float(np.median(times)), "mean_tokens": float(np.mean(tokens))})
        print(
            f"{task.name:18s} {mode:12s} acc {out[mode]['acc']:.3f}  macro-F1 {out[mode]['macro_f1']:.3f}  invalid {out[mode]['unmatched']:.3f}  p50 {out[mode]['p50_ms']:.0f} ms",
            flush=True,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tasks", nargs="+", default=["heldout"], help="task names or a role")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default=None, help="default: results/prompting/<model>.json")
    args = ap.parse_args()

    model, tok = load_model(args.model)
    tasks = [
        t
        for name in args.tasks
        for t in (tasks_with_role(name) if name in {"train", "heldout", "sanity"} else [get_task(name)])
    ]
    results = {"model": args.model, "tasks": {task.name: evaluate_task(model, tok, task, args.limit) for task in tasks}}
    out = Path(args.out or f"results/prompting/{args.model.split('/')[-1]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
