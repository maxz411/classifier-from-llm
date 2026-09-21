"""Generation baselines for the latency comparison.

Both use an explicit greedy decode loop with a KV cache, so the measured cost is
the real per-step decode cost of the same model.

* :func:`constrained_generate` — decode the option string token by token with the
  logits masked to a trie over the rendered options. This is what a
  constrained-decoding library (outlines, vLLM guided decoding) does.
* :func:`json_generate` — free generation of ``{"reason": ..., "label": ...}``,
  the usual "classify with an LLM" pattern.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import torch

from optscore.formatting import DEFAULT_FORMAT, OptionFormat
from optscore.scoring import encode_options


@dataclass
class GenResult:
    choice: int | None
    text: str
    n_steps: int


class Trie:
    def __init__(self):
        self.children: dict[int, "Trie"] = {}
        self.leaf: int | None = None

    def insert(self, ids: list[int], idx: int):
        node = self
        for t in ids:
            node = node.children.setdefault(t, Trie())
        node.leaf = idx


@torch.no_grad()
def _prefill(model, ids: list[int]):
    out = model(input_ids=torch.tensor([ids], device=model.device), use_cache=True)
    return out.logits[0, -1], out.past_key_values


@torch.no_grad()
def _step(model, token: int, cache):
    out = model(input_ids=torch.tensor([[token]], device=model.device), past_key_values=cache, use_cache=True)
    return out.logits[0, -1], out.past_key_values


@torch.no_grad()
def constrained_generate(model, tok, prompt: str, options: list[str], fmt: OptionFormat = DEFAULT_FORMAT) -> GenResult:
    prompt_ids, opt_ids = encode_options(tok, prompt, options, fmt)
    trie = Trie()
    for i, ids in enumerate(opt_ids):
        trie.insert(ids, i)
    logits, cache = _prefill(model, prompt_ids)
    node, steps, produced = trie, 0, []
    while node.leaf is None:
        allowed = torch.tensor(list(node.children), device=logits.device)
        masked = torch.full_like(logits, float("-inf"))
        masked[allowed] = logits[allowed]
        t = int(torch.argmax(masked))
        produced.append(t)
        node = node.children[t]
        steps += 1
        if node.leaf is None:
            logits, cache = _step(model, t, cache)
    return GenResult(choice=node.leaf, text=tok.decode(produced), n_steps=steps)


JSON_INSTRUCTION = (
    'Respond with a JSON object of the form {"reason": "<one sentence>", "label": "<one of the options>"}.\n'
)


@torch.no_grad()
def json_generate(model, tok, prompt: str, options: list[str], max_new_tokens: int = 48) -> GenResult:
    text = prompt.rstrip()
    if text.endswith("Answer:") or text.endswith("Label:"):
        text = text.rsplit("\n", 1)[0]
    text = text + "\n" + JSON_INSTRUCTION + "Options: " + ", ".join(options) + "\nJSON: {"
    ids = tok(text, add_special_tokens=True).input_ids
    logits, cache = _prefill(model, ids)
    produced: list[int] = []
    for _ in range(max_new_tokens):
        t = int(torch.argmax(logits))
        produced.append(t)
        if t == tok.eos_token_id:
            break
        decoded = tok.decode(produced)
        if "}" in decoded:
            break
        logits, cache = _step(model, t, cache)
    decoded = "{" + tok.decode(produced)
    return GenResult(choice=_parse_choice(decoded, options), text=decoded, n_steps=len(produced))


def _parse_choice(decoded: str, options: list[str]) -> int | None:
    m = re.search(r"\{.*?\}", decoded, flags=re.S)
    label = None
    if m:
        try:
            label = json.loads(m.group(0)).get("label")
        except json.JSONDecodeError:
            label = None
    if label is None:
        m2 = re.search(r'"label"\s*:\s*"([^"]*)"', decoded)
        label = m2.group(1) if m2 else None
    if label is None:
        return None
    label = label.strip().lower()
    for i, o in enumerate(options):
        if o.strip().lower() == label:
            return i
    return None
