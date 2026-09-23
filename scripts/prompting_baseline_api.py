"""Prompt a large hosted model to name the label, through the Vercel AI Gateway (OpenAI-compatible).

AI_GATEWAY_API_KEY=... uv run scripts/prompting_baseline_api.py --model openai/gpt-5 --provider openai --tasks heldout --limit 500

The same user message as scripts/prompting_baseline.py, greedy (temperature 0), the reply matched to
an option; an unmatched reply counts as wrong and is recorded. Every request is restricted to the
specified providers, and the provider that served each reply is recorded. Bedrock is excluded.
Only accuracy and macro-F1 are reported: API latency says nothing about the model's compute. Writes
results/prompting/<model>.json in the format of the local baseline, with the "free" mode only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

from packreadout.bench.prompting import score_replies, user_message
from packreadout.tasks import get_task, tasks_with_role

URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
MODELS_URL = "https://ai-gateway.vercel.sh/v1/models"
# Routing policy for these experiments; this is not a legal certification of a provider.
# Pin a single provider for new runs.
ALLOWED_PROVIDERS = {
    "openai",
    "anthropic",
    "google",
    "xai",
    "mistral",
    "meta",
    "deepseek",
    "alibaba",
    "zai",
    "moonshot",
    "deepinfra",
    "novita",
    "groq",
    "baseten",
    "nebius",
}


def allowed_providers_of(model: str, key: str) -> list[str]:
    """Intersect the gateway's live endpoints with the experiment's provider allowlist."""
    r = requests.get(f"{MODELS_URL}/{model}/endpoints", headers={"Authorization": f"Bearer {key}"}, timeout=60)
    r.raise_for_status()
    serving = {e["provider_name"] for e in r.json()["data"]["endpoints"]}
    return sorted(serving & ALLOWED_PROVIDERS)


def served_by(response: dict) -> str:
    """The provider named in the gateway's response metadata."""
    meta = (response["choices"][0]["message"].get("provider_metadata") or {}).get("gateway") or {}
    routing = meta.get("routing") or {}
    for source in (meta, routing):
        for key in ("provider", "finalProvider", "final_provider"):
            if key in source:
                return str(source[key])
    raise RuntimeError(f"Gateway response {response.get('id', '<no id>')} has no serving-provider metadata")


def ask(model: str, prompt: str, key: str, providers: list[str], extra: dict, retries: int = 8) -> tuple[str, str, int]:
    """One chat completion at temperature 0, pinned to ``providers``: (reply, provider that served it,
    tokens generated including hidden reasoning).
    Retried with backoff on rate limits, capacity limits and server errors."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_message(prompt)}],
        "temperature": 0,
        "providerOptions": {"gateway": {"only": providers}},
        **extra,  # model-specific fields, e.g. a reasoning effort
    }
    for attempt in range(retries):
        try:
            r = requests.post(URL, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=120)
        except requests.exceptions.RequestException:  # a hung or dropped connection: retry like a server error
            if attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            raise
        if r.status_code == 200:
            d = r.json()
            provider = served_by(d)
            if provider not in providers:
                raise RuntimeError(f"{model}: response served by {provider!r}, outside requested providers {providers}")
            return (
                d["choices"][0]["message"]["content"] or "",
                provider,
                int(d.get("usage", {}).get("completion_tokens") or 0),
            )
        if (
            r.status_code in (408, 429, 498, 500, 502, 503, 504) and attempt < retries - 1
        ):  # 498: pinned providers at capacity
            time.sleep(2**attempt)
            continue
        raise RuntimeError(f"{model}: HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("unreachable")


def evaluate_task(model: str, task, limit: int, key: str, providers: list[str], extra: dict, workers: int) -> dict:
    examples = task.load("test", limit)
    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(lambda e: ask(model, e.prompt, key, providers, extra), examples))
    replies, served = [a[0] for a in answers], sorted({a[1] for a in answers})
    out = {"n": len(examples), "served_by": served, "free": score_replies(replies, examples)}
    out["served_by_per_reply"] = [a[1] for a in answers]
    out["free"]["mean_tokens"] = float(sum(a[2] for a in answers) / len(answers))  # decode steps per decision
    f = out["free"]
    print(
        f"{task.name:18s} acc {f['acc']:.3f}  macro-F1 {f['macro_f1']:.3f}  invalid {f['unmatched']:.3f} (empty {f['unmatched_empty']:.3f})  served by {served}",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-5", help="a gateway model id, provider/model")
    ap.add_argument(
        "--provider",
        default="auto",
        help="comma-separated gateway providers to pin, or auto: every allowed provider that serves the model",
    )
    ap.add_argument("--tasks", nargs="+", default=["heldout"], help="task names or a role")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extra-json", default="{}", help='extra request fields, e.g. \'{"reasoning_effort": "minimal"}\'')
    ap.add_argument("--out", default=None, help="default: results/prompting/<model>.json")
    args = ap.parse_args()
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if not key:
        raise SystemExit("AI_GATEWAY_API_KEY is missing; supply the existing gateway key through the environment")
    providers = allowed_providers_of(args.model, key) if args.provider == "auto" else args.provider.split(",")
    for provider in providers:
        if provider not in ALLOWED_PROVIDERS:
            raise SystemExit(f"{provider}: excluded by this experiment's provider allowlist")
    if not providers:
        raise SystemExit(f"{args.model}: no serving endpoint matches this experiment's provider allowlist")
    print(f"{args.model}: pinned to {providers}", flush=True)

    tasks = [
        t
        for name in args.tasks
        for t in (tasks_with_role(name) if name in {"train", "heldout", "sanity"} else [get_task(name)])
    ]
    results = {
        "model": args.model,
        "provider": ",".join(providers),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "extra": json.loads(args.extra_json),
        "temperature": 0,
        "workers": args.workers,
        "limit": args.limit,
        "tasks": {},
    }
    out = Path(args.out or f"results/api/prompting/{args.model.replace('/', '-')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        results["tasks"][task.name] = evaluate_task(
            args.model, task, args.limit, key, providers, results["extra"], args.workers
        )
        out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
