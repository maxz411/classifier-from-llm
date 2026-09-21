"""Latency benchmark: option scoring vs. constrained vs. JSON generation, one timing per input.

uv run scripts/bench_latency.py --model Qwen/Qwen3-1.7B-Base --texts 20
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from optscore.bench.latency import METHODS, sample_texts, sweep_n_options, sweep_option_length, table
from optscore.model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--texts", type=int, default=20, help="number of inputs; each method is timed once per input")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--ns", nargs="+", type=int, default=[2, 4, 8, 16, 77, 255])
    ap.add_argument("--draws", type=int, default=3, help="random label sets per option count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    model, tok = load_model(args.model, device=args.device)
    dev = model.device
    devname = torch.cuda.get_device_name(0) if dev.type == "cuda" else f"{platform.machine()} {dev.type}"
    print(f"model={args.model} device={devname} dtype={model.dtype}")
    texts = sample_texts(args.texts)
    kw = dict(warmup=args.warmup, methods=tuple(args.methods))

    print("\n### sweep: number of options (intent labels)")
    t_n = sweep_n_options(model, tok, texts, ns=tuple(args.ns), draws=args.draws, seed=args.seed, **kw)
    print(table(t_n))
    print("\n### sweep: option length in words (8 options)")
    t_len = sweep_option_length(model, tok, texts, **kw)
    print(table(t_len))

    tag = args.model.split("/")[-1]
    path = Path(args.out) / f"latency__{tag}__{dev.type}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": args.model,
                "device": devname,
                "dtype": str(model.dtype),
                "draws": args.draws,
                "n_options_sweep": [asdict(t) for t in t_n],
                "option_length_sweep": [asdict(t) for t in t_len],
            },
            indent=1,
        )
    )
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
