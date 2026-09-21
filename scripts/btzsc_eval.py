"""Score a base model or a trained adapter on the BTZSC zero-shot classification benchmark.

uv run scripts/btzsc_eval.py --model Qwen/Qwen3-1.7B-Base --max-samples 500
uv run scripts/btzsc_eval.py --model Qwen/Qwen3-1.7B-Base --adapter runs/1.7b-set-lmhead-btzscclean-s0 --max-samples 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optscore
import optscore.gptoss
from optscore.bench.btzsc_adapter import OptionScoringModel, evaluate_btzsc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--adapter", default=None, help="run directory written by scripts/train.py")
    ap.add_argument("--tasks", nargs="*", default=None, help="dataset names; default all 22")
    ap.add_argument("--max-samples", type=int, default=None, help="per-dataset cap: a seeded random subset of examples")
    ap.add_argument("--out", default=None, help="default: results/btzsc/<model or run name>.json")
    args = ap.parse_args()

    impl = optscore.gptoss if "gpt-oss" in args.model else optscore  # two complete implementations of the method
    model, tok = impl.model.load_model(args.model)
    readout, head, name = "lm_head", None, args.model.split("/")[-1]
    if args.adapter:
        model, readout, head = impl.train.load_adapter(model, args.adapter)
        name = Path(args.adapter).name

    results = evaluate_btzsc(
        OptionScoringModel(model, tok, name, readout, head, impl.scoring.score_examples),
        max_examples=args.max_samples,
        datasets=args.tasks,
    )
    out = Path(args.out or f"results/btzsc/{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("summary:", json.dumps(results["task_summary"]))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
