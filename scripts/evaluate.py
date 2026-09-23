"""Evaluate a base model or a trained adapter on tasks; one report per task under --out.

uv run scripts/evaluate.py --tasks heldout --limit 500                      # zero-shot, option text
uv run scripts/evaluate.py --tasks heldout --method symbol                  # first-token letter baseline
uv run scripts/evaluate.py --tasks sanity --limit 400                       # lm-eval format check
uv run scripts/evaluate.py --model Qwen/Qwen3-1.7B-Base --adapter runs/1.7b-set-scalar-s0 --tasks heldout --method paraphrase
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import packreadout
import packreadout.gptoss
from packreadout.eval import evaluate, save_report, summary_table
from packreadout.tasks import get_task, tasks_with_role


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--device", default=None, help="cuda, mps, cpu, or auto: shard a big model over every GPU")
    ap.add_argument("--adapter", default=None, help="run directory written by scripts/train.py")
    ap.add_argument("--tasks", nargs="+", default=["heldout"], help="task names or a role: train | heldout | sanity")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument(
        "--method", default="text", help="text | symbol | paraphrase | fewshot<k> (k training examples in the prompt)"
    )
    ap.add_argument("--uncond", action="store_true", help="also compute PMI normalization")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: results/<model or run name>/<method>")
    args = ap.parse_args()

    impl = packreadout.gptoss if "gpt-oss" in args.model else packreadout  # two complete implementations of the method
    model, tok = impl.model.load_model(args.model, args.device)
    readout, head, tag = "lm_head", None, args.model.split("/")[-1]
    if args.adapter:
        model, readout, head = impl.train.load_adapter(model, args.adapter)
        tag = Path(args.adapter).name
    out = Path(args.out or f"results/{tag}/{args.method}")
    print(f"model={args.model} adapter={args.adapter} device={model.device} dtype={model.dtype} -> {out}")

    tasks = [
        t
        for name in args.tasks
        for t in (tasks_with_role(name) if name in {"train", "heldout", "sanity"} else [get_task(name)])
    ]
    summary = {}
    for task in tasks:
        try:
            report = evaluate(
                model,
                tok,
                task,
                args.limit,
                args.batch_size,
                args.method,
                readout,
                head,
                args.uncond,
                impl.scoring.score_examples,
            )
        except ValueError as err:  # e.g. symbol scoring with more than 26 options, or no paraphrases
            print(f"\n## {task.name}: skipped ({err})")
            continue
        save_report(report, out / f"{task.name}.json")
        summary[task.name] = {norm: m["t1"] for norm, m in report["by_norm"].items()}
        print(f"\n## {task.name}  (n={report['n']}, {report['seconds']:.1f}s)\n{summary_table(report)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
