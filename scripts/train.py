"""Fine-tune on the training mixture, then evaluate the adapter on the held-out tasks.

uv run scripts/train.py --model Qwen/Qwen3-1.7B-Base --steps 2000 --out runs/1.7b-set-lmhead-s0

The adapter goes to --out; its evaluations go to results/<run name>/<method>/, like scripts/evaluate.py.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

import optscore
import optscore.gptoss
from optscore.eval import evaluate, save_report, summary_table
from optscore.tasks import get_task, tasks_with_role
from optscore.train import TrainConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--device", default=None, help="cuda, mps, cpu, or auto: shard a big model over every GPU")
    ap.add_argument("--readout", default="lm_head", choices=["lm_head", "scalar", "both"])
    ap.add_argument("--objective", default="set", choices=["set", "gold", "brier", "reinforce"])
    ap.add_argument("--length-norm", default="raw", choices=["raw", "byte"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--examples-per-task", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--train-tasks", nargs="*", default=None, help="default: every task with role=train")
    ap.add_argument("--eval-tasks", nargs="*", default=None, help="default: every task with role=heldout")
    ap.add_argument("--eval-limit", type=int, default=500)
    ap.add_argument("--eval-methods", nargs="+", default=["text"], help="text, plus paraphrase and/or symbol")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    args = ap.parse_args()
    cfg = TrainConfig(**{k: getattr(args, k) for k in TrainConfig.__dataclass_fields__})
    impl = optscore.gptoss if "gpt-oss" in cfg.model else optscore  # two complete implementations of the method

    model, tok = impl.model.load_model(cfg.model, args.device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    train_tasks = [get_task(n) for n in args.train_tasks] if args.train_tasks else tasks_with_role("train")
    print(f"training on {[t.name for t in train_tasks]}")
    out = impl.train.train(model, tok, train_tasks, cfg)

    # the adapter only exists on this machine for as long as the job runs, so every evaluation of it happens here;
    # the trained model is freed first, so that the fresh copy and the adapter fit where the training run fit
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model, tok = impl.model.load_model(cfg.model, args.device)
    model, readout, head = impl.train.load_adapter(model, out)
    eval_tasks = [get_task(n) for n in args.eval_tasks] if args.eval_tasks else tasks_with_role("heldout")
    for method in args.eval_methods:
        folder = Path("results") / out.name / method
        summary = {}
        for task in eval_tasks:
            try:
                report = evaluate(
                    model,
                    tok,
                    task,
                    args.eval_limit,
                    method=method,
                    readout=readout,
                    head=head,
                    score=impl.scoring.score_examples,
                )
            except ValueError as err:  # symbol scoring with more than 26 options, or no paraphrases
                print(f"\n## {task.name} ({method}): skipped ({err})")
                continue
            save_report(report, folder / f"{task.name}.json")
            summary[task.name] = report["by_norm"]["raw"]["t1"]
            print(f"\n## {task.name} ({method})\n{summary_table(report)}", flush=True)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
