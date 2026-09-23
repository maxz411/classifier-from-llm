"""Fine-tune the option-set distribution directly.

For every training example the packed forward gives one logit per option
(:func:`packreadout.packing.readout_logits`): ``s_i = sum_t log P(opt_i[t] | prompt, opt_i[:t])``
with the LM-head readout, or ``s_i = w . h_i`` with ``h_i`` the hidden state at the
option's last token (scalar-head readout). The objective is the cross-entropy of the
gold option under ``softmax(s)``: it trains exactly the distribution used at inference
and is a proper scoring rule, so calibration is part of the objective.

Other objectives, for comparison:

* ``gold``: the SFT baseline, maximize the gold option's likelihood without the softmax
  over the option set (loss only on the option tokens).
* ``brier``: the Brier score of the option distribution. With ground-truth labels a
  proper-scoring-rule "reward" is a differentiable function of the distribution, so
  training with it as a loss is exactly what an RL method with that reward would
  optimize, without the sampling noise.
* ``reinforce``: a genuine policy gradient. Sample a decision from the distribution,
  reward 1 if it is the gold option, running-mean baseline. It optimizes accuracy, not
  calibration, and is expected to produce over-confident distributions.

LoRA adapters via peft; the scalar head, when used, is a single trainable vector. A run
directory holds the adapter (peft's files), ``config.json`` (this TrainConfig),
``head.pt`` (the scalar head, if any) and ``log.jsonl`` (loss per 10 steps).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from packreadout.formatting import DEFAULT_FORMAT, OptionFormat
from packreadout.packing import packed_forward, readout_logits
from packreadout.scoring import encode_options
from packreadout.tasks.base import Example, Task


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen3-0.6B-Base"
    readout: str = "lm_head"  # lm_head | scalar | both (sum of the two)
    objective: str = "set"  # set | gold | brier | reinforce (see module docstring)
    length_norm: str = "raw"  # LM-head logit = raw sum | byte (sum / bytes of the rendered option)
    steps: int = 2000
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 2e-4
    warmup: int = 100
    lora_rank: int = 16
    examples_per_task: int = 2000
    seed: int = 0
    out: str = "runs/default"


def option_logits(
    model,
    tok,
    examples: list[Example],
    readout: str,
    head: torch.nn.Linear | None = None,
    length_norm: str = "raw",
    fmt: OptionFormat = DEFAULT_FORMAT,
) -> list[torch.Tensor]:
    """One logit vector per example (differentiable): encode, one packed forward, read out."""
    items = [encode_options(tok, e.prompt, e.options, fmt) for e in examples]
    hidden, packed = packed_forward(model, tok, items)
    logits = readout_logits(model, hidden, packed, items, readout, head)
    if length_norm == "byte":  # the ablation: the LM-head logit per byte of the rendered option
        if readout != "lm_head":
            raise ValueError("length normalization applies to the LM-head readout only")
        logits = [
            s / torch.tensor([len(fmt.render(o).encode()) for o in e.options], device=s.device)
            for s, e in zip(logits, examples)
        ]
    return logits


def loss_fn(logits: list[torch.Tensor], golds: list[int], objective: str, state: dict) -> torch.Tensor:
    """Mean loss of a batch; ``state`` keeps the running statistics some objectives need."""
    losses = []
    for s, g in zip(logits, golds):
        if objective == "set":
            losses.append(-torch.log_softmax(s, 0)[g])
        elif objective == "gold":  # SFT: -log P(gold option); the other options are not scored against it
            losses.append(-s[g])
        elif objective == "brier":
            p = torch.softmax(s, 0)
            losses.append(((p - torch.nn.functional.one_hot(torch.tensor(g, device=s.device), len(s))) ** 2).sum())
        elif objective == "reinforce":
            log_p = torch.log_softmax(s, 0)
            action = torch.multinomial(log_p.exp().detach(), 1).item()
            reward = float(action == g)
            baseline = state.setdefault("baseline", 0.5)
            state["baseline"] = 0.99 * baseline + 0.01 * reward
            losses.append(-(reward - baseline) * log_p[action])
        else:
            raise ValueError(objective)
    return torch.stack(losses).mean()


def build_mixture(tasks: list[Task], per_task: int, seed: int) -> list[Example]:
    examples = [e for t in tasks for e in t.load("train", per_task)]
    random.Random(seed).shuffle(examples)
    return examples


def add_lora(model, rank: int):
    config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, config)


def load_adapter(model, adapter_dir: str | Path):
    """Attach a trained adapter to its base model. Returns ``(model, readout, head)``;
    ``head`` is the scalar head for the ``scalar`` and ``both`` readouts, else None.
    The adapter is merged into the weights, which is faster at inference and changes nothing else."""
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    readout = json.loads((Path(adapter_dir) / "config.json").read_text())["readout"]
    head = None
    if readout in ("scalar", "both"):
        head = torch.nn.Linear(model.config.hidden_size, 1, bias=False, device=model.device, dtype=model.dtype)
        head.load_state_dict(torch.load(Path(adapter_dir) / "head.pt", map_location=model.device))
    return model, readout, head


def train(model, tok, tasks: list[Task], cfg: TrainConfig) -> Path:
    """Train and save an adapter into ``cfg.out``. The model passed in is left LoRA-wrapped;
    reload the base model and call :func:`load_adapter` to evaluate."""
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))

    examples = build_mixture(tasks, cfg.examples_per_task, cfg.seed)
    model = add_lora(model, cfg.lora_rank)
    params = [p for p in model.parameters() if p.requires_grad]
    head = None
    if cfg.readout in ("scalar", "both"):
        head = torch.nn.Linear(model.config.hidden_size, 1, bias=False, device=model.device, dtype=torch.float32)
        params.append(head.weight)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)

    def lr_at(step: int) -> float:  # linear warmup, then cosine decay to zero at the last step
        return cfg.lr * min(1.0, step / max(cfg.warmup, 1)) * 0.5 * (1 + math.cos(math.pi * step / cfg.steps))

    model.train()
    state: dict = {}
    t0, cursor = time.time(), 0
    with open(out / "log.jsonl", "w") as log:
        for step in range(1, cfg.steps + 1):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            total = 0.0
            for _ in range(cfg.grad_accum):
                batch = examples[
                    cursor : cursor + cfg.batch_size
                ]  # one pass over the shuffled mixture, wrapping around
                cursor = (cursor + cfg.batch_size) % len(examples)
                logits = option_logits(model, tok, batch, cfg.readout, head, cfg.length_norm)
                loss = (
                    loss_fn([s.float() for s in logits], [e.gold for e in batch], cfg.objective, state) / cfg.grad_accum
                )
                loss.backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 10 == 0 or step == 1:
                rec = {"step": step, "loss": total, "lr": lr_at(step), "seconds": time.time() - t0}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)

    model.save_pretrained(out)
    if head is not None:
        torch.save(head.state_dict(), out / "head.pt")
    return out
