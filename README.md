# optscore

Turn a pretrained causal language model into a classifier over arbitrary label sets.
Options share one packed forward pass; a learned scalar head produces the option
probabilities. The repository contains the model, data loaders, and training and
evaluation code.

## Installation

```sh
uv sync --locked
```

## Training

```sh
uv run scripts/train.py --model Qwen/Qwen3-1.7B-Base --readout scalar --steps 2000 --out runs/1.7b-scalar
```

The default recipe trains on the 18-task mixture defined in `src/optscore/tasks/`
and evaluates on the held-out tasks. The output directory contains the LoRA
adapter, scalar head, training configuration and loss log. Use `--help` for the
existing training options.

```sh
uv run scripts/evaluate.py --model Qwen/Qwen3-1.7B-Base --adapter runs/1.7b-scalar --tasks heldout --method text
```

## Inference

After training:

```python
from optscore import load_model, score_examples
from optscore.train import load_adapter

model, tokenizer = load_model("Qwen/Qwen3-1.7B-Base")
model, readout, head = load_adapter(model, "runs/1.7b-scalar")
labels = ["positive", "negative", "neutral"]
prompt = "Review: The food was excellent.\nLabels: positive, negative, neutral\nLabel:"
result = score_examples(model, tokenizer, [prompt], [labels], readout=readout, head=head)[0]
print(dict(zip(labels, result.probs().tolist())))
```

`src/optscore/` contains model loading, packed attention, scoring and training;
`tasks/` contains the data loaders. `gptoss/` implements the same method for the
gpt-oss architecture. The training and evaluation scripts select it automatically.

## Baselines and benchmarks

The comparisons in the paper are run by `scripts/prompting_baseline.py` (prompting an
instruction-tuned model, free and constrained replies), `scripts/prompting_baseline_api.py`
(hosted models through the Vercel AI Gateway; needs `AI_GATEWAY_API_KEY`),
`scripts/bench_latency.py` (packed scoring against KV replication, one sequence per option,
constrained decoding and JSON generation) and `scripts/btzsc_eval.py` (the BTZSC zero-shot
classification benchmark; `scripts/fetch_btzsc_leaderboard.py` downloads its leaderboard).
`src/optscore/bench/` holds their shared code. The per-example evaluation reports, the
prompted baselines' replies, the latency sweeps and the leaderboard snapshot behind every
number in the paper are in
[`paper-results.tar.gz`](https://github.com/maxz411/classifier-from-llm/releases/download/v0.1.0/paper-results.tar.gz)
(108 MB, [release v0.1.0](https://github.com/maxz411/classifier-from-llm/releases/tag/v0.1.0));
extract it at the repository root to restore `results/`.

## Tests

```sh
uv sync --locked --extra dev
uv run pytest    # downloads Qwen3-0.6B-Base on first use
```

## License and acknowledgements

The code is licensed under [Apache-2.0](LICENSE). Base models and datasets retain
their own terms; see [DATA.md](DATA.md). Training downloads data from its original
sources. Model weights are not bundled with the code. Citation metadata is in
[CITATION.cff](CITATION.cff).

Claude Code and OpenAI Codex assisted with training code, substantial manuscript
revisions and figure preparation. The research idea originated with Max Zhang,
who takes responsibility for the work. The work was entirely self-funded; the
author declares no competing company relationships.
