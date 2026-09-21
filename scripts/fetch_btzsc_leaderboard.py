"""Download the BTZSC leaderboard (btzsc/btzsc-results on the Hugging Face Hub) into one JSON file:
one entry per model with its family, parameter count and per-dataset macro-F1.

uv run scripts/fetch_btzsc_leaderboard.py --out results/btzsc_leaderboard.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def params_in_billions(s: str) -> float:
    return float(s[:-1]) / (1000 if s.endswith("M") else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/btzsc_leaderboard.json")
    args = ap.parse_args()

    root = snapshot_download("btzsc/btzsc-results", repo_type="dataset", allow_patterns=["results/*"])
    models = []
    for path in sorted(glob.glob(f"{root}/results/*/*.json")):
        j = json.loads(Path(path).read_text())
        models.append(
            {
                "name": j["model"]["name"],
                "family": j["model"]["model_type"],
                "params": params_in_billions(j["model"]["params"]),
                "overall": j["results"]["overall"]["macro_f1"],
                "by_task": {k: v["macro_f1"] for k, v in j["results"]["by_task"].items()},
                "by_dataset": {k: v["macro_f1"] for k, v in j["results"]["by_dataset"].items()},
            }
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(models, indent=1))
    print(f"saved {len(models)} models to {args.out}")


if __name__ == "__main__":
    main()
