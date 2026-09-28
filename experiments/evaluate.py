"""Aggregate SAGE counterfactual and surrogate-rule metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import load_config
from utils.metrics import evaluate_counterfactuals
from preprocessing.common import read_jsonl


def run(config, dataset):
    output = ROOT / config["project"]["output_dir"] / dataset
    source = output / "counterfactuals.jsonl"
    with source.open(encoding="utf-8") as handle:
        results = [json.loads(line) for line in handle]
    reference_cases = read_jsonl(ROOT / config["data"]["processed_dir"] / dataset / "train.jsonl")
    metrics = evaluate_counterfactuals(results, reference_cases)
    destination = output / "counterfactual_metrics.json"
    destination.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=("bpic2017", "bpic2012", "helpdesk", "sepsis"))
    args = parser.parse_args()
    run(load_config(ROOT / args.config), args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
