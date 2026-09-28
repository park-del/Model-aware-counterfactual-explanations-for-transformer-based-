"""Generate SAGE counterfactuals for correctly predicted test prefixes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models.io import load_checkpoint
from models.tokenizer import ProcessTokenizer
from preprocessing.common import read_jsonl
from sage.counterfactual import SAGEExplainer
from utils.config import load_config
from utils.seed import set_seed


def run(config, dataset, limit=None, prefix_length=None):
    seed = int(config["project"]["seed"])
    set_seed(seed)
    processed = ROOT / config["data"]["processed_dir"] / dataset
    output = ROOT / config["project"]["output_dir"] / dataset
    tokenizer = ProcessTokenizer.load(output / "tokenizer.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(output / "model.pt", device)
    train_cases = read_jsonl(processed / "train.jsonl")
    test_cases = read_jsonl(processed / "test.jsonl")
    sage_config = {**config["sage"], "seed": seed}
    explainer = SAGEExplainer(model, tokenizer, train_cases, sage_config, device)
    limit = int(limit or config["experiment"]["explanation_limit"])
    configured_prefix = int(prefix_length or config["experiment"]["explanation_prefix_length"])

    destination = output / "counterfactuals.jsonl"
    generated = attempted = 0
    with destination.open("w", encoding="utf-8") as handle:
        for case in test_cases:
            current_prefix = min(configured_prefix, max(1, len(case.activities) - 1))
            predicted, _ = explainer.predict(case, current_prefix)
            if predicted != case.label:
                continue
            attempted += 1
            started = time.perf_counter()
            result = explainer.explain(case, current_prefix)
            result.update({"case_id": case.case_id, "true_label": case.label, "runtime_seconds": time.perf_counter() - started})
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            generated += 1
            print(f"[{generated}/{limit}] {case.case_id}: validity={result['validity']}")
            if generated >= limit:
                break
    summary = {"dataset": dataset, "attempted_correct_prefixes": attempted, "generated": generated, "path": str(destination)}
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=("bpic2017", "bpic2012", "helpdesk", "sepsis"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prefix-length", type=int)
    args = parser.parse_args()
    run(load_config(ROOT / args.config), args.dataset, args.limit, args.prefix_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
