"""Run the paper's SAGE sensitivity sweeps and four ablations."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.evaluate import run as evaluate_run
from experiments.generate_counterfactuals import run as generate_run
from utils.config import load_config


SWEEPS = {
    "alpha": [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    "population_size": [250, 500, 1000, 1500],
    "generations": [5, 10, 15, 20],
    "mutation_probability": [0.05, 0.10, 0.20, 0.30, 0.40],
    "temperature": [0.10, 0.25, 0.50, 0.75, 1.00],
}

ABLATIONS = {
    "sage": {},
    "random_initialization": {"initialization": "random"},
    "random_mutation": {"mutation_strategy": "random"},
    "without_feasibility": {"enforce_feasibility": False},
    "distance_only": {"fitness_mode": "distance_only"},
}


def configurations(study):
    if study == "ablation":
        yield from ABLATIONS.items()
        return
    for value in SWEEPS[study]:
        changes = {study: value}
        if study == "alpha":
            changes["beta"] = 1.0 - value
        yield str(value), changes


def run(config, dataset, study, seeds, limit):
    output = ROOT / config["project"]["output_dir"] / dataset
    records = []
    for label, changes in configurations(study):
        for seed in seeds:
            current = copy.deepcopy(config)
            current["project"]["seed"] = seed
            current["sage"].update(changes)
            generate_run(current, dataset, limit=limit)
            metrics = evaluate_run(current, dataset)
            destination = output / "studies" / study / label / f"seed_{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / "counterfactuals.jsonl", destination / "counterfactuals.jsonl")
            (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            records.append({"configuration": label, "seed": seed, "changes": changes, "metrics": metrics})
    summary = output / "studies" / study / "summary.json"
    summary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="bpic2017", choices=("bpic2017", "bpic2012", "helpdesk", "sepsis"))
    parser.add_argument("--study", required=True, choices=tuple(SWEEPS) + ("ablation",))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    run(load_config(ROOT / args.config), args.dataset, args.study, args.seeds, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
