"""Preprocess one paper dataset into case-level JSONL splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.bpic2012 import preprocess_bpic2012
from preprocessing.bpic2017 import preprocess_bpic2017
from preprocessing.helpdesk import preprocess_helpdesk
from preprocessing.sepsis import preprocess_sepsis
from utils.config import load_config


RUNNERS = {
    "bpic2017": preprocess_bpic2017,
    "bpic2012": preprocess_bpic2012,
    "helpdesk": preprocess_helpdesk,
    "sepsis": preprocess_sepsis,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=RUNNERS, required=True)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    dataset_config = config["datasets"][args.dataset]
    input_path = ROOT / dataset_config["raw_path"]
    output_dir = ROOT / config["data"]["processed_dir"] / args.dataset
    options = {
        **config["data"],
        **dataset_config,
        "seed": config["project"]["seed"],
    }
    metadata = RUNNERS[args.dataset](input_path, output_dir, options)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
