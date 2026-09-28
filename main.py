"""Command-line entry point for the SAGE reproduction project."""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the SAGE experiments")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML experiment configuration",
    )
    parser.add_argument(
        "--stage",
        choices=("download", "preprocess", "train", "counterfactual", "baseline", "evaluate"),
        required=True,
        help="Pipeline stage to run",
    )
    parser.add_argument("--dataset", choices=("bpic2017", "bpic2012", "helpdesk", "sepsis"))
    parser.add_argument("--method", choices=("dice", "face", "growing_spheres", "loreley"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage != "download" and not args.dataset:
        raise SystemExit("--dataset is required for this stage")
    if args.stage == "baseline" and not args.method:
        raise SystemExit("--method is required for the baseline stage")
    commands = {
        "download": [sys.executable, "scripts/download_datasets.py"],
        "preprocess": [sys.executable, "scripts/preprocess.py", "--config", args.config, "--dataset", args.dataset],
        "train": [sys.executable, "experiments/train_predictor.py", "--config", args.config, "--dataset", args.dataset],
        "counterfactual": [sys.executable, "experiments/generate_counterfactuals.py", "--config", args.config, "--dataset", args.dataset],
        "baseline": [sys.executable, "experiments/run_baselines.py", "--config", args.config, "--dataset", args.dataset, "--method", args.method],
        "evaluate": [sys.executable, "experiments/evaluate.py", "--config", args.config, "--dataset", args.dataset] + (["--method", args.method] if args.method else []),
    }
    subprocess.run(commands[args.stage], check=True)


if __name__ == "__main__":
    main()
