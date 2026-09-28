"""Command-line entry point for the reproduction project."""

import argparse

from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the SAGE experiments")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML experiment configuration",
    )
    parser.add_argument(
        "--stage",
        choices=("preprocess", "train", "counterfactual", "evaluate"),
        required=True,
        help="Pipeline stage to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    print(f"Stage '{args.stage}' selected for dataset '{config['data']['dataset']}'.")
    print("The repository structure is ready; implementation will proceed module by module.")


if __name__ == "__main__":
    main()
