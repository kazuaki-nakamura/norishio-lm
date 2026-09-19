"""Command-line entry point for the plastic-memory toy mechanism test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .plastic_memory_experiment import run_plastic_memory_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("codex/work_output/plastic-memory-demo"))
    parser.add_argument("--seed", type=int, default=37)
    arguments = parser.parse_args()
    print(json.dumps(run_plastic_memory_experiment(arguments.output_dir, seed=arguments.seed), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
