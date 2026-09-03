"""Evaluate dense, compressed, and retrained variants on a task suite."""

from __future__ import annotations

import argparse

from ..evaluation_suite import EvaluationSuiteConfig, run_evaluation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to an evaluation-suite JSON config")
    args = parser.parse_args()
    run_evaluation_suite(EvaluationSuiteConfig.from_json(args.config))


if __name__ == "__main__":
    main()
