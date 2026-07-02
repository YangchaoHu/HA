"""List registered benchmark problems and algorithms."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moea_benchmark.algorithms import list_algorithms
from moea_benchmark.benchmarks import list_problem_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-expensive", action="store_true")
    parser.add_argument("--tiers", nargs="*", default=None, help="Filter tiers: math engineering cae")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Algorithms:")
    for name in list_algorithms():
        print(f"  - {name}")

    print("\nProblems:")
    for spec in list_problem_specs(tiers=args.tiers, include_expensive=args.include_expensive):
        expensive = " expensive" if spec.expensive else ""
        print(f"  - {spec.name:<22} tier={spec.tier:<12}{expensive} source={spec.source}")
        print(f"    {spec.description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

