"""Run one real MAPDL structural-case evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moea_benchmark.cae.mapdl_structural_cases import (
    CAE_PROBLEM_CLASSES,
    analytical_cantilever_displacement,
    create_mapdl_problem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=sorted(CAE_PROBLEM_CLASSES),
        default="mapdl_cantilever_beam",
        help="MAPDL case to evaluate.",
    )
    parser.add_argument("--width", type=float, default=0.05)
    parser.add_argument("--height", type=float, default=0.08)
    parser.add_argument("--x", nargs="*", type=float, help="Override complete design vector.")
    parser.add_argument("--work-root", default="experiments_results/mapdl_smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problem = create_mapdl_problem(args.case, work_root=args.work_root)
    try:
        x = args.x if args.x else problem.default_x()
        if args.case == "mapdl_cantilever_beam" and not args.x:
            x = [args.width, args.height]
        objectives = problem.evaluate_design(x, raise_on_failure=True)
        print(f"MAPDL case: {args.case}")
        print(f"Design vector: {list(map(float, x))}")
        print(f"MAPDL objectives [volume, response]: {objectives}")
        if args.case == "mapdl_cantilever_beam":
            estimated = analytical_cantilever_displacement(float(x[0]), float(x[1]))
            print(f"Euler-Bernoulli displacement estimate: {estimated:.6e}")
        print(f"Evaluation log: {problem.eval_log.resolve()}")
    finally:
        problem.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
