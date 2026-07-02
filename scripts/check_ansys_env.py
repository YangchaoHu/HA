"""Inspect local ANSYS/PyMAPDL readiness.

Run with:
    python scripts/check_ansys_env.py --output experiments_results/ansys_env.json
    python scripts/check_ansys_env.py --launch-smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moea_benchmark.cae.ansys_env import inspect_ansys_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments_results/ansys_env_report.json")
    parser.add_argument(
        "--launch-smoke",
        action="store_true",
        help="Actually launch MAPDL and run a minimal /PREP7 smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_ansys_environment(run_launch_smoke=args.launch_smoke)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")
    print(report.to_json())
    print(f"\nWrote: {output.resolve()}")
    return 0 if report.ready_for_mapdl else 2


if __name__ == "__main__":
    raise SystemExit(main())

