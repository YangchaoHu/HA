"""Fit a surrogate from an evaluation archive and save validation metrics."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moea_benchmark.surrogates import EvaluationArchive, fit_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--kind", choices=["rbf", "gp"], default="rbf")
    parser.add_argument("--output-dir", default="experiments_results/surrogates")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    x, f, _ = EvaluationArchive(args.archive).load()
    surrogate = fit_surrogate(x, f, kind=args.kind, random_state=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.archive).stem}_{args.kind}"
    with (output_dir / f"{stem}.pkl").open("wb") as fh:
        pickle.dump(surrogate, fh)
    (output_dir / f"{stem}_metrics.json").write_text(
        json.dumps(surrogate.metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(surrogate.metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

