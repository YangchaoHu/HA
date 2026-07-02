"""Run the multi-level MOEA benchmark matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moea_benchmark.benchmarks import list_problem_specs
from moea_benchmark.runner import run_matrix
from moea_benchmark.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON/YAML config file.")
    parser.add_argument("--problems", nargs="+")
    parser.add_argument("--tiers", nargs="+", choices=["math", "engineering", "cae"])
    parser.add_argument("--include-expensive", action="store_true")
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--n-gen", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _defaults() -> dict[str, Any]:
    return {
        "problems": None,
        "tiers": ["math"],
        "include_expensive": False,
        "algorithms": ["nsga2", "nsga3"],
        "seeds": [1, 2, 3],
        "pop_size": 80,
        "n_gen": 100,
        "save_every": 1,
        "output_root": "experiments_results/moea_benchmark",
        "verbose": False,
    }


def _merge_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = _defaults()
    merged.update(config)

    cli_fields = {
        "problems": args.problems,
        "tiers": args.tiers,
        "include_expensive": True if args.include_expensive else None,
        "algorithms": args.algorithms,
        "seeds": args.seeds,
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "save_every": args.save_every,
        "output_root": args.output_root,
        "verbose": True if args.verbose else None,
    }
    for key, value in cli_fields.items():
        if value is not None:
            merged[key] = value
    return merged


def _select_problems(config: dict[str, Any]) -> list[str]:
    if config.get("problems"):
        return [str(name).lower() for name in config["problems"]]

    specs = list_problem_specs(
        tiers=config.get("tiers"),
        include_expensive=bool(config.get("include_expensive", False)),
    )
    return [spec.name for spec in specs if spec.enabled_by_default or config.get("include_expensive")]


def main() -> int:
    args = parse_args()
    config = load_config(args.config) if args.config else {}
    cfg = _merge_args(config, args)
    problems = _select_problems(cfg)

    print("Benchmark configuration:")
    print(json.dumps({**cfg, "problems": problems}, indent=2, ensure_ascii=False))

    results = run_matrix(
        problems=problems,
        algorithms=[str(x).lower() for x in cfg["algorithms"]],
        seeds=[int(x) for x in cfg["seeds"]],
        pop_size=int(cfg["pop_size"]),
        n_gen=int(cfg["n_gen"]),
        output_root=cfg["output_root"],
        save_every=int(cfg["save_every"]),
        verbose=bool(cfg.get("verbose", False)),
    )

    print("\nCompleted jobs:")
    failures = 0
    for item in results:
        print(f"  {item.status:<7} {item.problem:<22} {item.algorithm:<10} seed={item.seed:<5} {item.run_dir}")
        if item.status != "ok":
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

