"""
将同一 problem 下多个 seed 的 gp trace CSV 按 generation 对齐求平均。

示例（QuickSimu1 三种子）:
    python compare_proxy_models/aggregate_gp_trace_seeds.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按 generation 对多 seed gp trace 求平均")
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            ROOT / "results" / "QuickSimu1_gp_seed_2026.csv",
            ROOT / "results" / "QuickSimu1_gp_seed_2027.csv",
            ROOT / "results" / "QuickSimu1_gp_seed_2028.csv",
        ],
        help="输入 CSV 路径列表",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "QuickSimu1_gp_seeds_mean.csv",
        help="输出 CSV 路径",
    )
    return p.parse_args()


def load_rows(paths: List[Path]) -> Tuple[str, Dict[int, List[dict]]]:
    """返回 problem 名与 generation -> 行列表"""
    by_gen: Dict[int, List[dict]] = defaultdict(list)
    problem_name = ""
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                g = int(row["generation"])
                by_gen[g].append(row)
                if not problem_name:
                    problem_name = row["problem"]
    return problem_name, by_gen


def main() -> int:
    args = parse_args()
    problem_name, by_gen = load_rows(list(args.inputs))

    rows_out: List[dict] = []
    for gen in sorted(by_gen.keys()):
        group = by_gen[gen]
        n = len(group)
        if n == 0:
            continue

        fes_vals = [int(r["fes"]) for r in group]
        best_f_vals = [float(r["best_f"]) for r in group]
        xs = []
        for r in group:
            xs.append(json.loads(r["best_x"]))
        X = np.asarray(xs, dtype=float)

        x_mean = X.mean(axis=0).tolist()
        row = {
            "problem": problem_name,
            "generation": gen,
            "n_seeds": n,
            "fes_mean": float(np.mean(fes_vals)),
            "best_f_mean": float(np.mean(best_f_vals)),
            "best_x_mean": json.dumps(x_mean, ensure_ascii=False),
        }
        rows_out.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem",
        "generation",
        "n_seeds",
        "fes_mean",
        "best_f_mean",
        "best_x_mean",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"[OK] 写入 {args.output}（共 {len(rows_out)} 行，按 generation 平均 {len(args.inputs)} 个 seed）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
