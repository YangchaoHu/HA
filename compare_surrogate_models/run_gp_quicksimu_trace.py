"""
run_gp_quicksimu_trace.py
-------------------------

对 QuickSimu1 / QuickSimu2 使用 HA_Nelder_Mead(method='gp') 运行三种子实验，
记录每一代的 fes 与 best 个体，并保存到 compare_proxy_models 目录。

特性：
1) 固定种子：2026, 2027, 2028
2) 结果 CSV 存放在 compare_proxy_models/results/
3) 断点恢复：若 CSV 已存在，读取最后 generation，仅追加缺失代次
   （说明：算法本身未提供中间状态恢复，这里采用“重跑后只追加缺失记录”的输出恢复策略）
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# 允许在 compare_proxy_models 目录下直接运行脚本
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiment_runner import PROBLEMS, run_ha_nelder_mead_method


DEFAULT_SEEDS = [2026, 2027, 2028]
DEFAULT_PROBLEMS = ["QuickSimu1", "QuickSimu2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HA_Nelder_Mead(gp) on QuickSimu1/2 and export generation trace CSVs."
    )
    parser.add_argument("--pop-size", type=int, default=50, help="Population size.")
    parser.add_argument("--n-gen", type=int, default=30, help="Number of generations.")
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=DEFAULT_PROBLEMS,
        default=DEFAULT_PROBLEMS,
        help="Problems to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory to save csv results.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to save per-run logs. Default: <output-dir>/logs",
    )
    return parser.parse_args()


def load_existing_last_gen(csv_path: Path) -> int:
    """读取已有 CSV 的最后 generation（无文件或空文件返回 0）。"""
    if not csv_path.exists():
        return 0
    last_gen = 0
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "generation" in row and row["generation"]:
                    last_gen = max(last_gen, int(row["generation"]))
    except Exception:
        # 文件损坏则按 0 处理，让后续覆盖重建
        return 0
    return last_gen


def _evaluate_volume_and_constraint(problem, x: np.ndarray) -> Tuple[float, float]:
    """
    对单个解 x 重新评估，返回:
      - volume: 目标值（当前问题定义中的 F）
      - constraint_size: 约束违反度 sum(max(0, G))
    """
    out: Dict = {}
    problem._evaluate(np.atleast_2d(x).astype(float), out)
    f_arr = np.asarray(out.get("F", [np.nan]), dtype=float).reshape(-1)
    volume = float(f_arr[0]) if f_arr.size > 0 else float("nan")

    g_raw = out.get("G", None)
    if g_raw is None:
        constraint_size = 0.0
    else:
        g_arr = np.asarray(g_raw, dtype=float)
        g_arr = np.atleast_2d(g_arr).reshape(1, -1)
        constraint_size = float(np.sum(np.maximum(0.0, g_arr[0])))
    return volume, constraint_size


def trace_rows(problem_name: str, seed: int, history_data: List, problem) -> List[Dict]:
    rows: List[Dict] = []
    for g in history_data:
        best_x = np.asarray(g.best_X, dtype=float).tolist()
        volume, constraint_size = _evaluate_volume_and_constraint(problem, np.asarray(g.best_X, dtype=float))
        rows.append(
            {
                "problem": problem_name,
                "seed": seed,
                "generation": int(g.gen),
                "fes": int(g.fes),
                "volume": float(volume),
                "constraint_size": float(constraint_size),
                "best_x": json.dumps(best_x, ensure_ascii=False),
            }
        )
    return rows


def write_rows(csv_path: Path, rows: List[Dict], append: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with csv_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem",
                "seed",
                "generation",
                "fes",
                "volume",
                "constraint_size",
                "best_x",
            ],
        )
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def _update_progress_json(
    progress_path: Path,
    *,
    problem_name: str,
    seed: int,
    status: str,
    written_rows: int,
    final_gen: int,
    error: str | None = None,
) -> None:
    """
    记录中间结果进度，便于中断后查看执行状态。
    """
    now = datetime.now().isoformat(timespec="seconds")
    payload: Dict = {
        "updated_at": now,
        "runs": [],
    }
    if progress_path.exists():
        try:
            with progress_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if "runs" not in payload or not isinstance(payload["runs"], list):
                payload["runs"] = []
        except Exception:
            payload = {"updated_at": now, "runs": []}

    runs: List[Dict] = payload["runs"]
    run_key = f"{problem_name}|{seed}"
    row = {
        "key": run_key,
        "problem": problem_name,
        "seed": seed,
        "status": status,
        "written_rows": int(written_rows),
        "final_gen": int(final_gen),
        "error": error,
        "updated_at": now,
    }
    idx = next((i for i, r in enumerate(runs) if r.get("key") == run_key), None)
    if idx is None:
        runs.append(row)
    else:
        runs[idx] = row

    payload["updated_at"] = now
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_one(
    problem_name: str,
    seed: int,
    pop_size: int,
    n_gen: int,
    output_dir: Path,
    progress_path: Path,
) -> Tuple[int, int]:
    """
    返回 (written_rows, final_gen)
    """
    csv_path = output_dir / f"{problem_name}_gp_seed_{seed}.csv"
    last_gen = load_existing_last_gen(csv_path)
    if last_gen >= n_gen:
        print(f"[SKIP] {problem_name} seed={seed}: already has gen {last_gen} >= target {n_gen}")
        _update_progress_json(
            progress_path,
            problem_name=problem_name,
            seed=seed,
            status="skipped",
            written_rows=0,
            final_gen=last_gen,
            error=None,
        )
        return 0, last_gen

    if problem_name not in PROBLEMS:
        raise KeyError(f"Unknown problem: {problem_name}")
    problem = PROBLEMS[problem_name]()

    print(
        f"[RUN ] {problem_name} seed={seed}: pop_size={pop_size}, n_gen={n_gen}, "
        f"resume_from_gen={last_gen + 1}"
    )
    _, history_data = run_ha_nelder_mead_method(
        problem=problem,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        method="gp",
        initial_pop=None,
    )

    all_rows = trace_rows(problem_name, seed, history_data, problem)
    new_rows = [r for r in all_rows if int(r["generation"]) > last_gen]
    if not new_rows:
        print(f"[INFO] {problem_name} seed={seed}: no new rows to append.")
        _update_progress_json(
            progress_path,
            problem_name=problem_name,
            seed=seed,
            status="no_new_rows",
            written_rows=0,
            final_gen=last_gen,
            error=None,
        )
        return 0, last_gen

    append = csv_path.exists() and last_gen > 0
    # 若文件存在但解析失败(last_gen=0)，采用覆盖写避免混杂
    write_rows(csv_path, new_rows, append=append)
    final_gen = int(new_rows[-1]["generation"])
    print(f"[SAVE] {csv_path.name}: +{len(new_rows)} rows (final_gen={final_gen})")
    _update_progress_json(
        progress_path,
        problem_name=problem_name,
        seed=seed,
        status="done",
        written_rows=len(new_rows),
        final_gen=final_gen,
        error=None,
    )
    return len(new_rows), final_gen


def build_summary(output_dir: Path) -> Path:
    """合并所有单文件为 summary CSV。"""
    summary_path = output_dir / "all_gp_trace.csv"
    files = sorted(output_dir.glob("QuickSimu*_gp_seed_*.csv"))
    rows: List[Dict] = []
    for fp in files:
        with fp.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))
    write_rows(summary_path, rows, append=False)
    return summary_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir if args.log_dir is not None else (args.output_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "gp_trace_progress.json"

    total_written = 0
    failures: List[str] = []

    for problem_name in args.problems:
        for seed in args.seeds:
            log_path = log_dir / f"{problem_name}_gp_seed_{seed}.log"
            try:
                with log_path.open("w", encoding="utf-8") as logf:
                    with contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
                        print(f"[START] {datetime.now().isoformat(timespec='seconds')}")
                        print(
                            f"problem={problem_name}, seed={seed}, pop_size={args.pop_size}, "
                            f"n_gen={args.n_gen}, output_dir={args.output_dir}"
                        )
                        written, _ = run_one(
                            problem_name=problem_name,
                            seed=seed,
                            pop_size=args.pop_size,
                            n_gen=args.n_gen,
                            output_dir=args.output_dir,
                            progress_path=progress_path,
                        )
                        print(f"[END] {datetime.now().isoformat(timespec='seconds')}")
                total_written += written
                print(f"[LOG ] {problem_name}/seed={seed}: {log_path}")
            except Exception as exc:
                msg = f"{problem_name}/seed={seed}: {type(exc).__name__}: {exc}"
                failures.append(msg)
                print(f"[FAIL] {msg}")
                _update_progress_json(
                    progress_path,
                    problem_name=problem_name,
                    seed=seed,
                    status="failed",
                    written_rows=0,
                    final_gen=0,
                    error=msg,
                )

    summary = build_summary(args.output_dir)
    print(f"[DONE] rows_written={total_written}, summary={summary}")
    if failures:
        print("[WARN] failures:")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
