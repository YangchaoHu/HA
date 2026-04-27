"""
run_bandit.py
-------------

统一运行脚本：支持两个 Bandit V2 版本的对比实验。

    QL-V2   ── 改进 ε-greedy Q-learning（双选择器，绝对改进奖励）
    UCB-V2  ── 改进 UCB1（双选择器，绝对改进奖励）

用法：
    # 运行所有版本（依次运行）
    python run_bandit.py

    # 只运行指定版本（逗号分隔）
    python run_bandit.py --versions QL-V2,UCB-V2

输出：
    在 EXISTING_RESULTS_DIR 下生成：
        {problem_name}_{method_name}_summary.csv
    例：F2_QL-V2_summary.csv、Ackley_UCB-V2_summary.csv

绘图：
    结果保存完毕后，直接运行 evaluate_metrics.py 查看对比图。
"""

import argparse
import contextlib
import csv
import os
import warnings
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================================
# 线程限制（防止多进程 + BLAS 线程爆炸）
# ============================================================================
os.environ.setdefault("OMP_NUM_THREADS",         "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",    "1")
os.environ.setdefault("MKL_NUM_THREADS",         "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS",  "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",     "1")

import numpy as np
from numpy.typing import NDArray

from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

from problem import (
    F2Problem, F3Problem, F4Problem,
    RastriginProblem, GriewankProblem, AckleyProblem,
)
from ha_bandit import HA_QL, HA_UCB
from experiment_runner import generate_initial_population

# ============================================================================
# ★ 用户配置区 ★
# ============================================================================

# 将结果写入此目录（与其他方法的对照实验共用同一目录）
EXISTING_RESULTS_DIR = Path(__file__).parent / "experiments_results" / "result_20260115_224425"

POP_SIZE     = 50
N_GEN        = 30
N_RUNS       = 10
RANDOM_SEEDS = list(range(42, 42 + N_RUNS))

PROBLEMS: Dict[str, Callable] = {
    "F2":        lambda: F2Problem(n_var=10, m=10, xl=0.0, xu=np.pi),
    "F3":        lambda: F3Problem(n_var=10, xl=-100.0, xu=100.0),
    "F4":        lambda: F4Problem(n_var=10, xl=-100.0, xu=100.0),
    "Rastrigin": lambda: RastriginProblem(n_var=10, xl=-5.12, xu=5.12),
    "Griewank":  lambda: GriewankProblem(n_var=10, xl=-600.0, xu=600.0),
    "Ackley":    lambda: AckleyProblem(n_var=10, xl=-32.768, xu=32.768),
}

# ============================================================================
# 四个版本的完整配置
# ============================================================================

VARIANT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "QL": {
        "cls": HA_QL,
        "desc": "改进 ε-greedy QL（双选择器，绝对改进奖励）",
        "kwargs": {
            "global_best_actions": ["Nelder-Mead", "L-BFGS-B"],
            "elite_actions":       ["rbf", "gp"],
            "epsilon":             1.0,
            "epsilon_decay":       0.95,
            "epsilon_min":         0.05,
        },
    },
    "UCB": {
        "cls": HA_UCB,
        "desc": "改进 UCB1（双选择器，绝对改进奖励）",
        "kwargs": {
            "global_best_actions": ["Nelder-Mead", "L-BFGS-B"],
            "elite_actions":       ["rbf", "gp"],
            "c":                   1.0,
            "alpha":               0.1,
            "reward_scale_beta":   0.1,
        },
    },
}

# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class GenerationData:
    gen:    int
    fes:    int
    X:      NDArray
    F:      NDArray
    best_X: NDArray
    best_F: float


@dataclass
class RunResult:
    problem_name: str
    method_name:  str
    run_id:       int
    seed:         int
    generations:  List[GenerationData] = field(default_factory=list)
    final_best_X: Optional[NDArray]    = None
    final_best_F: Optional[float]      = None
    runtime:      float                = 0.0


# ============================================================================
# 回调
# ============================================================================

class DataCollectorCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data: List[GenerationData] = []

    def notify(self, algorithm):
        pop  = algorithm.pop
        X    = pop.get("X").copy()
        F    = pop.get("F").copy()
        fes  = int(getattr(algorithm.problem, "fes", 0))
        best = int(np.argmin(F))
        self.data.append(GenerationData(
            gen=algorithm.n_gen, fes=fes, X=X, F=F,
            best_X=X[best].copy(), best_F=float(F[best]),
        ))


# ============================================================================
# Worker 初始化（Windows spawn 模式下必须）
# ============================================================================

_RESULTS_DIR: Path = EXISTING_RESULTS_DIR

def _worker_init(results_dir: Path, variant_name: str) -> None:
    global _RESULTS_DIR, _CURRENT_VARIANT
    _RESULTS_DIR     = results_dir
    _CURRENT_VARIANT = variant_name


# ============================================================================
# 单次运行（子进程）
# ============================================================================

def _run_single(args: Tuple) -> RunResult:
    """在子进程中运行一次实验。"""
    import time
    problem_name, run_id, seed, variant_name = args

    warnings.filterwarnings("ignore")

    log_dir  = _RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{problem_name}_{variant_name}_run{run_id:02d}.log"

    class _FileSink:
        def __init__(self, path):  self.f = open(path, "w", encoding="utf-8")
        def write(self, t):        self.f.write(t); self.f.flush()
        def flush(self):           self.f.flush()
        def __enter__(self):       return self
        def __exit__(self, *_):    self.f.close()

    cfg = VARIANT_CONFIGS[variant_name]
    with _FileSink(log_file) as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            print(f"[{datetime.now()}] 开始: {problem_name}-{variant_name}-Run{run_id} (seed={seed})")
            problem     = PROBLEMS[problem_name]()
            initial_pop = generate_initial_population(problem, POP_SIZE, seed)
            callback    = DataCollectorCallback()

            algorithm = cfg["cls"](
                pop_size=POP_SIZE,
                niche_num=3,
                mutation_rate=1.0,
                inherit_rate=1.0,
                activate_method=True,
                cluster_method="kmeans",
                X=initial_pop,
                seed=seed,
                **cfg["kwargs"],
            )

            start = time.time()
            try:
                result  = minimize(
                    problem, algorithm,
                    termination=get_termination("n_gen", N_GEN),
                    seed=seed,
                    callback=callback,
                    verbose=False,
                )
                runtime = time.time() - start
                algorithm.print_selector_stats()

                run_result = RunResult(
                    problem_name=problem_name,
                    method_name=variant_name,
                    run_id=run_id,
                    seed=seed,
                    generations=callback.data,
                    final_best_X=result.X.copy() if result.X is not None else None,
                    final_best_F=float(result.F) if result.F is not None else None,
                    runtime=runtime,
                )
                print(f"[{datetime.now()}] 完成: F={run_result.final_best_F:.6e}, t={runtime:.1f}s")
            except Exception as exc:
                import traceback
                runtime = time.time() - start
                print(f"[{datetime.now()}] 错误: {exc}")
                traceback.print_exc()
                run_result = RunResult(
                    problem_name=problem_name,
                    method_name=variant_name,
                    run_id=run_id,
                    seed=seed,
                    runtime=runtime,
                )
    return run_result


# ============================================================================
# CSV 保存
# ============================================================================

def _save_to_csv(results: List[RunResult], problem_name: str, variant_name: str) -> None:
    """将多次运行结果汇总为 summary CSV（格式与 experiment_runner.py 一致）。"""
    valid = [r for r in results if r.generations]
    if not valid:
        print(f"  [{problem_name}-{variant_name}] 无有效数据，跳过")
        return

    n_gen = len(valid[0].generations)
    n_var = valid[0].generations[0].best_X.shape[0]
    out   = _RESULTS_DIR / f"{problem_name}_{variant_name}_summary.csv"

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "fes_mean", "best_f_mean"]
                   + [f"x{i+1}_mean" for i in range(n_var)])
        for gi in range(n_gen):
            fes_v, bf_v, bx_v = [], [], []
            for r in valid:
                if gi < len(r.generations):
                    g = r.generations[gi]
                    fes_v.append(g.fes)
                    bf_v.append(g.best_F)
                    bx_v.append(g.best_X.reshape(-1))
            if not fes_v:
                continue
            w.writerow(
                [gi + 1, float(np.mean(fes_v)), float(np.mean(bf_v))]
                + np.mean(np.stack(bx_v, axis=0), axis=0).tolist()
            )
    print(f"  [{problem_name}-{variant_name}] → {out.name}")


# ============================================================================
# 运行单个版本
# ============================================================================

def run_variant(variant_name: str) -> None:
    """并行运行一个版本的全部 (问题 × 次数) 实验并保存 CSV。"""
    cfg = VARIANT_CONFIGS[variant_name]
    print(f"\n{'='*65}")
    print(f"  版本: {variant_name}  —  {cfg['desc']}")
    print(f"  问题: {list(PROBLEMS.keys())}")
    print(f"  runs={N_RUNS}, pop={POP_SIZE}, gen={N_GEN}")
    print(f"{'='*65}")

    tasks = [
        (prob, run_id, seed, variant_name)
        for prob in PROBLEMS
        for run_id, seed in enumerate(RANDOM_SEEDS)
    ]
    n_workers = min(len(tasks), mp.cpu_count(), 32)
    print(f"  并行进程: {n_workers}，总任务: {len(tasks)}\n")

    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(_RESULTS_DIR, variant_name),
    ) as pool:
        all_results = pool.map(_run_single, tasks)

    # 按问题分组保存 CSV
    by_prob: Dict[str, List[RunResult]] = {}
    for r in all_results:
        by_prob.setdefault(r.problem_name, []).append(r)

    print(f"\n  保存 CSV：")
    for pname, rs in by_prob.items():
        _save_to_csv(rs, pname, variant_name)

    # 控制台摘要
    print(f"\n  ── 数值摘要 [{variant_name}] ──")
    for pname, rs in by_prob.items():
        valid = [r for r in rs if r.final_best_F is not None]
        if valid:
            vals = [r.final_best_F for r in valid]
            print(
                f"  {pname:<12} "
                f"mean={np.mean(vals):.4e}  std={np.std(vals):.4e}  "
                f"min={np.min(vals):.4e}  [{len(valid)}/{N_RUNS}]"
            )


# ============================================================================
# 主函数
# ============================================================================

def main():
    global _RESULTS_DIR
    _RESULTS_DIR = EXISTING_RESULTS_DIR
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="运行 Bandit 局部搜索版本对比实验",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--versions",
        type=str,
        default=",".join(VARIANT_CONFIGS.keys()),
        help=(
            "指定要运行的版本，逗号分隔，默认全部运行。\n"
            "可选值: QL-V2, UCB-V2\n"
            "示例:   --versions QL-V2,UCB-V2"
        ),
    )
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    unknown  = [v for v in versions if v not in VARIANT_CONFIGS]
    if unknown:
        print(f"[错误] 未知版本: {unknown}，可选: {list(VARIANT_CONFIGS.keys())}")
        return

    print(f"\n结果目录: {_RESULTS_DIR}")
    print(f"可选版本: {list(VARIANT_CONFIGS.keys())}")
    print(f"计划运行版本: {versions}\n")

    for v in versions:
        run_variant(v)

    print(f"\n{'='*65}")
    print(f"全部完成！CSV 已写入: {_RESULTS_DIR}")
    print("下一步：运行 python evaluate_metrics.py 查看对比图")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
