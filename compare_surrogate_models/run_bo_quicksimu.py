"""
run_bo_quicksimu.py
-------------------

用主动学习贝叶斯优化（active_learning_bo.BayesianOptimizer）对
QuickSimu1 / QuickSimu2 运行完整流程。

真实仿真函数：
    QuickSimu1Problem / QuickSimu2Problem（来自 problem.py），
    以 volume 为优化目标，位移约束用惩罚函数折算进目标值。

结果输出：
    compare_surrogate_models/results_bo/<problem>_seed<seed>_bo_trace.csv

用法示例::

    # 完整运行（两个问题 × 三个 seed）
    python compare_surrogate_models/run_bo_quicksimu.py

    # 冒烟测试（QuickSimu1，单 seed，快速验证）
    python compare_surrogate_models/run_bo_quicksimu.py \\
        --problems QuickSimu1 --seeds 2026 --n-init 5 --max-iter 10 \\
        --ha-pop 10 --ha-gen 5

    # 使用 LCB 采集函数
    python compare_surrogate_models/run_bo_quicksimu.py --acquisition lcb
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from active_learning_bo import BayesianOptimizer  # noqa: E402

# ============================================================================
# 问题元数据（与 run_surrogate_ha_eval.py 保持一致）
# ============================================================================

PROBLEM_META: Dict[str, Dict] = {
    "QuickSimu1": {
        "xl": np.array([0.4,  0.01, 0.01], dtype=float),
        "xu": np.array([1.6,  0.08, 0.08], dtype=float),
        "disp_limit": 0.027867728805696976,
    },
    "QuickSimu2": {
        "xl": np.array([0.2, 0.2], dtype=float),
        "xu": np.array([1.6, 1.6], dtype=float),
        "disp_limit": 3.6164872159562756e-07,
    },
}

DEFAULT_PROBLEMS = ["QuickSimu1", "QuickSimu2"]
DEFAULT_SEEDS    = [2026, 2027, 2028]


# ============================================================================
# 真实仿真包装
# ============================================================================

_prob_cache: Dict[str, object] = {}


def _get_problem(name: str):
    """懒加载真实 pymoo Problem 对象（避免重复初始化 ANSYS）。"""
    if name not in _prob_cache:
        if name == "QuickSimu1":
            from problem import QuickSimu1Problem  # type: ignore
            _prob_cache[name] = QuickSimu1Problem()
        elif name == "QuickSimu2":
            from problem import QuickSimu2Problem  # type: ignore
            _prob_cache[name] = QuickSimu2Problem()
        else:
            raise ValueError(f"未知问题: {name}")
    return _prob_cache[name]


def make_sim_func(problem_name: str):
    """
    返回真实仿真函数：f(x) -> float。

    目标：最小化 volume（f_real）。
    约束：abs(max_disp) ≤ disp_limit，以惩罚项追加到目标值。
    惩罚系数：adaptive，与 ha_Nelder_Mead._local_search 策略一致。
    位移限值已体现在 problem._evaluate 输出的 G 中（G≤0 可行），此处无需再读 disp_limit。
    """
    def sim_func(x: np.ndarray) -> float:
        prob = _get_problem(problem_name)
        out: Dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob._evaluate(np.atleast_2d(x).astype(float), out)

        f_arr = np.asarray(out.get("F", [np.nan])).flatten()
        g_arr = np.asarray(out.get("G", [np.nan])).flatten()

        volume = float(f_arr[0]) if f_arr.size > 0 else float("nan")
        g_val  = float(g_arr[0]) if g_arr.size > 0 else float("nan")

        # 惩罚：若约束违反（g > 0），对目标值加惩罚
        cv = float(np.maximum(0.0, g_val)) if not np.isnan(g_val) else 0.0
        alpha = abs(volume) * 10 if abs(volume) > 1e-9 else 10.0
        return volume + alpha * cv

    return sim_func


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="主动学习 BO（HA 内层优化）对 QuickSimu1/2 的完整实验"
    )
    p.add_argument(
        "--problems", nargs="+",
        choices=DEFAULT_PROBLEMS,
        default=DEFAULT_PROBLEMS,
    )
    p.add_argument("--seeds",    nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--n-init",   type=int,   default=15,    help="LHS 初始样本数")
    p.add_argument("--max-iter", type=int,   default=60,    help="外层最大迭代次数（真实仿真次数）")
    p.add_argument(
        "--acquisition", choices=["ei", "lcb"], default="ei",
        help="采集函数：ei（期望改进，默认）或 lcb（置信下界）",
    )
    p.add_argument("--kappa",    type=float, default=2.0,   help="LCB 的 κ 参数")
    p.add_argument("--xi",       type=float, default=0.01,  help="EI 的 ξ 参数")
    p.add_argument("--patience", type=int,   default=8,     help="收敛容忍轮数")
    p.add_argument("--tol",      type=float, default=1e-7,  help="收敛容忍值")
    p.add_argument("--ha-pop",   type=int,   default=30,    help="内层 HA 种群大小")
    p.add_argument("--ha-gen",   type=int,   default=20,    help="内层 HA 迭代代数")
    p.add_argument(
        "--ha-method", default="L-BFGS-B",
        help="HA 局部搜索方法（L-BFGS-B / Nelder-Mead / gp）",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "results_bo",
    )
    return p.parse_args()


# ============================================================================
# 主流程
# ============================================================================

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 72
    print(sep)
    print(
        f"[{_ts()}] 启动 BO 实验\n"
        f"  problems  = {args.problems}\n"
        f"  seeds     = {args.seeds}\n"
        f"  n_init    = {args.n_init}  max_iter = {args.max_iter}\n"
        f"  acq       = {args.acquisition}  kappa={args.kappa}  xi={args.xi}\n"
        f"  patience  = {args.patience}  tol={args.tol}\n"
        f"  HA        = pop={args.ha_pop}  gen={args.ha_gen}  method={args.ha_method}\n"
        f"  output    = {args.output_dir}"
    )
    print(sep, flush=True)

    all_results = []

    for prob_name in args.problems:
        meta = PROBLEM_META[prob_name]
        sim  = make_sim_func(prob_name)

        for seed in args.seeds:
            print(f"\n{sep}")
            print(f"[{_ts()}] 开始 {prob_name} / seed={seed}", flush=True)
            print(sep)

            out_dir = args.output_dir / prob_name
            out_dir.mkdir(parents=True, exist_ok=True)
            # 每个 (problem, seed) 独立 CSV
            trace_path = out_dir / f"{prob_name}_seed{seed}_bo_trace.csv"
            # 若文件已存在则先删除（本次重跑）
            if trace_path.exists():
                trace_path.unlink()

            bo = BayesianOptimizer(
                sim_func    = sim,
                xl          = meta["xl"],
                xu          = meta["xu"],
                n_init      = args.n_init,
                max_iter    = args.max_iter,
                acquisition = args.acquisition,
                kappa       = args.kappa,
                xi          = args.xi,
                tol         = args.tol,
                patience    = args.patience,
                ha_pop_size = args.ha_pop,
                ha_n_gen    = args.ha_gen,
                ha_method   = args.ha_method,
                seed        = seed,
                output_dir  = out_dir,
                verbose     = True,
            )

            result = bo.run()

            # 把 bo_trace.csv 重命名为带 seed 的名字（BayesianOptimizer 固定输出 bo_trace.csv）
            generic = out_dir / "bo_trace.csv"
            if generic.exists() and not trace_path.exists():
                generic.rename(trace_path)

            all_results.append({
                "problem": prob_name,
                "seed":    seed,
                "best_f":  result["best_f"],
                "best_x":  result["best_x"].tolist(),
                "n_sim":   result["n_sim_calls"],
            })

            print(
                f"[{_ts()}] 完成 {prob_name}/seed={seed}  "
                f"best_f={result['best_f']:.6e}  "
                f"n_sim={result['n_sim_calls']}",
                flush=True,
            )

    # 汇总摘要
    print(f"\n{sep}")
    print(f"[{_ts()}] 全部完成，结果摘要：")
    for r in all_results:
        print(
            f"  {r['problem']:12s} seed={r['seed']}  "
            f"best_f={r['best_f']:.6e}  n_sim={r['n_sim']}"
        )
    print(sep, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
