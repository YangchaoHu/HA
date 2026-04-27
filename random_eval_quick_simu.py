"""
在 QuickSimu1 / QuickSimu2 的决策空间内各随机采样 5 个点并调用仿真评估。

用法（需在能启动 MAPDL 的环境中运行）:
    python random_eval_quick_simu.py
    python random_eval_quick_simu.py --seed 123
"""

from __future__ import annotations

import argparse

import numpy as np

from problem import QuickSimu1Problem, QuickSimu2Problem


def sample_box(rng: np.random.Generator, xl: np.ndarray, xu: np.ndarray, n: int) -> np.ndarray:
    """在 [xl, xu] 上均匀独立采样 n 个点，形状 (n, n_var)。"""
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    u = rng.uniform(size=(n, xl.size))
    return xl + u * (xu - xl)


def eval_points(name: str, problem, X: np.ndarray) -> None:
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    out: dict = {}
    problem._evaluate(X, out)
    F = np.asarray(out["F"]).reshape(-1)
    for i, (x, f) in enumerate(zip(X, F)):
        print(f"  #{i+1}  x={np.array2string(x, precision=6, separator=', ')}  F={f:.6e}")
    print(f"  (problem.fes={problem.fes})")


def main() -> None:
    parser = argparse.ArgumentParser(description="随机评估 QuickSimu1 / QuickSimu2 各 5 点")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--n", type=int, default=5, help="每个问题采样点数")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n = max(1, args.n)

    p1 = QuickSimu1Problem()
    X1 = sample_box(rng, p1.xl, p1.xu, n)
    eval_points("QuickSimu1 (len1, width1, width2) → min |disp_x|", p1, X1)

    p2 = QuickSimu2Problem()
    X2 = sample_box(rng, p2.xl, p2.xu, n)
    eval_points("QuickSimu2 (len1, len2) → min |disp_y|", p2, X2)


if __name__ == "__main__":
    main()
