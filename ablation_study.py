"""
ablation_study.py
-----------------
消融实验：分析 HA_NSGA3 两个核心组件的贡献。

消融组：
    - NSGA-III   : 纯 pymoo NSGA-III（基线）
    - HA-Full    : 完整 HA_NSGA3（RBF+PBI 局部搜索 + 自定义混合遗传算子）
    - HA-NoLS    : 关闭局部搜索（activate_method=False），保留自定义遗传算子
                   → 与 HA-Full 对比，揭示 _clustering_and_learning 的贡献
    - HA-StdGA   : 保留 RBF+PBI 局部搜索，但将 _inheritance_mo 替换为
                   pymoo NSGA-III 相同的遗传算子（二元锦标赛+标准 SBX）
                   → 与 HA-Full 对比，揭示 _inheritance_mo 的贡献

指标：IGD（越小越好）、FEs（函数评估次数）

设置：
    - 种群大小：50
    - 代数：30
    - 3 个随机种子取均值 ± 标准差

输出：
    图片与日志保存在 NSHA_VS_NSGAIII/<日期时间>/ 下，文件名以 ablation_ 为前缀。
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions

from problem import (
    ZDT1Problem, ZDT2Problem, ZDT3Problem, ZDT4Problem, ZDT6Problem
)
from ha_nsga3 import HA_NSGA3

# ============================================================
# 实验配置（与 compare_nsga3_vs_ha.py 保持一致）
# ============================================================
POP_SIZE = 50
N_GEN    = 30
SEEDS    = [42, 2024, 2025]

RESULT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "NSHA_VS_NSGAIII",
)


# ============================================================
# 真实 Pareto 前沿
# ============================================================

def _pf_zdt1(n: int = 500) -> np.ndarray:
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])

def _pf_zdt2(n: int = 500) -> np.ndarray:
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1.0 - f1 ** 2])

def _pf_zdt3(n: int = 500) -> np.ndarray:
    regions = [
        (0.0,       0.0830015),
        (0.1822287, 0.2577623),
        (0.4093136, 0.4538522),
        (0.6183967, 0.6525117),
        (0.8233317, 0.8518328),
    ]
    parts = []
    for lo, hi in regions:
        f1 = np.linspace(lo, hi, max(2, n // 5))
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
        parts.append(np.column_stack([f1, f2]))
    return np.vstack(parts)

def _pf_zdt4(n: int = 500) -> np.ndarray:
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])

def _pf_zdt6(n: int = 500) -> np.ndarray:
    x1 = np.linspace(0.0, 1.0, 5000)
    f1 = 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)
    f1_min = f1.min()
    f1_pf  = np.linspace(f1_min, 1.0, n)
    f2_pf  = 1.0 - f1_pf ** 2
    return np.column_stack([f1_pf, f2_pf])

PROBLEMS = [
    ("ZDT1", ZDT1Problem, {"n_var": 10}, _pf_zdt1),
    ("ZDT2", ZDT2Problem, {"n_var": 10}, _pf_zdt2),
    ("ZDT3", ZDT3Problem, {"n_var": 10}, _pf_zdt3),
    ("ZDT4", ZDT4Problem, {"n_var": 10}, _pf_zdt4),
    ("ZDT6", ZDT6Problem, {"n_var": 10}, _pf_zdt6),
]


# ============================================================
# 辅助工具
# ============================================================

class _DevNull:
    def write(self, *a, **k): pass
    def flush(self): pass


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def _front0(F: np.ndarray) -> np.ndarray:
    nds = NonDominatedSorting()
    fronts = nds.do(F)
    return F[fronts[0]]


# ============================================================
# 消融组 2：HA_StdGA
# 将 _inheritance_mo 替换为与 pymoo NSGA-III 相同的遗传算子：
#   - 父本选择：二元锦标赛（比较 front_rank，同 rank 比 d2）
#   - 交叉：标准 SBX（eta=15，无 fitness 偏向）
# ============================================================

class NSHA_StdGA(HA_NSGA3):
    """
    消融变体：仅将 _inheritance_mo 替换为 pymoo NSGA-III 相同的遗传算子。
    其余（RBF+PBI 局部搜索、NSGA-III 环境选择）与 NSHA-Full 完全一致。
    """

    def _inheritance_mo(  # type: ignore[override]
        self,
        offspring_size: int,
        pop: np.ndarray,
        F: np.ndarray,
        front_rank: np.ndarray,
        d2: np.ndarray,
    ) -> np.ndarray:
        """
        pymoo NSGA-III 风格遗传算子：
          1. 二元锦标赛选父本（front_rank 优先，同 rank 比 d2 小的胜）
          2. 标准 SBX（eta=15），两个子代随机取其一，无 fitness 偏向
        """
        n_pop = len(pop)
        eta   = 15
        offspring = np.zeros((offspring_size, self.dim))

        def _tournament(a: int, b: int) -> int:
            if front_rank[a] < front_rank[b]:
                return a
            if front_rank[a] > front_rank[b]:
                return b
            return a if d2[a] <= d2[b] else b

        def _sbx(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            child = np.empty(self.dim)
            for j in range(self.dim):
                y1, y2 = p1[j], p2[j]
                if abs(y1 - y2) < 1e-14:
                    child[j] = y1
                    continue
                yl, yu = min(y1, y2), max(y1, y2)
                u = self._rng.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))
                c_lo = 0.5 * ((yl + yu) - beta * (yu - yl))
                c_hi = 0.5 * ((yl + yu) + beta * (yu - yl))
                # 等概率选两个子代之一（无 fitness 偏向）
                child[j] = c_lo if self._rng.random() < 0.5 else c_hi
            return child

        for i in range(offspring_size):
            # 二元锦标赛选两个父本
            c1, c2 = self._rng.integers(n_pop, size=2)
            c3, c4 = self._rng.integers(n_pop, size=2)
            p1 = pop[_tournament(int(c1), int(c2))]
            p2 = pop[_tournament(int(c3), int(c4))]
            offspring[i] = _sbx(p1, p2)

        return np.clip(offspring, self.lb, self.ub)


# ============================================================
# Runners
# ============================================================

def run_nsga3(
    problem_cls, problem_kwargs: dict, seed: int
) -> Tuple[np.ndarray, int]:
    """pymoo 内置 NSGA-III（基线）。"""
    problem = problem_cls(**problem_kwargs)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=POP_SIZE - 1)
    algo = NSGA3(ref_dirs=ref_dirs, pop_size=POP_SIZE, seed=seed)

    real_stdout = sys.stdout
    sys.stdout = _DevNull()
    try:
        res = minimize(
            problem, algo,
            termination=get_termination("n_gen", N_GEN),
            seed=seed, verbose=False, save_history=False,
        )
    finally:
        sys.stdout = real_stdout

    F_final = np.asarray(res.pop.get("F"))
    return _front0(F_final), problem.fes


def _run_ha_nsga3(
    algo: HA_NSGA3,
    problem_cls, problem_kwargs: dict, seed: int
) -> Tuple[np.ndarray, int]:
    """通用 HA_NSGA3 系列运行器，直接传入已构造的算法实例。"""
    problem = problem_cls(**problem_kwargs)
    # algo 里的 problem 在 minimize 时会被重新 setup，problem 必须是新的
    real_stdout = sys.stdout
    sys.stdout = _DevNull()
    try:
        res = minimize(
            problem, algo,
            termination=get_termination("n_gen", N_GEN),
            seed=seed, verbose=False, save_history=False,
            copy_algorithm=False,
        )
    finally:
        sys.stdout = real_stdout

    F_final = np.asarray(res.pop.get("F"))
    return _front0(F_final), problem.fes


def run_nsha_full(
    problem_cls, problem_kwargs: dict, seed: int
) -> Tuple[np.ndarray, int]:
    """NSHA-Full：完整 HA_NSGA3（RBF+PBI + 自定义混合遗传算子）。"""
    algo = HA_NSGA3(
        method="rbf",
        pop_size=POP_SIZE,
        niche_num=4,
        mutation_rate=0.2,
        activate_method=True,
        niche_strategy="ref_dirs",
        pbi_theta=5.0,
        scalarization="pbi",
        seed=seed,
    )
    return _run_ha_nsga3(algo, problem_cls, problem_kwargs, seed)


def run_nsha_no_ls(
    problem_cls, problem_kwargs: dict, seed: int
) -> Tuple[np.ndarray, int]:
    """
    NSHA-NoLS：关闭局部搜索（activate_method=False）。
    与 NSHA-Full 对比，差值 = _clustering_and_learning 的贡献。
    """
    algo = HA_NSGA3(
        method="rbf",
        pop_size=POP_SIZE,
        niche_num=4,
        mutation_rate=0.2,
        activate_method=False,
        niche_strategy="ref_dirs",
        pbi_theta=5.0,
        scalarization="pbi",
        seed=seed,
    )
    return _run_ha_nsga3(algo, problem_cls, problem_kwargs, seed)


def run_nsha_std_ga(
    problem_cls, problem_kwargs: dict, seed: int
) -> Tuple[np.ndarray, int]:
    """
    NSHA-StdGA：保留 RBF+PBI 局部搜索，但将 _inheritance_mo
    替换为 pymoo NSGA-III 相同的遗传算子（二元锦标赛 + 标准 SBX）。
    与 NSHA-Full 对比，差值 = _inheritance_mo 的贡献。
    """
    algo = NSHA_StdGA(
        method="rbf",
        pop_size=POP_SIZE,
        niche_num=4,
        mutation_rate=0.2,
        activate_method=True,
        niche_strategy="ref_dirs",
        pbi_theta=5.0,
        scalarization="pbi",
        seed=seed,
    )
    return _run_ha_nsga3(algo, problem_cls, problem_kwargs, seed)


# ============================================================
# 算法注册表
# ============================================================

ALGO_RUNNERS: List[Tuple[str, object]] = [
    ("NSHA-Full",   run_nsha_full),
    ("NSHA-NoLS",   run_nsha_no_ls),
    ("NSHA-StdGA",  run_nsha_std_ga),
]

ALGO_STYLES: Dict[str, dict] = {
    "NSHA-Full":   dict(color="tab:orange", marker="s", s=40),
    "NSHA-NoLS":   dict(color="tab:green",  marker="^", s=40),
    "NSHA-StdGA":  dict(color="tab:red",    marker="D", s=40),
}

# 两组消融对比定义
ABLATION_PAIRS = [
    ("ablation1_ls",   "NSHA-Full", "NSHA-NoLS",
     "消融1：局部搜索贡献（NSHA-Full vs NSHA-NoLS）"),
    ("ablation2_ga",   "NSHA-Full", "NSHA-StdGA",
     "消融2：遗传算子贡献（NSHA-Full vs NSHA-StdGA）"),
]


# ============================================================
# 主实验
# ============================================================

def main() -> None:
    """运行消融实验，结果写入 NSHA_VS_NSGAIII/<日期时间>/。"""
    run_tag = datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(RESULT_ROOT, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "ablation_run.log")
    log_file = open(log_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, log_file)

    try:
        _run_experiment(out_dir)
    finally:
        sys.stdout = orig_stdout
        log_file.close()
        print(f"\n日志已保存: {log_path}")


def _run_experiment(out_dir: str) -> None:
    """主实验：运行所有消融组，生成汇总与图表。"""
    print("=" * 70)
    print("消融实验：NSHA 组件贡献分析")
    print(f"  NSHA-Full vs NSHA-NoLS  → _clustering_and_learning 贡献")
    print(f"  NSHA-Full vs NSHA-StdGA → _inheritance_mo 贡献")
    print(f"输出目录: {out_dir}")
    print(f"pop={POP_SIZE}, gen={N_GEN}, seeds={SEEDS}")
    print("=" * 70)

    # 写配置
    config_path = os.path.join(out_dir, "ablation_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"pop_size={POP_SIZE}\n")
        f.write(f"n_gen={N_GEN}\n")
        f.write(f"seeds={SEEDS}\n")
        f.write(f"algorithms={[name for name, _ in ALGO_RUNNERS]}\n")
        f.write(f"problems={[p[0] for p in PROBLEMS]}\n")
    print(f"配置已保存: {config_path}")

    RecordType = Dict[str, Dict[str, Dict[str, object]]]
    records: RecordType = {}

    for prob_name, prob_cls, prob_kw, pf_func in PROBLEMS:
        pf = pf_func()
        records[prob_name] = {
            name: {"igd": [], "fes": [], "F0_last": None}
            for name, _ in ALGO_RUNNERS
        }

        print(f"\n{'='*70}")
        print(f"  Problem: {prob_name}  (n_var={prob_kw['n_var']}, "
              f"pop={POP_SIZE}, gen={N_GEN})")
        print(f"{'='*70}")

        for seed in SEEDS:
            for algo_name, runner in ALGO_RUNNERS:
                try:
                    F0, fes = runner(prob_cls, prob_kw, seed)
                    igd = float(IGD(pf).do(F0))
                except Exception as e:
                    print(f"  [{algo_name} seed={seed}] ERROR: {e}")
                    F0, fes, igd = np.empty((0, 2)), 0, float("inf")

                records[prob_name][algo_name]["igd"].append(igd)
                records[prob_name][algo_name]["fes"].append(fes)
                records[prob_name][algo_name]["F0_last"] = F0
                print(f"  {algo_name:<12}  seed={seed}: "
                      f"IGD={igd:.4e}  FEs={fes}")

    # --------------------------------------------------------
    # 汇总表
    # --------------------------------------------------------
    n_algos = len(ALGO_RUNNERS)
    col_w = 14  # 算法列宽

    header = (f"{'Problem':<8} {'Algorithm':<{col_w}} "
              f"{'IGD mean':>12} {'IGD std':>10} "
              f"{'FEs mean':>10} {'FEs std':>8}")
    sep = "-" * (8 + col_w + 46)

    print(f"\n{'='*len(sep)}")
    print(f"{'消融实验 Summary':^{len(sep)}}")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)

    summary_lines: List[str] = [
        "Problem,Algorithm,IGD_mean,IGD_std,FEs_mean,FEs_std",
    ]
    for prob_name, _, _, _ in PROBLEMS:
        for algo_name, _ in ALGO_RUNNERS:
            igds = records[prob_name][algo_name]["igd"]
            fess = records[prob_name][algo_name]["fes"]
            igd_mean = np.mean(igds)
            igd_std  = np.std(igds)
            fes_mean = np.mean(fess)
            fes_std  = np.std(fess)
            line = (f"{prob_name:<8} {algo_name:<{col_w}} "
                    f"{igd_mean:>12.4e} {igd_std:>10.4e} "
                    f"{fes_mean:>10.0f} {fes_std:>8.0f}")
            print(line)
            summary_lines.append(
                f"{prob_name},{algo_name},{igd_mean:.6e},{igd_std:.6e},"
                f"{fes_mean:.0f},{fes_std:.0f}"
            )
        print()

    summary_txt = os.path.join(out_dir, "ablation_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(sep + "\n")
        for line in summary_lines[1:]:
            parts = line.split(",")
            if len(parts) >= 6:
                f.write(
                    f"{parts[0]:<8} {parts[1]:<{col_w}} "
                    f"{float(parts[2]):>12.4e} {float(parts[3]):>10.4e} "
                    f"{float(parts[4]):>10.0f} {float(parts[5]):>8.0f}\n"
                )
    summary_csv = os.path.join(out_dir, "ablation_summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"汇总已保存: {summary_txt}")
    print(f"汇总已保存: {summary_csv}")

    # --------------------------------------------------------
    # 可视化：每个问题 × 每个消融对比 → 各一张图
    # 共 5 问题 × 2 对比 = 10 张图，每张图两条曲线（NSHA-Full vs 另一方）
    # --------------------------------------------------------
    for prob_name, _, _, pf_func in PROBLEMS:
        pf = pf_func(n=400)

        for file_prefix, algo_a, algo_b, pair_title in ABLATION_PAIRS:
            fig_p, ax_p = plt.subplots(figsize=(7, 5))
            ax_p.plot(pf[:, 0], pf[:, 1], "k--", lw=1.2, alpha=0.5,
                      label="True PF", zorder=1)

            for algo_name in (algo_a, algo_b):
                style = ALGO_STYLES[algo_name]
                F0    = records[prob_name][algo_name]["F0_last"]
                igds  = records[prob_name][algo_name]["igd"]
                fess  = records[prob_name][algo_name]["fes"]
                if F0 is not None and len(F0) > 0:
                    ax_p.scatter(
                        F0[:, 0], F0[:, 1],
                        c=style["color"], marker=style["marker"], s=style["s"],
                        alpha=0.82, edgecolors="black", linewidths=0.3, zorder=2,
                        label=(f"{algo_name}  "
                               f"IGD={np.mean(igds):.3e}±{np.std(igds):.1e}  "
                               f"FEs={int(np.mean(fess))}"),
                    )

            ax_p.set_title(
                f"{prob_name} — {pair_title}\n"
                f"(pop={POP_SIZE}, gen={N_GEN}, {len(SEEDS)} seeds)",
                fontsize=10,
            )
            ax_p.set_xlabel("f₁", fontsize=10)
            ax_p.set_ylabel("f₂", fontsize=10)
            ax_p.legend(fontsize=8, loc="upper right")
            ax_p.grid(True, alpha=0.25)
            fig_p.tight_layout()

            save_path = os.path.join(out_dir, f"{file_prefix}_{prob_name}.png")
            fig_p.savefig(save_path, dpi=140, bbox_inches="tight")
            plt.close(fig_p)
            print(f"  [PF 图] 已保存: {save_path}")

    # --------------------------------------------------------
    # 合并大图：2 列（每列一个消融对比） × 5 行（每行一个问题）
    # --------------------------------------------------------
    n_pairs = len(ABLATION_PAIRS)
    fig, axes = plt.subplots(
        len(PROBLEMS), n_pairs,
        figsize=(6 * n_pairs, 3.8 * len(PROBLEMS)),
        squeeze=False,
    )
    fig.suptitle(
        f"NSHA 消融实验：局部搜索 & 遗传算子 贡献分析\n"
        f"(pop={POP_SIZE}, gen={N_GEN}, last seed={SEEDS[-1]})",
        fontsize=13, y=1.002,
    )

    for row, (prob_name, _, _, pf_func) in enumerate(PROBLEMS):
        pf = pf_func(n=300)
        for col, (_, algo_a, algo_b, pair_title) in enumerate(ABLATION_PAIRS):
            ax = axes[row][col]
            ax.plot(pf[:, 0], pf[:, 1], "k--", lw=1.0, alpha=0.45,
                    label="True PF")

            for algo_name in (algo_a, algo_b):
                style = ALGO_STYLES[algo_name]
                F0    = records[prob_name][algo_name]["F0_last"]
                igds  = records[prob_name][algo_name]["igd"]
                fess  = records[prob_name][algo_name]["fes"]
                if F0 is not None and len(F0) > 0:
                    ax.scatter(
                        F0[:, 0], F0[:, 1],
                        c=style["color"], marker=style["marker"], s=18,
                        alpha=0.85, edgecolors="none",
                        label=f"{algo_name} IGD={np.mean(igds):.3e}",
                    )

            ax.set_title(
                f"{prob_name}\n{pair_title}",
                fontsize=8,
            )
            ax.set_xlabel("f₁", fontsize=8)
            ax.set_ylabel("f₂", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5, loc="upper right")
            ax.grid(True, alpha=0.25)

    fig.tight_layout()
    combined_path = os.path.join(out_dir, "ablation_compare.png")
    fig.savefig(combined_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[合并大图] 已保存: {combined_path}")
    print(f"\n全部结果目录: {out_dir}")


if __name__ == "__main__":
    main()
