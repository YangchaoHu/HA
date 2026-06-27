"""
compare_nsga3_vs_ha.py
----------------------
对比实验：NSGA-III 基准 vs NSHA 的 6 个变体，逐项验证三项多样性改动的效果。

变体：
    NSGA-III    — pymoo 内置基准
    NSHA-Base   — 原始 NSHA（无任何改动）
    NSHA+Nadir  — 仅启用改动一：动态 nadir 估计
    NSHA+Theta  — 仅启用改动二：pbi_theta 从 5.0 降至 1.0
    NSHA+Inject — 仅启用改动三：覆盖率触发 + 定向注入
    NSHA+All    — 三项改动全部启用

输出结构：<run_dir>/<problem>/NSHA_<variant>_vs_NSHA_Base.png（每问题 4 张）
指标：IGD（越小越好，兼顾收敛性与分布性）/ FEs
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
# 实验配置
# ============================================================
POP_SIZE   = 50
N_GEN      = 30
SEEDS      = [42, 2024, 2025]

HA_LOCAL_SEARCH_METHOD = "rbf"    # HA 局部搜索方法
HA_SCALARIZATION       = "pbi"    # HA 标量化策略
HA_PBI_THETA_BASE      = 5.0      # 改动二前的原始值
HA_PBI_THETA_NEW       = 1.0      # 改动二：降低 PBI 惩罚系数

# 每个问题的解析 Pareto 前沿生成函数
def _pf_zdt1(n: int = 500) -> np.ndarray:
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])

def _pf_zdt2(n: int = 500) -> np.ndarray:
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1.0 - f1 ** 2])

def _pf_zdt3(n: int = 500) -> np.ndarray:
    """ZDT3 Pareto 前沿（不连续的 5 段）"""
    # 精确的 5 段区间
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
    # ZDT4 真实 PF 形状与 ZDT1 相同
    f1 = np.linspace(0, 1, n)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])

def _pf_zdt6(n: int = 500) -> np.ndarray:
    # ZDT6: f1 = 1 - exp(-4*x1)*sin^6(6*pi*x1)，在 x1 ∈ [0,1] 取得最小
    # PF 近似：f1 取 [0.2807753..., 1]
    # 精确下界可用 scipy 求，这里用密集均匀采样 + 筛选
    x1 = np.linspace(0.0, 1.0, 5000)
    f1 = 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)
    # g=1 时 f1 最小值约 0.2807
    f1_min = f1.min()
    f1_pf  = np.linspace(f1_min, 1.0, n)
    f2_pf  = 1.0 - f1_pf ** 2
    return np.column_stack([f1_pf, f2_pf])

PROBLEMS = [
    ("ZDT1", ZDT1Problem,  {"n_var": 10}, _pf_zdt1),
    ("ZDT2", ZDT2Problem,  {"n_var": 10}, _pf_zdt2),
    ("ZDT3", ZDT3Problem,  {"n_var": 10}, _pf_zdt3),
    ("ZDT4", ZDT4Problem,  {"n_var": 10}, _pf_zdt4),
    ("ZDT6", ZDT6Problem,  {"n_var": 10}, _pf_zdt6),
]

# ============================================================
# 辅助函数
# ============================================================

class _DevNull:
    """静默输出流，抑制算法运行时的冗余 print。"""
    def write(self, *a, **k): pass
    def flush(self): pass


class _Tee:
    """同时写入终端与日志文件。"""
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


RESULT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "NSHA_VS_NSGAIII",
)


def _front0(F: np.ndarray) -> np.ndarray:
    """从最终种群目标值中提取非支配 front 0。"""
    nds = NonDominatedSorting()
    fronts = nds.do(F)
    return F[fronts[0]]


def run_nsga3(problem_cls, problem_kwargs: dict, seed: int) -> Tuple[float, int]:
    """
    运行 pymoo 内置 NSGA-III，返回 (IGD_front0, FEs)。

    使用与 HA_NSGA3 相同的 Das-Dennis 参考方向（n_partitions 使 H ≈ POP_SIZE）。
    """
    problem = problem_cls(**problem_kwargs)

    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=POP_SIZE - 1)
    algo = NSGA3(ref_dirs=ref_dirs, pop_size=POP_SIZE, seed=seed)

    real_stdout = sys.stdout
    sys.stdout = _DevNull()
    try:
        res = minimize(
            problem,
            algo,
            termination=get_termination("n_gen", N_GEN),
            seed=seed,
            verbose=False,
            save_history=False,
        )
    finally:
        sys.stdout = real_stdout

    F_final = np.asarray(res.pop.get("F"))
    F0 = _front0(F_final)
    return F0, problem.fes


def _run_ha(problem_cls, problem_kwargs: dict, seed: int,
            pbi_theta: float = HA_PBI_THETA_BASE,
            use_dynamic_nadir: bool = False,
            use_coverage_injection: bool = False) -> Tuple[np.ndarray, int]:
    """通用 HA_NSGA3 运行器，支持三项改动的独立开关，返回 (F_front0, FEs)。"""
    problem = problem_cls(**problem_kwargs)
    algo = HA_NSGA3(
        method=HA_LOCAL_SEARCH_METHOD,
        pop_size=POP_SIZE,
        niche_num=4,
        mutation_rate=0.2,
        activate_method=True,
        niche_strategy="ref_dirs",
        pbi_theta=pbi_theta,
        scalarization=HA_SCALARIZATION,
        use_dynamic_nadir=use_dynamic_nadir,
        use_coverage_injection=use_coverage_injection,
        seed=seed,
    )
    real_stdout = sys.stdout
    sys.stdout = _DevNull()
    try:
        res = minimize(
            problem,
            algo,
            termination=get_termination("n_gen", N_GEN),
            seed=seed,
            verbose=False,
            save_history=False,
            copy_algorithm=False,
        )
    finally:
        sys.stdout = real_stdout
    F_final = np.asarray(res.pop.get("F"))
    return _front0(F_final), problem.fes


# ── 6 个变体的 runner 包装 ────────────────────────────────────────────────────

def run_ha_base(prob_cls, prob_kw, seed):
    return _run_ha(prob_cls, prob_kw, seed,
                   pbi_theta=HA_PBI_THETA_BASE,
                   use_dynamic_nadir=False,
                   use_coverage_injection=False)

def run_ha_nadir(prob_cls, prob_kw, seed):
    """仅改动一：动态 nadir 估计"""
    return _run_ha(prob_cls, prob_kw, seed,
                   pbi_theta=HA_PBI_THETA_BASE,
                   use_dynamic_nadir=True,
                   use_coverage_injection=False)

def run_ha_theta(prob_cls, prob_kw, seed):
    """仅改动二：降低 PBI theta"""
    return _run_ha(prob_cls, prob_kw, seed,
                   pbi_theta=HA_PBI_THETA_NEW,
                   use_dynamic_nadir=False,
                   use_coverage_injection=False)

def run_ha_inject(prob_cls, prob_kw, seed):
    """仅改动三：覆盖率触发注入"""
    return _run_ha(prob_cls, prob_kw, seed,
                   pbi_theta=HA_PBI_THETA_BASE,
                   use_dynamic_nadir=False,
                   use_coverage_injection=True)

def run_ha_all(prob_cls, prob_kw, seed):
    """三项改动全部启用"""
    return _run_ha(prob_cls, prob_kw, seed,
                   pbi_theta=HA_PBI_THETA_NEW,
                   use_dynamic_nadir=True,
                   use_coverage_injection=True)


# 算法注册表：(显示名, runner函数)
ALGO_RUNNERS = [
    ("NSGA-III",   run_nsga3),
    ("NSHA-Base",  run_ha_base),
    ("NSHA+Nadir", run_ha_nadir),
    ("NSHA+Theta", run_ha_theta),
    ("NSHA+Inject",run_ha_inject),
    ("NSHA+All",   run_ha_all),
]

ALGO_STYLES = {
    "NSGA-III":   dict(color="tab:blue",   marker="o", s=28),
    "NSHA-Base":  dict(color="tab:orange", marker="s", s=28),
    "NSHA+Nadir": dict(color="tab:green",  marker="^", s=28),
    "NSHA+Theta": dict(color="tab:red",    marker="D", s=22),
    "NSHA+Inject":dict(color="tab:purple", marker="v", s=28),
    "NSHA+All":   dict(color="tab:brown",  marker="*", s=40),
}

# 两两对比：(改动变体, 基线)，每个问题生成 4 张图
COMPARISON_PAIRS = [
    ("NSHA+Nadir",  "NSHA-Base"),
    ("NSHA+Theta",  "NSHA-Base"),
    ("NSHA+Inject", "NSHA-Base"),
    ("NSHA+All",    "NSHA-Base"),
]


def main() -> None:
    """运行对比实验，结果写入 NSHA_VS_NSGAIII/<日期时间>/。"""
    run_tag = datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(RESULT_ROOT, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "run.log")
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
    """主实验：对比、汇总、绘图，全部写入 out_dir。"""
    print("=" * 70)
    print("多样性改动隔离对比实验")
    print(f"输出目录: {out_dir}")
    print(f"pop={POP_SIZE}, gen={N_GEN}, seeds={SEEDS}")
    print(f"参与变体: {[name for name, _ in ALGO_RUNNERS]}")
    print(f"HA 局部搜索: {HA_LOCAL_SEARCH_METHOD!r}  标量化: {HA_SCALARIZATION!r}")
    print(f"pbi_theta: Base={HA_PBI_THETA_BASE}  New={HA_PBI_THETA_NEW}")
    print("=" * 70)

    config_path = os.path.join(out_dir, "config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"pop_size={POP_SIZE}\n")
        f.write(f"n_gen={N_GEN}\n")
        f.write(f"seeds={SEEDS}\n")
        f.write(f"algorithms={[n for n,_ in ALGO_RUNNERS]}\n")
        f.write(f"problems={[p[0] for p in PROBLEMS]}\n")
        f.write(f"pbi_theta_base={HA_PBI_THETA_BASE}\n")
        f.write(f"pbi_theta_new={HA_PBI_THETA_NEW}\n")
    print(f"配置已保存: {config_path}")

    RecordType = Dict[str, Dict[str, Dict[str, list]]]
    records: RecordType = {}

    for prob_name, prob_cls, prob_kw, pf_func in PROBLEMS:
        pf = pf_func()
        records[prob_name] = {
            name: {"igd": [], "fes": [], "F0_last": None}
            for name, _ in ALGO_RUNNERS
        }

        print(f"\n{'='*60}")
        print(f"  Problem: {prob_name}  (n_var={prob_kw['n_var']}, "
              f"pop={POP_SIZE}, gen={N_GEN})")
        print(f"{'='*60}")

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

    # ============================================================
    # 打印汇总表并写入 summary.txt / summary.csv
    # ============================================================
    COL_W = 13
    header = (f"{'Problem':<8} {'Algorithm':<12} "
              f"{'IGD mean':>{COL_W}} {'IGD std':>10} "
              f"{'FEs mean':>10} {'FEs std':>8}")
    sep = "-" * len(header)
    print(f"\n{'Summary':^{len(header)}}")
    print("=" * len(header))
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
            print(f"{prob_name:<8} {algo_name:<12} "
                  f"{igd_mean:>{COL_W}.4e} {igd_std:>10.4e} "
                  f"{fes_mean:>10.0f} {fes_std:>8.0f}")
            summary_lines.append(
                f"{prob_name},{algo_name},{igd_mean:.6e},{igd_std:.6e},"
                f"{fes_mean:.0f},{fes_std:.0f}"
            )
        print()

    summary_txt = os.path.join(out_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(header + "\n" + sep + "\n")
        for line in summary_lines[1:]:
            parts = line.split(",")
            if len(parts) >= 6:
                f.write(
                    f"{parts[0]:<8} {parts[1]:<12} "
                    f"{float(parts[2]):>{COL_W}.4e} {float(parts[3]):>10.4e} "
                    f"{float(parts[4]):>10.0f} {float(parts[5]):>8.0f}\n"
                )
            else:
                f.write(line + "\n")
    summary_csv = os.path.join(out_dir, "summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"汇总已保存: {summary_txt}")
    print(f"汇总已保存: {summary_csv}")

    # ============================================================
    # 可视化：按问题划分子文件夹，每个文件夹 4 张两两对比图
    # ============================================================
    print(f"\n{'='*60}")
    print("  生成两两对比图（改动变体 vs NSHA-Base）")
    print(f"{'='*60}")

    for prob_name, _, _, pf_func in PROBLEMS:
        prob_dir = os.path.join(out_dir, prob_name.lower())
        os.makedirs(prob_dir, exist_ok=True)
        pf = pf_func(n=400)

        for variant_name, base_name in COMPARISON_PAIRS:
            fig, ax = plt.subplots(figsize=(7, 5.5))

            # True PF
            ax.plot(pf[:, 0], pf[:, 1], "k--", lw=1.4, alpha=0.55,
                    label="True PF", zorder=1)

            # NSHA-Base
            base_style = ALGO_STYLES[base_name]
            base_F0    = records[prob_name][base_name]["F0_last"]
            base_igds  = records[prob_name][base_name]["igd"]
            if base_F0 is not None and len(base_F0) > 0:
                ax.scatter(
                    base_F0[:, 0], base_F0[:, 1],
                    c=base_style["color"], marker=base_style["marker"],
                    s=base_style["s"],
                    alpha=0.75, edgecolors="black", linewidths=0.25, zorder=2,
                    label=(f"{base_name}  "
                           f"IGD={np.mean(base_igds):.3e}±{np.std(base_igds):.1e}"),
                )

            # 改动变体
            var_style = ALGO_STYLES[variant_name]
            var_F0    = records[prob_name][variant_name]["F0_last"]
            var_igds  = records[prob_name][variant_name]["igd"]
            if var_F0 is not None and len(var_F0) > 0:
                ax.scatter(
                    var_F0[:, 0], var_F0[:, 1],
                    c=var_style["color"], marker=var_style["marker"],
                    s=var_style["s"],
                    alpha=0.80, edgecolors="black", linewidths=0.25, zorder=3,
                    label=(f"{variant_name}  "
                           f"IGD={np.mean(var_igds):.3e}±{np.std(var_igds):.1e}"),
                )

            ax.set_title(
                f"{prob_name}: {variant_name} vs {base_name}\n"
                f"(pop={POP_SIZE}, gen={N_GEN}, {len(SEEDS)} seeds)",
                fontsize=11,
            )
            ax.set_xlabel("f₁", fontsize=10)
            ax.set_ylabel("f₂", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()

            safe_variant = variant_name.replace("+", "_")
            safe_base    = base_name.replace("-", "_")
            img_name     = f"{safe_variant}_vs_{safe_base}.png"
            img_path     = os.path.join(prob_dir, img_name)
            fig.savefig(img_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"  [对比图] {prob_name}/{img_name}")

    print(f"\n全部结果目录: {out_dir}")


if __name__ == "__main__":
    main()
