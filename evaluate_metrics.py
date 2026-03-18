"""
evaluate_metrics.py
-------------------

根据实验结果计算三个评估指标并绘图：
1. Optimality (最优性): 1 - |f_o - f̂_o| / |f̄ - f_|
2. Accuracy (准确性): 1 - ||x_o - x̂_o|| / ||x̄ - x_||
3. Sensitivity (灵敏度): |Δf̂| / |Δx̂| * ||x̄ - x_|| / |f̄ - f_|

横坐标: FEs (函数评估次数)
纵坐标: 指标值
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Type
from dataclasses import dataclass

# 导入问题类用于数值评估 Sensitivity
from problem import (
    F2Problem, F3Problem, F4Problem,
    RastriginProblem, GriewankProblem, AckleyProblem
)

# ============================================================================
# 问题定义：理论最优值和搜索边界
# ============================================================================

@dataclass
class ProblemInfo:
    """问题的理论信息"""
    name: str
    n_var: int
    xl: float  # 下界
    xu: float  # 上界
    optimal_f: float  # 理论最优适应度
    optimal_x: np.ndarray  # 理论最优解
    problem_class: Type  # 问题类，用于数值评估


# 定义各个问题的理论信息
PROBLEM_INFO = {
    "F2": ProblemInfo(
        name="F2 (Michalewicz)",
        n_var=10,
        xl=0.0,
        xu=np.pi,
        optimal_f=-9.66,  # F2 (Michalewicz) 的理论最优值（D=10, m=10）
        optimal_x=np.array([2.2029, 1.5708, 1.2850, 1.9231, 1.7205, 1.5708, 1.4544, 1.7561, 1.6557, 1.5708]),  # 近似最优解
        problem_class=F2Problem
    ),
    "F3": ProblemInfo(
        name="F3",
        n_var=10,
        xl=-100.0,
        xu=100.0,
        optimal_f=0.0,  # F3 在原点处达到最小值
        optimal_x=np.zeros(10),
        problem_class=F3Problem
    ),
    "F4": ProblemInfo(
        name="F4 (Mixed)",
        n_var=10,
        xl=-100.0,
        xu=100.0,
        optimal_f=0.0,  # F4 在原点处达到最小值（近似）
        optimal_x=np.zeros(10),
        problem_class=F4Problem
    ),
    "Rastrigin": ProblemInfo(
        name="Rastrigin",
        n_var=10,
        xl=-5.12,
        xu=5.12,
        optimal_f=0.0,
        optimal_x=np.zeros(10),
        problem_class=RastriginProblem
    ),
    "Griewank": ProblemInfo(
        name="Griewank",
        n_var=10,
        xl=-600.0,
        xu=600.0,
        optimal_f=0.0,
        optimal_x=np.zeros(10),
        problem_class=GriewankProblem
    ),
    "Ackley": ProblemInfo(
        name="Ackley",
        n_var=10,
        xl=-32.768,
        xu=32.768,
        optimal_f=0.0,
        optimal_x=np.zeros(10),
        problem_class=AckleyProblem
    ),
}

# 方法列表和颜色：使用高对比度调色板，增加线宽和标志区分度
METHODS = {
    "GA": {
        "color": "#0000FF",      # 蓝色
        "linestyle": "-", 
        "marker": "o", 
        "markersize": 6,
        "linewidth": 2.0,
        "label": "GA"
    },
    "Nelder-Mead": {
        "color": "#1F77B4",      # 深蓝色
        "linestyle": "-", 
        "marker": "o", 
        "markersize": 6,
        "linewidth": 2.0,
        "label": "Nelder-Mead"
    },
    "rbf": {
        "color": "#FF7F0E",      # 橙色
        "linestyle": "--", 
        "marker": "s", 
        "markersize": 6,
        "linewidth": 2.0,
        "label": "RBF"
    },
    "gp": {
        "color": "#E377C2",      # 粉红色
        "linestyle": "--", 
        "marker": "X", 
        "markersize": 6,
        "linewidth": 2.0,
        "label": "GP"
    },
    "history-ladder": {
        "color": "#17BECF",      # 青色
        "linestyle": "-.", 
        "marker": "+", 
        "markersize": 7,
        "linewidth": 2.0,
        "label": "History-Ladder"
    },
    "Adam": {
        "color": "#2CA02C",      # 绿色
        "linestyle": "-", 
        "marker": "^", 
        "markersize": 7,
        "linewidth": 2.0,
        "label": "Adam"
    },
    "Sophia": {
        "color": "#D62728",      # 红色
        "linestyle": "-.", 
        "marker": "D", 
        "markersize": 6,
        "linewidth": 2.0,
        "label": "Sophia"
    },
    "Lion": {
        "color": "#9467BD",      # 紫色
        "linestyle": "-", 
        "marker": "v", 
        "markersize": 6,
        "linewidth": 2.0,
        "label": "Lion"
    },
    "AdamW": {
        "color": "#8C564B",      # 棕色
        "linestyle": "--", 
        "marker": "p", 
        "markersize": 6,
        "linewidth": 2.0,
        "label": "AdamW"
    },
    "L-BFGS-B": {
        "color": "#000000",      # 黑色
        "linestyle": "-", 
        "marker": "*", 
        "markersize": 7,
        "linewidth": 2.0,
        "label": "L-BFGS-B"
    },
}

# FEs 截止值
MAX_FES = 1500

# 结果目录 - 指向指定的实验结果目录
RESULTS_DIR = Path(__file__).parent / "experiments_results" / "result_20260115_224425"

# ============================================================================
# 数据读取
# ============================================================================

def read_summary_csv(problem_name: str, method_name: str) -> Optional[Dict]:
    """读取汇总 CSV 文件。"""
    csv_path = RESULTS_DIR / f"{problem_name}_{method_name}_summary.csv"
    
    if not csv_path.exists():
        return None
    
    data = {
        "generation": [],
        "fes": [],
        "best_f": [],
        "best_x": []
    }
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["generation"].append(int(row["generation"]))
            data["fes"].append(float(row["fes_mean"]))
            data["best_f"].append(float(row["best_f_mean"]))
            
            x_mean = []
            for i in range(1, 11):
                key = f"x{i}_mean"
                if key in row:
                    x_mean.append(float(row[key]))
            data["best_x"].append(np.array(x_mean))
    
    return data


# ============================================================================
# 指标计算
# ============================================================================

def compute_metrics_over_generations(data: Dict, problem_info: ProblemInfo) -> Dict:
    """
    计算每一代的三个指标。
    使用相对归一化，使图表更具可读性（指标从 0 附近开始，向 1 进化）。
    
    Returns:
        包含 fes, optimality, accuracy, sensitivity 数组的字典
    """
    n_gen = len(data["generation"])
    
    fes = np.array(data["fes"])
    optimality = np.zeros(n_gen)
    accuracy = np.zeros(n_gen)
    sensitivity = np.zeros(n_gen)
    
    # 理论最优
    f_hat_o = problem_info.optimal_f
    x_hat_o = problem_info.optimal_x
    
    # 初始状态（用于相对归一化）
    f_initial = data["best_f"][0]
    x_initial = data["best_x"][0]
    
    # 初始差距
    f_gap_initial = abs(f_initial - f_hat_o)
    x_dist_initial = np.linalg.norm(x_initial - x_hat_o)
    
    # 搜索空间范围（用于 Sensitivity）
    x_bar = np.full(problem_info.n_var, problem_info.xu)
    x_underline = np.full(problem_info.n_var, problem_info.xl)
    x_range_total = np.linalg.norm(x_bar - x_underline)
    
    # 创建问题实例用于数值评估 Sensitivity
    problem = problem_info.problem_class(n_var=problem_info.n_var)
    
    for i in range(n_gen):
        f_o = data["best_f"][i]
        x_o = data["best_x"][i]
        
        # 1. 相对 Optimality = 1 - (当前F差距 / 初始F差距)
        # 这样第一代是 0，完美达到最优是 1
        if f_gap_initial > 1e-12:
            opt = 1.0 - abs(f_o - f_hat_o) / f_gap_initial
            optimality[i] = max(0.0, min(1.0, opt))
        else:
            optimality[i] = 1.0
        
        # 2. 相对 Accuracy = 1 - (当前X距离 / 初始X距离)
        # 这样第一代是 0，完美达到最优是 1
        if x_dist_initial > 1e-12:
            acc = 1.0 - np.linalg.norm(x_o - x_hat_o) / x_dist_initial
            accuracy[i] = max(0.0, min(1.0, acc))
        else:
            accuracy[i] = 1.0
        
        # 3. Sensitivity = |Δf̂| / |Δx̂| * ||x̄ - x_|| / |f̄ - f_|
        # 使用全局范围作为分母以保持 Sensitivity 的绝对含义
        f_range_total = f_gap_initial if f_gap_initial > 1e-12 else 1.0
        
        if f_range_total > 1e-12:
            delta_x_norm = 0.01 * x_range_total
            direction = np.random.randn(problem_info.n_var)
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            delta_x = direction * delta_x_norm
            
            x_perturbed = np.clip(x_o + delta_x, problem_info.xl, problem_info.xu)
            
            out_original = {}
            out_perturbed = {}
            problem._evaluate(x_o.reshape(1, -1), out_original)
            problem._evaluate(x_perturbed.reshape(1, -1), out_perturbed)
            
            delta_f = abs(out_perturbed["F"][0] - out_original["F"][0])
            actual_delta_x = np.linalg.norm(x_perturbed - x_o)
            
            if actual_delta_x > 1e-12:
                sens = (delta_f / actual_delta_x) * (x_range_total / f_range_total)
                sensitivity[i] = sens
            else:
                sensitivity[i] = 0.0
        else:
            sensitivity[i] = 0.0
    
    return {
        "fes": fes,
        "optimality": optimality,
        "accuracy": accuracy,
        "sensitivity": sensitivity
    }


# ============================================================================
# 绘图函数
# ============================================================================

def plot_metrics_for_problem(problem_name: str, problem_info: ProblemInfo):
    """
    为单个问题绘制 Optimality 指标的收敛曲线。
    """
    # 收集所有方法的数据
    all_metrics = {}
    
    for method_name in METHODS.keys():
        data = read_summary_csv(problem_name, method_name)
        if data is not None:
            metrics = compute_metrics_over_generations(data, problem_info)
            all_metrics[method_name] = metrics
    
    if not all_metrics:
        print(f"警告: {problem_name} 没有可用数据")
        return
    
    # 创建图表 - 只绘制 Optimality
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'{problem_info.name} - Optimality vs FEs', fontsize=14, fontweight='bold')
    
    for method_name, style in METHODS.items():
        if method_name in all_metrics:
            metrics = all_metrics[method_name]
            # 过滤 FEs <= MAX_FES 的数据
            fes = metrics["fes"]
            mask = fes <= MAX_FES
            fes_filtered = fes[mask]
            n_points = len(fes_filtered)
            
            # 根据数据点数量调整标记密度
            # GA方法通常有30个点（每50 FEs一个），应该显示更多标记
            # 其他方法数据点更密集，可以显示较少标记
            if method_name == "GA":
                # GA方法：每2个点显示一个标记（约15个标记）
                markevery_val = max(1, 2)
            elif n_points <= 30:
                # 数据点较少的方法：每2个点显示一个标记
                markevery_val = max(1, 2)
            else:
                # 数据点较多的方法：每8个点显示一个标记
                markevery_val = max(1, n_points // 8)
            
            ax.plot(
                fes_filtered,
                metrics["optimality"][mask],
                color=style["color"],
                linestyle=style.get("linestyle", "-"),
                marker=style.get("marker", "o"),
                markersize=style.get("markersize", 6),
                markevery=markevery_val,
                linewidth=style.get("linewidth", 2.0),
                label=style.get("label", method_name),
                alpha=0.8
            )
    
    ax.set_xlabel("FEs (Function Evaluations)", fontsize=12)
    ax.set_ylabel("Relative Optimality", fontsize=12)
    ax.set_title("Optimality", fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(0, MAX_FES)  # 设置 x 轴范围
    ax.set_ylim(-0.05, 1.05)  # Optimality 范围 [0, 1]
    
    plt.tight_layout()
    
    # 保存图表
    output_path = RESULTS_DIR / f"{problem_name}_optimality.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存: {output_path}")


def plot_all_problems():
    """
    为所有问题绘制指标图。
    """
    print("=" * 60)
    print("绘制评估指标图表")
    print("=" * 60)
    
    for problem_name, problem_info in PROBLEM_INFO.items():
        print(f"\n处理: {problem_info.name}")
        plot_metrics_for_problem(problem_name, problem_info)
    
    print("\n" + "=" * 60)
    print("所有图表已生成完毕！")
    print("=" * 60)


def plot_combined_metrics():
    """
    绘制所有问题的 Optimality 综合对比图。
    """
    print("\n绘制综合对比图...")
    
    n_problems = len(PROBLEM_INFO)
    # 使用 2x3 布局，因为有6个问题（F2, F3, F4, Rastrigin, Griewank, Ackley）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()  # 展平为1D数组便于索引
    
    fig.suptitle(f'Optimality Comparison Across All Benchmarks', fontsize=14, fontweight='bold')
    
    for idx, (problem_name, problem_info) in enumerate(PROBLEM_INFO.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        for method_name, style in METHODS.items():
            data = read_summary_csv(problem_name, method_name)
            if data is not None:
                metrics = compute_metrics_over_generations(data, problem_info)
                # 过滤 FEs <= MAX_FES 的数据
                fes = metrics["fes"]
                mask = fes <= MAX_FES
                fes_filtered = fes[mask]
                n_points = len(fes_filtered)
                
                # 根据数据点数量调整标记密度
                if method_name == "GA":
                    # GA方法：每2个点显示一个标记
                    markevery_val = max(1, 2)
                elif n_points <= 30:
                    # 数据点较少的方法：每2个点显示一个标记
                    markevery_val = max(1, 2)
                else:
                    # 数据点较多的方法：每10个点显示一个标记
                    markevery_val = max(1, n_points // 10)
                
                ax.plot(
                    fes_filtered,
                    metrics["optimality"][mask],
                    color=style["color"],
                    linestyle=style.get("linestyle", "-"),
                    marker=style.get("marker", "o"),
                    markersize=style.get("markersize", 4),
                    markevery=markevery_val,
                    linewidth=style.get("linewidth", 2.0),
                    label=style.get("label", method_name),
                    alpha=0.8
                )
        
        ax.set_xlabel("FEs", fontsize=11)
        ax.set_ylabel("Relative Optimality", fontsize=11)
        ax.set_title(problem_info.name, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(0, MAX_FES)  # 设置 x 轴范围
        ax.set_ylim(-0.05, 1.05)  # Optimality 范围 [0, 1]
    
    plt.tight_layout()
    
    output_path = RESULTS_DIR / f"all_optimality_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"已保存: {output_path}")


# ============================================================================
# 入口点
# ============================================================================

if __name__ == "__main__":
    # 设置 matplotlib 字体
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制每个问题的 Optimality 图表
    plot_all_problems()
    
    # 绘制综合对比图
    plot_combined_metrics()
