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
from problem import F2Problem, F3Problem, F4Problem, AckleyProblem, GriewankProblem

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
        optimal_f=-9.66,
        optimal_x=np.array([2.20, 1.57, 1.28, 1.11, 0.99, 
                           0.90, 0.83, 0.77, 0.72, 0.68]),
        problem_class=F2Problem
    ),
    "F3": ProblemInfo(
        name="F3 (Schaffer)",
        n_var=10,
        xl=-100.0,
        xu=100.0,
        optimal_f=0.0,
        optimal_x=np.zeros(10),
        problem_class=F3Problem
    ),
    "F4": ProblemInfo(
        name="F4 (Hybrid)",
        n_var=10,
        xl=-100.0,
        xu=100.0,
        optimal_f=0.0,
        optimal_x=np.zeros(10),
        problem_class=F4Problem
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
    "Griewank": ProblemInfo(
        name="Griewank",
        n_var=10,
        xl=-600.0,
        xu=600.0,
        optimal_f=0.0,
        optimal_x=np.zeros(10),
        problem_class=GriewankProblem
    ),
}

# 方法列表和颜色：使用高对比度调色板，增加线宽和标志区分度
METHODS = {
    "GA": {
        "color": "#000000",      # 黑色 (基准)
        "linestyle": "-", 
        "marker": "o", 
        "linewidth": 2.5,
        "label": "Standard GA"
    },
    "HA_kmeans": {
        "color": "#E31A1C",      # 鲜红色
        "linestyle": "-", 
        "marker": "s", 
        "linewidth": 2.5,
        "label": "HA (K-Means)"
    },
    "HA_meanshift": {
        "color": "#1F78B4",      # 鲜蓝色
        "linestyle": "--", 
        "marker": "^", 
        "linewidth": 2.5,
        "label": "HA (MeanShift)"
    },
    "HA_dbscan": {
        "color": "#33A02C",      # 鲜绿色
        "linestyle": "-.", 
        "marker": "D", 
        "linewidth": 2.5,
        "label": "HA (DBSCAN)"
    },
}

# FEs 截止值
MAX_FES = 5500

# 结果目录
RESULTS_DIR = Path(__file__).parent / "experiment_results"


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
    为单个问题绘制三个指标的收敛曲线。
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
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f'{problem_info.name} - Performance Metrics vs FEs', fontsize=14, fontweight='bold')
    
    metric_names = ["Optimality", "Accuracy", "Sensitivity"]
    metric_keys = ["optimality", "accuracy", "sensitivity"]
    
    for ax_idx, (ax, metric_name, metric_key) in enumerate(zip(axes, metric_names, metric_keys)):
        for method_name, style in METHODS.items():
            if method_name in all_metrics:
                metrics = all_metrics[method_name]
                # 过滤 FEs <= MAX_FES 的数据
                fes = metrics["fes"]
                mask = fes <= MAX_FES
                ax.plot(
                    fes[mask],
                    metrics[metric_key][mask],
                    color=style["color"],
                    linestyle=style.get("linestyle", "-"),
                    marker=style.get("marker", "o"),
                    markersize=6,
                    markevery=max(1, len(fes[mask]) // 8),  # 调整标记密度
                    linewidth=style.get("linewidth", 2.0),
                    label=style.get("label", method_name),
                    alpha=0.8
                )
        
        ax.set_xlabel("FEs (Function Evaluations)", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(0, MAX_FES)  # 设置 x 轴范围
        
        # Optimality 和 Accuracy 范围 [0, 1]
        if metric_key in ["optimality", "accuracy"]:
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(f"Relative {metric_name}", fontsize=11)
        
        # Sensitivity 使用对数坐标（如果值范围较大）
        if metric_key == "sensitivity":
            ax.set_ylabel("Sensitivity (log scale)", fontsize=11)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = RESULTS_DIR / f"{problem_name}_metrics.png"
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
    绘制所有问题的综合对比图（每个指标一张大图）。
    """
    print("\n绘制综合对比图...")
    
    metric_names = ["Optimality", "Accuracy", "Sensitivity"]
    metric_keys = ["optimality", "accuracy", "sensitivity"]
    
    for metric_name, metric_key in zip(metric_names, metric_keys):
        n_problems = len(PROBLEM_INFO)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()
        
        fig.suptitle(f'{metric_name} Comparison Across All Benchmarks', fontsize=14, fontweight='bold')
        
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
                    ax.plot(
                        fes[mask],
                        metrics[metric_key][mask],
                        color=style["color"],
                        linestyle=style.get("linestyle", "-"),
                        linewidth=style.get("linewidth", 2.0),
                        label=style.get("label", method_name),
                        alpha=0.8
                    )
            
            ax.set_xlabel("FEs", fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(problem_info.name, fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=8)
            ax.set_xlim(0, MAX_FES)  # 设置 x 轴范围
            
            if metric_key in ["optimality", "accuracy"]:
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel(f"Relative {metric_name}", fontsize=10)
            
            if metric_key == "sensitivity":
                ax.set_yscale('log')
                ax.set_ylabel("Sensitivity (log scale)", fontsize=10)
        
        # 隐藏多余的子图
        for idx in range(len(PROBLEM_INFO), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        output_path = RESULTS_DIR / f"all_{metric_key}_comparison.png"
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
    
    # 绘制每个问题的单独图表
    plot_all_problems()
    
    # 绘制综合对比图
    plot_combined_metrics()
