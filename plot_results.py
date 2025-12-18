#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据results目录中的CSV文件绘制对比图
横坐标：fes_mean
纵坐标：best_f_mean
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 从 experiment_runner 导入实验参数
from experiment_runner import POP_SIZE, N_GEN

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 结果目录
RESULTS_DIR = Path(__file__).parent / "experiment_results"

# 最大函数评估次数
MAX_FES = POP_SIZE * N_GEN

# 测试函数列表
TEST_FUNCTIONS = ["F2", "F3", "F4","Ackley","Griewank"]

# 方法列表及其显示名称和颜色
METHODS = {
    "GA": {"name": "GA", "color": "blue", "linestyle": "-"},
    "HA_kmeans": {"name": "HA_kmeans", "color": "red", "linestyle": "-"},
    "HA_meanshift": {"name": "HA_meanshift", "color": "green", "linestyle": "-"},
    "HA_dbscan": {"name": "HA_dbscan", "color": "orange", "linestyle": "-"},
    # "HA_original": {"name": "HA_original", "color": "purple", "linestyle": "-"}
}

def plot_function_comparison(function_name):
    """
    为指定测试函数绘制四种方法的对比图
    
    Args:
        function_name: 测试函数名称，如 "F2", "F3", "F4"
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_key, method_info in METHODS.items():
        csv_file = RESULTS_DIR / f"{function_name}_{method_key}_summary.csv"
        
        if not csv_file.exists():
            print(f"警告: 文件不存在 {csv_file}")
            continue
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 提取需要的列
            fes_mean = df['fes_mean'].values
            best_f_mean = df['best_f_mean'].values
            
            # 绘制曲线
            ax.plot(
                fes_mean,
                best_f_mean,
                label=method_info["name"],
                color=method_info["color"],
                linestyle=method_info["linestyle"],
                linewidth=2,
                marker='o',
                markersize=4
            )
            
        except Exception as e:
            print(f"读取文件 {csv_file} 时出错: {e}")
            continue
    
    # 设置图表属性
    ax.set_xlabel("fes_mean", fontsize=12)
    ax.set_ylabel("best_f_mean", fontsize=12)
    ax.set_title(f"对比图 - {function_name}", fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # 使用对数坐标，因为best_f_mean通常是很小的值
    ax.set_xlim(left=0, right=MAX_FES)  # 设置 x 轴最大值为 POP_SIZE * N_GEN
    
    # 保存图表
    output_file = RESULTS_DIR / f"{function_name}_comparison.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"图表已保存到: {output_file}")
    
    plt.close(fig)


def main():
    """主函数：为所有测试函数绘制对比图"""
    print("=" * 60)
    print("开始绘制对比图")
    print("=" * 60)
    
    if not RESULTS_DIR.exists():
        print(f"错误: 结果目录不存在: {RESULTS_DIR}")
        return
    
    for func_name in TEST_FUNCTIONS:
        print(f"\n正在处理: {func_name}")
        plot_function_comparison(func_name)
    
    print("\n" + "=" * 60)
    print("所有图表绘制完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

