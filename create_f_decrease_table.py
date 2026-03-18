"""
从 efficiency_table.csv 读取各方法在不同函数下的平均效率
并生成排名图表
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# 设置字体 - 使用英文避免乱码
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# CSV文件路径
CSV_FILE = Path(__file__).parent / "experiments_results" / "result_20260115_224425" / "efficiency_table.csv"

def read_csv_data():
    """从CSV文件读取数据"""
    data = defaultdict(dict)
    problems = []
    
    if not CSV_FILE.exists():
        print(f"错误: CSV文件不存在: {CSV_FILE}")
        return data, problems
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # 读取表头，获取所有方法名
        methods = [col for col in reader.fieldnames if col != '问题']
        
        # 读取数据行
        for row in reader:
            problem = row['问题']
            problems.append(problem)
            
            for method in methods:
                value_str = row.get(method, '0.0')
                try:
                    value = float(value_str)
                    data[method][problem] = value
                except ValueError:
                    data[method][problem] = 0.0
    
    return data, problems

def plot_ranking_chart(data, problems):
    """绘制排名图表，每个问题一个子图，显示各方法的排名"""
    methods = sorted(data.keys())
    
    if not methods:
        print("警告: 没有找到任何方法数据，跳过图表绘制")
        return
    
    if not problems:
        print("警告: 没有找到任何问题数据，跳过图表绘制")
        return
    
    n_problems = len(problems)
    
    # 创建子图：2行3列
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # 为每个问题绘制排名图表
    for idx, problem in enumerate(problems):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # 获取该问题下所有方法的值
        method_values = []
        for method in methods:
            value = data[method].get(problem, 0.0)
            method_values.append((method, value))
        
        # 按值从大到小排序（效率高的在前）
        method_values.sort(key=lambda x: x[1], reverse=True)
        
        methods_sorted = [m[0] for m in method_values]
        values_sorted = [m[1] for m in method_values]
        ranks = list(range(1, len(methods_sorted) + 1))  # 排名：1, 2, 3, ...
        
        # 绘制柱状图
        bars = ax.bar(range(len(methods_sorted)), values_sorted, color='steelblue', alpha=0.7)
        
        # 设置y轴为对数刻度（因为数值范围很大）
        ax.set_yscale('log')
        
        # 在每个柱子上标注排名（仅排名，不显示具体数值）
        for i, (bar, val, rank) in enumerate(zip(bars, values_sorted, ranks)):
            if val > 0:
                height = bar.get_height()
                # 标注排名（大字体，加粗）
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'#{rank}', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold', color='red')
        
        ax.set_xlabel('Method', fontsize=10)
        ax.set_ylabel('Average Efficiency (F Reduction / FEs)', fontsize=10)
        ax.set_title(f'{problem} - Method Ranking', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(range(len(methods_sorted)))
        ax.set_xticklabels(methods_sorted, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 设置总标题，并调整布局以确保标题可见
    plt.suptitle('Average Efficiency Ranking of Methods Across Different Functions', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    
    output_file = CSV_FILE.parent / "efficiency_ranking.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"排名图表已保存到: {output_file}")

if __name__ == "__main__":
    print(f"从CSV文件读取数据: {CSV_FILE}")
    data, problems = read_csv_data()
    
    if not data:
        print("错误: 无法读取数据，请检查CSV文件")
    else:
        print(f"成功读取 {len(problems)} 个问题，{len(data)} 个方法的数据")
        plot_ranking_chart(data, problems)

