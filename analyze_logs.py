#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析实验日志，计算局部搜索的效率指标：f_reduction / fes_safe

对于每行日志：
  Cluster rep idx=X : F: f_old -> f_new, FEs: n, STATUS
  
计算：
  F下降 = f_old - f_new（如果是改进，为正值）
  效率 = f_reduction / fes_safe（F下降 / FEs消耗）
  不降反升检测：f_new > f_old
"""

import re
import os
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# 结果目录
RESULTS_DIR = Path(__file__).parent / "experiments_results" / "result_20260115_224425"
LOGS_DIR = RESULTS_DIR / "logs"

# 正则表达式匹配日志行
PATTERN = re.compile(
    r'Cluster rep idx=(\d+)\s*(?:\(Global Best\))?\s*:\s*'
    r'F:\s*([\d.e+\-]+)\s*->\s*([\d.e+\-]+),\s*'
    r'FEs:\s*(\d+),\s*'
    r'(IMPROVED|-)'
)

def parse_log_file(log_file: Path) -> List[Dict]:
    """
    解析单个日志文件，提取局部搜索的数据。
    
    Returns:
        List[Dict]: 每条记录包含 {
            'idx': int,
            'f_old': float,
            'f_new': float,
            'fes': int,
            'improved': bool,
            'f_reduction': float,  # F下降（正值表示改进）
            'efficiency': float,   # f_reduction / fes_safe
            'worsened': bool        # 是否不降反升（f_new > f_old）
        }
    """
    records = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = PATTERN.search(line)
            if match:
                idx = int(match.group(1))
                f_old = float(match.group(2))
                f_new = float(match.group(3))
                fes = int(match.group(4))
                improved = (match.group(5) == "IMPROVED")
                
                # 计算F下降
                if f_new < f_old:
                    f_reduction = f_old - f_new  # 正值表示改进
                    worsened = False
                elif f_new > f_old:
                    f_reduction = 0.0  # 不降反升
                    worsened = True
                else:
                    f_reduction = 0.0  # 没有变化
                    worsened = False
                
                # 计算效率：f_reduction / fes_safe
                fes_safe = max(fes, 1)
                efficiency = f_reduction / fes_safe if fes_safe > 0 else 0.0
                
                records.append({
                    'idx': idx,
                    'f_old': f_old,
                    'f_new': f_new,
                    'fes': fes,
                    'fes_used': fes_safe,
                    'improved': improved,
                    'f_reduction': f_reduction,
                    'efficiency': efficiency,
                    'worsened': worsened  # 是否不降反升
                })
    
    return records


def analyze_method(method_name: str) -> Dict:
    """
    分析某个方法在所有问题上的表现。
    
    Args:
        method_name: 方法名称（如 "rbf", "Adam" 等）
        
    Returns:
        Dict: 统计信息
    """
    all_records = []
    problem_stats = defaultdict(list)
    
    # 遍历所有日志文件
    for log_file in LOGS_DIR.glob(f"*_{method_name}_run*.log"):
        # 提取问题名称
        match = re.match(r'(\w+)_' + re.escape(method_name) + r'_run', log_file.name)
        if match:
            problem_name = match.group(1)
            records = parse_log_file(log_file)
            
            for record in records:
                record['problem'] = problem_name
                all_records.append(record)
                problem_stats[problem_name].append(record)
    
    # 计算统计信息
    if not all_records:
        return None
    
    # 总体统计 - 只保留效率相关
    efficiencies = [r['efficiency'] for r in all_records]
    worsened_count = sum(1 for r in all_records if r['worsened'])
    
    # 计算统计值
    avg_efficiency = statistics.mean(efficiencies) if efficiencies else 0.0
    std_efficiency = statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0.0
    median_efficiency = statistics.median(efficiencies) if efficiencies else 0.0
    
    stats = {
        'method': method_name,
        'total_searches': len(all_records),
        'worsened_count': worsened_count,
        'worsened_ratio': worsened_count / len(all_records) if all_records else 0.0,
        'avg_efficiency': avg_efficiency,
        'std_efficiency': std_efficiency,
        'median_efficiency': median_efficiency,
        'problem_stats': {}
    }
    
    # 每个问题的统计
    for problem_name, records in problem_stats.items():
        if records:
            prob_efficiencies = [r['efficiency'] for r in records]
            prob_worsened_count = sum(1 for r in records if r['worsened'])
            
            stats['problem_stats'][problem_name] = {
                'count': len(records),
                'worsened_count': prob_worsened_count,
                'avg_efficiency': statistics.mean(prob_efficiencies) if prob_efficiencies else 0.0,
                'median_efficiency': statistics.median(prob_efficiencies) if prob_efficiencies else 0.0,
            }
    
    return stats


def print_summary(all_stats: List[Dict]):
    """打印统计摘要"""
    print("=" * 80)
    print("局部搜索效率分析（f_reduction / fes_safe）")
    print("=" * 80)
    
    for stats in all_stats:
        if stats is None:
            continue
            
        print(f"\n方法: {stats['method']}")
        print(f"  总搜索次数: {stats['total_searches']}")
        print(f"  不降反升次数: {stats['worsened_count']} ({stats['worsened_ratio']*100:.1f}%)")
        print(f"  平均效率: {stats['avg_efficiency']:.6e} ± {stats['std_efficiency']:.6e}")
        print(f"  中位数效率: {stats['median_efficiency']:.6e}")
        
        print(f"  各问题表现:")
        for problem_name, prob_stats in stats['problem_stats'].items():
            print(f"    {problem_name}: "
                  f"次数={prob_stats['count']}, "
                  f"不降反升={prob_stats['worsened_count']}, "
                  f"平均效率={prob_stats['avg_efficiency']:.6e}, "
                  f"中位数效率={prob_stats['median_efficiency']:.6e}")


def save_detailed_report(all_stats: List[Dict], output_file: Path):
    """保存详细报告到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("局部搜索效率分析报告（f_reduction / fes_safe）\n")
        f.write("=" * 80 + "\n\n")
        
        for stats in all_stats:
            if stats is None:
                continue
                
            f.write(f"\n方法: {stats['method']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"总搜索次数: {stats['total_searches']}\n")
            f.write(f"不降反升次数: {stats['worsened_count']} ({stats['worsened_ratio']*100:.1f}%)\n")
            f.write(f"平均效率: {stats['avg_efficiency']:.6e} ± {stats['std_efficiency']:.6e}\n")
            f.write(f"中位数效率: {stats['median_efficiency']:.6e}\n\n")
            
            f.write("各问题详细统计:\n")
            for problem_name, prob_stats in stats['problem_stats'].items():
                f.write(f"  {problem_name}:\n")
                f.write(f"    搜索次数: {prob_stats['count']}\n")
                f.write(f"    不降反升次数: {prob_stats['worsened_count']}\n")
                f.write(f"    平均效率: {prob_stats['avg_efficiency']:.6e}\n")
                f.write(f"    中位数效率: {prob_stats['median_efficiency']:.6e}\n")
            f.write("\n")


def main():
    """主函数"""
    print("开始分析日志文件...")
    print(f"日志目录: {LOGS_DIR}")
    
    # 获取所有方法（排除GA）
    log_files = list(LOGS_DIR.glob("*_run*.log"))
    methods = set()
    
    for log_file in log_files:
        # 提取方法名称（格式：Problem_Method_run*.log）
        parts = log_file.stem.split('_')
        if len(parts) >= 3 and parts[1] != "GA":
            methods.add(parts[1])
    
    methods = sorted(methods)
    print(f"\n找到的方法: {methods}")
    
    # 分析每个方法
    all_stats = []
    for method in methods:
        print(f"\n分析 {method}...")
        stats = analyze_method(method)
        if stats:
            all_stats.append(stats)
    
    # 打印摘要
    print_summary(all_stats)
    
    # 保存详细报告
    report_file = RESULTS_DIR / "local_search_efficiency_report.txt"
    save_detailed_report(all_stats, report_file)
    print(f"\n详细报告已保存到: {report_file}")


if __name__ == "__main__":
    main()

