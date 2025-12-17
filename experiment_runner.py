"""
experiment_runner.py
--------------------

实验脚本：对比标准 GA 和三种 HA 变体（kmeans, meanshift, dbscan）的性能。

测试问题：F2Problem, F3Problem, F4Problem
实验参数：pop_size=100, n_gen=30, 独立运行 30 次

使用 multiprocessing 并行运行，结果保存到 HA/results/ 目录。
"""

import os
import sys
import csv
import pickle
import warnings
import contextlib
import multiprocessing as mp
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# 线程数/并行设置，避免 OpenBLAS 和多进程创建过多线程
# ============================================================================

# 限制底层 BLAS / OpenMP 线程数，防止 "Thread creation failed"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from numpy.typing import NDArray

# pymoo 相关导入
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.core.population import Population

# 本地模块导入
from problem import F2Problem, F3Problem, F4Problem
from ha import HA

# ============================================================================
# 配置参数
# ============================================================================

# 实验参数
POP_SIZE = 100
N_GEN = 30
N_RUNS = 30
RANDOM_SEEDS = list(range(42, 42 + N_RUNS))  # 30 个不同的随机种子

# 问题配置
PROBLEMS = {
    "F2": lambda: F2Problem(n_var=10, m=5, xl=0.0, xu=np.pi),
    "F3": lambda: F3Problem(n_var=10, xl=-100.0, xu=100.0),
    "F4": lambda: F4Problem(n_var=10, xl=-100.0, xu=100.0),
}

# 算法配置
METHODS = {
    "GA": "standard",
    "HA_kmeans": "kmeans",
    "HA_meanshift": "meanshift",
    "HA_dbscan": "dbscan",
}

# 输出目录
RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class GenerationData:
    """单代数据"""
    gen: int
    fes: int  # 到当前代为止的函数评估次数（function evaluations）
    X: NDArray  # 所有个体的决策变量 (pop_size, n_var)
    F: NDArray  # 所有个体的适应度 (pop_size, 1)
    best_X: NDArray  # 当代最优个体
    best_F: float  # 当代最优适应度


@dataclass
class RunResult:
    """单次运行结果"""
    problem_name: str
    method_name: str
    run_id: int
    seed: int
    generations: List[GenerationData] = field(default_factory=list)
    final_best_X: Optional[NDArray] = None
    final_best_F: Optional[float] = None
    runtime: float = 0.0


# ============================================================================
# 回调类：记录每代数据
# ============================================================================

class DataCollectorCallback(Callback):
    """
    回调类，用于在每一代结束时收集种群数据。
    """
    
    def __init__(self):
        super().__init__()
        self.data: List[GenerationData] = []
    
    def notify(self, algorithm):
        """每代结束时调用"""
        pop = algorithm.pop
        
        X = pop.get("X").copy()
        F = pop.get("F").copy()
        # 从 problem 中读取当前累计的函数评估次数（在各 Problem 类中维护）
        fes = int(getattr(algorithm.problem, "fes", 0))
        
        # 找到当代最优
        best_idx = np.argmin(F)
        best_X = X[best_idx].copy()
        best_F = float(F[best_idx])
        
        gen_data = GenerationData(
            gen=algorithm.n_gen,
            fes=fes,
            X=X,
            F=F,
            best_X=best_X,
            best_F=best_F
        )
        self.data.append(gen_data)


# ============================================================================
# 初始种群生成
# ============================================================================

def generate_initial_population(
    problem,
    pop_size: int,
    seed: int
) -> NDArray:
    """
    生成初始种群。
    
    Args:
        problem: pymoo 问题对象
        pop_size: 种群大小
        seed: 随机种子
        
    Returns:
        NDArray: 初始种群 (pop_size, n_var)
    """
    rng = np.random.default_rng(seed)
    xl = np.array(problem.xl)
    xu = np.array(problem.xu)
    return rng.uniform(xl, xu, (pop_size, problem.n_var))


# ============================================================================
# 算法运行函数
# ============================================================================

def run_standard_ga(
    problem,
    initial_pop: NDArray,
    n_gen: int,
    seed: int
) -> Tuple[Any, List[GenerationData]]:
    """
    运行标准 GA 算法。
    """
    callback = DataCollectorCallback()
    
    # 创建初始 Population 对象
    pop = Population.new("X", initial_pop)
    
    algorithm = GA(
        pop_size=len(initial_pop),
        sampling=pop,
        eliminate_duplicates=True,
    )
    
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        callback=callback,
        verbose=False
    )
    
    return result, callback.data


def run_ha(
    problem,
    initial_pop: NDArray,
    n_gen: int,
    seed: int,
    cluster_method: str
) -> Tuple[Any, List[GenerationData]]:
    """
    运行 HA 算法。
    """
    callback = DataCollectorCallback()
    
    algorithm = HA(
        method="L-BFGS-B",
        pop_size=len(initial_pop),
        niche_num=3,
        mutation_rate=0.8,
        inherit_rate=0.5,
        activate_method=True,
        cluster_method=cluster_method,
        X=initial_pop.copy(),
    )
    
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        callback=callback,
        verbose=False
    )
    
    return result, callback.data


# ============================================================================
# 单次实验运行
# ============================================================================

def run_single_experiment(args: Tuple) -> RunResult:
    """
    运行单次实验（用于并行）。
    
    Args:
        args: (problem_name, method_name, run_id, seed)
        
    Returns:
        RunResult: 运行结果
    """
    problem_name, method_name, run_id, seed = args
    
    # 创建日志文件
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{problem_name}_{method_name}_run{run_id:02d}.log"
    
    # 重定向 stdout 和 stderr 到日志文件
    with open(log_file, 'w') as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            return _run_experiment_internal(problem_name, method_name, run_id, seed)


def _run_experiment_internal(
    problem_name: str,
    method_name: str,
    run_id: int,
    seed: int
) -> RunResult:
    """
    实际运行实验的内部函数。
    """
    import time
    
    # 忽略警告
    warnings.filterwarnings("ignore")
    
    print(f"[{datetime.now()}] 开始运行: {problem_name} - {method_name} - Run {run_id} (Seed={seed})")
    
    # 创建问题
    problem = PROBLEMS[problem_name]()
    
    # 生成初始种群（同一种子下所有方法使用相同初始种群）
    initial_pop = generate_initial_population(problem, POP_SIZE, seed)
    
    start_time = time.time()
    
    try:
        if method_name == "GA":
            result, gen_data = run_standard_ga(problem, initial_pop, N_GEN, seed)
        else:
            cluster_method = METHODS[method_name]
            result, gen_data = run_ha(problem, initial_pop, N_GEN, seed, cluster_method)
        
        runtime = time.time() - start_time
        
        run_result = RunResult(
            problem_name=problem_name,
            method_name=method_name,
            run_id=run_id,
            seed=seed,
            generations=gen_data,
            final_best_X=result.X.copy() if result.X is not None else None,
            final_best_F=float(result.F) if result.F is not None else None,
            runtime=runtime
        )
        
        print(f"[{datetime.now()}] 完成: {problem_name} - {method_name} - Run {run_id}")
        print(f"  最终最优值: {run_result.final_best_F:.6e}")
        print(f"  运行时间: {runtime:.2f}s")
        
    except Exception as e:
        print(f"[{datetime.now()}] 错误: {problem_name} - {method_name} - Run {run_id}")
        print(f"  异常: {e}")
        import traceback
        traceback.print_exc()
        
        run_result = RunResult(
            problem_name=problem_name,
            method_name=method_name,
            run_id=run_id,
            seed=seed,
            runtime=time.time() - start_time
        )
    
    return run_result


# ============================================================================
# 结果保存与加载
# ============================================================================

def save_results(results: List[RunResult], problem_name: str):
    """
    （已弃用）保存实验结果到 pickle 文件。
    
    按问题名称保存，每个问题一个文件，包含所有方法的所有运行结果。
    
    Args:
        results: 所有运行结果列表
        problem_name: 问题名称
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 按方法分组
    results_by_method = {}
    for r in results:
        if r.method_name not in results_by_method:
            results_by_method[r.method_name] = []
        results_by_method[r.method_name].append(r)
    
    # 保存为 pickle 文件
    output_file = RESULTS_DIR / f"{problem_name}_results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results_by_method, f)
    
    print(f"结果已保存到: {output_file}")


def load_results(problem_name: str) -> Dict[str, List[RunResult]]:
    """
    加载实验结果。
    
    Args:
        problem_name: 问题名称
        
    Returns:
        Dict[str, List[RunResult]]: 按方法名分组的结果
    """
    input_file = RESULTS_DIR / f"{problem_name}_results.pkl"
    with open(input_file, 'rb') as f:
        return pickle.load(f)


def save_results_to_csv(
    results: List[RunResult],
    problem_name: str
) -> None:
    """
    将同一 problem 的所有方法的 30 次运行结果汇总为 CSV。
    
    对于每个 (problem_name, method_name)：
        - 以“代”为单位，对 30 次运行取平均：
            * 平均 FEs（函数评估次数）
            * 平均最优适应度 best_F
            * 平均最优个体参数 best_X 各维度
        - 保存到: results/{problem_name}_{method_name}_summary.csv
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 按方法分组
    results_by_method: Dict[str, List[RunResult]] = {}
    for r in results:
        results_by_method.setdefault(r.method_name, []).append(r)
    
    for method_name, method_results in results_by_method.items():
        # 只保留有 generations 的结果
        method_results = [r for r in method_results if r.generations]
        if not method_results:
            continue
        
        # 假设所有运行的代数一致，使用第一条作为模板
        n_gen = len(method_results[0].generations)
        n_var = method_results[0].generations[0].best_X.shape[0]
        
        # 构造 CSV 路径与表头
        output_file = RESULTS_DIR / f"{problem_name}_{method_name}_summary.csv"
        header = (
            ["generation", "fes_mean", "best_f_mean"]
            + [f"x{i+1}_mean" for i in range(n_var)]
        )
        
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            # 逐代汇总 30 次运行的平均值
            for gen_idx in range(n_gen):
                fes_vals = []
                best_f_vals = []
                best_x_list = []
                
                for r in method_results:
                    if gen_idx < len(r.generations):
                        g = r.generations[gen_idx]
                        fes_vals.append(g.fes)
                        best_f_vals.append(g.best_F)
                        best_x_list.append(g.best_X.reshape(-1))
                
                if not fes_vals:
                    continue
                
                fes_mean = float(np.mean(fes_vals))
                best_f_mean = float(np.mean(best_f_vals))
                best_x_mean = np.mean(np.stack(best_x_list, axis=0), axis=0)
                
                row = [gen_idx + 1, fes_mean, best_f_mean] + best_x_mean.tolist()
                writer.writerow(row)
        
        print(f"CSV 汇总已保存到: {output_file}")


# ============================================================================
# 数据分析函数
# ============================================================================

def compute_convergence_curve(
    results: List[RunResult]
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    计算收敛曲线的均值和标准差。
    
    Args:
        results: 同一方法的多次运行结果
        
    Returns:
        Tuple: (generations, mean_best_f, std_best_f)
    """
    # 提取每次运行每代的最优值
    all_best_f = []
    
    for r in results:
        if r.generations:
            best_f_per_gen = [g.best_F for g in r.generations]
            all_best_f.append(best_f_per_gen)
    
    if not all_best_f:
        return np.array([]), np.array([]), np.array([])
    
    # 转换为数组
    all_best_f = np.array(all_best_f)  # (n_runs, n_gen)
    
    generations = np.arange(1, all_best_f.shape[1] + 1)
    mean_best_f = np.mean(all_best_f, axis=0)
    std_best_f = np.std(all_best_f, axis=0)
    
    return generations, mean_best_f, std_best_f


def print_summary(results_by_method: Dict[str, List[RunResult]], problem_name: str):
    """
    打印实验结果摘要。
    """
    print(f"\n{'='*60}")
    print(f"实验结果摘要: {problem_name}")
    print(f"{'='*60}")
    
    for method_name, results in results_by_method.items():
        valid_results = [r for r in results if r.final_best_F is not None]
        if valid_results:
            final_values = [r.final_best_F for r in valid_results]
            mean_f = np.mean(final_values)
            std_f = np.std(final_values)
            min_f = np.min(final_values)
            max_f = np.max(final_values)
            mean_time = np.mean([r.runtime for r in valid_results])
            
            print(f"\n{method_name}:")
            print(f"  成功运行: {len(valid_results)}/{len(results)}")
            print(f"  最终最优值: {mean_f:.6e} ± {std_f:.6e}")
            print(f"  最小值: {min_f:.6e}, 最大值: {max_f:.6e}")
            print(f"  平均运行时间: {mean_time:.2f}s")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：运行所有实验。
    """
    print(f"实验开始时间: {datetime.now()}")
    print(f"配置: pop_size={POP_SIZE}, n_gen={N_GEN}, n_runs={N_RUNS}")
    print(f"问题: {list(PROBLEMS.keys())}")
    print(f"方法: {list(METHODS.keys())}")
    cpu_count = mp.cpu_count()
    # 为了避免线程/进程数过多导致资源错误，限制实际 worker 数量
    n_workers = min(8, cpu_count)
    print(f"CPU 核心数: {cpu_count}")
    print(f"实际使用的并行进程数: {n_workers}")
    print(f"结果目录: {RESULTS_DIR}")
    
    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 构建所有实验任务
    all_tasks = []
    for problem_name in PROBLEMS.keys():
        for method_name in METHODS.keys():
            for run_id, seed in enumerate(RANDOM_SEEDS):
                all_tasks.append((problem_name, method_name, run_id, seed))
    
    print(f"\n总任务数: {len(all_tasks)}")
    print(f"开始并行运行（进程数 = {n_workers}）...")
    
    with mp.Pool(processes=n_workers) as pool:
        all_results = pool.map(run_single_experiment, all_tasks)
    
    # 按问题分组并保存结果（CSV 汇总）
    results_by_problem = {}
    for r in all_results:
        if r.problem_name not in results_by_problem:
            results_by_problem[r.problem_name] = []
        results_by_problem[r.problem_name].append(r)
    
    for problem_name, results in results_by_problem.items():
        # 保存为 CSV（每个方法一个 summary 文件）
        save_results_to_csv(results, problem_name)
        
        # 仍然在终端打印数值摘要，便于快速查看
        results_by_method: Dict[str, List[RunResult]] = {}
        for r in results:
            results_by_method.setdefault(r.method_name, []).append(r)
        print_summary(results_by_method, problem_name)
    
    print(f"\n实验结束时间: {datetime.now()}")
    print(f"所有结果已保存到: {RESULTS_DIR}")


# ============================================================================
# 数据读取示例
# ============================================================================

def example_load_and_plot():
    """
    数据读取示例：加载保存的文件并计算平均收敛曲线。
    
    使用方法：
        python experiment_runner.py --plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要安装 matplotlib 来绘制图表: pip install matplotlib")
        return
    
    print("=" * 60)
    print("数据读取示例")
    print("=" * 60)
    
    for problem_name in PROBLEMS.keys():
        result_file = RESULTS_DIR / f"{problem_name}_results.pkl"
        
        if not result_file.exists():
            print(f"结果文件不存在: {result_file}")
            continue
        
        print(f"\n加载结果: {problem_name}")
        results_by_method = load_results(problem_name)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {
            "GA": "blue",
            "HA_kmeans": "red",
            "HA_meanshift": "green",
            "HA_dbscan": "orange"
        }
        
        for method_name, results in results_by_method.items():
            gens, mean_f, std_f = compute_convergence_curve(results)
            
            if len(gens) == 0:
                continue
            
            color = colors.get(method_name, "black")
            
            # 绘制均值曲线
            ax.plot(gens, mean_f, label=method_name, color=color, linewidth=2)
            
            # 绘制标准差阴影
            ax.fill_between(
                gens,
                mean_f - std_f,
                mean_f + std_f,
                alpha=0.2,
                color=color
            )
        
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Best Fitness", fontsize=12)
        ax.set_title(f"Convergence Curve - {problem_name}", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")  # 对数坐标更容易观察
        
        # 保存图表
        plot_file = RESULTS_DIR / f"{problem_name}_convergence.png"
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        print(f"收敛曲线已保存到: {plot_file}")
        
        plt.close(fig)
        
        # 打印数值摘要
        print_summary(results_by_method, problem_name)
    
    print("\n" + "=" * 60)
    print("示例数据访问代码:")
    print("=" * 60)
    print("""
# 加载结果
from experiment_runner import load_results, compute_convergence_curve

results_by_method = load_results("F2")

# 访问 GA 的结果
ga_results = results_by_method["GA"]
print(f"GA 运行次数: {len(ga_results)}")

# 访问第一次运行的数据
run0 = ga_results[0]
print(f"Run 0 - Seed: {run0.seed}")
print(f"Run 0 - 最终最优值: {run0.final_best_F}")

# 访问每一代的数据
for gen_data in run0.generations:
    print(f"Gen {gen_data.gen}: best_F = {gen_data.best_F:.6e}")
    # gen_data.X 包含所有个体的决策变量 (pop_size, n_var)
    # gen_data.F 包含所有个体的适应度 (pop_size, 1)
    # gen_data.best_X 是当代最优个体
    # gen_data.best_F 是当代最优适应度

# 计算收敛曲线
gens, mean_f, std_f = compute_convergence_curve(ga_results)
print(f"各代平均最优值: {mean_f}")
""")


# ============================================================================
# 入口点
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        example_load_and_plot()
    else:
        main()

