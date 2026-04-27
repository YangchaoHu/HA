"""
experiment_runner.py
--------------------

实验脚本：对比标准 GA 和三种 HA 变体（kmeans, meanshift, dbscan）的性能。

测试问题：F2Problem, F3Problem, F4Problem
实验参数：pop_size=100, n_gen=30, 独立运行 30 次

使用 multiprocessing 并行运行，结果保存到 HA/results/ 目录。
"""

import os
import re
import sys
import csv
import pickle
import argparse
import warnings
import contextlib
import multiprocessing as mp
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# 同时输出到文件和控制台的类
# ============================================================================

class TeeOutput:
    """
    将输出写入文件的类（正常运行时不输出到控制台）。
    支持同时处理 stdout 和 stderr 的重定向。

    实现 isatty()供 PyMAPDL / colorama 等库探测终端能力，避免在重定向流上
    调用 sys.stdout.isatty() 时因自定义对象缺少该方法而崩溃。
    关闭后若仍有写入（例如 logging 与异常收尾顺序问题），则回退到真实 stderr。
    """
    def __init__(self, file_path, mode='w'):
        self._path = file_path
        self.file = open(file_path, mode, encoding='utf-8')

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def write(self, text):
        if not self.file or self.file.closed:
            try:
                sys.__stderr__.write(text)
                sys.__stderr__.flush()
            except Exception:
                pass
            return
        self.file.write(text)
        self.file.flush()

    def flush(self):
        if self.file and not self.file.closed:
            self.file.flush()

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
from tqdm import tqdm

# pymoo 相关导入
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.core.population import Population

# 本地模块导入
from problem import F2Problem, F3Problem, F4Problem
from ha import HA
from ha_original import HA as HA_Original
from ha_Nelder_Mead import HA as HA_Nelder_Mead
from ha_bandit import HA_QL, HA_UCB

# ============================================================================
# 配置参数
# ============================================================================

# 实验参数
POP_SIZE = 50
N_GEN = 30
N_RUNS = 10
RANDOM_SEEDS = list(range(42, 42 + N_RUNS))  # 10 个不同的随机种子

# 问题配置
from problem import (
    F2Problem, F3Problem, F4Problem,
    RastriginProblem, GriewankProblem, AckleyProblem,
    # QuickSimu1Problem, QuickSimu2Problem,
)

PROBLEMS = {
    "F2": lambda: F2Problem(n_var=10, m=10, xl=0.0, xu=np.pi),
    "F3": lambda: F3Problem(n_var=10, xl=-100.0, xu=100.0),
    "F4": lambda: F4Problem(n_var=10, xl=-100.0, xu=100.0),
    "Rastrigin": lambda: RastriginProblem(n_var=10, xl=-5.12, xu=5.12),
    "Griewank": lambda: GriewankProblem(n_var=10, xl=-600.0, xu=600.0),
    "Ackley": lambda: AckleyProblem(n_var=10, xl=-32.768, xu=32.768),
    # "QuickSimu1": lambda: QuickSimu1Problem(),
    # "QuickSimu2": lambda: QuickSimu2Problem(),
}

# 算法配置 - 包含标准 GA 和 HA_Nelder_Mead 的不同方法
METHODS = {
    # "GA": "GA",  # 标准 GA
    # "Nelder-Mead": "Nelder-Mead",
    "rbf": "rbf",
    "gp": "gp",  # 高斯过程代理模型
    # "history-ladder": "history-ladder",  # 历史阶梯跳跃局部搜索
    # "Adam": "Adam",
    # "Sophia": "Sophia",
    # "Lion": "Lion",
    # "AdamW": "AdamW",
    # "L-BFGS-B": "L-BFGS-B",
    # Bandit v2 方法
    "QL-V2": "QL-V2",
    "UCB-V2": "UCB-V2",
}

# 输出目录占位符 - 实际值由 main() 通过 Pool initializer 设置到 worker 进程
# 注意：不在模块顶层调用 datetime.now()，避免 Windows spawn 模式下每个 worker
# 重新 import 本模块时各自生成不同 timestamp，导致创建多个结果目录。
RESULTS_DIR: Path = Path(__file__).parent / "experiments_results" / "result_placeholder"


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
    
    def __init__(self, skip_first=False):
        super().__init__()
        self.data: List[GenerationData] = []
        self.skip_first = skip_first
        self.first_call = True
    
    def notify(self, algorithm):
        """每代结束时调用"""
        # 如果设置了skip_first，跳过第一次调用（因为初始种群已手动记录）
        if self.skip_first and self.first_call:
            self.first_call = False
            return
        
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
    
    使用固定的随机种子确保可重复性，同一种子下所有方法使用相同的初始种群。
    
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
    initial_pop = rng.uniform(xl, xu, (pop_size, problem.n_var))
    
    # 检查并处理重复个体（连续优化中概率极低，但为了确保一致性）
    # GA 的 eliminate_duplicates=True 会在初始化时去重并填充，导致初始种群不一致
    # 因此我们确保初始种群没有重复
    unique_pop = []
    for individual in initial_pop:
        is_duplicate = False
        for existing in unique_pop:
            if np.allclose(individual, existing, rtol=1e-10, atol=1e-10):
                is_duplicate = True
                break
        
        if is_duplicate:
            # 如果重复，生成新的个体，直到不重复
            max_attempts = 100
            for _ in range(max_attempts):
                new_individual = rng.uniform(xl, xu, problem.n_var)
                is_new_duplicate = False
                for existing in unique_pop:
                    if np.allclose(new_individual, existing, rtol=1e-10, atol=1e-10):
                        is_new_duplicate = True
                        break
                if not is_new_duplicate:
                    unique_pop.append(new_individual)
                    break
            else:
                # 如果100次尝试后仍然重复，直接添加（这种情况几乎不可能）
                unique_pop.append(individual)
        else:
            unique_pop.append(individual)
    
    return np.array(unique_pop)


# ============================================================================
# 算法运行函数
# ============================================================================

def run_standard_ga(
    problem,
    pop_size: int,
    n_gen: int,
    seed: int,
    initial_pop: Optional[NDArray] = None
) -> Tuple[Any, List[GenerationData]]:
    """
    运行标准 GA 算法。
    
    使用统一的初始种群生成函数，确保与其他方法完全一致。
    
    Args:
        initial_pop: 预生成的初始种群（可选），如果为None则使用seed生成
    """
    callback = DataCollectorCallback(skip_first=False)
    
    # 如果没有提供初始种群，使用统一函数生成
    if initial_pop is None:
        initial_pop = generate_initial_population(problem, pop_size, seed)
    print(f"GA使用的初始种群: initial_pop[0,0]={initial_pop[0,0]:.6f}")
    
    algorithm = GA(
        pop_size=pop_size,
        eliminate_duplicates=True,
        sampling=initial_pop,  # 使用预生成的初始种群
    )
    
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        callback=callback,
        verbose=False
    )
    
    # 调试：打印第一代的信息
    if len(callback.data) > 0:
        gen1 = callback.data[0]
        print(f"GA seed={seed}: gen1 best_F={gen1.best_F:.10e}, best_X[0]={gen1.best_X[0]:.6f}")
    
    return result, callback.data


def run_ha(
    problem,
    pop_size: int,
    n_gen: int,
    seed: int,
    cluster_method: str,
    initial_pop: Optional[NDArray] = None
) -> Tuple[Any, List[GenerationData]]:
    """
    运行 HA 算法。
    
    使用统一的初始种群生成函数，确保与其他方法完全一致。
    
    Args:
        initial_pop: 预生成的初始种群（可选），如果为None则使用seed生成
    """
    callback = DataCollectorCallback(skip_first=False)
    
    # 如果没有提供初始种群，使用统一函数生成
    if initial_pop is None:
        initial_pop = generate_initial_population(problem, pop_size, seed)
    print(f"HA_{cluster_method}使用的初始种群: initial_pop[0,0]={initial_pop[0,0]:.6f}")
    
    algorithm = HA(
        method="L-BFGS-B",
        pop_size=pop_size,
        niche_num=3,
        mutation_rate=1.0,
        inherit_rate=1.0,
        activate_method=True,
        cluster_method=cluster_method,
        X=initial_pop,  # 传入统一的初始种群
        seed=seed,  # 保留seed用于后续随机操作
    )
    
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        callback=callback,
        verbose=False
    )
    
    # 调试：打印第一代的信息
    if len(callback.data) > 0:
        gen1 = callback.data[0]
        print(f"HA_{cluster_method} seed={seed}: gen1 best_F={gen1.best_F:.10e}, best_X[0]={gen1.best_X[0]:.6f}")
    
    return result, callback.data


def run_ha_original(
    problem,
    pop_size: int,
    n_gen: int,
    seed: int,
    initial_pop: Optional[NDArray] = None
) -> Tuple[Any, List[GenerationData]]:
    """
    运行原始 HA 算法（ha_original.py）。
    
    使用统一的初始种群生成函数，确保与其他方法完全一致。
    
    Args:
        initial_pop: 预生成的初始种群（可选），如果为None则使用seed生成
    """
    callback = DataCollectorCallback(skip_first=False)
    
    # 如果没有提供初始种群，使用统一函数生成
    if initial_pop is None:
        initial_pop = generate_initial_population(problem, pop_size, seed)
    print(f"HA_original使用的初始种群: initial_pop[0,0]={initial_pop[0,0]:.6f}")
    
    algorithm = HA_Original(
        method="L-BFGS-B",
        pop_size=pop_size,
        niche_num=3,
        mutation_rate=1.0,
        inherit_rate=1.0,
        activate_method=True,
        X=initial_pop,  # 传入统一的初始种群
        seed=seed,  # 保留seed用于后续随机操作
    )
    
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        callback=callback,
        verbose=False
    )
    
    # 调试：打印第一代的信息
    if len(callback.data) > 0:
        gen1 = callback.data[0]
        print(f"HA_original seed={seed}: gen1 best_F={gen1.best_F:.10e}, best_X[0]={gen1.best_X[0]:.6f}")
    
    return result, callback.data


def run_ha_nelder_mead(
    problem,
    pop_size: int,
    n_gen: int,
    seed: int,
    cluster_method: str,
    initial_pop: Optional[NDArray] = None
) -> Tuple[Any, List[GenerationData]]:
    """
    运行 HA_Nelder_Mead 算法。
    
    使用统一的初始种群生成函数，确保与其他方法完全一致。
    
    Args:
        initial_pop: 预生成的初始种群（可选），如果为None则使用seed生成
    """
    callback = DataCollectorCallback(skip_first=False)
    
    # 如果没有提供初始种群，使用统一函数生成
    if initial_pop is None:
        initial_pop = generate_initial_population(problem, pop_size, seed)
    print(f"HA_Nelder_Mead_{cluster_method}使用的初始种群: initial_pop[0,0]={initial_pop[0,0]:.6f}")
    
    algorithm = HA_Nelder_Mead(
        method="Nelder-Mead",
        pop_size=pop_size,
        niche_num=3,
        mutation_rate=1.0,
        inherit_rate=1.0,
        activate_method=True,
        cluster_method=cluster_method,
        X=initial_pop,  # 传入统一的初始种群
        seed=seed,  # 保留seed用于后续随机操作
    )
    
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        callback=callback,
        verbose=False
    )
    
    # 调试：打印第一代的信息
    if len(callback.data) > 0:
        gen1 = callback.data[0]
        print(f"HA_Nelder_Mead_{cluster_method} seed={seed}: gen1 best_F={gen1.best_F:.10e}, best_X[0]={gen1.best_X[0]:.6f}")
    
    return result, callback.data


def run_ha_nelder_mead_method(
    problem,
    pop_size: int,
    n_gen: int,
    seed: int,
    method: str,
    initial_pop: Optional[NDArray] = None
) -> Tuple[Any, List[GenerationData]]:
    """
    运行 HA_Nelder_Mead 算法的指定方法。
    
    支持的方法：Nelder-Mead, rbf, Adam, Sophia, Lion, AdamW, L-BFGS-B
    
    Args:
        problem: 优化问题
        pop_size: 种群大小
        n_gen: 代数
        seed: 随机种子
        method: 局部搜索方法名称
        initial_pop: 预生成的初始种群（可选）
        
    Returns:
        Tuple: (优化结果, 代数据列表)
    """
    callback = DataCollectorCallback(skip_first=False)
    
    # 如果没有提供初始种群，使用统一函数生成
    if initial_pop is None:
        initial_pop = generate_initial_population(problem, pop_size, seed)
    print(f"HA_Nelder_Mead_{method}使用的初始种群: initial_pop[0,0]={initial_pop[0,0]:.6f}")
    
    algorithm = HA_Nelder_Mead(
        method=method,
        pop_size=pop_size,
        niche_num=3,
        mutation_rate=1.0,
        inherit_rate=1.0,
        activate_method=True,
        cluster_method="kmeans",  # 使用 kmeans 作为默认聚类方法
        X=initial_pop,  # 传入统一的初始种群
        seed=seed,  # 保留seed用于后续随机操作
    )
    
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        callback=callback,
        verbose=False
    )
    
    # 调试：打印第一代的信息
    if len(callback.data) > 0:
        gen1 = callback.data[0]
        print(f"HA_Nelder_Mead_{method} seed={seed}: gen1 best_F={gen1.best_F:.10e}, best_X[0]={gen1.best_X[0]:.6f}")
    
    return result, callback.data


def run_ha_ql(
    problem,
    pop_size: int,
    n_gen: int,
    seed: int,
    ls_actions: Optional[List[str]] = None,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.95,
    epsilon_min: float = 0.05,
    initial_pop: Optional[NDArray] = None,
) -> Tuple[Any, List[GenerationData]]:
    """
    运行 HA-QL 算法（Q-learning 动态局部搜索选择器）。

    使用 Q-learning（ε-贪婪多臂老虎机）在每次局部搜索时动态选择
    最优算法，并与单一固定方法的 HA 进行对比。

    Args:
        problem: 优化问题（pymoo Problem 对象）。
        pop_size: 种群大小。
        n_gen: 最大迭代代数。
        seed: 随机种子，保证可复现性。
        ls_actions: Q-learning 可选的局部搜索算法列表。
                    默认：["Nelder-Mead", "L-BFGS-B", "rbf", "gp", "Adam"]。
        epsilon: 初始探索率，默认 1.0。
        epsilon_decay: 探索率衰减因子，默认 0.95。
        epsilon_min: 探索率下限，默认 0.05。
        initial_pop: 预生成的初始种群（可选），为 None 时使用 seed 生成。

    Returns:
        Tuple: (优化结果, 代数据列表)
    """
    callback = DataCollectorCallback(skip_first=False)

    if initial_pop is None:
        initial_pop = generate_initial_population(problem, pop_size, seed)
    print(f"HA_QL使用的初始种群: initial_pop[0,0]={initial_pop[0,0]:.6f}")

    # 默认动作集：覆盖多种风格的局部搜索算法
    if ls_actions is None:
        ls_actions = ["Nelder-Mead", "L-BFGS-B", "rbf", "gp", "Adam"]

    algorithm = HA_QL(
        ls_actions=ls_actions,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        pop_size=pop_size,
        niche_num=3,
        mutation_rate=1.0,
        inherit_rate=1.0,
        activate_method=True,
        cluster_method="kmeans",
        X=initial_pop,
        seed=seed,
    )

    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        callback=callback,
        verbose=False
    )

    # 打印第一代信息
    if len(callback.data) > 0:
        gen1 = callback.data[0]
        print(f"HA_QL seed={seed}: gen1 best_F={gen1.best_F:.10e}, best_X[0]={gen1.best_X[0]:.6f}")

    # 打印 Q-learning 选择器最终统计信息
    algorithm.print_selector_stats()

    return result, callback.data


def run_ha_ucb(
    problem,
    pop_size: int,
    n_gen: int,
    seed: int,
    global_best_actions: Optional[List[str]] = None,
    elite_actions: Optional[List[str]] = None,
    c: float = 1.0,
    alpha: float = 0.1,
    reward_scale_beta: float = 0.1,
    initial_pop: Optional[NDArray] = None,
) -> Tuple[Any, List[GenerationData]]:
    """
    运行 HA-UCB-V2 算法（UCB 动态局部搜索选择器）。
    """
    callback = DataCollectorCallback(skip_first=False)

    if initial_pop is None:
        initial_pop = generate_initial_population(problem, pop_size, seed)
    print(f"HA_UCB使用的初始种群: initial_pop[0,0]={initial_pop[0,0]:.6f}")

    if global_best_actions is None:
        global_best_actions = ["Nelder-Mead", "L-BFGS-B"]
    if elite_actions is None:
        elite_actions = ["rbf", "gp"]

    algorithm = HA_UCB(
        global_best_actions=global_best_actions,
        elite_actions=elite_actions,
        c=c,
        alpha=alpha,
        reward_scale_beta=reward_scale_beta,
        pop_size=pop_size,
        niche_num=3,
        mutation_rate=1.0,
        inherit_rate=1.0,
        activate_method=True,
        cluster_method="kmeans",
        X=initial_pop,
        seed=seed,
    )

    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        callback=callback,
        verbose=False
    )

    if len(callback.data) > 0:
        gen1 = callback.data[0]
        print(f"HA_UCB seed={seed}: gen1 best_F={gen1.best_F:.10e}, best_X[0]={gen1.best_X[0]:.6f}")

    algorithm.print_selector_stats()

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
    
    # 使用 TeeOutput 同时输出到日志文件和控制台
    with TeeOutput(log_file, 'w') as tee:
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
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
    
    start_time = time.time()
    
    try:
        # 统一生成初始种群，确保三个方法使用完全相同的初始种群
        initial_pop = generate_initial_population(problem, POP_SIZE, seed)
        
        if method_name == "GA":
            # 标准 GA 算法
            result, gen_data = run_standard_ga(problem, POP_SIZE, N_GEN, seed, initial_pop)
        elif method_name == "HA_original":
            result, gen_data = run_ha_original(problem, POP_SIZE, N_GEN, seed, initial_pop)
        elif method_name.startswith("HA_Nelder_Mead_"):
            cluster_method = METHODS[method_name]
            result, gen_data = run_ha_nelder_mead(problem, POP_SIZE, N_GEN, seed, cluster_method, initial_pop)
        elif method_name == "QL-V2":
            result, gen_data = run_ha_ql(
                problem, POP_SIZE, N_GEN, seed,
                ls_actions=["Nelder-Mead", "L-BFGS-B", "rbf", "gp", "Adam"],
                epsilon=1.0,
                epsilon_decay=0.95,
                epsilon_min=0.05,
                initial_pop=initial_pop,
            )
        elif method_name == "UCB-V2":
            result, gen_data = run_ha_ucb(
                problem, POP_SIZE, N_GEN, seed,
                global_best_actions=["Nelder-Mead", "L-BFGS-B"],
                elite_actions=["rbf", "gp"],
                c=1.0,
                alpha=0.1,
                reward_scale_beta=0.1,
                initial_pop=initial_pop,
            )
        elif method_name in METHODS and method_name != "GA":
            # 使用 HA_Nelder_Mead 的不同方法（排除 GA，因为 GA 已经单独处理）
            method = METHODS[method_name]
            result, gen_data = run_ha_nelder_mead_method(problem, POP_SIZE, N_GEN, seed, method, initial_pop)
        else:
            cluster_method = METHODS.get(method_name, "kmeans")
            result, gen_data = run_ha(problem, POP_SIZE, N_GEN, seed, cluster_method, initial_pop)
        
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
        # 记录到日志文件（通过重定向的 stdout）
        error_msg = f"[{datetime.now()}] 错误: {problem_name} - {method_name} - Run {run_id}\n"
        error_detail = f"  异常: {e}\n"
        print(error_msg + error_detail)
        import traceback
        traceback.print_exc()
        
        # 同时强制输出到原始控制台，以便用户及时发现错误
        sys.__stdout__.write("\n" + "!"*60 + "\n")
        sys.__stdout__.write(error_msg + error_detail)
        sys.__stdout__.write(traceback.format_exc())
        sys.__stdout__.write("!"*60 + "\n")
        sys.__stdout__.flush()
        
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
# 失败日志检测与断点续跑
# ============================================================================

CHECKPOINT_NAME = "_checkpoint_all_results.pkl"


def is_log_failed(log_path: Path) -> bool:
    """判断单次运行的 .log 是否失败或未正常结束。"""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    if "Traceback (most recent call last):" in text:
        return True
    if "--- Logging error ---" in text:
        return True
    if "] 错误:" in text:
        return True
    if "开始运行:" not in text:
        return False
    if "完成:" in text and "最终最优值:" in text:
        return False
    return True


def parse_log_filename(filename: str) -> Optional[Tuple[str, str, int]]:
    """
    从 ``{problem}_{method}_run{nn}.log`` 解析 problem、method、run_id。
    method 名中可含连字符或下划线，problem 与 PROBLEMS 的键一致。
    """
    m = re.match(r"^(.+)_run(\d+)\.log$", filename, re.IGNORECASE)
    if not m:
        return None
    rest, run_s = m.group(1), m.group(2)
    run_id = int(run_s)
    for pn in sorted(PROBLEMS.keys(), key=len, reverse=True):
        prefix = pn + "_"
        if rest.startswith(prefix):
            method_name = rest[len(prefix):]
            return pn, method_name, run_id
    return None


def save_run_checkpoint(results: List[RunResult]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / CHECKPOINT_NAME
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"运行 checkpoint 已保存: {path}（共 {len(results)} 条 RunResult）")


def load_run_checkpoint() -> Optional[List[RunResult]]:
    path = RESULTS_DIR / CHECKPOINT_NAME
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def merge_run_results(
    previous: List[RunResult],
    updated: List[RunResult],
) -> List[RunResult]:
    def key(r: RunResult) -> Tuple[str, str, int]:
        return (r.problem_name, r.method_name, r.run_id)

    merged: Dict[Tuple[str, str, int], RunResult] = {key(r): r for r in previous}
    for r in updated:
        merged[key(r)] = r
    return list(merged.values())


def retry_failed_experiments(resume_dir: Path) -> None:
    """
    扫描 resume_dir/logs 下失败或未完成的日志，按相同 run_id -> RANDOM_SEEDS[run_id] 重跑，
    并与 _checkpoint_all_results.pkl 合并（若存在），重写 checkpoint 与 CSV 汇总。
    """
    global RESULTS_DIR
    resume_dir = resume_dir.resolve()
    if not resume_dir.is_dir():
        raise SystemExit(f"结果目录不存在: {resume_dir}")
    RESULTS_DIR = resume_dir
    log_dir = RESULTS_DIR / "logs"
    if not log_dir.is_dir():
        raise SystemExit(f"无 logs 子目录: {log_dir}")

    failed_tasks: List[Tuple[str, str, int, int]] = []
    for log_path in sorted(log_dir.glob("*.log")):
        if not is_log_failed(log_path):
            continue
        parsed = parse_log_filename(log_path.name)
        if not parsed:
            print(f"跳过无法解析的文件名: {log_path.name}")
            continue
        problem_name, method_name, run_id = parsed
        if problem_name not in PROBLEMS:
            print(f"跳过（PROBLEMS 中无此问题）: {problem_name} <- {log_path.name}")
            continue
        if method_name not in METHODS:
            print(f"跳过（METHODS 中无此方法）: {method_name} <- {log_path.name}")
            continue
        if run_id < 0 or run_id >= len(RANDOM_SEEDS):
            print(f"跳过 run_id 超出 RANDOM_SEEDS 范围: {log_path.name}")
            continue
        seed = RANDOM_SEEDS[run_id]
        failed_tasks.append((problem_name, method_name, run_id, seed))

    if not failed_tasks:
        print("未检测到失败或未完成的日志，无需重跑。")
        return

    print(f"将重跑 {len(failed_tasks)} 个任务:")
    for problem_name, method_name, run_id, seed in failed_tasks:
        print(f"  {problem_name} | {method_name} | run_id={run_id} | seed={seed}")

    prev = load_run_checkpoint()
    if not prev:
        print(
            "警告: 未找到 _checkpoint_all_results.pkl，无法与历史成功运行合并；"
            "本次仅重跑并更新日志与部分 checkpoint。"
        )

    has_ansys = any(t[0].startswith("QuickSimu") for t in failed_tasks)
    n_workers = 1 if has_ansys else min(48, mp.cpu_count())
    print(f"并行进程数: {n_workers}")

    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(RESULTS_DIR,),
    ) as pool:
        new_results = list(pool.imap(run_single_experiment, failed_tasks))

    merged = merge_run_results(prev or [], new_results)
    save_run_checkpoint(merged)

    if prev:
        by_problem: Dict[str, List[RunResult]] = {}
        for r in merged:
            by_problem.setdefault(r.problem_name, []).append(r)
        for pn, lst in by_problem.items():
            results_by_method: Dict[str, List[RunResult]] = {}
            for r in lst:
                results_by_method.setdefault(r.method_name, []).append(r)
            save_results_to_csv(lst, pn)
            print_summary(results_by_method, pn)
    else:
        print(
            "未合并历史 checkpoint：未重写 summary CSV。"
            "若需恢复完整汇总，请先有一次完整实验生成 _checkpoint_all_results.pkl，再执行重试。"
        )

    print(f"\n重跑结束，结果目录: {RESULTS_DIR}")


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

def _worker_init(results_dir: Path) -> None:
    """
    worker 进程初始化函数：将主进程计算好的 RESULTS_DIR 注入到 worker 的全局变量中。

    Windows 使用 "spawn" 模式创建子进程，每个 worker 重新 import 本模块，
    模块顶层的 RESULTS_DIR 会被重置为占位符值。通过 Pool initializer 在 worker
    启动后立即覆写全局变量，确保所有 worker 使用同一个结果目录。

    Args:
        results_dir: 主进程计算好的结果目录路径。
    """
    global RESULTS_DIR
    RESULTS_DIR = results_dir


def main():
    """
    主函数：运行所有实验。
    """
    # 在主进程中计算唯一的 timestamp，避免 worker 进程 re-import 时各自生成不同时间戳
    global RESULTS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = Path(__file__).parent / "experiments_results" / f"result_{timestamp}"

    print(f"实验开始时间: {datetime.now()}")
    print(f"配置: pop_size={POP_SIZE}, n_gen={N_GEN}, n_runs={N_RUNS}")
    print(f"问题: {list(PROBLEMS.keys())}")
    print(f"方法: {list(METHODS.keys())}")
    cpu_count = mp.cpu_count()
    # 对 ANSYS 仿真问题使用保守并行：MAPDL 多进程并行容易触发 Fortran I/O 冲突
    has_ansys_sim_problem = any(name.startswith("QuickSimu") for name in PROBLEMS.keys())
    if has_ansys_sim_problem:
        n_workers = 1
    else:
        # 非仿真问题可使用较高并行
        n_workers = min(48, cpu_count)
    print(f"CPU 核心数: {cpu_count}")
    print(f"实际使用的并行进程数: {n_workers}")
    if has_ansys_sim_problem:
        print("检测到 QuickSimu（ANSYS）问题：已自动切换为单进程运行，避免 MAPDL Fortran I/O 冲突。")
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

    # 通过 initializer 将主进程的 RESULTS_DIR 同步到每个 worker 进程
    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(RESULTS_DIR,),
    ) as pool:
        all_results: List[RunResult] = []
        for result in tqdm(
            pool.imap_unordered(run_single_experiment, all_tasks),
            total=len(all_tasks),
            desc="实验进度",
            unit="task",
            dynamic_ncols=True,
        ):
            all_results.append(result)

    save_run_checkpoint(all_results)

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
            "HA_original": "green",
            "HA_meanshift": "orange",
            "HA_dbscan": "purple",
            "HA_Nelder_Mead_kmeans": "cyan",
            "HA_Nelder_Mead_meanshift": "magenta",
            "HA_Nelder_Mead_dbscan": "yellow"
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
    parser = argparse.ArgumentParser(description="HA 实验运行 / 失败重试")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="加载 RESULTS_DIR 下的结果并绘制示例曲线（需 matplotlib）",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="已有实验结果目录（含 logs/），与 --retry-failed 联用以只重跑失败任务",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="扫描 resume-dir/logs 中失败或未完成的 .log，按相同 seed 重跑",
    )
    args = parser.parse_args()

    if args.plot:
        example_load_and_plot()
    elif args.retry_failed:
        if args.resume_dir is None:
            parser.error("--retry-failed 需要同时指定 --resume-dir")
        retry_failed_experiments(args.resume_dir)
    else:
        main()

