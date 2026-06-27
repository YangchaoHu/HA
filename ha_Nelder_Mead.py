"""
HA (Hybrid Algorithm) - 混合进化算法

该模块实现了一种结合聚类、局部搜索和遗传操作的混合优化算法。
主要特点：
    - K-means 聚类实现小生境划分
    - 支持多种局部搜索方法 (L-BFGS-B, Adam 等)
    - 自适应变异策略
    - 约束处理机制
"""

# ============================================================================
# 标准库导入
# ============================================================================
import logging
import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ============================================================================
# 第三方库导入
# ============================================================================
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize as scipy_minimize
from scipy.interpolate import RBFInterpolator
from scipy.linalg import lstsq
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# PyMoo 相关导入
from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.individual import Individual
from pymoo.core.population import Population

# ============================================================================
# 环境配置
# ============================================================================
os.environ["OMP_NUM_THREADS"] = "1"

# ============================================================================
# 类型别名
# ============================================================================
ArrayLike = NDArray[Any]


class PopulationHistory:
    """
    种群历史信息存储类（去重）
    
    用于存储种群所有历史信息，自动去重。
    基于决策变量 X 的容差判断是否为重复个体。
    
    Attributes:
        tolerance: 去重容差，用于判断两个解是否相同
        total_count: 总记录次数（包括重复）
        unique_count: 去重后的记录数量
        
    Example:
        >>> history = PopulationHistory(tolerance=1e-8)
        >>> # 添加种群信息
        >>> history.add(x1, f1, cv1)
        >>> history.add(x2, f2, cv2)
        >>> # 查询是否已存在
        >>> if history.contains(x1):
        ...     f, cv = history.get(x1)
        >>> # 获取统计信息
        >>> print(f"总记录: {history.total_count}, 去重后: {history.unique_count}")
    """
    
    def __init__(self, tolerance: float = 1e-8):
        """
        初始化种群历史信息存储
        
        Args:
            tolerance: 去重容差，用于判断两个解是否相同（默认 1e-8）
        """
        self.tolerance = tolerance
        self.total_count = 0
        self._X_list: List[ArrayLike] = []  # 存储去重后的决策变量
        self._F_list: List[ArrayLike] = []  # 存储对应的适应度
        self._CV_list: List[float] = []  # 存储对应的约束违反度
        
    def add(
        self,
        X: ArrayLike,
        F: ArrayLike,
        CV: Union[float, ArrayLike]
    ) -> bool:
        """
        添加种群信息（自动去重）
        
        如果该解已存在（在容差范围内），则更新其适应度和约束违反度（如果新值更好）。
        否则添加新的记录。
        
        Args:
            X: 决策变量，shape = (dim,) 或 (N, dim)
            F: 适应度值，shape = (1,) 或 (N, 1)
            CV: 约束违反度，标量或 shape = (1,) 或 (N, 1)
            
        Returns:
            bool: True 表示是新解（已添加），False 表示是重复解（已存在）
        """
        # 转换为 numpy 数组并确保正确的形状
        X = np.atleast_2d(X)
        F = np.atleast_2d(F)
        CV = np.atleast_1d(CV)
        
        # 如果 CV 是标量，转换为数组
        if CV.ndim == 0:
            CV = CV.reshape(1)
        elif CV.ndim == 2:
            CV = CV.flatten()
        
        # 批量处理
        added_count = 0
        for i in range(len(X)):
            x = X[i]
            f = F[i] if F.shape[0] > 1 else F[0]
            cv = CV[i] if len(CV) > 1 else CV[0]
            
            self.total_count += 1
            
            # 查找是否已存在（在容差范围内）
            idx = self._find_duplicate(x)
            
            if idx is None:
                # 新解，添加
                self._X_list.append(x.copy())
                self._F_list.append(f.copy() if f.ndim > 0 else np.array([f]))
                self._CV_list.append(float(cv))
                added_count += 1
            else:
                # 重复解，如果新值更好则直接替换整个记录（包括 X、F、CV）
                current_f = self._F_list[idx]
                current_cv = self._CV_list[idx]
                
                # 如果新解的约束违反度更小，或者约束违反度相同但适应度更好，则替换
                if (cv < current_cv) or (cv == current_cv and float(f) < float(current_f)):
                    # 替换整个记录：决策变量、适应度、约束违反度
                    self._X_list[idx] = x.copy()
                    self._F_list[idx] = f.copy() if f.ndim > 0 else np.array([f])
                    self._CV_list[idx] = float(cv)
        
        return added_count > 0
    
    def _find_duplicate(self, x: ArrayLike) -> Optional[int]:
        """
        查找是否存在重复解（在容差范围内）
        
        Args:
            x: 待查找的决策变量
            
        Returns:
            Optional[int]: 如果找到重复解，返回其索引；否则返回 None
        """
        if len(self._X_list) == 0:
            return None
        
        x = np.asarray(x).flatten()
        
        # 计算与所有已存储解的距离
        for idx, stored_x in enumerate(self._X_list):
            stored_x = np.asarray(stored_x).flatten()
            if len(x) != len(stored_x):
                continue
            
            # 使用 L∞ 范数（最大绝对差值）判断是否在容差范围内
            max_diff = np.max(np.abs(x - stored_x))
            if max_diff <= self.tolerance:
                return idx
        
        return None
    
    def contains(self, X: ArrayLike) -> bool:
        """
        检查是否已存在该解
        
        Args:
            X: 决策变量，shape = (dim,) 或 (N, dim)
            
        Returns:
            bool: 如果已存在返回 True，否则返回 False
        """
        X = np.atleast_2d(X)
        for x in X:
            if self._find_duplicate(x) is not None:
                return True
        return False
    
    def get(
        self,
        X: ArrayLike
    ) -> Optional[Tuple[ArrayLike, float]]:
        """
        获取已存在解的结果
        
        Args:
            X: 决策变量，shape = (dim,)
            
        Returns:
            Optional[Tuple[ArrayLike, float]]: 如果找到，返回 (适应度, 约束违反度)；
                                              否则返回 None
        """
        x = np.asarray(X).flatten()
        idx = self._find_duplicate(x)
        
        if idx is not None:
            return (self._F_list[idx].copy(), self._CV_list[idx])
        return None
    
    def get_all(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        获取所有去重后的历史记录
        
        Returns:
            Tuple[ArrayLike, ArrayLike, ArrayLike]: (X, F, CV)
                - X: 决策变量矩阵，shape = (N, dim)
                - F: 适应度矩阵，shape = (N, 1)
                - CV: 约束违反度数组，shape = (N,)
        """
        if len(self._X_list) == 0:
            return np.array([]), np.array([]), np.array([])
        
        X = np.vstack(self._X_list)
        F = np.vstack([f.reshape(1, -1) if f.ndim == 0 else f.reshape(1, -1) 
                       for f in self._F_list])
        CV = np.array(self._CV_list)
        
        return X, F, CV
    
    def clear(self) -> None:
        """清空所有历史记录"""
        self._X_list.clear()
        self._F_list.clear()
        self._CV_list.clear()
        self.total_count = 0
    
    @property
    def unique_count(self) -> int:
        """去重后的记录数量"""
        return len(self._X_list)
    
    def __len__(self) -> int:
        """返回去重后的记录数量"""
        return len(self._X_list)
    
    def __repr__(self) -> str:
        """返回类的字符串表示"""
        return (f"PopulationHistory(tolerance={self.tolerance}, "
                f"total={self.total_count}, "
                f"unique={self.unique_count})")


class NoOpEvaluator(Evaluator):
    """
    空操作评估器
    
    不执行任何评估操作，假设评估已在算法其他部分完成。
    用于避免 PyMoo 框架的自动评估行为。
    """
    
    def _eval(
        self,
        problem: Any,
        pop: Population,
        return_values_of: List[str],
        **kwargs
    ) -> None:
        """跳过评估，不做任何操作"""
        pass


class HA(Algorithm):
    """
    混合进化算法 (Hybrid Algorithm)
    
    结合聚类、局部搜索和遗传操作的优化算法，适用于连续优化问题。
    
    Attributes:
        method: 局部搜索方法，支持 "L-BFGS-B", "TNC", "SLSQP", "Powell", 
                "trust-constr", "Adam"
        pop_size: 种群大小
        niche_num: 小生境（聚类）数量，仅 kmeans 使用
        mutation_rate: 变异率
        inherit_rate: 遗传率，控制通过遗传操作产生的后代比例
        activate_method: 是否激活局部搜索方法
        cluster_method: 聚类方法，支持 "kmeans", "meanshift", "dbscan"
        dbscan_eps: DBSCAN 的邻域半径参数，None 表示自动估算
        dbscan_min_samples: DBSCAN 的最小样本数参数，None 表示自动估算
        
    Example:
        >>> from pymoo.optimize import minimize
        >>> # 使用 K-means 聚类
        >>> algorithm = HA(method="L-BFGS-B", pop_size=100, niche_num=3)
        >>> result = minimize(problem, algorithm, termination=('n_gen', 100))
        >>> 
        >>> # 使用 MeanShift 聚类（自动确定聚类数）
        >>> algorithm = HA(method="L-BFGS-B", pop_size=100, cluster_method="meanshift")
        >>> 
        >>> # 使用 DBSCAN 聚类（自动估算参数）
        >>> algorithm = HA(method="L-BFGS-B", pop_size=100, cluster_method="dbscan")
        >>> # 手动指定 DBSCAN 参数
        >>> algorithm = HA(method="L-BFGS-B", pop_size=100, cluster_method="dbscan", 
        ...                dbscan_eps=0.3, dbscan_min_samples=3)
    """
    
    # 支持边界约束的优化方法
    BOUNDED_METHODS = ["L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr", "Nelder-Mead"]
    
    # 支持的聚类方法
    CLUSTER_METHODS = ["kmeans", "meanshift", "dbscan"]
    
    # ========================================================================
    # 初始化方法
    # ========================================================================
    
    def __init__(
        self,
        method: str = "rbf",
        pop_size: int = 100,
        niche_num: int = 3,
        mutation_rate: float = 0.3,  # 提高变异率以增强探索
        inherit_rate: float = 0.8,
        activate_method: bool = True,
        cluster_method: str = "kmeans",
        dbscan_eps: Optional[float] = None,
        dbscan_min_samples: Optional[int] = None,
        X: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        初始化 HA 算法
        
        Args:
            method: 局部搜索方法名称
            pop_size: 种群大小
            niche_num: 聚类数量（小生境数量，仅 kmeans 使用）
            mutation_rate: 变异率，范围 [0, 1]
            inherit_rate: 遗传率，范围 [0, 1]
            activate_method: 是否启用局部搜索
            cluster_method: 聚类方法，支持 "kmeans", "meanshift", "dbscan"
            dbscan_eps: DBSCAN 的邻域半径参数，None 表示自动估算（推荐）
            dbscan_min_samples: DBSCAN 的最小样本数参数，None 表示自动估算（推荐）
            X: 初始种群（可选），若为 None 则使用seed随机生成
            seed: 随机种子，用于生成初始种群（当X为None时使用）
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(**kwargs)
        
        # 验证聚类方法
        if cluster_method.lower() not in self.CLUSTER_METHODS:
            raise ValueError(f"不支持的聚类方法: {cluster_method}，"
                           f"支持的方法: {self.CLUSTER_METHODS}")
        
        # 算法配置参数
        self.method = method
        self.pop_size = pop_size
        self.niche_num = niche_num
        self.mutation_rate = mutation_rate
        self.inherit_rate = inherit_rate
        self.activate_method = activate_method
        self.cluster_method = cluster_method.lower()
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.X = X
        self.seed = seed
        
        # 运行时状态
        self.step_size: float = 1.0
        self.improvement: bool = True
        self.stagnation_count: int = 0  # 增加停滞计数
        self.fun_cnt: int = 0
        self.best: float = float("inf")
        self.best_individual: ArrayLike = np.array([], dtype=float)
        
        # 将在 _setup 中初始化的参数
        self.elite_num: int = 0
        self.dim: int = 0
        self.lb: ArrayLike = np.array([], dtype=float)
        self.ub: ArrayLike = np.array([], dtype=float)
        self.FEs: int = 0
        self.local_elite_id: List[int] = []
        self.pop_cv: ArrayLike = np.array([], dtype=float)
        
        # 种群历史信息存储（去重）
        self.history = PopulationHistory()

    def _setup(self, problem: Any, **kwargs) -> None:
        """
        设置算法参数
        
        从问题对象中提取维度和边界信息，初始化相关参数。
        
        Args:
            problem: PyMoo 问题对象
            **kwargs: 其他参数
        """
        super()._setup(problem, **kwargs)
        
        self._rng = np.random.default_rng(self.seed)
        self.evaluator = NoOpEvaluator()  # 禁用 PyMoo 自动评估
        
        # 从问题中获取维度
        self.dim = problem.n_var
        
        # 处理边界：支持数组或标量形式
        if hasattr(problem.xl, '__len__'):
            self.lb = np.array(problem.xl)
            self.ub = np.array(problem.xu)
        else:
            self.lb = np.full(problem.n_var, problem.xl)
            self.ub = np.full(problem.n_var, problem.xu)
        
        # 精英数量：种群的 5%（至少 3 个）
        self.elite_num = max(3, self.pop_size * 5 // 100)
        self.FEs = 0
        self.local_elite_id = []

    # ========================================================================
    # 种群初始化与进化
    # ========================================================================
    
    def _initialize_infill(self) -> Population:
        """
        初始化种群
        
        生成初始种群并进行首次评估。
        如果通过 self.X 传入了初始种群，直接使用；
        否则使用seed（如果提供）或numpy默认随机状态生成。
        
        Returns:
            Population: 初始化后的种群对象
        """
        if self.lb.size == 0 or self.ub.size == 0 or self.dim == 0:
            raise RuntimeError("算法尚未完成 _setup，无法初始化种群")
        print("初始化种群...")
        
        # 生成或使用指定的初始种群
        if self.X is None:
            if self.seed is not None:
                print(f"使用seed={self.seed}生成随机初始种群（使用np.random.default_rng）")
                # 使用独立的随机数生成器，与GA保持完全一致
                rng = np.random.default_rng(self.seed)
                pop_x = rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
                print(f"生成的初始种群: pop_x[0,0]={pop_x[0,0]:.6f}")
            else:
                print("生成随机初始种群（无seed）")
                pop_x = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        else:
            print("使用指定的初始种群（self.X）")
            pop_x = self.X
        
        # 批量评估初始种群
        pop_f, pop_cv = self.evaluate_fitness_cv_batch(pop_x)
        self.pop_cv = np.array(pop_cv).reshape(-1, 1)
        pop_f = np.array(pop_f).reshape(-1, 1)
        
        print(f"pop.shape = {pop_x.shape}")
        print(f"fit.shape = {pop_f.shape}")
        print(f"cv.shape = {self.pop_cv.shape}")
        
        # 不记录 gen=1，因为会在 experiment_runner 中统一手动记录
        # 这样确保所有方法（GA 和 HA）的 gen=1 完全一致
        
        return Population.new("X", pop_x, "F", pop_f)

    def _infill(self) -> Population:
        """
        执行一代进化
        
        包括：更新最优解、执行 HA 算法步骤、更新种群。
        
        Returns:
            Population: 进化后的新种群
        """
        if self.n_gen is None or self.problem is None:
            raise RuntimeError("算法状态未初始化，无法进化")
        print(f"从 Generation {self.n_gen - 1} 进化到 Generation {self.n_gen}...")
        
        # 获取当前种群数据
        pop = np.asarray(self.pop.get("X"))
        fit = np.asarray(self.pop.get("F"))
        cv = np.asarray(self.pop_cv)
        
        # 找到当前代的最佳个体
        best_index = self._find_best_individual(fit, cv)
        
        # 打印当前代信息
        print(f"best 的 index 是: {best_index}")
        print("{:<6} | {:<8} | {:>13.4f} |  {:>13.10f} ".format(
            self.n_gen - 1, self.problem.fes, float(fit[best_index]), float(cv[best_index])
        ))
        print(f"best 在 pop_cv 中的 cv: {cv[best_index]}")
        
        # 更新全局最优（假设种群已按适应度排序，最优在索引 0）
        best_idx = 0
        current_best = fit[best_idx, 0]
        
        if current_best < self.best:
            self.best = current_best
            self.best_individual = pop[best_idx].copy()
            self.improvement = True
            self.stagnation_count = 0  # 重置停滞计数
        else:
            self.improvement = False
            self.stagnation_count += 1  # 增加停滞计数
        
        # 执行 HA 算法的一代进化
        new_pop, new_fit, new_cv = self._step_ha(pop, fit, cv)
        
        # 更新种群
        self.pop = Population.new("X", new_pop, "F", new_fit)
        self.pop_cv = new_cv
        
        # 数据记录由 experiment_runner 中的 callback 统一处理
        # 这样确保 GA 和 HA 的记录方式完全一致
        
        return self.pop

    def _find_best_individual(
        self,
        fit: ArrayLike,
        cv: ArrayLike
    ) -> int:
        """
        找到最佳个体索引
        
        优先选择可行解中适应度最小的；若无可行解，选择约束违反度最小的。
        
        Args:
            fit: 适应度数组
            cv: 约束违反度数组
            
        Returns:
            int: 最佳个体的索引
        """
        feasible_indices = np.where(cv <= 0)[0]
        
        if len(feasible_indices) > 0:
            # 从可行解中找适应度最小的
            return feasible_indices[np.argmin(fit[feasible_indices])]
        else:
            # 无可行解，返回约束违反度最小的
            return int(np.argmin(cv))

    # ========================================================================
    # 评估函数
    # ========================================================================
    
    def evaluate_fitness_cv(
        self,
        x: ArrayLike
    ) -> Tuple[ArrayLike, float]:
        """
        评估单个个体的适应度和约束违反度
        
        Args:
            x: 决策变量
            
        Returns:
            Tuple[ArrayLike, float]: (适应度值, 约束违反度)
            
        Raises:
            ValueError: 当 x 超出边界时
        """
        if self.problem is None:
            raise RuntimeError("问题未初始化，无法评估适应度")
        # 边界检查
        if np.any(x > self.ub) or np.any(x < self.lb):
            print(x)
            raise ValueError("x out of bounds")
        
        x = np.atleast_2d(x)
        out = {}
        self.problem._evaluate(x, out)
        fitness = out["F"]
        
        # 计算约束违反度
        if hasattr(self.problem, "evaluate") and self.problem.has_constraints():
            G = np.atleast_1d(out["G"])
            cv = float(np.sum(np.maximum(0, G)))
        else:
            cv = 0.0
        
        # 添加到种群历史信息
        self.history.add(x, fitness, cv)
        
        return fitness, cv

    def evaluate_fitness_cv_batch(
        self,
        X: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        批量评估多个个体的适应度和约束违反度
        
        Args:
            X: 决策变量矩阵，shape = (N, dim)
            
        Returns:
            Tuple[ArrayLike, ArrayLike]: (适应度数组, 约束违反度数组)
        """
        if self.problem is None:
            raise RuntimeError("问题未初始化，无法评估适应度")
        print(f"进入了 evaluate_fitness_cv_batch，X 大小 {X.shape[0]}")
        
        out = {}
        self.problem._evaluate(X, out)
        fitness = out["F"]
        G = out.get("G", None)
        
        # 处理约束
        if G is None:
            cv = np.zeros((len(X), 1))
        else:
            G = np.array(G)
            if G.ndim == 1:
                G = G.reshape(-1, 1)
            elif G.ndim == 0:
                G = G.reshape(1, 1)
            cv = np.sum(np.maximum(0, G), axis=1).reshape(-1, 1)
        
        # 添加到种群历史信息
        fitness_reshaped = fitness.reshape(-1, 1)
        self.history.add(X, fitness_reshaped, cv)
        
        return fitness_reshaped, cv

    # ========================================================================
    # HA 核心算法
    # ========================================================================
    
    def _step_ha(
        self,
        pop: ArrayLike,
        fit: ArrayLike,
        cv: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        HA 算法的一步进化
        
        包括：聚类学习、精英选择、遗传操作、变异、去重、选择。
        
        Args:
            pop: 当前种群
            fit: 适应度值
            cv: 约束违反度
            
        Returns:
            Tuple: (新种群, 新适应度, 新约束违反度)
        """
        # 聚类和局部学习
        elite_id = []
        pop, fit, cv, self.local_elite_id = self._clustering_and_learning(pop, fit, cv)
        
        if not self.check_bounds(pop):
            raise ValueError("聚类学习后种群越界")
        
        # 按约束优先规则排序
        sorted_indices = self._sort_by_constraint_dominance(fit, cv)
        pop = pop[sorted_indices]
        fit = fit[sorted_indices]
        cv = cv[sorted_indices]
        
        # 先加入局部精英（来自聚类学习）
        elite_id.extend(self.local_elite_id)
        
        # 再加入全局精英，补足至 self.elite_num（不与局部精英重复计数）
        global_elite_count = max(0, self.elite_num - len(elite_id))
        if global_elite_count > 0:
            global_elite_id = np.argsort(fit[:, 0])[:global_elite_count].tolist()
            elite_id.extend(global_elite_id)
        
        # 去重并保持顺序
        elite_id = self._unique_preserve_order(elite_id)
        print(f"精英个体数量: {len(elite_id)}")
        
        # 获取精英个体及其适应度（直接保留的精英不参与变异和评估，但可能被随机选中参与变异）
        elite_individuals = pop[elite_id]
        elite_fit = fit[elite_id]
        elite_cv = cv[elite_id]
        
        # 计算后代数量（需要生成的后代数量，不包括直接保留的精英）
        offspring_size = self.pop_size - len(elite_id)
        
        # 生成后代：只用遗传操作（与原版保持一致）
        offspring = self._inheritance(offspring_size, pop, fit)
        
        # 变异操作（对后代进行变异）
        mutate_num = round(offspring_size * self.mutation_rate)
        if mutate_num > 0 and len(offspring) > 0:
            mutate_id = self._rng.choice(len(offspring), min(mutate_num, len(offspring)), replace=False)
            offspring[mutate_id, :] = self._mutate(offspring[mutate_id, :])
        
        if not self.check_bounds(offspring):
            print("_mutate 越界")
            raise ValueError("变异后越界")
        
        # 去重并评估（评估后代）
        offspring, repeat = np.unique(offspring, axis=0, return_counts=True)
        print(f"处理重复个体后 offspring 的 shape: {offspring.shape}")
        
        offspring_fit, offspring_cv = self.evaluate_fitness_cv_batch(offspring)
        offspring_fit = offspring_fit.reshape(-1, 1)
        offspring_cv = offspring_cv.reshape(-1, 1)
        
        # 恢复重复个体
        offspring, offspring_fit, offspring_cv = self._restore_duplicates(
            offspring, offspring_fit, offspring_cv, repeat
        )
        
        # 合并父代和子代
        new_pop = np.vstack((pop[elite_id, :], offspring))
        new_fit = np.vstack((fit[elite_id, :], offspring_fit))
        new_cv = np.vstack((cv[elite_id, :], offspring_cv))
        
        # 选择前 pop_size 个个体
        sorted_indices = self._sort_by_constraint_dominance(new_fit, new_cv)
        selected_indices = sorted_indices[:self.pop_size]
        
        new_pop = new_pop[selected_indices]
        new_fit = new_fit[selected_indices]
        new_cv = new_cv[selected_indices]
        
        # 最终排序
        sorted_indices = self._sort_by_constraint_dominance(new_fit, new_cv)
        new_pop = new_pop[sorted_indices]
        new_fit = new_fit[sorted_indices]
        new_cv = new_cv[sorted_indices]
        
        # 检查种群多样性
        unique_count = np.unique(new_pop, axis=0).shape[0]
        print(f"种群中不重复的个体数量: {unique_count}")
        
        # 改进：不再强制终止，而是注入新鲜血液
        if unique_count <= max(2, self.pop_size // 20):
            print(">>> 检测到种群严重趋同，注入 30% 随机个体以增强后期探索性")
            replace_num = int(self.pop_size * 0.3)
            # 保持前部分个体，替换最后 30% 为随机个体
            random_pop = self._rng.uniform(self.lb, self.ub, (replace_num, self.dim))
            random_fit, random_cv = self.evaluate_fitness_cv_batch(random_pop)
            
            new_pop[-replace_num:] = random_pop
            new_fit[-replace_num:] = random_fit
            new_cv[-replace_num:] = random_cv
            
            # 重置步长，允许在大范围内重新搜索
            self.step_size = 0.5 
        
        return new_pop, new_fit, new_cv

    def _sort_by_constraint_dominance(
        self,
        fit: ArrayLike,
        cv: ArrayLike
    ) -> List[int]:
        """
        按约束支配规则排序
        
        同时适用于有约束和无约束问题：
        - 有约束问题：可行解（cv=0）优先于不可行解（cv>0），
          可行解之间按适应度排序，不可行解之间按约束违反度排序
        - 无约束问题：所有解的cv=0，直接按适应度排序
        
        Args:
            fit: 适应度数组
            cv: 约束违反度数组（无约束问题时全为0）
            
        Returns:
            List[int]: 排序后的索引列表
        """
        # 检查是否为无约束问题（所有cv都为0）
        if np.all(cv <= 0):
            # 无约束问题：直接按适应度排序
            return sorted(
                range(len(fit)),
                key=lambda i: fit[i, 0]
            )
        else:
            # 有约束问题：按约束支配规则排序
            def constraint_sort_key(fitness: float, cv_val: float) -> Tuple:
                return (cv_val > 0, cv_val, fitness)
            
            return sorted(
                range(len(fit)),
                key=lambda i: constraint_sort_key(fit[i, 0], cv[i, 0])
            )

    def _unique_preserve_order(self, items: List[int]) -> List[int]:
        """
        去重并保持原始顺序
        
        Args:
            items: 原始列表
            
        Returns:
            List[int]: 去重后的列表
        """
        _, unique_indices = np.unique(items, return_index=True)
        return [items[i] for i in sorted(unique_indices)]

    def _restore_duplicates(
        self,
        pop: ArrayLike,
        fit: ArrayLike,
        cv: ArrayLike,
    repeat: NDArray[np.integer]
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        恢复去重时删除的重复个体
        
        Args:
            pop: 去重后的种群
            fit: 去重后的适应度
            cv: 去重后的约束违反度
            repeat: 每个唯一个体的重复次数
            
        Returns:
            Tuple: 恢复后的 (种群, 适应度, 约束违反度)
        """
        repeat = repeat - 1  # 减去原本保留的一个
        repeat_index = np.nonzero(repeat)[0]
        
        if len(repeat_index) > 0:
            pop = np.vstack((
                pop,
                np.repeat(pop[repeat_index, :], repeat[repeat_index], axis=0)
            ))
            fit = np.vstack((
                fit,
                np.repeat(fit[repeat_index, :], repeat[repeat_index], axis=0)
            ))
            cv = np.vstack((
                cv,
                np.repeat(cv[repeat_index, :], repeat[repeat_index], axis=0)
            ))
        
        return pop, fit, cv

    # ========================================================================
    # 聚类与局部搜索
    # ========================================================================
    
    def _clustering_and_learning(
        self,
        pop: ArrayLike,
        fit: ArrayLike,
        cv: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, List[int]]:
        """
        聚类和局部学习
        
        对种群进行聚类，对每个聚类的最优个体执行局部搜索。
        不同聚类方法（kmeans, meanshift, dbscan）会产生不同的聚类结果，
        从而影响局部搜索的对象选择。
        
        策略：
        1. 周期性触发（每 5 代）或停滞时触发
        2. 对每个聚类的最优个体进行局部搜索
        3. 如果聚类数 > niche_num，只选取最优的 niche_num 个聚类代表
        
        Args:
            pop: 当前种群
            fit: 适应度值
            cv: 约束违反度
            
        Returns:
            Tuple: (更新后的种群, 适应度, 约束违反度, 精英索引列表)
        """
        if self.n_gen is None:
            raise RuntimeError("算法状态未初始化，无法进行聚类学习")
        if self.problem is None:
            raise RuntimeError("问题未初始化，无法进行聚类学习")
        if not self.activate_method:
            return pop, fit, cv, []

        # 触发条件：周期性（每5代）或停滞时
        is_periodic = (self.n_gen % 5 == 0)
        is_stagnant = (self.stagnation_count >= 3)
        
        if not (is_periodic or is_stagnant):
            return pop, fit, cv, []

        if not self.check_bounds(pop):
            print("传入的 pop 越界")
            raise ValueError("种群越界")
        
        elite_id = []
        print(f"进行聚类学习... Gen: {self.n_gen}, Stagnation: {self.stagnation_count}, "
              f"Method: {self.cluster_method}")
        
        # 执行聚类
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        labels, n_clusters = self._perform_clustering(pop)
        print(f"聚类结果: {n_clusters} 个簇")
        
        # 收集每个聚类的最优个体索引（种群已按适应度排序）
        cluster_best_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # 在该聚类中按约束优先规则选最优个体
            cluster_sorted = self._sort_by_constraint_dominance(
                fit[cluster_indices],
                cv[cluster_indices]
            )
            best_in_cluster = cluster_indices[cluster_sorted[0]]
            cluster_best_indices.append(best_in_cluster)
        
        # 如果聚类数 > niche_num，只选取最优的 niche_num 个聚类代表
        if len(cluster_best_indices) > self.niche_num:
            # 按适应度排序（考虑约束违反度）
            def sort_key(idx):
                cv_val = cv[idx, 0]
                fit_val = fit[idx, 0]
                return (cv_val > 0, cv_val, fit_val)
            
            cluster_best_indices = sorted(cluster_best_indices, key=sort_key)
            cluster_best_indices = cluster_best_indices[:self.niche_num]
            print(f"聚类数 > niche_num ({self.niche_num})，选取前 {self.niche_num} 个最优聚类代表")
        
        # 对选中的聚类代表执行局部搜索
        for idx in cluster_best_indices:
            # 搜索深度：全局最优深度搜索，其他浅层搜索
            is_global_best = (idx == np.argmin(fit[:, 0]))
            search_depth = 10 if (is_global_best and is_stagnant) else 3
            
            fes_before = self.problem.fes
            
            # 执行局部搜索
            new_solution = self._local_search(
                pop[idx, :], 
                fit[idx, 0],
                maxiter=search_depth
            )
            
            fes_after = self.problem.fes
            
            # 评估更新
            new_fitness, new_cv = self.evaluate_fitness_cv(new_solution)
            
            improved = new_fitness < fit[idx, 0]
            print(f"  Cluster rep idx={idx} {'(Global Best)' if is_global_best else ''}: "
                  f"F: {fit[idx, 0]:.6e} -> {new_fitness.item():.6e}, "
                  f"FEs: {fes_after - fes_before}, "
                  f"{'IMPROVED' if improved else '-'}")
            
            # 只有改进时才更新
            if improved:
                pop[idx, :] = new_solution
                fit[idx, 0] = new_fitness
                cv[idx, 0] = new_cv
            
            elite_id.append(idx)
        
        return pop, fit, cv, elite_id

    def _perform_clustering(
        self,
        pop: ArrayLike
    ) -> Tuple[ArrayLike, int]:
        """
        执行聚类操作
        
        根据 self.cluster_method 选择相应的聚类算法。
        
        Args:
            pop: 种群数据
            
        Returns:
            Tuple[ArrayLike, int]: (聚类标签, 聚类数量)
        """
        if self.cluster_method == "kmeans":
            return self._kmeans_clustering(pop)
        elif self.cluster_method == "meanshift":
            return self._meanshift_clustering(pop)
        elif self.cluster_method == "dbscan":
            return self._dbscan_clustering(pop)
        else:
            raise ValueError(f"不支持的聚类方法: {self.cluster_method}")

    def _kmeans_clustering(self, pop: ArrayLike) -> Tuple[ArrayLike, int]:
        """
        K-means 聚类
        
        Args:
            pop: 种群数据
            
        Returns:
            Tuple[ArrayLike, int]: (聚类标签, 聚类数量)
        """
        if self.n_gen is None:
            raise RuntimeError("算法状态未初始化，无法进行聚类")
        if self.n_gen <= 2:
            kmeans = KMeans(n_clusters=self.niche_num, n_init=1, random_state=42)
        else:
            # 使用前一代的精英作为初始聚类中心
            if len(self.local_elite_id) == self.niche_num:
                init_centers = pop[self.local_elite_id, :]
                kmeans = KMeans(
                    n_clusters=self.niche_num,
                    init=init_centers,
                    n_init=1,
                    random_state=42
                )
            else:
                kmeans = KMeans(n_clusters=self.niche_num, n_init=1, random_state=42)
        
        labels = kmeans.fit_predict(pop)
        return labels, self.niche_num

    def _meanshift_clustering(self, pop: ArrayLike) -> Tuple[ArrayLike, int]:
        """
        MeanShift 聚类
        
        自动确定聚类数量，基于密度估计。
        
        Args:
            pop: 种群数据
            
        Returns:
            Tuple[ArrayLike, int]: (聚类标签, 聚类数量)
        """
        # 标准化数据以提高聚类效果
        scaler = StandardScaler()
        pop_scaled = scaler.fit_transform(pop)
        
        # 显式估算 bandwidth，提高稳定性
        # quantile 参数控制带宽大小：较小值产生更多簇，较大值产生更少簇
        # 对于优化问题，使用 0.2-0.3 比较合适，平衡簇的数量和大小
        bandwidth = estimate_bandwidth(pop_scaled, quantile=0.25, n_samples=min(500, len(pop)))
        
        # 如果bandwidth为None或太小，使用默认值
        if bandwidth is None or bandwidth <= 0:
            # 使用启发式方法：数据范围的1/10
            bandwidth = np.mean(np.std(pop_scaled, axis=0))
            print(f"bandwidth估算失败，使用启发式值: {bandwidth:.4f}")
        
        # 确保bandwidth不会太小
        bandwidth = max(bandwidth, 0.1)
        print(f"MeanShift bandwidth = {bandwidth:.4f}")
        
        try:
            meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            labels = meanshift.fit_predict(pop_scaled)
        except ValueError as e:
            # 如果仍然失败，回退到单簇
            print(f"MeanShift聚类失败: {e}，回退到单个簇")
            labels = np.zeros(len(pop), dtype=int)
            return labels, 1
        
        n_clusters = len(set(labels))
        
        # 如果聚类数量为 0 或 1，则视为仅有一个簇
        if n_clusters <= 1:
            print(f"MeanShift 仅找到 {n_clusters} 个簇，视为单簇")
            labels = np.zeros(len(pop), dtype=int)
            n_clusters = 1
        
        return labels, n_clusters

    def _dbscan_clustering(self, pop: ArrayLike) -> Tuple[ArrayLike, int]:
        """
        DBSCAN 聚类
        
        基于密度的聚类，可以发现任意形状的簇。
        噪声点（标签为 -1）会被分配到最近的簇。
        
        如果 eps 或 min_samples 为 None，将根据数据自动估算合适的参数。
        
        Args:
            pop: 种群数据
            
        Returns:
            Tuple[ArrayLike, int]: (聚类标签, 聚类数量)
        """
        # 标准化数据
        scaler = StandardScaler()
        pop_scaled = scaler.fit_transform(pop)
        
        # 自动估算 eps（如果未指定）
        if self.dbscan_eps is None:
            # 使用 k-距离图方法估算 eps
            # 选择 k = min(4, pop_size/2)，计算每个点到第 k 个最近邻的距离
            k = min(4, max(2, len(pop) // 2))
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors_fit = neighbors.fit(pop_scaled)
            distances, _ = neighbors_fit.kneighbors(pop_scaled)
            distances = np.sort(distances, axis=0)
            k_distances = distances[:, -1]  # 取第 k 个最近邻距离
            # 使用 65% 分位数作为 eps 的估计值（比中位数稍大，减少噪声点）
            eps_estimated = float(np.percentile(k_distances, 65))
            # 确保eps不为0，设置最小值为0.01
            eps_estimated = max(eps_estimated, 0.01)
            print(f"自动估算 DBSCAN eps = {eps_estimated:.4f}")
            eps = eps_estimated
        else:
            eps = self.dbscan_eps
        
        # 自动估算 min_samples（如果未指定）
        if self.dbscan_min_samples is None:
            # 基于种群大小和维度设置
            # 对于 pop_size=100，期望 3 个簇，每个簇约 33 个样本
            # min_samples 设为 max(3, pop_size/30, dim) 更合理，避免过多噪声点
            dim = pop.shape[1]
            min_samples_estimated = max(3, len(pop) // 30, dim)
            print(f"自动估算 DBSCAN min_samples = {min_samples_estimated}")
            min_samples = min_samples_estimated
        else:
            min_samples = self.dbscan_min_samples
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(pop_scaled)
        
        # 处理噪声点（标签为 -1）
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.discard(-1)
            
            if len(unique_labels) == 0:
                # 所有点都是噪声，视为单簇
                print("DBSCAN 未找到有效簇，视为单簇")
                labels = np.zeros(len(pop), dtype=int)
                n_clusters = 1
            else:
                # 将噪声点分配到最近的簇
                noise_indices = np.where(labels == -1)[0]
                valid_indices = np.where(labels != -1)[0]
                
                for noise_idx in noise_indices:
                    # 找到最近的非噪声点
                    distances = np.linalg.norm(
                        pop_scaled[valid_indices] - pop_scaled[noise_idx], 
                        axis=1
                    )
                    nearest_idx = valid_indices[np.argmin(distances)]
                    labels[noise_idx] = labels[nearest_idx]
                
                n_clusters = len(unique_labels)
        else:
            n_clusters = len(unique_labels)
        
        # 如果聚类数量为 0 或 1，则视为单簇
        if n_clusters <= 1:
            print("DBSCAN 聚类数量为 0 或 1，视为单簇")
            labels = np.zeros(len(pop), dtype=int)
            n_clusters = 1
        
        return labels, n_clusters

    def _local_search(
        self,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        maxiter: int = 1
    ) -> ArrayLike:
        """
        局部搜索
        
        使用指定的优化方法对初始解进行局部优化。
        
        Args:
            x0: 初始解
            y0: 初始目标函数值（用于 Adam 方法）
            maxiter: 最大迭代次数
            
        Returns:
            ArrayLike: 优化后的解
            
        Raises:
            ValueError: 当指定了不支持的优化方法时
        """
        if not self.check_bounds([x0]):
            print("传入的 x0 越界")
            raise ValueError("初始解越界")
        
        def objective(x: ArrayLike) -> float:
            """带约束惩罚的目标函数"""
            # 先检查历史记录中是否已存在该解（极其接近）
            history_result = self.history.get(x)
            if history_result is not None:
                # 如果历史记录中存在，直接使用历史记录的值，避免重复评估
                fitness, cv = history_result
            else:
                # 如果历史记录中不存在，正常评估（会自动添加到 history）
                fitness, cv = self.evaluate_fitness_cv(x)
            
            # 计算惩罚系数和目标函数值
            fitness_array = np.atleast_1d(fitness)
            alpha = abs(fitness_array).mean() * 10
            fitness_val = float(fitness_array.item()) if fitness_array.size == 1 else float(fitness_array.mean())
            return fitness_val + alpha * cv
        y0 = objective(x0)
        
        # 执行局部搜索，获取结果
        if self.method == "Nelder-Mead":
            x_result = self._nelder_mead_local_search(x0, y0, objective, maxiter=3)
        elif self.method == "rbf":
            x_result = self._rbf_local_search(x0, y0, objective, maxiter=maxiter)
        elif self.method == "gp":
            x_result = self._gp_local_search(x0, y0, objective, maxiter=3)
        elif self.method == "history-ladder":
            x_result = self._history_ladder_local_search(x0, y0, objective, maxiter=maxiter)
        elif self.method in self.BOUNDED_METHODS:
            x_result = self._scipy_local_search(x0, objective, maxiter=1)
        elif self.method == "Adam":
            x_result = self._adam_local_search(x0, y0, objective)  # Adam 暂不支持动态 maxiter
        elif self.method == "AdamW":
            x_result = self._adamw_local_search(x0, y0, objective)  # AdamW 暂不支持动态 maxiter
        elif self.method == "Lion":
            x_result = self._lion_local_search(x0, y0, objective)  # Lion 暂不支持动态 maxiter
        elif self.method == "Sophia":
            x_result = self._sophia_local_search(x0, y0, objective)  # Sophia 暂不支持动态 maxiter
        else:
            raise ValueError(f"不支持的优化方法: {self.method}")
        
        # 使用约束优先规则判断是否改进（不重复评估）
        hist_x0 = self.history.get(x0)
        hist_x1 = self.history.get(x_result)
        if hist_x0 is None or hist_x1 is None:
            return x0
        f0, cv0 = hist_x0
        f1, cv1 = hist_x1
        fit_compare = np.vstack([np.atleast_1d(f0), np.atleast_1d(f1)])
        cv_compare = np.vstack([np.atleast_1d(cv0), np.atleast_1d(cv1)])
        best_idx = self._sort_by_constraint_dominance(fit_compare, cv_compare)[0]
        return x_result if best_idx == 1 else x0

    def _scipy_local_search(
        self,
        x0: ArrayLike,
        objective: Callable,
        maxiter: int = 1
    ) -> ArrayLike:
        """
        使用 SciPy 优化器进行局部搜索
        
        Args:
            x0: 初始解
            objective: 目标函数
            maxiter: 最大迭代次数
            
        Returns:
            ArrayLike: 优化后的解
        """
        bounds = [(self.lb[i], self.ub[i]) for i in range(self.dim)]
        x0 = np.clip(x0, self.lb, self.ub)
        
        minimize_kwargs = {
            "fun": objective,
            "x0": x0,
            "method": self.method,
            "bounds": bounds
        }
        
        # 设置优化选项
        if self.method in ["L-BFGS-B", "SLSQP", "Powell", "trust-constr"]:
            minimize_kwargs["options"] = {"maxiter": maxiter}
        else:
            minimize_kwargs["options"] = {"maxfun": 100 * maxiter, "disp": False}
        
        result = scipy_minimize(**minimize_kwargs)
        return np.clip(result.x, self.lb, self.ub)

    def _adam_local_search(
        self,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        objective: Callable
    ) -> ArrayLike:
        """
        使用 Adam 优化器进行局部搜索
        
        Args:
            x0: 初始解
            y0: 初始目标函数值
            objective: 目标函数
            
        Returns:
            ArrayLike: 优化后的解
        """
        return self._adam_optimize(
            f=objective,
            x0=x0,
            y0=y0,
            lb=self.lb,
            ub=self.ub,
            max_iter=self.dim,
            lr=0.01
        )

    def _adamw_local_search(
        self,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        objective: Callable
    ) -> ArrayLike:
        """
        使用 AdamW 优化器进行局部搜索
        
        AdamW 是 Adam 的改进版本，将权重衰减与梯度解耦，通常能获得更好的泛化性能。
        
        Args:
            x0: 初始解
            y0: 初始目标函数值
            objective: 目标函数
            
        Returns:
            ArrayLike: 优化后的解
        """
        return self._adamw_optimize(
            f=objective,
            x0=x0,
            y0=y0,
            lb=self.lb,
            ub=self.ub,
            max_iter=self.dim,
            lr=0.01,
            weight_decay=1e-4
        )

    def _lion_local_search(
        self,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        objective: Callable
    ) -> ArrayLike:
        """
        使用 Lion 优化器进行局部搜索
        
        Lion (EvoLved Sign Momentum) 是 Google 提出的优化算法，比 Adam 更简单高效：
        - 只跟踪动量，不需要二阶矩估计
        - 使用符号函数更新参数
        - 计算开销更小，内存占用更少
        
        Args:
            x0: 初始解
            y0: 初始目标函数值
            objective: 目标函数
            
        Returns:
            ArrayLike: 优化后的解
        """
        return self._lion_optimize(
            f=objective,
            x0=x0,
            y0=y0,
            lb=self.lb,
            ub=self.ub,
            max_iter=self.dim,
            lr=0.01,
            weight_decay=1e-4
        )

    def _sophia_local_search(
        self,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        objective: Callable
    ) -> ArrayLike:
        """
        使用 Sophia 优化器进行局部搜索
        
        Sophia (Second-order Clipped Stochastic Optimization) 是斯坦福大学提出的优化算法：
        - 使用二阶信息（Hessian 对角线近似）
        - 使用裁剪机制控制更新步长
        - 结合自适应学习率和二阶信息，在某些任务上表现优于 Adam
        
        Args:
            x0: 初始解
            y0: 初始目标函数值
            objective: 目标函数
            
        Returns:
            ArrayLike: 优化后的解
        """
        return self._sophia_optimize(
            f=objective,
            x0=x0,
            y0=y0,
            lb=self.lb,
            ub=self.ub,
            max_iter=self.dim,
            lr=0.01,
            weight_decay=1e-4
        )

    def _rbf_local_search(
        self,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        objective: Callable,
        maxiter: int = 1
    ) -> ArrayLike:
        """
        RBF 代理模型局部搜索
        
        从历史评估数据中找到离 x0 最近的点，构建 RBF 代理模型，
        然后在代理模型上进行优化。
        
        Args:
            x0: 初始解
            y0: 初始目标函数值（未使用，保留接口一致性）
            objective: 目标函数（已在 _local_search 中定义）
            maxiter: 最大迭代次数
            
        Returns:
            ArrayLike: 优化后的解
        """
        if not self.check_bounds([x0]):
            print("传入的 x0 越界")
            raise ValueError("初始解越界")
        
        # 检查是否有足够的历史数据
        if not hasattr(self, 'history') or self.history is None or len(self.history) < max(10, self.dim * 2):
            # 历史数据不足，直接返回原值
            print(f"历史数据不足 ({len(self.history) if hasattr(self, 'history') and self.history else 0} < {max(10, self.dim * 2)})，返回原值")
            return x0.copy()
        
        # 获取历史数据
        X_hist, F_hist, CV_hist = self.history.get_all()
        
        if len(X_hist) < max(5, self.dim):
            # 历史点太少，直接返回原值
            print(f"历史点不足 ({len(X_hist)} < {max(5, self.dim)})，返回原值")
            return x0.copy()
        
        # 找到离 x0 最近的邻居点
        # 使用最近邻方法，选择足够多的点用于构建代理模型
        n_neighbors = min(200, len(X_hist), max(50, self.dim * 5))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X_hist)
        distances, indices = nn.kneighbors(x0.reshape(1, -1))
        
        # 获取最近的邻居点
        X_nearby = X_hist[indices[0]]
        
        # 使用传入的 objective 函数计算历史数据的目标值（已包含约束惩罚）
        y_nearby = np.array([objective(x) for x in X_nearby])
        
        # 构建 RBF 代理模型
        try:
            # 直接拟合目标值（已包含约束惩罚）
            rbf = RBFInterpolator(
                X_nearby,
                y_nearby,
                kernel="thin_plate_spline",
                smoothing=0.0,
                neighbors=None
            )
        except Exception as e:
            # RBF 拟合失败，直接返回原值
            print(f"RBF 拟合失败: {e}，返回原值")
            return x0.copy()
        
        # 定义代理目标函数
        def surrogate_objective(x: ArrayLike) -> float:
            """使用 RBF 代理模型的目标函数"""
            x = np.clip(x, self.lb, self.ub)
            try:
                y_pred = rbf(x.reshape(1, -1))[0]
                return float(y_pred)
            except Exception:
                # 预测失败，使用最近邻方法
                nn_single = NearestNeighbors(n_neighbors=1)
                nn_single.fit(X_nearby)
                _, idx = nn_single.kneighbors(x.reshape(1, -1))
                return float(y_nearby[idx[0, 0]])
        
        # 使用 L-BFGS-B 在代理模型上优化
        bounds = [(self.lb[i], self.ub[i]) for i in range(self.dim)]
        x0_clipped = np.clip(x0, self.lb, self.ub)

        try:
            result = scipy_minimize(
                fun=surrogate_objective,
                x0=x0_clipped,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter},
            )

            optimized_x = np.clip(result.x, self.lb, self.ub)
            return optimized_x

        except Exception as e:
            warnings.warn(f"RBF 代理模型优化失败: {e}，返回原值")
            return x0.copy()

    def _gp_local_search(
        self,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        objective: Callable,
        maxiter: int = 1
    ) -> ArrayLike:
        """
        GP 代理模型局部搜索
        
        从历史评估数据中找到离 x0 最近的点，构建高斯过程（GP）代理模型，
        使用 LCB（Lower Confidence Bound）作为获取函数，然后在代理模型上进行优化。
        
        Args:
            x0: 初始解
            y0: 初始目标函数值（未使用，保留接口一致性）
            objective: 目标函数（已在 _local_search 中定义）
            maxiter: 最大迭代次数
            
        Returns:
            ArrayLike: 优化后的解
        """
        if not self.check_bounds([x0]):
            print("传入的 x0 越界")
            raise ValueError("初始解越界")
        
        # 检查是否有足够的历史数据
        if not hasattr(self, 'history') or self.history is None or len(self.history) < max(10, self.dim * 2):
            # 历史数据不足，直接返回原值
            print(f"历史数据不足 ({len(self.history) if hasattr(self, 'history') and self.history else 0} < {max(10, self.dim * 2)})，返回原值")
            return x0.copy()
        
        # 获取历史数据
        X_hist, F_hist, CV_hist = self.history.get_all()
        
        if len(X_hist) < max(5, self.dim):
            # 历史点太少，直接返回原值
            print(f"历史点不足 ({len(X_hist)} < {max(5, self.dim)})，返回原值")
            return x0.copy()
        
        # 找到离 x0 最近的邻居点
        # 使用最近邻方法，选择足够多的点用于构建代理模型
        n_neighbors = min(200, len(X_hist), max(50, self.dim * 5))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(X_hist)
        distances, indices = nn.kneighbors(x0.reshape(1, -1))
        
        # 1. 直接通过索引获取原始数据 (最安全，避开浮点数匹配风险)
        X_nearby = X_hist[indices[0]]
        F_nearby = F_hist[indices[0]]   # 形状应该是 (N,) 或 (N, 1)
        CV_nearby = CV_hist[indices[0]] # 形状应该是 (N,) 或 (N, 1)

        # 确保是一维数组
        F_nearby = F_nearby.flatten()
        CV_nearby = CV_nearby.flatten()

        # 2. 计算统一的局部惩罚系数 Alpha (关键优化)
        # 使用这组邻居的平均适应度来确定当前局部区域的惩罚力度
        # 这样可以保证 GP 拟合的是一个"惩罚标准统一"的光滑曲面
        avg_fitness = np.mean(np.abs(F_nearby))
        local_alpha = avg_fitness * 10 if avg_fitness > 1e-6 else 10.0 # 防止除零或过小

        # 3. 向量化计算 y_train (极速)
        # 这里模拟了你 objective 中的逻辑，但是是批量、统一处理
        y_train_raw = F_nearby + local_alpha * CV_nearby

        # 4. 数据标准化 (Standardization) - GP 必须步骤
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = StandardScaler()

        # 注意处理标准化可能抛出的异常（例如所有点y值都一样）
        try:
            X_train = scaler_X.fit_transform(X_nearby)
            y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
        except ValueError:
            return x0.copy()

        # 5. 构建 GP 代理模型
        try:
            # 使用 Matern 核，WhiteKernel 处理噪声/平滑
            kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
            
            # 拟合 GP 模型（自动优化超参数）
            gp.fit(X_train, y_train.flatten())
        except Exception as e:
            # GP 拟合失败，直接返回原值
            print(f"GP 拟合失败: {e}，返回原值")
            return x0.copy()
        
        # 6. 定义代理目标函数（使用 LCB 获取函数）
        def surrogate_objective(x_scaled: ArrayLike) -> float:
            """
            使用 GP 代理模型的 LCB 目标函数
            
            Args:
                x_scaled: 标准化后的决策变量（在 [0, 1] 范围内）
            
            Returns:
                float: LCB 值（均值 - 2倍标准差）
            """
            x_scaled = np.clip(x_scaled, 0, 1)  # 确保在标准化范围内
            try:
                # 预测时获取均值和标准差
                pred = gp.predict(x_scaled.reshape(1, -1), return_std=True)
                if isinstance(pred, tuple):
                    mu = pred[0]
                    std = pred[1]
                else:
                    mu = pred
                    std = np.zeros_like(mu)
                mu = float(np.atleast_1d(mu)[0])
                std = float(np.atleast_1d(std)[0])
                
                # LCB: 均值 - 2倍标准差 (鼓励探索未知区域)
                lcb = mu - 2.0 * std
                return float(lcb)
            except Exception:
                # 预测失败，使用最近邻方法
                nn_single = NearestNeighbors(n_neighbors=1)
                nn_single.fit(X_train)
                _, idx = nn_single.kneighbors(x_scaled.reshape(1, -1))
                return float(y_train[idx[0, 0]])
        
        # 7. 将 x0 标准化到 [0, 1] 范围
        x0_scaled = scaler_X.transform(x0.reshape(1, -1))[0]
        x0_scaled = np.clip(x0_scaled, 0, 1)

        # 8. 使用 L-BFGS-B 在代理模型上优化（在标准化空间中）
        bounds_scaled = [(0.0, 1.0) for _ in range(self.dim)]

        try:
            result = scipy_minimize(
                fun=surrogate_objective,
                x0=x0_scaled,
                method="L-BFGS-B",
                bounds=bounds_scaled,
                options={"maxiter": maxiter},
            )

            # 9. 将优化结果从标准化空间转换回原始空间
            x_optimized_scaled = np.clip(result.x, 0, 1)
            x_optimized = scaler_X.inverse_transform(
                x_optimized_scaled.reshape(1, -1)
            )[0]

            # 10. 确保在原始边界内
            optimized_x = np.clip(x_optimized, self.lb, self.ub)
            return optimized_x

        except Exception as e:
            warnings.warn(f"GP 代理模型优化失败: {e}，返回原值")
            return x0.copy()

    def _estimate_gradient_from_history(
        self,
        x0: ArrayLike,
        x0_fitness: float
    ) -> Optional[Tuple[ArrayLike, ArrayLike, Optional[ArrayLike]]]:
        """
        利用历史数据估计 x0 处的梯度方向（构建阶梯）
        
        通过找到 x0 的最近邻，使用最小二乘法拟合局部线性模型，
        估计梯度方向，从而确定下降方向。
        
        Args:
            x0: 当前中心点
            x0_fitness: 当前点的适应度值
            
        Returns:
            Optional[Tuple[ArrayLike, ArrayLike, Optional[ArrayLike]]]: 
                如果成功，返回 (下降方向, 邻居点, 残差)；
                如果无法估计则返回 None
        """
        # 1. 检查历史数据是否足够
        # 要拟合 D 维空间的平面，至少需要 D+1 个点
        if not hasattr(self, 'history') or self.history is None:
            return None
        
        X_hist, F_hist, _ = self.history.get_all()
        
        if len(X_hist) < self.dim + 2:
            return None
        
        # 2. 找到最近的 K 个邻居
        # K 的选择很关键：太少拟合不准，太多会引入非局部干扰
        # 推荐 K = 2 * dim (两倍维度) 到 3 * dim
        k_neighbors = min(len(X_hist), max(self.dim * 2 + 1, 10))
        
        try:
            nn = NearestNeighbors(n_neighbors=k_neighbors)
            nn.fit(X_hist)
            
            # 这里的 x0 需要 reshape
            dists, indices = nn.kneighbors(x0.reshape(1, -1))
            
            # 取出邻居（排除 x0 自身，如果它在历史里）
            # 这里的 indices[0] 包含了 k 个邻居的索引
            neighbors_X = X_hist[indices[0]]
            neighbors_y = F_hist[indices[0]].flatten()
            
            # 3. 构建线性回归方程: Delta_y = g * Delta_x
            # 我们计算所有邻居相对于 x0 的位移和适应度差
            
            # Delta X: shape (K, Dim)
            Delta_X = neighbors_X - x0
            
            # Delta y: shape (K,)
            Delta_y = neighbors_y - x0_fitness
            
            # 4. 使用最小二乘法求解梯度 g
            # 目标：最小化 || Delta_X * g - Delta_y ||^2
            # g 的方向就是适应度增加最快的方向，所以下降方向是 -g
            try:
                # lstsq 比求逆矩阵更稳定，能处理共线情况
                result = lstsq(Delta_X, Delta_y)
                if result is None:
                    return None
                g, residuals, rank, s = result
            except Exception:
                return None  # 计算失败（如矩阵奇异）
            
            # 5. 处理梯度
            norm = np.linalg.norm(g)
            if norm < 1e-9:
                return None  # 梯度极其微小（平坦区域）
            
            # 返回负梯度方向（下坡方向）、邻居点和残差
            descent_direction = -g / norm
            
            # residuals 可能是标量或数组，统一处理
            if isinstance(residuals, np.ndarray) and residuals.size > 0:
                residuals_value = residuals
            elif isinstance(residuals, (list, tuple)) and len(residuals) > 0:
                residuals_value = np.array(residuals)
            else:
                residuals_value = None
            
            return (descent_direction, neighbors_X, residuals_value)
            
        except Exception as e:
            # 任何异常都返回 None
            return None

    def _calculate_smart_step_size(
        self,
        x0: ArrayLike,
        direction: ArrayLike,
        neighbors_X: ArrayLike,
        residuals: Optional[ArrayLike] = None
    ) -> float:
        """
        计算基于信任域的自适应步长
        
        使用局部邻居的分布来确定信任域半径，从而自适应地计算步长。
        这种方法能够根据局部地形的复杂度自动调整步长大小。
        
        Args:
            x0: 当前中心点
            direction: 计算出的单位梯度方向
            neighbors_X: 用于拟合梯度的历史邻居点（包含 x0 的邻居）
            residuals: 线性回归的残差（可选，用于衡量地形复杂度）
            
        Returns:
            float: 推荐的步长
        """
        # 1. 计算局部信任域半径 (Radius)
        # 计算所有邻居到 x0 的欧氏距离
        # 注意：neighbors_X 应该只包含那 K 个最近邻，而不是全部历史
        dists = np.linalg.norm(neighbors_X - x0, axis=1)
        
        # 取最大距离作为"视野半径"
        # 如果邻居很聚拢，radius 就小；邻居很分散，radius 就大
        # 这天然实现了"早期大步，晚期小步"的效果
        if len(dists) == 0:
            # 如果没有邻居，使用启发式值
            trust_radius = np.mean(self.ub - self.lb) * 0.05
        else:
            trust_radius = np.max(dists)
        
        # 2. 基础步长设定
        # 保守策略：通常不超过半径的 1.0 倍，防止跳出数据支持的范围
        # 激进系数 (alpha)：0.5 (保守) ~ 1.2 (激进)
        alpha = 0.8
        step_size = trust_radius * alpha
        
        # 3. (可选) 基于拟合质量的动态调整
        # 如果 residuals 很大，说明地形不平坦，缩减步长
        if residuals is not None:
            try:
                # 计算残差的范数
                if isinstance(residuals, np.ndarray):
                    residual_norm = np.linalg.norm(residuals)
                else:
                    residual_norm = abs(float(residuals))
                
                # 如果残差相对较大，说明线性拟合质量差，缩减步长
                # 使用启发式：如果残差 > 步长带来的预期下降，说明不可信
                # 这里简化处理：残差越大，缩减因子越小
                if residual_norm > 0:
                    # 归一化残差（相对于适应度尺度）
                    # 这里使用一个简单的启发式：如果残差 > trust_radius，则缩减
                    if residual_norm > trust_radius:
                        reduction_factor = trust_radius / residual_norm
                        step_size = step_size * min(reduction_factor, 0.8)
            except Exception:
                # 如果计算残差因子失败，忽略
                pass
        
        # 4. (关键) 防止步长过小或过大
        # 最小值：参数范围的 0.1% (防止在极微小范围内死循环)
        domain_span = self.ub - self.lb
        min_step = np.mean(domain_span) * 0.001
        
        # 最大值：不能一步跨越整个搜索空间，限制在范围的 10%
        max_step = np.mean(domain_span) * 0.1
        
        # 确保 step_size 在合理范围内
        # 如果邻居太近(收敛后期)，trust_radius 可能小于 min_step，此时强制维持最小勘探步长
        step_size = np.clip(step_size, min_step, max_step)
        
        return float(step_size)

    def _history_ladder_local_search(
        self,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        objective: Callable,
        maxiter: int = 1
    ) -> ArrayLike:
        """
        历史阶梯跳跃局部搜索
        
        利用历史数据估计梯度方向，然后沿下降方向进行一步或多步跳跃。
        这是一种轻量级的局部搜索方法，特别适合历史数据丰富的情况。
        
        Args:
            x0: 初始解
            y0: 初始目标函数值（用于获取初始适应度）
            objective: 目标函数（带约束惩罚）
            maxiter: 最大迭代次数（跳跃步数）
            
        Returns:
            ArrayLike: 优化后的解
        """
        if not self.check_bounds([x0]):
            print("传入的 x0 越界")
            raise ValueError("初始解越界")
        
        # 检查历史数据是否足够
        if not hasattr(self, 'history') or self.history is None:
            return x0.copy()
        
        X_hist, _, _ = self.history.get_all()
        if len(X_hist) < self.dim + 2:
            return x0.copy()
        
        # 获取 x0 的初始适应度（带约束惩罚）
        fitness_x0 = objective(x0)
        x_current = x0.copy()
        fitness_current = fitness_x0
        
        # 迭代进行阶梯跳跃
        # 如果没有 break（一直有改进），就一直进行，不受 maxiter 限制
        iter_step = 0
        while True:
            iter_step += 1
            
            # 1. 从历史中"借"来梯度方向
            # 注意：这里使用 objective 函数的值，因为它已经包含了约束惩罚
            result = self._estimate_gradient_from_history(x_current, fitness_current)
            
            if result is None:
                # 无法构建阶梯，停止迭代
                break
            
            # 解包返回值：方向、邻居点、残差
            direction, neighbors_X, residuals = result
            
            # 2. 使用智能步长计算方法
            # 基于信任域的自适应步长，考虑局部邻居分布和拟合质量
            initial_step_size = self._calculate_smart_step_size(
                x_current, 
                direction, 
                neighbors_X, 
                residuals
            )
            
            # 3. 尝试两次：先尝试 initial_step_size，失败则尝试 initial_step_size * 0.5
            improved = False
            
            # 第一次尝试：使用初始步长
            step_size = initial_step_size
            x_candidate = x_current + direction * step_size
            x_candidate = np.clip(x_candidate, self.lb, self.ub)
            fitness_candidate = objective(x_candidate)
            
            if fitness_candidate < fitness_current:
                x_current = x_candidate
                fitness_current = fitness_candidate
                improved = True
            else:
                # 第二次尝试：使用一半步长
                step_size = initial_step_size * 0.5
                x_candidate = x_current + direction * step_size
                x_candidate = np.clip(x_candidate, self.lb, self.ub)
                fitness_candidate = objective(x_candidate)
                
                if fitness_candidate < fitness_current:
                    x_current = x_candidate
                    fitness_current = fitness_candidate
                    improved = True
            
            # 如果两次尝试都失败，停止迭代
            if not improved:
                break
            # 如果有改进，继续下一次迭代（不受 maxiter 限制，一直进行下去）
        
        return x_current

    def _nelder_mead_local_search(
        self,
        x0: ArrayLike,
        y0: float,  # 保留接口一致性，但本方法内部不再使用
        objective: Callable,
        maxiter: int = 1
    ) -> ArrayLike:
        """
        利用历史信息构建单纯形的 Nelder-Mead 搜索 (无缓存版)
        特点：利用正交基检查防止单纯形退化，自动兜底补齐维度
        """
        dim = self.dim  # 使用 self.dim 保持一致性
        
        # --- 1. 初始单纯形构建容器 ---
        # simplex_points: 存储顶点，首点必须是 x0
        simplex_points = [x0.copy()]
        # ortho_basis: 存储已选方向的标准正交基，用于检测线性无关性
        ortho_basis = [] 
        
        # --- 2. 尝试从历史构建 (筛选高质量邻居) ---
        if hasattr(self, 'history') and self.history is not None:
            X_hist, _, _ = self.history.get_all()
            
            # 检查历史数据是否足够且非空
            if len(X_hist) > 0 and len(X_hist) >= dim + 1:
                try:
                    # 寻找 x0 的最近邻 (多找几倍以备过滤)
                    n_neighbors = min(len(X_hist), dim * 5 + 2)
                    neigh = NearestNeighbors(n_neighbors=n_neighbors)
                    neigh.fit(X_hist)
                    dists, indices = neigh.kneighbors(x0.reshape(1, -1))
                    
                    # 遍历候选邻居，筛选出线性无关的点
                    for idx in indices[0]:
                        cand = X_hist[idx]
                        vec = cand - x0
                        
                        # A. 距离过滤：跳过重合点
                        if np.linalg.norm(vec) < 1e-6:
                            continue
                            
                        # B. 线性无关过滤 (Gram-Schmidt 正交化检查)
                        # 将向量 vec 投影到现有基的补空间
                        u = vec.copy()
                        for b in ortho_basis:
                            u -= np.dot(u, b) * b
                        
                        resid_norm = np.linalg.norm(u)
                        
                        # 如果残差显著(>1e-5)，说明提供了新的维度方向
                        if resid_norm > 1e-5:
                            simplex_points.append(cand)
                            ortho_basis.append(u / resid_norm) # 更新正交基
                        
                        if len(simplex_points) >= dim + 1:
                            break
                except Exception as e:
                    print(f"历史单纯形构建警告: {e}")

        # --- 3. 兜底补齐 (防止历史点不足或共线) ---
        # 如果历史点没凑够，沿标准轴扰动补齐，确保单纯形满秩
        if len(simplex_points) < dim + 1:
            eye = np.eye(dim)
            # 计算自适应步长，考虑边界距离
            range_sizes = self.ub - self.lb
            min_range = np.min(range_sizes)
            # 使用相对步长，但至少保证最小有效步长
            base_step = max(min_range * 0.05, 1e-6)
            
            for i in range(dim):
                if len(simplex_points) >= dim + 1:
                    break
                
                # 尝试沿第 i 轴扰动，考虑边界距离
                # 优先向远离边界的方向扰动
                dist_to_lb = x0[i] - self.lb[i]
                dist_to_ub = self.ub[i] - x0[i]
                
                # 选择有足够空间的方向
                if dist_to_ub > dist_to_lb and dist_to_ub > base_step:
                    step_vec = eye[i] * base_step
                elif dist_to_lb > base_step:
                    step_vec = -eye[i] * base_step
                else:
                    # 如果两个方向空间都不足，尝试双向扰动
                    step_vec = eye[i] * min(dist_to_ub, dist_to_lb, base_step)
                    if step_vec[i] < 1e-8:
                        # 如果空间实在太小，跳过这个维度，稍后用随机扰动
                        continue
                
                # 同样进行线性无关检查 (防止轴向扰动与历史点意外共线)
                u = step_vec.copy()
                for b in ortho_basis:
                    u -= np.dot(u, b) * b
                
                if np.linalg.norm(u) > 1e-6:
                    new_p = np.clip(x0 + step_vec, self.lb, self.ub)
                    # 再次确认 clip 后没有重合
                    if np.linalg.norm(new_p - x0) > 1e-6:
                        simplex_points.append(new_p)
                        ortho_basis.append(u / np.linalg.norm(u))
            
            # 如果仍然不够，使用随机扰动补齐
            if len(simplex_points) < dim + 1:
                attempts = 0
                max_attempts = dim * 10
                while len(simplex_points) < dim + 1 and attempts < max_attempts:
                    attempts += 1
                    # 随机方向扰动
                    random_dir = self._rng.normal(size=dim)
                    random_dir = random_dir / np.linalg.norm(random_dir)
                    step_size = base_step * (0.5 + self._rng.random())
                    step_vec = random_dir * step_size
                    
                    # 线性无关检查
                    u = step_vec.copy()
                    for b in ortho_basis:
                        u -= np.dot(u, b) * b
                    
                    if np.linalg.norm(u) > 1e-6:
                        new_p = np.clip(x0 + step_vec, self.lb, self.ub)
                        if np.linalg.norm(new_p - x0) > 1e-6:
                            simplex_points.append(new_p)
                            ortho_basis.append(u / np.linalg.norm(u))

        # --- 4. 验证单纯形有效性 ---
        # 确保单纯形有足够的点且满秩
        if len(simplex_points) < dim + 1:
            # 如果仍然不够，使用 SciPy 默认的初始单纯形构建方法
            # 这种情况下不提供 initial_simplex，让 SciPy 自动生成
            initial_simplex = None
        else:
            simplex_array = np.array(simplex_points)
            # 验证单纯形是否满秩（检查从 x0 出发的向量是否线性无关）
            vectors = simplex_array[1:] - simplex_array[0:1]
            if vectors.shape[0] >= dim:
                # 检查前 dim 个向量是否线性无关
                rank = np.linalg.matrix_rank(vectors[:dim])
                if rank < dim:
                    # 如果不满秩，使用 SciPy 默认方法
                    initial_simplex = None
                else:
                    initial_simplex = simplex_array
            else:
                initial_simplex = None

        # --- 5. 执行优化 ---
        options = {
            'maxiter': maxiter,
            'disp': False,
            'xatol': 1e-5,
            'fatol': 1e-5
        }
        if initial_simplex is not None:
            options['initial_simplex'] = initial_simplex

        try:
            # 直接调用 SciPy，依赖您修改后的 objective 处理评估逻辑
            # bounds 参数确保搜索在可行域内 (SciPy 1.7+ 有效)
            res = scipy_minimize(
                objective,
                x0,
                method='Nelder-Mead',
                bounds=[(l, u) for l, u in zip(self.lb, self.ub)],
                options=options
            )
            return np.clip(res.x, self.lb, self.ub)
            
        except Exception as e:
            print(f"NM 优化异常: {e}")
            return np.clip(x0, self.lb, self.ub)

    def _adam_optimize(
        self,
        f: Callable,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        lb: ArrayLike,
        ub: ArrayLike,
        max_iter: int = 100,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_eps: float = 1e-6,
        tol: float = 1e-6,
        verbose: bool = False
    ) -> ArrayLike:
        """
        Adam 优化算法实现
        
        Args:
            f: 目标函数
            x0: 初始解
            y0: 初始函数值（未使用，保留接口兼容）
            lb: 下界
            ub: 上界
            max_iter: 最大迭代次数
            lr: 学习率
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            eps: 数值稳定性常数
            grad_eps: 梯度估计的扰动量
            tol: 收敛阈值
            verbose: 是否打印详细信息
            
        Returns:
            ArrayLike: 优化后的解
        """
        x = x0.copy()
        m = np.zeros_like(x)  # 一阶矩
        v = np.zeros_like(x)  # 二阶矩
        
        for t in range(1, max_iter + 1):
            # 有限差分估计梯度
            grad = self._approximate_gradient(f, x, lb, ub, grad_eps)
            
            # 边界投影梯度（使用数值容差判断边界）
            grad = self._project_gradient(grad, x, lb, ub, tol=1e-8)
            
            # 梯度收敛检查
            if np.linalg.norm(grad) < tol:
                if verbose:
                    print(f"[Adam] 收敛于第 {t} 次迭代，梯度范数为 {np.linalg.norm(grad):.3e}")
                break
            
            # Adam 更新
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)  # 偏差修正
            v_hat = v / (1 - beta2 ** t)
            
            # 参数更新
            x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
            x_new = np.clip(x_new, lb, ub)
            
            # 位移收敛检查
            if np.linalg.norm(x_new - x) < tol:
                if verbose:
                    print(f"[Adam] 收敛于第 {t} 次迭代，Δx 范数为 {np.linalg.norm(x_new - x):.3e}")
                x = x_new
                break
            
            x = x_new
        
        return x

    @staticmethod
    def _approximate_gradient(
        f: Callable,
        x: ArrayLike,
        lb: Optional[ArrayLike],
        ub: Optional[ArrayLike],
        eps: float = 1e-6
    ) -> ArrayLike:
        """
        有限差分近似梯度
        
        使用前向差分，当点在边界上时使用单侧差分。
        
        Args:
            f: 目标函数
            x: 当前点
            lb: 下界
            ub: 上界
            eps: 扰动量
            
        Returns:
            ArrayLike: 近似梯度
        """
        grad = np.zeros_like(x)
        fx = f(x)
        tol = 1e-8
        
        for i in range(len(x)):
            # 检查是否在边界上
            at_lower_bound = lb is not None and x[i] <= lb[i] + tol
            at_upper_bound = ub is not None and x[i] >= ub[i] - tol
            
            if at_lower_bound:
                # 在下界上，使用前向差分
                x_eps = x.copy()
                x_eps[i] = min(x[i] + eps, ub[i] if ub is not None else x[i] + eps)
                grad[i] = (f(x_eps) - fx) / max(eps, x_eps[i] - x[i])
            elif at_upper_bound:
                # 在上界上，使用后向差分
                x_eps = x.copy()
                x_eps[i] = max(x[i] - eps, lb[i] if lb is not None else x[i] - eps)
                grad[i] = (fx - f(x_eps)) / max(eps, x[i] - x_eps[i])
            else:
                # 不在边界上，使用中心差分（前向差分作为简化）
                x_eps = x.copy()
                x_eps[i] += eps
                if ub is not None:
                    x_eps[i] = min(x_eps[i], ub[i])
                grad[i] = (f(x_eps) - fx) / eps
        
        return grad

    @staticmethod
    def _project_gradient(
        grad: ArrayLike,
        x: ArrayLike,
        lb: ArrayLike,
        ub: ArrayLike,
        tol: float = 1e-8
    ) -> ArrayLike:
        """
        投影梯度到可行方向
        
        当变量在边界上时，将指向边界外的梯度分量置零。
        
        Args:
            grad: 原始梯度
            x: 当前点
            lb: 下界
            ub: 上界
            tol: 边界容差，用于判断是否在边界上
            
        Returns:
            ArrayLike: 投影后的梯度
        """
        grad_proj = grad.copy()
        
        for i in range(len(x)):
            # 考虑数值容差，判断是否在边界上
            if x[i] <= lb[i] + tol and grad[i] < 0:
                # 在下界上且梯度指向边界外，置零
                grad_proj[i] = 0
            elif x[i] >= ub[i] - tol and grad[i] > 0:
                # 在上界上且梯度指向边界外，置零
                grad_proj[i] = 0
        
        return grad_proj

    def _adamw_optimize(
        self,
        f: Callable,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        lb: ArrayLike,
        ub: ArrayLike,
        max_iter: int = 100,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_eps: float = 1e-6,
        tol: float = 1e-6,
        verbose: bool = False
    ) -> ArrayLike:
        """
        AdamW 优化算法实现
        
        AdamW 是 Adam 的改进版本，主要区别是权重衰减（weight decay）的处理方式：
        - Adam: 权重衰减被添加到梯度中（L2正则化）
        - AdamW: 权重衰减直接应用到参数更新中，与梯度解耦
        
        这种解耦使得权重衰减和梯度更新相互独立，通常能获得更好的泛化性能。
        
        Args:
            f: 目标函数
            x0: 初始解
            y0: 初始函数值（未使用，保留接口兼容）
            lb: 下界
            ub: 上界
            max_iter: 最大迭代次数
            lr: 学习率
            weight_decay: 权重衰减系数（L2正则化强度）
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            eps: 数值稳定性常数
            grad_eps: 梯度估计的扰动量
            tol: 收敛阈值
            verbose: 是否打印详细信息
            
        Returns:
            ArrayLike: 优化后的解
        """
        x = x0.copy()
        m = np.zeros_like(x)  # 一阶矩
        v = np.zeros_like(x)  # 二阶矩
        
        for t in range(1, max_iter + 1):
            # 有限差分估计梯度（不包含权重衰减）
            grad = self._approximate_gradient(f, x, lb, ub, grad_eps)
            
            # 边界投影梯度（使用数值容差判断边界）
            grad = self._project_gradient(grad, x, lb, ub, tol=1e-8)
            
            # 梯度收敛检查
            if np.linalg.norm(grad) < tol:
                if verbose:
                    print(f"[AdamW] 收敛于第 {t} 次迭代，梯度范数为 {np.linalg.norm(grad):.3e}")
                break
            
            # Adam 更新（一阶矩和二阶矩）
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)  # 偏差修正
            v_hat = v / (1 - beta2 ** t)
            
            # AdamW 参数更新：梯度更新 + 权重衰减（解耦）
            # 关键区别：权重衰减直接应用到参数，而不是添加到梯度
            x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * weight_decay * x
            x_new = np.clip(x_new, lb, ub)
            
            # 位移收敛检查
            if np.linalg.norm(x_new - x) < tol:
                if verbose:
                    print(f"[AdamW] 收敛于第 {t} 次迭代，Δx 范数为 {np.linalg.norm(x_new - x):.3e}")
                x = x_new
                break
            
            x = x_new
        
        return x

    def _lion_optimize(
        self,
        f: Callable,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        lb: ArrayLike,
        ub: ArrayLike,
        max_iter: int = 100,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        grad_eps: float = 1e-6,
        tol: float = 1e-6,
        verbose: bool = False
    ) -> ArrayLike:
        """
        Lion (EvoLved Sign Momentum) 优化算法实现
        
        Lion 是 Google 在 2023 年提出的优化算法，主要特点：
        1. 只跟踪动量（momentum），不需要二阶矩估计，内存占用更少
        2. 使用符号函数（sign）来更新参数，更新规则更简单
        3. 计算开销更小，在某些任务上表现更好
        
        更新规则：
        - m_t = beta1 * m_{t-1} + (1 - beta1) * grad_t
        - x_t = x_{t-1} - lr * sign(m_t) - lr * weight_decay * x_{t-1}
        
        Args:
            f: 目标函数
            x0: 初始解
            y0: 初始函数值（未使用，保留接口兼容）
            lb: 下界
            ub: 上界
            max_iter: 最大迭代次数
            lr: 学习率
            weight_decay: 权重衰减系数（L2正则化强度）
            beta1: 动量衰减率（通常接近1，如0.9或0.99）
            beta2: 未使用，保留接口兼容性（Lion 不需要二阶矩）
            grad_eps: 梯度估计的扰动量
            tol: 收敛阈值
            verbose: 是否打印详细信息
            
        Returns:
            ArrayLike: 优化后的解
        """
        x = x0.copy()
        m = np.zeros_like(x)  # 动量
        
        for t in range(1, max_iter + 1):
            # 有限差分估计梯度
            grad = self._approximate_gradient(f, x, lb, ub, grad_eps)
            
            # 边界投影梯度（使用数值容差判断边界）
            grad = self._project_gradient(grad, x, lb, ub, tol=1e-8)
            
            # 梯度收敛检查
            if np.linalg.norm(grad) < tol:
                if verbose:
                    print(f"[Lion] 收敛于第 {t} 次迭代，梯度范数为 {np.linalg.norm(grad):.3e}")
                break
            
            # Lion 更新：动量更新
            m = beta1 * m + (1 - beta1) * grad
            
            # Lion 参数更新：使用符号函数 + 权重衰减
            # sign(m) 返回 {-1, 0, 1}，使得更新步长固定为 lr
            x_new = x - lr * np.sign(m) - lr * weight_decay * x
            x_new = np.clip(x_new, lb, ub)
            
            # 位移收敛检查
            if np.linalg.norm(x_new - x) < tol:
                if verbose:
                    print(f"[Lion] 收敛于第 {t} 次迭代，Δx 范数为 {np.linalg.norm(x_new - x):.3e}")
                x = x_new
                break
            
            x = x_new
        
        return x

    def _sophia_optimize(
        self,
        f: Callable,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        lb: ArrayLike,
        ub: ArrayLike,
        max_iter: int = 100,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        beta1: float = 0.965,
        beta2: float = 0.99,
        rho: float = 0.04,
        eps: float = 1e-8,
        grad_eps: float = 1e-6,
        hessian_update_freq: int = 10,
        tol: float = 1e-6,
        verbose: bool = False
    ) -> ArrayLike:
        """
        Sophia (Second-order Clipped Stochastic Optimization) 优化算法实现
        
        Sophia 是斯坦福大学在 2023 年提出的优化算法，主要特点：
        1. 使用二阶信息（Hessian 的对角线近似）
        2. 使用裁剪（clipping）机制来控制更新步长
        3. 结合了自适应学习率和二阶信息
        
        更新规则：
        - 估计 Hessian 对角线：h_t ≈ diag(Hessian)
        - 裁剪更新：h_clipped = clip(grad_t / (h_t + eps), -rho, rho)
        - 参数更新：x_t = x_{t-1} - lr * h_clipped - lr * weight_decay * x_{t-1}
        
        Args:
            f: 目标函数
            x0: 初始解
            y0: 初始函数值（未使用，保留接口兼容）
            lb: 下界
            ub: 上界
            max_iter: 最大迭代次数
            lr: 学习率
            weight_decay: 权重衰减系数（L2正则化强度）
            beta1: Hessian 估计的指数衰减率
            beta2: 梯度平滑的指数衰减率（未使用，保留接口兼容）
            rho: 裁剪阈值，控制更新步长的上限
            eps: 数值稳定性常数
            grad_eps: 梯度估计的扰动量
            hessian_update_freq: Hessian 估计的更新频率（每 N 次迭代更新一次）
            tol: 收敛阈值
            verbose: 是否打印详细信息
            
        Returns:
            ArrayLike: 优化后的解
        """
        x = x0.copy()
        grad_prev = None
        h = np.ones_like(x)  # Hessian 对角线估计，初始化为1
        
        for t in range(1, max_iter + 1):
            # 有限差分估计梯度
            grad = self._approximate_gradient(f, x, lb, ub, grad_eps)
            
            # 边界投影梯度（使用数值容差判断边界）
            grad = self._project_gradient(grad, x, lb, ub, tol=1e-8)
            
            # 梯度收敛检查
            if np.linalg.norm(grad) < tol:
                if verbose:
                    print(f"[Sophia] 收敛于第 {t} 次迭代，梯度范数为 {np.linalg.norm(grad):.3e}")
                break
            
            # 更新 Hessian 对角线估计（使用梯度变化）
            # 每 hessian_update_freq 次迭代更新一次，或第一次迭代
            if grad_prev is not None and (t % hessian_update_freq == 0 or t == 1):
                # 使用梯度变化来估计 Hessian 对角线
                # h ≈ |grad - grad_prev| / |x - x_prev|，简化版本
                grad_diff = np.abs(grad - grad_prev)
                # 避免除零，使用平滑估计
                h_update = grad_diff + eps
                # 指数移动平均更新 Hessian 估计
                h = beta1 * h + (1 - beta1) * h_update
            
            # 保存当前梯度用于下次更新
            grad_prev = grad.copy()
            
            # Sophia 裁剪更新：clip(grad / (h + eps), -rho, rho)
            h_safe = h + eps  # 避免除零
            h_clipped = np.clip(grad / h_safe, -rho, rho)
            
            # Sophia 参数更新：裁剪后的更新 + 权重衰减
            x_new = x - lr * h_clipped - lr * weight_decay * x
            x_new = np.clip(x_new, lb, ub)
            
            # 位移收敛检查
            if np.linalg.norm(x_new - x) < tol:
                if verbose:
                    print(f"[Sophia] 收敛于第 {t} 次迭代，Δx 范数为 {np.linalg.norm(x_new - x):.3e}")
                x = x_new
                break
            
            x = x_new
        
        return x

    # ========================================================================
    # 遗传操作
    # ========================================================================
    
    def _inheritance(
        self,
        offspring_size: int,
        pop: ArrayLike,
        fit: ArrayLike
    ) -> ArrayLike:
        """
        基于适应度的遗传操作 (隐性选择版)
        
        在交叉过程中实现隐性选择：适应度越好的父代，其基因越容易遗传给子代。
        无需显式选择步骤，选择压力内嵌于交叉操作中。
        
        Args:
            offspring_size: 需要生成的后代数量
            pop: 当前种群
            fit: 适应度值
            
        Returns:
            ArrayLike: 生成的后代
        """
        offspring = np.zeros((offspring_size, self.dim))
        
        # 计算适应度权重（用于隐性选择）
        # 使用排名来避免适应度尺度问题
        ranks = np.argsort(np.argsort(fit.reshape(-1)))  # ranks[i] = 第i个个体的排名
        
        # 将排名转换为选择权重：排名越小（越好），权重越大
        # 使用线性排名选择：weight = (N - rank) / N
        n_pop = len(pop)
        rank_weights = (n_pop - ranks) / n_pop  # 最好的个体权重接近1，最差的接近0
        
        for i in range(offspring_size):
            # 策略选择
            strategy = self._rng.random()
            
            if strategy < 0.6:
                # 策略1: 适应度加权 SBX 交叉 (60%)
                # 隐性选择：适应度好的父代贡献更多基因
                child = self._fitness_weighted_sbx(pop, fit, rank_weights)
                offspring[i, :] = child
                
            elif strategy < 0.85:
                # 策略2: 适应度加权多父代交叉 (25%)
                # 隐性选择：多个父代按适应度加权组合
                child = self._fitness_weighted_multiparent(pop, fit, rank_weights)
                offspring[i, :] = child
                
            else:
                # 策略3: DE/current-to-best (15%)
                # 隐性选择：向最优个体方向进化
                child = self._de_current_to_best(pop, fit, rank_weights)
                offspring[i, :] = child
        
        # 边界检查与投影
        offspring = np.clip(offspring, self.lb, self.ub)
        
        return offspring

    def _fitness_weighted_sbx(
        self,
        pop: ArrayLike,
        fit: ArrayLike,
        rank_weights: ArrayLike,
        eta: float = 15
    ) -> ArrayLike:
        """
        适应度加权 SBX 交叉（隐性选择）
        
        选择两个父代后，根据各自适应度决定每个维度更倾向于继承哪个父代的基因。
        适应度更好的父代，其基因被选中的概率更高。
        
        Args:
            pop: 种群
            fit: 适应度
            rank_weights: 排名权重
            eta: SBX 分布指数
            
        Returns:
            ArrayLike: 生成的子代
        """
        # 基于排名权重选择父代（好的个体更容易被选中）
        probs = rank_weights / rank_weights.sum()
        p1_idx, p2_idx = self._rng.choice(len(pop), size=2, replace=False, p=probs)
        
        parent1, parent2 = pop[p1_idx], pop[p2_idx]
        w1, w2 = rank_weights[p1_idx], rank_weights[p2_idx]
        
        # 计算父代1的遗传倾向（适应度越好，倾向越高）
        bias_to_p1 = w1 / (w1 + w2)  # ∈ (0, 1)
        
        child = np.zeros(self.dim)
        
        for j in range(self.dim):
            p1_val, p2_val = parent1[j], parent2[j]
            
            if abs(p1_val - p2_val) < 1e-14:
                # 父代相同，直接继承
                child[j] = p1_val
                continue
            
            # 以一定概率进行 SBX 交叉
            if self._rng.random() < 0.5:
                # SBX 交叉
                y_l, y_u = min(p1_val, p2_val), max(p1_val, p2_val)
                
                u = self._rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                
                c1 = 0.5 * ((y_l + y_u) - beta * (y_u - y_l))
                c2 = 0.5 * ((y_l + y_u) + beta * (y_u - y_l))
                
                # 隐性选择：根据适应度偏向选择子代
                # 适应度好的父代方向的子代更容易被选中
                if p1_val < p2_val:
                    # p1 在左边
                    child[j] = c1 if self._rng.random() < bias_to_p1 else c2
                else:
                    # p1 在右边
                    child[j] = c2 if self._rng.random() < bias_to_p1 else c1
            else:
                # 直接继承：根据适应度决定继承哪个父代
                child[j] = p1_val if self._rng.random() < bias_to_p1 else p2_val
        
        return np.clip(child, self.lb, self.ub)

    def _fitness_weighted_multiparent(
        self,
        pop: ArrayLike,
        fit: ArrayLike,
        rank_weights: ArrayLike
    ) -> ArrayLike:
        """
        适应度加权多父代交叉（隐性选择）
        
        选择多个父代，根据各自适应度加权组合产生子代。
        适应度越好的父代，在子代中的贡献越大。
        
        Args:
            pop: 种群
            fit: 适应度
            rank_weights: 排名权重
            
        Returns:
            ArrayLike: 生成的子代
        """
        # 选择 3-5 个父代
        n_parents = self._rng.integers(3, 6)
        
        # 基于排名权重选择父代
        probs = rank_weights / rank_weights.sum()
        parent_indices = self._rng.choice(len(pop), size=n_parents, replace=False, p=probs)
        
        parents = pop[parent_indices]
        weights = rank_weights[parent_indices]
        
        # 归一化权重（隐性选择：好的父代权重更大）
        weights = weights / weights.sum()
        
        # 加权组合
        child = np.sum(parents * weights.reshape(-1, 1), axis=0)
        
        # 添加适度扰动保持多样性
        # 扰动强度与最差父代权重成正比（增强探索）
        perturbation_scale = 0.02 * (1 - weights.max()) * (self.ub - self.lb)
        noise = self._rng.normal(0, 1, self.dim) * perturbation_scale
        child = child + noise
        
        return np.clip(child, self.lb, self.ub)

    def _de_current_to_best(
        self,
        pop: ArrayLike,
        fit: ArrayLike,
        rank_weights: ArrayLike
    ) -> ArrayLike:
        """
        DE/current-to-best/1 变异（隐性选择）
        
        从当前个体向最优个体方向进化，同时加入随机扰动。
        隐性选择：当前个体的选择基于适应度。
        
        Args:
            pop: 种群
            fit: 适应度
            rank_weights: 排名权重
            
        Returns:
            ArrayLike: 生成的子代
        """
        # 最优个体
        best_idx = np.argmin(fit)
        x_best = pop[best_idx]
        
        # 基于排名权重选择当前个体和两个随机个体
        probs = rank_weights / rank_weights.sum()
        
        # 选择当前个体（好的个体更容易被选中作为基础）
        current_idx = self._rng.choice(len(pop), p=probs)
        x_current = pop[current_idx]
        
        # 选择两个随机个体用于差分
        r_indices = self._rng.choice(len(pop), size=2, replace=False)
        x_r1, x_r2 = pop[r_indices[0]], pop[r_indices[1]]
        
        # DE/current-to-best/1: v = x_current + F1 * (x_best - x_current) + F2 * (x_r1 - x_r2)
        F1 = 0.3 + 0.4 * self._rng.random()  # 向最优方向的步长
        F2 = 0.3 + 0.3 * self._rng.random()  # 随机扰动的步长
        
        mutant = x_current + F1 * (x_best - x_current) + F2 * (x_r1 - x_r2)
        
        # 二项交叉
        CR = 0.8
        child = x_current.copy()
        crossover_mask = self._rng.random(self.dim) < CR
        crossover_mask[self._rng.integers(self.dim)] = True  # 确保至少一个维度交叉
        child[crossover_mask] = mutant[crossover_mask]
        
        return np.clip(child, self.lb, self.ub)

    # ========================================================================
    # 变异操作
    # ========================================================================
    
    def _mutate(self, offspring: ArrayLike) -> ArrayLike:
        """
        多项式变异 (Polynomial Mutation)
        
        经典的连续优化变异算子，由 NSGA-II 等算法广泛使用。
        具有自适应特性：靠近边界时变异幅度自动减小。
        
        Args:
            offspring: 待变异的个体
            
        Returns:
            ArrayLike: 变异后的个体
        """
        if self.n_gen is None:
            raise RuntimeError("算法状态未初始化，无法执行变异")
        # 自适应分布指数：搜索早期 eta 较小（探索），后期 eta 较大（利用）
        # 典型范围：eta ∈ [5, 100]
        progress = min(1.0, self.n_gen / 50)  # 假设50代达到收敛阶段
        eta_m = 20 + 80 * progress  # 从20逐渐增加到100
        
        # 如果连续多代没改进，降低 eta 增强探索
        if self.stagnation_count >= 5:
            eta_m = max(10, eta_m - self.stagnation_count * 5)
            if self.stagnation_count % 10 == 0:
                print(f">>> 停滞 {self.stagnation_count} 代，降低 eta_m 至 {eta_m:.1f}")
        
        mutated = offspring.copy()
        
        for i in range(offspring.shape[0]):
            for j in range(self.dim):
                # 以一定概率对每个维度进行变异
                if self._rng.random() > 1.0 / self.dim:
                    continue
                
                y = offspring[i, j]
                lb_j = self.lb[j]
                ub_j = self.ub[j]
                
                delta = ub_j - lb_j
                if delta < 1e-14:
                    continue
                
                # 计算相对位置
                delta1 = (y - lb_j) / delta
                delta2 = (ub_j - y) / delta
                
                u = self._rng.random()
                
                # 多项式变异公式
                if u < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta_m + 1.0))
                    deltaq = val ** (1.0 / (eta_m + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta_m + 1.0))
                    deltaq = 1.0 - val ** (1.0 / (eta_m + 1.0))
                
                y_new = y + deltaq * delta
                mutated[i, j] = np.clip(y_new, lb_j, ub_j)
        
        return mutated

    def _get_directions(
        self,
        mesh_size: float,
        x: ArrayLike,
        tol: float
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        获取搜索方向
        
        生成基础方向集和切锥方向（处理边界约束）。
        
        Args:
            mesh_size: 网格大小，影响方向的密度
            x: 当前点
            tol: 边界容差
            
        Returns:
            Tuple[ArrayLike, ArrayLike]: (基础方向矩阵, 切锥方向矩阵)
        """
        dim = x.shape[0]
        lb = np.expand_dims(self.lb, axis=1)
        ub = np.expand_dims(self.ub, axis=1)
        
        # 构造切锥：识别活跃边界
        I = np.eye(dim)
        active = (np.abs(x - lb) < tol) | (np.abs(x - ub) < tol)
        tangent_cone = I[:, active.flatten()]
        
        # 构造随机基础方向
        p = 1 / np.sqrt(mesh_size)
        lower_t = np.tril(np.round((p + 1) * self._rng.random((dim, dim)) - 0.5), -1)
        
        diag_temp = p * np.sign(self._rng.random((dim, 1)) - 0.5)
        diag_temp[diag_temp == 0] = p * np.sign(0.5 - self._rng.random())
        
        diag_t = np.diag(diag_temp.flatten())
        basis = lower_t + diag_t
        
        # 随机排列
        order = self._rng.choice(dim, dim, replace=False)
        basis = basis[order][:, order]
        
        return basis, tangent_cone

    def _is_feasible(self, x: ArrayLike, tol: float) -> bool:
        """
        检查解的可行性
        
        Args:
            x: 待检查的解
            tol: 容差
            
        Returns:
            bool: 是否可行
        """
        lb = np.expand_dims(self.lb, axis=1)
        ub = np.expand_dims(self.ub, axis=1)
        
        constraint = max(np.max(x - ub), np.max(lb - x), 0)
        return constraint < tol

    # ========================================================================
    # 备用变异策略
    # ========================================================================
    
    def _mutate_individual(
        self,
        x: ArrayLike,
        tol: float
    ) -> ArrayLike:
        """
        对单个个体进行变异（多策略组合）
        
        随机选择一种变异策略应用于个体。
        
        Args:
            x: 待变异的个体
            tol: 可行性容差
            
        Returns:
            ArrayLike: 变异后的个体
        """
        dim = x.shape[0]
        
        strategies = [
            self._gaussian_mutation,
            self._directional_mutation,
            self._boundary_mutation
        ]
        
        # 随机选择变异策略
        strategy_idx = int(self._rng.integers(len(strategies)))
        candidate = strategies[strategy_idx](x, tol)
        
        # 若变异失败，回退到小扰动
        if not self._is_feasible(candidate, tol):
            noise = self._rng.normal(0, self.step_size * 0.1, (dim, 1))
            candidate = np.clip(
                x + noise,
                self.lb.reshape(-1, 1),
                self.ub.reshape(-1, 1)
            )
        
        return candidate.flatten()

    def _gaussian_mutation(self, x: ArrayLike, tol: float) -> ArrayLike:
        """
        高斯变异
        
        对所有维度添加高斯噪声。
        
        Args:
            x: 待变异的个体
            tol: 容差（未使用，保持接口一致）
            
        Returns:
            ArrayLike: 变异后的个体
        """
        dim = x.shape[0]
        noise = self._rng.normal(0, self.step_size, (dim, 1))
        return np.clip(
            x + noise,
            self.lb.reshape(-1, 1),
            self.ub.reshape(-1, 1)
        )

    def _directional_mutation(self, x: ArrayLike, tol: float) -> ArrayLike:
        """
        方向性变异
        
        组合多个搜索方向进行变异。
        
        Args:
            x: 待变异的个体
            tol: 边界容差
            
        Returns:
            ArrayLike: 变异后的个体
        """
        basis, tangent_cone = self._get_directions(self.step_size, x, tol)
        
        if tangent_cone.shape[1] > 0:
            tangent_cone = tangent_cone[:, np.sum(tangent_cone == 1, axis=0) == 1]
        
        dir_vector = np.hstack((basis, tangent_cone))
        
        # 随机选择 1-3 个方向组合
        n_dirs = self._rng.integers(1, min(4, dir_vector.shape[1] + 1))
        selected_dirs = self._rng.choice(dir_vector.shape[1], n_dirs, replace=False)
        
        # 随机权重组合
        weights = self._rng.normal(0, 1, n_dirs)
        combined_direction = np.zeros((x.shape[0], 1))
        
        for i, dir_idx in enumerate(selected_dirs):
            sign = self._rng.choice([-1, 1])
            combined_direction += weights[i] * sign * dir_vector[:, dir_idx].reshape(-1, 1)
        
        # 归一化
        if np.linalg.norm(combined_direction) > 0:
            combined_direction = combined_direction / np.linalg.norm(combined_direction)
        
        return np.clip(
            x + self.step_size * combined_direction,
            self.lb.reshape(-1, 1),
            self.ub.reshape(-1, 1)
        )

    def _boundary_mutation(self, x: ArrayLike, tol: float) -> ArrayLike:
        """
        边界感知变异
        
        根据到边界的距离自适应调整变异强度。
        
        Args:
            x: 待变异的个体
            tol: 容差（未使用，保持接口一致）
            
        Returns:
            ArrayLike: 变异后的个体
        """
        dim = x.shape[0]
        lb = self.lb.reshape(-1, 1)
        ub = self.ub.reshape(-1, 1)
        
        # 计算到边界的距离
        dist_to_lb = x - lb
        dist_to_ub = ub - x
        
        # 自适应变异强度
        adaptive_step = np.minimum(dist_to_lb, dist_to_ub) * 0.1
        adaptive_step = np.maximum(adaptive_step, self.step_size * 0.01)
        
        noise = self._rng.normal(0, 1, (dim, 1)) * adaptive_step
        return np.clip(x + noise, lb, ub)

    # ========================================================================
    # 工具方法
    # ========================================================================
    
    def check_bounds(self, x: ArrayLike | List[ArrayLike]) -> bool:
        """
        检查个体是否在边界内
        
        Args:
            x: 个体或个体数组
            
        Returns:
            bool: 是否全部在边界内
        """
        x = np.atleast_2d(x)
        out_of_bounds = (
            np.any(x < self.lb, axis=1) | 
            np.any(x > self.ub, axis=1)
        )
        
        if np.any(out_of_bounds):
            print(f"越界个体索引：{np.where(out_of_bounds)[0]}")
            print(f"对应个体：{x[out_of_bounds]}")
            return False
        
        return True
