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
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

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
ArrayLike = NDArray[np.floating]


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
    BOUNDED_METHODS = ["L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr"]
    
    # 支持的聚类方法
    CLUSTER_METHODS = ["kmeans", "meanshift", "dbscan"]
    
    # ========================================================================
    # 初始化方法
    # ========================================================================
    
    def __init__(
        self,
        method: str = "L-BFGS-B",
        pop_size: int = 100,
        niche_num: int = 3,
        mutation_rate: float = 0.8,
        inherit_rate: float = 0.5,
        activate_method: bool = True,
        cluster_method: str = "kmeans",
        dbscan_eps: Optional[float] = None,
        dbscan_min_samples: Optional[int] = None,
        X: Optional[ArrayLike] = None,
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
            X: 初始种群（可选），若为 None 则随机生成
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
        
        # 运行时状态
        self.step_size: float = 1.0
        self.improvement: bool = True
        self.fun_cnt: int = 0
        self.best: float = float("inf")
        self.best_individual: Optional[ArrayLike] = None
        
        # 将在 _setup 中初始化的参数
        self.elite_num: Optional[int] = None
        self.dim: Optional[int] = None
        self.lb: Optional[ArrayLike] = None
        self.ub: Optional[ArrayLike] = None
        self.FEs: int = 0
        self.local_elite_id: List[int] = []
        self.pop_cv: Optional[ArrayLike] = None

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
        
        # 精英数量设为问题维度
        self.elite_num = self.dim
        self.FEs = 0
        self.local_elite_id = []

    # ========================================================================
    # 种群初始化与进化
    # ========================================================================
    
    def _initialize_infill(self) -> Population:
        """
        初始化种群
        
        生成初始种群并进行首次评估。
        
        Returns:
            Population: 初始化后的种群对象
        """
        print("初始化种群...")
        
        # 生成或使用指定的初始种群
        if self.X is None:
            print("self.X is None")
            pop_x = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        else:
            pop_x = self.X
        
        # 批量评估初始种群
        pop_f, pop_cv = self.evaluate_fitness_cv_batch(pop_x)
        
        # 保存约束违反度信息
        self.pop_cv = np.array(pop_cv).reshape(-1, 1)
        pop_f = np.array(pop_f).reshape(-1, 1)
        
        print(f"pop.shape = {pop_x.shape}")
        print(f"fit.shape = {pop_f.shape}")
        print(f"cv.shape = {self.pop_cv.shape}")
        
        return Population.new("X", pop_x, "F", pop_f)

    def _infill(self) -> Population:
        """
        执行一代进化
        
        包括：更新最优解、执行 HA 算法步骤、更新种群。
        
        Returns:
            Population: 进化后的新种群
        """
        print(f"从 Generation {self.n_gen - 1} 进化到 Generation {self.n_gen}...")
        
        # 获取当前种群数据
        pop = self.pop.get("X")
        fit = self.pop.get("F")
        cv = self.pop_cv
        
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
        else:
            self.improvement = False
        
        # 执行 HA 算法的一代进化
        new_pop, new_fit, new_cv = self._step_ha(pop, fit, cv)
        
        # 更新种群
        self.pop = Population.new("X", new_pop, "F", new_fit)
        self.pop_cv = new_cv
        
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
        
        return fitness.reshape(-1, 1), cv

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
        
        # 选择全局精英
        global_elite_id = np.argsort(fit[:, 0])[:self.elite_num].tolist()
        elite_id.extend(global_elite_id)
        
        # 去重并保持顺序
        elite_id = self._unique_preserve_order(elite_id)
        print(elite_id)
        
        # 计算后代数量
        offspring_size = self.pop_size - len(elite_id)
        
        # 生成后代：遗传 + 随机选择
        num_to_inherit = int(self.inherit_rate * offspring_size)
        offspring_inherit = self._inheritance(num_to_inherit, pop, fit)
        
        # 补充非遗传后代
        non_elite_indices = np.setdiff1d(np.arange(self.pop_size), elite_id)
        selected_indices = np.random.choice(
            non_elite_indices, 
            offspring_size - num_to_inherit, 
            replace=False
        )
        offspring_select = pop[selected_indices]
        
        # 合并精英和后代
        elite_individuals = pop[elite_id]
        offspring = np.vstack((offspring_inherit, offspring_select, elite_individuals))
        
        if not self.check_bounds(offspring) and len(offspring) != self.pop_size:
            raise ValueError("后代生成后越界")
        
        # 变异操作
        mutate_num = round(offspring_size * self.mutation_rate)
        if mutate_num > 0:
            mutate_id = np.random.choice(offspring_size, mutate_num, replace=False)
            offspring[mutate_id, :] = self._mutate(offspring[mutate_id, :])
        
        if not self.check_bounds(offspring):
            print("_mutate 越界")
            raise ValueError("变异后越界")
        
        
        
        # 去重并评估
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
        new_pop = np.vstack((offspring, pop))
        new_fit = np.vstack((offspring_fit, fit))
        new_cv = np.vstack((offspring_cv, cv))
        
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
        
        # 检查种群多样性，若过低则终止
        unique_count = np.unique(new_pop, axis=0).shape[0]
        print(f"种群中不重复的个体数量: {unique_count}")
        
        if unique_count <= 2:
            self.termination.force_termination = True
        
        return new_pop, new_fit, new_cv

    def _sort_by_constraint_dominance(
        self,
        fit: ArrayLike,
        cv: ArrayLike
    ) -> List[int]:
        """
        按约束支配规则排序
        
        排序优先级：可行解优先，同为可行解时按适应度排序，
        同为不可行解时按约束违反度排序。
        
        Args:
            fit: 适应度数组
            cv: 约束违反度数组
            
        Returns:
            List[int]: 排序后的索引列表
        """
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
        repeat: ArrayLike
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
        
        使用指定的聚类算法划分种群为多个小生境，
        对每个小生境的最优个体执行局部搜索。
        
        支持的聚类方法：
            - kmeans: K-means 聚类，需指定聚类数量
            - meanshift: MeanShift 聚类，自动确定聚类数量
            - dbscan: DBSCAN 聚类，基于密度的聚类
        
        Args:
            pop: 当前种群
            fit: 适应度值
            cv: 约束违反度
            
        Returns:
            Tuple: (更新后的种群, 适应度, 约束违反度, 精英索引列表)
        """
        if not self.check_bounds(pop):
            print("传入的 pop 越界")
            raise ValueError("种群越界")
        
        elite_id = []
        print(f"进行聚类和局部学习...{self.n_gen}，使用 {self.cluster_method} 聚类")
        
        # 执行聚类
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        labels, n_clusters = self._perform_clustering(pop)
        print(f"聚类结果：共 {n_clusters} 个簇")
        
        # 收集每个聚类的最优个体索引
        cluster_best_indices = []
        for i in range(n_clusters):
            cluster_idx = np.where(labels == i)[0]
            if len(cluster_idx) == 0:
                continue
            # 聚类中的最优个体（假设已排序，取第一个）
            cluster_best_indices.append(cluster_idx[0])
        
        # 如果聚类数量大于 niche_num，只选取适应度最好的 niche_num 个聚类代表
        if len(cluster_best_indices) > self.niche_num:
            # 按适应度排序（考虑约束违反度）
            def sort_key(idx):
                # 可行解优先，然后按适应度排序
                cv_val = cv[idx, 0]
                fit_val = fit[idx, 0]
                return (cv_val > 0, cv_val, fit_val)
            
            cluster_best_indices = sorted(cluster_best_indices, key=sort_key)
            cluster_best_indices = cluster_best_indices[:self.niche_num]
            print(f"聚类数 {n_clusters} > niche_num {self.niche_num}，"
                  f"选取前 {self.niche_num} 个最优聚类代表进行局部搜索")
        
        # 对选中的聚类代表执行局部搜索
        for best_individual_idx in cluster_best_indices:
            print(f"best_individual_idx = {best_individual_idx}")
            
            # 记录优化前后的函数评估次数
            fes_before = self.problem.fes
            print(f"局部优化之前：x: {pop[best_individual_idx, :]} "
                  f"fitness: {fit[best_individual_idx].item()} "
                  f"cv: {cv[best_individual_idx].item()}")
            
            # 执行局部搜索
            if self.n_gen <= 5 and self.activate_method:
                # 使用带约束惩罚的目标函数
                penalized_fitness = fit[best_individual_idx] + 10 * cv[best_individual_idx]
                new_solution = self._local_search(
                    pop[best_individual_idx, :], 
                    penalized_fitness
                )
            else:
                new_solution = pop[best_individual_idx, :]
            
            fes_after = self.problem.fes
            
            # 边界检查
            if np.any(new_solution > self.ub) or np.any(new_solution < self.lb):
                print(new_solution)
                raise ValueError("x out of bounds")
            
            # 更新个体
            pop[best_individual_idx, :] = new_solution
            
            if self.activate_method:
                new_fitness, new_cv = self.evaluate_fitness_cv(new_solution)
            else:
                new_fitness = fit[best_individual_idx]
                new_cv = cv[best_individual_idx]
            
            print(f"局部优化之后: x: {new_solution[:5]} "
                  f"fitness: {new_fitness.item()} cv: {new_cv}")
            print(f"局部优化消耗了 {fes_after - fes_before} 次仿真\n")
            
            fit[best_individual_idx, 0] = new_fitness
            cv[best_individual_idx, 0] = new_cv
            elite_id.append(best_individual_idx)
        
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
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        labels = meanshift.fit_predict(pop_scaled)
        
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
        y0: Union[float, ArrayLike]
    ) -> ArrayLike:
        """
        局部搜索
        
        使用指定的优化方法对初始解进行局部优化。
        
        Args:
            x0: 初始解
            y0: 初始目标函数值（用于 Adam 方法）
            
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
            fitness, cv = self.evaluate_fitness_cv(x)
            alpha = abs(fitness).mean() * 10
            return fitness + alpha * cv
        
        if self.method in self.BOUNDED_METHODS:
            return self._scipy_local_search(x0, objective)
        elif self.method == "Adam":
            return self._adam_local_search(x0, y0, objective)
        else:
            raise ValueError(f"不支持的优化方法: {self.method}")

    def _scipy_local_search(
        self,
        x0: ArrayLike,
        objective: Callable
    ) -> ArrayLike:
        """
        使用 SciPy 优化器进行局部搜索
        
        Args:
            x0: 初始解
            objective: 目标函数
            
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
            minimize_kwargs["options"] = {"maxiter": 1}
        else:
            minimize_kwargs["options"] = {"maxfun": 100, "disp": True}
        
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
            
            # 边界投影梯度
            grad = self._project_gradient(grad, x, lb, ub)
            
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
        
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += eps
            # 确保在边界内
            if lb is not None:
                x_eps[i] = min(max(x_eps[i], lb[i]), ub[i])
            grad[i] = (f(x_eps) - fx) / eps
        
        return grad

    @staticmethod
    def _project_gradient(
        grad: ArrayLike,
        x: ArrayLike,
        lb: ArrayLike,
        ub: ArrayLike
    ) -> ArrayLike:
        """
        投影梯度到可行方向
        
        当变量在边界上时，将指向边界外的梯度分量置零。
        
        Args:
            grad: 原始梯度
            x: 当前点
            lb: 下界
            ub: 上界
            
        Returns:
            ArrayLike: 投影后的梯度
        """
        grad_proj = grad.copy()
        
        for i in range(len(x)):
            if x[i] <= lb[i] and grad[i] < 0:
                grad_proj[i] = 0
            elif x[i] >= ub[i] and grad[i] > 0:
                grad_proj[i] = 0
        
        return grad_proj

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
        基于适应度的遗传操作
        
        使用排名选择法，从多个父代中选择基因组合产生后代。
        
        Args:
            offspring_size: 需要生成的后代数量
            pop: 当前种群
            fit: 适应度值
            
        Returns:
            ArrayLike: 生成的后代
        """
        # 基于排名的选择分数
        scores = np.arange(pop.shape[0], 0, -1)
        offspring = np.zeros((offspring_size, self.dim))
        
        for i in range(offspring_size):
            # 随机选择 dim 个父代
            parent_size = self.dim
            parent_idx = np.random.choice(pop.shape[0], parent_size, replace=False)
            parent_scores = scores[parent_idx]
            parent_pop = pop[parent_idx, :]
            
            # 基于分数的选择概率
            probs = parent_scores / np.sum(parent_scores)
            
            # 对每个维度独立选择
            for j in range(self.dim):
                offspring[i, j] = np.random.choice(parent_pop[:, j], p=probs)
        
        # 边界检查
        out_of_bounds = (
            np.any(offspring < self.lb, axis=1) | 
            np.any(offspring > self.ub, axis=1)
        )
        if np.any(out_of_bounds):
            print(f"越界个体索引：{np.where(out_of_bounds)[0]}")
            print(f"对应个体：{offspring[out_of_bounds]}")
            raise ValueError("有 offspring 个体越界")
        
        return offspring

    # ========================================================================
    # 变异操作
    # ========================================================================
    
    def _mutate(self, offspring: ArrayLike) -> ArrayLike:
        """
        自适应变异
        
        根据搜索进展动态调整步长，使用方向性变异策略。
        
        Args:
            offspring: 待变异的个体
            
        Returns:
            ArrayLike: 变异后的个体
        """
        # 自适应调整步长
        if self.n_gen <= 2:
            self.step_size = 1
        else:
            if self.improvement:
                self.step_size = min(1, self.step_size * 4)
            else:
                self.step_size = max(1e-6, self.step_size / 4)
        
        mutated = np.zeros_like(offspring)
        tol = 1e-3
        
        for i in range(offspring.shape[0]):
            x = offspring[i, :].reshape(-1, 1)
            
            # 获取搜索方向
            basis, tangent_cone = self._get_directions(self.step_size, x, tol)
            
            # 处理切锥方向
            if tangent_cone.shape[1] > 0:
                tangent_cone = tangent_cone[:, np.sum(tangent_cone == 1, axis=0) == 1]
            
            # 合并所有方向
            dir_vector = np.hstack((basis, tangent_cone))
            n_basis = basis.shape[1]
            n_tangent = tangent_cone.shape[1]
            n_total = n_basis + n_tangent
            
            # 构造方向索引和符号（正向和反向）
            index_vec = np.hstack((
                np.arange(n_basis), np.arange(n_basis),
                np.arange(n_basis, n_total), np.arange(n_basis, n_total)
            ))
            dir_sign = np.hstack((
                np.ones(n_basis), -np.ones(n_basis),
                np.ones(n_tangent), -np.ones(n_tangent)
            ))
            
            # 随机排列方向尝试顺序
            order = np.random.choice(len(index_vec), len(index_vec), replace=False)
            success = False
            mutated[i, :] = x.flatten()
            
            # 尝试每个方向直到找到可行解
            for k in order:
                direction = dir_sign[k] * dir_vector[:, index_vec[k]].reshape(-1, 1)
                noise = np.random.randn(*direction.shape) * 0.01
                direction += noise
                candidate = x + self.step_size * direction
                candidate = np.clip(candidate, self.lb.reshape(-1, 1), self.ub.reshape(-1, 1))
                
                if self._is_feasible(candidate, tol):
                    success = True
                    mutated[i, :] = candidate.flatten()
                    break
            
            if not success:
                print(f"第 {i} 个个体没有找到可行变异方向，保留原解")
        
        # 最终边界检查
        out_of_bounds = (
            np.any(mutated < self.lb, axis=1) | 
            np.any(mutated > self.ub, axis=1)
        )
        if np.any(out_of_bounds):
            print("变异越界：")
            print(f"越界个体索引：{np.where(out_of_bounds)[0]}")
            print(f"对应个体：{mutated[out_of_bounds]}")
            raise ValueError("有 offspring 个体越界")
        
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
        lower_t = np.tril(np.round((p + 1) * np.random.rand(dim, dim) - 0.5), -1)
        
        diag_temp = p * np.sign(np.random.rand(dim, 1) - 0.5)
        diag_temp[diag_temp == 0] = p * np.sign(0.5 - np.random.rand())
        
        diag_t = np.diag(diag_temp.flatten())
        basis = lower_t + diag_t
        
        # 随机排列
        order = np.random.choice(dim, dim, replace=False)
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
        strategy = np.random.choice(strategies)
        candidate = strategy(x, tol)
        
        # 若变异失败，回退到小扰动
        if not self._is_feasible(candidate, tol):
            noise = np.random.normal(0, self.step_size * 0.1, (dim, 1))
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
        noise = np.random.normal(0, self.step_size, (dim, 1))
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
        n_dirs = np.random.randint(1, min(4, dir_vector.shape[1] + 1))
        selected_dirs = np.random.choice(dir_vector.shape[1], n_dirs, replace=False)
        
        # 随机权重组合
        weights = np.random.normal(0, 1, n_dirs)
        combined_direction = np.zeros((x.shape[0], 1))
        
        for i, dir_idx in enumerate(selected_dirs):
            sign = np.random.choice([-1, 1])
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
        
        noise = np.random.normal(0, 1, (dim, 1)) * adaptive_step
        return np.clip(x + noise, lb, ub)

    # ========================================================================
    # 工具方法
    # ========================================================================
    
    def check_bounds(self, x: ArrayLike) -> bool:
        """
        检查个体是否在边界内
        
        Args:
            x: 个体或个体数组
            
        Returns:
            bool: 是否全部在边界内
        """
        out_of_bounds = (
            np.any(x < self.lb, axis=1) | 
            np.any(x > self.ub, axis=1)
        )
        
        if np.any(out_of_bounds):
            print(f"越界个体索引：{np.where(out_of_bounds)[0]}")
            print(f"对应个体：{x[out_of_bounds]}")
            return False
        
        return True
