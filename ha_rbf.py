"""
HA-RBF (Hybrid Algorithm with RBF Surrogate) - 带RBF代理模型的混合进化算法

该模块在 HA 算法基础上，使用径向基函数（RBF）拟合历史评估点，
在局部搜索时使用代理模型替代真实函数评估，减少计算开销。

主要特点：
    - 继承 HA 算法的所有功能
    - 维护评估历史缓存
    - RBF 代理模型辅助局部搜索
    - 局部搜索时使用代理模型，不再进行数值评估
"""

# ============================================================================
# 标准库导入
# ============================================================================
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ============================================================================
# 第三方库导入
# ============================================================================
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize as scipy_minimize
from sklearn.neighbors import NearestNeighbors

# 导入 HA 基类
from ha import HA

# ============================================================================
# 环境配置
# ============================================================================
os.environ["OMP_NUM_THREADS"] = "1"


class EvaluationHistory:
    """
    评估历史缓存
    
    存储所有评估过的点及其适应度和约束违反度。
    使用 KD-tree 加速最近邻查询。
    """
    
    def __init__(self, max_size: int = 10000):
        """
        初始化评估历史缓存
        
        Args:
            max_size: 最大缓存大小，超过后删除最旧的记录
        """
        self.max_size = max_size
        self.X: List[ArrayLike] = []
        self.F: List[float] = []
        self.CV: List[float] = []
        self._nn_tree: Optional[NearestNeighbors] = None
        self._needs_rebuild = True
    
    def add(self, x: ArrayLike, f: float, cv: float) -> None:
        """
        添加评估记录
        
        Args:
            x: 决策变量
            f: 适应度值
            cv: 约束违反度
        """
        self.X.append(x.copy())
        self.F.append(f)
        self.CV.append(cv)
        
        # 如果超过最大容量，删除最旧的
        if len(self.X) > self.max_size:
            self.X.pop(0)
            self.F.pop(0)
            self.CV.pop(0)
        
        self._needs_rebuild = True
    
    def add_batch(self, X: ArrayLike, F: ArrayLike, CV: ArrayLike) -> None:
        """
        批量添加评估记录
        
        Args:
            X: 决策变量矩阵
            F: 适应度数组
            CV: 约束违反度数组
        """
        for i in range(len(X)):
            self.add(X[i], float(F[i]), float(CV[i]))
    
    def get_nearby(
        self, 
        x0: ArrayLike, 
        radius: Optional[float] = None,
        max_points: int = 100
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        获取 x0 附近的历史点
        
        Args:
            x0: 查询点
            radius: 搜索半径，None 表示使用最近邻方法
            max_points: 最大返回点数
            
        Returns:
            Tuple: (X_nearby, F_nearby, CV_nearby)
        """
        if len(self.X) == 0:
            return np.array([]).reshape(0, len(x0)), np.array([]), np.array([])
        
        X_array = np.array(self.X)
        
        if radius is None:
            # 使用最近邻方法
            if self._needs_rebuild or self._nn_tree is None:
                self._nn_tree = NearestNeighbors(n_neighbors=min(max_points, len(self.X)))
                self._nn_tree.fit(X_array)
                self._needs_rebuild = False
            
            distances, indices = self._nn_tree.kneighbors([x0])
            indices = indices[0]
        else:
            # 使用半径方法
            if self._needs_rebuild or self._nn_tree is None:
                self._nn_tree = NearestNeighbors(radius=radius)
                self._nn_tree.fit(X_array)
                self._needs_rebuild = False
            
            distances, indices = self._nn_tree.radius_neighbors([x0], return_distance=True)
            indices = indices[0]
            distances = distances[0]
            
            # 按距离排序，取最近的 max_points 个
            if len(indices) > max_points:
                sorted_idx = np.argsort(distances)[:max_points]
                indices = indices[sorted_idx]
        
        return X_array[indices], np.array(self.F)[indices], np.array(self.CV)[indices]
    
    def size(self) -> int:
        """返回历史记录数量"""
        return len(self.X)
    
    def clear(self) -> None:
        """清空历史记录"""
        self.X.clear()
        self.F.clear()
        self.CV.clear()
        self._nn_tree = None
        self._needs_rebuild = True


class RBFSurrogate:
    """
    RBF 代理模型
    
    使用径向基函数拟合历史评估点，预测新点的适应度和约束违反度。
    """
    
    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        smoothing: float = 0.0,
        neighbors: Optional[int] = None
    ):
        """
        初始化 RBF 代理模型
        
        Args:
            kernel: RBF 核函数类型，可选 "linear", "thin_plate_spline", 
                   "cubic", "quintic", "multiquadric", "inverse_multiquadric", 
                   "gaussian"
            smoothing: 平滑参数，0.0 表示精确插值
            neighbors: 使用最近邻数量，None 表示使用所有点
        """
        self.kernel = kernel
        self.smoothing = smoothing
        self.neighbors = neighbors
        self.rbf_fitness: Optional[RBFInterpolator] = None
        self.rbf_cv: Optional[RBFInterpolator] = None
        self.X_train: Optional[ArrayLike] = None
        self.F_train: Optional[ArrayLike] = None
        self.CV_train: Optional[ArrayLike] = None
    
    def fit(
        self,
        X: ArrayLike,
        F: ArrayLike,
        CV: ArrayLike
    ) -> None:
        """
        拟合 RBF 模型
        
        Args:
            X: 训练点
            F: 适应度值
            CV: 约束违反度
        """
        if len(X) < 2:
            # 样本太少，无法拟合
            self.rbf_fitness = None
            self.rbf_cv = None
            return
        
        self.X_train = np.array(X)
        self.F_train = np.array(F).flatten()
        self.CV_train = np.array(CV).flatten()
        
        try:
            # 拟合适应度模型
            self.rbf_fitness = RBFInterpolator(
                self.X_train,
                self.F_train,
                kernel=self.kernel,
                smoothing=self.smoothing,
                neighbors=self.neighbors
            )
            
            # 拟合约束违反度模型
            self.rbf_cv = RBFInterpolator(
                self.X_train,
                self.CV_train,
                kernel=self.kernel,
                smoothing=self.smoothing,
                neighbors=self.neighbors
            )
        except Exception as e:
            # 拟合失败，可能是数值问题
            warnings.warn(f"RBF 拟合失败: {e}，将使用最近邻方法")
            self.rbf_fitness = None
            self.rbf_cv = None
    
    def predict(self, X: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        预测新点的适应度和约束违反度
        
        Args:
            X: 待预测的点
            
        Returns:
            Tuple: (预测适应度, 预测约束违反度)
        """
        X = np.atleast_2d(X)
        
        if self.rbf_fitness is None or self.rbf_cv is None:
            # 如果模型未拟合，使用最近邻方法
            return self._nearest_neighbor_predict(X)
        
        try:
            F_pred = self.rbf_fitness(X)
            CV_pred = self.rbf_cv(X)
            return F_pred.flatten(), CV_pred.flatten()
        except Exception as e:
            # 预测失败，回退到最近邻
            warnings.warn(f"RBF 预测失败: {e}，使用最近邻方法")
            return self._nearest_neighbor_predict(X)
    
    def _nearest_neighbor_predict(self, X: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """
        使用最近邻方法预测（回退方案）
        
        Args:
            X: 待预测的点
            
        Returns:
            Tuple: (预测适应度, 预测约束违反度)
        """
        if self.X_train is None or len(self.X_train) == 0:
            # 没有训练数据，返回默认值
            return np.full(len(X), np.inf), np.full(len(X), np.inf)
        
        X = np.atleast_2d(X)
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self.X_train)
        distances, indices = nn.kneighbors(X)
        
        F_pred = self.F_train[indices.flatten()]
        CV_pred = self.CV_train[indices.flatten()]
        
        return F_pred, CV_pred
    
    def is_fitted(self) -> bool:
        """检查模型是否已拟合"""
        return self.rbf_fitness is not None and self.rbf_cv is not None


class HA_RBF(HA):
    """
    带 RBF 代理模型的混合进化算法
    
    继承 HA 算法，在局部搜索时使用 RBF 代理模型替代真实函数评估。
    """
    
    def __init__(
        self,
        method: str = "L-BFGS-B",
        pop_size: int = 100,
        niche_num: int = 3,
        mutation_rate: float = 0.3,
        inherit_rate: float = 0.8,
        activate_method: bool = True,
        cluster_method: str = "kmeans",
        dbscan_eps: Optional[float] = None,
        dbscan_min_samples: Optional[int] = None,
        X: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
        rbf_kernel: str = "thin_plate_spline",
        rbf_smoothing: float = 0.0,
        rbf_neighbors: Optional[int] = None,
        history_max_size: int = 10000,
        local_search_radius: Optional[float] = None,
        **kwargs
    ):
        """
        初始化 HA-RBF 算法
        
        Args:
            method: 局部搜索方法名称（仅支持 L-BFGS-B）
            pop_size: 种群大小
            niche_num: 聚类数量
            mutation_rate: 变异率
            inherit_rate: 遗传率
            activate_method: 是否启用局部搜索
            cluster_method: 聚类方法
            dbscan_eps: DBSCAN 的邻域半径参数
            dbscan_min_samples: DBSCAN 的最小样本数参数
            X: 初始种群
            seed: 随机种子
            rbf_kernel: RBF 核函数类型
            rbf_smoothing: RBF 平滑参数
            rbf_neighbors: RBF 使用的最近邻数量
            history_max_size: 评估历史最大缓存大小
            local_search_radius: 局部搜索时查询历史点的半径
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(
            method=method,
            pop_size=pop_size,
            niche_num=niche_num,
            mutation_rate=mutation_rate,
            inherit_rate=inherit_rate,
            activate_method=activate_method,
            cluster_method=cluster_method,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            X=X,
            seed=seed,
            **kwargs
        )
        
        # RBF 相关参数
        self.rbf_kernel = rbf_kernel
        self.rbf_smoothing = rbf_smoothing
        self.rbf_neighbors = rbf_neighbors
        self.history_max_size = history_max_size
        self.local_search_radius = local_search_radius
        
        # 评估历史和代理模型
        self.history: Optional[EvaluationHistory] = None
        self.surrogate: Optional[RBFSurrogate] = None
    
    def _setup(self, problem: Any, **kwargs) -> None:
        """
        设置算法参数
        
        Args:
            problem: PyMoo 问题对象
            **kwargs: 其他参数
        """
        super()._setup(problem, **kwargs)
        
        # 初始化评估历史和代理模型
        self.history = EvaluationHistory(max_size=self.history_max_size)
        self.surrogate = RBFSurrogate(
            kernel=self.rbf_kernel,
            smoothing=self.rbf_smoothing,
            neighbors=self.rbf_neighbors
        )
    
    def evaluate_fitness_cv(
        self,
        x: ArrayLike
    ) -> Tuple[ArrayLike, float]:
        """
        评估单个个体的适应度和约束违反度（并记录历史）
        
        Args:
            x: 决策变量
            
        Returns:
            Tuple[ArrayLike, float]: (适应度值, 约束违反度)
        """
        # 调用父类方法进行真实评估
        fitness, cv = super().evaluate_fitness_cv(x)
        
        # 记录到历史
        if self.history is not None:
            self.history.add(x, float(fitness), float(cv))
        
        return fitness, cv
    
    def evaluate_fitness_cv_batch(
        self,
        X: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        批量评估多个个体的适应度和约束违反度（并记录历史）
        
        Args:
            X: 决策变量矩阵
            
        Returns:
            Tuple[ArrayLike, ArrayLike]: (适应度数组, 约束违反度数组)
        """
        # 调用父类方法进行真实评估
        fitness, cv = super().evaluate_fitness_cv_batch(X)
        
        # 记录到历史
        if self.history is not None:
            self.history.add_batch(X, fitness, cv)
        
        return fitness, cv
    
    def _local_search(
        self,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        maxiter: int = 1
    ) -> ArrayLike:
        """
        局部搜索（使用 RBF 代理模型）
        
        使用 RBF 代理模型替代真实函数评估，在代理模型上进行优化。
        
        Args:
            x0: 初始解
            y0: 初始目标函数值（未使用）
            maxiter: 最大迭代次数
            
        Returns:
            ArrayLike: 优化后的解
        """
        if not self.check_bounds([x0]):
            print("传入的 x0 越界")
            raise ValueError("初始解越界")
        
        # 检查是否有足够的历史数据
        if self.history is None or self.history.size() < max(10, self.dim * 2):
            # 历史数据不足，回退到父类的真实评估方法
            print(f"历史数据不足 ({self.history.size() if self.history else 0} < {max(10, self.dim * 2)})，使用真实评估")
            return super()._local_search(x0, y0, maxiter)
        
        # 获取 x0 附近的历史点
        X_nearby, F_nearby, CV_nearby = self.history.get_nearby(
            x0,
            radius=self.local_search_radius,
            max_points=min(200, self.history.size())
        )
        
        if len(X_nearby) < max(5, self.dim):
            # 附近点太少，回退到真实评估
            print(f"附近历史点不足 ({len(X_nearby)} < {max(5, self.dim)})，使用真实评估")
            return super()._local_search(x0, y0, maxiter)
        
        # 拟合 RBF 代理模型
        self.surrogate.fit(X_nearby, F_nearby, CV_nearby)
        
        if not self.surrogate.is_fitted():
            # 拟合失败，回退到真实评估
            print("RBF 拟合失败，使用真实评估")
            return super()._local_search(x0, y0, maxiter)
        
        # 定义代理目标函数（带约束惩罚）
        def surrogate_objective(x: ArrayLike) -> float:
            """使用 RBF 代理模型的目标函数"""
            x = np.clip(x, self.lb, self.ub)
            f_pred, cv_pred = self.surrogate.predict(x.reshape(1, -1))
            
            # 计算惩罚系数（使用历史数据的尺度）
            alpha = np.abs(F_nearby).mean() * 10 if len(F_nearby) > 0 else 1.0
            return float(f_pred[0] + alpha * max(0, cv_pred[0]))
        
        # 使用 L-BFGS-B 在代理模型上优化
        bounds = [(self.lb[i], self.ub[i]) for i in range(self.dim)]
        x0_clipped = np.clip(x0, self.lb, self.ub)
        
        try:
            result = scipy_minimize(
                fun=surrogate_objective,
                x0=x0_clipped,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter}
            )
            
            optimized_x = np.clip(result.x, self.lb, self.ub)
            
            # 验证：用真实函数评估一次（可选，用于验证代理模型质量）
            # 这里不进行验证，完全依赖代理模型
            
            return optimized_x
            
        except Exception as e:
            # 优化失败，回退到真实评估
            warnings.warn(f"代理模型优化失败: {e}，回退到真实评估")
            return super()._local_search(x0, y0, maxiter)
    
    def _clustering_and_learning(
        self,
        pop: ArrayLike,
        fit: ArrayLike,
        cv: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, List[int]]:
        """
        聚类和局部学习（使用 RBF 代理模型）
        
        重写父类方法，在局部搜索时使用代理模型。
        
        Args:
            pop: 当前种群
            fit: 适应度值
            cv: 约束违反度
            
        Returns:
            Tuple: (更新后的种群, 适应度, 约束违反度, 精英索引列表)
        """
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
        print(f"进行聚类学习（使用RBF代理模型）... Gen: {self.n_gen}, "
              f"Stagnation: {self.stagnation_count}, Method: {self.cluster_method}")
        print(f"历史评估点数: {self.history.size() if self.history else 0}")
        
        # 执行聚类
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        labels, n_clusters = self._perform_clustering(pop)
        print(f"聚类结果: {n_clusters} 个簇")
        
        # 收集每个聚类的最优个体索引
        cluster_best_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            cluster_fits = fit[cluster_indices, 0]
            best_in_cluster = cluster_indices[np.argmin(cluster_fits)]
            cluster_best_indices.append(best_in_cluster)
        
        # 如果聚类数 > niche_num，只选取最优的 niche_num 个聚类代表
        if len(cluster_best_indices) > self.niche_num:
            def sort_key(idx):
                cv_val = cv[idx, 0]
                fit_val = fit[idx, 0]
                return (cv_val > 0, cv_val, fit_val)
            
            cluster_best_indices = sorted(cluster_best_indices, key=sort_key)
            cluster_best_indices = cluster_best_indices[:self.niche_num]
            print(f"聚类数 > niche_num ({self.niche_num})，选取前 {self.niche_num} 个最优聚类代表")
        
        # 对选中的聚类代表执行局部搜索（使用代理模型）
        for idx in cluster_best_indices:
            is_global_best = (idx == np.argmin(fit[:, 0]))
            search_depth = 10 if (is_global_best and is_stagnant) else 3
            
            fes_before = self.problem.fes
            
            # 执行局部搜索（使用 RBF 代理模型，不进行真实评估）
            new_solution = self._local_search(
                pop[idx, :],
                fit[idx, 0],
                maxiter=search_depth
            )
            
            fes_after = self.problem.fes
            
            # 评估更新（这里才进行真实评估）
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
