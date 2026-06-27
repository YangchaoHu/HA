"""
HA-NSGA-III: Hybrid Algorithm with NSGA-III for Multi-Objective Optimization

该模块在 ha_Nelder_Mead.py 的 HA 类基础上扩展为多目标算法：
    - 用 NSGA-III 的非支配排序 + 参考方向小生境替换原约束支配排序与聚类
    - 用 PBI (Penalty-based Boundary Intersection) 标量化把局部搜索器
      (Nelder-Mead / RBF / GP / Adam / ...) 适配到多目标场景
    - 复用 HA 的遗传操作、变异、历史信息与局部搜索内部实现

主要类：
    - MOPopulationHistory: 多目标种群历史（支持 F shape (M,)，按支配关系判重复更新）
    - HA_NSGA3: 多目标 HA 算法，继承自 HA

用法示例：
    >>> from problem import ZDT1Problem
    >>> from pymoo.optimize import minimize
    >>> problem = ZDT1Problem(n_var=10)
    >>> algorithm = HA_NSGA3(method="Nelder-Mead", pop_size=50, seed=42)
    >>> result = minimize(problem, algorithm, termination=('n_gen', 30))
"""

# ============================================================================
# 标准库
# ============================================================================
import os
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

# ============================================================================
# 第三方库
# ============================================================================
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize as scipy_minimize

# pymoo 相关
from pymoo.core.population import Population
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# sklearn 相关（MO-NM 单纯形构建用最近邻）
from sklearn.neighbors import NearestNeighbors

# 本地模块
from ha_Nelder_Mead import HA, PopulationHistory

# ============================================================================
# 环境配置
# ============================================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ============================================================================
# 类型别名
# ============================================================================
ArrayLike = NDArray[np.floating]


# ============================================================================
# 多目标种群历史
# ============================================================================
class MOPopulationHistory(PopulationHistory):
    """
    多目标种群历史信息存储（去重）

    继承 PopulationHistory，主要差异：
        - F 形状从 (1,) 扩展为 (M,)（任意目标数）
        - 重复个体的"更优替换"规则改为：约束更小，或约束相同时按支配关系判断
          （旧解被新解 Pareto 支配则替换；互不支配则保留旧解）
    """

    @staticmethod
    def _dominates(a: ArrayLike, b: ArrayLike) -> bool:
        """检查 a 是否（弱）支配 b：所有维度 a[i] <= b[i] 且至少一个 a[i] < b[i]"""
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        return bool(np.all(a <= b) and np.any(a < b))

    def add(
        self,
        X: ArrayLike,
        F: ArrayLike,
        CV: Union[float, ArrayLike]
    ) -> bool:
        """
        添加种群信息（自动去重，支持多目标）

        Args:
            X: 决策变量，shape (dim,) 或 (N, dim)
            F: 目标值，shape (M,) 或 (N, M)
            CV: 约束违反度，标量或 (N,) / (N, 1)

        Returns:
            bool: 是否至少添加了一个新解
        """
        X = np.atleast_2d(X)
        F = np.atleast_2d(F)
        CV = np.atleast_1d(CV)

        if CV.ndim == 0:
            CV = CV.reshape(1)
        elif CV.ndim == 2:
            CV = CV.flatten()

        added_count = 0
        for i in range(len(X)):
            x = X[i]
            f = F[i] if F.shape[0] > 1 else F[0]
            cv = CV[i] if len(CV) > 1 else CV[0]

            self.total_count += 1

            idx = self._find_duplicate(x)
            if idx is None:
                self._X_list.append(x.copy())
                self._F_list.append(np.asarray(f).flatten().copy())
                self._CV_list.append(float(cv))
                added_count += 1
            else:
                current_f = self._F_list[idx]
                current_cv = self._CV_list[idx]
                new_f = np.asarray(f).flatten()

                # 多目标更优替换：cv 小者胜；cv 相等且 new 支配 old 则替换
                if cv < current_cv:
                    self._X_list[idx] = x.copy()
                    self._F_list[idx] = new_f.copy()
                    self._CV_list[idx] = float(cv)
                elif cv == current_cv and self._dominates(new_f, current_f):
                    self._X_list[idx] = x.copy()
                    self._F_list[idx] = new_f.copy()
                    self._CV_list[idx] = float(cv)

        return added_count > 0

    def get_all(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """获取所有去重后的历史记录（F shape (N, M)）"""
        if len(self._X_list) == 0:
            return np.array([]), np.array([]), np.array([])

        X = np.vstack(self._X_list)
        F = np.vstack([np.asarray(f).reshape(1, -1) for f in self._F_list])
        CV = np.array(self._CV_list)
        return X, F, CV


# ============================================================================
# HA-NSGA-III 主类
# ============================================================================
class   HA_NSGA3(HA):
    """
    HA-NSGA-III: 多目标混合进化算法

    在 HA 的基础上：
        - 用 NSGA-III 的非支配排序 + 参考方向小生境替换原约束支配排序与聚类
        - 用 PBI 标量化把局部搜索器（Nelder-Mead / RBF / GP / Adam / ...）
          适配到多目标场景，每个 niche 代表使用其关联的参考方向做局部下降
        - 保留 HA 的遗传操作、变异、历史信息基础设施

    Args:
        method: 局部搜索方法（与 HA 一致）
        pop_size: 种群大小
        niche_num: 每代选作局部搜索的 niche 代表数量
        mutation_rate: 变异率
        inherit_rate: 遗传率
        activate_method: 是否启用局部搜索
        ref_dirs: 用户自定义参考方向 (H, M)；为 None 时根据 n_obj 自动生成
        n_partitions: das-dennis 分区数；为 None 时按目标数自适应
        pbi_theta: PBI/IPBI 标量化中的惩罚系数 theta（默认 5.0）
        scalarization: 标量化方法，∈ {pbi, tchebycheff, ws, asf, aasf, ipbi}（默认 "pbi"）
            注：method="MO-Nelder-Mead" 时，scalarization 仅用于 fallback 通道
        niche_strategy: "ref_dirs"（NSGA-III 风格）或 "kmeans"（消融对比）
        cluster_method: 当 niche_strategy="kmeans" 时使用的聚类方法
        seed: 随机种子
        X: 初始种群（可选）
    """

    NICHE_STRATEGIES = ["ref_dirs", "kmeans"]
    SCALARIZATION_METHODS = ["pbi", "tchebycheff", "ws", "asf", "aasf", "ipbi"]

    # 模块级 WS 警告标志：仅在第一次使用 WS 时发警告
    _ws_warned: bool = False

    def __init__(
        self,
        method: str = "Nelder-Mead",
        pop_size: int = 100,
        niche_num: int = 3,
        mutation_rate: float = 0.3,
        inherit_rate: float = 0.8,
        activate_method: bool = True,
        ref_dirs: Optional[ArrayLike] = None,
        n_partitions: Optional[int] = None,
        pbi_theta: float = 5.0,
        scalarization: str = "pbi",
        niche_strategy: str = "ref_dirs",
        cluster_method: str = "kmeans",
        dbscan_eps: Optional[float] = None,
        dbscan_min_samples: Optional[int] = None,
        X: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
        use_dynamic_nadir: bool = False,
        use_coverage_injection: bool = False,
        **kwargs
    ) -> None:
        if niche_strategy not in self.NICHE_STRATEGIES:
            raise ValueError(
                f"不支持的 niche_strategy: {niche_strategy}，"
                f"支持: {self.NICHE_STRATEGIES}"
            )
        if scalarization.lower() not in self.SCALARIZATION_METHODS:
            raise ValueError(
                f"不支持的 scalarization: {scalarization}，"
                f"支持: {self.SCALARIZATION_METHODS}"
            )

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

        # 多目标专属参数
        self.ref_dirs: Optional[ArrayLike] = (
            np.asarray(ref_dirs, dtype=float) if ref_dirs is not None else None
        )
        self.n_partitions: Optional[int] = n_partitions
        self.pbi_theta: float = float(pbi_theta)
        self.scalarization: str = scalarization.lower()
        self.niche_strategy: str = niche_strategy
        # 改动一：动态 nadir 估计开关（默认 False = 原始累积行为）
        self.use_dynamic_nadir: bool = bool(use_dynamic_nadir)
        # 改动三：基于覆盖率的多样性注入开关（默认 False = 原始 x 空间唯一数）
        self.use_coverage_injection: bool = bool(use_coverage_injection)

        # 覆盖父类 history，启用多目标支持
        self.history = MOPopulationHistory()

        # 多目标运行时状态
        self.n_obj: int = 1
        self.z_star: Optional[ArrayLike] = None   # 理想点 (M,)
        self.z_nad: Optional[ArrayLike] = None    # 最差点 / 极值估计 (M,)
        self.front0: Optional[ArrayLike] = None   # 当前 Pareto 近似（决策空间）
        self.front0_F: Optional[ArrayLike] = None  # 当前 Pareto 近似（目标空间）
        self.last_indicator: float = float("inf")  # 用于停滞检测的指标

    # ========================================================================
    # 初始化
    # ========================================================================
    def _setup(self, problem: Any, **kwargs) -> None:
        """扩展父类 _setup：生成参考方向"""
        super()._setup(problem, **kwargs)

        self.n_obj = int(problem.n_obj)

        if self.ref_dirs is None:
            self.ref_dirs = self._build_ref_dirs(self.n_obj, self.pop_size,
                                                  self.n_partitions)

        # 初始化 z_star / z_nad
        self.z_star = np.full(self.n_obj, np.inf)
        self.z_nad = np.full(self.n_obj, -np.inf)

        print(f"[HA-NSGA3] n_obj={self.n_obj}, ref_dirs.shape={self.ref_dirs.shape}, "
              f"pbi_theta={self.pbi_theta}, niche_strategy={self.niche_strategy}")

    @staticmethod
    def _build_ref_dirs(
        n_obj: int,
        pop_size: int,
        n_partitions: Optional[int]
    ) -> ArrayLike:
        """根据目标数自适应生成 das-dennis 参考方向"""
        if n_obj <= 1:
            return np.array([[1.0]])

        if n_partitions is not None:
            return get_reference_directions(
                "das-dennis", n_obj, n_partitions=int(n_partitions)
            )

        if n_obj == 2:
            n_p = max(2, pop_size - 1)
            return get_reference_directions("das-dennis", 2, n_partitions=n_p)
        elif n_obj == 3:
            return get_reference_directions("das-dennis", 3, n_partitions=12)
        elif n_obj == 4:
            return get_reference_directions("das-dennis", 4, n_partitions=8)
        else:
            return get_reference_directions(
                "das-dennis", n_obj, n_partitions=max(2, 12 - n_obj)
            )

    # ========================================================================
    # 评估函数（多目标版本）
    # ========================================================================
    def evaluate_fitness_cv(
        self,
        x: ArrayLike
    ) -> Tuple[ArrayLike, float]:
        """评估单个个体（多目标 F shape (M,)）"""
        if self.problem is None:
            raise RuntimeError("问题未初始化，无法评估适应度")
        if np.any(x > self.ub) or np.any(x < self.lb):
            raise ValueError("x out of bounds")

        x_2d = np.atleast_2d(x)
        out: dict = {}
        self.problem._evaluate(x_2d, out)
        F = np.asarray(out["F"]).reshape(1, -1)  # (1, M)

        if hasattr(self.problem, "has_constraints") and self.problem.has_constraints():
            G = np.atleast_2d(out.get("G", np.zeros((1, 0))))
            cv = float(np.sum(np.maximum(0, G)))
        else:
            cv = 0.0

        self.history.add(x_2d, F, cv)
        return F.flatten(), cv

    def evaluate_fitness_cv_batch(
        self,
        X: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """批量评估个体（多目标 F shape (N, M)）"""
        if self.problem is None:
            raise RuntimeError("问题未初始化，无法评估适应度")
        print(f"进入 evaluate_fitness_cv_batch（MO），X.shape={X.shape}")

        out: dict = {}
        self.problem._evaluate(X, out)
        F = np.asarray(out["F"])  # 期望 (N, M)
        if F.ndim == 1:
            F = F.reshape(-1, 1)
        G = out.get("G", None)

        if G is None:
            cv = np.zeros((len(X), 1))
        else:
            G = np.asarray(G)
            if G.ndim == 1:
                G = G.reshape(-1, 1)
            cv = np.sum(np.maximum(0, G), axis=1).reshape(-1, 1)

        # 历史存储
        self.history.add(X, F, cv)
        return F, cv

    def _eval_cached(
        self,
        x: ArrayLike
    ) -> Tuple[ArrayLike, float]:
        """
        优先从历史缓存中获取评估结果，缓存未命中时才调用真实评估器。

        与 evaluate_fitness_cv 的唯一区别：先查 self.history，命中则直接返回，
        不产生额外的 FE。用于局部搜索结束后获取 new_F/new_cv 时避免重复评估。

        Returns:
            (F_flat, cv): 与 evaluate_fitness_cv 返回格式完全一致。
        """
        cached = self.history.get(x)
        if cached is not None:
            return cached
        return self.evaluate_fitness_cv(x)

    # ========================================================================
    # 主进化循环
    # ========================================================================
    def _initialize_infill(self) -> Population:
        """初始化种群（保留 F 为 (N, M)）"""
        if self.lb.size == 0 or self.ub.size == 0 or self.dim == 0:
            raise RuntimeError("算法尚未完成 _setup，无法初始化种群")
        print("[HA-NSGA3] 初始化种群...")

        if self.X is None:
            if self.seed is not None:
                rng = np.random.default_rng(self.seed)
                pop_x = rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
            else:
                pop_x = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        else:
            pop_x = np.asarray(self.X, dtype=float)

        pop_F, pop_cv = self.evaluate_fitness_cv_batch(pop_x)
        self.pop_cv = np.asarray(pop_cv).reshape(-1, 1)

        # 更新 z_star / z_nad
        self._update_ideal_nadir(pop_F, self.pop_cv)

        print(f"[HA-NSGA3] pop.shape={pop_x.shape}, F.shape={pop_F.shape}, "
              f"cv.shape={self.pop_cv.shape}")
        return Population.new("X", pop_x, "F", pop_F)

    def _infill(self) -> Population:
        """执行一代进化（多目标）"""
        if self.n_gen is None or self.problem is None:
            raise RuntimeError("算法状态未初始化，无法进化")
        print(f"\n[HA-NSGA3] === Generation {self.n_gen - 1} -> {self.n_gen} ===")

        pop = np.asarray(self.pop.get("X"))
        F = np.asarray(self.pop.get("F"))
        cv = np.asarray(self.pop_cv).reshape(-1, 1)

        # 更新理想点 / 最差点
        self._update_ideal_nadir(F, cv)

        # 当前 front 0（仅作监控/打印）
        front0_idx = self._first_front_indices(F, cv)
        self.front0 = pop[front0_idx]
        self.front0_F = F[front0_idx]

        # 计算当前指标用于停滞检测：front0 大小越大、IGD 越小越好
        current_indicator = self._compute_indicator(self.front0_F)
        if current_indicator < self.last_indicator - 1e-10:
            self.improvement = True
            self.stagnation_count = 0
            self.last_indicator = current_indicator
        else:
            self.improvement = False
            self.stagnation_count += 1

        # 日志
        best_idx = self._find_best_individual(F, cv)
        print(f"  front0 size = {len(front0_idx)}, indicator = {current_indicator:.6e}, "
              f"stagnation = {self.stagnation_count}")
        print(f"  best (PBI) idx = {best_idx}, F[best] = {F[best_idx]}")

        # 执行一步 HA 进化
        new_pop, new_F, new_cv = self._step_ha(pop, F, cv)

        self.pop = Population.new("X", new_pop, "F", new_F)
        self.pop_cv = new_cv
        return self.pop

    # ========================================================================
    # HA 核心步骤（多目标版本）
    # ========================================================================
    def _step_ha(
        self,
        pop: ArrayLike,
        F: ArrayLike,
        cv: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        多目标一步进化（标准 μ+μ）：
        局部搜索 -> 生成 N 个后代 -> P_t ∪ Q_t（2N）-> NSGA-III 环境选择保留 N
        """
        # 1) 更新理想点 / 最差点
        self._update_ideal_nadir(F, cv)

        # 2) 聚类与局部搜索（多目标版本）
        pop, F, cv, self.local_elite_id = self._clustering_and_learning(pop, F, cv)

        if not self.check_bounds(pop):
            raise ValueError("[HA-NSGA3] 聚类学习后种群越界")

        # 局部搜索后重新计算关联（F 可能已被更新）
        self._update_ideal_nadir(F, cv)
        niche_idx, d2 = self._associate_to_ref_dirs(F)

        # 3) 非支配排序 + 复合排序，为遗传操作提供 front_rank / d2
        front_rank = self._nondominated_rank(F, cv)
        sorted_indices = self._lexsort_by_front_and_d2(front_rank, d2)

        pop = pop[sorted_indices]
        F = F[sorted_indices]
        cv = cv[sorted_indices]
        front_rank = front_rank[sorted_indices]
        niche_idx = niche_idx[sorted_indices]
        d2 = d2[sorted_indices]

        # 4) 生成 N 个后代 Q_t（来自全量父代 P_t）
        offspring = self._inheritance_mo(
            self.pop_size, pop, F, front_rank, d2
        )
        mutate_num = round(self.pop_size * self.mutation_rate)
        if mutate_num > 0:
            mutate_id = self._rng.choice(
                len(offspring),
                min(mutate_num, len(offspring)),
                replace=False
            )
            offspring[mutate_id, :] = self._mutate(offspring[mutate_id, :])

        if not self.check_bounds(offspring):
            raise ValueError("[HA-NSGA3] 变异后越界")

        # 5) 后代去重后评估（重复个体瓜分 niche 名额无收益）
        offspring = np.unique(offspring, axis=0)
        print(f"[HA-NSGA3] 处理重复后 offspring.shape={offspring.shape}")

        offspring_F, offspring_cv = self.evaluate_fitness_cv_batch(offspring)
        offspring_cv = np.asarray(offspring_cv).reshape(-1, 1)

        # 6) 合并：全量父代 P_t（N）+ 全量后代 Q_t（≤N）→ R_t（≈2N）
        combined_pop = np.vstack((pop, offspring))
        combined_F = np.vstack((F, offspring_F))
        combined_cv = np.vstack((cv, offspring_cv))

        # 7) 合并后严格去重，避免重复个体瓜分小生境名额
        before_size = combined_pop.shape[0]
        combined_pop, combined_F, combined_cv = self._unique_with_tol(
            combined_pop, combined_F, combined_cv, tol=1e-8
        )
        after_size = combined_pop.shape[0]
        if after_size < before_size:
            print(
                f"[HA-NSGA3] 合并去重: {before_size} -> {after_size}"
                f" (移除 {before_size - after_size} 个近似重复个体)"
            )

        # 8) NSGA-III 环境选择：从 R_t（≈2N）中保留最好的 N 个
        new_pop, new_F, new_cv = self._nsga3_environmental_selection(
            combined_pop, combined_F, combined_cv
        )

        # 9) 监控多样性，必要时注入随机个体
        N = self.pop_size
        replace_num = int(N * 0.3)
        hard_floor = max(2, N // 20)

        if self.use_coverage_injection:
            # ── 改动三：基于参考方向覆盖率的自适应触发 ──────────────────────
            niche_idx_new, _ = self._associate_to_ref_dirs(new_F)
            H = len(self.ref_dirs)
            occupied = set(niche_idx_new.tolist())
            actual_coverage = len(occupied)
            # 自适应阈值：随 N/H 缩放，始终在物理可达范围内
            coverage_threshold = int(0.5 * min(N, H))
            coverage_triggered = actual_coverage < coverage_threshold
            hard_triggered = np.unique(new_pop, axis=0).shape[0] <= hard_floor
            print(f"[HA-NSGA3] niche覆盖={actual_coverage}/{H} "
                  f"(阈值={coverage_threshold})  唯一x={np.unique(new_pop,axis=0).shape[0]}")

            if coverage_triggered or hard_triggered:
                reason = []
                if coverage_triggered:
                    reason.append(f"覆盖率{actual_coverage}<{coverage_threshold}")
                if hard_triggered:
                    reason.append("x空间硬底线")
                print(f">>> [HA-NSGA3] 触发多样性注入（{'、'.join(reason)}），注入 {replace_num} 个")

                empty_niches = [k for k in range(H) if k not in occupied]

                # 生成候选并评估（候选数 = 注入量的 3 倍，提高命中空缺 niche 的概率）
                n_cand = replace_num * 3
                cand_x = self._rng.uniform(self.lb, self.ub, (n_cand, self.dim))
                cand_F, cand_cv = self.evaluate_fitness_cv_batch(cand_x)
                cand_cv = np.asarray(cand_cv).reshape(-1, 1)
                self._update_ideal_nadir(cand_F, cand_cv)
                cand_niche, _ = self._associate_to_ref_dirs(cand_F)

                # 为每个注入槽挑选候选：优先选落入空缺 niche 的候选
                injected_x, injected_F, injected_cv = [], [], []
                cycle_niches = (empty_niches * (replace_num + 1))[:replace_num]
                used_cand = set()
                for k in cycle_niches:
                    # 找落入 niche k 的未使用候选
                    hits = [i for i in np.where(cand_niche == k)[0]
                            if i not in used_cand]
                    if hits:
                        chosen = hits[0]
                    else:
                        # 没有候选落入该 niche：随机选一个未用候选
                        unused = [i for i in range(n_cand) if i not in used_cand]
                        chosen = unused[0] if unused else self._rng.integers(n_cand)
                    used_cand.add(chosen)
                    injected_x.append(cand_x[chosen])
                    injected_F.append(cand_F[chosen])
                    injected_cv.append(cand_cv[chosen])

                injected_x  = np.array(injected_x)
                injected_F  = np.array(injected_F)
                injected_cv = np.array(injected_cv)

                # 替换最拥挤 niche 中的个体（一驱一补，覆盖率提升最大）
                niche_count = np.zeros(H, dtype=int)
                for k in niche_idx_new:
                    niche_count[k] += 1
                niche_idx_tmp = niche_idx_new.copy()
                replace_indices = []
                for _ in range(replace_num):
                    crowded = int(np.argmax(niche_count))
                    candidates_in = np.where(niche_idx_tmp == crowded)[0]
                    if len(candidates_in) == 0:
                        break
                    victim = int(candidates_in[0])
                    replace_indices.append(victim)
                    niche_idx_tmp[victim] = -1
                    niche_count[crowded] -= 1

                for slot, idx in enumerate(replace_indices[:len(injected_x)]):
                    new_pop[idx] = injected_x[slot]
                    new_F[idx]   = injected_F[slot]
                    new_cv[idx]  = injected_cv[slot]
        else:
            # ── 原始行为：x 空间唯一数硬底线 ────────────────────────────────
            unique_count = np.unique(new_pop, axis=0).shape[0]
            print(f"[HA-NSGA3] 种群唯一个体数: {unique_count}")
            if unique_count <= hard_floor:
                print(">>> [HA-NSGA3] 检测到种群严重趋同，注入 30% 随机个体")
                random_pop = self._rng.uniform(self.lb, self.ub, (replace_num, self.dim))
                random_F, random_cv = self.evaluate_fitness_cv_batch(random_pop)
                new_pop[-replace_num:] = random_pop
                new_F[-replace_num:]   = random_F
                new_cv[-replace_num:]  = np.asarray(random_cv).reshape(-1, 1)

        return new_pop, new_F, new_cv

    # ========================================================================
    # NSGA-III 核心操作
    # ========================================================================
    def _update_ideal_nadir(self, F: ArrayLike, cv: ArrayLike) -> None:
        """根据当前 pop 与历史更新理想点/最差点（仅考虑可行解）。

        use_dynamic_nadir=False（默认）：z_nad 历史累积取最大，行为与原版一致。
        use_dynamic_nadir=True（改动一）：z_nad 每代用当前可行 front0 极值重估，
            随种群收敛动态收缩，避免早期劣解撑大归一化分母导致关联区分度塌缩。
        """
        cv_flat = np.asarray(cv).flatten()
        feasible_mask = cv_flat <= 0

        if np.any(feasible_mask):
            F_feas = F[feasible_mask]
            cur_min = np.min(F_feas, axis=0)
            cur_max = np.max(F_feas, axis=0)
        else:
            F_feas = np.asarray(F)
            cur_min = np.min(F_feas, axis=0)
            cur_max = np.max(F_feas, axis=0)

        # z_star：单调逼近理想点（两种模式均累积取最小）
        if self.z_star is None:
            self.z_star = cur_min.astype(float)
        else:
            self.z_star = np.minimum(self.z_star, cur_min)

        if not self.use_dynamic_nadir:
            # ── 原始行为：历史累积取最大 ──────────────────────────────────────
            if self.z_nad is None:
                self.z_nad = cur_max.astype(float)
            else:
                self.z_nad = np.maximum(self.z_nad, cur_max)
        else:
            # ── 改动一：每代用当前可行 front0 极值重估 ───────────────────────
            # 对可行解做非支配排序，取 front0 中的逐目标最大值
            front_rank = self._nondominated_rank(F_feas,
                                                 np.zeros(len(F_feas)))
            front0_mask = front_rank == 0
            if front0_mask.sum() >= 2:
                z_nad_new = np.max(F_feas[front0_mask], axis=0).astype(float)
            else:
                # front0 解过少时回退到全量可行解极值，保证稳健
                z_nad_new = cur_max.astype(float)
            # 加下限保护：分母至少为 1e-6，避免归一化退化
            self.z_nad = np.maximum(z_nad_new, self.z_star + 1e-6)

    def _normalize_F(self, F: ArrayLike) -> ArrayLike:
        """归一化目标值到 [0, 1]"""
        denom = self.z_nad - self.z_star
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        return (F - self.z_star) / denom

    def _associate_to_ref_dirs(
        self,
        F: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        将每个个体关联到最近的参考方向

        Returns:
            niche_idx: shape (N,)，每个个体关联的参考方向索引
            d2: shape (N,)，每个个体到关联方向的垂直距离
        """
        F_norm = self._normalize_F(F)
        # 参考方向归一化（每行）
        W = self.ref_dirs
        W_norm = W / np.maximum(
            np.linalg.norm(W, axis=1, keepdims=True), 1e-12
        )

        # d2[i, k] = ||F_norm[i] - (F_norm[i]·W_norm[k]) * W_norm[k]||
        # 利用广播
        # proj[i, k] = F_norm[i] @ W_norm[k]
        proj = F_norm @ W_norm.T  # (N, H)
        # F_proj[i, k, m] = proj[i, k] * W_norm[k, m]
        F_proj = proj[:, :, None] * W_norm[None, :, :]  # (N, H, M)
        diff = F_norm[:, None, :] - F_proj  # (N, H, M)
        d2_matrix = np.linalg.norm(diff, axis=2)  # (N, H)

        niche_idx = np.argmin(d2_matrix, axis=1)
        d2 = d2_matrix[np.arange(len(F)), niche_idx]
        return niche_idx, d2

    def _nondominated_rank(
        self,
        F: ArrayLike,
        cv: ArrayLike
    ) -> ArrayLike:
        """
        计算非支配 rank（含 CDP 约束支配）

        Returns:
            rank: shape (N,)，可行解按非支配 rank（0 起），不可行解按 cv 排序在可行解之后
        """
        cv_flat = np.asarray(cv).flatten()
        feasible_mask = cv_flat <= 0
        N = len(F)
        rank = np.full(N, 0, dtype=int)

        if np.any(feasible_mask):
            feas_idx = np.where(feasible_mask)[0]
            F_feas = F[feas_idx]
            nds = NonDominatedSorting()
            fronts = nds.do(F_feas)
            for r, front in enumerate(fronts):
                rank[feas_idx[front]] = r
            max_feas_rank = len(fronts) - 1
        else:
            max_feas_rank = -1

        if np.any(~feasible_mask):
            infeas_idx = np.where(~feasible_mask)[0]
            # 不可行解按 cv 升序，rank 在可行解之后
            sorted_infeas = infeas_idx[np.argsort(cv_flat[infeas_idx])]
            for j, idx in enumerate(sorted_infeas):
                rank[idx] = max_feas_rank + 1 + j

        return rank

    @staticmethod
    def _lexsort_by_front_and_d2(
        front_rank: ArrayLike,
        d2: ArrayLike
    ) -> ArrayLike:
        """复合排序：先按 front_rank，再按 d2"""
        return np.lexsort((d2, front_rank))

    @staticmethod
    def _unique_with_tol(
        X: ArrayLike,
        F: ArrayLike,
        CV: ArrayLike,
        tol: float = 1e-8,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        按 L∞ 距离容差去重决策变量，同步保留对应的 F 和 CV

        基于排序 + 相邻容差比较（O(N log N) + O(N) 扫描），适合 N <= 几千的种群规模。
        当某一行被判为重复时，保留**先出现**的那一行（即排序后较小的索引）。

        Args:
            X: (N, dim) 决策变量
            F: (N, M) 目标值
            CV: (N, 1) 或 (N,) 约束违反度
            tol: L∞ 距离阈值（默认 1e-8）

        Returns:
            (X_unique, F_unique, CV_unique)：按字典序去重后的子集
        """
        X = np.atleast_2d(X)
        if X.size == 0:
            return X, F, CV
        N = X.shape[0]
        if N <= 1:
            return X, F, CV

        # 按字典序排序
        order = np.lexsort(X.T[::-1])
        X_sorted = X[order]
        F_sorted = F[order]
        CV_sorted = CV[order]

        keep_mask = np.ones(N, dtype=bool)
        for i in range(1, N):
            if np.max(np.abs(X_sorted[i] - X_sorted[i - 1])) < tol:
                keep_mask[i] = False

        return X_sorted[keep_mask], F_sorted[keep_mask], CV_sorted[keep_mask]

    def _first_front_indices(
        self,
        F: ArrayLike,
        cv: ArrayLike
    ) -> ArrayLike:
        """返回当前可行解的第 0 个 front 索引（若无可行解，返回 cv 最小的索引）"""
        cv_flat = np.asarray(cv).flatten()
        feasible_mask = cv_flat <= 0
        if np.any(feasible_mask):
            feas_idx = np.where(feasible_mask)[0]
            nds = NonDominatedSorting()
            fronts = nds.do(F[feas_idx])
            return feas_idx[fronts[0]]
        else:
            return np.array([int(np.argmin(cv_flat))])

    def _nsga3_environmental_selection(
        self,
        X: ArrayLike,
        F: ArrayLike,
        CV: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        NSGA-III 环境选择：非支配排序 -> 累积填入 -> 临界 front 用 niche-count 选择
        """
        N = len(X)
        cv_flat = np.asarray(CV).flatten()
        feasible_mask = cv_flat <= 0
        target = self.pop_size

        selected: List[int] = []

        # 1) 可行解优先：非支配排序后从 front 0 开始填
        if np.any(feasible_mask):
            feas_idx = np.where(feasible_mask)[0]
            nds = NonDominatedSorting()
            fronts = nds.do(F[feas_idx])

            for front in fronts:
                front_global = feas_idx[front]
                if len(selected) + len(front_global) <= target:
                    selected.extend(front_global.tolist())
                else:
                    # 临界 front：使用 niche count 选择
                    last_front_global = front_global
                    remaining = target - len(selected)
                    chosen = self._niching(
                        F, CV, np.array(selected, dtype=int), last_front_global,
                        remaining
                    )
                    selected.extend(chosen.tolist())
                    break

            if len(selected) >= target:
                selected = selected[:target]

        # 2) 若可行解不足，补充约束违反度最小的不可行解
        if len(selected) < target:
            need = target - len(selected)
            if np.any(~feasible_mask):
                infeas_idx = np.where(~feasible_mask)[0]
                sorted_infeas = infeas_idx[np.argsort(cv_flat[infeas_idx])]
                # 排除已选
                already = set(selected)
                fill = [i for i in sorted_infeas.tolist() if i not in already][:need]
                selected.extend(fill)

        selected_arr = np.array(selected, dtype=int)
        # 最终再做一次复合排序（front_rank, d2）保证 pop[0] 是当前最佳
        front_rank = self._nondominated_rank(F[selected_arr], CV[selected_arr])
        _, d2 = self._associate_to_ref_dirs(F[selected_arr])
        order = self._lexsort_by_front_and_d2(front_rank, d2)
        selected_arr = selected_arr[order]

        return X[selected_arr], F[selected_arr], CV[selected_arr]

    def _niching(
        self,
        F: ArrayLike,
        CV: ArrayLike,
        already_selected: ArrayLike,
        last_front: ArrayLike,
        n_to_choose: int
    ) -> ArrayLike:
        """
        NSGA-III 小生境选择：在临界 front 中按 niche-count 选 n_to_choose 个

        Args:
            F: 全部候选目标值 (N, M)
            CV: 全部候选约束 (N, 1)
            already_selected: 已选解的索引（在合并 pop 中）
            last_front: 临界 front 在合并 pop 中的索引
            n_to_choose: 还需要选择的数量

        Returns:
            chosen: 选中的索引（在合并 pop 中）
        """
        # 计算每个解关联的 niche
        niche_idx, _ = self._associate_to_ref_dirs(F)
        H = self.ref_dirs.shape[0]

        # 已选解的 niche count
        niche_count = np.zeros(H, dtype=int)
        if len(already_selected) > 0:
            sel_niches = niche_idx[already_selected]
            for k in sel_niches:
                niche_count[k] += 1

        chosen: List[int] = []
        last_front_set = set(last_front.tolist())
        # 同时维护未被选的 last_front 索引
        remaining = list(last_front)

        # 计算 last_front 每个解到其 niche 的 d2
        _, d2_all = self._associate_to_ref_dirs(F)

        while len(chosen) < n_to_choose and len(remaining) > 0:
            # 1) 找出 niche_count 最小的 niche（在 last_front 涉及的 niche 中）
            remaining_niches = np.unique(niche_idx[remaining])
            if len(remaining_niches) == 0:
                break
            min_count = niche_count[remaining_niches].min()
            tied_niches = remaining_niches[
                niche_count[remaining_niches] == min_count
            ]
            # 2) 随机挑一个 niche 处理
            k = int(self._rng.choice(tied_niches))

            # 3) 在该 niche 的 last_front 候选中挑一个解
            cand = [i for i in remaining if niche_idx[i] == k]
            if len(cand) == 0:
                continue

            if min_count == 0:
                # 该 niche 当前为空，选 d2 最小（最贴近方向）
                cand_d2 = np.array([d2_all[i] for i in cand])
                pick = cand[int(np.argmin(cand_d2))]
            else:
                # 该 niche 已有解，随机挑一个
                pick = int(self._rng.choice(cand))

            chosen.append(pick)
            niche_count[k] += 1
            remaining.remove(pick)

        return np.array(chosen, dtype=int)

    # ========================================================================
    # 局部搜索（覆盖 HA._local_search）
    # ========================================================================
    def _make_scalarizer(
        self,
        method: str,
        w_k: ArrayLike,
        z_star: ArrayLike,
        z_nad: Optional[ArrayLike] = None,
        theta: Optional[float] = None,
        rho: float = 1e-4,
    ) -> Callable[[ArrayLike], float]:
        """
        多目标标量化目标函数工厂

        支持六种经典标量化方法：
            - "pbi"        : Penalty-based Boundary Intersection，d1 + theta * d2
            - "tchebycheff": max_i w_i * |f_i - z*_i|（MOEA/D 经典，凹凸前沿均匀）
            - "ws"         : weighted sum，Σ w_i * f_i（仅适合凸前沿）
            - "asf"        : max_i (f_i - z*_i) / w_i（NSGA-III 极值点估计同款）
            - "aasf"       : ASF + rho * Σ (f_i - z*_i) / w_i（可微版 ASF）
            - "ipbi"       : Inverted PBI，从 z_nad 反向投影（适合偏置/凹前沿）

        所有方法返回的标量目标自带约束惩罚：value + alpha * cv，其中
        alpha = abs(value) * 10 + 1，并对 NaN/Inf 做兜底（替换为 1e12）。

        Args:
            method: 标量化方法名（不区分大小写）
            w_k: 参考方向向量 (M,)
            z_star: 理想点 (M,)
            z_nad: 最差点 (M,)，IPBI 必需；其余方法可选用于归一化
            theta: PBI/IPBI 的惩罚系数，None 时退化为 self.pbi_theta
            rho: AASF 的增广系数（默认 1e-4，Deb 推荐值）

        Returns:
            Callable[[x], float]：给定决策变量 x，返回标量目标值
        """
        method = str(method).lower()
        if method not in self.SCALARIZATION_METHODS:
            raise ValueError(
                f"不支持的 scalarization 方法: {method}，"
                f"支持: {self.SCALARIZATION_METHODS}"
            )

        if theta is None:
            theta = self.pbi_theta

        w_k = np.asarray(w_k, dtype=float).flatten()
        z_star = np.asarray(z_star, dtype=float).flatten()
        w_norm = float(np.linalg.norm(w_k))
        w_hat = w_k / max(w_norm, 1e-12)
        # WS / TCH / ASF 用：对零权重的 w 加 1e-6 兜底
        w_safe = np.maximum(w_k, 1e-6)

        # IPBI 健壮性：z_nad 不稳定时退化到 PBI
        ipbi_fallback_to_pbi = False
        if method == "ipbi":
            if z_nad is None:
                z_nad_arr = (
                    np.asarray(self.z_nad, dtype=float).flatten()
                    if self.z_nad is not None
                    else None
                )
            else:
                z_nad_arr = np.asarray(z_nad, dtype=float).flatten()
            if z_nad_arr is None or (z_nad_arr - z_star).min() < 1e-6:
                ipbi_fallback_to_pbi = True
        else:
            z_nad_arr = (
                np.asarray(z_nad, dtype=float).flatten()
                if z_nad is not None
                else None
            )

        # WS 在多目标问题上的一次性警告
        if method == "ws" and self.n_obj >= 2 and not HA_NSGA3._ws_warned:
            warnings.warn(
                "Weighted Sum 仅适用于凸前沿，凹/不连续前沿请改用 tchebycheff 或 pbi",
                UserWarning,
                stacklevel=2,
            )
            HA_NSGA3._ws_warned = True

        def _safe_scalar(val: float) -> float:
            """处理 NaN / Inf"""
            if not np.isfinite(val):
                return 1e12
            return float(val)

        def objective(x: ArrayLike) -> float:
            x = np.asarray(x).flatten()
            hist = self.history.get(x)
            if hist is not None:
                F_x, cv = hist
            else:
                F_x, cv = self.evaluate_fitness_cv(x)
            F_x = np.asarray(F_x).flatten()
            F_shift = F_x - z_star

            if method == "pbi":
                d1 = float(np.dot(F_shift, w_hat))
                d2 = float(np.linalg.norm(F_shift - d1 * w_hat))
                val = d1 + theta * d2

            elif method == "tchebycheff":
                # 等价于 max_i w_i * |f_i - z*_i|（要求 z* <= F 时 |.| 简化为正）
                val = float(np.max(w_safe * np.abs(F_shift)))

            elif method == "ws":
                # 用 w_k（已经归一化或非负的 das-dennis）直接加权
                val = float(np.dot(w_k, F_x))

            elif method == "asf":
                val = float(np.max(F_shift / w_safe))

            elif method == "aasf":
                ratios = F_shift / w_safe
                val = float(np.max(ratios) + rho * np.sum(ratios))

            elif method == "ipbi":
                if ipbi_fallback_to_pbi:
                    d1 = float(np.dot(F_shift, w_hat))
                    d2 = float(np.linalg.norm(F_shift - d1 * w_hat))
                    val = d1 + theta * d2
                else:
                    # 从 nadir 反向投影：要让 F 远离 z_nad 且贴近方向
                    F_inv = z_nad_arr - F_x
                    d1p = float(np.dot(F_inv, w_hat))
                    d2p = float(np.linalg.norm(F_inv - d1p * w_hat))
                    # 越大越好 -> 取负号让"越小越好"统一
                    val = -(d1p - theta * d2p)
            else:
                raise ValueError(f"未实现的 scalarization: {method}")

            val = _safe_scalar(val)
            alpha = abs(val) * 10.0 + 1.0
            return val + alpha * float(cv)

        return objective

    def _make_pbi_objective(
        self,
        w_k: ArrayLike,
        z_star: ArrayLike,
        theta: Optional[float] = None
    ) -> Callable[[ArrayLike], float]:
        """
        生成 PBI 标量化目标函数（向后兼容薄包装层）

        新代码请直接使用 _make_scalarizer(method, ...)。

        Args:
            w_k: 参考方向向量
            z_star: 理想点
            theta: PBI 惩罚系数；None 时退化为 self.pbi_theta（注意 0.0 不被当作 None）
        """
        # 用 is None 显式判断，避免 theta=0.0 被误判
        effective_theta = self.pbi_theta if theta is None else theta
        return self._make_scalarizer(
            "pbi",
            w_k=w_k,
            z_star=z_star,
            theta=effective_theta,
        )

    # ========================================================================
    # 多目标支配版 Nelder-Mead
    # ========================================================================
    def _build_simplex(
        self,
        x0: ArrayLike,
        niche_filter_F: Optional[ArrayLike] = None,
        niche_filter_w: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        构建 Nelder-Mead 单纯形（共享工具函数，供 MO-NM 与标量 NM 使用）

        策略：
            1. 从 history 中找 x0 的最近邻（可选 niche 过滤：只选与 x0 同 niche 的历史点）
            2. 用 Gram-Schmidt 正交化筛选线性无关的方向
            3. 不足 dim+1 时沿标准轴扰动补齐
            4. 仍不足时随机扰动补齐
            5. 满秩性验证：若仍不满秩，返回 None（SciPy 会自动生成）

        Args:
            x0: 初始顶点 (dim,)
            niche_filter_F: 若提供 (N_hist, M)，配合 niche_filter_w 做 niche 过滤
            niche_filter_w: 该 niche 的参考方向 (M,)

        Returns:
            simplex_array: (dim+1, dim) 单纯形矩阵；构建失败时返回 None
        """
        dim = self.dim
        simplex_points: List[ArrayLike] = [x0.copy()]
        ortho_basis: List[ArrayLike] = []

        # 1. 从历史构建（可选 niche 过滤）
        if hasattr(self, "history") and self.history is not None:
            X_hist, F_hist, _ = self.history.get_all()
            if len(X_hist) > 0 and len(X_hist) >= dim + 1:
                X_candidate = X_hist

                # niche 过滤：优先选与 x0 同 niche 的历史点
                if (
                    niche_filter_F is not None
                    and niche_filter_w is not None
                    and len(F_hist) == len(X_hist)
                ):
                    try:
                        # 历史点按当前 ref_dirs 关联，找出与 niche_filter_w 同方向的点
                        nidx_hist, _ = self._associate_to_ref_dirs(F_hist)
                        # 找到 niche_filter_w 对应的 niche 索引
                        w_target = np.asarray(niche_filter_w).flatten()
                        w_norms = self.ref_dirs / np.maximum(
                            np.linalg.norm(self.ref_dirs, axis=1, keepdims=True), 1e-12
                        )
                        w_t_norm = w_target / max(float(np.linalg.norm(w_target)), 1e-12)
                        cos_sim = w_norms @ w_t_norm
                        target_niche = int(np.argmax(cos_sim))

                        # 按余弦相似度升序排序所有 niche，逐层扩大
                        order_niches = np.argsort(-cos_sim)
                        accumulated: List[int] = []
                        for kn in order_niches:
                            members = np.where(nidx_hist == int(kn))[0]
                            accumulated.extend(members.tolist())
                            if len(accumulated) >= max(dim * 5, 10):
                                break

                        if len(accumulated) >= dim + 1:
                            X_candidate = X_hist[accumulated]
                    except Exception:
                        X_candidate = X_hist  # 任何异常退化到全局

                try:
                    n_neighbors = min(len(X_candidate), dim * 5 + 2)
                    neigh = NearestNeighbors(n_neighbors=n_neighbors)
                    neigh.fit(X_candidate)
                    _, indices = neigh.kneighbors(x0.reshape(1, -1))
                    for idx in indices[0]:
                        cand = X_candidate[idx]
                        vec = cand - x0
                        if np.linalg.norm(vec) < 1e-6:
                            continue
                        # Gram-Schmidt 残差检查
                        u = vec.copy()
                        for b in ortho_basis:
                            u -= np.dot(u, b) * b
                        if np.linalg.norm(u) > 1e-5:
                            simplex_points.append(cand)
                            ortho_basis.append(u / np.linalg.norm(u))
                        if len(simplex_points) >= dim + 1:
                            break
                except Exception:
                    pass

        # 2. 沿标准轴扰动补齐
        if len(simplex_points) < dim + 1:
            eye = np.eye(dim)
            range_sizes = self.ub - self.lb
            base_step = max(float(np.min(range_sizes)) * 0.05, 1e-6)
            for i in range(dim):
                if len(simplex_points) >= dim + 1:
                    break
                dist_lb = x0[i] - self.lb[i]
                dist_ub = self.ub[i] - x0[i]
                if dist_ub > dist_lb and dist_ub > base_step:
                    step_vec = eye[i] * base_step
                elif dist_lb > base_step:
                    step_vec = -eye[i] * base_step
                else:
                    step_vec = eye[i] * min(dist_ub, dist_lb, base_step)
                    if abs(step_vec[i]) < 1e-8:
                        continue
                u = step_vec.copy()
                for b in ortho_basis:
                    u -= np.dot(u, b) * b
                if np.linalg.norm(u) > 1e-6:
                    new_p = np.clip(x0 + step_vec, self.lb, self.ub)
                    if np.linalg.norm(new_p - x0) > 1e-6:
                        simplex_points.append(new_p)
                        ortho_basis.append(u / np.linalg.norm(u))

        # 3. 随机扰动兜底
        if len(simplex_points) < dim + 1:
            base_step = max(float(np.min(self.ub - self.lb)) * 0.05, 1e-6)
            attempts = 0
            max_attempts = dim * 10
            while len(simplex_points) < dim + 1 and attempts < max_attempts:
                attempts += 1
                random_dir = self._rng.normal(size=dim)
                random_dir /= max(float(np.linalg.norm(random_dir)), 1e-12)
                step_size = base_step * (0.5 + self._rng.random())
                step_vec = random_dir * step_size
                u = step_vec.copy()
                for b in ortho_basis:
                    u -= np.dot(u, b) * b
                if np.linalg.norm(u) > 1e-6:
                    new_p = np.clip(x0 + step_vec, self.lb, self.ub)
                    if np.linalg.norm(new_p - x0) > 1e-6:
                        simplex_points.append(new_p)
                        ortho_basis.append(u / np.linalg.norm(u))

        # 4. 满秩验证
        if len(simplex_points) < dim + 1:
            return None
        simplex_array = np.array(simplex_points)
        vectors = simplex_array[1:] - simplex_array[0:1]
        if vectors.shape[0] >= dim:
            rank = np.linalg.matrix_rank(vectors[:dim])
            if rank < dim:
                return None
        return simplex_array

    def _d2_to_ref_dir(
        self,
        F_arr: ArrayLike,
        w_k: ArrayLike,
        z_star: ArrayLike,
        z_nad: ArrayLike,
    ) -> ArrayLike:
        """
        计算 F_arr 到参考方向 w_k 的归一化垂直距离 d2

        Args:
            F_arr: shape (M,) 或 (N, M)
            w_k: 参考方向 (M,)
            z_star/z_nad: 归一化用的理想点/最差点

        Returns:
            d2: 标量（输入 (M,)）或 (N,)（输入 (N, M)）
        """
        F_arr = np.atleast_2d(F_arr)
        denom = z_nad - z_star
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        F_norm = (F_arr - z_star) / denom
        w_hat = w_k / max(float(np.linalg.norm(w_k)), 1e-12)
        proj = F_norm @ w_hat
        diff = F_norm - proj[:, None] * w_hat[None, :]
        d2 = np.linalg.norm(diff, axis=1)
        return d2 if d2.shape[0] > 1 else float(d2[0])

    def _rank_vertices(
        self,
        F_verts: ArrayLike,
        CV_verts: ArrayLike,
        w_k: ArrayLike,
        z_star: ArrayLike,
        z_nad: ArrayLike,
    ) -> ArrayLike:
        """
        对 NM 单纯形顶点做 NSGA-III 风格复合排序

        排序优先级（best -> worst）：
            1. 可行性：cv <= 0 的顶点排在所有不可行顶点之前
            2. 不可行顶点之间：按 cv 升序
            3. 可行顶点之间：先按非支配 front rank 升序，再按到 w_k 的 d2 升序

        Args:
            F_verts: (V, M) 顶点目标值
            CV_verts: (V,) 或 (V, 1) 顶点约束违反度
            w_k: 当前 niche 的参考方向 (M,)
            z_star, z_nad: 归一化用的理想/最差点

        Returns:
            order: (V,) 顶点索引数组，从最优到最差
        """
        F_verts = np.atleast_2d(F_verts)
        cv_flat = np.asarray(CV_verts).flatten()
        V = F_verts.shape[0]

        # 1) 计算每个顶点的 front rank（仅在可行子集内做非支配排序）
        front_rank = np.full(V, V, dtype=int)  # 不可行先填一个大值
        feas_mask = cv_flat <= 0
        if np.any(feas_mask):
            feas_idx = np.where(feas_mask)[0]
            nds = NonDominatedSorting()
            fronts = nds.do(F_verts[feas_idx])
            for r, front in enumerate(fronts):
                front_rank[feas_idx[front]] = r

        # 2) 计算每个顶点到 w_k 的 d2
        d2_values = np.array(
            [
                self._d2_to_ref_dir(F_verts[i], w_k, z_star, z_nad)
                for i in range(V)
            ],
            dtype=float,
        )

        # 3) 复合 key：cv>0 优先级最低，cv 大者次低，front_rank 升序，d2 升序
        infeas_flag = (cv_flat > 0).astype(int)
        # np.lexsort 把最后一个 key 当作首要排序键
        order = np.lexsort(
            (d2_values, front_rank, cv_flat, infeas_flag)
        )
        return order

    def _is_better_mo(
        self,
        F_a: ArrayLike,
        cv_a: Union[float, ArrayLike],
        F_b: ArrayLike,
        cv_b: Union[float, ArrayLike],
        w_k: ArrayLike,
        z_star: ArrayLike,
        z_nad: ArrayLike,
    ) -> bool:
        """
        判断 a 是否"优于" b（用于 MO-NM 的接受准则）

        规则（与 _rank_vertices 一致）：
            - 可行 vs 不可行：可行胜
            - 不可行 vs 不可行：cv 小者胜
            - 可行 vs 可行：a 支配 b 则胜；互不支配时 d2 更小者胜
        """
        cv_a = float(np.atleast_1d(cv_a)[0])
        cv_b = float(np.atleast_1d(cv_b)[0])
        F_a_arr = np.asarray(F_a).flatten()
        F_b_arr = np.asarray(F_b).flatten()

        # 可行性优先
        if cv_a <= 0 and cv_b > 0:
            return True
        if cv_a > 0 and cv_b <= 0:
            return False
        if cv_a > 0 and cv_b > 0:
            return cv_a < cv_b

        # 都可行：先看支配
        a_dom_b = bool(np.all(F_a_arr <= F_b_arr) and np.any(F_a_arr < F_b_arr))
        b_dom_a = bool(np.all(F_b_arr <= F_a_arr) and np.any(F_b_arr < F_a_arr))
        if a_dom_b:
            return True
        if b_dom_a:
            return False

        # 互不支配：按 d2 距离判定
        d2_a = self._d2_to_ref_dir(F_a_arr, w_k, z_star, z_nad)
        d2_b = self._d2_to_ref_dir(F_b_arr, w_k, z_star, z_nad)
        return d2_a < d2_b

    def _mo_nelder_mead_local_search(
        self,
        x0: ArrayLike,
        w_k: ArrayLike,
        z_star: ArrayLike,
        z_nad: ArrayLike,
        maxiter: int = 30,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ) -> ArrayLike:
        """
        多目标支配版 Nelder-Mead 局部搜索

        手写 NM 主循环，所有顶点比较均用 NSGA-III 风格 (front rank + d2) 复合 key。
        与标量化驱动版相比：
            - 不受 PBI/TCH 偏好影响
            - 互不支配顶点用到 w_k 的 d2 距离 tie-break
            - 平局退化时自动切到 self.scalarization 驱动 2-3 步（fallback）

        约束：
            - 每次调用 FE 上限 max_fes_per_nm = 5 * (dim + 1)
            - 内部对 history 做缓存，避免重复评估
            - 单纯形退化（顶点到质心 d2 全 < tol）时提前终止

        Args:
            x0: 初始解 (dim,)
            w_k: 当前 niche 的参考方向 (M,)
            z_star, z_nad: 归一化用的理想点/最差点
            maxiter: 最大主循环迭代次数
            alpha/gamma/rho/sigma: NM 经典系数

        Returns:
            x_best: 主循环终止后最佳顶点的 x
        """
        if not self.check_bounds([x0]):
            raise ValueError("[HA-NSGA3:MO-NM] 初始解越界")

        dim = self.dim
        max_fes_per_nm = 5 * (dim + 1)

        # 局部 FE 计数（用 problem.fes 差值，不直接访问内部缓存）
        fes_before = self.problem.fes
        fes_used = 0  # 仅统计本次 NM 触发的新 FE

        def evaluate_with_cache(x: ArrayLike) -> Tuple[ArrayLike, float]:
            """优先查 history，未命中才真正调用 problem"""
            nonlocal fes_used
            x = np.clip(x, self.lb, self.ub)
            hit = self.history.get(x)
            if hit is not None:
                F_x, cv_x = hit
                return np.asarray(F_x).flatten(), float(cv_x)
            fes_before_eval = self.problem.fes
            F_x, cv_x = self.evaluate_fitness_cv(x)
            fes_used += self.problem.fes - fes_before_eval
            return np.asarray(F_x).flatten(), float(cv_x)

        # 1) 构建初始单纯形（可选用 niche 过滤）
        # 用历史 F 做 niche 过滤
        X_hist, F_hist, _ = self.history.get_all()
        niche_filter_F = F_hist if len(X_hist) > 0 else None

        simplex = self._build_simplex(
            x0, niche_filter_F=niche_filter_F, niche_filter_w=w_k
        )
        if simplex is None:
            # 极端兜底：返回 x0 不变
            print("[HA-NSGA3:MO-NM] 单纯形构建失败，返回 x0")
            return x0.copy()

        # 评估顶点
        F_verts = np.zeros((dim + 1, self.n_obj))
        CV_verts = np.zeros(dim + 1)
        for i in range(dim + 1):
            simplex[i] = np.clip(simplex[i], self.lb, self.ub)
            F_verts[i], CV_verts[i] = evaluate_with_cache(simplex[i])
            if fes_used >= max_fes_per_nm:
                # 评估初始顶点就用光了预算，返回当前最优
                order = self._rank_vertices(
                    F_verts[: i + 1], CV_verts[: i + 1], w_k, z_star, z_nad
                )
                return simplex[order[0]].copy()

        # 退化阈值（在归一化目标空间）
        d2_tol = 1e-6
        tied_tol = 1e-3  # 顶点 d2 差异小于此值视为平局退化

        # scalarizer fallback：在平局时使用
        scalarizer_for_fallback = self._make_scalarizer(
            self.scalarization, w_k=w_k, z_star=z_star, z_nad=z_nad
        )

        # 2) 主循环
        for it in range(maxiter):
            if fes_used >= max_fes_per_nm:
                break

            # 顶点排序
            order = self._rank_vertices(F_verts, CV_verts, w_k, z_star, z_nad)
            best, worst = int(order[0]), int(order[-1])
            second_worst = int(order[-2]) if len(order) >= 2 else best

            # 单纯形退化检测
            centroid = np.mean(simplex[order[:-1]], axis=0)
            d2_to_centroid = np.array(
                [
                    self._d2_to_ref_dir(F_verts[i], w_k, z_star, z_nad)
                    for i in range(dim + 1)
                ]
            )
            max_d2 = float(np.max(d2_to_centroid))
            min_d2 = float(np.min(d2_to_centroid))
            simplex_size = float(
                np.max(np.linalg.norm(simplex - centroid, axis=1))
            )
            if simplex_size < d2_tol:
                break

            # 平局退化检测：所有顶点都在 front 0 且 d2 差异极小 → scalarizer fallback
            all_front0 = np.all(
                [
                    self._rank_vertices(
                        F_verts, CV_verts, w_k, z_star, z_nad
                    )[i] == i
                    for i in range(min(2, dim + 1))
                ]
            )
            # 更精确的判定：直接看是否所有 feasible 顶点的 d2 都接近
            feas_mask = CV_verts <= 0
            if np.sum(feas_mask) >= 2:
                d2_feas = d2_to_centroid[feas_mask]
                if (d2_feas.max() - d2_feas.min()) < tied_tol:
                    # 切到 scalarizer 驱动 2 步反射
                    for _ in range(2):
                        if fes_used >= max_fes_per_nm:
                            break
                        s_vals = np.array(
                            [
                                scalarizer_for_fallback(simplex[i])
                                for i in range(dim + 1)
                            ]
                        )
                        scal_order = np.argsort(s_vals)
                        s_best = int(scal_order[0])
                        s_worst = int(scal_order[-1])
                        s_centroid = np.mean(
                            simplex[scal_order[:-1]], axis=0
                        )
                        x_r = np.clip(
                            s_centroid + alpha * (s_centroid - simplex[s_worst]),
                            self.lb,
                            self.ub,
                        )
                        F_r, cv_r = evaluate_with_cache(x_r)
                        s_r = scalarizer_for_fallback(x_r)
                        if s_r < s_vals[s_best]:
                            simplex[s_worst] = x_r
                            F_verts[s_worst] = F_r
                            CV_verts[s_worst] = cv_r
                        else:
                            # 内收缩
                            x_ic = np.clip(
                                s_centroid - rho * (s_centroid - simplex[s_worst]),
                                self.lb,
                                self.ub,
                            )
                            F_ic, cv_ic = evaluate_with_cache(x_ic)
                            s_ic = scalarizer_for_fallback(x_ic)
                            if s_ic < s_vals[s_worst]:
                                simplex[s_worst] = x_ic
                                F_verts[s_worst] = F_ic
                                CV_verts[s_worst] = cv_ic
                    continue  # 重新进入主循环（再用支配排序）

            # === 标准 NM 几何操作（用支配比较接受/拒绝） ===

            # 反射
            x_r = np.clip(
                centroid + alpha * (centroid - simplex[worst]),
                self.lb,
                self.ub,
            )
            F_r, cv_r = evaluate_with_cache(x_r)

            # x_r 是否优于 best
            r_better_than_best = self._is_better_mo(
                F_r, cv_r, F_verts[best], CV_verts[best], w_k, z_star, z_nad
            )
            # x_r 是否优于 second_worst
            r_better_than_2nd_worst = self._is_better_mo(
                F_r,
                cv_r,
                F_verts[second_worst],
                CV_verts[second_worst],
                w_k,
                z_star,
                z_nad,
            )
            # x_r 是否优于 worst
            r_better_than_worst = self._is_better_mo(
                F_r,
                cv_r,
                F_verts[worst],
                CV_verts[worst],
                w_k,
                z_star,
                z_nad,
            )

            if r_better_than_best:
                # 扩展
                x_e = np.clip(
                    centroid + gamma * (x_r - centroid), self.lb, self.ub
                )
                F_e, cv_e = evaluate_with_cache(x_e)
                e_better_than_r = self._is_better_mo(
                    F_e, cv_e, F_r, cv_r, w_k, z_star, z_nad
                )
                if e_better_than_r:
                    simplex[worst] = x_e
                    F_verts[worst] = F_e
                    CV_verts[worst] = cv_e
                else:
                    simplex[worst] = x_r
                    F_verts[worst] = F_r
                    CV_verts[worst] = cv_r
            elif r_better_than_2nd_worst:
                simplex[worst] = x_r
                F_verts[worst] = F_r
                CV_verts[worst] = cv_r
            elif r_better_than_worst:
                # 外收缩
                x_oc = np.clip(
                    centroid + rho * (x_r - centroid), self.lb, self.ub
                )
                F_oc, cv_oc = evaluate_with_cache(x_oc)
                oc_better_than_r = self._is_better_mo(
                    F_oc, cv_oc, F_r, cv_r, w_k, z_star, z_nad
                )
                if oc_better_than_r:
                    simplex[worst] = x_oc
                    F_verts[worst] = F_oc
                    CV_verts[worst] = cv_oc
                else:
                    # 单纯形收缩
                    best_pt = simplex[best].copy()
                    for i in range(dim + 1):
                        if i == best:
                            continue
                        simplex[i] = np.clip(
                            best_pt + sigma * (simplex[i] - best_pt),
                            self.lb,
                            self.ub,
                        )
                        F_verts[i], CV_verts[i] = evaluate_with_cache(simplex[i])
                        if fes_used >= max_fes_per_nm:
                            break
            else:
                # 内收缩
                x_ic = np.clip(
                    centroid - rho * (centroid - simplex[worst]),
                    self.lb,
                    self.ub,
                )
                F_ic, cv_ic = evaluate_with_cache(x_ic)
                ic_better_than_worst = self._is_better_mo(
                    F_ic,
                    cv_ic,
                    F_verts[worst],
                    CV_verts[worst],
                    w_k,
                    z_star,
                    z_nad,
                )
                if ic_better_than_worst:
                    simplex[worst] = x_ic
                    F_verts[worst] = F_ic
                    CV_verts[worst] = cv_ic
                else:
                    # 单纯形收缩
                    best_pt = simplex[best].copy()
                    for i in range(dim + 1):
                        if i == best:
                            continue
                        simplex[i] = np.clip(
                            best_pt + sigma * (simplex[i] - best_pt),
                            self.lb,
                            self.ub,
                        )
                        F_verts[i], CV_verts[i] = evaluate_with_cache(simplex[i])
                        if fes_used >= max_fes_per_nm:
                            break

        # 最终返回最佳顶点
        final_order = self._rank_vertices(F_verts, CV_verts, w_k, z_star, z_nad)
        x_best = simplex[final_order[0]].copy()
        return x_best

    def _local_search(
        self,
        x0: ArrayLike,
        y0: Union[float, ArrayLike],
        maxiter: int = 1,
        w_k: Optional[ArrayLike] = None,
        z_star: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """
        多目标局部搜索（基于 self.scalarization 的标量化 + 可选支配版 NM）

        覆盖父类 _local_search：
            - method == "MO-Nelder-Mead"：调用支配-niching 版手写 NM
              （直接用支配关系比较，不经过标量化）
            - 其他 method：用 self.scalarization 工厂构造标量目标，
              再调用现有子搜索方法（NM/RBF/GP/Adam 等，均无需修改）
        """
        if not self.check_bounds([x0]):
            raise ValueError("[HA-NSGA3] 初始解越界")

        if w_k is None:
            w_k = self.ref_dirs[0]
        if z_star is None:
            z_star = self.z_star if self.z_star is not None else np.zeros(self.n_obj)
        z_nad = (
            self.z_nad
            if self.z_nad is not None
            else (z_star + 1.0)
        )

        # 多目标支配版 Nelder-Mead：不经过 scalarizer
        if self.method == "MO-Nelder-Mead":
            x_result = self._mo_nelder_mead_local_search(
                x0,
                w_k=w_k,
                z_star=z_star,
                z_nad=z_nad,
                maxiter=max(self.dim, 10),
            )
            # 用支配关系判断是否接受新解
            F0, cv0 = self.evaluate_fitness_cv(x0)
            F1, cv1 = self.evaluate_fitness_cv(x_result)
            return (
                x_result
                if self._is_better_mo(F1, cv1, F0, cv0, w_k, z_star, z_nad)
                else x0
            )

        # 其余 method：基于 self.scalarization 构造标量目标
        objective = self._make_scalarizer(
            self.scalarization,
            w_k=w_k,
            z_star=z_star,
            z_nad=z_nad,
        )
        y0 = objective(x0)

        if self.method == "Nelder-Mead":
            x_result = self._nelder_mead_local_search(x0, y0, objective, maxiter=3)
        elif self.method == "rbf":
            x_result = self._rbf_local_search(x0, y0, objective, maxiter=maxiter)
        elif self.method == "gp":
            x_result = self._gp_local_search(x0, y0, objective, maxiter=3)
        elif self.method == "history-ladder":
            x_result = self._history_ladder_local_search(
                x0, y0, objective, maxiter=maxiter
            )
        elif self.method in self.BOUNDED_METHODS:
            x_result = self._scipy_local_search(x0, objective, maxiter=1)
        elif self.method == "Adam":
            x_result = self._adam_local_search(x0, y0, objective)
        elif self.method == "AdamW":   
            x_result = self._adamw_local_search(x0, y0, objective)
        elif self.method == "Lion":
            x_result = self._lion_local_search(x0, y0, objective)
        elif self.method == "Sophia":
            x_result = self._sophia_local_search(x0, y0, objective)
        else:
            raise ValueError(f"不支持的优化方法: {self.method}")

        # 标量空间比较：取较小者
        f0 = objective(x0)
        f1 = objective(x_result)
        return x_result if f1 < f0 else x0

    # ========================================================================
    # 替代聚类学习（参考方向 niching）
    # ========================================================================
    def _clustering_and_learning(
        self,
        pop: ArrayLike,
        F: ArrayLike,
        cv: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, List[int]]:
        """
        多目标聚类与局部学习

        策略：
            - niche_strategy == "ref_dirs"：基于参考方向关联选 niche 代表（推荐）
            - niche_strategy == "kmeans"：保留 HA 风格的聚类（消融对比）
        """
        if self.n_gen is None:
            raise RuntimeError("算法状态未初始化")
        if not self.activate_method:
            return pop, F, cv, []

        is_periodic = (self.n_gen % 5 == 0)
        is_stagnant = (self.stagnation_count >= 3)
        if not (is_periodic or is_stagnant):
            return pop, F, cv, []

        if not self.check_bounds(pop):
            raise ValueError("[HA-NSGA3] _clustering_and_learning 传入种群越界")

        print(f"[HA-NSGA3] 进行 niching+局部搜索: Gen={self.n_gen}, "
              f"Stagnation={self.stagnation_count}, Strategy={self.niche_strategy}")

        if self.niche_strategy == "ref_dirs":
            return self._refdir_niching_and_learning(pop, F, cv, is_stagnant)
        elif self.niche_strategy == "kmeans":
            return self._kmeans_niching_and_learning(pop, F, cv, is_stagnant)
        else:
            raise ValueError(f"不支持的 niche_strategy: {self.niche_strategy}")

    def _refdir_niching_and_learning(
        self,
        pop: ArrayLike,
        F: ArrayLike,
        cv: ArrayLike,
        is_stagnant: bool
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, List[int]]:
        """基于参考方向的 niche 代表选择 + 局部搜索

        选 niche 代表与改进判定均统一调用 self._make_scalarizer(self.scalarization, ...)，
        保证"选代表"和"局部搜索"目标口径一致；MO-Nelder-Mead 时同样使用标量化作为代表筛选指标
        （仅用于排序，不进入局部搜索）。
        """
        elite_id: List[int] = []
        z_star = self.z_star.copy() if self.z_star is not None else np.zeros(self.n_obj)
        z_nad = self.z_nad.copy() if self.z_nad is not None else (z_star + 1.0)

        # 关联：每个个体对应一个 niche
        niche_idx, d2 = self._associate_to_ref_dirs(F)
        cv_flat = np.asarray(cv).flatten()

        # 对每个被涉及的 niche，找其代表：基于 self.scalarization 的最优者
        H = self.ref_dirs.shape[0]
        niche_to_rep: List[Tuple[int, int, float]] = []  # (niche_k, rep_idx, score)
        for k in range(H):
            mask = (niche_idx == k)
            cand_idx = np.where(mask)[0]
            if len(cand_idx) == 0:
                continue

            scalarizer = self._make_scalarizer(
                self.scalarization,
                w_k=self.ref_dirs[k],
                z_star=z_star,
                z_nad=z_nad,
            )
            scores = np.array([scalarizer(pop[i]) for i in cand_idx])
            best_local = int(np.argmin(scores))
            rep_global = int(cand_idx[best_local])
            niche_to_rep.append((k, rep_global, float(scores[best_local])))

        if len(niche_to_rep) == 0:
            return pop, F, cv, []

        # 按 score 排序，取前 niche_num 个
        niche_to_rep.sort(key=lambda t: t[2])
        chosen = niche_to_rep[:self.niche_num]

        print(f"[HA-NSGA3]   niches涉及={len(niche_to_rep)}, 选取代表={len(chosen)} "
              f"(scalarizer={self.scalarization})")

        # 对每个代表执行局部搜索
        for niche_k, idx, score in chosen:
            w_k = self.ref_dirs[niche_k]
            # 搜索深度
            is_global_best = idx == int(np.argmin(
                [np.linalg.norm(F[i] - z_star) for i in range(len(F))]
            ))
            search_depth = 10 if (is_global_best and is_stagnant) else 3

            fes_before = self.problem.fes
            new_solution = self._local_search(
                pop[idx, :],
                score,
                maxiter=search_depth,
                w_k=w_k,
                z_star=z_star
            )
            fes_after = self.problem.fes

            new_F, new_cv = self._eval_cached(new_solution)

            # 改进判定：MO-Nelder-Mead 用支配关系；其余用 self.scalarization
            if self.method == "MO-Nelder-Mead":
                old_F_arr = np.asarray(F[idx]).flatten()
                old_cv_val = float(cv_flat[idx])
                improved = self._is_better_mo(
                    new_F, new_cv, old_F_arr, old_cv_val, w_k, z_star, z_nad
                )
                old_label = "dominance"
                old_obj = 0.0
                new_obj = 0.0
            else:
                scalarizer = self._make_scalarizer(
                    self.scalarization,
                    w_k=w_k,
                    z_star=z_star,
                    z_nad=z_nad,
                )
                old_obj = scalarizer(pop[idx, :])
                new_obj = scalarizer(new_solution)
                improved = new_obj < old_obj
                old_label = self.scalarization

            print(f"  niche={niche_k}, rep_idx={idx}{' (Global)' if is_global_best else ''}: "
                  f"{old_label} {old_obj:.6e} -> {new_obj:.6e}, "
                  f"FEs={fes_after - fes_before}, "
                  f"{'IMPROVED' if improved else '-'}")

            if improved:
                pop[idx, :] = new_solution
                F[idx, :] = new_F
                cv[idx, 0] = new_cv
            elite_id.append(idx)

        return pop, F, cv, elite_id

    def _kmeans_niching_and_learning(
        self,
        pop: ArrayLike,
        F: ArrayLike,
        cv: ArrayLike,
        is_stagnant: bool
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, List[int]]:
        """
        K-means 聚类风格的 niche 选择（消融对比用）

        在决策空间做 KMeans，对每个簇选一个代表，每个代表关联到其在目标空间
        最近的参考方向，再执行 PBI 局部搜索。
        """
        elite_id: List[int] = []
        z_star = self.z_star.copy() if self.z_star is not None else np.zeros(self.n_obj)
        z_nad = self.z_nad.copy() if self.z_nad is not None else (z_star + 1.0)
        cv_flat = np.asarray(cv).flatten()

        warnings.filterwarnings("ignore")
        labels, n_clusters = self._kmeans_clustering(pop)
        print(f"[HA-NSGA3]   kmeans 聚类: {n_clusters} 簇")

        cluster_best: List[int] = []
        for c in range(n_clusters):
            mask = (labels == c)
            cand = np.where(mask)[0]
            if len(cand) == 0:
                continue
            # 用全局 front rank 选最优
            ranks = self._nondominated_rank(F[cand], cv[cand])
            best_local = int(np.argmin(ranks))
            cluster_best.append(int(cand[best_local]))

        if len(cluster_best) > self.niche_num:
            # 用 rank 截断
            ranks_all = self._nondominated_rank(
                F[cluster_best], cv[cluster_best]
            )
            order = np.argsort(ranks_all)
            cluster_best = [cluster_best[i] for i in order[:self.niche_num]]

        # 为每个代表找最匹配的参考方向（在归一化目标空间）
        niche_idx_all, _ = self._associate_to_ref_dirs(F)

        for idx in cluster_best:
            niche_k = int(niche_idx_all[idx])
            w_k = self.ref_dirs[niche_k]
            search_depth = 10 if is_stagnant else 3

            new_solution = self._local_search(
                pop[idx, :], 0.0, maxiter=search_depth,
                w_k=w_k, z_star=z_star
            )
            new_F, new_cv = self._eval_cached(new_solution)

            if self.method == "MO-Nelder-Mead":
                old_F_arr = np.asarray(F[idx]).flatten()
                improved = self._is_better_mo(
                    new_F, new_cv, old_F_arr, float(cv_flat[idx]),
                    w_k, z_star, z_nad
                )
            else:
                scalarizer = self._make_scalarizer(
                    self.scalarization,
                    w_k=w_k,
                    z_star=z_star,
                    z_nad=z_nad,
                )
                old_obj = scalarizer(pop[idx, :])
                new_obj = scalarizer(new_solution)
                improved = new_obj < old_obj

            if improved:
                pop[idx, :] = new_solution
                F[idx, :] = new_F
                cv[idx, 0] = new_cv
            elite_id.append(idx)

        return pop, F, cv, elite_id

    # ========================================================================
    # 多目标遗传操作（替换 _inheritance / _de_current_to_best 的 rank 来源）
    # ========================================================================
    def _inheritance_mo(
        self,
        offspring_size: int,
        pop: ArrayLike,
        F: ArrayLike,
        front_rank: ArrayLike,
        d2: ArrayLike
    ) -> ArrayLike:
        """
        多目标版遗传操作：使用 NSGA-III rank（front + d2）作为隐性选择基础
        """
        offspring = np.zeros((offspring_size, self.dim))
        n_pop = len(pop)

        # 复合 rank：先 front，后 d2
        composite_order = np.lexsort((d2, front_rank))
        ranks = np.empty(n_pop, dtype=int)
        ranks[composite_order] = np.arange(n_pop)
        rank_weights = (n_pop - ranks) / n_pop  # 越前越大

        # 缓存 front 0 索引用于 DE
        front0_local = np.where(front_rank == 0)[0]
        if len(front0_local) == 0:
            front0_local = np.array([int(composite_order[0])])

        for i in range(offspring_size):
            strategy = self._rng.random()
            if strategy < 0.6:
                child = self._fitness_weighted_sbx(pop, F[:, 0:1], rank_weights)
            elif strategy < 0.85:
                child = self._fitness_weighted_multiparent(
                    pop, F[:, 0:1], rank_weights
                )
            else:
                child = self._de_current_to_best_mo(
                    pop, F, rank_weights, front0_local
                )
            offspring[i, :] = child

        offspring = np.clip(offspring, self.lb, self.ub)
        return offspring

    def _de_current_to_best_mo(
        self,
        pop: ArrayLike,
        F: ArrayLike,
        rank_weights: ArrayLike,
        front0_local: ArrayLike
    ) -> ArrayLike:
        """
        多目标版 DE/current-to-best/1：从 front 0 中随机抽取作为 x_best
        """
        if len(front0_local) > 0:
            best_idx = int(self._rng.choice(front0_local))
        else:
            best_idx = int(np.argmax(rank_weights))
        x_best = pop[best_idx]

        probs = rank_weights / rank_weights.sum()
        current_idx = int(self._rng.choice(len(pop), p=probs))
        x_current = pop[current_idx]

        r_indices = self._rng.choice(len(pop), size=2, replace=False)
        x_r1, x_r2 = pop[r_indices[0]], pop[r_indices[1]]

        F1 = 0.3 + 0.4 * self._rng.random()
        F2 = 0.3 + 0.3 * self._rng.random()
        mutant = x_current + F1 * (x_best - x_current) + F2 * (x_r1 - x_r2)

        CR = 0.8
        child = x_current.copy()
        mask = self._rng.random(self.dim) < CR
        mask[self._rng.integers(self.dim)] = True
        child[mask] = mutant[mask]
        return np.clip(child, self.lb, self.ub)

    # ========================================================================
    # 辅助：最佳个体（仅日志）
    # ========================================================================
    def _find_best_individual(
        self,
        fit: ArrayLike,
        cv: ArrayLike
    ) -> int:
        """
        返回当前 front 0 中 PBI 最小的个体索引（仅用于日志）。

        PBI 使用参考方向 ref_dirs[0]（或全局理想点方向），z_star 自动维护。
        """
        F = np.asarray(fit)
        cv_flat = np.asarray(cv).flatten()
        feasible_mask = cv_flat <= 0
        if np.any(feasible_mask):
            front0_idx = self._first_front_indices(F, cv)
            if len(front0_idx) == 0:
                return int(np.argmin(cv_flat))
            # 在 front 0 中按到 z_star 的欧氏距离选"最居中"个体
            if self.z_star is None:
                z_star = np.min(F[front0_idx], axis=0)
            else:
                z_star = self.z_star
            dists = np.linalg.norm(F[front0_idx] - z_star, axis=1)
            return int(front0_idx[np.argmin(dists)])
        else:
            return int(np.argmin(cv_flat))

    # ========================================================================
    # 监控指标：用于停滞检测
    # ========================================================================
    def _compute_indicator(self, F_front: ArrayLike) -> float:
        """
        计算当前 Pareto 近似的标量指标（越小越好）

        优先使用 problem._calc_pareto_front 的 IGD；否则用 hypervolume 的负值；
        最后回退到 front 大小的负值（鼓励更多 Pareto 解）。
        """
        if F_front is None or len(F_front) == 0:
            return float("inf")
        try:
            if hasattr(self.problem, "pareto_front"):
                pf = self.problem.pareto_front()
                if pf is not None and len(pf) > 0:
                    from pymoo.indicators.igd import IGD
                    return float(IGD(pf).do(F_front))
        except Exception:
            pass
        # 回退：用 front 的"中心到 z_star 的距离"作为代理指标
        try:
            z_star = self.z_star if self.z_star is not None else np.min(F_front, axis=0)
            return float(np.mean(np.linalg.norm(F_front - z_star, axis=1)))
        except Exception:
            return -float(len(F_front))


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == "__main__":
    import sys
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    try:
        from problem import ZDT1Problem
    except ImportError:
        print("无法导入 problem.ZDT1Problem，跳过 smoke test")
        sys.exit(0)

    print("\n========== HA-NSGA-III Smoke Test (ZDT1) ==========")

    # 三种配置对比：PBI+NM (原版), MO-NM (新支配版), TCH+NM (验证标量化扩展)
    configs = [
        {
            "name": "PBI+NM",
            "method": "Nelder-Mead",
            "scalarization": "pbi",
            "color": "tab:red",
            "marker": "o",
        },
        {
            "name": "MO-NM",
            "method": "MO-Nelder-Mead",
            "scalarization": "pbi",  # MO-NM 仅在 fallback 时使用
            "color": "tab:blue",
            "marker": "s",
        },
        {
            "name": "TCH+NM",
            "method": "Nelder-Mead",
            "scalarization": "tchebycheff",
            "color": "tab:green",
            "marker": "^",
        },
    ]

    # 解析 Pareto 前沿（ZDT1: f2 = 1 - sqrt(f1)）
    f1_arr = np.linspace(0, 1, 200)
    pf = np.column_stack([f1_arr, 1 - np.sqrt(f1_arr)])

    results: List[dict] = []
    for cfg in configs:
        print(f"\n----- 运行配置: {cfg['name']} "
              f"(method={cfg['method']}, scalarization={cfg['scalarization']}) -----")
        problem = ZDT1Problem(n_var=10)
        algorithm = HA_NSGA3(
            method=cfg["method"],
            pop_size=50,
            niche_num=4,
            mutation_rate=0.2,
            activate_method=True,
            niche_strategy="ref_dirs",
            pbi_theta=5.0,
            scalarization=cfg["scalarization"],
            seed=42,
        )

        fes_before = problem.fes
        res = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", 30),
            seed=42,
            verbose=False,
            save_history=False,
            copy_algorithm=False,
        )
        fes_after = problem.fes

        F_final = np.asarray(res.pop.get("F"))
        nds_final = NonDominatedSorting()
        fronts_final = nds_final.do(F_final)
        F_front0 = F_final[fronts_final[0]]

        try:
            from pymoo.indicators.igd import IGD
            igd_final = float(IGD(pf).do(F_final))
            igd_front0 = float(IGD(pf).do(F_front0))
        except Exception as e:
            print(f"[Smoke Test] IGD computation failed: {e}")
            igd_final = float("inf")
            igd_front0 = float("inf")

        print(f"[Smoke Test] {cfg['name']}: pop={F_final.shape}, "
              f"front0={len(F_front0)}, FEs={fes_after - fes_before}, "
              f"IGD(final)={igd_final:.4e}, IGD(front0)={igd_front0:.4e}")

        results.append({
            "name": cfg["name"],
            "color": cfg["color"],
            "marker": cfg["marker"],
            "F_final": F_final,
            "F_front0": F_front0,
            "igd_front0": igd_front0,
            "igd_final": igd_final,
            "fes": fes_after - fes_before,
        })

    # 同一张图三色对比
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot(pf[:, 0], pf[:, 1], "k--", lw=1.0, label="Analytical PF")
    for r in results:
        ax.scatter(
            r["F_front0"][:, 0],
            r["F_front0"][:, 1],
            s=42,
            c=r["color"],
            marker=r["marker"],
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            label=f"{r['name']} (IGD={r['igd_front0']:.3e})",
        )
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_title("HA-NSGA-III Variants on ZDT1 (30 gen, pop=50)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(os.path.dirname(__file__), "ha_nsga3_zdt1_smoke.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n[Smoke Test] image saved: {out_path}")

    # 汇总
    print("\n===== Smoke Test Summary =====")
    print(f"{'Config':<10} {'front0':<8} {'FEs':<8} {'IGD(front0)':<14} {'IGD(final)':<14}")
    for r in results:
        print(
            f"{r['name']:<10} {len(r['F_front0']):<8d} {r['fes']:<8d} "
            f"{r['igd_front0']:<14.4e} {r['igd_final']:<14.4e}"
        )
