"""
problem.py
----------

基准测试函数 f2(x)、f3(x) 和 f4(x) 的 `pymoo` 问题定义，用于配合 `HA` 算法进行优化。

f2(x) 公式：
    f2(x) = sum_{i=1}^D sin(x_i) * sin^{2m}(i * x_i^2 / pi)

f3(x) 公式：
    f3(x) = g(x1, x2) + g(x2, x3) + ... + g(xD-1, xD) + g(xD, x1)
    其中 g(x, y) = 0.5 + (sin²(√(x² + y²)) - 0.5) / (1 + 0.001(x² + y²))²

f4(x) 公式（混合函数）：
    f4(x) = 0.3 * f41(x) + 0.3 * f42(x) + 0.4 * f43(x)
    
    f41(x) = 418.9829 * D - sum_{i=1}^D g(z_i)  （修正的 Schwefel 函数）
    f42(x) = sum_{i=1}^D (x_i^2 - 10*cos(2*pi*x_i) + 10)  （Rastrigin 函数）
    f43(x) = sum_{i=1}^D (10^6)^{(i-1)/(D-1)} * x_i^2  （椭球函数）

其中：
    - D 为决策变量维度（n_var）
    - m 为 f2 的可调参数（正整数）
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pymoo.core.problem import Problem


ArrayLike = NDArray[np.floating]


class F2Problem(Problem):
    """
    单目标基准函数 f2 (Michalewicz) 的 `pymoo` 问题定义。

    目标函数：
        f2(x) = -sum_{i=1}^D sin(x_i) * sin^{2m}(i * x_i^2 / pi)
    
    注意：加负号使其成为标准 Michalewicz 函数，全局最小值约为 -9.66 (D=10, m=10)

    参数
    ----
    n_var : int, default 10
        决策变量维度 D。
    m : int, default 5
        指数参数 m（通常取正整数）。
    xl : float or array-like, default -10.0
        每个变量的下界，既可以是标量也可以是长度为 n_var 的数组。
    xu : float or array-like, default 10.0
        每个变量的上界，既可以是标量也可以是长度为 n_var 的数组。
    """

    def __init__(
        self,
        n_var: int = 10,
        m: int = 10,
        xl: float | ArrayLike = 0.0,
        xu: float | ArrayLike = np.pi,
    ) -> None:
        self.m: int = int(m)
        # 记录函数评估次数（function evaluations）
        # 供 HA 算法中的日志使用，如 self.problem.fes
        self.fes: int = 0

        # 调用父类构造函数：
        # - 单目标优化 (n_obj=1)
        # - 无不等式/等式约束 (n_ieq_constr=0, n_eq_constr=0)
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(
        self,
        X: ArrayLike,
        out: dict,
        *args,
        **kwargs,
    ) -> None:
        """
        批量评估目标函数值。

        参数
        ----
        X : ndarray, shape (N, D)
            N 个个体，每个个体为 D 维向量。
        out : dict
            结果字典，至少包含键 "F"。
        """

        X = np.atleast_2d(X).astype(float)  # 保证二维数组 (N, D)
        n_samples, n_var = X.shape

        # 更新函数评估次数：每一行视为一次评估
        self.fes += int(n_samples)

        # i = 1..D，形状 (D,)
        i = np.arange(1, n_var + 1, dtype=float)

        # term1 = sin(x_i)
        term1 = np.sin(X)  # (N, D)

        # term2 = sin^{2m}(i * x_i^2 / pi)
        # 先计算内部：i * x_i^2 / pi，利用广播：
        #   X**2         -> (N, D)
        #   i[None, :]  -> (1, D)
        inner = i[None, :] * (X ** 2) / np.pi  # (N, D)
        term2 = np.sin(inner) ** (2 * self.m)  # (N, D)

        # 按维度求和：-sum_{i=1}^D term1 * term2 (Michalewicz 函数需要负号)
        f = -np.sum(term1 * term2, axis=1)  # (N,)

        # 与 HA 中 evaluate_fitness_cv_batch 的接口兼容：
        # 该函数会对 out["F"] 调用 reshape(-1, 1)，
        # 因此这里可以返回一维数组或二维列向量。
        out["F"] = f


class F3Problem(Problem):
    """
    单目标基准函数 f3 的 `pymoo` 问题定义。

    目标函数：
        f3(x) = g(x1, x2) + g(x2, x3) + ... + g(xD-1, xD) + g(xD, x1)
        
        其中 g(x, y) = 0.5 + (sin²(√(x² + y²)) - 0.5) / (1 + 0.001(x² + y²))²

    参数
    ----
    n_var : int, default 10
        决策变量维度 D。
    xl : float or array-like, default -100.0
        每个变量的下界，既可以是标量也可以是长度为 n_var 的数组。
    xu : float or array-like, default 100.0
        每个变量的上界，既可以是标量也可以是长度为 n_var 的数组。

    注意
    ----
    定义域：x ∈ [-100, 100]^D
    范围：D=10 时 [0, 9.98]，D=30 时 [0, 29.93]
    """

    def __init__(
        self,
        n_var: int = 10,
        xl: float | ArrayLike = -100.0,
        xu: float | ArrayLike = 100.0,
    ) -> None:
        # 记录函数评估次数（function evaluations）
        # 供 HA 算法中的日志使用，如 self.problem.fes
        self.fes: int = 0

        # 调用父类构造函数：
        # - 单目标优化 (n_obj=1)
        # - 无不等式/等式约束 (n_ieq_constr=0, n_eq_constr=0)
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=xl,
            xu=xu,
        )

    @staticmethod
    def _g(x: ArrayLike, y: ArrayLike) -> ArrayLike:
        """
        辅助函数 g(x, y)。
        
        公式：
            g(x, y) = 0.5 + (sin²(√(x² + y²)) - 0.5) / (1 + 0.001(x² + y²))²
        
        参数
        ----
        x : ndarray
            第一个输入变量。
        y : ndarray
            第二个输入变量。
            
        返回
        ----
        ndarray
            g(x, y) 的计算结果。
        """
        # 计算 x² + y²
        x2_plus_y2 = x ** 2 + y ** 2
        
        # 计算 √(x² + y²)
        sqrt_x2_y2 = np.sqrt(x2_plus_y2)
        
        # 计算 sin²(√(x² + y²))
        sin_squared = np.sin(sqrt_x2_y2) ** 2
        
        # 计算分母：(1 + 0.001(x² + y²))²
        denominator = (1 + 0.001 * x2_plus_y2) ** 2
        
        # 计算 g(x, y) = 0.5 + (sin²(√(x² + y²)) - 0.5) / (1 + 0.001(x² + y²))²
        g_val = 0.5 + (sin_squared - 0.5) / denominator
        
        return g_val

    def _evaluate(
        self,
        X: ArrayLike,
        out: dict,
        *args,
        **kwargs,
    ) -> None:
        """
        批量评估目标函数值。

        参数
        ----
        X : ndarray, shape (N, D)
            N 个个体，每个个体为 D 维向量。
        out : dict
            结果字典，至少包含键 "F"。
        """

        X = np.atleast_2d(X).astype(float)  # 保证二维数组 (N, D)
        n_samples, n_var = X.shape

        # 更新函数评估次数：每一行视为一次评估
        self.fes += int(n_samples)

        # 初始化结果数组
        f = np.zeros(n_samples)

        # 计算相邻配对项：g(x1, x2), g(x2, x3), ..., g(xD-1, xD)
        for i in range(n_var - 1):
            f += self._g(X[:, i], X[:, i + 1])

        # 计算循环项：g(xD, x1)
        f += self._g(X[:, -1], X[:, 0])

        # 与 HA 中 evaluate_fitness_cv_batch 的接口兼容：
        # 该函数会对 out["F"] 调用 reshape(-1, 1)，
        # 因此这里可以返回一维数组或二维列向量。
        out["F"] = f


class F4Problem(Problem):
    """
    单目标混合基准函数 f4 的 `pymoo` 问题定义。

    目标函数：
        f4(x) = 0.3 * f41(x) + 0.3 * f42(x) + 0.4 * f43(x)
        
        其中：
        - f41(x) = 418.9829 * D - sum_{i=1}^D g(z_i)  （修正的 Schwefel 函数）
        - f42(x) = sum_{i=1}^D (x_i^2 - 10*cos(2*pi*x_i) + 10)  （Rastrigin 函数）
        - f43(x) = sum_{i=1}^D (10^6)^{(i-1)/(D-1)} * x_i^2  （椭球函数）
        
        z_i = x_i + 4.209687462275036 × 10^2
        
        g(z_i) 分段函数：
        - |z_i| <= 500: z_i * sin(sqrt(|z_i|))
        - z_i > 500: (500 - mod(z_i, 500)) * sin(sqrt(|500 - mod(z_i, 500)|)) 
                     - (z_i - 500)^2 / (10000*D)
        - z_i < -500: (mod(|z_i|, 500) - 500) * sin(sqrt(|mod(|z_i|, 500) - 500|)) 
                      - (z_i + 500)^2 / (10000*D)

    参数
    ----
    n_var : int, default 10
        决策变量维度 D。
    xl : float or array-like, default -100.0
        每个变量的下界。
    xu : float or array-like, default 100.0
        每个变量的上界。

    注意
    ----
    定义域：x ∈ [-100, 100]^D
    范围：D=10 时 [0, 5.54×10^8]，D=30 时 [0, 1.22×10^9]
    混合权重：0.3 (f41), 0.3 (f42), 0.4 (f43)
    """

    # Schwefel 函数的偏移常数
    SHIFT_CONSTANT = 4.209687462275036e2

    def __init__(
        self,
        n_var: int = 10,
        xl: float | ArrayLike = -100.0,
        xu: float | ArrayLike = 100.0,
    ) -> None:
        # 记录函数评估次数（function evaluations）
        # 供 HA 算法中的日志使用，如 self.problem.fes
        self.fes: int = 0

        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=xl,
            xu=xu,
        )

    def _g_schwefel(self, z: ArrayLike, D: int) -> ArrayLike:
        """
        修正的 Schwefel 函数中的 g(z_i) 分段函数。
        
        参数
        ----
        z : ndarray, shape (N, D)
            偏移后的变量 z_i = x_i + 420.9687462275036
        D : int
            维度
            
        返回
        ----
        ndarray, shape (N, D)
            每个 z_i 对应的 g(z_i) 值
        """
        g = np.zeros_like(z)
        
        # 情况 1: |z_i| <= 500
        mask1 = np.abs(z) <= 500
        g[mask1] = z[mask1] * np.sin(np.sqrt(np.abs(z[mask1])))
        
        # 情况 2: z_i > 500
        mask2 = z > 500
        if np.any(mask2):
            z_mod = np.mod(z[mask2], 500)
            term1 = (500 - z_mod) * np.sin(np.sqrt(np.abs(500 - z_mod)))
            term2 = (z[mask2] - 500) ** 2 / (10000 * D)
            g[mask2] = term1 - term2
        
        # 情况 3: z_i < -500
        mask3 = z < -500
        if np.any(mask3):
            z_abs_mod = np.mod(np.abs(z[mask3]), 500)
            term1 = (z_abs_mod - 500) * np.sin(np.sqrt(np.abs(z_abs_mod - 500)))
            term2 = (z[mask3] + 500) ** 2 / (10000 * D)
            g[mask3] = term1 - term2
        
        return g

    def _f41(self, X: ArrayLike) -> ArrayLike:
        """
        修正的 Schwefel 函数 f41(x)。
        
        公式：f41(x) = 418.9829 * D - sum_{i=1}^D g(z_i)
        其中 z_i = x_i + 4.209687462275036 × 10^2
        """
        n_samples, D = X.shape
        
        # 计算偏移后的 z
        z = X + self.SHIFT_CONSTANT
        
        # 计算 g(z_i)
        g_values = self._g_schwefel(z, D)
        
        # f41 = 418.9829 * D - sum(g(z_i))
        f41 = 418.9829 * D - np.sum(g_values, axis=1)
        
        return f41

    def _f42(self, X: ArrayLike) -> ArrayLike:
        """
        Rastrigin 函数 f42(x)。
        
        公式：f42(x) = sum_{i=1}^D (x_i^2 - 10*cos(2*pi*x_i) + 10)
        """
        term = X ** 2 - 10 * np.cos(2 * np.pi * X) + 10
        return np.sum(term, axis=1)

    def _f43(self, X: ArrayLike) -> ArrayLike:
        """
        椭球函数 f43(x)。
        
        公式：f43(x) = sum_{i=1}^D (10^6)^{(i-1)/(D-1)} * x_i^2
        """
        n_samples, D = X.shape
        
        # 计算系数：(10^6)^{(i-1)/(D-1)}，i = 1, 2, ..., D
        if D == 1:
            # 特殊情况：D=1 时避免除以零
            coeffs = np.array([1.0])
        else:
            i = np.arange(1, D + 1, dtype=float)
            exponents = (i - 1) / (D - 1)
            coeffs = (1e6) ** exponents  # shape (D,)
        
        # 计算加权平方和
        weighted_x2 = coeffs[None, :] * (X ** 2)  # (N, D)
        return np.sum(weighted_x2, axis=1)

    def _evaluate(
        self,
        X: ArrayLike,
        out: dict,
        *args,
        **kwargs,
    ) -> None:
        """
        批量评估目标函数值。

        参数
        ----
        X : ndarray, shape (N, D)
            N 个个体，每个个体为 D 维向量。
        out : dict
            结果字典，至少包含键 "F"。
        """
        X = np.atleast_2d(X).astype(float)
        n_samples, _ = X.shape

        # 更新函数评估次数：每一行视为一次评估
        self.fes += int(n_samples)

        # 计算三个子函数
        f41 = self._f41(X)
        f42 = self._f42(X)
        f43 = self._f43(X)

        # 混合：f4 = 0.3 * f41 + 0.3 * f42 + 0.4 * f43
        f = 0.3 * f41 + 0.3 * f42 + 0.4 * f43

        out["F"] = f


class AckleyProblem(Problem):
    """
    Ackley 函数的 `pymoo` 问题定义（经典多峰函数）。

    标准形式（D 维）：
        f(x) = -20 * exp( -0.2 * sqrt( 1/D * sum_{i=1}^D x_i^2 ) )
               - exp( 1/D * sum_{i=1}^D cos( 2*pi*x_i ) )
               + 20 + e

    常用定义域：x ∈ [-32.768, 32.768]^D
    全局最优：f(x*) = 0, x* = 0^D
    """

    def __init__(
        self,
        n_var: int = 10,
        xl: float | ArrayLike = -32.768,
        xu: float | ArrayLike = 32.768,
    ) -> None:
        # 记录函数评估次数
        self.fes: int = 0

        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(
        self,
        X: ArrayLike,
        out: dict,
        *args,
        **kwargs,
    ) -> None:
        """
        批量评估 Ackley 目标函数值。
        """
        X = np.atleast_2d(X).astype(float)
        n_samples, D = X.shape

        # 更新函数评估次数
        self.fes += int(n_samples)

        # 1/D * sum(x_i^2)
        sq_term = np.sum(X ** 2, axis=1) / D
        # 1/D * sum(cos(2*pi*x_i))
        cos_term = np.sum(np.cos(2 * np.pi * X), axis=1) / D

        f = -20.0 * np.exp(-0.2 * np.sqrt(sq_term)) - np.exp(cos_term) + 20.0 + np.e
        out["F"] = f


class GriewankProblem(Problem):
    """
    Griewank 函数的 `pymoo` 问题定义（经典多峰函数）。

    标准形式（D 维）：
        f(x) = 1 + 1/4000 * sum_{i=1}^D x_i^2 - prod_{i=1}^D cos( x_i / sqrt(i) )

    常用定义域：x ∈ [-600, 600]^D
    全局最优：f(x*) = 0, x* = 0^D
    """

    def __init__(
        self,
        n_var: int = 10,
        xl: float | ArrayLike = -600.0,
        xu: float | ArrayLike = 600.0,
    ) -> None:
        # 记录函数评估次数
        self.fes: int = 0

        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(
        self,
        X: ArrayLike,
        out: dict,
        *args,
        **kwargs,
    ) -> None:
        """
        批量评估 Griewank 目标函数值。
        """
        X = np.atleast_2d(X).astype(float)
        n_samples, D = X.shape

        # 更新函数评估次数
        self.fes += int(n_samples)

        # sum(x_i^2) / 4000
        sum_term = np.sum(X ** 2, axis=1) / 4000.0

        # prod(cos(x_i / sqrt(i)))
        i = np.arange(1, D + 1, dtype=float)
        cos_term = np.prod(np.cos(X / np.sqrt(i[None, :])), axis=1)

        f = 1.0 + sum_term - cos_term
        out["F"] = f


class RastriginProblem(Problem):
    """
    Rastrigin 函数的 `pymoo` 问题定义（经典多峰函数）。

    标准形式（D 维）：
        f(x) = sum_{i=1}^D (x_i^2 - 10*cos(2*pi*x_i) + 10)

    常用定义域：x ∈ [-5.12, 5.12]^D
    全局最优：f(x*) = 0, x* = 0^D
    """

    def __init__(
        self,
        n_var: int = 10,
        xl: float | ArrayLike = -5.12,
        xu: float | ArrayLike = 5.12,
    ) -> None:
        # 记录函数评估次数
        self.fes: int = 0

        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=0,
            n_eq_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(
        self,
        X: ArrayLike,
        out: dict,
        *args,
        **kwargs,
    ) -> None:
        """
        批量评估 Rastrigin 目标函数值。
        """
        X = np.atleast_2d(X).astype(float)
        n_samples, D = X.shape

        # 更新函数评估次数
        self.fes += int(n_samples)

        # 计算每一项：x_i^2 - 10*cos(2*pi*x_i) + 10
        term = X ** 2 - 10 * np.cos(2 * np.pi * X) + 10

        # 按维度求和
        f = np.sum(term, axis=1)

        out["F"] = f


__all__ = [
    "F2Problem",
    "F3Problem",
    "F4Problem",
    "AckleyProblem",
    "GriewankProblem",
    "RastriginProblem",
]


