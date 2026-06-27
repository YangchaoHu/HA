"""
active_learning_bo.py
---------------------

双层主动学习 / 贝叶斯优化框架（EGO - Efficient Global Optimization）。

外层循环（Outer Loop）：
    真实昂贵仿真 → 追加到训练集 → 重训 Kriging → 收敛判定

内层循环（Inner Loop）：
    HA（ha_Nelder_Mead.HA）在 Kriging 代理表面上最大化采集函数，
    每次评估只调用 Kriging.predict（毫秒级），不触碰真实仿真。

采集函数：
    EI  (Expected Improvement)   -- 用于找全局最优（默认）
    LCB (Lower Confidence Bound) -- 更激进的探索

典型用法::

    from active_learning_bo import BayesianOptimizer

    def my_sim(x):          # 真实昂贵仿真，x: ndarray shape (d,) -> float
        ...

    bo = BayesianOptimizer(
        sim_func=my_sim,
        xl=[0.4, 0.01, 0.01],
        xu=[1.6, 0.08, 0.08],
        n_init=15,
        max_iter=60,
        acquisition="ei",
    )
    result = bo.run()
    print(result["best_x"], result["best_f"])
"""

from __future__ import annotations

import csv
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# --------------------------------------------------------------------------
# 可选依赖：scipy.stats.qmc（Python ≥ 3.8 / scipy ≥ 1.7）
# --------------------------------------------------------------------------
try:
    from scipy.stats.qmc import LatinHypercube as _LHS
    _HAS_QMC = True
except ImportError:
    _HAS_QMC = False

from scipy.stats import norm as _norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

# HA 及配套工具
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ha_Nelder_Mead import HA as _HA                       # noqa: E402
from experiment_runner import DataCollectorCallback          # noqa: E402


# ============================================================================
# 工具函数
# ============================================================================

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(tag: str, msg: str) -> None:
    print(f"[{_ts()}][{tag}] {msg}", flush=True)


def _lhs_sample(xl: NDArray, xu: NDArray, n: int, seed: int) -> NDArray:
    """生成拉丁超立方样本，兼容 scipy 旧版本。"""
    d = len(xl)
    if _HAS_QMC:
        sampler = _LHS(d=d, seed=seed)
        unit = sampler.random(n=n)          # (n, d) in [0, 1]
    else:
        rng = np.random.default_rng(seed)
        # 朴素 LHS：每维等间隔随机置换
        unit = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            unit[:, j] = (perm + rng.random(n)) / n
    return xl + unit * (xu - xl)


# ============================================================================
# AcquisitionProblem：pymoo Problem，把 EI / LCB 包装为最小化目标
# ============================================================================

class AcquisitionProblem(Problem):
    """
    以已训练的 Kriging（GaussianProcessRegressor）为代理，
    将采集函数取负后包装为 pymoo 单目标最小化问题，
    供 HA 在内层循环快速寻优。

    Parameters
    ----------
    gp         : 已拟合的 GaussianProcessRegressor
    xl, xu     : 搜索边界（原始空间）
    f_best     : 当前已知最优目标值（EI 需要）
    acquisition: "ei" 或 "lcb"
    kappa      : LCB 的探索系数 κ（默认 2.0）
    xi         : EI 的改进阈值 ξ（默认 0.01，越大越倾向探索）
    scaler_X   : 若 Kriging 是在标准化空间训练的，传入 MinMaxScaler；
                 否则传 None（直接用原始坐标预测）
    """

    def __init__(
        self,
        gp: GaussianProcessRegressor,
        xl: NDArray,
        xu: NDArray,
        f_best: float,
        acquisition: str = "ei",
        kappa: float = 2.0,
        xi: float = 0.01,
        scaler_X=None,
    ) -> None:
        # 与 problem.py、HA(_infill/_local_search) 一致：供日志与局部搜索计数
        self.fes: int = 0
        super().__init__(n_var=len(xl), n_obj=1, xl=xl.copy(), xu=xu.copy())
        self._gp = gp
        self._f_best = f_best
        self._acq = acquisition.lower()
        self._kappa = kappa
        self._xi = xi
        self._scaler_X = scaler_X

    def _evaluate(self, X: NDArray, out: Dict, *args, **kwargs) -> None:
        X = np.atleast_2d(X).astype(float)
        self.fes += int(X.shape[0])

        X_input = (
            self._scaler_X.transform(X) if self._scaler_X is not None else X
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, std = self._gp.predict(X_input, return_std=True)
        mu = mu.flatten()
        std = np.maximum(std.flatten(), 1e-9)

        if self._acq == "ei":
            acq_vals = _ei(mu, std, self._f_best, self._xi)
        else:
            acq_vals = _lcb(mu, std, self._kappa)

        # 最小化 -acq（HA 是最小化器）
        out["F"] = (-acq_vals).reshape(-1, 1)


def _ei(mu: NDArray, std: NDArray, f_best: float, xi: float = 0.01) -> NDArray:
    """Expected Improvement（最小化问题版本）。"""
    z = (f_best - mu - xi) / std
    ei = (f_best - mu - xi) * _norm.cdf(z) + std * _norm.pdf(z)
    ei[std < 1e-10] = 0.0
    return np.maximum(ei, 0.0)


def _lcb(mu: NDArray, std: NDArray, kappa: float = 2.0) -> NDArray:
    """Lower Confidence Bound（取负后变成"越小越好的采集分数"）。"""
    # 内部对外暴露的是"值越大越值得采样"，与 EI 保持相同方向
    # 调用方会对返回值取负
    return -(mu - kappa * std)   # LCB 本身越小越好，这里先加负号让方向统一


# 注：_lcb 返回的是 -LCB，与 _ei 返回正值方向一致；
#     AcquisitionProblem._evaluate 再整体取负 → 最终目标 = LCB（最小化）


# ============================================================================
# BayesianOptimizer
# ============================================================================

class BayesianOptimizer:
    """
    双层主动学习贝叶斯优化器。

    外层循环：真实仿真 → 训练 Kriging → 收敛判定
    内层循环：HA 在 Kriging 代理面上最大化采集函数 → 候选点 x*

    Parameters
    ----------
    sim_func    : 真实仿真函数，签名 (x: ndarray shape (d,)) -> float
    xl, xu      : 搜索边界（list 或 ndarray，长度 d）
    n_init      : LHS 初始采样点数（默认 10）
    max_iter    : 外层最大迭代次数（每次调用一次真实仿真）
    acquisition : "ei"（默认）或 "lcb"
    kappa       : LCB 的 κ（默认 2.0）
    xi          : EI 的 ξ（默认 0.01）
    tol         : 连续 patience 轮最优值改进 < tol 则停止
    patience    : 容忍轮数（默认 5）
    ha_pop_size : 内层 HA 种群大小（默认 30）
    ha_n_gen    : 内层 HA 迭代代数（默认 20）
    ha_method   : HA 局部搜索方法（默认 "L-BFGS-B"，速度快）
    seed        : 随机种子
    output_dir  : 结果 CSV 输出目录（默认当前目录）
    verbose     : 是否打印每轮信息
    """

    def __init__(
        self,
        sim_func: Callable[[NDArray], float],
        xl,
        xu,
        n_init: int = 10,
        max_iter: int = 50,
        acquisition: str = "ei",
        kappa: float = 2.0,
        xi: float = 0.01,
        tol: float = 1e-6,
        patience: int = 5,
        ha_pop_size: int = 30,
        ha_n_gen: int = 20,
        ha_method: str = "L-BFGS-B",
        seed: int = 42,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> None:
        self.sim_func = sim_func
        self.xl = np.asarray(xl, dtype=float)
        self.xu = np.asarray(xu, dtype=float)
        self.n_init = n_init
        self.max_iter = max_iter
        self.acquisition = acquisition.lower()
        self.kappa = kappa
        self.xi = xi
        self.tol = tol
        self.patience = patience
        self.ha_pop_size = ha_pop_size
        self.ha_n_gen = ha_n_gen
        self.ha_method = ha_method
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.verbose = verbose

        # 运行时状态
        self._X_train: List[NDArray] = []
        self._y_train: List[float] = []
        self._gp: Optional[GaussianProcessRegressor] = None
        self._history: List[Dict[str, Any]] = []   # 每轮记录

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        执行完整贝叶斯优化流程。

        Returns
        -------
        dict with keys:
            best_x      : ndarray, 最优解
            best_f      : float,   最优目标值
            n_sim_calls : int,     总真实仿真调用次数
            history     : list,    每轮详细记录
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "bo_trace.csv"

        _log("BO", f"启动  dim={len(self.xl)}  n_init={self.n_init}  "
                   f"max_iter={self.max_iter}  acq={self.acquisition}")

        # ---- 1. 初始 LHS 采样 ----
        _log("BO", f"初始 LHS 采样 {self.n_init} 个点 ...")
        X_init = _lhs_sample(self.xl, self.xu, self.n_init, self.seed)
        for i, x in enumerate(X_init):
            f = self._call_sim(x, tag=f"init-{i+1}/{self.n_init}")
            self._X_train.append(x)
            self._y_train.append(f)

        best_f = min(self._y_train)
        best_x = self._X_train[int(np.argmin(self._y_train))].copy()
        _log("BO", f"初始化完毕  best_f={best_f:.6e}")

        self._append_csv(csv_path, iter_=0, x=best_x, f_real=best_f,
                         ei_max=float("nan"), note="init")

        # ---- 2. 外层迭代 ----
        no_improve = 0
        for it in range(1, self.max_iter + 1):
            t0 = time.perf_counter()

            # 2a. 重训 Kriging
            gp = self._fit_kriging()
            self._gp = gp

            # 2b. 内层 HA 最大化采集函数 → 候选点 x*
            x_star, acq_max = self._maximize_acquisition(gp, best_f, seed=self.seed + it)

            # 2c. 真实仿真回评
            f_star = self._call_sim(x_star, tag=f"iter-{it}")

            # 2d. 追加训练集
            self._X_train.append(x_star)
            self._y_train.append(f_star)

            # 2e. 更新最优
            improved = f_star < best_f - self.tol
            if improved:
                best_f = f_star
                best_x = x_star.copy()
                no_improve = 0
            else:
                no_improve += 1

            elapsed = time.perf_counter() - t0
            if self.verbose:
                _log("BO",
                     f"[{it:3d}/{self.max_iter}] f*={f_star:.6e}  "
                     f"best_f={best_f:.6e}  acq_max={acq_max:.4e}  "
                     f"no_improve={no_improve}/{self.patience}  "
                     f"耗时={elapsed:.1f}s")

            self._append_csv(csv_path, iter_=it, x=x_star, f_real=f_star,
                             ei_max=acq_max, note="")
            self._history.append({
                "iter": it, "x": x_star.tolist(), "f_real": f_star,
                "best_f": best_f, "acq_max": acq_max,
            })

            # 2f. 收敛判定
            if no_improve >= self.patience:
                _log("BO", f"收敛：连续 {self.patience} 轮无改进，提前停止")
                break

        n_sim = len(self._y_train)
        _log("BO", f"完成  best_f={best_f:.6e}  best_x={np.round(best_x, 5).tolist()}"
                   f"  总仿真次数={n_sim}")
        return {
            "best_x": best_x,
            "best_f": best_f,
            "n_sim_calls": n_sim,
            "history": self._history,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _call_sim(self, x: NDArray, tag: str = "") -> float:
        """调用真实仿真，带计时日志。"""
        t0 = time.perf_counter()
        f = float(self.sim_func(x))
        elapsed = time.perf_counter() - t0
        if self.verbose:
            _log("SIM", f"{tag}  x={np.round(x, 5).tolist()}  f={f:.6e}  耗时={elapsed:.2f}s")
        return f

    def _fit_kriging(self) -> GaussianProcessRegressor:
        """
        用当前训练集拟合 Kriging（GPR Matérn ν=2.5 + WhiteKernel）。
        核函数设置与 run_surrogate_ha_eval.py 保持一致。
        """
        X = np.array(self._X_train, dtype=float)
        y = np.array(self._y_train, dtype=float)

        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y)
        return gp

    def _maximize_acquisition(
        self,
        gp: GaussianProcessRegressor,
        f_best: float,
        seed: int,
    ) -> Tuple[NDArray, float]:
        """
        使用 HA 在 Kriging 代理面上最大化采集函数，返回 (x*, acq_value)。

        HA 是最小化器，AcquisitionProblem 将采集函数取负，
        因此 HA 找到的最小值对应采集函数的最大值。
        """
        acq_prob = AcquisitionProblem(
            gp=gp,
            xl=self.xl,
            xu=self.xu,
            f_best=f_best,
            acquisition=self.acquisition,
            kappa=self.kappa,
            xi=self.xi,
            scaler_X=None,      # 直接在原始坐标上预测
        )

        # 生成内层 HA 初始种群
        initial_pop = _lhs_sample(self.xl, self.xu, self.ha_pop_size, seed)

        # 构建并运行 HA
        callback = DataCollectorCallback(skip_first=False)
        algorithm = _HA(
            method=self.ha_method,
            pop_size=self.ha_pop_size,
            niche_num=min(3, self.ha_pop_size // 5),
            mutation_rate=1.0,
            inherit_rate=1.0,
            activate_method=True,
            cluster_method="kmeans",
            X=initial_pop,
            seed=seed,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pymoo_minimize(
                acq_prob,
                algorithm,
                termination=get_termination("n_gen", self.ha_n_gen),
                seed=seed,
                callback=callback,
                verbose=False,
            )

        # 提取最优候选点
        if result is not None and result.X is not None:
            x_star = np.asarray(result.X).flatten()
            acq_neg = float(result.F.flatten()[0]) if result.F is not None else float("nan")
        elif callback.data:
            best_gen = min(callback.data, key=lambda g: float(g.best_F))
            x_star = np.asarray(best_gen.best_X).flatten()
            acq_neg = float(best_gen.best_F)
        else:
            # 兜底：随机取一个点
            x_star = _lhs_sample(self.xl, self.xu, 1, seed)[0]
            acq_neg = float("nan")

        x_star = np.clip(x_star, self.xl, self.xu)
        acq_max = -acq_neg if not np.isnan(acq_neg) else float("nan")

        # 去重检查：若 x* 与历史点太近，在邻域内随机扰动
        x_star = self._deduplicate(x_star, seed=seed)

        return x_star, acq_max

    def _deduplicate(self, x: NDArray, seed: int, min_dist_ratio: float = 1e-3) -> NDArray:
        """
        若候选点 x 与训练集中任意点距离 < 域对角线的 min_dist_ratio，
        则在 x 周围施加微小扰动，避免重复评估。
        """
        if not self._X_train:
            return x
        X_exist = np.array(self._X_train)
        diag = float(np.linalg.norm(self.xu - self.xl))
        dists = np.linalg.norm(X_exist - x, axis=1)
        if dists.min() < diag * min_dist_ratio:
            rng = np.random.default_rng(seed)
            noise = rng.uniform(-1, 1, size=len(x)) * diag * min_dist_ratio * 2
            x = np.clip(x + noise, self.xl, self.xu)
        return x

    @staticmethod
    def _append_csv(
        path: Path,
        iter_: int,
        x: NDArray,
        f_real: float,
        ei_max: float,
        note: str,
    ) -> None:
        """追加一行到 bo_trace.csv。"""
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                x_cols = [f"x{i+1}" for i in range(len(x))]
                writer.writerow(["iter"] + x_cols + ["f_real", "acq_max", "note"])
            writer.writerow([iter_] + list(x) + [f_real, ei_max, note])
