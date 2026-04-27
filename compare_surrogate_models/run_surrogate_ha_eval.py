"""
run_surrogate_ha_eval.py
------------------------

对 QuickSimu1 / QuickSimu2 各 seed 各采样阶段 JSON，同时对 volume 与 max_disp
分别建立代理模型（Kriging / RBF / Polynomial / KAN / KAN-GP），

然后按照 problem.py 的问题定义构造代理优化问题：
    目标 F = volume（越小越好）
    约束 G = abs(max_disp) - disp_limit <= 0

用 HA_Nelder_Mead(method='gp') 优化代理，对最优个体做真实仿真回评，
并按 (problem, model_type, sample_count) 跨 seed 均值导出 CSV。

用法示例（全量）:
    python compare_surrogate_models/run_surrogate_ha_eval.py

冒烟测试（单问题、单采样数、快速验证）:
    python compare_surrogate_models/run_surrogate_ha_eval.py \
        --problems QuickSimu1 --counts 500 --pop-size 10 --n-gen 5

并行代理阶段（真实仿真仍串行）:
    python compare_surrogate_models/run_surrogate_ha_eval.py --jobs 4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 允许在仓库根目录或 compare_surrogate_models 目录下直接运行
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pymoo.core.problem import Problem
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import RBFInterpolator

from experiment_runner import run_ha_nelder_mead_method

# --------------------------------------------------------------------------
# 可选依赖：torch / gpytorch / kan_gp
# 若环境缺少这些包，KAN 与 KAN-GP 两类模型在 build_surrogate() 中会抛出清晰
# RuntimeError，由 run_one() 的 [SKIP] 分支接管，其他模型继续正常运行。
# --------------------------------------------------------------------------
_KAN_GP_DIR = Path(__file__).resolve().parent / "kan_gp_minimal"
if str(_KAN_GP_DIR) not in sys.path:
    sys.path.insert(0, str(_KAN_GP_DIR))

try:
    import torch
    import torch.nn as nn
    import gpytorch
    from kan_gp import KANFeatureExtractor, create_pimfgpkan

    _TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TORCH_AVAILABLE = False

# --------------------------------------------------------------------------
# KAN / KAN-GP 超参数常量
# --------------------------------------------------------------------------
KAN_HIDDEN_DIM: int = 16
KAN_FEATURE_DIM: int = 8
KAN_NUM_LAYERS: int = 2
KAN_ITERS: int = 200
KAN_LR: float = 0.01
KAN_GP_ITERS: int = 100
KAN_GP_LR: float = 0.03
_KAN_SEED: int = 42


# ============================================================================
# 问题元数据（与 problem.py 保持严格一致）
# F = volume，G = abs(max_disp) - disp_limit <= 0
# ============================================================================

PROBLEM_META: Dict[str, Dict] = {
    "QuickSimu1": {
        "n_var": 3,
        "xl": np.array([0.4, 0.01, 0.01], dtype=float),
        "xu": np.array([1.6, 0.08, 0.08], dtype=float),
        "json_dir": "quick_simu_1",
        "disp_key": "max_disp_x",
        "disp_limit": 0.027867728805696976,   # 与 problem.py QuickSimu1Problem 一致
    },
    "QuickSimu2": {
        "n_var": 2,
        "xl": np.array([0.2, 0.2], dtype=float),
        "xu": np.array([1.6, 1.6], dtype=float),
        "json_dir": "quick_simu_2",
        "disp_key": "max_disp_y",
        "disp_limit": 3.6164872159562756e-07, # 与 problem.py QuickSimu2Problem 一致
    },
}

JSON_BASE = ROOT_DIR / "proxy_models_jsons"

DEFAULT_SEEDS = [2026, 2027, 2028]

# ============================================================================
# 日志工具
# ============================================================================

def _ts() -> str:
    """返回 HH:MM:SS 格式的当前时间戳。"""
    return datetime.now().strftime("%H:%M:%S")


def _log(level: str, tag: str, msg: str) -> None:
    """统一日志格式：[HH:MM:SS][LEVEL] tag | msg"""
    print(f"[{_ts()}][{level:<4}] {tag} | {msg}", flush=True)


def _sec(elapsed: float) -> str:
    """将秒数格式化为易读字符串。"""
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    m, s = divmod(int(elapsed), 60)
    return f"{m}m{s:02d}s"
DEFAULT_PROBLEMS = ["QuickSimu1", "QuickSimu2"]
DEFAULT_COUNTS = list(range(100, 1501, 100))
MODEL_TYPES = ["kriging", "rbf", "polynomial", "kan", "kan_gp"]


# ============================================================================
# CLI 参数解析
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="代理模型 + HA 优化对比实验：五类代理（volume+disp 双目标）+ 真实仿真回评"
    )
    p.add_argument("--pop-size", type=int, default=50, help="HA 种群大小")
    p.add_argument("--n-gen", type=int, default=30, help="HA 迭代代数")
    p.add_argument(
        "--problems", nargs="+", choices=DEFAULT_PROBLEMS, default=DEFAULT_PROBLEMS
    )
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument(
        "--counts", nargs="+", type=int, default=DEFAULT_COUNTS,
        help="采样数列表，默认 100~1500 步长 100"
    )
    p.add_argument(
        "--model-types", nargs="+", choices=MODEL_TYPES, default=MODEL_TYPES
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "results_surrogate",
        help="结果输出目录"
    )
    p.add_argument(
        "--jobs", type=int, default=1,
        help="代理阶段并行进程数（--jobs 1 等价于串行，真实 ANSYS 回评始终串行）"
    )
    return p.parse_args()


# ============================================================================
# 数据加载：同时返回 volume 与 max_disp 两个标签
# ============================================================================

def load_training_data(
    problem_name: str, seed: int, count: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 seed_{seed}_{count}.json，提取 ok=True 的记录。
    返回 (X [n, n_var], y_volume [n,], y_disp [n,])。

    兼容多种历史 JSON 结构：
      - dict:   {"volume": ..., "max_disp_x"/"max_disp_y": ...}
      - list:   [max_disp_x, volume]  （QuickSimu1 旧格式）
      - scalar: disp_y               （QuickSimu2 旧格式，volume 按公式计算）
    """
    meta = PROBLEM_META[problem_name]
    disp_key: str = meta["disp_key"]
    path = JSON_BASE / meta["json_dir"] / f"seed_{seed}_{count}.json"
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    X_list: List[List[float]] = []
    y_vol_list: List[float] = []
    y_disp_list: List[float] = []

    for rec in data["records"]:
        if not rec.get("ok", False):
            continue
        xv = rec["x"]
        yv = rec.get("y")

        vol_val: Optional[float] = None
        disp_val: Optional[float] = None

        if isinstance(yv, dict):
            vol_val_raw = yv.get("volume")
            disp_val_raw = yv.get(disp_key)
            if vol_val_raw is None or disp_val_raw is None:
                continue
            vol_val = float(vol_val_raw)
            disp_val = float(disp_val_raw)

        elif isinstance(yv, list) and len(yv) >= 2:
            # QuickSimu1 旧格式: [max_disp_x, volume]
            disp_val = float(yv[0])
            vol_val = float(yv[1])

        elif yv is not None:
            # QuickSimu2 旧格式: scalar = disp_y；volume 由公式推算
            disp_val = float(yv)
            x1, x2 = float(xv[0]), float(xv[1])
            vol_val = (0.4 ** 2) * x1 + (0.2 ** 2) * x2 + (0.1 ** 2) * (2.0 - x1 - x2)

        else:
            continue

        X_list.append(xv)
        y_vol_list.append(vol_val)
        y_disp_list.append(disp_val)

    if len(X_list) < 10:
        raise ValueError(
            f"有效样本数 {len(X_list)} 不足 10，跳过: {path}"
        )

    return (
        np.array(X_list, dtype=float),
        np.array(y_vol_list, dtype=float),
        np.array(y_disp_list, dtype=float),
    )


# ============================================================================
# KAN / KAN-GP 工具
# ============================================================================

class KANRegressor:
    """KANFeatureExtractor + Linear 输出头，sklearn-style fit/predict，CPU 上运行。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = KAN_HIDDEN_DIM,
        feature_dim: int = KAN_FEATURE_DIM,
        num_layers: int = KAN_NUM_LAYERS,
        iters: int = KAN_ITERS,
        lr: float = KAN_LR,
        seed: int = _KAN_SEED,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.iters = iters
        self.lr = lr
        self.seed = seed
        self._net: Any = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def _build_net(self) -> Any:
        extractor = KANFeatureExtractor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            num_layers=self.num_layers,
        )
        head = nn.Linear(self.feature_dim, 1)
        return nn.Sequential(extractor, head)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KANRegressor":
        torch.manual_seed(self.seed)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        train_x = torch.tensor(X, dtype=torch.float32)
        train_y = torch.tensor(
            (y - self._y_mean) / self._y_std, dtype=torch.float32
        ).unsqueeze(-1)
        self._net = self._build_net()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self._net.train()
        for _ in range(self.iters):
            optimizer.zero_grad()
            pred = self._net(train_x)
            loss = loss_fn(pred, train_y)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("KANRegressor.fit() 尚未调用")
        self._net.eval()
        test_x = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            raw = self._net(test_x).squeeze(-1).numpy()
        return raw * self._y_std + self._y_mean


class KANGPSurrogate:
    """包装已训练 KAN-GP，提供 predict(X) 接口。"""

    def __init__(self, model: Any, likelihood: Any, y_mean: float, y_std: float) -> None:
        self._model = model
        self._likelihood = likelihood
        self._y_mean = y_mean
        self._y_std = y_std

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        self._likelihood.eval()
        test_x = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self._likelihood(self._model(test_x))
        return pred_dist.mean.numpy() * self._y_std + self._y_mean


def _build_kan_gp_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    iters: int = KAN_GP_ITERS,
    lr: float = KAN_GP_LR,
    seed: int = _KAN_SEED,
) -> KANGPSurrogate:
    torch.manual_seed(seed)
    y_mean = float(y.mean())
    y_std = float(y.std()) or 1.0
    train_x = torch.tensor(X, dtype=torch.float32)
    train_y = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
    model, likelihood = create_pimfgpkan(
        train_x=train_x,
        train_y=train_y,
        physics_mean_fn=None,
        input_dim=X.shape[1],
        hidden_dim=KAN_HIDDEN_DIM,
        feature_dim=KAN_FEATURE_DIM,
        num_kan_layers=KAN_NUM_LAYERS,
    )
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(iters):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    return KANGPSurrogate(model, likelihood, y_mean, y_std)


# ============================================================================
# 代理模型构建：单输出 + 双输出
# ============================================================================

def build_surrogate(model_type: str, X: np.ndarray, y: np.ndarray, _label: str = "") -> Any:
    """构建并返回已拟合的单输出代理。`_label` 用于日志标记（volume / disp）。"""
    tag = _label or model_type
    t0 = time.perf_counter()

    if model_type == "kriging":
        print(f"[{_ts()}][SURR]   拟合 kriging ({tag}, n={len(X)}) ...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
                n_restarts_optimizer=5,
                normalize_y=True,
            )
            gp.fit(X, y)
        model = gp

    elif model_type == "rbf":
        print(f"[{_ts()}][SURR]   拟合 rbf ({tag}, n={len(X)}) ...", flush=True)
        try:
            rbf = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=0.0)
            _ = rbf(X[:1])
            model = rbf
        except Exception:
            model = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=1e-3)

    elif model_type == "polynomial":
        print(f"[{_ts()}][SURR]   拟合 polynomial ({tag}, n={len(X)}) ...", flush=True)
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=True)),
            ("lr", LinearRegression()),
        ])
        pipe.fit(X, y)
        model = pipe

    elif model_type == "kan":
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch/gpytorch/kan_gp 不可用，无法构建 KAN 代理")
        print(
            f"[{_ts()}][SURR]   训练 KAN ({tag}, n={len(X)}, iters={KAN_ITERS}) ...",
            flush=True,
        )
        reg = KANRegressor(input_dim=X.shape[1])
        reg.fit(X, y)
        model = reg

    elif model_type == "kan_gp":
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch/gpytorch/kan_gp 不可用，无法构建 KAN-GP 代理")
        print(
            f"[{_ts()}][SURR]   训练 KAN-GP ({tag}, n={len(X)}, iters={KAN_GP_ITERS}) ...",
            flush=True,
        )
        model = _build_kan_gp_surrogate(X, y)

    else:
        raise ValueError(f"未知 model_type: {model_type!r}")

    elapsed = time.perf_counter() - t0
    print(f"[{_ts()}][SURR]   完成 {model_type}/{tag}  耗时 {_sec(elapsed)}", flush=True)
    return model


def build_surrogate_pair(
    model_type: str,
    X: np.ndarray,
    y_volume: np.ndarray,
    y_disp: np.ndarray,
    tag: str = "",
) -> Tuple[Any, Any]:
    """
    为 volume 和 max_disp 各建一个代理，返回 (vol_model, disp_model)。
    两个代理类型相同，分开训练以保证各自最优拟合。
    """
    surrogate_vol = build_surrogate(model_type, X, y_volume, _label=f"{tag}/volume")
    surrogate_disp = build_surrogate(model_type, X, y_disp, _label=f"{tag}/disp")
    return surrogate_vol, surrogate_disp


# ============================================================================
# SurrogateProblem：F = volume，G = abs(disp) - disp_limit
# ============================================================================

def _call_surrogate(model_type: str, surrogate: Any, X: np.ndarray) -> np.ndarray:
    """统一调用接口：rbf 是 callable，其余使用 .predict()。"""
    if model_type == "rbf":
        pred = surrogate(X)
    else:
        pred = surrogate.predict(X)
    return np.asarray(pred, dtype=float).reshape(-1)


class SurrogateProblem(Problem):
    """
    双代理代理问题，目标/约束与 problem.py 中真实问题对齐：
        F = volume_surrogate(x)        （最小化体积）
        G = abs(disp_surrogate(x)) - disp_limit  （位移约束 <= 0）
    """

    def __init__(
        self,
        surrogate_vol: Any,
        surrogate_disp: Any,
        model_type: str,
        n_var: int,
        xl: np.ndarray,
        xu: np.ndarray,
        disp_limit: float,
    ) -> None:
        self.fes: int = 0
        self._surrogate_vol = surrogate_vol
        self._surrogate_disp = surrogate_disp
        self._model_type = model_type
        self._disp_limit = disp_limit
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=1,
            n_eq_constr=0,
            xl=xl.copy(),
            xu=xu.copy(),
        )

    def _evaluate(self, X, out, *args, **kwargs) -> None:
        X = np.atleast_2d(X).astype(float)
        self.fes += X.shape[0]

        # 目标：预测 volume
        try:
            f = _call_surrogate(self._model_type, self._surrogate_vol, X)
        except Exception as exc:
            warnings.warn(f"[SurrogateProblem] volume 预测失败: {exc}")
            f = np.full(X.shape[0], 1e12, dtype=float)

        # 约束：abs(disp) - disp_limit
        try:
            d = _call_surrogate(self._model_type, self._surrogate_disp, X)
            g = np.abs(d) - self._disp_limit
        except Exception as exc:
            warnings.warn(f"[SurrogateProblem] disp 预测失败: {exc}")
            g = np.full(X.shape[0], 1e12, dtype=float)

        out["F"] = f.reshape(-1, 1)
        out["G"] = g.reshape(-1, 1)


# ============================================================================
# 真实仿真回评（始终在主进程串行执行，避免 MAPDL 资源冲突）
# ============================================================================

_real_problems: Dict[str, Any] = {}


def _get_real_problem(problem_name: str) -> Any:
    if problem_name not in _real_problems:
        if problem_name == "QuickSimu1":
            from problem import QuickSimu1Problem  # type: ignore
            _real_problems[problem_name] = QuickSimu1Problem()
        elif problem_name == "QuickSimu2":
            from problem import QuickSimu2Problem  # type: ignore
            _real_problems[problem_name] = QuickSimu2Problem()
        else:
            raise ValueError(f"未知问题: {problem_name}")
    return _real_problems[problem_name]


def evaluate_real(problem_name: str, x: np.ndarray) -> Dict[str, Any]:
    """
    真实 ANSYS 仿真评估单个个体 x。
    返回 {"f_real": volume, "g_real": abs(disp)-limit, "feasible_real": bool}。
    """
    prob = _get_real_problem(problem_name)
    out: Dict = {}
    prob._evaluate(np.atleast_2d(x).astype(float), out)
    f_arr = np.asarray(out.get("F", [np.nan])).flatten()
    g_arr = np.asarray(out.get("G", [np.nan])).flatten()
    f_real = float(f_arr[0]) if f_arr.size > 0 else float("nan")
    g_real = float(g_arr[0]) if g_arr.size > 0 else float("nan")
    feasible = bool(g_real <= 0) if not np.isnan(g_real) else False
    return {"f_real": f_real, "g_real": g_real, "feasible_real": feasible}


# ============================================================================
# 代理阶段（可并行）：加载数据 → 建双代理 → HA 优化 → 返回 x_best
# ============================================================================

def _surrogate_phase(
    problem_name: str,
    model_type: str,
    seed: int,
    count: int,
    pop_size: int,
    n_gen: int,
    task_label: str = "",
) -> Dict[str, Any]:
    """
    可并行执行的代理阶段。
    返回包含 x_best（list）的 dict，出错则抛出异常由调用方处理。
    """
    tag = f"{problem_name}|{model_type}|seed={seed}|count={count}"
    header = task_label or tag
    t_phase = time.perf_counter()

    # 1. 加载数据
    _log("STEP", header, "1/4  加载训练数据 ...")
    t0 = time.perf_counter()
    X_train, y_volume, y_disp = load_training_data(problem_name, seed, count)
    _log("STEP", header,
         f"1/4  数据加载完毕  n={len(X_train)}  耗时 {_sec(time.perf_counter()-t0)}")

    # 2. 构建双代理
    _log("STEP", header, f"2/4  构建双代理 ({model_type}) ...")
    t0 = time.perf_counter()
    surrogate_vol, surrogate_disp = build_surrogate_pair(
        model_type, X_train, y_volume, y_disp, tag=header
    )
    _log("STEP", header,
         f"2/4  双代理构建完毕  耗时 {_sec(time.perf_counter()-t0)}")

    # 3. 包装为 SurrogateProblem
    meta = PROBLEM_META[problem_name]
    surrogate_prob = SurrogateProblem(
        surrogate_vol=surrogate_vol,
        surrogate_disp=surrogate_disp,
        model_type=model_type,
        n_var=meta["n_var"],
        xl=meta["xl"],
        xu=meta["xu"],
        disp_limit=meta["disp_limit"],
    )

    # 4. HA 优化
    _log("STEP", header,
         f"3/4  HA 优化代理  pop={pop_size}  gen={n_gen} ...")
    t0 = time.perf_counter()
    result, history = run_ha_nelder_mead_method(
        problem=surrogate_prob,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        method="gp",
        initial_pop=None,
    )
    _log("STEP", header,
         f"3/4  HA 优化完毕  fes={surrogate_prob.fes}  耗时 {_sec(time.perf_counter()-t0)}")

    # 5. 提取最优个体
    x_best: Optional[np.ndarray] = None
    if result is not None and result.X is not None:
        x_best = np.asarray(result.X).flatten().astype(float)
    if x_best is None and history:
        best_gen = min(history, key=lambda g: float(g.best_F))
        x_best = np.asarray(best_gen.best_X).flatten().astype(float)
    if x_best is None:
        raise RuntimeError(f"[{tag}] 无法获取最优个体")

    _log("STEP", header,
         f"4/4  代理阶段完毕  x_best={[round(v,5) for v in x_best.tolist()]}"
         f"  总耗时 {_sec(time.perf_counter()-t_phase)}")

    return {
        "problem": problem_name,
        "model_type": model_type,
        "seed": seed,
        "sample_count": count,
        "x_best": x_best.tolist(),
    }


def _surrogate_phase_worker(kwargs: Dict) -> Dict:
    """
    顶层包装，供 ProcessPoolExecutor 调用（需可 pickle）。
    异常信息内联到返回 dict，不抛出，方便主进程统一收集。
    """
    try:
        return _surrogate_phase(**kwargs)
    except Exception as exc:
        import traceback
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "problem": kwargs.get("problem_name", ""),
            "model_type": kwargs.get("model_type", ""),
            "seed": kwargs.get("seed", -1),
            "sample_count": kwargs.get("count", -1),
        }


# ============================================================================
# 单次完整实验（串行时直接调用）
# ============================================================================

def run_one(
    problem_name: str,
    model_type: str,
    seed: int,
    count: int,
    pop_size: int,
    n_gen: int,
    task_idx: int = 0,
    total_tasks: int = 0,
) -> Optional[Dict]:
    """
    串行完整流程：代理阶段 + 真实仿真回评。
    失败返回 None。
    """
    tag = f"{problem_name}|{model_type}|seed={seed}|count={count}"
    progress = f"[{task_idx}/{total_tasks}]" if total_tasks > 0 else ""
    t_total = time.perf_counter()
    _log("RUN ", f"{progress}{tag}", "开始")

    try:
        sr = _surrogate_phase(
            problem_name, model_type, seed, count, pop_size, n_gen,
            task_label=f"{progress}{tag}",
        )
    except (FileNotFoundError, ValueError) as e:
        _log("SKIP", tag, f"数据加载失败: {e}")
        return None
    except Exception as e:
        _log("SKIP", tag, f"代理阶段失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    x_best = np.array(sr["x_best"])

    _log("REAL", tag, f"开始真实仿真  x={[round(v,5) for v in x_best.tolist()]}")
    t0 = time.perf_counter()
    try:
        real = evaluate_real(problem_name, x_best)
        _log("REAL", tag,
             f"完毕  f={real['f_real']:.6e}  g={real['g_real']:.6e}"
             f"  feasible={real['feasible_real']}"
             f"  耗时 {_sec(time.perf_counter()-t0)}")
    except Exception as e:
        _log("WARN", tag, f"真实仿真失败: {e}")
        real = {"f_real": float("nan"), "g_real": float("nan"), "feasible_real": False}

    _log("DONE", tag, f"全流程完毕  总耗时 {_sec(time.perf_counter()-t_total)}")
    return {**sr, **real}


# ============================================================================
# CSV 导出：明细 + 汇总
# ============================================================================

def write_detail_csv(
    rows: List[Dict], problem_name: str, output_dir: Path
) -> Path:
    """明细 CSV：每次实验一行，包含 f_real / g_real / feasible_real。"""
    if not rows:
        return output_dir / f"{problem_name}_surrogate_detail.csv"

    meta = PROBLEM_META[problem_name]
    n_var = meta["n_var"]
    x_cols = [f"x{i + 1}" for i in range(n_var)]
    fieldnames = (
        ["problem", "model_type", "seed", "sample_count"]
        + x_cols
        + ["f_real", "g_real", "feasible_real"]
    )

    path = output_dir / f"{problem_name}_surrogate_detail.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat: Dict = {
                "problem": row["problem"],
                "model_type": row["model_type"],
                "seed": row["seed"],
                "sample_count": row["sample_count"],
                "f_real": row.get("f_real", float("nan")),
                "g_real": row.get("g_real", float("nan")),
                "feasible_real": row.get("feasible_real", False),
            }
            for i, xv in enumerate(row.get("x_best", [])):
                flat[f"x{i + 1}"] = xv
            writer.writerow(flat)

    print(f"[SAVE] 明细 CSV: {path}  ({len(rows)} 行)")
    return path


def write_summary_csv(
    rows: List[Dict], problem_name: str, output_dir: Path
) -> Path:
    """
    汇总 CSV：按 (model_type, sample_count) 聚合三个 seed 求均值。
    新增字段：f_real_mean / g_real_mean / feasible_rate。
    """
    meta = PROBLEM_META[problem_name]
    n_var = meta["n_var"]
    x_cols = [f"x{i + 1}_mean" for i in range(n_var)]
    fieldnames = (
        ["model_type", "sample_count"]
        + x_cols
        + ["f_real_mean", "g_real_mean", "feasible_rate", "n_seeds"]
    )

    groups: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model_type"], row["sample_count"])].append(row)

    summary_rows = []
    for (model_type, count), grp in sorted(groups.items()):
        x_arr = np.array(
            [r["x_best"] for r in grp if r.get("x_best") is not None],
            dtype=float,
        )
        f_vals = [
            r["f_real"] for r in grp
            if r.get("f_real") is not None and not np.isnan(float(r["f_real"]))
        ]
        g_vals = [
            r["g_real"] for r in grp
            if r.get("g_real") is not None and not np.isnan(float(r["g_real"]))
        ]
        feasible_vals = [bool(r.get("feasible_real", False)) for r in grp]

        x_mean = x_arr.mean(axis=0).tolist() if len(x_arr) > 0 else [float("nan")] * n_var
        f_mean = float(np.mean(f_vals)) if f_vals else float("nan")
        g_mean = float(np.mean(g_vals)) if g_vals else float("nan")
        feasible_rate = float(np.mean(feasible_vals)) if feasible_vals else float("nan")

        sr: Dict = {
            "model_type": model_type,
            "sample_count": count,
            "f_real_mean": f_mean,
            "g_real_mean": g_mean,
            "feasible_rate": feasible_rate,
            "n_seeds": len(grp),
        }
        for i, xv in enumerate(x_mean):
            sr[f"x{i + 1}_mean"] = xv
        summary_rows.append(sr)

    path = output_dir / f"{problem_name}_surrogate_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[SAVE] 汇总 CSV: {path}  ({len(summary_rows)} 行)")
    return path


# ============================================================================
# 主流程：串行（--jobs 1）或代理阶段并行（--jobs N）
# ============================================================================

def _build_task_list(args: argparse.Namespace) -> List[Dict]:
    """构建所有任务参数列表（按 problem → count → model → seed 排列）。"""
    tasks = []
    for problem_name in args.problems:
        for count in args.counts:
            for model_type in args.model_types:
                for seed in args.seeds:
                    tasks.append({
                        "problem_name": problem_name,
                        "model_type": model_type,
                        "seed": seed,
                        "count": count,
                        "pop_size": args.pop_size,
                        "n_gen": args.n_gen,
                    })
    return tasks


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, List[Dict]] = defaultdict(list)
    failures: List[str] = []

    tasks = _build_task_list(args)
    n_tasks = len(tasks)
    t_main = time.perf_counter()

    sep = "=" * 72
    print(sep, flush=True)
    print(
        f"[{_ts()}] 启动评估  "
        f"problems={args.problems}  model_types={args.model_types}  "
        f"seeds={args.seeds}  counts({len(args.counts)})={args.counts[:3]}{'...' if len(args.counts)>3 else ''}  "
        f"总任务数={n_tasks}  jobs={args.jobs}",
        flush=True,
    )
    print(sep, flush=True)

    if args.jobs > 1:
        # ----------------------------------------------------------------
        # 并行代理阶段：ProcessPoolExecutor
        # 提交所有任务，收集 x_best；真实仿真在主进程串行执行。
        # ----------------------------------------------------------------
        print(
            f"[{_ts()}][INFO] 代理阶段并行：{args.jobs} 进程 × {n_tasks} 任务",
            flush=True,
        )
        surrogate_results: List[Dict] = [None] * n_tasks  # type: ignore
        done_count = 0
        t_surr = time.perf_counter()

        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_to_idx = {
                pool.submit(_surrogate_phase_worker, t): i
                for i, t in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                done_count += 1
                t = tasks[idx]
                tag = (
                    f"{t['problem_name']}|{t['model_type']}"
                    f"|seed={t['seed']}|count={t['count']}"
                )
                try:
                    surrogate_results[idx] = future.result()
                    status = "OK" if "error" not in surrogate_results[idx] else "ERR"
                except Exception as exc:
                    surrogate_results[idx] = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "problem": t["problem_name"],
                        "model_type": t["model_type"],
                        "seed": t["seed"],
                        "sample_count": t["count"],
                    }
                    status = "ERR"
                print(
                    f"[{_ts()}][PAR ] [{done_count}/{n_tasks}] {tag} → {status}",
                    flush=True,
                )

        print(
            f"[{_ts()}][INFO] 所有代理阶段完毕  耗时 {_sec(time.perf_counter()-t_surr)}",
            flush=True,
        )

        # 串行真实回评
        real_total = sum(1 for sr in surrogate_results if sr and "error" not in sr)
        real_done = 0
        print(f"[{_ts()}][INFO] 开始串行真实仿真  共 {real_total} 个有效任务", flush=True)
        for sr, task in zip(surrogate_results, tasks):
            tag = (
                f"{task['problem_name']}|{task['model_type']}"
                f"|seed={task['seed']}|count={task['count']}"
            )
            if sr is None or "error" in sr:
                err = (sr or {}).get("error", "未知错误")
                tb = (sr or {}).get("traceback", "")
                failures.append(f"{tag}: {err}")
                _log("SKIP", tag, f"代理阶段失败: {err}")
                if tb:
                    print(tb, flush=True)
                continue

            real_done += 1
            _log("REAL", tag,
                 f"[{real_done}/{real_total}] 开始真实仿真  x={[round(v,5) for v in sr['x_best']]}")
            t0 = time.perf_counter()
            try:
                x_best = np.array(sr["x_best"])
                real = evaluate_real(task["problem_name"], x_best)
                _log("REAL", tag,
                     f"完毕  f={real['f_real']:.6e}  g={real['g_real']:.6e}"
                     f"  feasible={real['feasible_real']}"
                     f"  耗时 {_sec(time.perf_counter()-t0)}")
                row = {**sr, **real}
                all_results[task["problem_name"]].append(row)
            except Exception as exc:
                failures.append(f"{tag}: 真实仿真失败 {exc}")
                _log("WARN", tag, f"真实仿真失败: {exc}")

    else:
        # ----------------------------------------------------------------
        # 串行模式（兼容原有行为）
        # ----------------------------------------------------------------
        for i, task in enumerate(tasks, 1):
            p = task["problem_name"]
            tag = f"{p}|{task['model_type']}|seed={task['seed']}|count={task['count']}"
            print(f"\n{sep}", flush=True)
            try:
                row = run_one(
                    problem_name=p,
                    model_type=task["model_type"],
                    seed=task["seed"],
                    count=task["count"],
                    pop_size=task["pop_size"],
                    n_gen=task["n_gen"],
                    task_idx=i,
                    total_tasks=n_tasks,
                )
                if row is not None:
                    all_results[p].append(row)
            except Exception as exc:
                failures.append(f"{tag}: {type(exc).__name__}: {exc}")
                _log("FAIL", tag, f"{type(exc).__name__}: {exc}")

    # 每个 problem 写 CSV（断点容忍）
    print(f"\n{sep}", flush=True)
    for problem_name in args.problems:
        if all_results[problem_name]:
            write_detail_csv(all_results[problem_name], problem_name, args.output_dir)
            write_summary_csv(all_results[problem_name], problem_name, args.output_dir)

    t_elapsed = time.perf_counter() - t_main
    ok = n_tasks - len(failures)
    print(sep, flush=True)
    if failures:
        print(f"[{_ts()}][WARN] 共 {len(failures)} 次失败:", flush=True)
        for msg in failures:
            print(f"  - {msg}", flush=True)
    print(
        f"[{_ts()}][DONE] 完成 {ok}/{n_tasks} 任务  总耗时 {_sec(t_elapsed)}  "
        f"结果目录: {args.output_dir}",
        flush=True,
    )
    print(sep, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
