"""
fit_surrogates_quicksimu1.py
----------------------------

对 quick_simu_1 与 quick_simu_2 的 seed_*_1500.json 数据建模，
对两个目标分别拟合：
  - volume
  - max_disp（quick_simu_1 为 max_disp_x，quick_simu_2 为 max_disp_y）

使用六种方法：
  - Kriging
  - RBF
  - 二次多项式
  - MLP(hidden=64)
  - KAN（经典 KAN + 线性回归头）
  - KAN-GP（KAN 深度核 + 精确 GP）

每个问题按 80/20 划分训练/测试，并导出 MAE / RMSE / R² 到 CSV。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 必须在 numpy/sklearn import 之前设置线程数，避免 Windows MKL 警告
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.interpolate import RBFInterpolator

# --------------------------------------------------------------------------
# 可选依赖：torch / gpytorch / kan_gp
# 若环境缺少这些包，KAN 与 KAN-GP 两类模型在运行时会捕获 ImportError 并回退 NaN，
# 其余模型不受影响。
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

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "results_surrogate"

SEEDS = [2026, 2027, 2028]
SAMPLE_COUNT = 1500
TEST_SIZE = 0.2
SPLIT_SEED = 42   # 保证 train_test_split 可复现

MODEL_NAMES = ["kriging", "rbf", "polynomial", "mlp64", "kan", "kan_gp"]

# --------------------------------------------------------------------------
# KAN / KAN-GP 超参数（集中管理，便于后续扫参）
# --------------------------------------------------------------------------
KAN_HIDDEN_DIM: int = 16       # KANFeatureExtractor 隐层宽度
KAN_FEATURE_DIM: int = 8       # KANFeatureExtractor 输出维度
KAN_NUM_LAYERS: int = 2        # KAN 层数
KAN_ITERS: int = 200           # 训练迭代轮数
KAN_LR: float = 0.01          # Adam 学习率
KAN_GP_ITERS: int = 100        # KAN-GP 训练迭代轮数
KAN_GP_LR: float = 0.03       # KAN-GP Adam 学习率

PROBLEM_CONFIG = {
    "quick_simu_1": {
        "json_dir": ROOT / "proxy_models_jsons" / "quick_simu_1",
        "disp_key": "max_disp_x",
        "out_prefix": "QuickSimu1",
    },
    "quick_simu_2": {
        "json_dir": ROOT / "proxy_models_jsons" / "quick_simu_2",
        "disp_key": "max_disp_y",
        "out_prefix": "QuickSimu2",
    },
}


# ============================================================================
# 数据加载
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuickSimu 代理模型双输出拟合评估")
    p.add_argument(
        "--problems",
        nargs="+",
        choices=list(PROBLEM_CONFIG.keys()),
        default=list(PROBLEM_CONFIG.keys()),
        help="要评估的问题",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="随机种子列表")
    p.add_argument("--sample-count", type=int, default=SAMPLE_COUNT, help="读取的样本规模文件后缀")
    p.add_argument("--test-size", type=float, default=TEST_SIZE, help="测试集比例")
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED, help="划分随机种子")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="输出目录")
    return p.parse_args()


def load_json_data(problem_name: str, seed: int, count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 seed_{seed}_{count}.json，提取 ok=True 记录。
    返回:
      X (n, d),
      y_volume (n,),
      y_disp (n,)
    """
    cfg = PROBLEM_CONFIG[problem_name]
    json_dir: Path = cfg["json_dir"]
    disp_key: str = cfg["disp_key"]

    path = json_dir / f"seed_{seed}_{count}.json"
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    X_list, y_vol_list, y_disp_list = [], [], []
    for rec in data["records"]:
        if not rec.get("ok", False):
            continue
        X_list.append(rec["x"])
        y = rec.get("y")
        if isinstance(y, dict):
            # 迁移后结构
            if problem_name == "quick_simu_1":
                disp_val = y.get("max_disp_x")
            else:
                disp_val = y.get("max_disp_y")
            vol_val = y.get("volume")
        elif isinstance(y, list) and len(y) >= 2:
            # 旧结构 quick_simu_1: [max_disp_x, volume]
            disp_val, vol_val = y[0], y[1]
        else:
            # 旧结构 quick_simu_2: y 为标量，volume 由公式计算
            disp_val = y
            x1, x2 = float(rec["x"][0]), float(rec["x"][1])
            vol_val = (0.4 ** 2) * x1 + (0.2 ** 2) * x2 + (0.1 ** 2) * (2.0 - x1 - x2)

        if disp_val is None or vol_val is None:
            continue
        y_disp_list.append(float(disp_val))
        y_vol_list.append(float(vol_val))

    X = np.array(X_list, dtype=float)
    y_volume = np.array(y_vol_list, dtype=float)
    y_disp = np.array(y_disp_list, dtype=float)
    return X, y_volume, y_disp


# ============================================================================
# 模型构建（每次调用返回新实例，保证 seed 间独立）
# ============================================================================

def make_kriging() -> GaussianProcessRegressor:
    return GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
        n_restarts_optimizer=5,
        normalize_y=True,
    )


def make_rbf():
    return None   # RBFInterpolator 在拟合时直接构建，此处占位


def make_polynomial() -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=True)),
        ("lr", LinearRegression()),
    ])


def make_mlp() -> Pipeline:
    """神经网络：隐藏层 64，前后加标准化。"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=SPLIT_SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )),
    ])


# ============================================================================
# KAN / KAN-GP 工具（需要 torch + gpytorch + kan_gp）
# ============================================================================

class KANRegressor:
    """经典 KAN 回归器：KANFeatureExtractor + 线性输出头，sklearn-style 接口。

    fit / predict 操作均在 CPU 上进行；训练前对目标值做 z-score 标准化以提升
    数值稳定性，预测时还原到原始量纲。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = KAN_HIDDEN_DIM,
        feature_dim: int = KAN_FEATURE_DIM,
        num_layers: int = KAN_NUM_LAYERS,
        iters: int = KAN_ITERS,
        lr: float = KAN_LR,
        seed: int = SPLIT_SEED,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.iters = iters
        self.lr = lr
        self.seed = seed

        self._net: Optional["nn.Module"] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def _build_net(self) -> "nn.Module":
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


def _train_kan_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    iters: int = KAN_GP_ITERS,
    lr: float = KAN_GP_LR,
    seed: int = SPLIT_SEED,
):
    """训练 KAN-GP 并返回 (model, likelihood, y_mean, y_std)。

    目标值在训练前做 z-score 标准化；返回的均值/标准差用于还原预测。
    """
    torch.manual_seed(seed)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) or 1.0

    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(
        (y_train - y_mean) / y_std, dtype=torch.float32
    )

    model, likelihood = create_pimfgpkan(
        train_x=train_x,
        train_y=train_y,
        physics_mean_fn=None,
        input_dim=X_train.shape[1],
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

    return model, likelihood, y_mean, y_std


# ============================================================================
# 单次实验：给定训练集/测试集，训练并评估一种模型
# ============================================================================

def evaluate_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    训练模型并返回 {'mae': ..., 'rmse': ..., 'r2': ...}。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if model_name == "kriging":
            model = make_kriging()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_name == "rbf":
            try:
                rbf = RBFInterpolator(X_train, y_train, kernel="thin_plate_spline", smoothing=0.0)
                y_pred = rbf(X_test)
            except Exception:
                rbf = RBFInterpolator(X_train, y_train, kernel="thin_plate_spline", smoothing=1e-3)
                y_pred = rbf(X_test)

        elif model_name == "polynomial":
            model = make_polynomial()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_name == "mlp64":
            model = make_mlp()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_name == "kan":
            if not _TORCH_AVAILABLE:
                raise RuntimeError("torch/gpytorch/kan_gp 不可用，无法运行 KAN 模型")
            reg = KANRegressor(input_dim=X_train.shape[1])
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

        elif model_name == "kan_gp":
            if not _TORCH_AVAILABLE:
                raise RuntimeError("torch/gpytorch/kan_gp 不可用，无法运行 KAN-GP 模型")
            gp_model, likelihood, y_mean, y_std = _train_kan_gp(X_train, y_train)
            gp_model.eval()
            likelihood.eval()
            test_x = torch.tensor(X_test, dtype=torch.float32)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_dist = likelihood(gp_model(test_x))
            y_pred = pred_dist.mean.numpy() * y_std + y_mean

        else:
            raise ValueError(f"未知模型: {model_name}")

    y_pred = np.asarray(y_pred, dtype=float).flatten()
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ============================================================================
# 主流程
# ============================================================================

def _save_rows(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_problem(
    *,
    problem_name: str,
    seeds: List[int],
    sample_count: int,
    test_size: float,
    split_seed: int,
    output_dir: Path,
) -> None:
    out_prefix = PROBLEM_CONFIG[problem_name]["out_prefix"]
    detail_rows: List[Dict] = []

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"[{problem_name}] seed={seed} 加载数据...")
        X, y_volume, y_disp = load_json_data(problem_name, seed, sample_count)
        print(f"[{problem_name}] 有效样本: {len(X)}")

        X_train, X_test, yv_train, yv_test = train_test_split(
            X, y_volume, test_size=test_size, random_state=split_seed
        )
        _, _, yd_train, yd_test = train_test_split(
            X, y_disp, test_size=test_size, random_state=split_seed
        )
        print(f"[{problem_name}] 训练集: {len(X_train)}, 测试集: {len(X_test)}")

        for model_name in MODEL_NAMES:
            print(f"[{problem_name}] seed={seed} 训练 {model_name} ...")
            # volume
            try:
                m_vol = evaluate_model(model_name, X_train, yv_train, X_test, yv_test)
            except Exception as exc:
                print(f"  volume 失败: {exc}")
                m_vol = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
            # max_disp
            try:
                m_disp = evaluate_model(model_name, X_train, yd_train, X_test, yd_test)
            except Exception as exc:
                print(f"  max_disp 失败: {exc}")
                m_disp = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

            detail_rows.append(
                {
                    "problem": problem_name,
                    "seed": seed,
                    "model": model_name,
                    "target": "volume",
                    "mae": m_vol["mae"],
                    "rmse": m_vol["rmse"],
                    "r2": m_vol["r2"],
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }
            )
            detail_rows.append(
                {
                    "problem": problem_name,
                    "seed": seed,
                    "model": model_name,
                    "target": "max_disp",
                    "mae": m_disp["mae"],
                    "rmse": m_disp["rmse"],
                    "r2": m_disp["r2"],
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }
            )

    detail_path = output_dir / f"{out_prefix}_dual_output_metrics_by_seed.csv"
    _save_rows(
        detail_path,
        detail_rows,
        ["problem", "seed", "model", "target", "mae", "rmse", "r2", "n_train", "n_test"],
    )
    print(f"\n[SAVE] 明细 CSV: {detail_path}")

    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"mae": [], "rmse": [], "r2": []}
    )
    for row in detail_rows:
        key = (row["model"], row["target"])
        for k in ("mae", "rmse", "r2"):
            v = float(row[k])
            if not np.isnan(v):
                grouped[key][k].append(v)

    mean_rows: List[Dict] = []
    for model_name in MODEL_NAMES:
        for target in ("volume", "max_disp"):
            g = grouped[(model_name, target)]
            mean_rows.append(
                {
                    "problem": problem_name,
                    "model": model_name,
                    "target": target,
                    "mae_mean": float(np.mean(g["mae"])) if g["mae"] else float("nan"),
                    "rmse_mean": float(np.mean(g["rmse"])) if g["rmse"] else float("nan"),
                    "r2_mean": float(np.mean(g["r2"])) if g["r2"] else float("nan"),
                    "n_seeds": len(g["mae"]),
                }
            )

    mean_path = output_dir / f"{out_prefix}_dual_output_metrics_mean.csv"
    _save_rows(
        mean_path,
        mean_rows,
        ["problem", "model", "target", "mae_mean", "rmse_mean", "r2_mean", "n_seeds"],
    )
    print(f"[SAVE] 均值 CSV:  {mean_path}")

    print(f"\n{'=' * 70}")
    print(f"{problem_name} 双输出拟合效果（三 seed 均值，按 target+RMSE 升序）")
    print(f"{'=' * 70}")
    mean_sorted = sorted(
        mean_rows,
        key=lambda r: (r["target"], float("inf") if np.isnan(r["rmse_mean"]) else r["rmse_mean"]),
    )
    for r in mean_sorted:
        print(
            f"{r['target']:<9} | {r['model']:<10} | "
            f"MAE={r['mae_mean']:.6e}, RMSE={r['rmse_mean']:.6e}, R2={r['r2_mean']:.4f}, n={r['n_seeds']}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for problem_name in args.problems:
        run_problem(
            problem_name=problem_name,
            seeds=list(args.seeds),
            sample_count=args.sample_count,
            test_size=float(args.test_size),
            split_seed=int(args.split_seed),
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
