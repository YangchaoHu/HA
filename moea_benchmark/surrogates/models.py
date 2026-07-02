"""Small surrogate-model adapters for multi-output objective prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.kernel_ridge import KernelRidge


SurrogateKind = Literal["rbf", "gp"]


@dataclass
class MultiOutputSurrogate:
    """Predicts multiple objectives from a common decision vector."""

    model: object
    kind: str
    metrics: dict[str, float]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.asarray(self.model.predict(x), dtype=float)


def fit_surrogate(
    x: np.ndarray,
    f: np.ndarray,
    kind: SurrogateKind = "rbf",
    random_state: int = 1,
) -> MultiOutputSurrogate:
    """Fit a compact multi-output surrogate model."""

    x = np.atleast_2d(np.asarray(x, dtype=float))
    f = np.atleast_2d(np.asarray(f, dtype=float))
    if len(x) != len(f):
        raise ValueError("x and f must have the same number of samples.")
    if len(x) < 4:
        raise ValueError("At least 4 samples are required to fit a surrogate.")

    if kind == "rbf":
        estimator = KernelRidge(kernel="rbf", alpha=1e-8, gamma=None)
        model = make_pipeline(StandardScaler(), MultiOutputRegressor(estimator))
    elif kind == "gp":
        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-8)
        estimator = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=random_state,
            n_restarts_optimizer=1,
        )
        model = make_pipeline(StandardScaler(), MultiOutputRegressor(estimator))
    else:
        raise ValueError(f"Unknown surrogate kind: {kind}")

    model.fit(x, f)
    pred = np.asarray(model.predict(x), dtype=float)
    metrics = {
        "train_mae": float(mean_absolute_error(f, pred)),
        "train_rmse": float(mean_squared_error(f, pred) ** 0.5),
        "train_r2": float(r2_score(f, pred, multioutput="uniform_average")),
        "n_samples": float(len(x)),
    }
    return MultiOutputSurrogate(model=model, kind=kind, metrics=metrics)

