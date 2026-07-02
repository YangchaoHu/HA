"""Performance indicators for multi-objective experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def feasible_mask(cv: np.ndarray | None, n: int) -> np.ndarray:
    """Return a feasibility mask using pymoo's non-positive CV convention."""

    if cv is None:
        return np.ones(n, dtype=bool)
    arr = np.asarray(cv, dtype=float).reshape(-1)
    if arr.size != n:
        return np.ones(n, dtype=bool)
    return arr <= 1e-12


def nondominated_feasible_front(
    f_values: np.ndarray,
    cv: np.ndarray | None = None,
) -> np.ndarray:
    """Return feasible nondominated objective vectors."""

    f = np.asarray(f_values, dtype=float)
    if f.size == 0:
        return np.empty((0, 0), dtype=float)
    f = np.atleast_2d(f)
    mask = feasible_mask(cv, len(f))
    feasible = f[mask]
    feasible = feasible[np.all(np.isfinite(feasible), axis=1)]
    if len(feasible) == 0:
        return np.empty((0, f.shape[1]), dtype=float)
    front_indices = NonDominatedSorting().do(feasible, only_non_dominated_front=True)
    return feasible[front_indices]


def default_ref_point(
    front: np.ndarray,
    reference_front: np.ndarray | None = None,
    margin_ratio: float = 0.1,
) -> np.ndarray | None:
    """Construct a conservative hypervolume reference point."""

    candidates = []
    if reference_front is not None and len(reference_front) > 0:
        candidates.append(np.asarray(reference_front, dtype=float))
    if front is not None and len(front) > 0:
        candidates.append(np.asarray(front, dtype=float))
    if not candidates:
        return None

    all_f = np.vstack(candidates)
    finite = all_f[np.all(np.isfinite(all_f), axis=1)]
    if len(finite) == 0:
        return None
    f_min = finite.min(axis=0)
    f_max = finite.max(axis=0)
    span = np.maximum(f_max - f_min, 1.0)
    return f_max + margin_ratio * span


def compute_indicator_row(
    f_values: np.ndarray,
    cv: np.ndarray | None = None,
    reference_front: np.ndarray | None = None,
    ref_point: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Compute core indicators for one population snapshot."""

    front = nondominated_feasible_front(f_values, cv=cv)
    key = f"{prefix}_" if prefix else ""
    row: dict[str, Any] = {
        f"{key}n_feasible_nd": int(len(front)),
        f"{key}hv": math.nan,
        f"{key}igd": math.nan,
        f"{key}igd_plus": math.nan,
    }

    if len(front) == 0:
        return row

    rp = np.asarray(ref_point, dtype=float) if ref_point is not None else default_ref_point(front, reference_front)
    if rp is not None:
        try:
            row[f"{key}hv"] = float(HV(ref_point=rp).do(front))
        except Exception:
            row[f"{key}hv"] = math.nan

    if reference_front is not None and len(reference_front) > 0:
        pf = np.asarray(reference_front, dtype=float)
        try:
            row[f"{key}igd"] = float(IGD(pf).do(front))
        except Exception:
            row[f"{key}igd"] = math.nan
        try:
            row[f"{key}igd_plus"] = float(IGDPlus(pf).do(front))
        except Exception:
            row[f"{key}igd_plus"] = math.nan

    return row

