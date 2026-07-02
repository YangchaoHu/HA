"""Experiment recording callbacks and run directory helpers."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.core.callback import Callback

from .metrics.indicators import compute_indicator_row, nondominated_feasible_front


def utc_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def make_run_dir(
    output_root: str | Path,
    problem_name: str,
    algorithm_name: str,
    seed: int,
) -> Path:
    root = Path(output_root)
    stamp = utc_timestamp()
    run_dir = root / stamp / problem_name / algorithm_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _population_get(pop, key: str) -> np.ndarray | None:
    try:
        values = pop.get(key)
    except Exception:
        return None
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    return np.atleast_2d(arr) if arr.ndim == 1 else arr


def _flatten_cv(cv: np.ndarray | None, n: int) -> np.ndarray | None:
    if cv is None:
        return None
    arr = np.asarray(cv, dtype=float).reshape(-1)
    if arr.size != n:
        return None
    return arr


def write_population_csv(path: str | Path, x: np.ndarray, f: np.ndarray, cv: np.ndarray | None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    x = np.atleast_2d(np.asarray(x, dtype=float))
    f = np.atleast_2d(np.asarray(f, dtype=float))
    n_rows = len(f)
    cv_flat = _flatten_cv(cv, n_rows)

    headers = [f"x{i}" for i in range(x.shape[1])] + [f"f{i}" for i in range(f.shape[1])]
    if cv_flat is not None:
        headers.append("cv")

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for i in range(n_rows):
            row = list(map(float, x[i])) + list(map(float, f[i]))
            if cv_flat is not None:
                row.append(float(cv_flat[i]))
            writer.writerow(row)


class ExperimentRecorder(Callback):
    """Save every generation and append per-generation metrics."""

    def __init__(
        self,
        run_dir: str | Path,
        reference_front: np.ndarray | None = None,
        ref_point: np.ndarray | None = None,
        save_every: int = 1,
    ) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.population_dir = self.run_dir / "populations"
        self.reference_front = reference_front
        self.ref_point = ref_point
        self.save_every = max(1, int(save_every))
        self.metric_path = self.run_dir / "metrics_by_generation.csv"
        self._metric_headers: list[str] | None = None

    def notify(self, algorithm) -> None:
        gen = int(getattr(algorithm, "n_gen", 0) or 0)
        if gen % self.save_every != 0:
            return

        pop = getattr(algorithm, "pop", None)
        if pop is None:
            return

        x = _population_get(pop, "X")
        f = _population_get(pop, "F")
        cv = _population_get(pop, "CV")
        if x is None or f is None:
            return

        n_eval = int(getattr(getattr(algorithm, "evaluator", None), "n_eval", 0) or 0)
        write_population_csv(self.population_dir / f"gen_{gen:04d}.csv", x, f, cv)

        row = {
            "generation": gen,
            "n_eval": n_eval,
            "pop_size": int(len(f)),
        }
        row.update(compute_indicator_row(f, cv=cv, reference_front=self.reference_front, ref_point=self.ref_point))
        self._append_metric_row(row)

    def _append_metric_row(self, row: dict[str, Any]) -> None:
        row = {key: _csv_value(value) for key, value in row.items()}
        is_new = not self.metric_path.exists()
        if self._metric_headers is None:
            self._metric_headers = list(row.keys())

        with self.metric_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._metric_headers)
            if is_new:
                writer.writeheader()
            writer.writerow(row)


def save_final_artifacts(
    run_dir: str | Path,
    result,
    reference_front: np.ndarray | None = None,
    ref_point: np.ndarray | None = None,
) -> dict[str, Any]:
    """Save final population/front and return final metrics."""

    run_path = Path(run_dir)
    pop = getattr(result, "pop", None)
    if pop is None:
        return {"status": "no_population"}

    x = _population_get(pop, "X")
    f = _population_get(pop, "F")
    cv = _population_get(pop, "CV")
    if x is None or f is None:
        return {"status": "invalid_population"}

    write_population_csv(run_path / "final_population.csv", x, f, cv)
    front = nondominated_feasible_front(f, cv=cv)
    if len(front) > 0:
        np.savetxt(run_path / "final_front.csv", front, delimiter=",", header=",".join(f"f{i}" for i in range(front.shape[1])), comments="")

    metrics = compute_indicator_row(f, cv=cv, reference_front=reference_front, ref_point=ref_point)
    metrics["status"] = "ok"
    metrics["n_final_pop"] = int(len(f))
    metrics["n_final_front"] = int(len(front))
    write_json(run_path / "final_metrics.json", metrics)
    return metrics


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value

