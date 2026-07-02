"""Smoke tests for the MOEA benchmark scaffold."""

from __future__ import annotations

import numpy as np

from moea_benchmark.benchmarks.catalog import create_problem, list_problem_specs, reference_front
from moea_benchmark.metrics.indicators import compute_indicator_row
from moea_benchmark.surrogates import fit_surrogate


def test_math_problem_reference_front() -> None:
    problem = create_problem("zdt1")
    pf = reference_front(problem)
    assert problem.n_obj == 2
    assert pf is not None
    assert pf.shape[1] == 2


def test_catalog_includes_three_tiers() -> None:
    specs = list_problem_specs(include_expensive=True)
    tiers = {spec.tier for spec in specs}
    assert {"math", "engineering", "cae"}.issubset(tiers)


def test_indicator_row_smoke() -> None:
    f = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    pf = f.copy()
    row = compute_indicator_row(f, reference_front=pf)
    assert row["n_feasible_nd"] == 3
    assert row["igd"] == 0.0


def test_surrogate_smoke() -> None:
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    f = np.column_stack([x[:, 0] ** 2, 1.0 - x[:, 0]])
    surrogate = fit_surrogate(x, f, kind="rbf")
    pred = surrogate.predict([[0.25]])
    assert pred.shape == (1, 2)

