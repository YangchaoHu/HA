"""Experiment runner for one or many benchmark jobs."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .algorithms import create_algorithm
from .benchmarks.catalog import create_problem, get_problem_spec, reference_front
from .metrics.indicators import default_ref_point
from .recording import ExperimentRecorder, make_run_dir, save_final_artifacts, write_json


@dataclass(frozen=True)
class RunResult:
    problem: str
    algorithm: str
    seed: int
    run_dir: Path
    status: str
    elapsed_sec: float
    final_metrics: dict[str, Any]


def run_single(
    problem_name: str,
    algorithm_name: str,
    seed: int,
    pop_size: int,
    n_gen: int,
    output_root: str | Path = "experiments_results/moea_benchmark",
    save_every: int = 1,
    algorithm_options: dict[str, Any] | None = None,
    verbose: bool = False,
) -> RunResult:
    """Run one algorithm/problem/seed job and save all intermediate results."""

    spec = get_problem_spec(problem_name)
    run_dir = make_run_dir(output_root, spec.name, algorithm_name.lower(), seed)
    problem = create_problem(spec.name)
    pf = reference_front(problem)
    ref_point = default_ref_point(pf, reference_front=pf) if pf is not None else None

    metadata = {
        "problem": spec.name,
        "tier": spec.tier,
        "description": spec.description,
        "source": spec.source,
        "algorithm": algorithm_name,
        "seed": int(seed),
        "pop_size": int(pop_size),
        "n_gen": int(n_gen),
        "save_every": int(save_every),
        "algorithm_options": algorithm_options or {},
        "problem_meta": {
            "n_var": int(getattr(problem, "n_var", -1)),
            "n_obj": int(getattr(problem, "n_obj", -1)),
            "n_ieq_constr": int(getattr(problem, "n_ieq_constr", 0) or 0),
            "xl": _tolist(getattr(problem, "xl", None)),
            "xu": _tolist(getattr(problem, "xu", None)),
        },
        "reference_point": _tolist(ref_point),
    }
    write_json(run_dir / "metadata.json", metadata)

    if pf is not None:
        np.savetxt(run_dir / "reference_front.csv", pf, delimiter=",")

    start = perf_counter()
    final_metrics: dict[str, Any]
    status = "ok"
    try:
        algorithm = create_algorithm(
            algorithm_name,
            problem=problem,
            pop_size=pop_size,
            seed=seed,
            options=algorithm_options,
        )
        recorder = ExperimentRecorder(
            run_dir=run_dir,
            reference_front=pf,
            ref_point=ref_point,
            save_every=save_every,
        )
        result = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", n_gen),
            seed=seed,
            verbose=verbose,
            save_history=False,
            copy_algorithm=False,
            callback=recorder,
        )
        final_metrics = save_final_artifacts(run_dir, result, reference_front=pf, ref_point=ref_point)
    except Exception as exc:
        status = "failed"
        final_metrics = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "final_metrics.json", final_metrics)
    finally:
        if hasattr(problem, "close"):
            try:
                problem.close()
            except Exception:
                pass

    elapsed = perf_counter() - start
    summary = {
        "status": status,
        "elapsed_sec": elapsed,
        "final_metrics": final_metrics,
    }
    write_json(run_dir / "run_summary.json", summary)

    return RunResult(
        problem=spec.name,
        algorithm=algorithm_name,
        seed=int(seed),
        run_dir=run_dir,
        status=status,
        elapsed_sec=elapsed,
        final_metrics=final_metrics,
    )


def run_matrix(
    problems: list[str],
    algorithms: list[str],
    seeds: list[int],
    pop_size: int,
    n_gen: int,
    output_root: str | Path,
    save_every: int = 1,
    verbose: bool = False,
) -> list[RunResult]:
    """Run a Cartesian product of problems, algorithms, and seeds."""

    results: list[RunResult] = []
    for problem_name in problems:
        for algorithm_name in algorithms:
            for seed in seeds:
                results.append(
                    run_single(
                        problem_name=problem_name,
                        algorithm_name=algorithm_name,
                        seed=seed,
                        pop_size=pop_size,
                        n_gen=n_gen,
                        output_root=output_root,
                        save_every=save_every,
                        verbose=verbose,
                    )
                )
    return results


def _tolist(value) -> Any:
    if value is None:
        return None
    arr = np.asarray(value)
    return arr.tolist()
