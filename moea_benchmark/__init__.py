"""Multi-level benchmark scaffold for multi-objective evolutionary algorithms."""

from .benchmarks.catalog import create_problem, list_problem_specs

__all__ = ["create_problem", "list_problem_specs"]

