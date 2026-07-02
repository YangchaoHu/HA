"""Surrogate-model helpers for expensive benchmark cases."""

from .dataset import EvaluationArchive
from .models import MultiOutputSurrogate, fit_surrogate
from .problem import SurrogateProblem

__all__ = ["EvaluationArchive", "MultiOutputSurrogate", "SurrogateProblem", "fit_surrogate"]

