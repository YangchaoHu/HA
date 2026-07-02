"""pymoo problem wrapper backed by a fitted surrogate."""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

from .models import MultiOutputSurrogate


class SurrogateProblem(Problem):
    """Vectorized pymoo problem that evaluates a fitted objective surrogate."""

    def __init__(
        self,
        surrogate: MultiOutputSurrogate,
        xl: np.ndarray,
        xu: np.ndarray,
        n_obj: int,
    ) -> None:
        self.surrogate = surrogate
        super().__init__(
            n_var=len(xl),
            n_obj=int(n_obj),
            n_ieq_constr=0,
            xl=np.asarray(xl, dtype=float),
            xu=np.asarray(xu, dtype=float),
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        out["F"] = self.surrogate.predict(x)

