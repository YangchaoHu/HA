"""Algorithm factory used by the benchmark runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions


_ALGORITHM_NAMES = ("nsga2", "nsga3", "unsga3", "moead", "sms-emoa", "ha-nsga3")


def list_algorithms() -> tuple[str, ...]:
    return _ALGORITHM_NAMES


def reference_directions(n_obj: int, pop_size: int, seed: int | None = None) -> np.ndarray:
    """Create reference directions for decomposition/reference-vector algorithms."""

    if n_obj == 2:
        return get_reference_directions("das-dennis", n_obj, n_partitions=max(1, pop_size - 1))
    return get_reference_directions("energy", n_obj, pop_size, seed=seed)


def create_algorithm(
    name: str,
    problem,
    pop_size: int,
    seed: int | None = None,
    options: Mapping[str, Any] | None = None,
):
    """Create a pymoo-compatible algorithm instance."""

    key = name.lower()
    opts = dict(options or {})
    n_obj = int(getattr(problem, "n_obj", 2))

    if key == "nsga2":
        return NSGA2(pop_size=pop_size, **opts)

    if key == "nsga3":
        ref_dirs = opts.pop("ref_dirs", None)
        if ref_dirs is None:
            ref_dirs = reference_directions(n_obj, pop_size, seed=seed)
        return NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, **opts)

    if key == "unsga3":
        ref_dirs = opts.pop("ref_dirs", None)
        if ref_dirs is None:
            ref_dirs = reference_directions(n_obj, pop_size, seed=seed)
        return UNSGA3(ref_dirs=ref_dirs, pop_size=pop_size, **opts)

    if key == "moead":
        ref_dirs = opts.pop("ref_dirs", None)
        if ref_dirs is None:
            ref_dirs = reference_directions(n_obj, pop_size, seed=seed)
        return MOEAD(ref_dirs=ref_dirs, **opts)

    if key == "sms-emoa":
        return SMSEMOA(pop_size=pop_size, **opts)

    if key == "ha-nsga3":
        try:
            from ha_nsga3 import HA_NSGA3
        except Exception as exc:
            raise RuntimeError(
                "Could not import project-local HA_NSGA3. "
                "Check that ha_nsga3.py and its dependencies are available."
            ) from exc

        defaults: dict[str, Any] = {
            "method": "rbf",
            "pop_size": pop_size,
            "niche_num": 4,
            "mutation_rate": 0.2,
            "activate_method": True,
            "niche_strategy": "ref_dirs",
            "scalarization": "pbi",
            "pbi_theta": 1.0,
            "seed": seed,
        }
        defaults.update(opts)
        return HA_NSGA3(**defaults)

    raise KeyError(f"Unknown algorithm '{name}'. Available algorithms: {', '.join(_ALGORITHM_NAMES)}")

