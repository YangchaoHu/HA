"""Problem catalog for the three-level MOEA benchmark.

The catalog deliberately keeps the external benchmark implementations behind a
thin factory layer. This makes it possible to run official pymoo problems,
project-local problems, and expensive MAPDL-backed cases through the same
experiment pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions


ProblemFactory = Callable[[], object]


@dataclass(frozen=True)
class ProblemSpec:
    """Metadata and factory for a benchmark problem."""

    name: str
    tier: str
    factory: ProblemFactory
    description: str
    source: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    expensive: bool = False
    enabled_by_default: bool = True


def _pymoo_problem(name: str, **kwargs) -> ProblemFactory:
    return lambda: get_problem(name, **kwargs)


_SPECS: tuple[ProblemSpec, ...] = (
    ProblemSpec(
        name="zdt1",
        tier="math",
        factory=_pymoo_problem("zdt1"),
        description="ZDT1 two-objective convex Pareto-front benchmark.",
        source="https://pymoo.org/problems/multi/zdt.html",
        tags=("zdt", "two_objective", "unconstrained"),
    ),
    ProblemSpec(
        name="zdt2",
        tier="math",
        factory=_pymoo_problem("zdt2"),
        description="ZDT2 two-objective non-convex Pareto-front benchmark.",
        source="https://pymoo.org/problems/multi/zdt.html",
        tags=("zdt", "two_objective", "unconstrained"),
    ),
    ProblemSpec(
        name="zdt3",
        tier="math",
        factory=_pymoo_problem("zdt3"),
        description="ZDT3 two-objective discontinuous Pareto-front benchmark.",
        source="https://pymoo.org/problems/multi/zdt.html",
        tags=("zdt", "two_objective", "discontinuous"),
    ),
    ProblemSpec(
        name="zdt4",
        tier="math",
        factory=_pymoo_problem("zdt4"),
        description="ZDT4 multimodal two-objective benchmark.",
        source="https://pymoo.org/problems/multi/zdt.html",
        tags=("zdt", "two_objective", "multimodal"),
    ),
    ProblemSpec(
        name="zdt6",
        tier="math",
        factory=_pymoo_problem("zdt6"),
        description="ZDT6 biased-density two-objective benchmark.",
        source="https://pymoo.org/problems/multi/zdt.html",
        tags=("zdt", "two_objective", "biased"),
    ),
    ProblemSpec(
        name="dtlz1",
        tier="math",
        factory=_pymoo_problem("dtlz1", n_obj=3),
        description="DTLZ1 scalable many-objective benchmark with local fronts.",
        source="https://pymoo.org/problems/many/dtlz.html",
        tags=("dtlz", "three_objective", "scalable"),
    ),
    ProblemSpec(
        name="dtlz2",
        tier="math",
        factory=_pymoo_problem("dtlz2", n_obj=3),
        description="DTLZ2 spherical Pareto-front many-objective benchmark.",
        source="https://pymoo.org/problems/many/dtlz.html",
        tags=("dtlz", "three_objective", "scalable"),
    ),
    ProblemSpec(
        name="dtlz3",
        tier="math",
        factory=_pymoo_problem("dtlz3", n_obj=3),
        description="DTLZ3 multimodal many-objective benchmark.",
        source="https://pymoo.org/problems/many/dtlz.html",
        tags=("dtlz", "three_objective", "multimodal"),
    ),
    ProblemSpec(
        name="dtlz4",
        tier="math",
        factory=_pymoo_problem("dtlz4", n_obj=3),
        description="DTLZ4 biased many-objective benchmark.",
        source="https://pymoo.org/problems/many/dtlz.html",
        tags=("dtlz", "three_objective", "biased"),
    ),
    ProblemSpec(
        name="welded_beam",
        tier="engineering",
        factory=_pymoo_problem("welded_beam"),
        description="Official pymoo welded beam design benchmark.",
        source="https://pymoo.org/problems/multi/welded_beam.html",
        tags=("official_pymoo", "constrained", "mechanical_design"),
    ),
    ProblemSpec(
        name="truss2d",
        tier="engineering",
        factory=_pymoo_problem("truss2d"),
        description="Official pymoo two-bar truss design benchmark.",
        source="https://pymoo.org/problems/multi/truss2d.html",
        tags=("official_pymoo", "constrained", "structural_design"),
    ),
    ProblemSpec(
        name="bnh",
        tier="constrained_math",
        factory=_pymoo_problem("bnh"),
        description="BNH constrained analytical two-objective benchmark.",
        source="https://pymoo.org/problems/index.html",
        tags=("official_pymoo", "constrained", "analytical"),
    ),
    ProblemSpec(
        name="tnk",
        tier="constrained_math",
        factory=_pymoo_problem("tnk"),
        description="TNK constrained analytical two-objective benchmark.",
        source="https://pymoo.org/problems/index.html",
        tags=("official_pymoo", "constrained", "analytical"),
    ),
    ProblemSpec(
        name="osy",
        tier="constrained_math",
        factory=_pymoo_problem("osy"),
        description="OSY constrained analytical two-objective benchmark.",
        source="https://pymoo.org/problems/index.html",
        tags=("official_pymoo", "constrained", "analytical"),
    ),
    ProblemSpec(
        name="carside",
        tier="engineering",
        factory=_pymoo_problem("carside"),
        description="Car side-impact constrained three-objective benchmark.",
        source="https://pymoo.org/problems/index.html",
        tags=("official_pymoo", "constrained", "crashworthiness"),
    ),
    ProblemSpec(
        name="mapdl_cantilever_beam",
        tier="cae",
        factory=lambda: _create_mapdl_problem("mapdl_cantilever_beam"),
        description=(
            "Public PyMAPDL BEAM188 rectangular-beam example adapted into a "
            "real MAPDL multi-objective cantilever case; objectives are "
            "structural volume and tip displacement."
        ),
        source="https://mapdl.docs.pyansys.com/version/stable/examples/gallery_examples/00-mapdl-examples/modal_beam.html",
        tags=("ansys", "pymapdl", "finite_element", "expensive"),
        expensive=True,
        enabled_by_default=False,
    ),
    ProblemSpec(
        name="mapdl_simply_supported_beam",
        tier="cae",
        factory=lambda: _create_mapdl_problem("mapdl_simply_supported_beam"),
        description="Real MAPDL BEAM188 simply supported beam with mid-span load.",
        source="https://mapdl.docs.pyansys.com/version/stable/examples/gallery_examples/00-mapdl-examples/modal_beam.html",
        tags=("ansys", "pymapdl", "finite_element", "beam", "expensive"),
        expensive=True,
        enabled_by_default=False,
    ),
    ProblemSpec(
        name="mapdl_portal_frame",
        tier="cae",
        factory=lambda: _create_mapdl_problem("mapdl_portal_frame"),
        description="Real MAPDL BEAM188 portal frame under combined lateral and vertical load.",
        source="https://mapdl.docs.pyansys.com/version/stable/examples/gallery_examples/00-mapdl-examples/modal_beam.html",
        tags=("ansys", "pymapdl", "finite_element", "frame", "expensive"),
        expensive=True,
        enabled_by_default=False,
    ),
    ProblemSpec(
        name="mapdl_two_bar_truss",
        tier="cae",
        factory=lambda: _create_mapdl_problem("mapdl_two_bar_truss"),
        description="Real MAPDL LINK180 two-bar truss with variable areas and height.",
        source="https://mapdl.docs.pyansys.com/version/stable/examples/gallery_examples/00-mapdl-examples/index.html",
        tags=("ansys", "pymapdl", "finite_element", "truss", "expensive"),
        expensive=True,
        enabled_by_default=False,
    ),
    ProblemSpec(
        name="mapdl_plane_stress_plate",
        tier="cae",
        factory=lambda: _create_mapdl_problem("mapdl_plane_stress_plate"),
        description="Real MAPDL PLANE182 clamped plate in tension.",
        source="https://mapdl.docs.pyansys.com/version/stable/examples/gallery_examples/00-mapdl-examples/2d_plate_with_a_hole.html",
        tags=("ansys", "pymapdl", "finite_element", "plane_stress", "expensive"),
        expensive=True,
        enabled_by_default=False,
    ),
)


def _create_mapdl_problem(name: str):
    from moea_benchmark.cae.mapdl_structural_cases import create_mapdl_problem

    return create_mapdl_problem(name)


def list_problem_specs(
    tiers: Iterable[str] | None = None,
    include_expensive: bool = False,
) -> list[ProblemSpec]:
    """Return problem specs filtered by tier and expense."""

    tier_set = {tier.lower() for tier in tiers} if tiers else None
    specs: list[ProblemSpec] = []
    for spec in _SPECS:
        if tier_set is not None and spec.tier.lower() not in tier_set:
            continue
        if spec.expensive and not include_expensive:
            continue
        specs.append(spec)
    return specs


def get_problem_spec(name: str) -> ProblemSpec:
    """Return a problem spec by name."""

    key = name.lower()
    for spec in _SPECS:
        if spec.name == key:
            return spec
    available = ", ".join(spec.name for spec in _SPECS)
    raise KeyError(f"Unknown problem '{name}'. Available problems: {available}")


def create_problem(name: str):
    """Create a fresh problem instance."""

    return get_problem_spec(name).factory()


def reference_front(problem, n_points: int = 1000) -> np.ndarray | None:
    """Best-effort reference Pareto-front extraction.

    Some pymoo problems generate the front directly. Many-objective DTLZ
    problems expect a set of reference directions, so this helper supplies one.
    Expensive simulation cases normally have no known analytical front.
    """

    if not hasattr(problem, "pareto_front"):
        return None

    for kwargs in (
        {"n_pareto_points": n_points},
        {},
    ):
        try:
            pf = problem.pareto_front(**kwargs)
            if pf is not None:
                return np.asarray(pf, dtype=float)
        except TypeError:
            continue
        except Exception:
            break

    n_obj = int(getattr(problem, "n_obj", 0) or 0)
    if n_obj >= 3:
        try:
            partitions = 12 if n_obj <= 3 else 8
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=partitions)
            pf = problem.pareto_front(ref_dirs)
            if pf is not None:
                return np.asarray(pf, dtype=float)
        except Exception:
            return None

    return None
