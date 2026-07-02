"""Backward-compatible imports for the original MAPDL cantilever beam case."""

from .mapdl_structural_cases import (
    MAPDLCantileverBeamProblem,
    analytical_cantilever_displacement,
)

__all__ = ["MAPDLCantileverBeamProblem", "analytical_cantilever_displacement"]

