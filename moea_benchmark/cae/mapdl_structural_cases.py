"""Real MAPDL-backed structural benchmark problems.

Each class is a small but genuine finite-element model solved by ANSYS MAPDL
through PyMAPDL. The cases are intentionally compact so smoke tests can run on
a desktop license, while still exercising different structural model families:
beam, frame, truss, and 2D plane-stress continuum elements.
"""

from __future__ import annotations

import csv
import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .ansys_env import inspect_ansys_environment


class MAPDLStructuralProblem(ElementwiseProblem, ABC):
    """Base class for expensive MAPDL-backed structural MO problems."""

    case_name = "mapdl_structural_problem"
    objective_names = ("volume", "max_displacement")

    def __init__(
        self,
        n_var: int,
        xl: np.ndarray,
        xu: np.ndarray,
        work_root: str | Path | None = None,
        keep_mapdl_alive: bool = True,
        failure_value: float = 1.0e6,
    ) -> None:
        self.keep_mapdl_alive = bool(keep_mapdl_alive)
        self.failure_value = float(failure_value)
        self._mapdl = None
        self._eval_id = 0
        self.last_status = "not_run"
        self.last_message = ""
        self.last_row: dict[str, Any] | None = None

        root = Path(work_root) if work_root else Path("experiments_results") / "mapdl_work"
        self.work_root = root.resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.eval_log = self.work_root / f"{self.case_name}_evaluations.csv"

        super().__init__(
            n_var=int(n_var),
            n_obj=2,
            n_ieq_constr=0,
            xl=np.asarray(xl, dtype=float),
            xu=np.asarray(xu, dtype=float),
        )

    def default_x(self) -> np.ndarray:
        """Return a robust mid-range smoke-test design."""

        return (np.asarray(self.xl, dtype=float) + np.asarray(self.xu, dtype=float)) / 2.0

    def evaluate_design(self, x, raise_on_failure: bool = False) -> np.ndarray:
        """Evaluate one design and append a persistent evaluation log row."""

        start = time.perf_counter()
        x = np.asarray(x, dtype=float).reshape(-1)
        self._eval_id += 1

        try:
            f, meta = self._solve_design(x)
            status = "ok"
            message = ""
        except Exception as exc:
            f = np.full(self.n_obj, self.failure_value, dtype=float)
            meta = {}
            status = "failed"
            message = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - start
        row: dict[str, Any] = {
            "eval_id": self._eval_id,
            "case": self.case_name,
            "elapsed_sec": elapsed,
            "status": status,
            "message": message,
        }
        row.update({f"x{i}": float(value) for i, value in enumerate(x)})
        row.update({f"f{i}_{name}": float(value) for i, (name, value) in enumerate(zip(self.objective_names, f))})
        row.update(meta)

        self.last_status = status
        self.last_message = message
        self.last_row = row
        self._append_eval_log(row)

        if raise_on_failure and status != "ok":
            raise RuntimeError(message)
        return np.asarray(f, dtype=float)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        out["F"] = self.evaluate_design(x, raise_on_failure=False)

    @abstractmethod
    def _solve_design(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Solve one design and return objective vector plus scalar metadata."""

    def _get_mapdl(self):
        if self.keep_mapdl_alive and self._mapdl is not None:
            return self._mapdl

        report = inspect_ansys_environment(run_launch_smoke=False)
        if not report.ready_for_mapdl:
            raise RuntimeError("ANSYS MAPDL is not ready. Run scripts/check_ansys_env.py for details.")

        from ansys.mapdl.core import launch_mapdl

        run_location = Path(tempfile.mkdtemp(prefix=f"{self.case_name}_", dir=str(self.work_root)))
        mapdl = launch_mapdl(
            exec_file=report.mapdl_executables[0],
            run_location=str(run_location),
            override=True,
            loglevel=os.environ.get("MOEA_MAPDL_LOGLEVEL", "ERROR"),
            start_timeout=int(os.environ.get("MOEA_MAPDL_START_TIMEOUT", "120")),
        )
        if self.keep_mapdl_alive:
            self._mapdl = mapdl
        return mapdl

    def _prepare_static_model(self):
        mapdl = self._get_mapdl()
        mapdl.clear()
        mapdl.prep7()
        mapdl.units("SI")
        mapdl.mp("EX", 1, 210e9)
        mapdl.mp("PRXY", 1, 0.3)
        mapdl.mp("DENS", 1, 7850.0)
        return mapdl

    @staticmethod
    def _solve_static(mapdl) -> None:
        mapdl.allsel(mute=True)
        mapdl.run("/SOLU")
        mapdl.antype("STATIC")
        mapdl.solve()
        mapdl.finish()
        mapdl.post1()
        mapdl.set(1, 1)

    @staticmethod
    def _max_abs_disp(mapdl, component: str) -> float:
        values = np.asarray(mapdl.post_processing.nodal_displacement(component), dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"MAPDL returned invalid {component} displacement results.")
        return float(np.nanmax(np.abs(values)))

    @staticmethod
    def _max_planar_disp(mapdl) -> float:
        ux = np.asarray(mapdl.post_processing.nodal_displacement("X"), dtype=float)
        uy = np.asarray(mapdl.post_processing.nodal_displacement("Y"), dtype=float)
        if ux.size == 0 or uy.size == 0 or ux.size != uy.size:
            raise RuntimeError("MAPDL returned invalid planar displacement results.")
        return float(np.nanmax(np.sqrt(ux**2 + uy**2)))

    def _append_eval_log(self, row: dict[str, Any]) -> None:
        is_new = not self.eval_log.exists()
        with self.eval_log.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def close(self) -> None:
        if self._mapdl is not None:
            try:
                self._mapdl.exit()
            finally:
                self._mapdl = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MAPDLCantileverBeamProblem(MAPDLStructuralProblem):
    """BEAM188 cantilever beam: minimize volume and vertical deflection."""

    case_name = "mapdl_cantilever_beam"

    def __init__(self, work_root: str | Path | None = None, keep_mapdl_alive: bool = True) -> None:
        self.length = 1.0
        self.force_y = -1000.0
        self.n_elem = 24
        super().__init__(
            n_var=2,
            xl=np.array([0.02, 0.02]),
            xu=np.array([0.12, 0.20]),
            work_root=work_root,
            keep_mapdl_alive=keep_mapdl_alive,
        )

    def _solve_design(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        width, height = map(float, x)
        mapdl = self._prepare_static_model()
        mapdl.et(1, "BEAM188")
        mapdl.sectype(1, "BEAM", "RECT")
        mapdl.secdata(width, height)
        mapdl.k(1, 0.0, 0.0, 0.0)
        mapdl.k(2, self.length, 0.0, 0.0)
        mapdl.l(1, 2)
        mapdl.lesize("ALL", "", "", self.n_elem)
        mapdl.lmesh("ALL")
        mapdl.nsel("S", "LOC", "X", 0.0)
        mapdl.d("ALL", "ALL")
        mapdl.nsel("S", "LOC", "X", self.length)
        mapdl.f("ALL", "FY", self.force_y)
        self._solve_static(mapdl)
        volume = width * height * self.length
        disp = self._max_abs_disp(mapdl, "Y")
        return np.array([volume, disp], dtype=float), {"width": width, "height": height}


class MAPDLSimplySupportedBeamProblem(MAPDLStructuralProblem):
    """BEAM188 simply supported beam with a mid-span transverse load."""

    case_name = "mapdl_simply_supported_beam"

    def __init__(self, work_root: str | Path | None = None, keep_mapdl_alive: bool = True) -> None:
        self.length = 1.2
        self.force_y = -1500.0
        self.n_elem = 24
        super().__init__(
            n_var=2,
            xl=np.array([0.02, 0.02]),
            xu=np.array([0.14, 0.22]),
            work_root=work_root,
            keep_mapdl_alive=keep_mapdl_alive,
        )

    def _solve_design(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        width, height = map(float, x)
        mapdl = self._prepare_static_model()
        mapdl.et(1, "BEAM188")
        mapdl.sectype(1, "BEAM", "RECT")
        mapdl.secdata(width, height)
        mapdl.k(1, 0.0, 0.0, 0.0)
        mapdl.k(2, self.length, 0.0, 0.0)
        mapdl.l(1, 2)
        mapdl.lesize("ALL", "", "", self.n_elem)
        mapdl.lmesh("ALL")

        mapdl.nsel("S", "LOC", "X", 0.0)
        mapdl.d("ALL", "UX")
        mapdl.d("ALL", "UY")
        mapdl.d("ALL", "UZ")
        mapdl.d("ALL", "ROTX")
        mapdl.nsel("S", "LOC", "X", self.length)
        mapdl.d("ALL", "UY")
        mapdl.d("ALL", "UZ")
        mapdl.nsel("S", "LOC", "X", self.length / 2.0)
        mapdl.f("ALL", "FY", self.force_y)

        self._solve_static(mapdl)
        volume = width * height * self.length
        disp = self._max_abs_disp(mapdl, "Y")
        return np.array([volume, disp], dtype=float), {"width": width, "height": height}


class MAPDLPortalFrameProblem(MAPDLStructuralProblem):
    """BEAM188 portal frame under combined vertical and lateral load."""

    case_name = "mapdl_portal_frame"

    def __init__(self, work_root: str | Path | None = None, keep_mapdl_alive: bool = True) -> None:
        self.span = 1.2
        self.height_frame = 0.8
        self.force_x = 500.0
        self.force_y = -1000.0
        self.n_elem_per_member = 8
        super().__init__(
            n_var=2,
            xl=np.array([0.025, 0.025]),
            xu=np.array([0.14, 0.22]),
            work_root=work_root,
            keep_mapdl_alive=keep_mapdl_alive,
        )

    def _solve_design(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        width, depth = map(float, x)
        mapdl = self._prepare_static_model()
        mapdl.et(1, "BEAM188")
        mapdl.sectype(1, "BEAM", "RECT")
        mapdl.secdata(width, depth)
        mapdl.k(1, 0.0, 0.0, 0.0)
        mapdl.k(2, 0.0, self.height_frame, 0.0)
        mapdl.k(3, self.span, self.height_frame, 0.0)
        mapdl.k(4, self.span, 0.0, 0.0)
        mapdl.l(1, 2)
        mapdl.l(2, 3)
        mapdl.l(3, 4)
        mapdl.lesize("ALL", "", "", self.n_elem_per_member)
        mapdl.lmesh("ALL")

        mapdl.nsel("S", "LOC", "Y", 0.0)
        mapdl.d("ALL", "ALL")
        mapdl.nsel("S", "LOC", "X", self.span)
        mapdl.nsel("R", "LOC", "Y", self.height_frame)
        mapdl.f("ALL", "FX", self.force_x)
        mapdl.f("ALL", "FY", self.force_y)

        self._solve_static(mapdl)
        volume = width * depth * (2.0 * self.height_frame + self.span)
        disp = self._max_planar_disp(mapdl)
        return np.array([volume, disp], dtype=float), {"width": width, "depth": depth}


class MAPDLTwoBarTrussProblem(MAPDLStructuralProblem):
    """LINK180 two-bar truss with variable member areas and height."""

    case_name = "mapdl_two_bar_truss"

    def __init__(self, work_root: str | Path | None = None, keep_mapdl_alive: bool = True) -> None:
        self.span = 1.0
        self.force_y = -2000.0
        super().__init__(
            n_var=3,
            xl=np.array([1.0e-4, 1.0e-4, 0.35]),
            xu=np.array([5.0e-3, 5.0e-3, 1.20]),
            work_root=work_root,
            keep_mapdl_alive=keep_mapdl_alive,
        )

    def _solve_design(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        area_left, area_right, height = map(float, x)
        mapdl = self._prepare_static_model()
        mapdl.et(1, "LINK180")
        mapdl.sectype(1, "LINK")
        mapdl.secdata(area_left)
        mapdl.sectype(2, "LINK")
        mapdl.secdata(area_right)
        mapdl.n(1, 0.0, 0.0, 0.0)
        mapdl.n(2, self.span, 0.0, 0.0)
        mapdl.n(3, self.span / 2.0, height, 0.0)
        mapdl.type(1)
        mapdl.mat(1)
        mapdl.secnum(1)
        mapdl.e(1, 3)
        mapdl.secnum(2)
        mapdl.e(3, 2)

        mapdl.nsel("S", "NODE", "", 1)
        mapdl.nsel("A", "NODE", "", 2)
        mapdl.d("ALL", "ALL")
        mapdl.nsel("S", "NODE", "", 3)
        mapdl.d("ALL", "UZ")
        mapdl.f("ALL", "FY", self.force_y)

        self._solve_static(mapdl)
        member_len = float(np.sqrt((self.span / 2.0) ** 2 + height**2))
        volume = area_left * member_len + area_right * member_len
        disp = self._max_abs_disp(mapdl, "Y")
        return (
            np.array([volume, disp], dtype=float),
            {"area_left": area_left, "area_right": area_right, "height": height},
        )


class MAPDLPlaneStressPlateProblem(MAPDLStructuralProblem):
    """PLANE182 clamped rectangular plate in tension."""

    case_name = "mapdl_plane_stress_plate"

    def __init__(self, work_root: str | Path | None = None, keep_mapdl_alive: bool = True) -> None:
        self.length = 1.0
        self.force_x = 5000.0
        super().__init__(
            n_var=2,
            xl=np.array([0.15, 0.005]),
            xu=np.array([0.60, 0.030]),
            work_root=work_root,
            keep_mapdl_alive=keep_mapdl_alive,
        )

    def _solve_design(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        plate_height, thickness = map(float, x)
        mapdl = self._prepare_static_model()
        mapdl.et(1, "PLANE182")
        mapdl.keyopt(1, 3, 3)
        mapdl.r(1, thickness)
        mapdl.blc4(0.0, -plate_height / 2.0, self.length, plate_height)
        mapdl.esize(min(self.length / 24.0, plate_height / 8.0))
        mapdl.amesh("ALL")

        mapdl.nsel("S", "LOC", "X", 0.0)
        mapdl.d("ALL", "ALL")
        mapdl.nsel("S", "LOC", "X", self.length)
        n_right = max(1, int(round(float(mapdl.get("NRIGHT", "NODE", 0, "COUNT")))))
        mapdl.f("ALL", "FX", self.force_x / n_right)

        self._solve_static(mapdl)
        volume = self.length * plate_height * thickness
        disp = self._max_abs_disp(mapdl, "X")
        return (
            np.array([volume, disp], dtype=float),
            {"plate_height": plate_height, "thickness": thickness, "right_nodes": n_right},
        )


def analytical_cantilever_displacement(
    width: float,
    height: float,
    length: float = 1.0,
    force_y: float = -1000.0,
    young_modulus: float = 210e9,
) -> float:
    """Euler-Bernoulli tip displacement estimate for cantilever sanity checks."""

    inertia = width * height**3 / 12.0
    return abs(force_y) * length**3 / (3.0 * young_modulus * inertia)


CAE_PROBLEM_CLASSES = {
    "mapdl_cantilever_beam": MAPDLCantileverBeamProblem,
    "mapdl_simply_supported_beam": MAPDLSimplySupportedBeamProblem,
    "mapdl_portal_frame": MAPDLPortalFrameProblem,
    "mapdl_two_bar_truss": MAPDLTwoBarTrussProblem,
    "mapdl_plane_stress_plate": MAPDLPlaneStressPlateProblem,
}


def create_mapdl_problem(name: str, **kwargs) -> MAPDLStructuralProblem:
    key = name.lower()
    if key not in CAE_PROBLEM_CLASSES:
        available = ", ".join(sorted(CAE_PROBLEM_CLASSES))
        raise KeyError(f"Unknown MAPDL problem '{name}'. Available: {available}")
    return CAE_PROBLEM_CLASSES[key](**kwargs)
