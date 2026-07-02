"""ANSYS and PyMAPDL environment validation utilities."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnsysEnvironmentReport:
    """Serializable ANSYS environment report."""

    python_executable: str
    python_version: str
    platform: str
    conda_default_env: str | None
    inferred_conda_env: str | None
    awp_roots: dict[str, str] = field(default_factory=dict)
    license_file: str | None = None
    mapdl_executables: list[str] = field(default_factory=list)
    python_packages: dict[str, str | None] = field(default_factory=dict)
    can_import_pymapdl: bool = False
    launch_attempted: bool = False
    launch_ok: bool = False
    launch_message: str = ""

    @property
    def ready_for_mapdl(self) -> bool:
        return self.can_import_pymapdl and bool(self.mapdl_executables)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ready_for_mapdl"] = self.ready_for_mapdl
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name, fromlist=["__version__"])
        return str(getattr(module, "__version__", "installed"))
    except Exception:
        return None


def _discover_awp_roots() -> dict[str, str]:
    roots: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.upper().startswith("AWP_ROOT") and value:
            roots[key] = value
    return dict(sorted(roots.items()))


def _discover_mapdl_executables(awp_roots: dict[str, str]) -> list[str]:
    found: list[str] = []

    for exe_name in ("MAPDL.exe", "MAPDL252.exe", "ANSYS.exe", "ANSYS252.exe"):
        exe = shutil.which(exe_name)
        if exe:
            found.append(str(Path(exe).resolve()))

    for root in awp_roots.values():
        root_path = Path(root)
        candidates = [
            root_path / "ANSYS" / "bin" / "winx64" / "MAPDL.exe",
            root_path / "ANSYS" / "bin" / "winx64" / "MAPDL252.exe",
            root_path / "ANSYS" / "bin" / "winx64" / "ANSYS.exe",
            root_path / "ANSYS" / "bin" / "winx64" / "ANSYS252.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                found.append(str(candidate.resolve()))

    unique: list[str] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


def inspect_ansys_environment(run_launch_smoke: bool = False) -> AnsysEnvironmentReport:
    """Inspect local Python, ANSYS, license, and PyMAPDL readiness."""

    awp_roots = _discover_awp_roots()
    report = AnsysEnvironmentReport(
        python_executable=sys.executable,
        python_version=sys.version.replace("\n", " "),
        platform=platform.platform(),
        conda_default_env=os.environ.get("CONDA_DEFAULT_ENV"),
        inferred_conda_env=_infer_conda_env_name(sys.executable),
        awp_roots=awp_roots,
        license_file=os.environ.get("ANSYSLMD_LICENSE_FILE"),
        mapdl_executables=_discover_mapdl_executables(awp_roots),
        python_packages={
            "ansys": _package_version("ansys"),
            "ansys.mapdl.core": _package_version("ansys.mapdl.core"),
            "pymoo": _package_version("pymoo"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scipy": _package_version("scipy"),
            "pyvista": _package_version("pyvista"),
        },
        can_import_pymapdl=importlib.util.find_spec("ansys.mapdl.core") is not None,
    )

    if run_launch_smoke:
        _run_launch_smoke(report)

    return report


def _infer_conda_env_name(python_executable: str) -> str | None:
    parts = Path(python_executable).parts
    lowered = [part.lower() for part in parts]
    if "envs" in lowered:
        idx = lowered.index("envs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return os.environ.get("CONDA_DEFAULT_ENV")


def _run_launch_smoke(report: AnsysEnvironmentReport) -> None:
    report.launch_attempted = True
    if not report.ready_for_mapdl:
        report.launch_message = "PyMAPDL import or MAPDL executable discovery failed."
        return

    try:
        from ansys.mapdl.core import launch_mapdl

        with tempfile.TemporaryDirectory(prefix="mapdl_smoke_") as tmp:
            mapdl = launch_mapdl(
                exec_file=report.mapdl_executables[0],
                run_location=tmp,
                override=True,
                loglevel="ERROR",
                start_timeout=90,
            )
            try:
                version = str(getattr(mapdl, "version", "unknown"))
                mapdl.prep7()
                mapdl.finish()
                report.launch_ok = True
                report.launch_message = f"MAPDL launched successfully; version={version}"
            finally:
                mapdl.exit()
    except Exception as exc:
        report.launch_ok = False
        report.launch_message = f"{type(exc).__name__}: {exc}"


def write_environment_report(path: str | Path, run_launch_smoke: bool = False) -> AnsysEnvironmentReport:
    """Inspect and write the ANSYS environment report as JSON."""

    report = inspect_ansys_environment(run_launch_smoke=run_launch_smoke)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")
    return report
