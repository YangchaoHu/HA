"""Dataset utilities for expensive CAE evaluations."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class EvaluationArchive:
    """Append-only archive of decision vectors, objectives, and metadata."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        x: np.ndarray,
        f: np.ndarray,
        cv: float | None = None,
        meta: dict[str, object] | None = None,
    ) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        f = np.asarray(f, dtype=float).reshape(-1)
        row: dict[str, object] = {}
        row.update({f"x{i}": float(value) for i, value in enumerate(x)})
        row.update({f"f{i}": float(value) for i, value in enumerate(f)})
        if cv is not None:
            row["cv"] = float(cv)
        if meta:
            row.update(meta)

        is_new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        frame = pd.read_csv(self.path)
        x_cols = sorted([col for col in frame.columns if col.startswith("x")], key=_col_index)
        f_cols = sorted([col for col in frame.columns if col.startswith("f")], key=_col_index)
        x = frame[x_cols].to_numpy(dtype=float)
        f = frame[f_cols].to_numpy(dtype=float)
        cv = frame["cv"].to_numpy(dtype=float) if "cv" in frame.columns else None
        return x, f, cv


def _col_index(name: str) -> int:
    return int("".join(ch for ch in name if ch.isdigit()) or 0)

