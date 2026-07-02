"""Aggregate benchmark outputs into paper-ready tables and convergence plots."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRICS = ["hv", "igd", "igd_plus", "n_final_front", "elapsed_sec"]
MINIMIZE_METRICS = {"igd", "igd_plus", "elapsed_sec"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="experiments_results/moea_benchmark")
    parser.add_argument("--output-dir", default="experiments_results/moea_benchmark_paper")
    parser.add_argument("--metric", default="igd", choices=["hv", "igd", "igd_plus"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(input_root)
    if runs.empty:
        print(f"No runs found under {input_root.resolve()}")
        return 2

    runs.to_csv(output_dir / "all_runs.csv", index=False)
    summary = summarize_runs(runs)
    summary.to_csv(output_dir / "summary_by_problem_algorithm.csv", index=False)
    write_latex_table(summary, output_dir / "paper_table.tex", metric=args.metric)
    aggregate_convergence(input_root, output_dir, metric=args.metric)
    print(f"Wrote aggregated results to {output_dir.resolve()}")
    return 0


def collect_runs(input_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in input_root.rglob("final_metrics.json"):
        run_dir = metrics_path.parent
        metadata_path = run_dir / "metadata.json"
        summary_path = run_dir / "run_summary.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

        row = {
            "run_dir": str(run_dir),
            "problem": metadata.get("problem"),
            "tier": metadata.get("tier"),
            "algorithm": metadata.get("algorithm"),
            "seed": metadata.get("seed"),
            "status": run_summary.get("status", metrics.get("status")),
            "elapsed_sec": run_summary.get("elapsed_sec"),
            "source": metadata.get("source"),
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)) or value is None:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    ok = runs[runs["status"] == "ok"].copy()
    for metric in METRICS:
        if metric in ok:
            ok[metric] = pd.to_numeric(ok[metric], errors="coerce")

    grouped = ok.groupby(["tier", "problem", "algorithm"], dropna=False)
    pieces = []
    for metric in METRICS:
        if metric not in ok:
            continue
        stat = grouped[metric].agg(["mean", "std", "count"]).reset_index()
        stat = stat.rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "count": f"{metric}_count",
            }
        )
        pieces.append(stat)

    if not pieces:
        return pd.DataFrame()

    summary = pieces[0]
    for item in pieces[1:]:
        summary = summary.merge(item, on=["tier", "problem", "algorithm"], how="outer")

    for metric in ["hv", "igd", "igd_plus"]:
        col = f"{metric}_mean"
        if col in summary:
            ascending = metric in MINIMIZE_METRICS
            summary[f"{metric}_rank"] = summary.groupby("problem")[col].rank(
                method="min", ascending=ascending, na_option="bottom"
            )

    return summary.sort_values(["tier", "problem", "algorithm"]).reset_index(drop=True)


def write_latex_table(summary: pd.DataFrame, output: Path, metric: str) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    rank_col = f"{metric}_rank"
    if mean_col not in summary:
        output.write_text("% No metric available.\n", encoding="utf-8")
        return

    lines = [
        "\\begin{tabular}{lllrr}",
        "\\toprule",
        "Tier & Problem & Algorithm & Metric & Rank \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        mean = _fmt(row.get(mean_col))
        std = _fmt(row.get(std_col))
        rank = row.get(rank_col, math.nan)
        metric_text = f"{mean} $\\pm$ {std}" if std != "" else mean
        rank_text = "" if pd.isna(rank) else str(int(rank))
        lines.append(
            f"{_tex(row.get('tier'))} & {_tex(row.get('problem'))} & "
            f"{_tex(row.get('algorithm'))} & {metric_text} & {rank_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def aggregate_convergence(input_root: Path, output_dir: Path, metric: str) -> None:
    frames: list[pd.DataFrame] = []
    for metrics_path in input_root.rglob("metrics_by_generation.csv"):
        run_dir = metrics_path.parent
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(metrics_path)
        frame["problem"] = metadata.get("problem")
        frame["algorithm"] = metadata.get("algorithm")
        frame["seed"] = metadata.get("seed")
        frames.append(frame)

    if not frames:
        return

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output_dir / "convergence_all.csv", index=False)
    metric_col = metric
    if metric_col not in data:
        return

    data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
    grouped = (
        data.groupby(["problem", "algorithm", "generation"], dropna=False)[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.to_csv(output_dir / f"convergence_{metric}.csv", index=False)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for problem, sub in grouped.groupby("problem"):
        plt.figure(figsize=(7, 4.5))
        for algorithm, line in sub.groupby("algorithm"):
            line = line.sort_values("generation")
            plt.plot(line["generation"], line["mean"], label=str(algorithm), linewidth=1.8)
            if "std" in line:
                lo = line["mean"] - line["std"].fillna(0)
                hi = line["mean"] + line["std"].fillna(0)
                plt.fill_between(line["generation"], lo, hi, alpha=0.15)
        plt.xlabel("Generation")
        plt.ylabel(metric.upper())
        plt.title(str(problem))
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{problem}_{metric}_convergence.png", dpi=200)
        plt.close()


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.4f}"


def _tex(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    return text.replace("_", "\\_")


if __name__ == "__main__":
    raise SystemExit(main())

