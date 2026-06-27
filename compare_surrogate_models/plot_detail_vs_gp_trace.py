"""
单图对比：
- QuickSimu1_gp_seed_2026/2027/2028.csv：按 generation 对齐，对 fes、volume 取三 seed 均值后
  以 FES 为横轴、均值为纵轴（volume 即 f_real）。
- QuickSimu1_surrogate_detail.csv：横轴 sample_count，纵轴 f_real（按 model_type 跨 seed 均值）。
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent

_DEFAULT_GP_TRACES = (
    ROOT / "results_ha" / "QuickSimu1_gp_seed_2026.csv",
    ROOT / "results_ha" / "QuickSimu1_gp_seed_2027.csv",
    ROOT / "results_ha" / "QuickSimu1_gp_seed_2028.csv",
)


def _configure_matplotlib_fonts(preferred: str | None = None) -> str:
    import matplotlib
    from matplotlib import font_manager

    all_names = [f.name for f in font_manager.fontManager.ttflist]
    name_set = set(all_names)

    def first_substring(keyword: str) -> str | None:
        kw = keyword.lower()
        for n in all_names:
            if kw in n.lower():
                return n
        return None

    chosen: str | None = None
    if preferred and preferred in name_set:
        chosen = preferred
    elif preferred:
        hit = first_substring(preferred)
        if hit:
            chosen = hit

    if chosen is None:
        for exact in (
            "Microsoft YaHei",
            "Microsoft YaHei UI",
            "SimHei",
            "SimSun",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
        ):
            if exact in name_set:
                chosen = exact
                break

    if chosen is None:
        for key in ("yahei", "simhei", "simsun", "noto sans cjk", "source han"):
            hit = first_substring(key)
            if hit:
                chosen = hit
                break

    fallback = ["DejaVu Sans"]
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen] + fallback
    else:
        matplotlib.rcParams["font.sans-serif"] = fallback
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen or "DejaVu Sans"


def load_gp_mean_across_seeds(paths: Iterable[Path]) -> tuple[list[float], list[float]]:
    """按 generation 聚合多条轨迹，对每条内的 (fes, volume) 取算术均值。"""
    by_gen: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"GP 轨迹文件不存在: {path}")
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                g = int(row["generation"])
                by_gen[g].append((float(row["fes"]), float(row["volume"])))

    gens = sorted(by_gen.keys())
    fes_mean: list[float] = []
    vol_mean: list[float] = []
    for g in gens:
        pairs = by_gen[g]
        fes_mean.append(sum(p[0] for p in pairs) / len(pairs))
        vol_mean.append(sum(p[1] for p in pairs) / len(pairs))
    return fes_mean, vol_mean


def load_surrogate_by_model(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    buckets: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mt = row["model_type"]
            sc = int(row["sample_count"])
            buckets[mt][sc].append(float(row["f_real"]))

    out: dict[str, tuple[list[float], list[float]]] = {}
    for mt, by_sc in buckets.items():
        xs = sorted(by_sc.keys())
        ys = [float(sum(by_sc[x]) / len(by_sc[x])) for x in xs]
        out[mt] = (xs, ys)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="surrogate_detail 与 GP HA 轨迹（多 seed 均值）同图对比"
    )
    p.add_argument(
        "--surrogate-detail",
        type=Path,
        default=ROOT / "results_surrogate" / "QuickSimu1_surrogate_detail.csv",
    )
    p.add_argument(
        "--gp-traces",
        type=Path,
        nargs="+",
        default=list(_DEFAULT_GP_TRACES),
        help="多条 QuickSimu1_gp_seed_*.csv，按 generation 对齐取 fes、volume 均值",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results_ha" / "QuickSimu1_surrogate_vs_gp_mean.png",
    )
    p.add_argument("--font-family", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _configure_matplotlib_fonts(args.font_family)
    import matplotlib.pyplot as plt

    gp_x, gp_y = load_gp_mean_across_seeds(args.gp_traces)
    by_model = load_surrogate_by_model(args.surrogate_detail)

    colors = {
        "kriging": "#ff7f0e",
        "rbf": "#2ca02c",
        "polynomial": "#d62728",
        "kan": "#9467bd",
        "kan_gp": "#8c564b",
    }
    markers = {
        "kriging": "s",
        "rbf": "^",
        "polynomial": "D",
        "kan": "v",
        "kan_gp": "p",
    }
    mt_cn = {
        "kriging": "克里金(GP)",
        "rbf": "RBF",
        "polynomial": "多项式",
        "kan": "KAN",
        "kan_gp": "KAN-GP",
    }

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    ax.plot(
        gp_x,
        gp_y,
        color="#1f77b4",
        linewidth=2.4,
        marker="o",
        markersize=5,
        label="HA",
        zorder=5,
    )

    order = ("kriging", "rbf", "polynomial", "kan", "kan_gp")
    for mt in order:
        if mt not in by_model:
            continue
        xs, ys = by_model[mt]
        ax.plot(
            xs,
            ys,
            color=colors.get(mt, "#333333"),
            linewidth=1.8,
            marker=markers.get(mt, "o"),
            markersize=5,
            label=f"{mt_cn.get(mt, mt)}",
        )

    ax.set_xlabel(
        "FES（函数评估次数）或 sample_count（代理训练样本数）",
        fontsize=11,
    )
    ax.set_ylabel(r"$f_{\mathrm{real}}$（体积 volume）", fontsize=11)
    ax.set_title(
        "HA优化真实仿真 vs HA优化各代理模型",
        fontsize=12,
    )
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8, ncol=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 已保存: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
