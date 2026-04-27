"""
对比 QuickSimu1：
- results/QuickSimu1_gp_seeds_mean.csv：横轴 fes_mean，纵轴 best_f_mean（真实 HA 三 seed 平均）
- results_surrogate/QuickSimu1_surrogate_summary.csv：横轴 sample_count，纵轴 y_real_mean

输出 PNG 到 compare_proxy_models/results/ 目录。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _configure_matplotlib_fonts(preferred: str | None = None) -> str:
    """
    Windows 上为中文标题/图例设置可用无衬线字体，避免方框或缺字。
    优先顺序：用户指定 -> 微软雅黑系 -> 黑体 -> 宋体 -> 其它 CJK -> DejaVu Sans。
    """
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
            "NSimSun",
            "KaiTi",
            "FangSong",
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

    if chosen is None:
        print(
            "[WARN] 未在系统中匹配到常见中文字体，中文可能仍显示异常；"
            "可用 --font-family 指定已安装字体名（如 Microsoft YaHei）。",
            file=sys.stderr,
        )
    else:
        print(f"[INFO] matplotlib 使用字体: {chosen}", file=sys.stderr)

    return chosen or "DejaVu Sans"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gp-mean",
        type=Path,
        default=ROOT / "results" / "QuickSimu1_gp_seeds_mean.csv",
    )
    p.add_argument(
        "--surrogate-summary",
        type=Path,
        default=ROOT / "results_surrogate" / "QuickSimu1_surrogate_summary.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "QuickSimu1_gp_vs_surrogate.png",
    )
    p.add_argument(
        "--font-family",
        type=str,
        default=None,
        help="matplotlib 无衬线字体名（如 Microsoft YaHei），用于正确显示中文",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _configure_matplotlib_fonts(args.font_family)
    import matplotlib.pyplot as plt

    # --- GP mean ---
    gp_x, gp_y = [], []
    with args.gp_mean.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gp_x.append(float(row["fes_mean"]))
            gp_y.append(float(row["best_f_mean"]))

    # --- Surrogate summary ---
    by_model: dict[str, tuple[list[float], list[float]]] = {}
    with args.surrogate_summary.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mt = row["model_type"]
            if mt not in by_model:
                by_model[mt] = ([], [])
            by_model[mt][0].append(float(row["sample_count"]))
            by_model[mt][1].append(float(row["y_real_mean"]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        gp_x,
        gp_y,
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="HA 真实仿真",
    )  

    colors = {"kriging": "#ff7f0e", "rbf": "#2ca02c", "polynomial": "#d62728"}
    markers = {"kriging": "s", "rbf": "^", "polynomial": "D"}
    mt_cn = {"kriging": "克里金", "rbf": "径向基 RBF", "polynomial": "二次多项式"}
    for mt in ("kriging", "rbf", "polynomial"):
        if mt not in by_model:
            continue
        xs, ys = by_model[mt]
        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        ax.plot(
            xs,
            ys,
            color=colors.get(mt, None),
            linewidth=1.8,
            marker=markers.get(mt, "o"),
            markersize=5,
            label=f"{mt_cn.get(mt, mt)}",
        )

    ax.set_xlabel("FES（HA 真实仿真）or 训练样本数 （构建代理模型）", fontsize=11)
    ax.set_ylabel("目标 y（|disp_x|）", fontsize=11)
    ax.set_title(
        "真实 HA 仿真 vs 代理模型",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 已保存: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
