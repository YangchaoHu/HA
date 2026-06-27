"""
可视化 ZDT1/ZDT2/ZDT3 的 Pareto 前沿，以及参考方向射线与前沿的交点。
射线从各问题的理想点（最好点）z* = (min f1, min f2) 出发，
与 NSGA-III 的实际做法一致（先减去理想点归一化，射线在原空间即从 z* 出发）。
目的：展示均匀参考方向在不同形状前沿上的交点是否均匀分布。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import brentq, minimize_scalar

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

# ── Pareto 前沿解析函数 ──────────────────────────────────────────────────────

def zdt1_f2(f1):
    return 1.0 - np.sqrt(f1)

def zdt2_f2(f1):
    return 1.0 - f1 ** 2

def zdt3_f2(f1):
    return 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)

# ZDT3 Pareto 最优段的 f1 区间（文献标准值）
ZDT3_SEGMENTS = [
    (0.0000, 0.0830),
    (0.1822, 0.2583),
    (0.4093, 0.4538),
    (0.6183, 0.6525),
    (0.8233, 0.8518),
]

# ── 理想点（最好点）计算 ──────────────────────────────────────────────────────

def compute_ideal_point(func, segments):
    """
    理想点 z* = (min f1 on front, min f2 on front)。
    各目标分量独立取最优值。
    """
    z1 = min(seg[0] for seg in segments)          # min f1 = 前沿起点
    z2 = np.inf
    for seg_lo, seg_hi in segments:
        # 在该段上最小化 f2
        res = minimize_scalar(func, bounds=(seg_lo, seg_hi), method="bounded")
        if res.fun < z2:
            z2 = res.fun
    return z1, z2

# ── 参考方向生成（NSGA-III 风格，2D）────────────────────────────────────────

def make_reference_directions(n_divisions: int):
    """
    在 2D 单纯形上均匀生成 n_divisions+1 个参考方向，归一化为单位向量。
    单纯形参考点为 (i/p, 1-i/p)，归一化后得方向向量。
    """
    dirs = []
    for i in range(n_divisions + 1):
        w1 = i / n_divisions
        w2 = 1.0 - w1
        norm = np.hypot(w1, w2)
        dirs.append((w1 / norm, w2 / norm))
    return dirs

# ── 从理想点出发的射线与前沿求交 ──────────────────────────────────────────────

def ray_intersect_segment(z1, z2, d1, d2, front_func, f1_lo, f1_hi):
    """
    射线：P(t) = (z1 + t*d1,  z2 + t*d2),  t >= 0
    前沿：f2 = front_func(f1)
    求满足 z2 + t*d2 = front_func(z1 + t*d1) 的 t，
    等价于对 f1 = z1 + t*d1 求根：
        h(f1) = (z2 + (f1 - z1)*(d2/d1)) - front_func(f1) = 0
    返回 (f1, f2) 或 None。
    """
    EPS = 1e-9

    if d1 < EPS:
        # 沿 f2 轴方向的竖直射线，f1 恒等于 z1
        if f1_lo - EPS <= z1 <= f1_hi + EPS:
            return (z1, front_func(z1))
        return None

    slope = d2 / d1

    def h(f1):
        f2_ray = z2 + (f1 - z1) * slope
        return f2_ray - front_func(f1)

    # f1 范围：从 z1（t=0）起，沿 d1>0 方向增大
    a = max(f1_lo, z1 + EPS)
    b = min(f1_hi, 1.0 - EPS)
    if a > b:
        return None

    ha, hb = h(a), h(b)
    if ha * hb > 0:
        return None  # 同号无根

    try:
        f1_sol = brentq(h, a, b, xtol=1e-10)
        return (f1_sol, front_func(f1_sol))
    except Exception:
        return None


def find_all_intersections(z1, z2, d1, d2, front_func, segments):
    """在所有段上收集交点（ZDT3 不连续前沿）。"""
    pts = []
    for seg_lo, seg_hi in segments:
        pt = ray_intersect_segment(z1, z2, d1, d2, front_func, seg_lo, seg_hi)
        if pt is not None:
            pts.append(pt)
    return pts

# ── 问题定义 ──────────────────────────────────────────────────────────────────

N_DIVISIONS = 12   # 参考方向条数 = N_DIVISIONS + 1

PROBLEMS = [
    {
        "name": "ZDT1（凸前沿）",
        "func": zdt1_f2,
        "segments": [(0.0, 1.0)],
        "f1_dense": np.linspace(0.0, 1.0, 600),
    },
    {
        "name": "ZDT2（凹前沿）",
        "func": zdt2_f2,
        "segments": [(0.0, 1.0)],
        "f1_dense": np.linspace(0.0, 1.0, 600),
    },
    {
        "name": "ZDT3（非连续前沿）",
        "func": zdt3_f2,
        "segments": ZDT3_SEGMENTS,
        "f1_dense": None,
    },
]

# 预计算各问题理想点
for prob in PROBLEMS:
    z1, z2 = compute_ideal_point(prob["func"], prob["segments"])
    prob["ideal"] = (z1, z2)
    print(f"{prob['name']}  理想点 z* = ({z1:.4f}, {z2:.4f})")

dirs = make_reference_directions(N_DIVISIONS)
colors = cm.tab20(np.linspace(0, 1, len(dirs)))

# ── 绘图 ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle(
    f"NSGA-III 参考方向射线（从理想点 z* 出发）与 Pareto 前沿的交点  [{N_DIVISIONS + 1} 条射线]\n"
    "射线在单纯形上均匀采样 —— 观察交点在不同形状前沿上的分布是否仍然均匀",
    fontsize=12,
)

for ax, prob in zip(axes, PROBLEMS):
    name   = prob["name"]
    func   = prob["func"]
    segs   = prob["segments"]
    iz1, iz2 = prob["ideal"]

    # ── 绘制 Pareto 前沿 ──
    if prob["f1_dense"] is not None:
        f1v = prob["f1_dense"]
        ax.plot(f1v, func(f1v), color="steelblue", linewidth=2.5,
                label="Pareto 前沿", zorder=2)
    else:
        first = True
        for seg_lo, seg_hi in segs:
            f1v = np.linspace(seg_lo, seg_hi, 300)
            ax.plot(f1v, func(f1v), color="steelblue", linewidth=2.5,
                    label="Pareto 前沿" if first else "", zorder=2)
            first = False

    # ── 标注理想点 ──
    ax.plot(iz1, iz2, "r*", markersize=11, zorder=7,
            label=f"理想点 z*=({iz1:.2f},{iz2:.2f})")

    # ── 绘制射线及交点 ──
    for idx, (d1, d2) in enumerate(dirs):
        pts = find_all_intersections(iz1, iz2, d1, d2, func, segs)

        # 射线终点：到最远交点后延伸 15%，或固定长度
        if pts:
            t_vals = [(p[0] - iz1) / d1 if d1 > 1e-9 else (p[1] - iz2) / d2
                      for p in pts]
            t_end = max(t_vals) * 1.15
        else:
            t_end = 1.8

        rx_end = iz1 + t_end * d1
        ry_end = iz2 + t_end * d2
        ax.plot(
            [iz1, rx_end], [iz2, ry_end],
            color=colors[idx], linewidth=0.9, linestyle="--", alpha=0.55,
            zorder=1,
        )

        for pt in pts:
            ax.plot(
                pt[0], pt[1],
                "o", color=colors[idx],
                markersize=7, markeredgecolor="k", markeredgewidth=0.5,
                zorder=5,
            )

    ax.set_xlabel("$f_1$", fontsize=11)
    ax.set_ylabel("$f_2$", fontsize=11)
    ax.set_title(name, fontsize=12)

    # 根据各问题自动设置轴范围（留出理想点下方空间）
    all_f1 = np.concatenate([np.linspace(s[0], s[1], 200) for s in segs])
    all_f2 = func(all_f1)
    x_margin = 0.08
    y_margin = 0.12
    ax.set_xlim(iz1 - x_margin, max(all_f1) + x_margin)
    ax.set_ylim(iz2 - y_margin, max(all_f2) + y_margin)

    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
out_path = "pf_convex_vs_concave.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n图像已保存至 {out_path}")
