from __future__ import annotations

import math
import shutil
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\dell\Desktop\HA\HA")
OUT = ROOT / "output" / "poster_ha"
BG_SRC = Path(
    r"C:\Users\dell\.codex\generated_images\019eede2-2e85-76f3-a2a0-6bdd6ff49461"
    r"\ig_033b11996fd75738016a38e05b0e78819b9afd03285eee6e50.png"
)
BG = OUT / "img2_algorithm_background_v3.png"
POSTER_PNG = OUT / "HA_algorithm_poster_v3.png"
POSTER_PDF = OUT / "HA_algorithm_poster_v3.pdf"

ZDT_DIR = ROOT / "NSHA_VS_NSGAIII" / "20260616144338"
ZDT_IMAGES = [ZDT_DIR / "pf_ZDT1.png", ZDT_DIR / "pf_ZDT2.png", ZDT_DIR / "pf_ZDT3.png"]

FONT_DIR = Path(r"C:\Windows\Fonts")
FONT_REG = str(FONT_DIR / "msyh.ttc")
FONT_BOLD = str(FONT_DIR / "msyhbd.ttc")
FONT_LIGHT = str(FONT_DIR / "msyhl.ttc")
FONT_SERIF = str(FONT_DIR / "NotoSerifSC-VF.ttf")

W, H = 1800, 4500


def font(size: int, bold: bool = False, light: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    if serif and Path(FONT_SERIF).exists():
        return ImageFont.truetype(FONT_SERIF, size=size)
    return ImageFont.truetype(FONT_BOLD if bold else FONT_LIGHT if light else FONT_REG, size=size)


def bbox(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=f)


def text_w(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont) -> int:
    b = bbox(draw, text, f)
    return b[2] - b[0]


def line_h(draw: ImageDraw.ImageDraw, f: ImageFont.FreeTypeFont, gap: int) -> int:
    b = bbox(draw, "测Ay", f)
    return b[3] - b[1] + gap


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        for ch in para:
            if text_w(draw, cur + ch, f) <= width or not cur:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width: int,
    f: ImageFont.FreeTypeFont,
    fill,
    gap: int = 6,
    max_lines: int | None = None,
) -> int:
    x, y = xy
    lines = wrap(draw, text, f, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    lh = line_h(draw, f, gap)
    for line in lines:
        draw.text((x, y), line, font=f, fill=fill)
        y += lh
    return y


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=None, width=2, radius=18):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def panel(draw: ImageDraw.ImageDraw, box, title: str, accent, subtitle: str | None = None):
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=(1, 21, 56, 220), outline=(80, 180, 230, 160), width=2, radius=18)
    draw.rectangle((x1, y1, x1 + 10, y2), fill=accent)
    draw.text((x1 + 28, y1 + 18), title, font=font(28, bold=True), fill=(255, 255, 255))
    if subtitle:
        draw.text((x1 + 28, y1 + 58), subtitle, font=font(19), fill=(198, 226, 245))
    draw.line((x1 + 28, y1 + 88, x2 - 26, y1 + 88), fill=accent, width=2)


def chip(draw: ImageDraw.ImageDraw, box, text: str, fill, fg=(255, 255, 255)):
    rounded(draw, box, fill=fill, outline=None, width=0, radius=13)
    f = font(18, bold=True)
    x1, y1, x2, y2 = box
    b = bbox(draw, text, f)
    draw.text((x1 + (x2 - x1 - b[2] + b[0]) // 2, y1 + (y2 - y1 - b[3] + b[1]) // 2 - 2), text, font=f, fill=fg)


def arrow(draw: ImageDraw.ImageDraw, start, end, color, width=4):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 14
    pts = [
        (x2, y2),
        (x2 - size * math.cos(ang - 0.45), y2 - size * math.sin(ang - 0.45)),
        (x2 - size * math.cos(ang + 0.45), y2 - size * math.sin(ang + 0.45)),
    ]
    draw.polygon(pts, fill=color)


def formula_box(
    draw: ImageDraw.ImageDraw,
    box,
    title: str,
    formulas: list[str],
    note: str | None = None,
    accent=(255, 206, 72),
):
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=(3, 38, 88, 235), outline=accent, width=2, radius=14)
    draw.text((x1 + 18, y1 + 12), title, font=font(23, bold=True), fill=accent)
    y = y1 + 55
    ff = font(23, serif=True)
    for item in formulas:
        draw.text((x1 + 18, y), item, font=ff, fill=(246, 250, 255))
        y += 36
    if note:
        draw_wrapped(draw, (x1 + 18, y + 4), note, x2 - x1 - 36, font(17), (205, 230, 244), 5)


def crop_chart(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    diff = ImageChops.difference(im, Image.new("RGB", im.size, "white"))
    diff = Image.eval(diff, lambda p: 255 if p > 12 else 0)
    b = diff.getbbox()
    if b:
        l, t, r, bot = b
        im = im.crop((max(0, l - 8), max(0, t - 8), min(im.width, r + 8), min(im.height, bot + 8)))
    return im


def paste_fit(base: Image.Image, src: Image.Image, box, bg=(255, 255, 255)):
    x1, y1, x2, y2 = box
    canvas = Image.new("RGB", (x2 - x1, y2 - y1), bg)
    im = src.copy()
    im.thumbnail((canvas.width - 12, canvas.height - 12), Image.Resampling.LANCZOS)
    canvas.paste(im, ((canvas.width - im.width) // 2, (canvas.height - im.height) // 2))
    base.paste(canvas, (x1, y1))


def bullet_block(draw, x, y, width, items, size=21, color=(232, 244, 255), accent=(255, 206, 72), gap=8):
    f = font(size)
    for item in items:
        draw.ellipse((x, y + 9, x + 9, y + 18), fill=accent)
        y = draw_wrapped(draw, (x + 22, y), item, width - 22, f, color, gap)
        y += 4
    return y


def draw_background() -> Image.Image:
    if BG_SRC.exists() and not BG.exists():
        shutil.copy2(BG_SRC, BG)
    if BG.exists():
        bg = Image.open(BG).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    else:
        bg = Image.new("RGB", (W, H), (0, 16, 48))
    navy = Image.new("RGB", (W, H), (0, 16, 48))
    poster = Image.blend(navy, bg, 0.33)
    overlay = Image.new("RGBA", (W, H), (0, 13, 40, 120))
    poster = Image.alpha_composite(poster.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(poster, "RGBA")
    draw.rectangle((0, 0, W, 420), fill=(0, 17, 52, 210))
    draw.rectangle((0, H - 90, W, H), fill=(0, 16, 48, 230))
    return poster


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    poster = draw_background()
    draw = ImageDraw.Draw(poster, "RGBA")

    white = (248, 252, 255)
    pale = (214, 235, 248)
    cyan = (73, 219, 255)
    gold = (255, 206, 72)
    orange = (255, 142, 70)
    green = (115, 236, 160)
    violet = (164, 152, 255)

    title = "基于聚类自适应局部搜索与混合交叉池的混合进化算法研究与应用"
    draw.text((58, 36), title, font=font(52, bold=True), fill=white)
    draw.text((62, 112), "通用混合进化算法：面向高维、多模态、昂贵黑箱与多目标约束优化", font=font(30), fill=pale)
    draw.text((62, 160), "胡杨超  |  电子科技大学（深圳）高等研究院  |  计算机技术  |  导师：李耘、王华山", font=font(24), fill=gold)

    intro = (
        "研究对象不是某一个固定工程零件，而是一套可迁移的优化算法框架。算法以“全局进化负责探索、"
        "聚类局部搜索负责开发、混合交叉池负责重组、多臂老虎机与标量化负责自适应决策”为主线，"
        "在 CEC/ZDT 基准与工程仿真案例中验证其收敛精度、函数评估效率和可扩展性。"
    )
    rounded(draw, (58, 220, 1742, 358), fill=(2, 36, 84, 215), outline=(73, 219, 255, 130), width=2, radius=20)
    draw_wrapped(draw, (88, 247), intro, 1625, font(24), white, 8)

    # Layout boxes
    xL, xM, xR = 48, 625, 1202
    colw = 550
    y = 425

    p1 = (xL, y, xL + colw, 1045)
    p2 = (xM, y, xM + colw, 1045)
    p3 = (xR, y, xR + colw, 1045)
    p4 = (xL, 1085, xL + colw, 1775)
    p5 = (xM, 1085, xM + colw, 1775)
    p6 = (xR, 1085, xR + colw, 1775)
    p7 = (xL, 1818, xL + colw, 2575)
    p8 = (xM, 1818, xM + colw, 2575)
    p9 = (xR, 1818, xR + colw, 2575)
    p10 = (48, 2618, 1752, 3435)
    p11 = (48, 3478, 887, 4390)
    p12 = (913, 3478, 1752, 4390)

    panel(draw, p1, "问题定义与研究动机", cyan, "通用算法目标，而非单一工程问题")
    yy = 535
    yy = bullet_block(
        draw,
        xL + 35,
        yy,
        485,
        [
            "复杂优化常表现为高维、非线性、多模态、黑箱、不可导和计算昂贵。",
            "标准 GA 全局性较强，但局部开发弱，后期收敛慢，且容易破坏长距离优良图式。",
            "全员局部搜索虽然能加速收敛，但会消耗大量函数评估次数，昂贵仿真环境下不可接受。",
            "本研究希望构建从单目标到多目标、从无约束到有约束均可扩展的混合进化框架。"
        ],
        size=21,
        accent=cyan,
    )
    formula_box(
        draw,
        (xL + 34, 840, xL + colw - 32, 1010),
        "通用形式",
        ["min / max  F(x)", "s.t.  g_j(x) ≤ 0,  h_k(x)=0", "x ∈ Ω,  F(x) may be black-box"],
        accent=cyan,
    )

    panel(draw, p2, "算法总体框架", gold, "Lamarckian learning + population evolution")
    nodes = [
        ("P_t", "初始化/保留精英", xM + 42, 550),
        ("聚类", "识别局部盆地", xM + 210, 550),
        ("局部搜索", "代表个体后天学习", xM + 378, 550),
        ("回写", "x' 替换 x", xM + 42, 735),
        ("交叉池", "SBX/多父代/DE", xM + 210, 735),
        ("选择", "约束优先+精英保留", xM + 378, 735),
    ]
    for head, sub, nx, ny in nodes:
        rounded(draw, (nx, ny, nx + 132, ny + 115), fill=(4, 47, 101, 230), outline=gold, width=2, radius=16)
        draw.text((nx + 18, ny + 13), head, font=font(23, bold=True), fill=gold)
        draw_wrapped(draw, (nx + 14, ny + 50), sub, 104, font(17), pale, 4)
    arrow(draw, (xM + 174, 607), (xM + 210, 607), gold, 3)
    arrow(draw, (xM + 342, 607), (xM + 378, 607), gold, 3)
    arrow(draw, (xM + 444, 665), (xM + 108, 735), gold, 3)
    arrow(draw, (xM + 174, 792), (xM + 210, 792), gold, 3)
    arrow(draw, (xM + 342, 792), (xM + 378, 792), gold, 3)
    arrow(draw, (xM + 444, 735), (xM + 108, 550), gold, 3)
    draw_wrapped(
        draw,
        (xM + 38, 892),
        "核心思想：学习结果不是只改变适应度评价，而是以 Lamarckian 方式回写到染色体，直接改变下一代可继承信息。聚类控制“在哪里学”，交叉池控制“如何重组”。",
        486,
        font(21),
        white,
        6,
    )

    panel(draw, p3, "聚类自适应局部搜索", orange, "本课题重点强调机制")
    yy = 535
    yy = bullet_block(
        draw,
        xR + 35,
        yy,
        490,
        [
            "对种群在决策空间进行 K-means / MeanShift / DBSCAN 聚类，兼容固定簇数与密度自适应场景。",
            "每 5 代周期触发，或连续 3 代无改进触发；停滞时加深搜索深度。",
            "每簇只选择约束优先意义下的代表个体；簇数过多时保留最有潜力的 niche_num 个代表。",
            "改进后才回写，避免局部搜索噪声污染种群。"
        ],
        size=21,
        accent=orange,
    )
    formula_box(
        draw,
        (xR + 34, 858, xR + colw - 32, 1010),
        "触发与代表",
        ["trigger ⇔ (t mod 5=0) ∨ (stagnation≥3)", "r_c = arg best_{i∈C_c} (CV_i, f_i)", "depth = 10 if stagnant else 3"],
        accent=orange,
    )

    panel(draw, p4, "局部代理与伪梯度", cyan, "面向小样本昂贵黑箱")
    draw_wrapped(
        draw,
        (xL + 35, 1195),
        "局部搜索并不假设真实目标可导。算法复用簇内邻域和历史样本，构建局部 RBF/GP 近似，获得低代价的搜索方向；当样本不足或代理不可靠时，回退到直接搜索/原始解。",
        480,
        font(21),
        white,
        7,
    )
    formula_box(
        draw,
        (xL + 34, 1368, xL + colw - 32, 1608),
        "代理模型目标",
        [
            "RBF:  ŷ(x)=Σ_i λ_i φ(||x-x_i||)+p(x)",
            "GP:   x_next = argmin μ(x)-κσ(x)",
            "penalty: y=F+α·CV,  α≈10·mean(|F|)"
        ],
        "小样本场景的关键不是全局代理处处准确，而是在候选盆地内快速变准。",
        accent=cyan,
    )
    draw_wrapped(draw, (xL + 35, 1632), "工程仿真闭环只作为验证场景之一：参数 → 仿真响应 → 代理更新 → HA 候选 → 真实回评。", 480, font(20), pale, 5)

    panel(draw, p5, "混合交叉池", gold, "显式比例 + 隐性选择压力")
    formula_box(
        draw,
        (xM + 34, 1195, xM + colw - 32, 1448),
        "三类重组算子",
        [
            "SBX 60%: β=(2u)^{1/(η+1)},  u≤0.5",
            "              β=[1/(2(1-u))]^{1/(η+1)},  u>0.5",
            "Multi-parent 25%: x_child=Σ_i ω_i x_i + ε",
            "DE 15%: v=x+F1(x_best-x)+F2(x_r1-x_r2)"
        ],
        accent=gold,
    )
    draw_wrapped(
        draw,
        (xM + 35, 1475),
        "设计理由：SBX 保证稳定遗传，多父代融合群体统计信息，DE/current-to-best 提供方向性开发。三者共同缓解“纯随机缺方向”和“单一算子易失效”的问题。",
        480,
        font(21),
        white,
        7,
    )
    formula_box(
        draw,
        (xM + 34, 1610, xM + colw - 32, 1738),
        "多项式变异",
        ["p_m=1/d,  η_m=20+80·min(1,t/50)", "stagnant ⇒ η_m ↓,  mutation radius ↑"],
        accent=gold,
    )

    panel(draw, p6, "多臂老虎机自适应策略选择", violet, "让搜索器选择随阶段改变")
    formula_box(
        draw,
        (xR + 34, 1195, xR + colw - 32, 1515),
        "QL / UCB 更新",
        [
            "ε-greedy:  random if p<ε, else argmax Q(a)",
            "R_t=max(0, f_before-f_after)",
            "Q(a)←Q(a)+(1/n_a)(R_t-Q(a))",
            "UCB(a)=Q(a)+c√(log t / n_a)",
            "Q←Q+α(R_norm-Q),  ε←max(0.95ε,0.05)"
        ],
        "将局部搜索器视作“臂”，用真实改进量学习当前阶段最值得调用的搜索方式。",
        accent=violet,
    )
    draw_wrapped(draw, (xR + 35, 1545), "双选择器思想：全局最优个体偏深度开发，普通精英个体偏代理/探索，避免一种经验支配所有个体。", 480, font(21), white, 7)

    panel(draw, p7, "约束处理与多样性维护", green, "防止不可行和早熟收敛")
    yy = 1930
    yy = bullet_block(
        draw,
        xL + 35,
        yy,
        485,
        [
            "约束排序采用可行解优先思想：可行解优于不可行解；同可行比较目标，同不可行比较违反度。",
            "标量局部搜索统一叠加约束惩罚，使无约束局部搜索器也能感知可行性。",
            "重复个体过多时注入随机个体并重置步长，维持后期探索能力。",
            "工程仿真异常、网格失败或无效响应返回大罚值，保证算法流程不断裂。"
        ],
        size=21,
        accent=green,
    )
    formula_box(
        draw,
        (xL + 34, 2308, xL + colw - 32, 2538),
        "约束惩罚",
        ["g(x)=value(x)+α·CV(x)", "α=10|value(x)|+1", "CV(x)=Σ_j max(0,g_j(x))"],
        "与 Deb 可行性规则共同工作：选择时重可行性，局部搜索时用连续惩罚引导方向。",
        accent=green,
    )

    panel(draw, p8, "多目标扩展：NSHA", orange, "局部搜索与 Pareto 前沿相容")
    draw_wrapped(
        draw,
        (xM + 35, 1930),
        "NSHA 将多目标问题沿 NSGA-III 参考方向分解。每个 niche 选择代表个体，并用对应参考方向构造标量化目标；局部改进后回到多目标选择框架中维护前沿分布。",
        480,
        font(21),
        white,
        7,
    )
    formula_box(
        draw,
        (xM + 34, 2110, xM + colw - 32, 2458),
        "标量化公式",
        [
            "F_s=F(x)-z*,   ŵ=w/||w||",
            "PBI: d1=<F_s,ŵ>, d2=||F_s-d1ŵ||",
            "g_PBI=d1+θd2",
            "TCH: g=max_i w_i|f_i-z_i*|",
            "ASF: g=max_i (f_i-z_i*)/w_i"
        ],
        "所有 scalarizer 叠加 value+α·CV；ZDT2/ZDT3 暴露了凹/不连续前沿下进一步改进 θ 和 nadir 的必要性。",
        accent=orange,
    )

    panel(draw, p9, "理论解释与验证路径", cyan, "为什么它有研究价值")
    yy = 1930
    yy = bullet_block(
        draw,
        xR + 35,
        yy,
        485,
        [
            "图式定理扩展：多父代概率采样降低长定义距图式被交叉破坏的概率。",
            "马尔可夫链收敛：精英保留使状态转移满足收敛分析所需的单调保优条件。",
            "消融实验：Full HA / 去聚类 / 去局部搜索 / 标准交叉 / 标准 GA。",
            "统计检验：CEC/ZDT 上结合 IGD、HV、最优值、FEs 与 Wilcoxon 秩和检验。"
        ],
        size=21,
        accent=cyan,
    )
    formula_box(
        draw,
        (xR + 34, 2260, xR + colw - 32, 2538),
        "图式与收敛表达",
        ["E[m(H,t+1)] ≥ m(H,t)·f(H)/f_avg·(1-p_d)", "Elitism: best(P_{t+1}) ≤ best(P_t)", "P(lim_{t→∞} P_t∩X*≠∅)=1"],
        "海报中只给核心表达，完整证明应放论文正文。",
        accent=cyan,
    )

    panel(draw, p10, "阶段实验：ZDT 多目标验证（算法通用性的一部分）", gold, "图示为 NSGA-III 与 HA-RBF-PBI 的 Pareto 前沿对比")
    chart_boxes = [(92, 2748, 594, 3175), (650, 2748, 1152, 3175), (1208, 2748, 1710, 3175)]
    for path, box in zip(ZDT_IMAGES, chart_boxes):
        paste_fit(poster, crop_chart(path), box)
        rounded(draw, box, fill=(255, 255, 255, 0), outline=gold, width=4, radius=12)
    result_text = (
        "结果解读：ZDT1 上 HA-RBF-PBI 的 IGD=6.04e-02，优于 NSGA-III 的 1.21e-01，说明聚类局部搜索能够提升部分前沿逼近。"
        "ZDT2/ZDT3 上表现不稳定，反映当前 PBI 局部搜索对凹前沿和不连续前沿仍敏感。该结果不是失败，而是明确指出下一步："
        "动态 nadir、θ 自适应、支配式局部搜索与标量化消融。"
    )
    draw_wrapped(draw, (92, 3210), result_text, 1580, font(23), white, 7)
    draw_wrapped(
        draw,
        (92, 3322),
        "工程案例定位：SolidWorks/CAE 只作为昂贵黑箱验证场景，用于检验小样本代理、异常惩罚、真实回评和算法接口的可用性；不作为算法研究对象本身。",
        1580,
        font(22, bold=True),
        pale,
        7,
    )

    panel(draw, p11, "已完成工作", green, "从算法原型到扩展验证")
    yy_done = bullet_block(
        draw,
        88,
        3588,
        760,
        [
            "实现单目标 HA：聚类局部搜索、混合交叉池、自适应多项式变异与多样性维护。",
            "实现 Bandit 版本：QL/UCB 根据真实改进量动态选择局部搜索器。",
            "实现 NSHA：参考方向 niche、PBI/Tchebycheff/WS/ASF/AASF/IPBI 标量化接口。",
            "搭建小样本验证流程：LHS 初始采样、GP/KAN-GP 代理、CEI 采集、top-k 真实回评。",
            "整理 ZDT 与工程仿真初步结果，形成下一步消融和统计检验方向。"
        ],
        size=22,
        accent=green,
    )
    formula_box(
        draw,
        (88, yy_done + 15, 850, 4330),
        "当前代码能力",
        [
            "HA: single-objective + constraints",
            "NSHA: ref-dir niching + scalarizers",
            "Bandit: QL / UCB local-search selector",
            "Surrogate: GP / KAN-GP + CEI acquisition"
        ],
        "这些模块共同服务于通用算法框架；工程案例只承担黑箱昂贵评估验证。",
        accent=green,
    )

    panel(draw, p12, "下一步研究重点", orange, "让算法故事闭环")
    yy_next = bullet_block(
        draw,
        953,
        3588,
        760,
        [
            "聚类自适应：比较 K-means、MeanShift、DBSCAN 在多峰/高维/噪声场景下的收益与开销。",
            "交叉池自适应：由固定 60/25/15 走向基于反馈的算子概率更新。",
            "多目标局部搜索：改进 PBI 参数、动态理想/最差点、Pareto-Nelder-Mead 和共同下降方向。",
            "理论证明：补全多亲概率采样图式传播模型与精英马尔可夫链收敛条件。",
            "验证体系：CEC + ZDT + 一个工程黑箱案例，主次清晰地证明通用算法能力。"
        ],
        size=22,
        accent=orange,
    )
    formula_box(
        draw,
        (953, yy_next + 15, 1715, 4330),
        "论文实验矩阵",
        [
            "Benchmark: CEC + ZDT + engineering black-box",
            "Ablation: Full / no-cluster / no-LS / no-pool",
            "Metrics: best, FEs, IGD, HV, feasibility rate",
            "Statistics: Wilcoxon rank-sum test"
        ],
        "主结论应证明通用算法能力，而不是某个具体零件调参成功。",
        accent=orange,
    )

    draw.line((58, 4410, 1742, 4410), fill=gold, width=3)
    draw.text((62, 4430), "HA / NSHA  |  Clustering Adaptive Local Search  |  Hybrid Crossover Pool  |  Bandit Local Search Selection", font=font(23), fill=pale)
    draw.text((1260, 4430), "2026 Academic Poster", font=font(23), fill=gold)

    poster.save(POSTER_PNG, quality=96)
    poster.save(POSTER_PDF, "PDF", resolution=220)
    print(POSTER_PNG)
    print(POSTER_PDF)


if __name__ == "__main__":
    main()
