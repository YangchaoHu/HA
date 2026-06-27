import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "C:/Users/dell/Desktop/HA/HA/output/poster_ha";
const WORK = `${OUT}/editable_v4_work`;
const PPTX = `${OUT}/HA_algorithm_poster_editable_v4.pptx`;
const PREVIEW = `${OUT}/HA_algorithm_poster_editable_v4_preview.png`;
const LAYOUT = `${OUT}/HA_algorithm_poster_editable_v4_layout.json`;
const BG = `${OUT}/img2_algorithm_background_v3.png`;
const ZDT = [
  "C:/Users/dell/Desktop/HA/HA/NSHA_VS_NSGAIII/20260616144338/pf_ZDT1.png",
  "C:/Users/dell/Desktop/HA/HA/NSHA_VS_NSGAIII/20260616144338/pf_ZDT2.png",
  "C:/Users/dell/Desktop/HA/HA/NSHA_VS_NSGAIII/20260616144338/pf_ZDT3.png",
];

const W = 1800;
const H = 4500;
const C = {
  navy: "#061735",
  panel: "#08224F",
  panel2: "#0A2D67",
  white: "#F8FAFC",
  pale: "#CDE6F7",
  muted: "#9BB9CE",
  cyan: "#49DBFF",
  gold: "#FFCE48",
  orange: "#FF8E46",
  green: "#73ECA0",
  violet: "#A498FF",
};

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function imageBytes(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addShape(slide, geometry, x, y, w, h, fill, line = "none", width = 0, name = undefined) {
  const config = {
    geometry,
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width },
  };
  if (geometry === "rect" || geometry === "textbox" || geometry === "roundRect") {
    config.borderRadius = geometry === "roundRect" ? "rounded-lg" : 14;
  }
  return slide.shapes.add(config);
}

function addText(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: opts.name,
    position: { left: x, top: y, width: w, height: h },
    fill: opts.fill ?? "none",
    line: { style: "solid", fill: opts.line ?? "none", width: opts.lineWidth ?? 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: opts.size ?? 22,
    bold: opts.bold ?? false,
    color: opts.color ?? C.white,
    typeface: opts.typeface ?? "Microsoft YaHei",
    alignment: opts.align ?? "left",
  };
  return shape;
}

function panel(slide, x, y, w, h, title, accent, subtitle = "") {
  addShape(slide, "rect", x, y, w, h, `${C.panel}/88`, `${accent}/90`, 2, `panel-${title}`);
  addShape(slide, "rect", x, y, 10, h, accent, "none", 0);
  addText(slide, title, x + 28, y + 16, w - 56, 34, { size: 28, bold: true });
  if (subtitle) addText(slide, subtitle, x + 28, y + 52, w - 56, 26, { size: 18, color: C.muted });
  addShape(slide, "line", x + 28, y + 88, w - 56, 0, "none", accent, 2);
}

function formulaBox(slide, x, y, w, h, title, formulas, note, accent) {
  addShape(slide, "rect", x, y, w, h, "#092B62", accent, 2, `formula-${title}`);
  addText(slide, title, x + 18, y + 12, w - 36, 28, { size: 22, bold: true, color: accent });
  addText(slide, formulas.join("\n"), x + 18, y + 54, w - 36, h - 76, {
    size: 20,
    color: C.white,
    typeface: "Cambria Math",
  });
  if (note) {
    addText(slide, note, x + 18, y + h - 52, w - 36, 42, { size: 15, color: C.pale });
  }
}

function bullets(slide, items, x, y, w, h, accent, size = 20) {
  const text = items.map((item) => `• ${item}`).join("\n");
  addText(slide, text, x, y, w, h, { size, color: C.white });
}

function flowNode(slide, x, y, w, h, head, body, accent) {
  addShape(slide, "roundRect", x, y, w, h, "#0B3577", accent, 2);
  addText(slide, head, x + 14, y + 12, w - 28, 28, { size: 22, bold: true, color: accent });
  addText(slide, body, x + 14, y + 48, w - 28, h - 54, { size: 16, color: C.pale });
}

function arrow(slide, x, y, w, h, accent) {
  addShape(slide, "rightArrow", x, y, w, h, accent, "none", 0);
}

async function main() {
  await fs.mkdir(WORK, { recursive: true });

  const prs = Presentation.create({ slideSize: { width: W, height: H } });
  const slide = prs.slides.add();
  slide.background.fill = C.navy;

  slide.images.add({
    blob: await imageBytes(BG),
    contentType: "image/png",
    alt: "abstract algorithm background",
    fit: "cover",
    position: { left: 0, top: 0, width: W, height: H },
  });
  addShape(slide, "rect", 0, 0, W, H, "#061735/82", "none", 0);
  addShape(slide, "rect", 0, 0, W, 410, "#061735/92", "none", 0);

  addText(slide, "基于聚类自适应局部搜索与混合交叉池的混合进化算法研究与应用", 58, 34, 1688, 70, {
    size: 49,
    bold: true,
  });
  addText(slide, "通用混合进化算法：面向高维、多模态、昂贵黑箱与多目标约束优化", 62, 112, 1500, 38, {
    size: 29,
    color: C.pale,
  });
  addText(slide, "胡杨超  |  电子科技大学（深圳）高等研究院  |  计算机技术  |  导师：李耘、王华山", 62, 160, 1300, 32, {
    size: 23,
    color: C.gold,
  });
  addShape(slide, "rect", 58, 220, 1684, 138, "#08224F/86", `${C.cyan}/70`, 2);
  addText(
    slide,
    "研究对象不是某一个固定工程零件，而是一套可迁移的优化算法框架。算法以“全局进化负责探索、聚类局部搜索负责开发、混合交叉池负责重组、多臂老虎机与标量化负责自适应决策”为主线，在 CEC/ZDT 基准与工程仿真案例中验证其收敛精度、函数评估效率和可扩展性。",
    88,
    247,
    1625,
    92,
    { size: 23, color: C.white },
  );

  const xL = 48, xM = 625, xR = 1202, col = 550;
  const pY = 425;

  panel(slide, xL, pY, col, 620, "问题定义与研究动机", C.cyan, "通用算法目标，而非单一工程问题");
  bullets(slide, [
    "复杂优化常表现为高维、非线性、多模态、黑箱、不可导和计算昂贵。",
    "标准 GA 全局性较强，但局部开发弱，后期收敛慢，且容易破坏长距离优良图式。",
    "全员局部搜索会消耗大量函数评估次数，在昂贵仿真环境下不可接受。",
    "本研究构建从单目标到多目标、从无约束到有约束均可扩展的混合进化框架。"
  ], xL + 35, 535, 485, 280, C.cyan, 20);
  formulaBox(slide, xL + 34, 840, col - 66, 170, "通用形式", [
    "min / max   F(x)",
    "s.t.   gⱼ(x) ≤ 0,   hₖ(x)=0",
    "x ∈ Ω,   F(x) may be black-box"
  ], "", C.cyan);

  panel(slide, xM, pY, col, 620, "算法总体框架", C.gold, "Lamarckian learning + population evolution");
  const fx = xM + 42, fy = 550, nw = 132, nh = 112;
  flowNode(slide, fx, fy, nw, nh, "P_t", "初始化/保留精英", C.gold);
  arrow(slide, fx + 140, fy + 42, 54, 28, C.gold);
  flowNode(slide, fx + 200, fy, nw, nh, "聚类", "识别局部盆地", C.gold);
  arrow(slide, fx + 340, fy + 42, 54, 28, C.gold);
  flowNode(slide, fx + 400, fy, nw, nh, "局部搜索", "代表个体后天学习", C.gold);
  flowNode(slide, fx, fy + 185, nw, nh, "回写", "x' 替换 x", C.gold);
  arrow(slide, fx + 140, fy + 227, 54, 28, C.gold);
  flowNode(slide, fx + 200, fy + 185, nw, nh, "交叉池", "SBX/多父代/DE", C.gold);
  arrow(slide, fx + 340, fy + 227, 54, 28, C.gold);
  flowNode(slide, fx + 400, fy + 185, nw, nh, "选择", "约束优先+精英保留", C.gold);
  addText(slide, "核心思想：学习结果不是只改变适应度评价，而是以 Lamarckian 方式回写到染色体，直接改变下一代可继承信息。聚类控制“在哪里学”，交叉池控制“如何重组”。", xM + 38, 892, 486, 105, { size: 20 });

  panel(slide, xR, pY, col, 620, "聚类自适应局部搜索", C.orange, "本课题重点强调机制");
  bullets(slide, [
    "在决策空间进行 K-means / MeanShift / DBSCAN 聚类，兼容固定簇数与密度自适应场景。",
    "每 5 代周期触发，或连续 3 代无改进触发；停滞时加深搜索深度。",
    "每簇只选择约束优先意义下的代表个体；簇数过多时保留最有潜力的 niche_num 个代表。",
    "改进后才回写，避免局部搜索噪声污染种群。"
  ], xR + 35, 535, 485, 300, C.orange, 20);
  formulaBox(slide, xR + 34, 858, col - 66, 152, "触发与代表", [
    "trigger ⇔ (t mod 5 = 0) ∨ (s ≥ 3)",
    "r_c = arg best_{i∈C_c}(CV_i, f_i)",
    "depth = 10 if stagnant else 3"
  ], "", C.orange);

  panel(slide, xL, 1085, col, 690, "局部代理与伪梯度", C.cyan, "面向小样本昂贵黑箱");
  addText(slide, "局部搜索并不假设真实目标可导。算法复用簇内邻域和历史样本，构建局部 RBF/GP 近似，获得低代价的搜索方向；当样本不足或代理不可靠时，回退到直接搜索/原始解。", xL + 35, 1195, 480, 145, { size: 20 });
  formulaBox(slide, xL + 34, 1368, col - 66, 240, "代理模型目标", [
    "ŷ(x) = Σᵢ λᵢ φ(‖x − xᵢ‖) + p(x)",
    "x_next = arg min_x [ μ(x) − κσ(x) ]",
    "y = F + α·CV,    α ≈ 10·mean(|F|)"
  ], "小样本场景的关键不是全局代理处处准确，而是在候选盆地内快速变准。", C.cyan);
  addText(slide, "工程仿真闭环只作为验证场景之一：参数 → 仿真响应 → 代理更新 → HA 候选 → 真实回评。", xL + 35, 1632, 480, 65, { size: 19, color: C.pale });

  panel(slide, xM, 1085, col, 690, "混合交叉池", C.gold, "显式比例 + 隐性选择压力");
  formulaBox(slide, xM + 34, 1195, col - 66, 270, "三类重组算子", [
    "SBX 60%:   β = (2u)^{1/(η+1)},    u ≤ 0.5",
    "          β = [1/(2(1−u))]^{1/(η+1)},    u > 0.5",
    "Multi-parent 25%:   x_c = Σᵢ ωᵢ xᵢ + ε",
    "DE 15%:   v = x + F₁(x_best−x) + F₂(x_r1−x_r2)"
  ], "", C.gold);
  addText(slide, "设计理由：SBX 保证稳定遗传，多父代融合群体统计信息，DE/current-to-best 提供方向性开发。三者共同缓解“纯随机缺方向”和“单一算子易失效”的问题。", xM + 35, 1488, 480, 95, { size: 20 });
  formulaBox(slide, xM + 34, 1610, col - 66, 128, "多项式变异", [
    "p_m = 1/d,    η_m = 20 + 80·min(1,t/50)",
    "stagnant ⇒ η_m ↓,   mutation radius ↑"
  ], "", C.gold);

  panel(slide, xR, 1085, col, 690, "多臂老虎机自适应策略选择", C.violet, "让搜索器选择随阶段改变");
  formulaBox(slide, xR + 34, 1195, col - 66, 320, "QL / UCB 更新", [
    "ε-greedy:   a ∼ Uniform(A),   if p < ε",
    "             a = arg max_a Q(a),   otherwise",
    "R_t = max(0, f_before − f_after)",
    "Q(a) ← Q(a) + (1/n_a)[R_t − Q(a)]",
    "UCB(a) = Q(a) + c √(log t / n_a)"
  ], "将局部搜索器视作“臂”，用真实改进量学习当前阶段最值得调用的搜索方式。", C.violet);
  addText(slide, "双选择器思想：全局最优个体偏深度开发，普通精英个体偏代理/探索，避免一种经验支配所有个体。", xR + 35, 1545, 480, 78, { size: 20 });

  panel(slide, xL, 1818, col, 757, "约束处理与多样性维护", C.green, "防止不可行和早熟收敛");
  bullets(slide, [
    "约束排序采用可行解优先思想：可行解优于不可行解；同可行比较目标，同不可行比较违反度。",
    "标量局部搜索统一叠加约束惩罚，使无约束局部搜索器也能感知可行性。",
    "重复个体过多时注入随机个体并重置步长，维持后期探索能力。",
    "工程仿真异常、网格失败或无效响应返回大罚值，保证算法流程不断裂。"
  ], xL + 35, 1930, 485, 305, C.green, 20);
  formulaBox(slide, xL + 34, 2308, col - 66, 230, "约束惩罚", [
    "g(x) = value(x) + α·CV(x)",
    "α = 10|value(x)| + 1",
    "CV(x) = Σⱼ max(0, gⱼ(x))"
  ], "与 Deb 可行性规则共同工作：选择时重可行性，局部搜索时用连续惩罚引导方向。", C.green);

  panel(slide, xM, 1818, col, 757, "多目标扩展：NSHA", C.orange, "局部搜索与 Pareto 前沿相容");
  addText(slide, "NSHA 将多目标问题沿 NSGA-III 参考方向分解。每个 niche 选择代表个体，并用对应参考方向构造标量化目标；局部改进后回到多目标选择框架中维护前沿分布。", xM + 35, 1930, 480, 145, { size: 20 });
  formulaBox(slide, xM + 34, 2110, col - 66, 348, "标量化公式", [
    "F_s = F(x) − z*,    ŵ = w / ‖w‖",
    "d₁ = ⟨F_s, ŵ⟩,    d₂ = ‖F_s − d₁ŵ‖",
    "g_PBI = d₁ + θd₂",
    "g_TCH = max_i  w_i |f_i − z_i*|",
    "g_ASF = max_i  (f_i − z_i*) / w_i"
  ], "所有 scalarizer 叠加 value+αCV；ZDT2/ZDT3 暴露了凹/不连续前沿下进一步改进 θ 和 nadir 的必要性。", C.orange);

  panel(slide, xR, 1818, col, 757, "理论解释与验证路径", C.cyan, "为什么它有研究价值");
  bullets(slide, [
    "图式定理扩展：多父代概率采样降低长定义距图式被交叉破坏的概率。",
    "马尔可夫链收敛：精英保留使状态转移满足收敛分析所需的单调保优条件。",
    "消融实验：Full HA / 去聚类 / 去局部搜索 / 标准交叉 / 标准 GA。",
    "统计检验：CEC/ZDT 上结合 IGD、HV、最优值、FEs 与 Wilcoxon 秩和检验。"
  ], xR + 35, 1930, 485, 270, C.cyan, 20);
  formulaBox(slide, xR + 34, 2260, col - 66, 278, "图式与收敛表达", [
    "E[m(H,t+1)] ≥ m(H,t) · f(H)/f̄ · (1−p_d)",
    "Elitism:   best(P_{t+1}) ≤ best(P_t)",
    "P{ lim_{t→∞} P_t ∩ X* ≠ ∅ } = 1"
  ], "海报中只给核心表达，完整证明应放论文正文。", C.cyan);

  panel(slide, 48, 2618, 1704, 817, "阶段实验：ZDT 多目标验证（算法通用性的一部分）", C.gold, "图示为 NSGA-III 与 HA-RBF-PBI 的 Pareto 前沿对比");
  for (let i = 0; i < 3; i += 1) {
    slide.images.add({
      blob: await imageBytes(ZDT[i]),
      contentType: "image/png",
      alt: `ZDT${i + 1} Pareto front comparison`,
      fit: "contain",
      position: { left: 92 + i * 558, top: 2748, width: 502, height: 427 },
    });
    addShape(slide, "rect", 92 + i * 558, 2748, 502, 427, "none", C.gold, 4);
  }
  addText(slide, "结果解读：ZDT1 上 HA-RBF-PBI 的 IGD=6.04e-02，优于 NSGA-III 的 1.21e-01，说明聚类局部搜索能够提升部分前沿逼近。ZDT2/ZDT3 上表现不稳定，反映当前 PBI 局部搜索对凹前沿和不连续前沿仍敏感。该结果不是失败，而是明确指出下一步：动态 nadir、θ 自适应、支配式局部搜索与标量化消融。", 92, 3210, 1580, 80, { size: 21 });
  addText(slide, "工程案例定位：SolidWorks/CAE 只作为昂贵黑箱验证场景，用于检验小样本代理、异常惩罚、真实回评和算法接口的可用性；不作为算法研究对象本身。", 92, 3322, 1580, 55, { size: 21, bold: true, color: C.pale });

  panel(slide, 48, 3478, 839, 912, "已完成工作", C.green, "从算法原型到扩展验证");
  bullets(slide, [
    "实现单目标 HA：聚类局部搜索、混合交叉池、自适应多项式变异与多样性维护。",
    "实现多臂老虎机版本：QL/UCB 根据真实改进量动态选择局部搜索器。",
    "实现 NSHA：参考方向 niche、PBI/Tchebycheff/WS/ASF/AASF/IPBI 标量化接口。",
    "搭建小样本验证流程：LHS 初始采样、GP/KAN-GP 代理、CEI 采集、top-k 真实回评。",
    "整理 ZDT 与工程仿真初步结果，形成下一步消融和统计检验方向。"
  ], 88, 3588, 760, 265, C.green, 20);
  formulaBox(slide, 88, 3892, 762, 438, "当前代码能力", [
    "HA: single-objective + constraints",
    "NSHA: ref-dir niching + scalarizers",
    "Bandit: QL / UCB local-search selector",
    "Surrogate: GP / KAN-GP + CEI acquisition"
  ], "这些模块共同服务于通用算法框架；工程案例只承担黑箱昂贵评估验证。", C.green);

  panel(slide, 913, 3478, 839, 912, "下一步研究重点", C.orange, "让算法故事闭环");
  bullets(slide, [
    "聚类自适应：比较 K-means、MeanShift、DBSCAN 在多峰/高维/噪声场景下的收益与开销。",
    "交叉池自适应：由固定 60/25/15 走向基于反馈的算子概率更新。",
    "多目标局部搜索：改进 PBI 参数、动态理想/最差点、Pareto-Nelder-Mead 和共同下降方向。",
    "理论证明：补全多亲概率采样图式传播模型与精英马尔可夫链收敛条件。",
    "验证体系：CEC + ZDT + 一个工程黑箱案例，主次清晰地证明通用算法能力。"
  ], 953, 3588, 760, 285, C.orange, 20);
  formulaBox(slide, 953, 3892, 762, 438, "论文实验矩阵", [
    "Benchmark: CEC + ZDT + engineering black-box",
    "Ablation: Full / no-cluster / no-LS / no-pool",
    "Metrics: best, FEs, IGD, HV, feasibility rate",
    "Statistics: Wilcoxon rank-sum test"
  ], "主结论应证明通用算法能力，而不是某个具体零件调参成功。", C.orange);

  addShape(slide, "line", 58, 4410, 1684, 0, "none", C.gold, 3);
  addText(slide, "HA / NSHA  |  Clustering Adaptive Local Search  |  Hybrid Crossover Pool  |  Bandit Local Search Selection", 62, 4430, 1180, 32, { size: 22, color: C.pale });
  addText(slide, "2026 Academic Poster", 1260, 4430, 300, 32, { size: 22, color: C.gold });

  await writeBlob(PREVIEW, await prs.export({ slide, format: "png", scale: 1 }));
  await fs.writeFile(LAYOUT, await (await slide.export({ format: "layout" })).text(), "utf8");
  const pptx = await PresentationFile.exportPptx(prs);
  await pptx.save(PPTX);
  console.log(PPTX);
  console.log(PREVIEW);
  console.log(LAYOUT);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
