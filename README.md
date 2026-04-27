# HA — 混合进化优化算法

**HA（Hybrid Algorithm）** 是一种结合聚类小生境、多策略局部搜索和遗传操作的混合进化优化算法。  
本仓库包含核心算法实现、多臂老虎机（Bandit）自适应变体、代理模型对比实验，以及对两个工程仿真问题（QuickSimu1/QuickSimu2）的完整实验结果。

---

## 目录结构

```
HA/
├── ha_Nelder_Mead.py            # HA 核心实现（聚类 + 局部搜索 + 遗传操作）
├── ha_bandit.py                 # Bandit 自适应版本（HA_QL / HA_UCB）
├── ha_selectors.py              # Q-learning / UCB 选择器（V1 / V2）
├── ha_index.py                  # 统一导出入口
├── selectors.py                 # 选择器兼容导出
├── local_search_selector.py     # 兼容旧导入路径
├── ucb_local_search_selector.py # 兼容旧导入路径
├── ha.py                        # HA 原始版（早期实现）
├── ha_rbf.py                    # HA-RBF 单独实现
├── ha_original.py               # HA 消融对比版本
├── problem.py                   # 工程优化问题定义（QuickSimu1 / QuickSimu2）
├── experiment_runner.py         # 主实验驱动（批量运行、绘图、CSV 汇总）
├── evaluate_metrics.py          # 指标评估与可视化
├── run_bandit.py                # 快速运行 Bandit 对比脚本
├── random_eval_quick_simu.py    # 随机基线评估
├── proxy_models_jsons/          # 预生成 LHS 采样数据（JSON 格式，供代理模型训练）
│   ├── quick_simu_1/            # QuickSimu1 各 seed 各采样数 JSON
│   └── quick_simu_2/            # QuickSimu2 各 seed 各采样数 JSON
├── compare_surrogate_models/    # 代理模型对比实验（Kriging / RBF / Poly / KAN / KAN-GP）
│   ├── run_surrogate_ha_eval.py # 主入口：建代理 → HA 优化 → 真实回评
│   ├── run_gp_quicksimu_trace.py# 真实 GP 运行轨迹（HA 直接优化仿真）
│   ├── fit_surrogates_quicksimu1.py # 拟合精度对比
│   ├── plot_gp_vs_surrogate_quicksimu1.py # 绘图：GP 轨迹 vs 代理优化结果
│   ├── results_ha/              # 真实 HA 三 seed 运行结果 CSV
│   └── results_surrogate/       # 五类代理模型跨 seed 汇总 CSV
├── results/                     # HA 各方法对比实验结果（CSV + PNG）
├── experiment_results/          # Nelder-Mead 版 HA 实验结果
├── experiment_results_nelder_mead/ # Nelder-Mead 扩展对比
├── simulation_models/           # 仿真模型文件
├── README_HA.md                 # 代码整理说明（Bandit 版本详细说明）
└── 指标说明.md                  # 局部搜索效率指标定义
```

---

## 快速开始

### 环境依赖

```bash
pip install numpy scipy scikit-learn pymoo
# 可选（KAN / KAN-GP 代理模型）：
pip install torch gpytorch
```

### 1. 运行 Bandit 版本对比（推荐入口）

```bash
# 运行 HA_QL 与 HA_UCB 全量对比
python run_bandit.py

# 仅运行指定版本
python run_bandit.py --versions QL-V2,UCB-V2
```

### 2. 运行核心实验（全问题全方法）

```bash
python experiment_runner.py
```

### 3. 代理模型对比实验

```bash
# 全量运行（QuickSimu1 + QuickSimu2 × 3 seed × 15 采样规模 × 5 代理类型）
python compare_surrogate_models/run_surrogate_ha_eval.py

# 冒烟测试（快速验证）
python compare_surrogate_models/run_surrogate_ha_eval.py \
    --problems QuickSimu1 --counts 500 --pop-size 10 --n-gen 5

# 并行代理阶段（真实仿真仍串行）
python compare_surrogate_models/run_surrogate_ha_eval.py --jobs 4
```

### 4. 绘制 GP vs 代理模型对比图

```bash
python compare_surrogate_models/plot_gp_vs_surrogate_quicksimu1.py
```

---

## 算法结构

```
HA
├── 聚类（K-means / MeanShift / DBSCAN）→ 小生境划分
├── 局部搜索（可动态切换）
│   ├── Nelder-Mead          （精细化开发，用于全局最优个体）
│   ├── L-BFGS-B             （精细化开发，用于全局最优个体）
│   ├── RBF  ★ 当前最优      （探索性代理模型，用于普通精英个体）
│   └── GP   ★ 当前最优      （探索性代理模型，用于普通精英个体）
└── 遗传操作（变异 + 选择）
```

**Bandit 自适应版本**（`ha_bandit.py`）在运行时用 Q-learning（`HA_QL`）或 UCB（`HA_UCB`）动态决定每个个体使用哪种局部搜索策略：

| 类名 | 奖励机制 | 选择器结构 |
|------|----------|------------|
| `HA_QL_V1` | 改进量 / FE 数 | 单选择器（共享） |
| `HA_QL_V2`（默认 `HA_QL`）| `max(0, f_before − f_after)` | 双选择器（全局最优 / 普通精英分离）|
| `HA_UCB_V1` | 改进量 / FE 数 | 单选择器（共享） |
| `HA_UCB_V2`（默认 `HA_UCB`）| `max(0, f_before − f_after)` | 双选择器（全局最优 / 普通精英分离）|

---

## 实验结论

> **当前阶段，RBF（径向基函数）与 GP（高斯过程）作为局部搜索代理模型，在所有对比方法中综合表现最优。**

具体表现：

- **RBF**：在 QuickSimu1 / QuickSimu2 两个工程问题上，随训练样本增加收敛稳定，真实回评目标值最低，适合中等规模数据（500–1000 样本）。
- **GP（Kriging）**：在样本量充足（≥ 700）时，拟合精度高，HA 优化后真实回评结果与 RBF 并列最优，置信区间窄，适用于精度要求高的场景。
- **Polynomial**：拟合简单、速度快，但在非线性约束区域误差较大，最终目标值明显劣于 RBF/GP。
- **KAN / KAN-GP**：依赖 PyTorch，小样本下不稳定，大样本下有潜力，当前实验规模下略逊于 RBF/GP。

结论来源：`compare_surrogate_models/results_surrogate/` 下的汇总 CSV（跨 seed 均值），以及 `compare_surrogate_models/results_ha/` 中的真实 HA 运行对照数据。

---

## 问题定义

| 问题 | 决策变量数 | 目标 | 约束 | 位移限制 |
|------|-----------|------|------|----------|
| `QuickSimu1` | 3 | min volume | `abs(max_disp_x) ≤ 限制` | 0.02787 |
| `QuickSimu2` | 2 | min volume | `abs(max_disp_y) ≤ 限制` | 3.616e-7 |

详见 [`problem.py`](problem.py)。

---

## 代码导入方式

```python
# Bandit HA（推荐）
from ha_bandit import HA_QL, HA_UCB
from ha_bandit import HA_QL_V1, HA_QL_V2, HA_UCB_V1, HA_UCB_V2

# 统一入口
from ha_index import HA_QL, HA_UCB

# 基础 HA（无 Bandit）
from ha_Nelder_Mead import HA
```

---

## 参数说明（`experiment_runner.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `POP_SIZE` | 50 | 种群大小 |
| `N_GEN` | 100 | 最大迭代代数 |
| `seed` | 多组 | 随机种子（用于重复性验证）|

代理模型实验参数（`run_surrogate_ha_eval.py`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pop-size` | 50 | HA 种群大小 |
| `--n-gen` | 30 | HA 迭代代数 |
| `--problems` | QuickSimu1 QuickSimu2 | 测试问题 |
| `--seeds` | 2026 2027 2028 | 随机种子 |
| `--counts` | 100~1500（步长100）| 训练样本数 |
| `--model-types` | 全部5种 | 代理模型类型 |
| `--jobs` | 1 | 代理阶段并行进程数 |

---

## 相关文档

- [`README_HA.md`](README_HA.md) — Bandit 版本详细说明与维护建议
- [`指标说明.md`](指标说明.md) — 局部搜索效率指标定义与解读
