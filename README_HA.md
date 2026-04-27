# HA 代码整理说明

本项目中的 HA（Hybrid Algorithm）相关代码已经按“核心实现 / 兼容入口 / 运行脚本”整理，目标是：

- 保持旧脚本可继续运行；
- 将多版本 Bandit（QL/UCB，V1/V2）集中管理；
- 降低后续对比实验和维护成本。

## 1. 目录与职责

- `ha_Nelder_Mead.py`
  - HA 基础实现（聚类、小生境、局部搜索、遗传操作等）。
  - 可视作 Bandit 扩展的父类。

- `ha_bandit.py`
  - Bandit 版本 HA 的**核心实现文件**（已集中）：
  - `HA_QL_V1`：QL 原始版（单选择器 + 改进量/FE）
  - `HA_QL_V2`：QL 改进版（双选择器 + 绝对改进量）
  - `HA_UCB_V1`：UCB 原始版（单选择器 + 改进量/FE）
  - `HA_UCB_V2`：UCB 改进版（双选择器 + 绝对改进量）
  - 默认别名：
    - `HA_QL = HA_QL_V2`
    - `HA_UCB = HA_UCB_V2`

- `selectors.py`
  - QL/UCB 选择器实现集中地（V1/V2），被 `ha_bandit.py` 使用。
  - 包含：
    - `QLSelectorV1` / `QLSelectorV2`
    - `UCBSelectorV1` / `UCBSelectorV2`

- `ha_index.py`
  - 新增的统一导出入口，便于脚本统一 import。

- 兼容入口（不承载核心逻辑）：
  - `local_search_selector.py`
  - `ucb_local_search_selector.py`
  - 这些文件保留是为了兼容旧导入路径。

## 2. 当前推荐导入方式

优先从 `ha_bandit.py` 或 `ha_index.py` 导入：

```python
from ha_bandit import HA_QL_V1, HA_QL_V2, HA_UCB_V1, HA_UCB_V2
# 或
from ha_index import HA_QL_V1, HA_QL_V2, HA_UCB_V1, HA_UCB_V2
```

## 3. 运行方式

### 3.1 跑四个版本对比（推荐）

```bash
python run_bandit.py
```

只跑指定版本：

```bash
python run_bandit.py --versions QL-V2,UCB-V2
python run_bandit.py --versions QL-V1,UCB-V1
```

### 3.2 单独跑 QL 或 UCB（兼容脚本）

```bash
python run_ql_only.py
python run_ucb_only.py
```

> 这两个脚本已改为直接从 `ha_bandit.py` 导入改进版类，不依赖中间 stub。

### 3.3 画图

```bash
python evaluate_metrics.py
```

## 4. V1 与 V2 的核心差异

- 奖励定义：
  - V1：`(f_before - f_after) / delta_FEs`
  - V2：`max(0, f_before - f_after)`

- 策略结构：
  - V1：单选择器（所有个体共享）
  - V2：双选择器（全局最优个体 / 普通精英个体分开建模）

## 5. 维护建议

- 新增 Bandit 变体时，优先在 `ha_bandit.py` 与 `selectors.py` 扩展；
- 兼容入口文件只做重导出，不建议继续写业务逻辑；
- 结果文件命名建议保持 `{problem}_{method}_summary.csv`，避免绘图脚本额外适配。

