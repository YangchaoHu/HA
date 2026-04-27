"""
selectors.py
------------

局部搜索策略选择器汇总文件（QL 和 UCB，原始版与改进版）。

├── ε-贪婪 Q-learning 选择器
│   ├── QLSelectorV1  ── 原始版：奖励 = 改进量 / delta_FEs
│   └── QLSelectorV2  ── 改进版：奖励 = 绝对改进量（推荐）
│
└── UCB1 选择器
    ├── UCBSelectorV1 ── 原始版：原始奖励 = 改进量 / delta_fes
    └── UCBSelectorV2 ── 改进版：原始奖励 = 绝对改进量（推荐）

版本差异对比：
┌──────────────┬─────────────────────────────────┬───────────────────────────────┐
│ 特性          │ V1（原始版）                      │ V2（改进版）                   │
├──────────────┼─────────────────────────────────┼───────────────────────────────┤
│ 奖励公式      │ (f_before - f_after) / delta_FEs │ max(0, f_before - f_after)    │
│ FEs 惩罚      │ 有（惩罚高消耗算法如 gp/rbf）      │ 无（关注总改进量）              │
│ 非平稳适应    │ 较差（gp 早期大奖励会主导 Q 值）   │ UCBSelectorV2 通过 baseline_r │
│              │                                  │ EMA 归一化缓解此问题           │
└──────────────┴─────────────────────────────────┴───────────────────────────────┘

向后兼容别名（供现有代码直接使用）：
    LocalSearchSelector    = QLSelectorV2
    UCBLocalSearchSelector = UCBSelectorV2
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# 第一部分：ε-贪婪 Q-learning 选择器
# ============================================================================

class QLSelectorV1:
    """
    原始 ε-贪婪 Q-learning 选择器。

    奖励公式：R_t = (f_before - f_after) / delta_FEs
    ── 单位 FE 改进量，鼓励"便宜且有效"的算法。
    ── 问题：系统性惩罚 gp/rbf 等每次消耗 FE 多但改进大的算法。

    Q 值更新：增量均值法（Sample Average），无固定学习率。
    """

    def __init__(
        self,
        actions: List[str],
        epsilon: float = 1.0,
        epsilon_decay: float = 0.95,
        epsilon_min: float = 0.05,
        convergence_threshold: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        if not actions:
            raise ValueError("actions 不能为空")
        if not (0 < epsilon <= 1.0):
            raise ValueError(f"epsilon 必须在 (0,1]，当前: {epsilon}")

        self.actions: List[str] = list(actions)
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.convergence_threshold: float = convergence_threshold

        self.q_table: Dict[str, float] = {a: 0.0 for a in self.actions}
        self.n_table: Dict[str, int]   = {a: 0   for a in self.actions}
        self.converged_action: Optional[str] = None
        self._rng = np.random.default_rng(seed)
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # 收敛判定
    # ------------------------------------------------------------------

    def is_converged(self) -> bool:
        """当 ε 达到最小、所有动作均试过、且最优 Q 值明显领先时判定收敛。"""
        if self.epsilon > self.epsilon_min + 1e-9:
            return False
        if any(n == 0 for n in self.n_table.values()):
            return False
        if len(self.actions) == 1:
            self.converged_action = self.actions[0]
            return True
        q_sorted = sorted(self.q_table.values(), reverse=True)
        if q_sorted[0] - q_sorted[1] > self.convergence_threshold:
            self.converged_action = max(self.q_table, key=self.q_table.get)
            return True
        return False

    # ------------------------------------------------------------------
    # 策略选择（ε-greedy）
    # ------------------------------------------------------------------

    def select_action(self) -> str:
        if self.is_converged() and self.converged_action is not None:
            return self.converged_action
        p = self._rng.uniform(0.0, 1.0)
        if p < self.epsilon:
            return str(self._rng.choice(self.actions))
        max_q = max(self.q_table.values())
        best = [a for a, q in self.q_table.items() if q >= max_q - 1e-12]
        return str(self._rng.choice(best))

    # ------------------------------------------------------------------
    # 奖励计算（V1：除以 delta_FEs）
    # ------------------------------------------------------------------

    @staticmethod
    def compute_reward(
        f_best: float,
        f_prime_best: float,
        delta_FEs: int = 1,
        minimize: bool = True,
    ) -> float:
        """
        V1 奖励：单位 FE 改进量。

        R_t = (f_best - f'_best) / delta_FEs   （最小化）
        """
        if delta_FEs <= 0:
            return 0.0
        if minimize:
            return (f_best - f_prime_best) / float(delta_FEs)
        return (f_prime_best - f_best) / float(delta_FEs)

    # ------------------------------------------------------------------
    # Q 值更新（增量均值法）
    # ------------------------------------------------------------------

    def update_q_value(self, action: str, reward: float) -> None:
        if action not in self.q_table:
            raise ValueError(f"未知动作: {action}")
        self.n_table[action] += 1
        n = self.n_table[action]
        self.q_table[action] += (1.0 / n) * (reward - self.q_table[action])
        self._history.append({
            "action": action, "reward": reward,
            "q_after": self.q_table[action], "n": n, "epsilon": self.epsilon,
        })

    # ------------------------------------------------------------------
    # ε 衰减
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    # ------------------------------------------------------------------
    # 统计与工具
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        return {
            "q_table": dict(self.q_table),
            "n_table": dict(self.n_table),
            "epsilon": self.epsilon,
            "total_steps": sum(self.n_table.values()),
            "best_action": max(self.q_table, key=self.q_table.get),
            "converged": self.is_converged(),
            "converged_action": self.converged_action,
        }

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def reset(self) -> None:
        self.q_table = {a: 0.0 for a in self.actions}
        self.n_table = {a: 0   for a in self.actions}
        self.converged_action = None
        self._history.clear()

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"{self.__class__.__name__}("
            f"actions={self.actions}, ε={self.epsilon:.4f}, "
            f"best='{s['best_action']}', steps={s['total_steps']})"
        )


class QLSelectorV2(QLSelectorV1):
    """
    改进 ε-贪婪 Q-learning 选择器。

    奖励公式：R_t = max(0, f_before - f_after)   （绝对改进量）
    ── 移除 FEs 惩罚，关注总改进量而非单位 FE 效率。
    ── 在固定总 FE 预算下，绝对改进量是更合理的优化目标。

    Q 值更新：同 V1（增量均值法）。
    """

    @staticmethod
    def compute_reward(
        f_best: float,
        f_prime_best: float,
        delta_FEs: int = 0,
        minimize: bool = True,
    ) -> float:
        """
        V2 奖励：绝对改进量（不除以 FEs）。

        R_t = max(0, f_best - f'_best)   （最小化）
        """
        if minimize:
            return max(0.0, f_best - f_prime_best)
        return max(0.0, f_prime_best - f_best)


# ============================================================================
# 第二部分：UCB1 选择器
# ============================================================================

class UCBSelectorV1:
    """
    原始 UCB1 选择器。

    原始奖励：R_t = (f_before - f_after) / delta_fes（单位 FE 改进量）
    Q 值更新：固定学习率 alpha（EMA），适应非平稳环境。
    在线归一化：通过 baseline_r 对奖励缩放，缓解早期大奖励主导问题。
    """

    def __init__(
        self,
        actions: List[str],
        c: float = 1.0,
        alpha: float = 0.1,
        reward_scale_beta: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        if not actions:
            raise ValueError("actions 不能为空")
        if c < 0:
            raise ValueError("c 不能为负数")
        if not (0 < alpha <= 1):
            raise ValueError("alpha 必须在 (0,1]")
        if not (0 < reward_scale_beta <= 1):
            raise ValueError("reward_scale_beta 必须在 (0,1]")

        self.actions: List[str] = list(actions)
        self.c: float = c
        self.alpha: float = alpha
        self.reward_scale_beta: float = reward_scale_beta

        self.q_table: Dict[str, float] = {a: 0.0 for a in self.actions}
        self.n_table: Dict[str, int]   = {a: 0   for a in self.actions}
        self.total_steps: int = 0
        self.baseline_r: float = 1.0
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 策略选择（UCB1 + 冷启动）
    # ------------------------------------------------------------------

    def select_action(self) -> str:
        untried = [a for a in self.actions if self.n_table[a] == 0]
        if untried:
            return str(self._rng.choice(untried))
        log_total = math.log(max(1, self.total_steps))
        scores = {
            a: self.q_table[a] + self.c * math.sqrt(log_total / self.n_table[a])
            for a in self.actions
        }
        max_s = max(scores.values())
        candidates = [a for a, s in scores.items() if abs(s - max_s) <= 1e-12]
        return str(self._rng.choice(candidates))

    # ------------------------------------------------------------------
    # 奖励计算（V1：除以 delta_fes）
    # ------------------------------------------------------------------

    @staticmethod
    def compute_raw_reward(
        f_before: float,
        f_after: float,
        delta_fes: int = 1,
        minimize: bool = True,
    ) -> float:
        """
        V1 原始奖励：单位 FE 改进量。

        R_t = (f_before - f_after) / delta_fes   （最小化）
        """
        if delta_fes <= 0:
            return 0.0
        if minimize:
            return (f_before - f_after) / float(delta_fes)
        return (f_after - f_before) / float(delta_fes)

    # ------------------------------------------------------------------
    # Q 值更新（固定学习率 EMA + baseline_r 归一化）
    # ------------------------------------------------------------------

    def update(self, action: str, raw_reward: float) -> float:
        """
        归一化 + EMA 更新。

        1) R_norm = raw_reward / (baseline_r + 1e-8)
        2) baseline_r ← EMA(|raw_reward|)
        3) Q(a) ← Q(a) + alpha * (R_norm - Q(a))

        Returns:
            归一化后的奖励 R_norm（便于日志记录）。
        """
        if action not in self.q_table:
            raise ValueError(f"未知动作: {action}")
        reward_norm = float(raw_reward) / (self.baseline_r + 1e-8)
        abs_r = abs(float(raw_reward))
        self.baseline_r = (
            (1.0 - self.reward_scale_beta) * self.baseline_r
            + self.reward_scale_beta * abs_r
        )
        self.baseline_r = max(self.baseline_r, 1e-8)
        self.q_table[action] += self.alpha * (reward_norm - self.q_table[action])
        self.n_table[action] += 1
        self.total_steps += 1
        return reward_norm

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        return {
            "q_table": dict(self.q_table),
            "n_table": dict(self.n_table),
            "total_steps": self.total_steps,
            "baseline_r": self.baseline_r,
            "best_action": max(self.q_table, key=self.q_table.get),
            "c": self.c,
            "alpha": self.alpha,
            "reward_scale_beta": self.reward_scale_beta,
        }

    def reset(self) -> None:
        self.q_table = {a: 0.0 for a in self.actions}
        self.n_table = {a: 0   for a in self.actions}
        self.total_steps = 0
        self.baseline_r = 1.0

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"{self.__class__.__name__}("
            f"actions={self.actions}, c={self.c}, α={self.alpha}, "
            f"best='{s['best_action']}', steps={self.total_steps})"
        )


class UCBSelectorV2(UCBSelectorV1):
    """
    改进 UCB1 选择器。

    原始奖励：R_t = max(0, f_before - f_after)   （绝对改进量）
    ── 移除 FEs 惩罚，其余（EMA 更新、baseline_r 归一化）与 V1 相同。
    """

    @staticmethod
    def compute_raw_reward(
        f_before: float,
        f_after: float,
        delta_fes: int = 0,
        minimize: bool = True,
    ) -> float:
        """
        V2 原始奖励：绝对改进量（不除以 FEs）。

        R_t = max(0, f_before - f_after)   （最小化）
        """
        if minimize:
            return max(0.0, f_before - f_after)
        return max(0.0, f_after - f_before)


# ============================================================================
# 向后兼容别名（现有代码无需修改）
# ============================================================================

#: 供 ha_ql.py / run_ql_only.py 等旧代码直接使用
LocalSearchSelector = QLSelectorV2

#: 供 ha_ucb.py / run_ucb_only.py 等旧代码直接使用
UCBLocalSearchSelector = UCBSelectorV2


# ============================================================================
# 独立运行演示
# ============================================================================

if __name__ == "__main__":
    import numpy as np

    print("=" * 70)
    print("V1 vs V2 奖励对比演示（相同场景，不同奖励公式）")
    print("=" * 70)

    actions = ["Nelder-Mead", "gp", "rbf"]
    rng = np.random.default_rng(42)

    # 模拟一次局部搜索：gp 每次改进 1.0，消耗 20 FEs；Nelder-Mead 改进 0.3，消耗 3 FEs
    scenarios = [
        ("gp",          1.0, 20),
        ("Nelder-Mead", 0.3,  3),
        ("rbf",         0.5, 10),
    ]

    print(f"\n{'算法':<15} {'改进量':>8} {'FEs':>5} {'V1奖励(改进/FEs)':>18} {'V2奖励(绝对改进)':>18}")
    print("-" * 70)
    for alg, improvement, fes in scenarios:
        f_b, f_a = 10.0, 10.0 - improvement
        v1 = QLSelectorV1.compute_reward(f_b, f_a, fes)
        v2 = QLSelectorV2.compute_reward(f_b, f_a, fes)
        print(f"{alg:<15} {improvement:>8.3f} {fes:>5d} {v1:>18.6f} {v2:>18.6f}")

    print("\n说明：V1 使 gp（高消耗）奖励低于 Nelder-Mead，导致 Q 表偏向后者。")
    print("      V2 关注绝对改进，gp 的 1.0 改进量正确地优于 Nelder-Mead 的 0.3。")
