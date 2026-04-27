"""
ha_bandit.py
------------

所有基于多臂老虎机（Bandit）的局部搜索动态选择 HA 类（汇总文件）。

├── ε-贪婪 Q-learning 版本
│   └── HA_QL  ── 双选择器（全局最优/普通精英分离），绝对改进奖励
│
└── UCB1 版本
    └── HA_UCB ── 双选择器（全局最优/普通精英分离），绝对改进奖励

HA_QL / HA_UCB 特性：
    - 全局最优个体 → selector_global（Nelder-Mead + L-BFGS-B，精细化开发）
    - 普通精英个体 → selector_elite（rbf + gp，探索性代理模型）
    - 奖励公式：max(0, f_before - f_after)
    - 根据是否全局最优自动路由选择器（上下文感知）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ha_Nelder_Mead import HA as HA_Base
from ha_selectors import (
    QLSelectorV2,
    UCBSelectorV2,
)

logger = logging.getLogger(__name__)
ArrayLike = NDArray[np.floating]


# ============================================================================
# 默认动作集（按上下文分离）
# ============================================================================

_GLOBAL_BEST_ACTIONS: List[str] = ["Nelder-Mead", "L-BFGS-B"]  # 精细化开发
_ELITE_ACTIONS:       List[str] = ["rbf", "gp"]                 # 探索性代理模型


# ============================================================================
# 公共工具：全局最优判定
# ============================================================================

def _is_global_best(ha_instance: HA_Base, f_now: float) -> bool:
    """判断 f_now 是否对应当前种群的全局最优个体。"""
    try:
        current_best = float(np.min(ha_instance.pop.get("F")))
        tol = abs(current_best) * 1e-9 + 1e-15
        return f_now <= current_best + tol
    except Exception:
        return False


# ============================================================================
# 第一部分：HA-QL（ε-贪婪 Q-learning）
# ============================================================================

class HA_QL(HA_Base):
    """
    HA-QL：双选择器（上下文感知）+ 绝对改进奖励。

    - 全局最优个体 → selector_global（Nelder-Mead + L-BFGS-B，精细化开发）；
    - 普通精英个体 → selector_elite（rbf + gp，探索性代理模型）；
    - 奖励 = max(0, f_before - f_after)；
    - 通过比较 y0 与种群全局最优适应度自动路由。

    Args:
        global_best_actions: 全局最优算法集，默认 ["Nelder-Mead","L-BFGS-B"]。
        elite_actions: 普通精英算法集，默认 ["rbf","gp"]。
        ls_actions: 已弃用，兼容旧代码（自动拆分为两组）。
        epsilon/epsilon_decay/epsilon_min/convergence_threshold: QL 参数。
    """

    def __init__(
        self,
        global_best_actions: Optional[List[str]] = None,
        elite_actions: Optional[List[str]] = None,
        ls_actions: Optional[List[str]] = None,   # 向后兼容
        epsilon: float = 1.0,
        epsilon_decay: float = 0.95,
        epsilon_min: float = 0.05,
        convergence_threshold: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("method", "Nelder-Mead")
        super().__init__(**kwargs)

        if ls_actions and not global_best_actions and not elite_actions:
            mid = max(1, len(ls_actions) // 2)
            global_best_actions = ls_actions[mid:]
            elite_actions = ls_actions[:mid]

        self.global_best_actions = global_best_actions or _GLOBAL_BEST_ACTIONS[:]
        self.elite_actions        = elite_actions        or _ELITE_ACTIONS[:]

        ql_kw = dict(epsilon=epsilon, epsilon_decay=epsilon_decay,
                     epsilon_min=epsilon_min, convergence_threshold=convergence_threshold,
                     seed=self.seed)
        self.selector_global = QLSelectorV2(actions=self.global_best_actions, **ql_kw)
        self.selector_elite  = QLSelectorV2(actions=self.elite_actions,        **ql_kw)

        self._ql_call_count: int = 0
        self._global_call_count: int = 0
        self._elite_call_count: int = 0

    def _local_search(self, x0: ArrayLike, y0: Union[float, ArrayLike], maxiter: int = 1) -> ArrayLike:
        self._ql_call_count += 1
        f_before = float(np.atleast_1d(y0)[0])
        is_gb = _is_global_best(self, f_before)

        if is_gb:
            selector = self.selector_global
            self._global_call_count += 1
        else:
            selector = self.selector_elite
            self._elite_call_count += 1

        action = selector.select_action()
        fes_before = int(getattr(self.problem, "fes", 0))

        orig_method, self.method = self.method, action
        extra_eval = 0
        try:
            x_result = super()._local_search(x0, y0, maxiter)
        except Exception as exc:
            logger.warning("HA_QL: '%s' 失败（%s），回退 x0", action, exc)
            x_result = x0.copy()
        finally:
            self.method = orig_method

        fes_after = int(getattr(self.problem, "fes", 0))
        delta_fes = max(0, fes_after - fes_before)

        hist = self.history.get(x_result)
        if hist is not None:
            f_after = float(np.atleast_1d(hist[0])[0])
        else:
            fa, _ = self.evaluate_fitness_cv(x_result)
            f_after = float(np.atleast_1d(fa)[0])
            extra_eval = 1
        delta_fes += extra_eval

        reward = QLSelectorV2.compute_reward(f_before, f_after, minimize=True)
        selector.update_q_value(action, reward)
        selector.decay_epsilon()

        logger.debug(
            "HA_QL [%d, %s]: %s | f:%.4e→%.4e | R:%.4e | ε:%.4f",
            self._ql_call_count, "gb" if is_gb else "el",
            action, f_before, f_after, reward, selector.epsilon,
        )
        return x_result

    def get_selector_stats(self) -> Dict:
        return {
            "ql_call_count":    self._ql_call_count,
            "global_call_count": self._global_call_count,
            "elite_call_count":  self._elite_call_count,
            "global": self.selector_global.get_stats(),
            "elite":  self.selector_elite.get_stats(),
        }

    def print_selector_stats(self) -> None:
        s = self.get_selector_stats()
        print("\n" + "=" * 60)
        print(f"HA-QL 统计（总:{s['ql_call_count']}，global:{s['global_call_count']}，elite:{s['elite_call_count']}）")
        for ctx, label, actions in [
            ("global", "全局最优路径", self.global_best_actions),
            ("elite",  "普通精英路径", self.elite_actions),
        ]:
            cs = s[ctx]
            print(f"\n  [{label}] ε={cs['epsilon']:.4f}，最优={cs['best_action']}")
            print(f"  {'算法':<15} {'Q值':>14} {'次数':>8}")
            print(f"  {'-'*15} {'-'*14} {'-'*8}")
            for a in actions:
                mk = " ◀" if a == cs["best_action"] else ""
                print(f"  {a:<15} {cs['q_table'][a]:>14.6e} {cs['n_table'][a]:>8d}{mk}")
        print("=" * 60)


# ============================================================================
# 第二部分：HA-UCB（UCB1）
# ============================================================================

class HA_UCB(HA_Base):
    """
    HA-UCB：双选择器（上下文感知）+ 绝对改进奖励。

    - 全局最优个体 → selector_global（Nelder-Mead + L-BFGS-B，精细化开发）；
    - 普通精英个体 → selector_elite（rbf + gp，探索性代理模型）；
    - 原始奖励 = max(0, f_before - f_after)；
    - baseline_r 在线归一化（EMA）缓解早期奖励主导问题。

    Args:
        global_best_actions: 全局最优算法集，默认 ["Nelder-Mead","L-BFGS-B"]。
        elite_actions: 普通精英算法集，默认 ["rbf","gp"]。
        ls_actions: 已弃用，兼容旧代码（自动拆分为两组）。
        c/alpha/reward_scale_beta: UCB1 参数。
    """

    def __init__(
        self,
        global_best_actions: Optional[List[str]] = None,
        elite_actions: Optional[List[str]] = None,
        ls_actions: Optional[List[str]] = None,   # 向后兼容
        c: float = 1.0,
        alpha: float = 0.1,
        reward_scale_beta: float = 0.1,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("method", "Nelder-Mead")
        super().__init__(**kwargs)

        if ls_actions and not global_best_actions and not elite_actions:
            mid = max(1, len(ls_actions) // 2)
            global_best_actions = ls_actions[mid:]
            elite_actions = ls_actions[:mid]

        self.global_best_actions = global_best_actions or _GLOBAL_BEST_ACTIONS[:]
        self.elite_actions        = elite_actions        or _ELITE_ACTIONS[:]

        ucb_kw = dict(c=c, alpha=alpha, reward_scale_beta=reward_scale_beta, seed=self.seed)
        self.selector_global = UCBSelectorV2(actions=self.global_best_actions, **ucb_kw)
        self.selector_elite  = UCBSelectorV2(actions=self.elite_actions,        **ucb_kw)

        self._ucb_call_count: int = 0
        self._global_call_count: int = 0
        self._elite_call_count: int = 0

    def _local_search(self, x0: ArrayLike, y0: Union[float, ArrayLike], maxiter: int = 1) -> ArrayLike:
        self._ucb_call_count += 1
        f_before = float(np.atleast_1d(y0)[0])
        is_gb = _is_global_best(self, f_before)

        if is_gb:
            selector = self.selector_global
            self._global_call_count += 1
        else:
            selector = self.selector_elite
            self._elite_call_count += 1

        action = selector.select_action()
        fes_before = int(getattr(self.problem, "fes", 0))

        orig_method, self.method = self.method, action
        extra_eval = 0
        try:
            x_result = super()._local_search(x0, y0, maxiter)
        except Exception as exc:
            logger.warning("HA_UCB: '%s' 失败（%s），回退 x0", action, exc)
            x_result = x0.copy()
        finally:
            self.method = orig_method

        fes_after = int(getattr(self.problem, "fes", 0))
        delta_fes = max(0, fes_after - fes_before)

        hist = self.history.get(x_result)
        if hist is not None:
            f_after = float(np.atleast_1d(hist[0])[0])
        else:
            fa, _ = self.evaluate_fitness_cv(x_result)
            f_after = float(np.atleast_1d(fa)[0])
            extra_eval = 1
        delta_fes += extra_eval

        raw_reward = UCBSelectorV2.compute_raw_reward(f_before, f_after, minimize=True)
        norm_reward = selector.update(action, raw_reward)

        logger.debug(
            "HA_UCB [%d, %s]: %s | f:%.4e→%.4e | raw:%.4e | norm:%.4f",
            self._ucb_call_count, "gb" if is_gb else "el",
            action, f_before, f_after, raw_reward, norm_reward,
        )
        return x_result

    def get_selector_stats(self) -> Dict:
        return {
            "ucb_call_count":    self._ucb_call_count,
            "global_call_count": self._global_call_count,
            "elite_call_count":  self._elite_call_count,
            "global": self.selector_global.get_stats(),
            "elite":  self.selector_elite.get_stats(),
        }

    def print_selector_stats(self) -> None:
        s = self.get_selector_stats()
        print("\n" + "=" * 60)
        print(f"HA-UCB 统计（总:{s['ucb_call_count']}，global:{s['global_call_count']}，elite:{s['elite_call_count']}）")
        for ctx, label, actions in [
            ("global", "全局最优路径", self.global_best_actions),
            ("elite",  "普通精英路径", self.elite_actions),
        ]:
            cs = s[ctx]
            print(
                f"\n  [{label}] c={cs['c']:.2f}, α={cs['alpha']:.2f}, "
                f"baseline_R={cs['baseline_r']:.4e}，最优={cs['best_action']}"
            )
            print(f"  {'算法':<15} {'Q值':>14} {'次数':>8}")
            print(f"  {'-'*15} {'-'*14} {'-'*8}")
            for a in actions:
                mk = " ◀" if a == cs["best_action"] else ""
                print(f"  {a:<15} {cs['q_table'][a]:>14.6e} {cs['n_table'][a]:>8d}{mk}")
        print("=" * 60)

