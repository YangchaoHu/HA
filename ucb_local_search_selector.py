"""
ucb_local_search_selector.py  ──  向后兼容入口
-----------------------------------------------

本文件已整合至 selectors.py，此处仅作重导出 stub，
现有代码（run_ucb_only.py 等）无需任何修改。

推荐直接从 selectors.py 导入：
    from selectors import UCBSelectorV1, UCBSelectorV2

版本说明：
    UCBLocalSearchSelector = UCBSelectorV2（改进版，绝对改进奖励）
    如需原始版请使用：from selectors import UCBSelectorV1
"""

from ha_selectors import (  # noqa: F401
    UCBSelectorV1,
    UCBSelectorV2,
    UCBLocalSearchSelector,   # = UCBSelectorV2
)

__all__ = ["UCBLocalSearchSelector", "UCBSelectorV1", "UCBSelectorV2"]
