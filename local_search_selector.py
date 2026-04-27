"""
local_search_selector.py  ──  向后兼容入口
------------------------------------------

本文件已整合至 selectors.py，此处仅作重导出 stub，
现有代码（run_ql_only.py 等）无需任何修改。

推荐直接从 selectors.py 导入：
    from selectors import QLSelectorV1, QLSelectorV2

版本说明：
    LocalSearchSelector = QLSelectorV2（改进版，绝对改进奖励）
    如需原始版请使用：from selectors import QLSelectorV1
"""

from ha_selectors import (  # noqa: F401
    QLSelectorV1,
    QLSelectorV2,
    LocalSearchSelector,      # = QLSelectorV2
)

# 向后兼容：旧代码 from local_search_selector import LocalSearchSelector 仍可用
__all__ = ["LocalSearchSelector", "QLSelectorV1", "QLSelectorV2"]
