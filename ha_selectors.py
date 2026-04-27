"""
ha_selectors.py
---------------

为项目内 `selectors.py` 提供稳定导入入口，避免与 Python 标准库
`selectors` 同名冲突。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_LOCAL_SELECTORS_PATH = Path(__file__).with_name("selectors.py")
_SPEC = importlib.util.spec_from_file_location("_ha_local_selectors", _LOCAL_SELECTORS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"无法加载本地 selectors 模块: {_LOCAL_SELECTORS_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

QLSelectorV1 = _MODULE.QLSelectorV1
QLSelectorV2 = _MODULE.QLSelectorV2
UCBSelectorV1 = _MODULE.UCBSelectorV1
UCBSelectorV2 = _MODULE.UCBSelectorV2
LocalSearchSelector = _MODULE.LocalSearchSelector
UCBLocalSearchSelector = _MODULE.UCBLocalSearchSelector

__all__ = [
    "QLSelectorV1",
    "QLSelectorV2",
    "UCBSelectorV1",
    "UCBSelectorV2",
    "LocalSearchSelector",
    "UCBLocalSearchSelector",
]
