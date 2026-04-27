"""
generate_lhs_jsons.py
---------------------

使用 LHS（拉丁超立方）采样对 quick_simu(1) 和 quick_simu(2) 批量生成仿真数据。
每累计 100 条保存一次快照 JSON，文件名中的数字 = 已累计样本总数。
支持中断后自动断点续跑（读取最新快照，从断点处继续）。

目录结构：
    proxy_models_jsons/
    ├── quick_simu_1/
    │   ├── seed_2026_100.json    # 样本 1-100
    │   ├── seed_2026_200.json    # 样本 1-200
    │   ├── ...
    │   └── seed_2026_1500.json   # 样本 1-1500（完整）
    ├── quick_simu_2/
    │   └── ...（同上结构）
    └── generate_lhs_jsons.py  （本文件）

运行示例：
    python generate_lhs_jsons.py
    python generate_lhs_jsons.py --n-samples 1500 --chunk-size 100 --seeds 2026 2027 2028
    python generate_lhs_jsons.py --models quick_simu_1   # 仅跑模型 1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # proxy_models_jsons/
ROOT_DIR = _HERE.parent                          # 项目根目录
SIM_DIR = ROOT_DIR / "simulation_models"
OUT_ROOT = _HERE

# 文件名格式: seed_{seed}_{cumulative_count}.json，例如 seed_2026_200.json
SNAP_PATTERN = re.compile(r"^seed_(?P<seed>\d+)_(?P<count>\d+)\.json$")


# ---------------------------------------------------------------------------
# LHS 采样
# ---------------------------------------------------------------------------

def lhs_samples(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """生成 n × d 的拉丁超立方样本矩阵（值域 [0, 1]）。"""
    result = np.zeros((n, d), dtype=float)
    for j in range(d):
        cutpoints = (np.arange(n, dtype=float) + rng.random(n)) / float(n)
        rng.shuffle(cutpoints)
        result[:, j] = cutpoints
    return result


def scale_lhs(
    unit_samples: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """将 [0,1] 的 LHS 样本线性映射到各参数的真实取值范围。"""
    scaled = np.zeros_like(unit_samples, dtype=float)
    for idx, (low, high) in enumerate(bounds):
        scaled[:, idx] = low + unit_samples[:, idx] * (high - low)
    return scaled


# ---------------------------------------------------------------------------
# JSON 原子写入（先写 .tmp 临时文件再重命名，防止中断产生损坏 JSON）
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# 序列化辅助（numpy 转 Python 原生类型）
# ---------------------------------------------------------------------------

def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# 断点恢复：找到最新有效快照，返回已有记录列表与已完成样本数
# ---------------------------------------------------------------------------

def load_latest_snapshot(
    out_dir: Path,
    seed: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    扫描输出目录，找到 seed 对应的最大累计数量的有效快照文件。

    Returns:
        (done_count, records)
        done_count = 0 且 records = [] 表示没有任何有效快照。
    """
    if not out_dir.exists():
        return 0, []

    candidates: List[Tuple[int, Path]] = []
    for fp in out_dir.iterdir():
        m = SNAP_PATTERN.match(fp.name)
        if m and int(m.group("seed")) == seed:
            candidates.append((int(m.group("count")), fp))

    # 从最大到最小尝试，找第一个可完整解析的文件
    for count, fp in sorted(candidates, reverse=True):
        try:
            with fp.open(encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("records", [])
            if len(records) == count:
                return count, records
            # 记录数与文件名不符 → 损坏，删除
            fp.unlink()
            print(f"[WARN] 删除记录数不符的快照: {fp.name} (期望 {count} 条，实有 {len(records)} 条)")
        except (json.JSONDecodeError, OSError):
            try:
                fp.unlink()
                print(f"[WARN] 删除损坏快照: {fp.name}")
            except OSError:
                pass

    return 0, []


# ---------------------------------------------------------------------------
# MAPDL 缓存清理（每次个体评估后调用，防止累积导致仿真变慢）
# ---------------------------------------------------------------------------

def clear_mapdl_cache(module: Any) -> None:
    """
    每评估完一个个体后调用，清理 MAPDL 内存与临时文件。
    步骤：
      1. /FINISH  ── 退出当前处理器，回到 Begin Level
      2. /CLEAR   ── 清除数据库（等同 mapdl.clear()）
      3. 删除工作目录下的 .rst / .esav / .emat 等仿真临时文件
    """
    mapdl = getattr(module, "mapdl", None)
    if mapdl is None:
        return
    try:
        mapdl.finish()
    except Exception:
        pass
    try:
        mapdl.clear()
    except Exception:
        pass
    # 删除 MAPDL 工作目录下的临时结果文件，防止磁盘占满或被旧文件干扰
    try:
        work_dir = Path(mapdl.directory)
        for suffix in ("*.rst", "*.esav", "*.emat", "*.mntr", "*.stat", "*.err"):
            for tmp_file in work_dir.glob(suffix):
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 动态加载仿真模块
# ---------------------------------------------------------------------------

def dynamic_load_module(py_file: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(py_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从以下路径加载模块: {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------

def build_model_configs() -> List[Dict[str, Any]]:
    """
    返回两个模型的配置字典，包含：
        name        : 输出子目录名 / 标识
        path        : .py 文件路径
        module_name : 动态加载时的模块名
        bounds      : 各参数取值范围 [(low, high), ...]
        runner      : Callable[[module, x: list], y]
    """

    def run_quick1(module: Any, x: List[float]) -> Any:
        """
        quick_simu(1) — CAEModel(paras)
        paras = (len1, width1, width2)
        返回: (max_disp_x, volume)  二元元组
        """
        model = module.CAEModel(tuple(x))
        return model.mapping_func()

    def run_quick2(module: Any, x: List[float]) -> Any:
        """
        quick_simu(2) — CAE(lens)
        lens = [len1, len2]，len3 = total_len - len1 - len2
        返回: max_disp_y  标量（仿真失败时返回 None）
        """
        model = module.CAE(x)
        return model.mapping_func()

    return [
        {
            "name": "quick_simu_1",
            "path": SIM_DIR / "quick_simu(1).py",
            "module_name": "quick_simu_1_module",
            # len1 ∈ [0.4, 1.6], width1 ∈ [0.01, 0.08], width2 ∈ [0.005, 0.04]
            "bounds": [(0.4, 1.6), (0.01, 0.08), (0.005, 0.04)],
            "runner": run_quick1,
        },
        {
            "name": "quick_simu_2",
            "path": SIM_DIR / "quick_simu(2).py",
            "module_name": "quick_simu_2_module",
            # len1 + len2 < total_len=2；各段均取 (0.2, 0.8) 以保证 len3 >= 0.4
            "bounds": [(0.2, 0.8), (0.2, 0.8)],
            "runner": run_quick2,
        },
    ]


# ---------------------------------------------------------------------------
# 核心运行器
# ---------------------------------------------------------------------------

def run_one_model(
    model_cfg: Dict[str, Any],
    seeds: Sequence[int],
    n_samples: int,
    chunk_size: int,
) -> bool:
    """
    对单个模型运行所有 seeds。
    每累计 chunk_size 条写一次快照，文件名 = seed_{seed}_{累计总数}.json，
    每个快照包含从第 1 条到当前累计总数的全部记录。
    返回 True 表示全部成功（单条仿真失败不中断批处理，记录 ok=False）。
    """
    model_name: str = model_cfg["name"]
    py_path: Path = model_cfg["path"]
    bounds: Sequence[Tuple[float, float]] = model_cfg["bounds"]
    runner: Callable[[Any, List[float]], Any] = model_cfg["runner"]
    module_name: str = model_cfg["module_name"]

    if not py_path.exists():
        print(f"[ERROR] 仿真文件不存在: {py_path}", file=sys.stderr)
        return False

    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    total_snaps = (n_samples + chunk_size - 1) // chunk_size

    print(f"\n{'=' * 60}")
    print(f"  模型: {model_name}")
    print(f"  样本数: {n_samples}  分片大小: {chunk_size}  共 {total_snaps} 个快照")
    print(f"  输出目录: {out_dir}")
    print(f"{'=' * 60}")

    print(f"[INFO] 加载仿真模块: {py_path.name} ...")
    module = dynamic_load_module(py_path, module_name)
    print(f"[INFO] 模块加载成功。")

    for seed in seeds:
        # 每个 seed 独立的 RNG，固定种子保证可复现
        rng = np.random.default_rng(seed)
        sample_unit = lhs_samples(n_samples, len(bounds), rng)
        sample_x = scale_lhs(sample_unit, bounds)

        # 断点恢复：读取最新有效快照
        done_count, all_records = load_latest_snapshot(out_dir, seed)

        if done_count >= n_samples:
            print(f"[SKIP] {model_name} seed={seed}: 已完成全部 {n_samples} 条，跳过。")
            continue

        if done_count > 0:
            print(
                f"[RESUME] {model_name} seed={seed}: "
                f"已有 {done_count} 条记录，从第 {done_count + 1} 条续跑。"
            )
        else:
            print(f"[START] {model_name} seed={seed}: 从第 1 条开始。")

        snap_idx = done_count // chunk_size  # 已写入的快照数
        fail_count = 0

        for sample_idx in range(done_count, n_samples):
            x_list: List[float] = [float(v) for v in sample_x[sample_idx]]
            item: Dict[str, Any] = {
                "index": sample_idx,      # 全局样本序号（0 起）
                "x": x_list,
            }
            try:
                raw_y = runner(module, x_list)
                item["y"] = to_jsonable(raw_y)
                item["ok"] = True
            except Exception as exc:
                item["y"] = None
                item["ok"] = False
                item["error"] = f"{type(exc).__name__}: {exc}"
                fail_count += 1
                print(
                    f"[WARN]  sample_idx={sample_idx} 失败: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                # 每个个体评估完毕后立即清理 MAPDL 缓存，防止仿真变慢
                clear_mapdl_cache(module)
            all_records.append(item)

            cumulative = sample_idx + 1  # 当前已累计条数（1-based）

            # 每隔 chunk_size 或到达总数时保存快照
            if cumulative % chunk_size == 0 or cumulative == n_samples:
                snap_idx += 1
                target = out_dir / f"seed_{seed}_{cumulative}.json"
                payload: Dict[str, Any] = {
                    "model": model_name,
                    "seed": seed,
                    "range": f"1-{cumulative}",
                    "records": all_records,
                }
                atomic_write_json(target, payload)

                ok_count = cumulative - fail_count
                status_str = f"{ok_count} 成功"
                if fail_count:
                    status_str += f" / {fail_count} 失败"
                print(
                    f"  [SNAP {snap_idx:03d}/{total_snaps}]"
                    f"  seed={seed}  样本 1-{cumulative}"
                    f"  {status_str}"
                    f"  -> {target.name}"
                )

    # 尽量关闭 MAPDL 进程，避免资源泄漏
    mapdl = getattr(module, "mapdl", None)
    if mapdl is not None:
        try:
            mapdl.exit()
            print(f"[INFO] MAPDL 会话已关闭: {model_name}")
        except Exception:
            pass

    return True


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用 LHS 对 quick_simu(1)/(2) 批量仿真，"
            "每 chunk-size 条保存一次累积快照 JSON，支持断点续跑。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-samples", type=int, default=1500,
        help="每个 seed 的样本总量",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100,
        help="每隔多少条保存一次快照",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2026, 2027, 2028],
        help="重复实验的随机种子（默认 3 个）",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["quick_simu_1", "quick_simu_2"],
        default=["quick_simu_1", "quick_simu_2"],
        help="指定运行哪些模型（默认全部）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.n_samples <= 0:
        print("--n-samples 必须为正整数。", file=sys.stderr)
        return 2
    if args.chunk_size <= 0:
        print("--chunk-size 必须为正整数。", file=sys.stderr)
        return 2
    if len(args.seeds) < 1:
        print("至少需要一个 seed。", file=sys.stderr)
        return 2

    cfg_map = {cfg["name"]: cfg for cfg in build_model_configs()}

    all_ok = True
    for model_name in args.models:
        try:
            ok = run_one_model(
                model_cfg=cfg_map[model_name],
                seeds=args.seeds,
                n_samples=args.n_samples,
                chunk_size=args.chunk_size,
            )
            if not ok:
                all_ok = False
        except Exception:
            print(f"\n[FATAL] 模型 {model_name} 发生未捕获异常：", file=sys.stderr)
            traceback.print_exc()
            all_ok = False

    if all_ok:
        print("\n[DONE] 所有模型运行完毕。")
        return 0
    else:
        print("\n[DONE] 部分模型运行异常，请检查上方日志。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
