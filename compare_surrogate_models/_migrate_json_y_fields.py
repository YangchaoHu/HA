import json
from pathlib import Path

root = Path("C:/Users/dell/Desktop/HA/HA/proxy_models_jsons")

# quick_simu_1: y -> {"max_disp_x": ..., "volume": ...}
for fp in (root / "quick_simu_1").glob("seed_*.json"):
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for rec in data.get("records", []):
        y = rec.get("y")
        if isinstance(y, dict):
            continue
        if isinstance(y, list) and len(y) >= 2:
            rec["y"] = {"max_disp_x": y[0], "volume": y[1]}
        elif y is None:
            rec["y"] = {"max_disp_x": None, "volume": None}
        else:
            rec["y"] = {"max_disp_x": y, "volume": None}
        changed = True

    if changed:
        with fp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# quick_simu_2: y -> {"max_disp_y": ..., "volume": ...}
for fp in (root / "quick_simu_2").glob("seed_*.json"):
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for rec in data.get("records", []):
        x = rec.get("x") or [None, None]
        x1 = x[0] if len(x) > 0 else None
        x2 = x[1] if len(x) > 1 else None

        volume = None
        if x1 is not None and x2 is not None:
            x1f = float(x1)
            x2f = float(x2)
            volume = (0.4 ** 2) * x1f + (0.2 ** 2) * x2f + (0.1 ** 2) * (2.0 - x1f - x2f)

        y = rec.get("y")
        if isinstance(y, dict):
            max_disp_y = y.get("max_disp_y", y.get("max_disp", y.get("disp_y")))
        else:
            max_disp_y = y

        rec["y"] = {"max_disp_y": max_disp_y, "volume": volume}
        changed = True

    if changed:
        with fp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

print("done")
