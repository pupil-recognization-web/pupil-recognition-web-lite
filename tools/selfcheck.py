#!/usr/bin/env python3
"""轻量版自检：依赖、模型、以及一条 CSV 的「识别→临床报告→PDF」链路。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import config  # noqa: E402

fails, warns, oks = [], [], []
for mod in ("numpy", "pandas", "cv2", "matplotlib", "gradio", "torch", "ultralytics"):
    try:
        __import__(mod)
        oks.append(mod)
    except Exception as e:  # noqa: BLE001
        fails.append(f"依赖缺失 {mod}: {e}")

for key, info in config.MODEL_REGISTRY.items():
    for fn in ("model.yaml", "plr_model.pt"):
        if not (Path(info["dir"]) / fn).exists():
            fails.append(f"模型文件缺失 {key}/{fn}")

import clinical  # noqa: E402
import lite_pdf  # noqa: E402
from config import VIDEO_OUTPUT_DIR  # noqa: E402

csvs = sorted(Path(VIDEO_OUTPUT_DIR).glob("*.csv"))
if not csvs:
    warns.append("outputs/segmentation_results 下还没有 CSV，跑一次识别后再自检")
else:
    rep = clinical.build(csvs[0].name, video_name="selfcheck")
    if rep is None:
        warns.append(f"{csvs[0].name} 数据不足，无法生成临床报告")
    else:
        oks.append(f"{csvs[0].name}: 反射={rep.reflex} 直径={rep.baseline_mm}mm NPi={rep.npi}")
        path = lite_pdf.export(rep)
        if Path(path).stat().st_size > 10_000:
            oks.append(f"PDF 生成成功 {Path(path).name}")
        else:
            fails.append("PDF 生成异常（文件过小）")

print(f"依赖: {len([m for m in oks if m in ('numpy','pandas','cv2','matplotlib','gradio','torch','ultralytics')])}/7")
for x in oks:
    if x not in ("numpy", "pandas", "cv2", "matplotlib", "gradio", "torch", "ultralytics"):
        print(" OK  ", x)
for x in warns:
    print("WARN ", x)
for x in fails:
    print("FAIL ", x)
print(f"\n完成：FAIL {len(fails)}，WARN {len(warns)}")
raise SystemExit(1 if fails else 0)
