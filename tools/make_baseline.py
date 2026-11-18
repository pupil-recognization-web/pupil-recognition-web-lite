#!/usr/bin/env python3
"""把当前 CSV 的临床结论固化成基线（轻量版）。"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-baseline")

import clinical
import store


def snapshot():
    out = {}
    for p in sorted(store.seg_dir().glob("*.csv")):
        name = p.name
        mark, _ = None, None
        try:
            import analysis_backend as ab
            mark, _ = ab._load_stimulus_time(name)
        # 关键：把当时用的刺激时刻固化成基线的一部分，避免以后重标导致误报
        except Exception:
            pass
        rep = clinical.build(name, video_name=name, stimulus_time=mark)
        if rep is None:
            continue
        out[name] = {
            "_stim": (round(mark, 4) if mark else None),
            "reflex": rep.reflex,
            "baseline_mm": round(rep.baseline_mm, 2) if rep.baseline_mm is not None else None,
            "constriction_pct": round(rep.constriction_pct, 1) if rep.constriction_pct is not None else None,
            "npi": rep.npi,
            "gaze_stability": rep.gaze_stability,
            "triage": rep.assessment.triage if rep.assessment else None,
        }
    return out


if __name__ == "__main__":
    data = snapshot()
    p = ROOT / "tools" / "baseline.json"
    if "--update" in sys.argv:
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        print(f"已写入 {p}（{len(data)} 份 CSV）")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True))
