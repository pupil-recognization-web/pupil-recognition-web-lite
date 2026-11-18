#!/usr/bin/env python3
"""轻量版冒烟测试：改完代码跑一条命令，确认核心链路没退化。

覆盖：
  1. 单眼：分割 → 临床报告 → PDF + 同名 txt
  2. 关键帧：单眼 2 张（刺激前 / 最低点）
  3. 历次趋势：同一患者第二次分析出现"与上次对比"，且历史每次只 +1
  4. 双眼：build_pair 能跑通，4 张关键帧，PDF + txt，一次检查只记 1 条历史
  5. PDF 页数合理（单眼 ≥1，双眼 ≥2）

用法：
    python tools/smoke_test.py                # 默认从 OpenEDS 取一段视频切 36 帧
    python tools/smoke_test.py --video /path/to.mp4 --frames 40
    python tools/smoke_test.py --keep         # 保留产物（默认只清测试患者记录）
    python tools/smoke_test.py --no-infer     # 跳过分割，直接用已有 CSV（更快）

退出码：0 = 全过，1 = 有失败项。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-smoke")

import config  # noqa: E402

TEST_PATIENT = "__smoke_test__"
RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = ""):
    RESULTS.append((bool(ok), name, detail))
    print(f"{'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))


def pdf_pages(path) -> int:
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    try:
        return int(out.split("Pages:")[1].split()[0])
    except Exception:  # noqa: BLE001
        return -1


def cut_clip(src: Path, dst: Path, frames: int):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-frames:v", str(frames),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)],
        check=True,
    )


def segment(clip: Path):
    from inference import predict_video_iter

    last = None
    for item in predict_video_iter(str(clip), model_name=config.DEFAULT_MODEL,
                                   max_frames=200, flow_fill=False, plr=True):
        last = item
    return Path(last["csv_path"]).name, last["out_path"]


def main() -> int:
    ap = argparse.ArgumentParser(description="轻量版冒烟测试")
    ap.add_argument("--video", default=None, help="源视频（默认取数据集里第一个）")
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--keep", action="store_true", help="保留测试产物")
    ap.add_argument("--no-infer", action="store_true", help="跳过分割，用已有 CSV")
    args = ap.parse_args()

    import clinical
    import lite_pdf

    print("== 环境 ==")
    print("（提示：服务正在运行时，本测试会额外加载一份模型，内存紧张时先停服务）")
    check(shutil.which("ffmpeg") is not None, "ffmpeg 可用")
    check(shutil.which("wkhtmltopdf") is not None, "wkhtmltopdf 可用")
    for key, info in config.MODEL_REGISTRY.items():
        if key != config.DEFAULT_MODEL:
            continue
        ok = (Path(info["dir"]) / "model.pt").exists() and (Path(info["dir"]) / "plr_model.pt").exists()
        check(ok, f"模型权重（{key}）",
              "" if ok else f"缺 .pt，用 PUPIL_MODELS_DIR 指定权重目录（当前 {info['dir']}）")

    tmp = Path(tempfile.mkdtemp(prefix="smoke_", dir="/tmp"))
    patient_file = Path(config.PLOT_DIR).parent / "patients" / f"{TEST_PATIENT}.json"
    if patient_file.exists():
        patient_file.unlink()      # 每次从干净的历史开始

    # ---------- 准备素材 ----------
    clips = []
    if args.no_infer:
        csvs = sorted(Path(config.VIDEO_OUTPUT_DIR).glob("*_plr_seg.csv"))
        if len(csvs) < 2:
            print("❌ --no-infer 需要至少 2 份已有 CSV")
            return 1
        pairs = [(c.name, None) for c in csvs[:2]]
    else:
        src = Path(args.video) if args.video else None
        if src is None:
            for d in config.DATASET_DIRS.values():
                vids = sorted(Path(d).glob("*.mp4")) if Path(d).is_dir() else []
                if vids:
                    src = vids[0]
                    break
        if src is None or not Path(src).exists():
            print("❌ 找不到源视频，用 --video 指定")
            return 1
        for i in (1, 2):
            clip = tmp / f"smoke_{i}.mp4"
            cut_clip(Path(src), clip, args.frames)
            clips.append(clip)
        pairs = []
        print("\n== 分割（两段）==")
        for i, clip in enumerate(clips, 1):
            csv_name, out_video = segment(clip)
            pairs.append((csv_name, out_video))
            check(Path(config.VIDEO_OUTPUT_DIR, csv_name).exists(), f"分割产出 CSV #{i}", csv_name)

    csv1, vid1 = pairs[0]
    csv2, vid2 = pairs[1]

    # ---------- 单眼 ----------
    print("\n== 单眼链路 ==")
    rep1 = clinical.build(csv1, video_name="冒烟测试1.mp4", stimulus_time=1.0,
                          patient=TEST_PATIENT, examiner="smoke", gcs=13,
                          signs={"ptosis": True}, meds={"opioid": True},
                          accommodation="存在", video_path=vid1)
    check(rep1 is not None, "生成单眼报告")
    if rep1:
        check(len(rep1.keyframes) == 2, "单眼关键帧 2 张", f"{len(rep1.keyframes)} 张")
        check(len(rep1.history) == 1, "首次分析历史 1 条", f"{len(rep1.history)} 条")
        pdf1 = lite_pdf.export(rep1)
        check(Path(pdf1).exists(), "单眼 PDF 生成", Path(pdf1).name)
        check(Path(pdf1).with_suffix(".txt").exists(), "单眼文本版生成")
        check(pdf_pages(pdf1) >= 1, "单眼 PDF 页数", str(pdf_pages(pdf1)))
        check(rep1.assessment is not None, "单眼分型与处置")
        check("处置等级" in lite_pdf.build_html(rep1), "PDF 含处置等级")

    # ---------- 第二次：趋势 ----------
    print("\n== 历次趋势 ==")
    rep2 = clinical.build(csv1, video_name="冒烟测试2.mp4", stimulus_time=1.0,
                          patient=TEST_PATIENT, video_path=vid1)
    check(len(rep2.history) == 2, "第二次分析历史 +1", f"{len(rep2.history)} 条")
    check(bool(rep2.history_plot) and Path(rep2.history_plot).exists(),
          "第二次起出现历次趋势图", Path(rep2.history_plot).name if rep2.history_plot else "无")
    check(TEST_PATIENT in clinical.list_patients(), "患者列表可检索到刚测的患者")
    check(bool(rep2.delta) and "baseline_mm_diff" in rep2.delta, "出现与上次对比",
          str({k: v for k, v in rep2.delta.items() if k.endswith("_diff")}))

    # ---------- 双眼 ----------
    print("\n== 双眼链路 ==")
    dual = clinical.build_pair(csv1, csv2, stim_l=1.0, stim_r=1.0,
                               video_l="冒烟左.mp4", video_r="冒烟右.mp4",
                               patient=TEST_PATIENT, examiner="smoke", gcs=13,
                               signs={"ptosis": True}, meds={"opioid": True},
                               accommodation="存在",
                               video_path_l=vid1, video_path_r=vid2)
    check(dual is not None, "生成双眼报告")
    if dual:
        check(len(dual.keyframes) == 4, "双眼关键帧 4 张", f"{len(dual.keyframes)} 张")
        check(len(dual.history) == 3, "一次双眼检查只 +1 条历史", f"{len(dual.history)} 条")
        pdf2 = lite_pdf.export_dual(dual)
        check(Path(pdf2).exists(), "双眼 PDF 生成", Path(pdf2).name)
        check(Path(pdf2).with_suffix(".txt").exists(), "双眼文本版生成")
        check(pdf_pages(pdf2) >= 2, "双眼 PDF 页数", str(pdf_pages(pdf2)))
        check(dual.assessment is not None and dual.assessment.triage in ("紧急", "尽快", "复查", "常规"),
              "双眼处置等级", dual.assessment.triage if dual.assessment else "—")

    # ---------- 兜底能力 ----------
    print("\n== 兜底：未做光刺激 / 双眼同步确认 ==")
    rep_ns = clinical.build(csv1, video_name="无刺激.mp4", no_stimulus=True)
    check(rep_ns is not None, "未做光刺激也能出报告")
    if rep_ns:
        check(str(rep_ns.reflex).startswith("未评估"), "反射标记为未评估", rep_ns.reflex)
        check(rep_ns.npi is None and rep_ns.constriction_pct is None, "PLR 数值留空而不是编造")
        pdf_ns = lite_pdf.export(rep_ns)
        txt_ns = Path(pdf_ns).with_suffix(".txt").read_text(encoding="utf-8")
        check("未做光刺激" in txt_ns, "报告写明未做光刺激")

    rep_sync = clinical.build_pair(csv1, csv2, stim_l=1.0, stim_r=1.0,
                                   video_l="左.mp4", video_r="右.mp4",
                                   video_path_l=vid1, video_path_r=vid2,
                                   sync_confirmed=True)
    check(getattr(rep_sync, "sync_confirmed", None) is True, "双眼同步确认可写入报告")

    # ---------- 数值回归 ----------
    print("\n== 数值回归（对比 tools/baseline.json）==")
    import json
    bpath = ROOT / "tools" / "baseline.json"
    if not bpath.exists():
        check(False, "缺少 tools/baseline.json", "先跑 tools/make_baseline.py --update")
    else:
        base = json.loads(bpath.read_text(encoding="utf-8"))
        drift, checked = [], 0
        import analysis_backend as _ab
        for name, exp in base.items():
            if not Path(config.VIDEO_OUTPUT_DIR, name).exists():
                continue
            mark, _ = _ab._load_stimulus_time(name)     # 与 make_baseline 同口径
            rep = clinical.build(name, video_name=name, stimulus_time=mark)
            if rep is None:
                drift.append(f"{name}: 现在算不出报告")
                continue
            for k, v in exp.items():
                if k == "_stim":
                    continue
                cur = {"reflex": rep.reflex, "baseline_mm": rep.baseline_mm,
                       "constriction_pct": rep.constriction_pct, "npi": rep.npi,
                       "gaze_stability": rep.gaze_stability,
                       "triage": rep.assessment.triage if rep.assessment else None}.get(k)
                if v is None or cur is None:
                    if v != cur:
                        drift.append(f"{name}:{k} {v}→{cur}")
                    continue
                if isinstance(v, str):
                    if cur != v:
                        drift.append(f"{name}:{k} {v}→{cur}")
                    continue
                checked += 1
                if abs(float(cur) - float(v)) > max(0.02, abs(float(v)) * 0.01):
                    drift.append(f"{name}:{k} {v}→{cur}")
        check(not drift, f"临床数值与基线一致（{checked} 个指标）",
              "；".join(drift[:3]) if drift else "")

    # ---------- 收尾 ----------
    if not args.keep:
        if patient_file.exists():
            patient_file.unlink()
        # 清掉本次产生的分割结果 / 报告 / 图片，避免污染 outputs
        for pat, folder in (("smoke_*", config.VIDEO_OUTPUT_DIR),
                            ("__smoke_test__*", config.PDF_DIR),
                            ("*smoke*", config.PLOT_DIR)):
            for f in Path(folder).glob(pat):
                try:
                    f.unlink()
                except Exception:  # noqa: BLE001
                    pass
        shutil.rmtree(tmp, ignore_errors=True)

    ok_n = sum(1 for ok, _, _ in RESULTS if ok)
    print(f"\n结果：{ok_n}/{len(RESULTS)} 项通过")
    return 0 if ok_n == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
