"""瞳孔评估（轻量版）——三步引导，单眼 / 双眼两种模式。

    ① 选录像 → 瞳孔识别（PLR 双类模型，得到 mm）
    ② 标定光刺激时刻（双眼模式可分别标，也可一键与左眼相同）
    ③ 结果与报告（临床指标 + 提示 + 建议 + PDF）

刻意不做：训练、去模糊、光流、ROI、参数表、CSV、像素/置信度/fps。
这些在「专业版」（pupil_recognition_web_v2）。

注意：Gradio 的 State 输出必须返回**裸值**，不能返回 gr.State(...) 对象。
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import gradio as gr

import clinical
import lite_pdf
from config import (
    CONF_THRESHOLD, DATASET_DIRS, IOU_THRESHOLD, RAW_VIDEO_DIR, SERVER_HOST,
    SERVER_PORT, STIMULUS_DIR, VIDEO_MAX_FRAMES, VIDEO_OUTPUT_DIR, allowed_paths,
)
from inference import predict_video_iter

MODEL_CHOICES = [
    ("标准（推荐）", "baseline"),
    ("边缘模糊时更稳", "mac"),
    ("睫毛/反光遮挡时更稳", "strip"),
]

CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
h1 { color: #1f4e79; letter-spacing: 1px; }
.stepbar { display:flex; gap:10px; margin: 6px 0 16px 0; }
.stepbar .s { flex:1; padding:10px 12px; border-radius:8px; background:#eef3f8;
              border:1px solid #d6e0ea; color:#5b6b7c; font-size:14px; }
.stepbar .s.on { background:#1f4e79; color:#fff; border-color:#1f4e79; font-weight:600; }
.stepbar .s b { font-size:16px; margin-right:6px; }
.card { background:#fff; border:1px solid #d8e1ea; border-radius:10px; padding:12px 14px; }
.card .k { color:#5b6b7c; font-size:13px; }
.card .v { color:#1f4e79; font-size:22px; font-weight:700; line-height:1.3; }
.verdict { background:#f4f7fa; border-left:5px solid #1f4e79; padding:12px 16px;
           border-radius:0 8px 8px 0; font-size:16px; margin:6px 0 14px 0; }
.hint { color:#6b7a89; font-size:13px; }
table .cmp { border-collapse: collapse; }
table.cmp { border-collapse: collapse; width:100%; margin:6px 0 10px 0; }
table.cmp th, table.cmp td { border:1px solid #d5dde6; padding:6px 9px; font-size:14px; }
table.cmp th { background:#eef3f8; color:#33475b; font-weight:normal; text-align:left; }
table.cmp td.eye { color:#1f4e79; font-weight:600; }
"""


def _banner(step: int) -> str:
    items = [("1", "选择录像并识别"), ("2", "标定光刺激"), ("3", "查看结果与报告")]
    cells = "".join(
        f'<div class="s{" on" if i + 1 == step else ""}"><b>{n}</b>{t}</div>'
        for i, (n, t) in enumerate(items)
    )
    return f'<div class="stepbar">{cells}</div>'


def _natural_key(name: str):
    """自然排序：S_2 排在 S_10 前面，而不是按字典序 S_1,S_10,S_100…

    把文件名按“文字段 / 数字段”切开，数字段用整数比较；整串再兜底比一次，
    保证同名不同扩展名、纯字母文件名也有稳定顺序。
    """
    stem = Path(name).stem
    key = []
    for part in re.split(r"(\d+)", stem):
        if part == "":
            continue
        key.append((0, int(part), "") if part.isdigit() else (1, 0, part.lower()))
    return (key, name.lower())


def _list_videos(dataset: str):
    """数据集里的视频列表：按文件名自然排序（S_0, S_1, S_2 … S_10）。"""
    d = DATASET_DIRS.get(dataset)
    if not d or not Path(d).is_dir():
        return []
    files = [f.name for f in Path(d).glob("*.mp4")]
    try:
        return sorted(files, key=_natural_key)
    except Exception:  # noqa: BLE001 - 排序失败也要能列出文件
        return sorted(files)


def _stage(dataset, name):
    d = DATASET_DIRS.get(dataset)
    if not d or not name:
        return None
    src = Path(d) / name
    if not src.exists():
        return None
    dst = Path(RAW_VIDEO_DIR) / f"{dataset}_{name}"
    if not dst.exists():
        shutil.copy(src, dst)
    return str(dst)


def _seg(video_l, video_r, model, mode, progress=gr.Progress()):
    """第 1 步：识别（双眼模式依次跑两只眼）。"""
    dual = (mode == "双眼对比")
    if not video_l:
        raise gr.Error("请先选择左眼（或单眼）录像")
    if dual and not video_r:
        raise gr.Error("双眼模式需要两段录像：请再选择右眼录像")
    if dual and Path(str(video_l)).stem == Path(str(video_r)).stem:
        raise gr.Error("左右眼选了同一段录像：识别结果会互相覆盖，请换一段（或改用单眼评估）")

    jobs = [("L", video_l)]
    if dual:
        jobs.append(("R", video_r))

    out, quality_lines, state = {}, [], {"mode": "dual" if dual else "single"}
    for tag, path in jobs:
        label = "左眼" if tag == "L" else "右眼"
        last = None
        for item in predict_video_iter(
            path, model_name=model, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            max_frames=VIDEO_MAX_FRAMES, progress=progress, flow_fill=False, plr=True,
        ):
            if item["partial"]:
                yield (gr.update(value=f"{label}识别中…已处理 {len(item['rows'])} 帧"),
                       gr.update(), gr.update(), gr.update(), state,
                       gr.update(interactive=False))
            else:
                last = item
        stats = last["stats"]
        rate = stats["detected"] / max(stats["frames"], 1) * 100
        level, note = clinical._detection_level(rate)
        quality_lines.append(f"**{label}**：识别质量 {level}　{note}")
        out[tag] = {"video": str(path), "csv": Path(last["csv_path"]).name,
                    "out_video": last["out_path"]}
        state[tag] = out[tag]

    summary = "已完成识别：" + "；".join(
        f"{'左眼' if k == 'L' else '右眼'} {Path(v['out_video']).stat().st_size // 1024} KB"
        for k, v in out.items()
    )
    yield (gr.update(value=summary),
           gr.update(value=out["L"]["out_video"]),
           gr.update(value=out.get("R", {}).get("out_video")),
           gr.update(value="\n\n".join(quality_lines)),
           state, gr.update(interactive=True))


def _stim_table(st) -> str:
    """常驻的标定状态表：两只眼都列出来，互不覆盖。"""
    st = st or {}
    dual = st.get("mode") == "dual"
    items = [("L", "左眼" if dual else "本次录像")]
    if dual:
        items.append(("R", "右眼"))
    body = ""
    for tag, name in items:
        t = (st.get(tag) or {}).get("stim")
        when = f"{t:.2f} 秒" if t else "未标定"
        flag = "✅ 已标定" if t else "— 待标定"
        body += (f"<tr><th>{name}</th><td class='eye'>{when}</td>"
                 f"<td>{flag}</td></tr>")
    return ('<table class="cmp" style="max-width:460px;">'
            "<thead><tr><th>眼别</th><th>光刺激时刻</th><th>状态</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def _mark_eye(mark_value, eye, video_path, st):
    """第 2 步：记录某一只眼的光刺激时刻。"""
    st = dict(st or {})
    if not video_path:
        return _stim_table(st), "请先完成第 1 步", gr.update(), st
    try:
        t = float(mark_value)
    except Exception:  # noqa: BLE001
        t = 0.0
    if t <= 0:
        return (_stim_table(st),
                "⚠️ 请播放录像到光刺激出现的瞬间，再点击「记录此刻」",
                gr.update(), st)

    stem = Path(video_path).stem
    marks_dir = Path(STIMULUS_DIR)
    marks_dir.mkdir(parents=True, exist_ok=True)
    out = marks_dir / f"{stem}.json"
    data = {"video": Path(video_path).name, "marks": [], "current": None}
    if out.exists():
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    data.setdefault("marks", [])
    if data["marks"] and abs(float(data["marks"][-1].get("time", -1)) - t) < 1e-6:
        data["current"] = t
    else:
        data["marks"].append({"time": t, "source": "lite_mark"})
        data["current"] = t
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    eye_name = "左眼" if eye == "L" else "右眼"
    st[eye] = dict(st.get(eye) or {}, stim=t)
    ready = bool(st.get("L", {}).get("stim")) and (
        st.get("mode") != "dual" or st.get("R", {}).get("stim"))
    note = f"{eye_name}已记录：{t:.2f} 秒"
    if not ready:
        other = "右眼" if eye == "L" else "左眼"
        note += f"（{other}还没标定）"
    return _stim_table(st), note, gr.update(interactive=ready), st


def _to_step2(video_l, video_r, mode, st):
    if not video_l:
        raise gr.Error("请先选择录像")
    dual = (mode == "双眼对比")
    return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
            _banner(2),
            gr.update(value=video_l), gr.update(value=video_r, visible=dual),
            gr.update(visible=dual))


def _to_step3():
    return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
            _banner(3))


def _cards(reps) -> str:
    """reps: [(眼别, ClinicalReport)]"""
    html = []
    for name, rep in reps:
        head = f"<b>{name}</b>　" if len(reps) > 1 else ""
        html.append(f'<div class="verdict">{head}对光反射<b>{rep.reflex}</b>；'
                    f'瞳孔 <b>{rep.baseline_mm:.1f} mm</b>（{rep.size_label.split("（")[0]}）；'
                    f'形态<b>{rep.shape_label}</b>。</div>')
    rows = [("瞳孔直径", lambda r: f"{r.baseline_mm:.1f} mm" if r.baseline_mm is not None else "—"),
            ("收缩幅度", lambda r: f"{r.constriction_pct:.0f}%" if r.constriction_pct is not None else "—"),
            ("收缩速度", lambda r: f"{r.velocity_mm_s:.2f} mm/s" if r.velocity_mm_s else "—"),
            ("潜伏期", lambda r: f"{r.latency_s:.2f} 秒" if r.latency_s is not None else "—"),
            ("NPi", lambda r: f"{r.npi:.1f}" if r.npi is not None else "—"),
            ("光刺激时刻", lambda r: f"{r.stim_time:.2f} 秒" if r.stim_time is not None else "—"),
            ("注视稳定性", lambda r: f"{r.gaze_stability:.2f} mm" if r.gaze_stability is not None else "—"),
            ("眼位偏移", lambda r: (f"{r.gaze_offset:.2f} mm（{r.gaze_dir}）"
                                if r.gaze_offset is not None else "—"))]
    if len(reps) == 1:
        rep = reps[0][1]
        html.append('<div style="display:flex;gap:10px;flex-wrap:wrap;">')
        for label, fn in rows:
            html.append(f'<div class="card" style="flex:1 1 18%;min-width:150px;">'
                        f'<div class="k">{label}</div><div class="v">{fn(rep)}</div></div>')
        html.append("</div>")
    else:
        l, r = reps[0][1], reps[1][1]
        body = "".join(
            f"<tr><th>{label}</th><td class='eye'>{fn(l)}</td><td class='eye'>{fn(r)}</td></tr>"
            for label, fn in rows
        )
        html.append('<table class="cmp"><thead><tr><th>指标</th><th>左眼</th><th>右眼</th>'
                    f'</tr></thead><tbody>{body}</tbody></table>')
    return "".join(html)


_TRIAGE_COLOR = {"紧急": "#c62828", "尽快": "#ef6c00", "复查": "#f9a825", "常规": "#2e7d32"}


def _assess_html(a) -> str:
    """分型与处置：处置等级 + 符合项（含排查方向）+ 人工核对清单。"""
    if a is None:
        return ""
    color = _TRIAGE_COLOR.get(a.triage, "#5b6b7c")
    html = [f'<div style="margin:14px 0 8px 0;"><b style="font-size:15px;">异常分型与处置</b>　'
            f'<span style="background:{color};color:#fff;padding:3px 12px;border-radius:6px;'
            f'font-size:13px;">处置等级：{a.triage}</span>'
            f'<span style="color:#6b7a89;font-size:13px;margin-left:8px;">{a.triage_reason}</span></div>']
    pos = a.positives
    if pos:
        rows = "".join(
            f"<tr><th style='width:34%;'>{c.name}</th>"
            f"<td><b>{c.verdict}</b><div class='k'>{c.basis}</div>"
            + (f"<div class='k' style='color:#44515e;'>排查：{c.workup}</div>" if c.workup != "—" else "")
            + "</td></tr>"
            for c in pos)
        html.append(f'<table class="cmp"><tbody>{rows}</tbody></table>')
    else:
        html.append('<div class="hint">未符合 PPT 中的任何异常分型。</div>')
    rest = [c for c in a.categories if not c.is_positive]
    if rest:
        html.append('<div class="hint" style="margin-top:4px;">未符合：'
                    + "、".join(c.name for c in rest) + "</div>")
    html.append('<div style="margin-top:8px;"><b style="font-size:14px;">需人工核对</b>'
                '<ul style="margin:4px 0 0 18px;color:#44515e;font-size:13px;">')
    html += [f"<li>{x}</li>" for x in a.checklist]
    html.append("</ul></div>")
    return "".join(html)


def _diag_html(rep) -> str:
    """未引出时把原因摊开：到底是没收缩、还是标定偏了。"""
    if isinstance(rep, clinical.DualReport):
        blocks = [(name, r.diagnostics) for name, r in
                  (("左眼", rep.left), ("右眼", rep.right)) if r.diagnostics]
    else:
        blocks = [("", rep.diagnostics or [])]
    if not any(b[1] for b in blocks):
        return ""
    html = ['<div style="margin-top:10px;padding:10px 14px;background:#fff8f0;'
            'border-left:4px solid #ef6c00;border-radius:0 8px 8px 0;font-size:13px;">'
            '<b>为什么判为“未引出”</b>']
    for name, items in blocks:
        if not items:
            continue
        if name:
            html.append(f'<div style="margin-top:4px;"><b>{name}</b></div>')
        html.append("<ul style='margin:2px 0 0 18px;color:#5b4a3a;'>")
        html += [f"<li>{x}</li>" for x in items]
        html.append("</ul>")
    html.append('<div style="margin-top:6px;color:#6b7a89;">提示：回到第 2 步把刺激时刻对准'
                '“瞳孔开始下降”的前 0.1–0.2 秒，或用上方输入框微调后重新生成。</div></div>')
    return "".join(html)


def _analyze(st, adj_l=0, adj_r=0, patient="", examiner="",
             gcs=0, diplopia=False, ptosis=False, anhidrosis=False,
             med_opioid=False, med_anti=False, med_miotic=False, med_mydriatic=False,
             accommodation="未查", no_stim=False, sync_confirmed=False):
    st = dict(st or {})
    dual = st.get("mode") == "dual"
    if not st.get("L", {}).get("csv"):
        raise gr.Error("请先完成第 1 步的识别")

    signs = {"diplopia": bool(diplopia), "ptosis": bool(ptosis), "anhidrosis": bool(anhidrosis)}
    meds = {"opioid": bool(med_opioid), "anticholinergic": bool(med_anti),
            "miotic": bool(med_miotic), "mydriatic": bool(med_mydriatic)}
    gcs_val = int(gcs) if gcs else None
    if no_stim:
        stim_l = stim_r = None          # 未做光刺激：不要求标定
    else:
        stim_l = float(adj_l) if adj_l else st.get("L", {}).get("stim")
    stim_r = float(adj_r) if adj_r else st.get("R", {}).get("stim")

    if dual:
        if not st.get("R", {}).get("csv"):
            raise gr.Error("双眼模式缺少右眼数据")
        rep = clinical.build_pair(
            st["L"]["csv"], st["R"]["csv"],
            stim_l=stim_l, stim_r=stim_r,
            video_l=Path(st["L"]["video"]).name, video_r=Path(st["R"]["video"]).name,
            patient=patient, examiner=examiner, gcs=gcs_val, signs=signs,
            meds=meds, accommodation=accommodation,
            video_path_l=st["L"]["video"], video_path_r=st["R"]["video"],
            no_stimulus=bool(no_stim), sync_confirmed=bool(sync_confirmed),
        )
        if rep is None:
            raise gr.Error("至少一只眼的数据不足，无法生成双眼报告，请重拍")
        cards = (_cards([("左眼", rep.left), ("右眼", rep.right)])
                 + _diag_html(rep) + _assess_html(rep.assessment))
        notes = ["**临床提示与建议**\n" + "\n".join(f"{i+1}. {x}" for i, x in enumerate(rep.advice))]
        if rep.quality:
            notes.append("**数据说明**\n" + "\n".join(f"- {x}" for x in rep.quality))
        plot = rep.plot
    else:
        rep = clinical.build(st["L"]["csv"], video_name=Path(st["L"]["video"]).name,
                             stimulus_time=stim_l, patient=patient, examiner=examiner,
                             gcs=gcs_val, signs=signs, meds=meds,
                             accommodation=accommodation,
                             video_path=st["L"]["video"], no_stimulus=bool(no_stim))
        if rep is None:
            raise gr.Error("这份录像的有效数据不足，无法生成结论，建议重拍")
        cards = _cards([("本次", rep)]) + _diag_html(rep) + _assess_html(rep.assessment)
        notes = []
        if rep.findings:
            notes.append("**临床提示**\n" + "\n".join(f"- {x}" for x in rep.findings))
        notes.append("**建议**\n" + "\n".join(f"{i+1}. {x}" for i, x in enumerate(rep.advice)))
        if rep.quality:
            notes.append("**数据说明**\n" + "\n".join(f"- {x}" for x in rep.quality))
        plot = rep.plot

    banner = ""
    if rep.assessment is not None and rep.assessment.triage == "紧急":
        banner = ('<div style="background:#fdecea;border:2px solid #c62828;color:#8e1b1b;'
                  'padding:12px 16px;border-radius:8px;font-size:16px;font-weight:600;'
                  'margin-bottom:10px;">⚠️ 紧急：请立即通知医生 · '
                  + rep.assessment.triage_reason[:80] + '</div>')
    if getattr(rep, "delta", None) and "baseline_mm_diff" in rep.delta:
        d = rep.delta
        trend = (f"\n\n**与上次对比**（{d.get('prev_time','')}）："
                 f"直径 {d['baseline_mm_diff']:+.2f} mm")
        if "npi_diff" in d:
            trend += f"，NPi {d['npi_diff']:+.1f}"
        notes.insert(0, trend)

    shots = getattr(rep, "keyframes", None) or []
    st["report"] = rep
    hist_img = getattr(rep, "history_plot", None)
    try:
        names = clinical.list_patients()
    except Exception:  # noqa: BLE001
        names = []
    return (gr.update(value=banner + cards), gr.update(value=plot),
            gr.update(value="\n\n".join(notes)), st, gr.update(interactive=True),
            gr.update(value=shots),
            gr.update(value=hist_img, visible=bool(hist_img)),
            gr.update(choices=names, value=(patient or None)))


def _pdf(st):
    st = dict(st or {})
    rep = st.get("report")
    if rep is None:
        raise gr.Error("请先生成结果")
    path = lite_pdf.export_dual(rep) if isinstance(rep, clinical.DualReport) else lite_pdf.export(rep)
    return gr.update(value=path, visible=True), f"已生成：`{path}`"


def _restart():
    import os
    import sys
    import threading
    import time

    def _do():
        time.sleep(1.0)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_do, daemon=True).start()
    return gr.update(value="正在重启，约 5 秒后刷新页面…")


def _reset():
    return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
            _banner(1), {}, gr.update(value=""), gr.update(value=None),
            gr.update(value=None), gr.update(value=""), gr.update(value=""),
            gr.update(interactive=False), _stim_table({}), "")


with gr.Blocks(title="瞳孔评估（轻量版）") as demo:
    gr.Markdown("# 瞳孔评估")
    gr.Markdown('<div class="hint">三步完成：选择录像 → 标定光刺激 → 查看结论并导出报告。</div>')
    mode_rd = gr.Radio(["单眼评估", "双眼对比"], value="单眼评估", label="评估方式")
    banner = gr.HTML(_banner(1))
    state = gr.State({})

    # ---------------- 第 1 步 ----------------
    with gr.Column() as grp1:
        gr.Markdown("### 第 1 步 · 选择录像并完成瞳孔识别")
        with gr.Row():
            vid_l = gr.Video(label="左眼 / 单眼录像")
            vid_r = gr.Video(label="右眼录像", visible=False)
        with gr.Accordion("或从数据集选择（演示用）", open=False):
            with gr.Row():
                ds_l = gr.Dropdown(choices=list(DATASET_DIRS.keys()),
                                   value=next(iter(DATASET_DIRS)), label="数据集")
                vd_l = gr.Dropdown(choices=_list_videos(next(iter(DATASET_DIRS))), label="左眼录像")
                load_l = gr.Button("载入左眼")
            with gr.Row(visible=False) as ds_row_r:
                ds_r = gr.Dropdown(choices=list(DATASET_DIRS.keys()),
                                   value=next(iter(DATASET_DIRS)), label="数据集（右眼）")
                vd_r = gr.Dropdown(choices=_list_videos(next(iter(DATASET_DIRS))), label="右眼录像")
                load_r = gr.Button("载入右眼")
        model_rd = gr.Radio(MODEL_CHOICES, value="baseline", label="识别模型")
        gr.Markdown('<div class="hint">不确定时用「标准」；画面偏模糊或睫毛遮挡时再换其他模型。</div>')
        btn_seg = gr.Button("开始识别", variant="primary", size="lg")
        seg_status = gr.Markdown("")
        with gr.Row():
            out_l = gr.Video(label="识别结果（红色为瞳孔轮廓）", interactive=False)
            out_r = gr.Video(label="右眼识别结果", interactive=False, visible=False)
        quality_md = gr.Markdown("")
        btn_next1 = gr.Button("下一步：标定光刺激 →", variant="primary", interactive=False)

    # ---------------- 第 2 步 ----------------
    with gr.Column(visible=False) as grp2:
        gr.Markdown("### 第 2 步 · 标定光刺激时刻")
        gr.Markdown('<div class="hint">播放录像，在光刺激亮起那一刻点击「记录此刻」；'
                    '记不准也可直接输入秒数。这一步决定所有时间指标，务必标对。</div>')
        gr.Markdown("**标定结果**（两只眼分别记录，互不覆盖）")
        stim_table = gr.HTML(_stim_table({}))
        with gr.Row():
            with gr.Column():
                player_l = gr.Video(label="左眼 / 单眼录像", interactive=False, elem_id="lite-video-l")
                mark_l = gr.HTML(
                    value="", html_template="""
                    <div style="margin:6px 0;">
                      <button id="lite-mark-l-btn" style="padding:10px 20px; cursor:pointer;
                        background:#1f4e79; color:#fff; border:none; border-radius:8px; font-size:15px;">
                        ⏺ 记录此刻</button>
                      <span id="lite-mark-l-info" style="margin-left:10px;color:#6b7a89;font-size:13px;"></span>
                    </div>""", js_on_load="""
                    const btn = element.querySelector('#lite-mark-l-btn');
                    const info = element.querySelector('#lite-mark-l-info');
                    if (btn) btn.addEventListener('click', () => {
                      const v = document.querySelector('#lite-video-l video');
                      if (!v) { info.textContent = '找不到播放器'; return; }
                      info.textContent = '已记录 ' + v.currentTime.toFixed(2) + ' 秒';
                      props.value = String(v.currentTime);
                    });
                    document.addEventListener('keydown', (ev) => {
                      if (ev.code === 'Space' && !['INPUT','TEXTAREA'].includes(ev.target.tagName)
                          && btn && btn.offsetParent !== null) {
                        ev.preventDefault();
                        btn.click();
                      }
                    });
                    document.addEventListener('keydown', (ev) => {
                      if (ev.code === 'Space' && !['INPUT','TEXTAREA'].includes(ev.target.tagName)
                          && btn && btn.offsetParent !== null) {
                        ev.preventDefault();
                        btn.click();
                      }
                    });""")
                with gr.Row():
                    manual_l = gr.Number(value=0, label="或手动输入秒数（左眼）", precision=2)
                    save_l = gr.Button("保存左眼标定", variant="primary")
            with gr.Column(visible=False) as col_r:
                player_r = gr.Video(label="右眼录像", interactive=False, elem_id="lite-video-r")
                mark_r = gr.HTML(
                    value="", html_template="""
                    <div style="margin:6px 0;">
                      <button id="lite-mark-r-btn" style="padding:10px 20px; cursor:pointer;
                        background:#c62828; color:#fff; border:none; border-radius:8px; font-size:15px;">
                        ⏺ 记录此刻</button>
                      <span id="lite-mark-r-info" style="margin-left:10px;color:#6b7a89;font-size:13px;"></span>
                    </div>""", js_on_load="""
                    const btn = element.querySelector('#lite-mark-r-btn');
                    const info = element.querySelector('#lite-mark-r-info');
                    if (btn) btn.addEventListener('click', () => {
                      const v = document.querySelector('#lite-video-r video');
                      if (!v) { info.textContent = '找不到播放器'; return; }
                      info.textContent = '已记录 ' + v.currentTime.toFixed(2) + ' 秒';
                      props.value = String(v.currentTime);
                    });""")
                with gr.Row():
                    manual_r = gr.Number(value=0, label="或手动输入秒数（右眼）", precision=2)
                    save_r = gr.Button("保存右眼标定", variant="primary")
                    copy_r = gr.Button("与左眼相同")
        mark_status = gr.Markdown("")
        with gr.Row():
            btn_back2 = gr.Button("← 上一步")
            btn_next2 = gr.Button("下一步：查看结果 →", variant="primary", interactive=False)

    # ---------------- 第 3 步 ----------------
    with gr.Column(visible=False) as grp3:
        gr.Markdown("### 第 3 步 · 结果与报告")
        gr.Markdown('<div class="hint">指标已按临床口径整理；异常项在「临床提示」里单列。</div>')
        with gr.Accordion("患者与查体（可选，填了判读更准）", open=False):
            with gr.Row():
                patient_pick = gr.Dropdown(choices=[], label="最近患者（选填，选了自动填床号）",
                                           value=None, allow_custom_value=False)
                patient_in = gr.Textbox(label="患者 / 床号（写入抬头与文件名）", placeholder="如：12床 王五")
                examiner_in = gr.Textbox(label="检查者", placeholder="姓名/工号")
            with gr.Row():
                gcs_in = gr.Number(value=0, label="GCS（3–15，0=未记录）", precision=0)
                diplopia_in = gr.Checkbox(label="复视", value=False)
                ptosis_in = gr.Checkbox(label="上睑下垂", value=False)
                anhidrosis_in = gr.Checkbox(label="同侧面部少汗", value=False)
            with gr.Row():
                med_opioid = gr.Checkbox(label="阿片类", value=False)
                med_anti = gr.Checkbox(label="抗胆碱药（阿托品等）", value=False)
                med_miotic = gr.Checkbox(label="缩瞳滴眼液", value=False)
                med_mydriatic = gr.Checkbox(label="散瞳滴眼液", value=False)
            accommodation_in = gr.Radio(["未查", "存在", "减弱", "消失"], value="未查",
                                        label="调节反射（阿-罗/Adie 鉴别用）")
            no_stim_in = gr.Checkbox(label="本次未做光刺激（只评大小/形态/眼位）", value=False)
            sync_in = gr.Checkbox(label="双眼为同一时段采集（已确认）", value=False, visible=False)
        with gr.Accordion("标定微调（高级）", open=False):
            with gr.Row():
                adj_l = gr.Number(value=0, label="左眼刺激时刻（秒，0=用第 2 步标定值）", precision=2)
                adj_r = gr.Number(value=0, label="右眼刺激时刻（秒）", precision=2, visible=False)
        btn_analyze = gr.Button("生成结果", variant="primary", size="lg")
        result_html = gr.HTML("")
        curve_img = gr.Image(label="对光反射曲线", interactive=False)
        notes_md = gr.Markdown("")
        with gr.Accordion("历次趋势（同一患者 ≥2 次记录时可见）", open=False):
            trend_img = gr.Image(label="", interactive=False, visible=False)
        with gr.Accordion("分割关键帧（判断识别是否可信）", open=False):
            kf_gallery = gr.Gallery(label="", columns=2, height="auto", object_fit="contain")
        with gr.Row():
            btn_back3 = gr.Button("← 上一步")
            btn_pdf = gr.Button("导出 PDF 报告", variant="primary", interactive=False)
        pdf_file = gr.File(label="报告文件", visible=False, interactive=False)
        pdf_status = gr.Markdown("")
        gr.Markdown("---")
        with gr.Row():
            btn_reset = gr.Button("开始新的一次评估", size="sm")
            btn_restart = gr.Button("重启服务（更新代码后使用）", size="sm", variant="secondary")
        restart_note = gr.Markdown("")

    # ---------------- 事件 ----------------
    def _on_mode(mode):
        dual = (mode == "双眼对比")
        return (gr.update(visible=dual), gr.update(visible=dual),
                gr.update(visible=dual), gr.update(visible=dual), gr.update(visible=dual),
                gr.update(visible=dual), gr.update(visible=dual), {}, gr.update(interactive=False),
                _stim_table({}), "", gr.update(visible=dual))

    mode_rd.change(_on_mode, inputs=[mode_rd],
                   outputs=[vid_r, out_r, ds_row_r, col_r, player_r, copy_r, adj_r, state,
                            btn_next1, stim_table, mark_status, sync_in])

    patient_pick.change(lambda name: gr.update(value=name or ""), inputs=[patient_pick], outputs=[patient_in])

    ds_l.change(lambda d: gr.update(choices=_list_videos(d), value=None), inputs=[ds_l], outputs=[vd_l])
    ds_r.change(lambda d: gr.update(choices=_list_videos(d), value=None), inputs=[ds_r], outputs=[vd_r])

    def _load(dataset, name):
        p = _stage(dataset, name)
        if not p:
            raise gr.Error("载入失败：找不到这段录像")
        return gr.update(value=p)

    load_l.click(_load, inputs=[ds_l, vd_l], outputs=[vid_l])
    load_r.click(_load, inputs=[ds_r, vd_r], outputs=[vid_r])

    def _mark_l(value, path, st):
        return _mark_eye(value, "L", path, st)

    def _mark_r(value, path, st):
        return _mark_eye(value, "R", path, st)

    btn_seg.click(_seg, inputs=[vid_l, vid_r, model_rd, mode_rd],
                  outputs=[seg_status, out_l, out_r, quality_md, state, btn_next1],
                  api_name="segment")

    btn_next1.click(_to_step2, inputs=[vid_l, vid_r, mode_rd, state],
                    outputs=[grp1, grp2, grp3, banner, player_l, player_r, col_r])

    mark_l.change(_mark_l, inputs=[mark_l, player_l, state],
                  outputs=[stim_table, mark_status, btn_next2, state])
    save_l.click(_mark_l, inputs=[manual_l, player_l, state],
                 outputs=[stim_table, mark_status, btn_next2, state])
    mark_r.change(_mark_r, inputs=[mark_r, player_r, state],
                  outputs=[stim_table, mark_status, btn_next2, state])
    save_r.click(_mark_r, inputs=[manual_r, player_r, state],
                 outputs=[stim_table, mark_status, btn_next2, state])

    def _copy(st):
        st = dict(st or {})
        stim = (st.get("L") or {}).get("stim")
        if not stim:
            return _stim_table(st), "请先标定左眼", gr.update(), st
        st["R"] = dict(st.get("R") or {}, stim=stim)
        return (_stim_table(st), f"右眼已与左眼相同：{stim:.2f} 秒",
                gr.update(interactive=True), st)

    copy_r.click(_copy, inputs=[state],
                 outputs=[stim_table, mark_status, btn_next2, state])

    btn_next2.click(_to_step3, outputs=[grp1, grp2, grp3, banner])
    btn_analyze.click(_analyze,
                      inputs=[state, adj_l, adj_r, patient_in, examiner_in,
                              gcs_in, diplopia_in, ptosis_in, anhidrosis_in,
                              med_opioid, med_anti, med_miotic, med_mydriatic,
                              accommodation_in, no_stim_in, sync_in],
                      outputs=[result_html, curve_img, notes_md, state, btn_pdf, kf_gallery,
                               trend_img, patient_pick],
                      api_name="analyze")
    btn_pdf.click(_pdf, inputs=[state], outputs=[pdf_file, pdf_status])

    btn_back2.click(lambda: (gr.update(visible=True), gr.update(visible=False),
                             gr.update(visible=False), _banner(1)),
                    outputs=[grp1, grp2, grp3, banner])
    btn_back3.click(lambda: (gr.update(visible=True), gr.update(visible=False), _banner(2)),
                    outputs=[grp2, grp3, banner])
    btn_restart.click(_restart, outputs=[restart_note], api_name="restart")
    btn_reset.click(_reset,
                    outputs=[grp1, grp2, grp3, banner, state, out_l, vid_l, vid_r,
                             curve_img, seg_status, btn_next1, stim_table, mark_status])

if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=2)   # 长任务排队，不堵界面
    demo.launch(server_name=SERVER_HOST, server_port=SERVER_PORT,
                allowed_paths=allowed_paths() + [str(VIDEO_OUTPUT_DIR)],
                show_error=True, css=CSS, quiet=False)
