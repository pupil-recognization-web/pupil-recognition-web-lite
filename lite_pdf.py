"""轻量版 PDF：一页看完的临床报告。

排版原则（给医生看的）：
  · 结论在最上面，大字号，先说结果再说数据；
  · 指标只用临床单位（mm / 秒 / NPi 0–5），不出现像素和置信度；
  · 只保留"光刺激 / 最低点 / 恢复"三个标注的曲线；
  · 提示与建议分开，建议按优先级编号；
  · 最后一段说明数据可信度 + 免责。
"""
from __future__ import annotations

import html
import subprocess
from datetime import datetime
from pathlib import Path

from config import PDF_DIR

OUT_DIR = Path(PDF_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSS = """
<style>
  @page { size: A4; margin: 0; }
  body { font-family: "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Noto Serif CJK SC",
         "AR PL UMing CN", sans-serif; color: #1a1a1a; font-size: 13px; line-height: 1.65; }
  .page { padding: 12mm 12mm; }
  body { font-size: 12.5px; }
  .head { border-bottom: 2px solid #1f4e79; padding-bottom: 6px; margin-bottom: 12px; }
  .head h1 { font-size: 22px; margin: 0 0 3px 0; color: #1f4e79; letter-spacing: 1px; }
  .head .meta { font-size: 11px; color: #666; }
  .verdict { background: #f4f7fa; border-left: 5px solid #1f4e79; padding: 8px 12px;
             margin: 8px 0 10px 0; font-size: 14.5px; }
  .verdict b { color: #1f4e79; }
  h2 { font-size: 14px; margin: 12px 0 4px 0; color: #1f4e79;
       border-bottom: 1px solid #cfd8e3; padding-bottom: 3px; }
  table.metrics { border-collapse: collapse; width: 100%; margin: 4px 0 8px 0; }
  table.metrics th, table.metrics td { border: 1px solid #d5dde6; padding: 4px 7px; }
  table.metrics th { background: #eef3f8; width: 26%; text-align: left; font-weight: normal; color: #33475b; }
  table.metrics td { text-align: left; }
  .big { font-size: 16px; font-weight: bold; color: #1f4e79; }
  ul { margin: 3px 0 6px 18px; padding: 0; }
  li { margin: 2px 0; }
  .advice li { margin: 5px 0; }
  img.curve { width: 92%; display:block; margin: 2px auto 2px auto; }
  .cap { font-size: 10.5px; color: #6b7a89; margin-bottom: 10px; }
  .note { background: #fbfbfb; border: 1px solid #e2e8ee; padding: 6px 10px; font-size: 11px; color: #55606b; }
  .foot { margin-top: 10px; font-size: 9.5px; color: #98a4b0; text-align: right; }
</style>
"""


def _esc(x):
    return html.escape("" if x is None else str(x))


_MED_LABELS = [("opioid", "阿片类"), ("anticholinergic", "抗胆碱药"),
               ("miotic", "缩瞳滴眼液"), ("mydriatic", "散瞳滴眼液")]


def _meds_text(rep) -> str:
    meds = getattr(rep, "meds", {}) or {}
    hit = [name for key, name in _MED_LABELS if meds.get(key)]
    return "、".join(hit) if hit else "未记录"


def _section_trend(rep) -> str:
    d = getattr(rep, "delta", {}) or {}
    if not d:
        return ""
    rows = []
    for key, label in (("baseline_mm", "瞳孔直径 (mm)"), ("npi", "NPi"),
                       ("constriction_pct", "收缩幅度 (%)")):
        if f"{key}_diff" in d:
            prev = d.get(f"{key}_prev")
            diff = d[f"{key}_diff"]
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
            rows.append(f"<tr><th>{label}</th><td>{prev} → {arrow} {diff:+.2f}</td></tr>")
    if "reflex_prev" in d:
        rows.append(f"<tr><th>对光反射</th><td>{_esc(d['reflex_prev'])} → {_esc(rep.reflex)}</td></tr>")
    if not rows:
        return ""
    when = d.get("prev_time", "上次")
    return (f"<h2>与上次对比</h2><div class='cap'>上次记录：{_esc(when)}</div>"
            f"<table class='metrics'>{''.join(rows)}</table>")


def _section_keyframes(rep) -> str:
    shots = getattr(rep, "keyframes", None) or []
    if isinstance(rep, object) and hasattr(rep, "left"):
        shots = (rep.left.keyframes or []) + (rep.right.keyframes or [])
    if not shots:
        return ""
    cells = "".join(
        f'<div style="display:inline-block;width:48%;vertical-align:top;margin:0 1% 6px 0;">'
        f'<img src="file://{Path(p).resolve()}" style="width:100%;">'
        f'<div class="cap">{_esc(cap)}</div></div>'
        for p, cap in shots[:4])
    return f"<h2>分割关键帧</h2>{cells}"


def _signs_text(rep) -> str:
    signs = getattr(rep, "signs", {}) or {}
    labels = [("diplopia", "复视"), ("ptosis", "上睑下垂"), ("anhidrosis", "面部少汗")]
    hit = [name for key, name in labels if signs.get(key)]
    return "、".join(hit) if hit else "未记录"


def _verdict(rep) -> str:
    short = rep.size_label.split("（")[0]          # 正常 / 偏小 / 散大 / 无法测量
    diam = f"<b>{rep.baseline_mm:.1f} mm</b>" if rep.baseline_mm is not None else "未测得"
    return (f"对光反射<b>{_esc(rep.reflex)}</b>；"
            f"瞳孔直径 {diam}（{_esc(short)}）；"
            f"形态<b>{_esc(rep.shape_label)}</b>。")


def _metrics(rep):
    if getattr(rep, "no_stim", False):
        body = (
            f"<tr><th>基线瞳孔直径</th><td>{rep.baseline_mm:.2f} mm</td></tr>"
            if rep.baseline_mm is not None else ""
        )
        body += (f"<tr><th>瞳孔大小判定</th><td>{_esc(rep.size_label)}</td></tr>"
                 f"<tr><th>形态</th><td>{_esc(rep.shape_label)}</td></tr>"
                 "<tr><th>对光反射</th><td>—（本次未做光刺激）</td></tr>"
                 "<tr><th>PLR / NPi</th><td>—（未评估）</td></tr>")
        return f'<table class="metrics">{body}</table>'
    no_resp = str(rep.reflex).startswith("未引出")
    rows = [
        ("基线瞳孔直径", f"{rep.baseline_mm:.2f} mm" if rep.baseline_mm is not None else "—"),
        ("最小瞳孔直径", ("—（未见收缩）" if no_resp else
                     (f"{rep.minimum_mm:.2f} mm" if rep.minimum_mm is not None else "—"))),
        ("收缩幅度", ("0（未见收缩）" if no_resp else
                  (f"{rep.constriction_mm:.2f} mm（{rep.constriction_pct:.1f}%）"
                   if rep.constriction_mm is not None else "—"))),
        ("最大收缩速度", f"{rep.velocity_mm_s:.2f} mm/s" if rep.velocity_mm_s is not None else "未测得"),
        ("潜伏期", f"{rep.latency_s:.2f} 秒" if rep.latency_s is not None else "未测得"),
        ("NPi（0–5，≥3 正常）", f"{rep.npi:.1f}" if rep.npi is not None else "不适用"),
        ("光刺激时刻", f"{rep.stim_time:.2f} 秒" if rep.stim_time is not None else "—"),
        ("注视稳定性（瞳孔中心漂移）",
         f"{rep.gaze_stability:.2f} mm（{rep.gaze_label}）" if rep.gaze_stability is not None else "—"),
        ("眼位偏移（瞳孔相对虹膜中心）",
         f"{rep.gaze_offset:.2f} mm（{rep.gaze_dir}）" if rep.gaze_offset is not None else "—"),
    ]
    body = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows)
    return f'<table class="metrics">{body}</table>'


_TRIAGE_COLOR = {"紧急": "#c62828", "尽快": "#ef6c00", "复查": "#f9a825", "常规": "#2e7d32"}


def _triage_badge(a) -> str:
    color = _TRIAGE_COLOR.get(a.triage, "#5b6b7c")
    return (f'<div style="margin:6px 0 8px 0;">'
            f'<span style="background:{color};color:#fff;padding:4px 12px;border-radius:6px;'
            f'font-size:14px;font-weight:600;">处置等级：{_esc(a.triage)}</span>'
            f'<span style="color:#55606b;font-size:12px;margin-left:10px;">{_esc(a.triage_reason)}</span>'
            f"</div>")


def _section_assessment(a) -> str:
    if a is None:
        return ""
    parts = ["<h2>异常分型与处置</h2>", _triage_badge(a)]
    pos = a.positives
    if pos:
        rows = "".join(
            f"<tr><th>{_esc(c.name)}</th><td>{_esc(c.verdict)}<br>"
            f"<span style='color:#6b7a89;font-size:11px;'>{_esc(c.basis)}</span></td></tr>"
            for c in pos)
        parts.append(f'<table class="metrics">{rows}</table>')
        parts.append("<div style='font-size:11.5px;color:#44515e;'>排查方向：<ul>")
        parts += [f"<li>{_esc(c.name)}：{_esc(c.workup)}</li>" for c in pos if c.workup != "—"]
        parts.append("</ul></div>")
    else:
        parts.append("<div class='note'>未符合 PPT 中的任何异常分型。</div>")
    rest = [c for c in a.categories if not c.is_positive]
    if rest:
        parts.append("<div style='font-size:11px;color:#6b7a89;margin-top:4px;'>"
                     "未符合：" + "、".join(_esc(c.name) for c in rest) + "</div>")
    parts.append('<h2>需人工核对</h2><ul style="font-size:11.5px;">')
    parts += [f"<li>{_esc(x)}</li>" for x in a.checklist]
    parts.append("</ul>")
    return "".join(parts)


def _sections(rep) -> str:
    out = [f'<div class="verdict">{_verdict(rep)}</div>',
           "<h2>关键指标</h2>", _metrics(rep)]
    out.append(_section_trend(rep))
    if getattr(rep, "history_plot", None) and Path(rep.history_plot).exists():
        out.append('<h2>历次趋势</h2>')
        out.append(f'<img class="curve" src="file://{Path(rep.history_plot).resolve()}">')
    if getattr(rep, "no_stim", False):
        out.append('<div class="note" style="margin:8px 0;">本次未做光刺激，报告只含瞳孔大小、形态与眼位；'
                   "PLR、收缩速度、潜伏期与本 NPi 均未评估。</div>")
    elif rep.plot and Path(rep.plot).exists():
        out.append("<h2>对光反射曲线</h2>")
        out.append(f'<img class="curve" src="file://{Path(rep.plot).resolve()}">')
        out.append(f'<div class="cap">光刺激发生于 {rep.stim_time:.2f} 秒；'
                   f"曲线为瞳孔直径随时间变化，最低点即最大收缩。</div>")
    if getattr(rep, "diagnostics", None):
        out.append('<h2>为什么判为"未引出"</h2><ul style="font-size:11.5px;">')
        out += [f"<li>{_esc(x)}</li>" for x in rep.diagnostics]
        out.append("</ul>")
    if rep.findings:
        out.append("<h2>临床提示</h2><ul>")
        out += [f"<li>{_esc(x)}</li>" for x in rep.findings]
        out.append("</ul>")
    out.append('<h2>建议</h2><ul class="advice">')
    out += [f"<li>{_esc(x)}</li>" for x in rep.advice]
    out.append("</ul>")
    if rep.quality:
        out.append('<h2>数据说明</h2><div class="note">')
        out += [f"<div>· {_esc(x)}</div>" for x in rep.quality]
        out.append("</div>")
    out.append(_section_keyframes(rep))
    if getattr(rep, "assessment", None) is not None:
        out.append('<div style="page-break-before: always;">')
        out.append(_section_assessment(rep.assessment))
        out.append("</div>")
    return "".join(out)


def build_html(rep) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">{CSS}</head><body><div class="page">
<div class="head">
  <h1>瞳孔评估报告</h1>
  <div class="meta">患者/床号：{_esc(rep.patient) or "—"} &nbsp;|&nbsp; 检查者：{_esc(rep.examiner) or "—"}
    &nbsp;|&nbsp; GCS：{rep.gcs if rep.gcs else "—"} &nbsp;|&nbsp; 体征：{_signs_text(rep)}
    &nbsp;|&nbsp; 用药：{_meds_text(rep)} &nbsp;|&nbsp; 调节反射：{rep.accommodation or "未查"}
    &nbsp;|&nbsp; 检查对象：{_esc(rep.video)} &nbsp;|&nbsp; 录像时长约 {rep.frames} 帧
    &nbsp;|&nbsp; 光刺激时刻 {_esc(rep.stim_time)} 秒 &nbsp;|&nbsp; 生成时间 {ts}</div>
</div>
{_sections(rep)}
<div class="note" style="margin-top:12px;">
  本报告由瞳孔录像自动分析生成，仅供临床参考，不能替代医生查体与综合判断。
  瞳孔变化需结合意识、GCS、用药、生命体征与影像学结果综合评估。
</div>
<div class="foot">瞳孔评估 · 轻量版 &nbsp;|&nbsp; 报告编号 {_esc(Path(rep.csv_name).stem)}</div>
</div></body></html>"""


def build_text(rep) -> str:
    """纯文本版报告：给 EMR 粘贴用，不受 PDF 字体 ToUnicode 影响。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["瞳孔评估报告", "=" * 32,
             f"患者/床号：{getattr(rep, 'patient', '') or '—'}　检查者：{getattr(rep, 'examiner', '') or '—'}",
             f"检查对象：{getattr(rep, 'video', '')}　生成时间：{ts}"]
    if hasattr(rep, "gcs"):
        lines.append(f"GCS：{rep.gcs or '—'}　体征：{_signs_text(rep)}　"
                     f"用药：{_meds_text(rep)}　调节反射：{getattr(rep, 'accommodation', '未查')}")
    lines.append("")
    if hasattr(rep, "left"):
        for name, r in (("左眼", rep.left), ("右眼", rep.right)):
            dia = f"{r.baseline_mm:.1f} mm" if r.baseline_mm is not None else "未测得"
            lines += [f"[{name}] 对光反射{r.reflex}；瞳孔 {dia}"
                      f"（{r.size_label.split('（')[0]}）；形态{r.shape_label}",
                      f"      基线 {r.baseline_mm:.2f} / 最小 {r.minimum_mm:.2f} mm；"
                      f"收缩 {r.constriction_pct:.1f}%"
                      + (f"；速度 {r.velocity_mm_s:.2f} mm/s" if r.velocity_mm_s else "")
                      + (f"；潜伏期 {r.latency_s:.2f} s" if r.latency_s is not None else "")
                      + (f"；NPi {r.npi:.1f}" if r.npi is not None else "")]
        lines += ["", f"双眼差：{rep.diff_base:.2f} mm，{rep.anisocoria}"
                  if rep.diff_base is not None else ""]
    elif getattr(rep, "no_stim", False):
        dia = f"{rep.baseline_mm:.2f} mm" if rep.baseline_mm is not None else "未测得"
        lines.append("结论：本次未做光刺激，仅评估瞳孔大小、形态与眼位")
        lines.append(f"瞳孔直径：{dia}（{rep.size_label.split('（')[0]}）；形态：{rep.shape_label}")
        lines.append("对光反射 / PLR / NPi：未评估")
    else:
        lines.append(f"结论：对光反射{rep.reflex}；瞳孔 {rep.baseline_mm:.1f} mm"
                     f"（{rep.size_label.split('（')[0]}）；形态{rep.shape_label}")
        lines.append(f"光刺激时刻：{rep.stim_time} 秒")
        lines.append(f"基线 {rep.baseline_mm:.2f} mm；最小 {rep.minimum_mm:.2f} mm；"
                     f"收缩 {rep.constriction_pct:.1f}%"
                     + (f"；速度 {rep.velocity_mm_s:.2f} mm/s" if rep.velocity_mm_s else "")
                     + (f"；潜伏期 {rep.latency_s:.2f} s" if rep.latency_s is not None else "")
                     + (f"；NPi {rep.npi:.1f}" if rep.npi is not None else ""))
    if getattr(rep, "delta", None):
        d = rep.delta
        bits = []
        for key, label in (("baseline_mm", "直径"), ("npi", "NPi"), ("constriction_pct", "收缩幅度")):
            if f"{key}_diff" in d:
                bits.append(f"{label} {d[key + '_diff']:+.2f}")
        if bits:
            lines += ["", f"与上次对比（{d.get('prev_time','')}）：" + "；".join(bits)]
    a = getattr(rep, "assessment", None)
    if a is not None:
        lines += ["", f"处置等级：{a.triage}（{a.triage_reason}）"]
        for c in a.positives:
            lines.append(f"  · {c.name}：{c.verdict}｜{c.basis}")
        if a.checklist:
            lines.append("  需人工核对：" + "；".join(a.checklist[:4]) + "……")
    tips = getattr(rep, "advice", None) or []
    if tips:
        lines += ["", "建议："] + [f"  {i+1}. {x}" for i, x in enumerate(tips)]
    lines += ["", "本报告由自动分析生成，仅供临床参考，不能替代医生查体与综合判断。"]
    return "\n".join(x for x in lines if x is not None)


def export(rep, out_path=None) -> str:
    """导出 PDF，返回文件路径。"""
    stem = Path(rep.video).stem if rep.video else Path(rep.csv_name).stem
    prefix = f"{rep.patient}_" if getattr(rep, "patient", "") else ""
    out = Path(out_path) if out_path else OUT_DIR / f"{prefix}{stem}_瞳孔评估报告.pdf"
    html_path = Path("/tmp") / f"_lite_{stem}.html"
    html_path.write_text(build_html(rep), encoding="utf-8")

    cmd = [
        "wkhtmltopdf", "--enable-local-file-access", "--encoding", "utf-8",
        "--page-size", "A4",
        "--margin-top", "0mm", "--margin-bottom", "0mm",
        "--margin-left", "0mm", "--margin-right", "0mm",
        "--footer-font-size", "8", "--footer-right", "第 [page] / [topage] 页",
        "--footer-spacing", "4",
        str(html_path), str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"PDF 生成失败：{(proc.stderr or '')[-500:]}")
    try:
        Path(out).with_suffix(".txt").write_text(build_text(rep), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - 文本版失败不影响 PDF
        print(f"文本版报告写入失败: {e}")
    return str(out)


# ============================================================
# 双眼报告
# ============================================================
def _fmt(v, unit="", nd=2, dash="—"):
    if v is None:
        return dash
    return f"{v:.{nd}f}{unit}"


def _dual_table(rep):
    l, r = rep.left, rep.right
    rows = [
        ("瞳孔直径 (mm)", _fmt(l.baseline_mm), _fmt(r.baseline_mm),
         _fmt(rep.diff_base, " mm")),
        ("最小直径 (mm)", _fmt(l.minimum_mm), _fmt(r.minimum_mm), "—"),
        ("收缩幅度 (%)", _fmt(l.constriction_pct, "", 1), _fmt(r.constriction_pct, "", 1),
         _fmt(rep.diff_amp, " 个百分点", 1)),
        ("最大收缩速度 (mm/s)", _fmt(l.velocity_mm_s), _fmt(r.velocity_mm_s), "—"),
        ("潜伏期 (s)", _fmt(l.latency_s), _fmt(r.latency_s), _fmt(rep.diff_latency, " s")),
        ("NPi（0–5）", _fmt(l.npi, "", 1), _fmt(r.npi, "", 1), _fmt(rep.diff_npi, "", 1)),
        ("光刺激时刻 (s)", _fmt(l.stim_time), _fmt(r.stim_time),
         _fmt(abs((l.stim_time or 0) - (r.stim_time or 0)), " s")),
        ("注视稳定性 (mm)", _fmt(l.gaze_stability), _fmt(r.gaze_stability),
         _fmt(abs((l.gaze_stability or 0) - (r.gaze_stability or 0)))),
        ("眼位偏移 (mm)", _fmt(l.gaze_offset), _fmt(r.gaze_offset),
         _fmt(abs((l.gaze_offset or 0) - (r.gaze_offset or 0)))),
        ("对光反射", l.reflex, r.reflex, "—"),
        ("瞳孔大小", l.size_label.split("（")[0], r.size_label.split("（")[0], "—"),
        ("形态", l.shape_label, r.shape_label, "—"),
    ]
    body = "".join(
        f"<tr><th>{_esc(a)}</th><td>{_esc(b)}</td><td>{_esc(c)}</td><td>{_esc(d)}</td></tr>"
        for a, b, c, d in rows
    )
    head = "<thead><tr><th>指标</th><th>左眼</th><th>右眼</th><th>差值</th></tr></thead>"
    return f'<table class="metrics">{head}{body}</table>'


def build_html_dual(rep) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f'<div class="verdict">{_esc(rep.verdict)}</div>',
             "<h2>双眼对照</h2>", _dual_table(rep)]
    if rep.plot and Path(rep.plot).exists():
        parts += ["<h2>对光反射曲线对比</h2>",
                  f'<img class="curve" src="file://{Path(rep.plot).resolve()}">',
                  '<div class="cap">实线为瞳孔直径随时间变化，虚线为各自的光刺激时刻，圆点为最低点。</div>']
    parts.append(_section_keyframes(rep))
    if rep.advice:
        parts.append('<h2>临床提示与建议</h2><ul class="advice">')
        parts += [f"<li>{_esc(x)}</li>" for x in rep.advice]
        parts.append("</ul>")
    if rep.quality:
        parts.append('<h2>数据说明</h2><div class="note">')
        parts += [f"<div>· {_esc(x)}</div>" for x in rep.quality]
        parts.append("</div>")
    if getattr(rep, "assessment", None) is not None:
        parts.append('<div style="page-break-before: always;">')
        parts.append(_section_assessment(rep.assessment))
        parts.append("</div>")
    body = "".join(parts)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">{CSS}</head><body><div class="page">
<div class="head">
  <h1>瞳孔评估报告（双眼对比）</h1>
  <div class="meta">患者/床号：{_esc(getattr(rep, "patient", "")) or "—"} &nbsp;|&nbsp;
    检查者：{_esc(getattr(rep, "examiner", "")) or "—"} &nbsp;|&nbsp;
    GCS：{rep.gcs if rep.gcs else "—"} &nbsp;|&nbsp; 体征：{_signs_text(rep)} &nbsp;|&nbsp;
    同步性：{"已确认同时段" if getattr(rep, "sync_confirmed", False) else "未确认（差值仅供参考）"} &nbsp;|&nbsp;
    检查对象：{_esc(rep.video)} &nbsp;|&nbsp; 生成时间 {ts}</div>
</div>
{body}
<div class="note" style="margin-top:12px;">
  本报告由双眼瞳孔录像自动分析生成，仅供临床参考。双眼对比要求两条录像在同一时段、同一光照条件下采集；
  若采集条件不同，直径差值仅供参考。
</div>
<div class="foot">瞳孔评估 · 轻量版 · 双眼</div>
</div></body></html>"""


def export_dual(rep, out_path=None) -> str:
    stem = f"{Path(rep.left.video).stem}_vs_{Path(rep.right.video).stem}"
    prefix = f"{rep.patient}_" if getattr(rep, "patient", "") else ""
    out = Path(out_path) if out_path else OUT_DIR / f"{prefix}{stem}_瞳孔评估报告.pdf"
    html_path = Path("/tmp") / f"_lite_dual_{abs(hash(stem))}.html"
    html_path.write_text(build_html_dual(rep), encoding="utf-8")
    cmd = [
        "wkhtmltopdf", "--enable-local-file-access", "--encoding", "utf-8",
        "--page-size", "A4",
        "--margin-top", "0mm", "--margin-bottom", "0mm",
        "--margin-left", "0mm", "--margin-right", "0mm",
        "--footer-font-size", "8", "--footer-right", "第 [page] / [topage] 页",
        "--footer-spacing", "4",
        str(html_path), str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"PDF 生成失败：{(proc.stderr or '')[-500:]}")
    try:
        Path(out).with_suffix(".txt").write_text(build_text(rep), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - 文本版失败不影响 PDF
        print(f"文本版报告写入失败: {e}")
    return str(out)
