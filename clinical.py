"""临床层：把逐帧数据翻译成医生看得懂的话。

轻量版**只**暴露这些东西：
  瞳孔直径（基线 / 最小）、收缩幅度、收缩速度、潜伏期、NPi、
  对光反射状态、瞳孔大小、形态、临床提示、建议、数据是否够用。

不出现：像素、置信度、mAP、fps、CSV 列名、长短轴比数值、AUC 之类工程指标。
数值不可信时用「需复测 / 仅供参考」这类临床措辞表达，而不是把误差丢给医生。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import analysis_backend as ab
import plr_core
import store
from config import CORNEA_MM, PLOT_DIR

SIZE_NORMAL = (2.0, 5.0)

#: 从判读结果里保留给医生看的项（其余属于工程/质控范畴，轻量版不展示）
#: 依赖 MCV / NPi / "对光反射消失" 的提示：未引出收缩时不可评估，不展示
_RESPONSE_DEPENDENT = ("PLR 变差", "阿-罗瞳孔", "强直性瞳孔", "散大固定提示", "对光反射")

_KEEP_FINDINGS = (
    "针尖样瞳孔", "瞳孔散大", "散大程度", "对光反射", "PLR 变差",
    "形态欠规则", "小瞳提示", "散大固定提示", "阿-罗瞳孔", "强直性瞳孔",
    "Horner",
)


@dataclass
class ClinicalReport:
    video: str
    csv_name: str
    frames: int
    detection_level: str          # 稳定 / 一般 / 不稳定
    detection_pct: float
    baseline_mm: Optional[float]
    minimum_mm: Optional[float]
    constriction_mm: Optional[float]
    constriction_pct: Optional[float]
    velocity_mm_s: Optional[float]
    latency_s: Optional[float]
    npi: Optional[float]
    reflex: str                   # 存在 / 减弱 / 未引出（消失）
    size_label: str
    shape_label: str
    stim_time: Optional[float]
    findings: list = field(default_factory=list)
    advice: list = field(default_factory=list)
    quality: list = field(default_factory=list)
    plot: Optional[str] = None
    assessment: Optional["Assessment"] = None
    global_min_s: Optional[float] = None
    diagnostics: list = field(default_factory=list)
    gaze_stability: Optional[float] = None      # 瞳孔中心相对虹膜中心的漂移（mm，越小越稳）
    gaze_offset: Optional[float] = None         # 平均眼位偏移（mm）
    gaze_dir: str = "—"                         # 偏移方向
    gaze_label: str = "—"                       # 稳定 / 轻度漂移 / 明显漂移
    patient: str = ""                           # 患者/床号（报告抬头与文件名）
    examiner: str = ""                          # 检查者
    gcs: Optional[int] = None                   # 格拉斯哥昏迷评分（3–15）
    signs: dict = field(default_factory=dict)   # 查体：diplopia / ptosis / anhidrosis
    meds: dict = field(default_factory=dict)    # 用药：opioid / anticholinergic / miotic / mydriatic
    accommodation: str = "未查"                 # 调节反射：存在 / 减弱 / 消失 / 未查
    history: list = field(default_factory=list)  # 该患者历次记录（旧→新）
    delta: dict = field(default_factory=dict)   # 与上次相比的变化
    keyframes: list = field(default_factory=list)  # [(图片路径, 说明)]
    no_stim: bool = False                       # 本次未做光刺激：只评形态/直径/眼位
    history_plot: Optional[str] = None          # 历次趋势折线（≥2 次记录时才有）

    @property
    def usable(self) -> bool:
        return self.detection_level != "不稳定"


def _detection_level(pct: float):
    if pct >= 90:
        return "稳定", "瞳孔识别稳定，数据可采信。"
    if pct >= 70:
        return "一般", "部分画面未能稳定识别，数值仅供参考。"
    return "不稳定", "识别不稳定，建议重新拍摄（对焦、补光、避免睫毛遮挡）。"


def _size_label(d0: Optional[float]) -> str:
    if d0 is None:
        return "无法测量"
    if d0 < 2.0:
        return f"偏小（{d0:.1f} mm，<2 mm）"
    if d0 >= 9.0:
        return f"极度散大（{d0:.1f} mm，>9 mm）"
    if d0 >= 6.0:
        return f"显著散大（{d0:.1f} mm，6–9 mm）"
    if d0 > 5.0:
        return f"散大（{d0:.1f} mm，5–6 mm）"
    return f"正常（{d0:.1f} mm，2–5 mm）"


def _mm_median(rows) -> Optional[float]:
    """从逐帧行里取 pupil_mm 的中位数（仅 PLR 模式才有）。"""
    vals = [r.get("pupil_mm") for r in rows
            if r.get("detected") == 1 and r.get("pupil_mm") not in (None, "")]
    if not vals:
        return None
    try:
        return float(np.median([float(v) for v in vals]))
    except Exception:  # noqa: BLE001
        return None


def _shape_label(csv_name: str):
    rows = ab._read_rows(csv_name) or []
    ratios = [r["minor"] / r["major"] for r in rows
              if r["detected"] == 1 and r["major"] and r["minor"] and r["major"] > 0]
    if not ratios:
        return "无法评估"
    m = float(np.median(ratios))
    if m >= 0.85:
        return "规则"
    if m >= 0.80:
        return "轻度偏椭圆"
    return "欠规则"


def _reflex(resp: bool, pct: Optional[float]) -> str:
    if not resp:
        if pct is not None and pct >= 3.0:
            return f"可疑（幅度 {pct:.1f}%，未达 5% 阈值）"
        return "未引出（需复核刺激与录像）"
    if pct is not None and pct < 15:
        return "减弱"
    return "存在"


def list_patients(limit: int = 20) -> list:
    """最近看过的患者（按记录时间倒序），给首页下拉用。"""
    d = Path(PLOT_DIR).parent / "patients"
    if not d.exists():
        return []
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return [f.stem for f in files]


def history_plot(patient: str) -> Optional[str]:
    """历次趋势折线：直径与 NPi 随时间变化（≥2 次才有意义）。"""
    hist = load_history(patient)
    if len(hist) < 2:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Serif CJK SC",
                                       "WenQuanYi Zen Hei", "AR PL UMing CN", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    xs = list(range(1, len(hist) + 1))
    dia = [h.get("baseline_mm") for h in hist]
    npi = [h.get("npi") for h in hist]
    labels = [str(h.get("time", ""))[5:16] for h in hist]

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    axes[0].plot(xs, dia, marker="o", color="#1f4e79", lw=1.8)
    axes[0].set_ylabel("瞳孔直径 (mm)")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(xs, npi, marker="s", color="#c62828", lw=1.8)
    axes[1].set_ylabel("NPi (0–5)")
    axes[1].set_xlabel("历次检查")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_title(f"{patient} 历次趋势（{len(hist)} 次）")
    fig.tight_layout()
    out_dir = Path(PLOT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{patient}_history.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _patient_file(patient: str) -> Optional[Path]:
    name = (patient or "").strip()
    if not name:
        return None
    safe = "".join(ch for ch in name if ch not in '\\/:*?"<>|').strip() or "unknown"
    d = Path(PLOT_DIR).parent / "patients"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{safe}.json"


def load_history(patient: str) -> list:
    f = _patient_file(patient)
    if not f or not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def save_history(patient: str, entry: dict) -> list:
    """追加一条记录（同一患者保留最近 50 次）。"""
    f = _patient_file(patient)
    if not f:
        return []
    hist = load_history(patient)
    hist.append(entry)
    hist = hist[-50:]
    f.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    return hist


def _delta(hist: list) -> dict:
    """与上一条记录比较（hist 已含本次）。"""
    if len(hist) < 2:
        return {}
    cur, prev = hist[-1], hist[-2]
    out = {"prev_time": prev.get("time", "")}
    for key, label in (("baseline_mm", "直径"), ("npi", "NPi"),
                       ("constriction_pct", "收缩幅度"), ("gaze_stability", "注视漂移"),
                       ("anisocoria", "双眼差")):
        a, b = cur.get(key), prev.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[f"{key}_prev"] = b
            out[f"{key}_diff"] = round(a - b, 2)
    if cur.get("reflex") != prev.get("reflex"):
        out["reflex_prev"] = prev.get("reflex")
    return out


def _gaze_metrics(csv_name: str):
    """免模型的注视代理指标。

    以**虹膜中心**为参考点，看瞳孔中心怎么动：
      · 稳定性 = 偏移量的标准差（mm）——头部轻微移动会被虹膜中心抵消；
      · 眼位   = 平均偏移量与方向（mm）。
    这两项不需要 3D 模型，但注意：单目相机的绝对偏移会受拍摄角度影响，
    绝对值只能同一次录像内/双眼同时刻比较，不能跨次录像直接比。
    """
    import pandas as pd  # 懒加载，避免模块导入变慢

    path = Path(ab.CSV_DIR) / csv_name
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    if not {"iris_cx", "iris_cy"} <= set(df.columns):
        return None
    if "iris_detected" in df.columns:
        df = df[df.iris_detected == 1]
    v = df[(df.detected == 1) & df.iris_cx.notna() & df.cx.notna() & df.iris_major.notna()]
    if len(v) < 10:
        return None

    mm_per_px = CORNEA_MM / float(np.median(v.iris_major))
    dx = (v.cx.values - v.iris_cx.values) * mm_per_px
    dy = (v.cy.values - v.iris_cy.values) * mm_per_px
    stab = float(np.hypot(np.std(dx), np.std(dy)))
    ox, oy = float(np.mean(dx)), float(np.mean(dy))
    mag = float(np.hypot(ox, oy))

    parts = []
    if abs(ox) >= 0.20:
        parts.append("偏" + ("右" if ox > 0 else "左"))
    if abs(oy) >= 0.20:
        parts.append("偏" + ("下" if oy > 0 else "上"))
    direction = "、".join(parts) if parts else "基本居中"

    # 经验阈值（含分割抖动约 0.05–0.1 mm），换设备后需重新标定
    if stab < 0.20:
        label = "稳定"
    elif stab < 0.40:
        label = "轻度漂移"
    else:
        label = "明显漂移"

    return {"stability": stab, "offset": mag, "dx": ox, "dy": oy,
            "direction": direction, "label": label, "frames": len(v)}


def _keyframes(video_path, stim_time, t_min, stem: str) -> list:
    """从分割视频里抠两张图：刺激前、瞳孔最低点。给医生快速判断分割是否可信。"""
    if not video_path or not Path(str(video_path)).exists():
        return []
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_dir = Path(PLOT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    shots = []
    points = [(max((stim_time or 0) - 0.2, 0.0), "stim", "光刺激前")]
    if t_min is not None:
        points.append((t_min, "min", "瞳孔最低点"))
    else:
        points = [(0.0, "stim", "首帧")]
    for t, tag, label in points:
        idx = int(round(t * fps))
        if total:
            idx = max(0, min(idx, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        path = out_dir / f"{stem}_{tag}.png"
        cv2.imwrite(str(path), frame)
        shots.append((str(path), f"{label}（{t:.2f} s）"))
    cap.release()
    return shots


def _diagnose(res, curve) -> list:
    """未引出收缩时，告诉医生"为什么"以及标定是否可能偏了。"""
    if res.response:
        return []
    p, t, d = res.params, curve.t, curve.d
    out = []
    if p.amplitude <= 1e-9:
        out.append(f"标定后 2 秒内瞳孔没有收缩：基线 {p.baseline:.2f} mm，"
                   f"窗口内最低 {p.minimum:.2f} mm（反而更高）。")
    else:
        out.append(f"标定后 2 秒内只收缩了 {p.percent:.1f}%（判定阈值 5%），"
                   f"幅度不足以判为有反应。")
    gmin_t = float(t[int(np.nanargmin(d))])
    if gmin_t > curve.stimulus_time + 2.0:
        out.append(f"全段瞳孔最低点在 {gmin_t:.2f} 秒，比标定点晚 {gmin_t - curve.stimulus_time:.1f} 秒；"
                   "如果光刺激确实发生在标定处，说明这段录像里没有对应的收缩，建议复测。")
    elif gmin_t < curve.stimulus_time:
        out.append(f"全段瞳孔最低点在 {gmin_t:.2f} 秒，早于标定点——标定可能偏晚"
                   "（瞳孔已在恢复段），可把标定往前移到下降起点前 0.1–0.2 秒再算一次。")
    pre = d[(t >= curve.stimulus_time - 0.5) & (t < curve.stimulus_time)]
    if len(pre) >= 2 and pre[-1] < pre[0] * 0.98:
        out.append("标定前 0.5 秒瞳孔已在下降，光刺激可能比标定点更早。")
    return out


def _collect_findings(csv_name: str, response: bool = True):
    """从完整判读里挑出临床相关项，去掉工程/质控条目。

    ``response=False``（未引出收缩）时，MCV/NPi 不可评估，凡是依赖它们的提示
    （PLR 变差、阿-罗、强直性、散大固定）一律不展示——否则会出现
    "MCV 极低" 与 "MCV 未测得" 同时出现的自相矛盾。
    """
    try:
        rows, _ = ab.clinical_interpretation(csv_name)
    except Exception:  # noqa: BLE001 - 判读失败不应阻断报告
        return []
    out = []
    for item in rows or []:
        if len(item) < 3:
            continue
        name, verdict, basis = item[0], str(item[1]), item[2]
        if not any(k in name for k in _KEEP_FINDINGS):
            continue
        if "✅" in verdict:
            continue
        if not response and any(k in name for k in _RESPONSE_DEPENDENT):
            continue
        out.append(f"{name}：{basis}")
    # 去重保序
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _advice(rep: ClinicalReport) -> list:
    tips = []
    if rep.gcs is not None and rep.gcs <= 8:
        tips.append(f"GCS {rep.gcs} 分（≤8，重型颅脑损伤）：瞳孔任何异常都应按急症处理，"
                    "立即通知医生并复评意识与生命体征。")
    if rep.reflex.startswith("未引出"):
        tips.append("本次未观察到明确的对光反射收缩。请先确认光刺激确实照入、录像亮度足够，"
                    "并复测一次；若复测仍无收缩，需结合意识状态、用药与影像进一步评估。")
    elif rep.reflex == "减弱":
        tips.append("对光反射减弱，建议缩短复测间隔（如每 30–60 分钟），并记录变化趋势。")
    else:
        tips.append("对光反射存在，按常规间隔动态观察即可。")

    if rep.baseline_mm is not None and rep.baseline_mm > 5.0:
        tips.append("瞳孔散大：需结合意识、GCS、用药（阿托品/东莨菪碱等）与颅脑影像，"
                    "警惕颅内压升高；若为单侧新发散大，按急症处理。")
    if rep.baseline_mm is not None and rep.baseline_mm < 2.0:
        tips.append("瞳孔偏小：需排查阿片类药物、脑桥病变、有机磷中毒或缩瞳滴眼液。")
    if rep.shape_label == "欠规则":
        tips.append("瞳孔形态欠规则：建议人工检查调节反射，以鉴别阿-罗瞳孔等情况。")
    if rep.npi is not None and rep.npi < 3 and rep.reflex != "未引出（需复核刺激与录像）":
        tips.append("NPi 偏低（<3）：提示对光反射功能下降，建议加密监测并复查。")
    if rep.detection_level != "稳定":
        tips.append("本次录像质量一般，建议在光线稳定、患者安静的状态下重拍一次以复核。")
    return tips


def _plot(res: plr_core.PlrResult, csv_name: str) -> Optional[str]:
    """画一张医生看得懂的曲线：只标光刺激、最低点、恢复点。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config import PLOT_DIR

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Zen Hei",
        "AR PL UMing CN", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    c, p = res.curve, res.params
    out_dir = Path(PLOT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(c.t, c.d, color="#1f4e79", lw=1.8)
    ax.axvline(c.stimulus_time, color="#2e7d32", ls="--", lw=1.2)
    x_frac = c.stimulus_time / max(float(c.t[-1]), 1e-6)
    ax.annotate("光刺激", (c.stimulus_time, ax.get_ylim()[1]),
                xytext=(6 if x_frac < 0.8 else -52, -12), textcoords="offset points",
                color="#2e7d32", fontsize=10)
    ax.scatter([c.t_min], [p.minimum], color="#c62828", zorder=5, s=40)
    label = "最低" if res.response else "窗口最低"
    ax.annotate(f"{label} {p.minimum:.2f} mm", (c.t_min, p.minimum),
                xytext=(10, 14), textcoords="offset points", color="#c62828", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    if c.recovery_idx is not None and res.response:
        ax.axvline(c.t_recovery, color="#ef6c00", ls=":", lw=1.1)
        ax.annotate("恢复至 90%", (c.t_recovery, ax.get_ylim()[1]),
                    xytext=(4, -26), textcoords="offset points", color="#ef6c00", fontsize=9)
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("瞳孔直径（mm）")
    ax.set_title("对光反射曲线")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out_dir / f"{Path(csv_name).stem}_lite_plr.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def build(csv_name: str, video_name: str = "", stimulus_time=None,
          patient: str = "", examiner: str = "",
          gcs=None, signs=None, meds=None, accommodation="未查",
          video_path=None, save_hist=True, no_stimulus=False) -> Optional[ClinicalReport]:
    """生成一份面向医生的报告数据；数据不足时返回 None。"""
    rows = ab._read_rows(csv_name) or []
    if not rows:
        return None
    frames = len(rows)
    det = sum(1 for r in rows if r["detected"] == 1)
    pct = det / frames * 100 if frames else 0.0
    level, level_note = _detection_level(pct)

    # 「本次未做光刺激」：跳过 PLR，只出形态/直径/眼位，避免自动检测编出一个假刺激
    if no_stimulus:
        rep = ClinicalReport(
            video=video_name or Path(csv_name).stem, csv_name=csv_name, frames=frames,
            detection_level=level, detection_pct=pct,
            baseline_mm=_mm_median(rows), minimum_mm=None, constriction_mm=None,
            constriction_pct=None, velocity_mm_s=None, latency_s=None, npi=None,
            reflex="未评估（本次未做光刺激）",
            size_label=_size_label(_mm_median(rows)),
            shape_label=_shape_label(csv_name), stim_time=None, no_stim=True,
            patient=(patient or "").strip(), examiner=(examiner or "").strip(),
            gcs=(int(gcs) if gcs else None), signs=dict(signs or {}),
            meds=dict(meds or {}), accommodation=(accommodation or "未查"),
        )
        rep.quality = [level_note]
        if frames < 60:
            rep.quality.append("录像偏短，建议延长至 10 秒以上。")
        rep.advice = ["本次仅评估瞳孔大小、形态与眼位，未做光刺激，PLR 与 NPi 未评估。",
                      "如需 PLR/对光反射结论，请在标注光刺激时刻后重新生成。"]
        rep.assessment = assess(rep)
        rep.keyframes = _keyframes(video_path, None, None, Path(csv_name).stem)
        return rep

    res = plr_core.analyze(Path(ab.CSV_DIR) / csv_name,
                           stimulus_time=stimulus_time)
    if res is None:
        return None
    c, p, q = res.curve, res.params, res.quality

    rep = ClinicalReport(
        video=video_name or Path(csv_name).stem,
        csv_name=csv_name,
        frames=frames,
        detection_level=level,
        detection_pct=pct,
        baseline_mm=round(p.baseline, 2),
        minimum_mm=round(p.minimum, 2),
        constriction_mm=round(p.amplitude, 2),
        constriction_pct=round(p.percent, 1),
        velocity_mm_s=round(p.mcv, 2) if res.response else None,
        latency_s=(round(p.latency, 2) if np.isfinite(p.latency) and res.response else None),
        npi=round(p.npi_like, 1) if res.response else None,
        reflex=_reflex(res.response, p.percent),
        size_label=_size_label(p.baseline),
        shape_label=_shape_label(csv_name),
        stim_time=round(c.stimulus_time, 2),
        findings=_collect_findings(csv_name, response=res.response),
        patient=(patient or "").strip(),
        examiner=(examiner or "").strip(),
        gcs=(int(gcs) if gcs else None),
        signs=dict(signs or {}),
        meds=dict(meds or {}),
        accommodation=(accommodation or "未查"),
    )

    rep.quality = [level_note]
    if frames < 60:
        rep.quality.append("录像偏短，可能未覆盖完整的恢复过程，建议延长至 10 秒以上。")
    if q.frames_dropped_closed:
        rep.quality.append(f"已自动剔除 {q.frames_dropped_closed} 帧闭眼/眯眼画面。")
    if q.fps and q.fps < 15:
        rep.quality.append("录像帧率偏低，潜伏期只能粗略判断。")
    rep.advice = _advice(rep)
    rep.plot = _plot(res, csv_name)
    rep.assessment = assess(rep)
    rep.global_min_s = float(c.t[int(np.nanargmin(c.d))])
    rep.diagnostics = _diagnose(res, c)

    g = _gaze_metrics(csv_name)
    if g:
        rep.gaze_stability = round(g["stability"], 2)
        rep.gaze_offset = round(g["offset"], 2)
        rep.gaze_dir = g["direction"]
        rep.gaze_label = g["label"]
        if g["label"] != "稳定":
            rep.quality.append(
                f"注视{g['label']}（瞳孔中心漂移 {g['stability']:.2f} mm）："
                "PLR 的潜伏期/速度会受轻微影响，必要时让患者注视固定目标后复测。")
        if g["stability"] >= 0.40:
            rep.advice.append(
                f"注视明显漂移（瞳孔中心相对虹膜中心波动 {g['stability']:.2f} mm）："
                "本次 PLR 数值需谨慎，建议让患者注视固定目标后复测。")
        if g["offset"] >= 0.8:
            rep.advice.append(
                f"瞳孔相对虹膜中心偏移 {g['offset']:.2f} mm（{g['direction']}）："
                "提示眼位不正（斜视/眼外肌）可能，但单目拍摄角度也会造成偏移，需结合查体。")
    else:
        rep.gaze_label = "无数据（需重新识别以记录虹膜中心）"

    # 关键帧（刺激前 / 最低点）
    try:
        rep.keyframes = _keyframes(video_path, rep.stim_time, rep.global_min_s,
                                   Path(csv_name).stem)
    except Exception as e:  # noqa: BLE001
        print(f"关键帧生成失败: {e}")

    # 历次趋势（按患者/床号）；双眼模式由 build_pair 统一记一次，避免重复
    if rep.patient and save_hist:
        hist = save_history(rep.patient, {
            "time": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
            "video": rep.video,
            "baseline_mm": rep.baseline_mm,
            "npi": rep.npi,
            "constriction_pct": rep.constriction_pct,
            "reflex": rep.reflex,
            "gaze_stability": rep.gaze_stability,
        })
        rep.history = hist
        rep.delta = _delta(hist)
        try:
            rep.history_plot = history_plot(rep.patient)
        except Exception as e:  # noqa: BLE001 - 画图失败不影响报告
            print(f"趋势图生成失败: {e}")
    return rep


# ============================================================
# 双眼对比
# ============================================================
@dataclass
class DualReport:
    left: ClinicalReport
    right: ClinicalReport
    diff_base: Optional[float]
    diff_amp: Optional[float]
    diff_npi: Optional[float]
    diff_latency: Optional[float]
    anisocoria: str
    verdict: str
    advice: list = field(default_factory=list)
    quality: list = field(default_factory=list)
    plot: Optional[str] = None
    video: str = "双眼对比"
    assessment: Optional["Assessment"] = None
    patient: str = ""
    examiner: str = ""
    gcs: Optional[int] = None
    signs: dict = field(default_factory=dict)
    meds: dict = field(default_factory=dict)
    accommodation: str = "未查"
    history: list = field(default_factory=list)
    delta: dict = field(default_factory=dict)
    keyframes: list = field(default_factory=list)
    no_stim: bool = False
    sync_confirmed: Optional[bool] = None
    history_plot: Optional[str] = None


def _anisocoria(diff: Optional[float]) -> str:
    if diff is None:
        return "无法比较"
    if diff < 1.0:
        return "生理性（<1 mm）"
    if diff < 2.0:
        return "需关注（1–2 mm）"
    return "明显不等大（>2 mm）"


def _dual_plot(l: ClinicalReport, r: ClinicalReport) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config import PLOT_DIR

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Zen Hei",
        "AR PL UMing CN", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    curves = []
    for rep, color, label in ((l, "#1f4e79", "左眼"), (r, "#c62828", "右眼")):
        res = plr_core.analyze(Path(ab.CSV_DIR) / rep.csv_name, stimulus_time=rep.stim_time)
        if res:
            curves.append((rep, res, color, label))
    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for rep, res, color, label in curves:
        ax.plot(res.curve.t, res.curve.d, color=color, lw=1.7, label=label)
        ax.axvline(res.curve.stimulus_time, color=color, ls="--", lw=1.0, alpha=0.6)
        ax.scatter([res.curve.t_min], [res.params.minimum], color=color, s=28, zorder=5)
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("瞳孔直径（mm）")
    ax.set_title("双眼对光反射对比（虚线为各自的光刺激时刻）")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_dir = Path(PLOT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dual_lite_plr.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def build_pair(left_csv: str, right_csv: str,
               stim_l=None, stim_r=None,
               video_l: str = "左眼", video_r: str = "右眼",
               patient: str = "", examiner: str = "",
               gcs=None, signs=None, meds=None, accommodation="未查",
               video_path_l=None, video_path_r=None,
               no_stimulus=False, sync_confirmed=False):
    """双眼报告。任一眼数据不足时返回 None。

    注意：两只眼各自 build 时不写历史（save_hist=False），
    由本函数在最后统一记一条，避免同一检查被记成两次。
    """
    l = build(left_csv, video_name=video_l, stimulus_time=stim_l,
              patient=patient, examiner=examiner, gcs=gcs, signs=signs,
              meds=meds, accommodation=accommodation,
              video_path=video_path_l, save_hist=False, no_stimulus=no_stimulus)
    r = build(right_csv, video_name=video_r, stimulus_time=stim_r,
              patient=patient, examiner=examiner, gcs=gcs, signs=signs,
              meds=meds, accommodation=accommodation,
              video_path=video_path_r, save_hist=False, no_stimulus=no_stimulus)
    if l is None or r is None:
        return None

    diff_base = (abs(l.baseline_mm - r.baseline_mm)
                 if l.baseline_mm is not None and r.baseline_mm is not None else None)
    diff_amp = (abs(l.constriction_pct - r.constriction_pct)
                if l.constriction_pct is not None and r.constriction_pct is not None else None)
    diff_npi = (abs(l.npi - r.npi)
                if l.npi is not None and r.npi is not None else None)
    diff_lat = (abs(l.latency_s - r.latency_s)
                if l.latency_s is not None and r.latency_s is not None else None)

    aniso = _anisocoria(diff_base)
    verdict = (f"左眼 {l.baseline_mm:.1f} mm、对光反射{l.reflex}；"
               f"右眼 {r.baseline_mm:.1f} mm、对光反射{r.reflex}；"
               f"双眼差 {diff_base:.2f} mm，{aniso}。"
               if diff_base is not None else
               f"左眼对光反射{l.reflex}；右眼对光反射{r.reflex}；直径无法比较。")

    tips = []
    if diff_base is not None and diff_base >= 1.0:
        tips.append(f"双眼直径差 {diff_base:.2f} mm（≥1 mm）：需结合意识、GCS、用药与颅脑影像，"
                    "警惕颅内压升高/脑疝；若为新发变化，按急症处理。")
    if l.reflex.startswith("未引出") != r.reflex.startswith("未引出"):
        weak = "左眼" if l.reflex.startswith("未引出") else "右眼"
        tips.append(f"{weak}未引出明确的对光反射，对侧存在：先复核该眼光刺激与录像，再复测一次；"
                    "若仍无反应，需尽快请上级评估。")
    elif l.reflex.startswith("未引出") and r.reflex.startswith("未引出"):
        tips.append("双眼均未引出对光反射：先确认刺激与录像条件，复测仍无反应需紧急评估。")
    if diff_npi is not None and diff_npi >= 0.5:
        tips.append(f"双眼 NPi 相差 {diff_npi:.1f}：提示单侧反应异常，建议加密监测。")
    if diff_amp is not None and diff_amp >= 10:
        tips.append(f"双眼收缩幅度相差 {diff_amp:.0f}%：提示一侧反应偏弱，建议复测并记录趋势。")
    for eye, rep in (("左眼", l), ("右眼", r)):
        if rep.baseline_mm is not None and rep.baseline_mm > 5.0:
            tips.append(f"{eye}瞳孔散大（{rep.baseline_mm:.1f} mm）：警惕颅内压升高/动眼神经受压，"
                        "结合意识与影像判断。")
        if rep.baseline_mm is not None and 1.0 <= rep.baseline_mm <= 2.0:
            tips.append(f"{eye}瞳孔偏小（{rep.baseline_mm:.1f} mm）：需排查 Horner、阿片类药物、"
                        "脑桥病变或缩瞳滴眼液。")
    if not tips:
        tips.append("双眼瞳孔大小对称、对光反射存在，按常规间隔观察即可。")

    # 两眼相同的问题合并成一条（"左眼/右眼：录像偏短…"），避免重复刷屏
    bucket = {}
    for eye, rep_ in (("左眼", l), ("右眼", r)):
        for x in rep_.quality:
            bucket.setdefault(x, []).append(eye)
    quality = [f"{'/'.join(eyes)}：{text}" if len(eyes) > 1 else f"{eyes[0]}：{text}"
               for text, eyes in bucket.items()]

    dual = DualReport(
        left=l, right=r,
        diff_base=diff_base, diff_amp=diff_amp, diff_npi=diff_npi, diff_latency=diff_lat,
        anisocoria=aniso, verdict=verdict, advice=tips, quality=quality,
        plot=_dual_plot(l, r), video=f"{video_l} / {video_r}",
        patient=(patient or "").strip(), examiner=(examiner or "").strip(),
        gcs=(int(gcs) if gcs else None), signs=dict(signs or {}),
    )
    if l.gaze_offset is not None and r.gaze_offset is not None:
        gdx = abs((l.gaze_offset or 0) - (r.gaze_offset or 0))
        if gdx >= 0.4:
            tips.append(f"双眼眼位偏移相差 {gdx:.2f} mm（左 {l.gaze_dir} / 右 {r.gaze_dir}）："
                        "需除外斜视、眼外肌麻痹（单目拍摄角度也会影响，需同机位对比）。")
    dual.assessment = assess(l, r)
    dual.keyframes = (l.keyframes or []) + (r.keyframes or [])
    dual.meds = dict(meds or {})
    dual.accommodation = (accommodation or "未查")
    dual.no_stim = bool(no_stimulus)
    dual.sync_confirmed = bool(sync_confirmed)

    if dual.patient:
        import datetime as _dt
        hist = save_history(dual.patient, {
            "time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "video": f"{video_l} / {video_r}",
            "mode": "dual",
            "baseline_mm": l.baseline_mm,
            "baseline_mm_r": r.baseline_mm,
            "anisocoria": diff_base,
            "npi": l.npi,
            "npi_r": r.npi,
            "constriction_pct": l.constriction_pct,
            "reflex": f"L:{l.reflex} / R:{r.reflex}",
            "gaze_stability": l.gaze_stability,
        })
        dual.history = hist
        dual.delta = _delta(hist)
    return dual


# ============================================================
# 异常分型与处置（依据三份共识，见 CLINICAL.md）
# ============================================================
@dataclass
class Category:
    name: str
    verdict: str        # 符合 / 部分符合 / 不符合 / 需人工核对（可带括号补充，如"符合（三联征）"）
    basis: str
    workup: str

    @property
    def is_positive(self) -> bool:
        """注意用前缀判断：verdict 可能带括号说明，精确 == 会漏掉。"""
        return self.verdict.startswith(("符合", "部分符合"))


@dataclass
class Assessment:
    triage: str         # 紧急 / 尽快 / 复查 / 常规
    triage_reason: str
    categories: list = field(default_factory=list)
    checklist: list = field(default_factory=list)

    @property
    def positives(self):
        return [c for c in self.categories if c.is_positive]


def _dilated(x):  return x is not None and x.baseline_mm is not None and x.baseline_mm > 5.0
def _pinpoint(x): return x is not None and x.baseline_mm is not None and x.baseline_mm < 2.0
def _small(x):    return x is not None and x.baseline_mm is not None and 1.0 <= x.baseline_mm <= 2.0
def _absent(x):   return x is not None and x.reflex.startswith("未引出")
def _weak(x):     return x is not None and x.reflex == "减弱"
def _normal_size(x): return x is not None and x.baseline_mm is not None and 2.0 <= x.baseline_mm <= 5.0


def _reflex_short(x) -> str:
    """把“未引出（需复核刺激与录像）”压成“未引出”，用于表格与分级理由。"""
    return "未引出" if _absent(x) else (x.reflex if x is not None else "—")


def assess(l: ClinicalReport, r: Optional[ClinicalReport] = None) -> Assessment:
    """给出异常分型、危险分级与人工核对清单。

    只做"提示 + 排查方向"，不下诊断：每条都写依据，且要求结合查体/影像。
    """
    dual = r is not None
    eyes = [l] + ([r] if r else [])
    signs = (l.signs or {})
    has_ptosis = bool(signs.get("ptosis"))
    has_anhidrosis = bool(signs.get("anhidrosis"))
    has_diplopia = bool(signs.get("diplopia"))
    cats: list[Category] = []

    # 1. 双侧针尖样
    if dual:
        if _pinpoint(l) and _pinpoint(r):
            v, basis = "符合", f"双眼 {l.baseline_mm:.1f} / {r.baseline_mm:.1f} mm，均 <2 mm"
        elif _pinpoint(l) or _pinpoint(r):
            side = "左眼" if _pinpoint(l) else "右眼"
            v, basis = "部分符合", f"{side} <2 mm，对侧未同步变小"
        else:
            v, basis = "不符合", "双眼均 ≥2 mm"
    else:
        v = "部分符合" if _pinpoint(l) else "不符合"
        basis = f"本眼 {l.baseline_mm:.1f} mm（需对侧确认是否对称）" if l.baseline_mm else "直径未测得"
    cats.append(Category(
        "双侧针尖样瞳孔（<2 mm）", v, basis,
        "优先排查：阿片类（芬太尼/舒芬太尼/吗啡，纳洛酮可逆转）→ 脑桥出血/梗死 → "
        "有机磷/胆碱酯酶抑制剂 → 缩瞳滴眼液"))

    # 2. 单侧缩小（Horner 方向）
    if dual:
        diff = abs((l.baseline_mm or 0) - (r.baseline_mm or 0)) if l.baseline_mm and r.baseline_mm else None
        small_eye = l if _small(l) else (r if _small(r) else None)
        if small_eye and diff is not None and diff >= 1.0:
            v = "符合"
            basis = (f"一侧 {small_eye.baseline_mm:.1f} mm（1–2 mm），双眼差 {diff:.2f} mm ≥1 mm")
        elif small_eye and _small(l) and _small(r):
            v, basis = "不符合", f"双眼均 1–2 mm 且对称（差 {diff:.2f} mm），不支持单侧 Horner，考虑药物或脑桥病变" if diff is not None else "双眼对称偏小"
        elif small_eye:
            v, basis = "部分符合", f"有眼处于 1–2 mm，但双眼差 {diff:.2f} mm <1 mm" if diff is not None else "单侧 1–2 mm"
        else:
            v, basis = "不符合", "双眼均不在 1–2 mm"
    else:
        v = "部分符合" if _small(l) else "不符合"
        basis = f"本眼 {l.baseline_mm:.1f} mm 处于 1–2 mm" if _small(l) else "本眼不在 1–2 mm"
    if has_ptosis and has_anhidrosis and v in ("符合", "部分符合"):
        v, basis = "符合（三联征）", basis + "；伴同侧上睑下垂 + 面部少汗"
    elif v in ("符合", "部分符合") and not (has_ptosis or has_anhidrosis):
        basis += "；未见上睑下垂/少汗（如查体确认则更支持 Horner）"
    cats.append(Category(
        "单侧瞳孔缩小（Horner 方向）", v, basis,
        "核对同侧上睑下垂（Müller 肌仅贡献 1–2 mm）与面部少汗；排查延髓背外侧梗死"
        "（Wallenberg）、颈内动脉夹层、颈/纵隔占位、颈交感链或中心静脉置管损伤"))

    # 3. 单侧散大 + 反射迟钝/消失
    dil = [x for x in eyes if _dilated(x)]
    if dual and dil and len(dil) == 1 and (_absent(dil[0]) or _weak(dil[0])):
        side = "左眼" if dil[0] is l else "右眼"
        v = "符合"
        basis = f"{side} {dil[0].baseline_mm:.1f} mm（>5 mm），对光反射{dil[0].reflex}"
    elif dil and (_absent(dil[0]) or _weak(dil[0])):
        v, basis = "部分符合", f"本眼 {dil[0].baseline_mm:.1f} mm（>5 mm）、反射{dil[0].reflex}（需对侧比较）"
    else:
        v, basis = "不符合", "未见 >5 mm 且反射减弱的组合"
    cats.append(Category(
        "单侧瞳孔散大（>5 mm，反射迟钝/消失）", v, basis,
        "急症方向：小脑幕切迹疝（血肿/水肿/恶性梗死，需紧急降颅压）→ 后交通动脉瘤压迫动眼神经"
        "（常伴剧烈头痛）→ 颅底骨折/动眼神经损伤 → 误滴散瞳药或眼外伤；"
        "糖尿病性动眼神经麻痹多为“瞳孔回避”"))

    # 4. 双侧散大：固定型 vs 反射尚存
    if dual and all(_dilated(x) for x in eyes):
        if all(_absent(x) for x in eyes):
            v, basis = "符合（固定型）", "双眼 >5 mm 且对光反射均未引出"
            workup = ("晚期脑疝/脑干功能衰竭、抗胆碱药（阿托品/东莨菪碱）；"
                      "脑死亡需整套规范评估，不能单靠瞳孔")
        else:
            v, basis = "符合（反射尚存）", "双眼 >5 mm，但仍有收缩反应"
            workup = "躁动/疼痛/麻醉恢复期、抗胆碱药、暗环境；先排除可逆因素再谈病理"
    elif dil:
        v, basis = "部分符合", f"本眼 {dil[0].baseline_mm:.1f} mm（>5 mm），需对侧确认是否双侧"
        workup = "双侧固定型多为脑干功能衰竭/抗胆碱药；反射尚存多为躁动、疼痛、麻醉恢复或暗环境"
    else:
        v, basis, workup = "不符合", "未见 >5 mm", "—"
    cats.append(Category("双侧瞳孔散大（>5 mm）", v, basis, workup))

    # 4.5 动眼神经麻痹（PPT/共识：急性痛性+瞳孔受累 → 后交通动脉瘤；糖尿病性多为瞳孔回避）
    if has_ptosis or has_diplopia:
        if dil and any(_absent(x) or _weak(x) for x in dil):
            v = "符合（瞳孔受累型）"
            basis = "上睑下垂/复视 + 瞳孔散大且对光反射减弱/未引出"
        elif all(_normal_size(x) for x in eyes):
            v = "部分符合（瞳孔回避型）"
            basis = "上睑下垂/复视存在，但瞳孔大小与对光反射基本正常"
        else:
            v, basis = "需人工核对", "有上睑下垂/复视，瞳孔改变不典型"
    else:
        v, basis = "不符合", "未记录上睑下垂或复视"
    cats.append(Category(
        "动眼神经麻痹（瞳孔受累型 / 回避型）", v, basis,
        "急性痛性 + 瞳孔受累要优先排除后交通动脉瘤（急诊 MRI/MRA）；颅底骨折、脑疝也会压迫动眼神经；"
        "糖尿病性多为“瞳孔回避”。眼肌型重症肌无力可模拟，需看疲劳性"))

    # 5. 阿-罗瞳孔
    smallish = [x for x in eyes if x.baseline_mm is not None and x.baseline_mm < 3.0]
    irregular = [x for x in eyes if x.shape_label == "欠规则"]
    if smallish and irregular and any(_absent(x) for x in eyes):
        v = "部分符合"
        basis = "瞳孔偏小 + 形态欠规则 + 对光反射未引出"
    elif irregular or (smallish and any(_absent(x) for x in eyes)):
        v, basis = "需人工核对", "部分条件满足（偏小/欠规则/反射未引出）"
    else:
        v, basis = "不符合", "未见“偏小 + 欠规则 + 反射消失”的组合"
    cats.append(Category(
        "阿-罗瞳孔（Argyll-Robertson）", v, basis,
        "必须人工查调节反射（对光消失但调节存在）；ICU 少见，考虑神经梅毒、中脑顶盖病变、多发性硬化"))

    # 6. 强直性瞳孔（Adie）
    big = [x for x in eyes if x.baseline_mm is not None and x.baseline_mm > 4.0]
    slow = [x for x in eyes if x.velocity_mm_s is not None and x.velocity_mm_s < 0.5]
    if big and slow:
        v, basis = "部分符合", f"瞳孔偏大（{big[0].baseline_mm:.1f} mm）且收缩很慢（{slow[0].velocity_mm_s:.2f} mm/s）"
    elif slow:
        v, basis = "需人工核对", "收缩速度很慢，需结合瞳孔大小与调节反射"
    else:
        v, basis = "不符合", "收缩速度未见明显减慢"
    cats.append(Category(
        "强直性瞳孔（Adie）", v, basis,
        "强光下收缩极慢、调节也慢；中青年多见、多为良性，ICU 少见；需人工查调节反射"))

    # 6.5 对光反射未引出 / 减弱（瞳孔未必散大）
    if any(_absent(x) for x in eyes):
        side = "、".join("左眼" if x is l else "右眼" for x in eyes if _absent(x)) if dual else "本眼"
        size_txt = "瞳孔已散大" if any(_dilated(x) for x in eyes) else "瞳孔未散大"
        v, basis = "符合", f"{side}未引出对光反射（{size_txt}）"
    elif any(_weak(x) for x in eyes):
        v, basis = "部分符合", "对光反射减弱"
    else:
        v, basis = "不符合", "对光反射存在"
    cats.append(Category(
        "对光反射未引出 / 减弱", v, basis,
        "首要复核：光刺激是否照入、录像亮度、标定时刻；排除后考虑早期颅内压增高、"
        "视神经损伤、中脑损害、缺氧/脓毒症/代谢性脑病、作用于瞳孔的药物"))

    # 7. 尺寸正常但 PLR 变差
    if any(_absent(x) for x in eyes):
        v, basis = "需人工核对", "本次未引出收缩，PLR 变差无法量化（先复核刺激）"
    elif any(_normal_size(x) and ((x.velocity_mm_s is not None and x.velocity_mm_s < 1.0)
                                  or (x.npi is not None and x.npi < 3)) for x in eyes):
        v, basis = "符合", "瞳孔 2–5 mm，但收缩速度 <1 mm/s 或 NPi <3"
    else:
        v, basis = "不符合", "尺寸与反应未见该组合"
    cats.append(Category(
        "尺寸正常但 PLR 变差", v, basis,
        "早期颅内压增高（尚未散大）、视神经损伤、中脑早期损害；缺氧性/脓毒症/代谢性脑病。"
        "注意：NPi 的前提是视神经与动眼神经完好，否则数值不能代表脑干功能"))

    # 8. 双侧不等大
    if dual and l.baseline_mm and r.baseline_mm:
        diff = abs(l.baseline_mm - r.baseline_mm)
        if diff < 1.0:
            v, basis = "不符合（生理性）", f"双眼差 {diff:.2f} mm <1 mm"
        else:
            v, basis = "符合", f"双眼差 {diff:.2f} mm ≥1 mm"
    else:
        v, basis = "需人工核对", "单眼模式无法比较，请用双眼对比"
    cats.append(Category(
        "双侧瞳孔不等大（anisocoria）", v, basis,
        "≥1 mm 且有新发变化时按病理性处理：一侧动眼神经受压/损伤（脑疝、动脉瘤）、"
        "一侧交感通路损伤（Horner）、既往眼部病变。长期稳定、反射灵敏多无病理意义"))

    # ---- 用药与调节反射：把"排查方向"变成"已知条件" ----
    meds = (l.meds or {})
    acc = (l.accommodation or "未查")
    for c in cats:
        if c.is_positive and c.name.startswith("双侧针尖样") and meds.get("opioid"):
            c.basis += "；医嘱含阿片类（纳洛酮可逆转，先核对）"
        if c.is_positive and c.name.startswith("双侧瞳孔散大") and meds.get("anticholinergic"):
            c.basis += "；医嘱含抗胆碱药（阿托品/东莨菪碱）"
        if c.is_positive and c.name.startswith("单侧瞳孔缩小") and meds.get("miotic"):
            c.basis += "；在用缩瞳滴眼液？"
        if c.is_positive and c.name.startswith("单侧瞳孔散大") and meds.get("mydriatic"):
            c.basis += "；在用散瞳滴眼液/散瞳检查？"
        if c.name.startswith("阿-罗") and c.verdict != "不符合":
            if acc == "存在" and any(_absent(x) for x in eyes):
                c.verdict = "符合（需人工确认调节）"
                c.basis += "；调节反射存在"
            elif acc in ("减弱", "消失"):
                c.verdict = "部分符合（调节亦异常，倾向 Adie/其他）"
                c.basis += f"；调节反射{acc}"
        if c.name.startswith("强直性瞳孔") and c.is_positive and acc in ("减弱", "消失"):
            c.verdict = "符合（调节亦慢）"
            c.basis += f"；调节反射{acc}"

    # ---- 危险分级 ----
    # 尺寸类证据（直径、对称性）不受眼动影响；反射类证据（对光反射、NPi）会被注视漂移污染，
    # 所以眼动明显时只降级"反射类"触发条件，尺寸类照旧。
    unstable = any(x.gaze_stability is not None and x.gaze_stability >= 0.40 for x in eyes)
    urgent_size, urgent_reflex, soon_size, soon_reflex, recheck = [], [], [], [], []

    for x in eyes:
        if _dilated(x) and (_absent(x) or _weak(x)):
            urgent_reflex.append(f"瞳孔散大（{x.baseline_mm:.1f} mm）伴对光反射{_reflex_short(x)}")
        elif _dilated(x):
            urgent_size.append(f"瞳孔散大（{x.baseline_mm:.1f} mm）")
    if dual and l.baseline_mm and r.baseline_mm and abs(l.baseline_mm - r.baseline_mm) >= 1.0 and (
            _dilated(l) or _dilated(r)):
        urgent_size.append("双眼差 ≥1 mm 且一侧散大")

    for x in eyes:
        if not _dilated(x) and (_absent(x) or _weak(x)):
            soon_reflex.append(f"对光反射{_reflex_short(x)}（瞳孔未散大）")
        if x.npi is not None and x.npi < 3 and not _absent(x):
            soon_reflex.append(f"NPi {x.npi:.1f} <3")
        if x.shape_label == "欠规则" and _absent(x):
            soon_reflex.append("形态欠规则且反射未引出")
    for x in eyes:
        if _pinpoint(x):
            soon_size.append(f"瞳孔针尖样（{x.baseline_mm:.1f} mm）")
        elif _small(x):
            recheck.append(f"瞳孔偏小 {x.baseline_mm:.1f} mm（Horner 方向）")
        if x.shape_label == "欠规则":
            recheck.append("瞳孔形态欠规则")

    gaze_note = ""
    if unstable:
        gaze_note = "注视明显漂移，反射类指标暂不作为紧急依据"

    if urgent_size:
        triage = "紧急"
        reason = "；".join(dict.fromkeys(urgent_size + (urgent_reflex if not unstable else [])))
    elif urgent_reflex:
        if unstable:
            triage = "尽快"
            reason = "；".join(dict.fromkeys(urgent_reflex)) + "（" + gaze_note + "，请先复测）"
        else:
            triage, reason = "紧急", "；".join(dict.fromkeys(urgent_reflex))
    elif soon_size:
        triage, reason = "尽快", "；".join(dict.fromkeys(soon_size))
    elif soon_reflex:
        if unstable:
            triage = "复查"
            reason = "；".join(dict.fromkeys(soon_reflex)) + "（" + gaze_note + "）"
        else:
            triage, reason = "尽快", "；".join(dict.fromkeys(soon_reflex))
    elif recheck:
        triage, reason = "复查", "；".join(dict.fromkeys(recheck))
    else:
        triage, reason = "常规", "瞳孔大小与对光反射未见明显异常"
        if l.gcs is not None and l.gcs <= 8:
            triage = "复查"
            reason += f"；但 GCS {l.gcs}（≤8）属重型颅脑损伤，需按重症监护频率复评瞳孔"

    checklist = [
        "上睑下垂 / 睑裂宽度（Horner 的 Müller 肌仅解释 1–2 mm）",
        "面部少汗（同侧）",
        "调节反射（鉴别阿-罗、Adie 必需）",
        "用药：阿片类、抗胆碱药（阿托品/东莨菪碱）、缩瞳/散瞳滴眼液",
        "意识与 GCS、生命体征",
        "外伤史、眼部手术/疾病史、既往是否长期不等大",
        "年龄与环境光（老年常 2–3 mm、暗处 4–5 mm，对称且反射好即为生理）",
    ]
    if has_diplopia:
        checklist.append("复视：交替遮盖试验、马氏杆/红玻璃（滑车神经麻痹的垂直位微小差异）")
    return Assessment(triage, reason, cats, checklist)
