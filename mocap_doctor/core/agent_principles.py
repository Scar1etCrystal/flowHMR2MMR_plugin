"""agent_principles.py — 动画原理写工具 anticipation（预备）/ follow_through（跟随）/
overshoot（过冲）+ 配套读工具 analyze_motion（找发力起点/峰值/停止点/幅度/轴，
给可直接用的建议参数，修后复测）。

主通道时间线（四个工具共用）
--------------------------
主骨 = ``main_bone``（默认：窗口内峰值角速度最大的骨）。速度 =
``P.smooth(P.angular_speed_deg(q), smooth)``（局部旋转，度/帧，默认 3 帧平滑——
本 fixture 的手臂原始速度逐帧抖 1~2°/帧，不平滑起止点会乱跳）。
``P.detect_events`` 在主骨速度上找（窗口内索引，对外一律报帧号）：

- onset = 峰值前最后一个速度 ≤ onset_frac·峰值 的帧（发力起点：速度开始陡升）
- peak  = 速度峰值帧
- stop  = 峰值后第一个速度 ≤ stop_frac·峰值 的帧（"速度归零"）

``onset_frame`` / ``stop_frame`` 可钉死（钉了之后峰值只在 [onset, stop] 内找）。

每骨在这条时间线上量：
amplitude = 从 onset 位姿出发、[onset, stop] 内的最大角位移（度）；
运动轴 = ``P.motion_axis(q(onset), q(onset+k))``（动作初始方向，局部轴）；
接近轴 = ``P.motion_axis(q(stop−k), q(stop))``（停下前的运动方向）。
k = axis_frames（默认 3），k 帧内位移不足 1° 时自动加长。

三个写工具的区别（都写 preview delta strip，只写 frame_range 以内，params 全录、
可 reapply、可 dry_run）
-----------------------------------------------------------------------------
- anticipation：动作**开始前**先往反方向蓄一下。每骨绕 −运动轴 转
  amount×amplitude_b（平滑鼓包：onset−lead 升起、新起点 onset+delay 达峰、
  重映射后的峰值帧回落到 0），同时整个部位共用一条时间重映射：起点后挪
  delay 帧，动作段 [onset, E] 压缩进 [onset+delay, E]，E（=stop）之后完全不变。
- follow_through：动作**停下之后**的衰减振荡：绕接近轴
  A·exp(−τ/decay)·sin(2πτ/period)，第一瓣顺惯性继续往前、然后摆回、正负交替
  多瓣，cycles·period 帧后平滑归零；stop 之前偏移恒为 0。amount 定义为
  **第一瓣峰值**/amplitude（形状已归一化）。propagate>0 时链深度 d 的骨起振
  延后 d·propagate 帧、幅度 ×gain_per_depth^d（链末端甩得更晚）。
- overshoot：模拟"停下太快"：**单瓣**沿接近方向冲过终点 amount×amplitude
  （stop+peak_after 帧达峰），再用 settle 帧平滑回到原终点位姿，不振荡。过冲从
  stop 前 lead_in 帧（默认 = peak_after，不早于速度峰值）开始累积，把急停变成
  "带惯性冲过头再回来"，速度连续不跳变。

所有角度对外用度；四元数 (w,x,y,z)；帧号含两端。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import agent_ops, agent_pose as P

_BLEND = 3            # 三个写工具默认 taper 帧数
_MIN_AXIS_DEG = 1.0   # 取轴时 k 帧内位移不足它就加长 k
_MIN_AMP_DEG = 0.5    # 幅度低于它的骨不加偏移（只跟着时间重映射）
_LOBE_EPS_DEG = 0.2   # 数瓣/变号时忽略的小量
_NOISE_RATIO = 0.35   # 主骨速度中位数/峰值 超过它 → 提示噪声大

# ---------------------------------------------------------------------------
# small helpers


def _r(v, nd=2):
    return None if v is None else round(float(v), nd)


def _vec(v, nd=3):
    return [round(float(x), nd) for x in v]


def _speed(q: np.ndarray, smooth) -> np.ndarray:
    sp = P.angular_speed_deg(q)
    w = int(smooth or 1)
    return P.smooth(sp, w) if w > 1 else sp


def _rot(axis, deg) -> np.ndarray:
    """(T,) degrees about one fixed local axis → (T,4) quats."""
    rv = (np.asarray(axis, dtype=np.float64)[None, :]
          * np.radians(np.asarray(deg, dtype=np.float64))[:, None])
    return P.rotvec_to_quat(rv)


def _rotvec_deg(q_ref, q) -> np.ndarray:
    """Rotation taking q_ref → q, as a rotvec in degrees (q_ref's local frame)."""
    return np.degrees(P.quat_to_rotvec(P.qmul(P.qconj(q_ref), q)))


def _proj_deg(q_ref, q, axis) -> np.ndarray:
    """Signed rotation (deg) of q relative to q_ref about a local `axis`."""
    return _rotvec_deg(q_ref, q) @ np.asarray(axis, dtype=np.float64)


def _amplitude(q: np.ndarray, i0: int, i1: int) -> tuple[float, int]:
    """Max angular displacement (deg) from q[i0] over q[i0..i1], and its index."""
    i1 = max(int(i0), int(i1))
    ang = P.qangle_deg(q[i0], q[i0:i1 + 1])
    j = int(np.argmax(ang))
    return float(ang[j]), int(i0) + j


def _axis_forward(q: np.ndarray, i0: int, k0: int, limit: int):
    """Initial motion axis from q[i0] → q[i0+k]; k grows from k0 until the
    rotation reaches _MIN_AXIS_DEG (never past `limit`).  → (axis, deg, k)"""
    n = len(q)
    if i0 >= n - 1:
        return np.array([1.0, 0.0, 0.0]), 0.0, 0
    limit = max(i0 + 1, min(int(limit), n - 1))
    k = max(1, int(k0))
    while True:
        j = min(i0 + k, limit)
        ax, ang = P.motion_axis(q[i0], q[j])
        if ang >= _MIN_AXIS_DEG or j >= limit:
            return ax, ang, j - i0
        k += 1


def _axis_backward(q: np.ndarray, i1: int, k0: int, lower: int):
    """Approach axis q[i1−k] → q[i1]; k grows until ≥ _MIN_AXIS_DEG (never
    before `lower`).  → (axis, deg, k)"""
    if i1 <= 0:
        return np.array([1.0, 0.0, 0.0]), 0.0, 0
    lower = max(0, min(int(lower), i1 - 1))
    k = max(1, int(k0))
    while True:
        j = max(i1 - k, lower)
        ax, ang = P.motion_axis(q[j], q[i1])
        if ang >= _MIN_AXIS_DEG or j <= lower:
            return ax, ang, i1 - j
        k += 1


def _lobes(x: np.ndarray, eps: float) -> list[tuple[int, float]]:
    """Signed extreme of each same-sign run of x (|x| ≤ eps ignored)."""
    lobes: list[list] = []
    for i, v in enumerate(np.asarray(x, dtype=np.float64)):
        if abs(v) <= eps:
            continue
        sgn = 1 if v > 0 else -1
        if not lobes or lobes[-1][0] != sgn:
            lobes.append([sgn, i, float(v)])
        elif abs(v) > abs(lobes[-1][2]):
            lobes[-1][1], lobes[-1][2] = i, float(v)
    return [(int(i), float(v)) for _s, i, v in lobes]


def _events(speed: np.ndarray, a: int, *, onset_frac, stop_frac,
            onset_frame=None, stop_frame=None) -> dict:
    """Main-channel events (INDEX space).  Pins are absolute frames; with a
    pin the peak is searched only inside [onset, stop]."""
    T = len(speed)
    pins = {}
    for key, fr in (("onset", onset_frame), ("stop", stop_frame)):
        if fr is None:
            continue
        i = int(round(float(fr))) - a
        if not 0 <= i < T:
            raise RuntimeError(
                f"{key}_frame={fr} 不在 frame_range [{a}, {a + T - 1}] 内："
                f"放宽 frame_range 或去掉 {key}_frame")
        pins[key] = i
    peak_index = None
    if pins:
        lo, hi = pins.get("onset", 0), pins.get("stop", T - 1)
        if hi <= lo:
            raise RuntimeError(
                f"onset_frame（{a + lo}）必须早于 stop_frame（{a + hi}）")
        peak_index = lo + int(np.argmax(speed[lo:hi + 1]))
    ev = P.detect_events(speed, onset_frac=float(onset_frac),
                         stop_frac=float(stop_frac), peak_index=peak_index)
    ev.update(pins)
    ev["pinned"] = sorted(pins)
    return ev


def _pick_main(bones: Sequence[str], speeds: dict, main_bone=None) -> str:
    if main_bone:
        return main_bone
    return max(bones, key=lambda bn: float(np.max(speeds[bn])))


def _prepare(scene, armature, bones, frames, main_bone, smooth):
    """ONE frame sweep for the written bones (+ a read-only main bone)."""
    read = list(bones)
    if main_bone and main_bone not in read:
        read.append(main_bone)
    smp = P.sample_visible(scene, armature, read, frames)
    speeds = {bn: _speed(smp["quat"][bn], smooth) for bn in read}
    return _pick_main(bones, speeds, main_bone), smp, speeds


def _range_hint(lo: int, hi: int) -> str:
    return f"建议 frame_range=[{int(lo)}, {int(hi)}]"


def _ignored(extra: dict) -> list:
    return sorted(extra)


def _merge_info(metrics: dict, info: dict, strength: float, keys) -> None:
    """pose_deltas info → per-bone rows; offsets reported AFTER strength."""
    k = float(strength)
    for bn, row in metrics["bones"].items():
        row.update(info.get(bn, {}))
        for key in keys:
            if row.get(key) is not None:
                row[key] = _r(row[key] * k)
        if "lobes" in row:
            row["lobes"] = [[f, _r(v * k)] for f, v in row["lobes"]]


# ---------------------------------------------------------------------------
# anticipation


def _anticipation_remap(T: int, o: int, E: int, p: int, lead: int, delay: int):
    """New-time → original-time map T(t) (monotone, piecewise linear) and the
    counter-move bump g(t) ∈ [0,1].  Index space."""
    t = np.arange(T, dtype=np.float64)
    tm = t.copy()
    r0 = o - lead
    rise = float(lead + delay)
    m1 = (t >= r0) & (t <= o + delay)
    tm[m1] = r0 + (t[m1] - r0) * (lead / rise)          # 拉长静止段
    span = float(E - o)
    scale = span / (span - delay)
    m2 = (t > o + delay) & (t <= E)
    tm[m2] = o + (t[m2] - (o + delay)) * scale          # 压缩动作段
    p_new = (o + delay) + (p - o) / scale               # 重映射后的峰值帧
    g = np.zeros(T)
    g[m1] = P.smoothstep((t[m1] - r0) / rise)
    f_end = max(o + delay + 1.0, p_new)
    m3 = (t > o + delay) & (t < f_end)
    g[m3] = 1.0 - P.smoothstep((t[m3] - (o + delay)) / (f_end - (o + delay)))
    return tm, g, scale, p_new, f_end


def _solve_anticipation(quat: dict, bones, main, speed_main, a, *, amount,
                        lead, delay, onset_frac, stop_frac, onset_frame,
                        stop_frame, axis_frames, blend):
    T = len(speed_main)
    b = a + T - 1
    ev = _events(speed_main, a, onset_frac=onset_frac, stop_frac=stop_frac,
                 onset_frame=onset_frame, stop_frame=stop_frame)
    o, p, s = ev["onset"], ev["peak"], ev["stop"]
    if o is None:
        raise RuntimeError(
            f"窗口 {a}-{b} 内找不到发力起点：主骨 {main} 的速度在峰值帧 {a + p} 之前"
            f"从没降到 {onset_frac}×峰值。把 frame_range 起点往前挪到动作前的静止段，"
            f"{_range_hint(a - 15, b)}；或直接给 onset_frame")
    E = s if s is not None else T - 1 - blend
    probs = []
    lo_need = a + o - lead - blend
    hi_need = (a + s + blend) if s is not None else b
    if o - lead < blend:
        probs.append(f"起点余量不够：onset {a + o} − lead {lead} = {a + o - lead}，"
                     f"要 ≥ 窗口起点+blend = {a + blend}")
    if s is not None and s > T - 1 - blend:
        probs.append(f"停止点 {a + s} 离窗口末端太近：要 ≤ 窗口末端−blend = {b - blend}"
                     "（E 之后必须完全不变，taper 区不能有改动）")
    if E <= p:
        probs.append(f"峰值 {a + p} 不在 E={a + E} 之前（没找到停止点，窗口末端太早）")
    elif E - o < delay + 2:
        probs.append(f"动作段太短：onset {a + o} → E {a + E} 只有 {E - o} 帧，"
                     f"至少要 delay+2 = {delay + 2} 帧")
    if probs:
        hi = max(b, hi_need + 2) if s is not None else b + 15
        raise RuntimeError("anticipation 窗口不合适：" + "；".join(probs)
                           + "。" + _range_hint(min(a, lo_need - 2), hi))
    tm, g, scale, p_new, f_end = _anticipation_remap(T, o, E, p, lead, delay)
    desired, per = {}, {}
    for bn in bones:
        q = quat[bn]
        amp, imax = _amplitude(q, o, E)
        axis, ang, k = _axis_forward(q, o, axis_frames, E)
        C = float(amount) * amp if (amp >= _MIN_AMP_DEG and ang > 1e-6) else 0.0
        q_rm = P.resample_quats(q, 0, tm)
        desired[bn] = P.qmul(q_rm, _rot(-axis, C * g))
        per[bn] = {"amplitude_deg": _r(amp), "max_disp_frame": a + imax,
                   "counter_deg": _r(C), "axis": _vec(axis), "axis_frames": int(k)}
    metrics = {
        "main_bone": main,
        "onset_frame": a + o, "new_onset_frame": a + o + delay,
        "peak_frame": a + p, "new_peak_frame": _r(a + p_new, 1),
        "stop_frame": (a + s) if s is not None else None,
        "E_frame": a + E, "stop_detected": s is not None,
        "peak_speed": _r(ev["peak_speed"]),
        "amplitude_deg": per[main]["amplitude_deg"] if main in per else None,
        "counter_deg": per[main]["counter_deg"] if main in per else None,
        "time_scale": _r(scale, 3),
        "counter_rise": [a + o - lead, a + o + delay],
        "counter_fall_end": _r(a + f_end, 1),
        "changed_frames": [a + o - lead, a + E],
        "pinned": ev["pinned"],
        "bones": per,
    }
    if main not in per:   # read-only main bone: still report its numbers
        amp, _ = _amplitude(quat[main], o, E)
        metrics["amplitude_deg"] = _r(amp)
        metrics["counter_deg"] = _r(float(amount) * amp)
    return desired, metrics


def anticipation(scene, armature, *, bones, frame_range, main_bone=None,
                 onset_frame=None, stop_frame=None, amount=0.15, lead=6,
                 delay=2, onset_frac=0.15, stop_frac=0.12, smooth=3,
                 axis_frames=3, strength=1.0, blend=_BLEND,
                 op_mode="preview", data_dir=None, track_name=None,
                 dry_run=False, record=True, **extra):
    """预备：在发力起点前加反方向小位移 + 起点后挪 delay 帧（时间重映射补回总时长）。"""
    a, b, frames = P.strip_window(frame_range)
    bones = P.resolve_pose_bones(armature, bones)
    lead, delay, blend = int(lead), int(delay), int(blend)
    if lead < 0 or delay < 0 or lead + delay < 1:
        raise RuntimeError("lead/delay 不能为负，且 lead+delay ≥ 1（反向位移要有帧数起势）")
    main, smp, speeds = _prepare(scene, armature, bones, frames, main_bone, smooth)
    desired, metrics = _solve_anticipation(
        smp["quat"], bones, main, speeds[main], a, amount=float(amount),
        lead=lead, delay=delay, onset_frac=float(onset_frac),
        stop_frac=float(stop_frac), onset_frame=onset_frame,
        stop_frame=stop_frame, axis_frames=int(axis_frames), blend=blend)
    scalars, quats, info = P.pose_deltas(armature, smp, desired_quat=desired,
                                         strength=float(strength))
    _merge_info(metrics, info, strength, ("counter_deg",))
    if metrics["counter_deg"] is not None:
        metrics["counter_deg"] = _r(metrics["counter_deg"] * float(strength))
    metrics["verify"] = (
        f"analyze_motion 复测（同 frame_range、main_bone，不钉 onset）："
        f"main.onset_frame≈{metrics['new_onset_frame']}、"
        f"main.counter_move_deg≈{metrics['counter_deg']}、counter_dir_cos≈-1；"
        f"{metrics['E_frame']} 帧之后位姿不变")
    params = {"bones": list(bones), "frame_range": [a, b],
              "main_bone": main_bone, "onset_frame": onset_frame,
              "stop_frame": stop_frame, "amount": float(amount),
              "lead": lead, "delay": delay, "onset_frac": float(onset_frac),
              "stop_frac": float(stop_frac), "smooth": int(smooth),
              "axis_frames": int(axis_frames), "strength": float(strength),
              "blend": blend}
    if _ignored(extra):
        metrics["ignored_args"] = _ignored(extra)
    if dry_run:
        return {"dry_run": True, "params": params, "metrics": metrics,
                "frames": [a, b]}
    track, strip = P.write_pose(armature, f"agent_antic_{a}_{b}", a, scalars,
                                quats, blend=blend, track_name=track_name)
    op = agent_ops._new_op("anticipation", params, (a, b), strip.name,
                           op_mode, metrics, track=track.name)
    return agent_ops._record(data_dir, op) if (data_dir and record) else op


# ---------------------------------------------------------------------------
# follow_through


def _ft_unit(tau, period: float, decay: float, cycles: float) -> np.ndarray:
    """exp(−τ/decay)·sin(2πτ/period), 0 for τ≤0, tapered to 0 over the last
    period before τ = cycles·period (zero value AND slope at the end)."""
    L = cycles * period
    taper = min(period, L)
    tau = np.asarray(tau, dtype=np.float64)
    out = np.zeros_like(tau)
    m = (tau > 0) & (tau < L)
    x = tau[m]
    out[m] = (np.exp(-x / decay) * np.sin(2.0 * np.pi * x / period)
              * (1.0 - P.smoothstep((x - (L - taper)) / taper)))
    return out


def _ft_norm(period: float, decay: float, cycles: float) -> float:
    """1 / first-lobe peak of the unit shape → amount = first-lobe / amplitude."""
    x = np.linspace(0.0, period / 2.0, 401)
    m = float(_ft_unit(x, period, decay, cycles).max())
    return 1.0 / m if m > 1e-9 else 0.0


def _solve_follow(quat: dict, bones, main, speed_main, a, depths: dict, *,
                  amount, period, decay, cycles, propagate, gain_per_depth,
                  onset_frac, stop_frac, onset_frame, stop_frame, axis_frames,
                  blend):
    T = len(speed_main)
    b = a + T - 1
    if period < 2 or decay <= 0 or cycles < 0.5:
        raise RuntimeError("period ≥ 2 帧、decay > 0、cycles ≥ 0.5")
    if propagate < 0 or gain_per_depth <= 0:
        raise RuntimeError("propagate ≥ 0、gain_per_depth > 0")
    ev = _events(speed_main, a, onset_frac=onset_frac, stop_frac=stop_frac,
                 onset_frame=onset_frame, stop_frame=stop_frame)
    o, p, s = ev["onset"], ev["peak"], ev["stop"]
    L = cycles * period
    if s is None:
        raise RuntimeError(
            f"窗口 {a}-{b} 内找不到停止点：主骨 {main} 峰值帧 {a + p} 之后速度没降到 "
            f"{stop_frac}×峰值。把 frame_range 终点往后延（还要留 cycles·period="
            f"{L:g} 帧振荡 + blend），{_range_hint(a, b + int(L) + blend + 15)}；"
            "或直接给 stop_frame")
    o_eff = o if o is not None else 0
    start = {bn: s + depths.get(bn, 0) * propagate for bn in bones}
    end = max(start.values()) + L
    probs = []
    if s < blend:
        probs.append(f"停止点 {a + s} 离窗口起点太近：要 ≥ 窗口起点+blend = {a + blend}")
    if end > T - 1 - blend:
        probs.append(f"振荡要到 {a + end:g} 帧才结束（stop {a + s} + "
                     f"{'深度延迟 + ' if propagate else ''}cycles·period {L:g}），"
                     f"窗口末端要 ≥ {a + end + blend:g}")
    if probs:
        raise RuntimeError("follow_through 窗口不合适：" + "；".join(probs) + "。"
                           + _range_hint(min(a, a + s - blend - 4),
                                         max(b, int(np.ceil(a + end)) + blend + 2)))
    N = _ft_norm(period, decay, cycles)
    t = np.arange(T, dtype=np.float64)
    desired, per = {}, {}
    for bn in bones:
        q = quat[bn]
        d = int(depths.get(bn, 0))
        amp, _ = _amplitude(q, o_eff, s)
        axis, ang, k = _axis_backward(q, s, axis_frames, lower=min(p, s - 1))
        A = (float(amount) * amp * (gain_per_depth ** d)
             if (amp >= _MIN_AMP_DEG and ang > 1e-6) else 0.0)
        theta = A * N * _ft_unit(t - start[bn], period, decay, cycles)
        desired[bn] = P.qmul(q, _rot(axis, theta))
        lob = _lobes(theta, _LOBE_EPS_DEG * 0.25)
        per[bn] = {"depth": d, "start_frame": _r(a + start[bn], 1),
                   "amplitude_deg": _r(amp), "first_lobe_deg": _r(A),
                   "approach_axis": _vec(axis), "axis_frames": int(k),
                   "lobes": [[a + i, _r(v)] for i, v in lob[:6]]}
    metrics = {
        "main_bone": main,
        "onset_frame": (a + o) if o is not None else None,
        "peak_frame": a + p, "stop_frame": a + s,
        "peak_speed": _r(ev["peak_speed"]),
        "oscillation_end_frame": _r(a + end, 1),
        "changed_frames": [a + s, int(np.ceil(a + end))],
        "pinned": ev["pinned"],
        "bones": per,
    }
    src = per.get(main)
    if src is None:   # read-only main bone
        amp, _ = _amplitude(quat[main], o_eff, s)
        src = {"amplitude_deg": _r(amp), "first_lobe_deg": _r(float(amount) * amp),
               "lobes": []}
    metrics["amplitude_deg"] = src["amplitude_deg"]
    metrics["first_lobe_deg"] = src["first_lobe_deg"]
    metrics["expected_lobes"] = [list(x) for x in src["lobes"]]
    if o is None:
        metrics["note"] = "窗口内没找到 onset，幅度从窗口起点量"
    return desired, metrics


def follow_through(scene, armature, *, bones, frame_range, main_bone=None,
                   stop_frame=None, onset_frame=None, amount=0.12, period=8,
                   decay=6.0, cycles=2.0, onset_frac=0.15, stop_frac=0.12,
                   propagate=0, gain_per_depth=1.0, smooth=3, axis_frames=3,
                   strength=1.0, blend=_BLEND, op_mode="preview",
                   data_dir=None, track_name=None, dry_run=False, record=True,
                   **extra):
    """跟随：停止点之后加衰减振荡（第一瓣顺惯性往前，正负交替，按链深度可延后）。"""
    a, b, frames = P.strip_window(frame_range)
    bones = P.resolve_pose_bones(armature, bones)
    blend = int(blend)
    main, smp, speeds = _prepare(scene, armature, bones, frames, main_bone, smooth)
    depths = {bn: P.depth_in(armature, bn, bones) for bn in bones}
    desired, metrics = _solve_follow(
        smp["quat"], bones, main, speeds[main], a, depths,
        amount=float(amount), period=float(period), decay=float(decay),
        cycles=float(cycles), propagate=float(propagate),
        gain_per_depth=float(gain_per_depth), onset_frac=float(onset_frac),
        stop_frac=float(stop_frac), onset_frame=onset_frame,
        stop_frame=stop_frame, axis_frames=int(axis_frames), blend=blend)
    scalars, quats, info = P.pose_deltas(armature, smp, desired_quat=desired,
                                         strength=float(strength))
    _merge_info(metrics, info, strength, ("first_lobe_deg",))
    if metrics["first_lobe_deg"] is not None:
        metrics["first_lobe_deg"] = _r(metrics["first_lobe_deg"] * float(strength))
        metrics["expected_lobes"] = [[f, _r(v * float(strength))]
                                     for f, v in metrics["expected_lobes"]]
    metrics["verify"] = (
        f"analyze_motion 复测（同 frame_range、main_bone，stop_frame={metrics['stop_frame']}，"
        f"baseline_op=<本 op_id>）：vs_baseline.bones[主骨].approach_lobes 第一瓣≈"
        f"+{metrics['first_lobe_deg']}°、至少 2 次变号；stop 之前 / "
        f"{metrics['changed_frames'][1]} 之后 diff≈0")
    params = {"bones": list(bones), "frame_range": [a, b],
              "main_bone": main_bone, "onset_frame": onset_frame,
              "stop_frame": stop_frame, "amount": float(amount),
              "period": float(period), "decay": float(decay),
              "cycles": float(cycles), "propagate": float(propagate),
              "gain_per_depth": float(gain_per_depth),
              "onset_frac": float(onset_frac), "stop_frac": float(stop_frac),
              "smooth": int(smooth), "axis_frames": int(axis_frames),
              "strength": float(strength), "blend": blend}
    if _ignored(extra):
        metrics["ignored_args"] = _ignored(extra)
    if dry_run:
        return {"dry_run": True, "params": params, "metrics": metrics,
                "frames": [a, b]}
    track, strip = P.write_pose(armature, f"agent_follow_{a}_{b}", a, scalars,
                                quats, blend=blend, track_name=track_name)
    op = agent_ops._new_op("follow_through", params, (a, b), strip.name,
                           op_mode, metrics, track=track.name)
    return agent_ops._record(data_dir, op) if (data_dir and record) else op


# ---------------------------------------------------------------------------
# overshoot


def _solve_overshoot(quat: dict, bones, main, speed_main, a, *, amount,
                     peak_after, settle, lead_in, onset_frac, stop_frac,
                     onset_frame, stop_frame, axis_frames, blend):
    T = len(speed_main)
    b = a + T - 1
    if peak_after < 0 or settle < 1 or (lead_in is not None and lead_in < 0):
        raise RuntimeError("peak_after ≥ 0、settle ≥ 1、lead_in ≥ 0")
    ev = _events(speed_main, a, onset_frac=onset_frac, stop_frac=stop_frac,
                 onset_frame=onset_frame, stop_frame=stop_frame)
    o, p, s = ev["onset"], ev["peak"], ev["stop"]
    if s is None:
        raise RuntimeError(
            f"窗口 {a}-{b} 内找不到停止点：主骨 {main} 峰值帧 {a + p} 之后速度没降到 "
            f"{stop_frac}×峰值。把 frame_range 终点往后延，"
            f"{_range_hint(a, b + peak_after + settle + blend + 15)}；或直接给 stop_frame")
    lead_in = int(peak_after if lead_in is None else lead_in)
    o_eff = o if o is not None else 0
    t_pk = s + peak_after
    t_end = t_pk + settle
    t0 = max(s - lead_in, p)            # 不早于速度峰值（还在减速段里累积）
    probs = []
    if t_pk - t0 < 1:
        probs.append("peak_after + lead_in 至少要 1 帧（过冲要有帧数起势）")
    if t0 < blend:
        probs.append(f"过冲起点 {a + t0} 离窗口起点太近：要 ≥ 窗口起点+blend = {a + blend}")
    if t_end > T - 1 - blend:
        probs.append(f"回位要到 {a + t_end} 帧（stop {a + s} + peak_after {peak_after}"
                     f" + settle {settle}），窗口末端要 ≥ {a + t_end + blend}")
    if probs:
        raise RuntimeError("overshoot 窗口不合适：" + "；".join(probs) + "。"
                           + _range_hint(min(a, a + t0 - blend - 2),
                                         max(b, a + t_end + blend + 2)))
    t = np.arange(T, dtype=np.float64)
    g = np.zeros(T)
    m = (t >= t0) & (t <= t_pk)
    g[m] = P.smoothstep((t[m] - t0) / float(max(t_pk - t0, 1)))
    m = (t > t_pk) & (t < t_end)
    g[m] = 1.0 - P.smoothstep((t[m] - t_pk) / float(settle))
    desired, per = {}, {}
    for bn in bones:
        q = quat[bn]
        amp, _ = _amplitude(q, o_eff, s)
        axis, ang, k = _axis_backward(q, s, axis_frames, lower=min(p, s - 1))
        A = float(amount) * amp if (amp >= _MIN_AMP_DEG and ang > 1e-6) else 0.0
        desired[bn] = P.qmul(q, _rot(axis, A * g))
        per[bn] = {"amplitude_deg": _r(amp), "overshoot_deg": _r(A),
                   "approach_axis": _vec(axis), "axis_frames": int(k)}
    amp_main = (per[main]["amplitude_deg"] if main in per
                else _r(_amplitude(quat[main], o_eff, s)[0]))
    metrics = {
        "main_bone": main,
        "onset_frame": (a + o) if o is not None else None,
        "peak_frame": a + p, "stop_frame": a + s,
        "peak_speed": _r(ev["peak_speed"]),
        "rise_start_frame": a + t0, "overshoot_peak_frame": a + t_pk,
        "settle_end_frame": a + t_end,
        "amplitude_deg": amp_main,
        "overshoot_deg": (per[main]["overshoot_deg"] if main in per
                          else _r(float(amount) * amp_main)),
        "changed_frames": [a + t0, a + t_end],
        "pinned": ev["pinned"],
        "bones": per,
    }
    if o is None:
        metrics["note"] = "窗口内没找到 onset，幅度从窗口起点量"
    return desired, metrics


def overshoot(scene, armature, *, bones, frame_range, main_bone=None,
              stop_frame=None, onset_frame=None, amount=0.08, peak_after=2,
              settle=6, lead_in=None, onset_frac=0.15, stop_frac=0.12,
              smooth=3, axis_frames=3, strength=1.0, blend=_BLEND,
              op_mode="preview", data_dir=None, track_name=None,
              dry_run=False, record=True, **extra):
    """过冲：停下太快 → 单瓣冲过终点 amount×amplitude，settle 帧平滑回原终点（不振荡）。"""
    a, b, frames = P.strip_window(frame_range)
    bones = P.resolve_pose_bones(armature, bones)
    blend = int(blend)
    main, smp, speeds = _prepare(scene, armature, bones, frames, main_bone, smooth)
    desired, metrics = _solve_overshoot(
        smp["quat"], bones, main, speeds[main], a, amount=float(amount),
        peak_after=int(peak_after), settle=int(settle),
        lead_in=None if lead_in is None else int(lead_in),
        onset_frac=float(onset_frac), stop_frac=float(stop_frac),
        onset_frame=onset_frame, stop_frame=stop_frame,
        axis_frames=int(axis_frames), blend=blend)
    scalars, quats, info = P.pose_deltas(armature, smp, desired_quat=desired,
                                         strength=float(strength))
    _merge_info(metrics, info, strength, ("overshoot_deg",))
    if metrics["overshoot_deg"] is not None:
        metrics["overshoot_deg"] = _r(metrics["overshoot_deg"] * float(strength))
    metrics["verify"] = (
        f"analyze_motion 复测（同 frame_range、main_bone，stop_frame={metrics['stop_frame']}，"
        f"baseline_op=<本 op_id>）：vs_baseline.bones[主骨].approach_peak_deg≈"
        f"{metrics['overshoot_deg']} @ {metrics['overshoot_peak_frame']}，单瓣；"
        f"{metrics['settle_end_frame']} 之后 diff≈0")
    params = {"bones": list(bones), "frame_range": [a, b],
              "main_bone": main_bone, "onset_frame": onset_frame,
              "stop_frame": stop_frame, "amount": float(amount),
              "peak_after": int(peak_after), "settle": int(settle),
              "lead_in": None if lead_in is None else int(lead_in),
              "onset_frac": float(onset_frac), "stop_frac": float(stop_frac),
              "smooth": int(smooth), "axis_frames": int(axis_frames),
              "strength": float(strength), "blend": blend}
    if _ignored(extra):
        metrics["ignored_args"] = _ignored(extra)
    if dry_run:
        return {"dry_run": True, "params": params, "metrics": metrics,
                "frames": [a, b]}
    track, strip = P.write_pose(armature, f"agent_overshoot_{a}_{b}", a,
                                scalars, quats, blend=blend,
                                track_name=track_name)
    op = agent_ops._new_op("overshoot", params, (a, b), strip.name, op_mode,
                           metrics, track=track.name)
    return agent_ops._record(data_dir, op) if (data_dir and record) else op


# ---------------------------------------------------------------------------
# analyze_motion (read-only)


def _series(frames, sp, max_points: int):
    """Bucket-max downsample (keeps peaks) → [[frame, deg/frame], ...]."""
    T = len(sp)
    n = max(2, int(max_points))
    if T <= n:
        return [[int(f), _r(v)] for f, v in zip(frames, sp)], False
    edges = np.linspace(0, T, n + 1).astype(int)
    out = []
    for i0, i1 in zip(edges[:-1], edges[1:]):
        if i1 <= i0:
            continue
        j = i0 + int(np.argmax(sp[i0:i1]))
        out.append([int(frames[j]), _r(sp[j])])
    return out, True


# 抖动闸门阈值（tools_io.md §8.1）。2026-10-06：用户要求 analyze_motion 自己报警，
# 并把 peak 阈值从 1.5 放宽到 2.0（"更干脆/力量感"这类任务峰速上升是预期的）。
_GATE_SPIKE_RATIO = 2.0      # 某帧速度 / 前后两帧均值 > 此值 = 孤峰（不需要修前数据）
_GATE_JITTER_RATIO = 2.0     # 修后 jitter / 修前 jitter > 此值 = 不通过
_GATE_PEAK_RATIO = 2.0       # 修后 peak_speed / 修前 > 此值 = 不通过（原 1.5，已放宽）


def _jitter_mean(q) -> float:
    """每帧偏离"前后两帧 slerp 中点"的平均角度（度）——和 _bone_report 同口径。"""
    if q is None or len(q) < 3:
        return 0.0
    return float(P.qangle_deg(q[1:-1], P.slerp(q[:-2], q[2:], 0.5)).mean())


def _bone_report(q, sp, a, *, o, p, s, onset_frac, stop_frac, axis_frames,
                 lookback):
    """Per-bone numbers: own events + measurements on the MAIN timeline."""
    T = len(q)
    own = P.detect_events(sp, onset_frac=onset_frac, stop_frac=stop_frac)
    amp, imax = _amplitude(q, o, s)
    ax, ax_deg, k1 = _axis_forward(q, o, axis_frames, s)
    apx, ap_deg, k2 = _axis_backward(q, s, axis_frames, lower=min(p, s - 1))
    row = {
        "peak_frame": a + own["peak"], "peak_speed": _r(own["peak_speed"]),
        "onset_frame": None if own["onset"] is None else a + own["onset"],
        "stop_frame": None if own["stop"] is None else a + own["stop"],
        "amplitude_deg": _r(amp), "max_disp_frame": a + imax,
        "axis": _vec(ax), "axis_deg": _r(ax_deg), "axis_frames": int(k1),
        "approach_axis": _vec(apx), "approach_deg": _r(ap_deg),
    }
    # 抖动：每帧偏离"前后两帧 slerp 中点"的角度（度）。匀速转动≈0，越大越抖。
    # 去抖（clean_jitter）前后对比看这个数。
    if T >= 3:
        jit = P.qangle_deg(q[1:-1], P.slerp(q[:-2], q[2:], 0.5))
        row["jitter_deg"] = _r(float(jit.mean()))
        row["jitter_max_deg"] = _r(float(jit.max()))
        top = np.argsort(jit)[::-1][:5]          # 最抖的 5 帧（帧号, 度）
        row["jitter_top_frames"] = [[int(a + 1 + i), _r(float(jit[i]))] for i in top]
    # counter-move: earlier poses that sit AHEAD (along +axis) of the onset
    # pose = the bone came BACK (−axis) before launching.  ≈ amount×amplitude
    # after anticipation, ≈0 on a plain start.
    lo = max(0, o - int(lookback))
    if o > lo:
        rv = _rotvec_deg(q[o], q[lo:o])
        pr = rv @ ax
        j = int(np.argmax(pr))
        row["counter_move_deg"] = _r(max(0.0, float(pr[j])))
        row["counter_from_frame"] = a + lo + j
        nrm = float(np.linalg.norm(rv[j]))
        row["counter_dir_cos"] = (_r(-float(pr[j]) / nrm, 3)
                                  if (nrm > 1e-6 and pr[j] > 0) else None)
    else:
        row["counter_move_deg"] = 0.0
        row["counter_from_frame"] = None
        row["counter_dir_cos"] = None
    # after the stop: excursion past the settled (window-end) pose along the
    # approach direction.  +ve = beyond the end pose (overshoot / lobes).
    e = _proj_deg(q[T - 1], q[s:], apx)
    eps = max(_LOBE_EPS_DEG, 0.05 * float(np.abs(e).max()) if len(e) else 0.0)
    lob = _lobes(e, eps)
    j = int(np.argmax(e)) if len(e) else 0
    row["overshoot_deg"] = _r(max(0.0, float(e[j])) if len(e) else 0.0)
    row["overshoot_frame"] = (a + s + j) if len(e) and e[j] > 0 else None
    row["post_stop_lobes"] = [[a + s + i, _r(v)] for i, v in lob[:6]]
    row["post_stop_sign_changes"] = max(0, len(lob) - 1)
    return row


def _baseline_report(qb, qc, a, *, o, p, s, axis_frames):
    """Current vs baseline (op muted): exact per-frame difference."""
    rv = _rotvec_deg(qb, qc)                     # (T,3) baseline-local frame
    mag = np.linalg.norm(rv, axis=1)
    ch = np.nonzero(mag > 0.05)[0]
    ax, _d, _k = _axis_forward(qb, o, axis_frames, s)
    apx, _d, _k = _axis_backward(qb, s, axis_frames, lower=min(p, s - 1))
    along_m = rv @ ax
    along_a = rv[s:] @ apx
    lob = _lobes(along_a, max(_LOBE_EPS_DEG, 0.05 * float(np.abs(along_a).max())))
    jm = int(np.argmax(mag))
    jn = int(np.argmin(along_m))
    ja = int(np.argmax(along_a)) if len(along_a) else 0
    return {
        "max_diff_deg": _r(mag[jm]), "max_diff_frame": a + jm,
        "changed_frames": ([a + int(ch[0]), a + int(ch[-1])] if len(ch) else None),
        "before_stop_max_deg": _r(mag[:s].max()) if s > 0 else 0.0,
        "along_motion_min_deg": _r(along_m[jn]), "along_motion_min_frame": a + jn,
        "approach_peak_deg": _r(along_a[ja]) if len(along_a) else None,
        "approach_peak_frame": (a + s + ja) if len(along_a) else None,
        "approach_lobes": [[a + s + i, _r(v)] for i, v in lob[:6]],
        "approach_sign_changes": max(0, len(lob) - 1),
    }


def analyze_motion(scene, armature, *, bones, frame_range, main_bone=None,
                   onset_frame=None, stop_frame=None, onset_frac=0.15,
                   stop_frac=0.12, max_points=40, smooth=3, axis_frames=3,
                   lookback=12, baseline_tracks=None, **extra):
    """只读：主通道事件 + 每骨幅度/轴/反向位移/过冲 + 降采样速度 + 三个写工具的建议参数。

    复测约定（修后再调一次）：
    - anticipation：不钉 onset → main.onset_frame 应后移 delay，
      main.counter_move_deg ≈ amount×amplitude，counter_dir_cos ≈ −1
      （lookback 要 ≥ lead+delay，默认 12 帧）。
    - follow_through / overshoot：钉住修前的 stop_frame（振荡/过冲本身会把重新检测
      的 stop 挪走），并给 baseline_op=<op_id> → vs_baseline 里是"当前 − 该 op 之前"
      的逐帧精确差（approach_lobes / approach_peak_deg / changed_frames）。
      main.overshoot_deg 是相对窗口末端位姿量的绝对值，stop 后原动作还在漂时不准。

    baseline_tracks：要临时静音的轨（某个 op 的轨），读完恢复原静音状态。"""
    mask = extra.pop("mask_tracks", None)    # 桥内部参数（见 vs_baseline），不对外
    a, b, frames = P.strip_window(frame_range)
    bones = P.resolve_pose_bones(armature, bones)
    onset_frac, stop_frac = float(onset_frac), float(stop_frac)
    main, smp, speeds = _prepare(scene, armature, bones, frames, main_bone, smooth)
    T = len(frames)
    ev = _events(speeds[main], a, onset_frac=onset_frac, stop_frac=stop_frac,
                 onset_frame=onset_frame, stop_frame=stop_frame)
    o, p, s = ev["onset"], ev["peak"], ev["stop"]
    o_eff = o if o is not None else 0
    s_eff = s if s is not None else T - 1
    names = list(bones) + ([main] if main not in bones else [])
    rows = {}
    for bn in names:
        rows[bn] = _bone_report(smp["quat"][bn], speeds[bn], a, o=o_eff, p=p,
                                s=s_eff, onset_frac=onset_frac,
                                stop_frac=stop_frac,
                                axis_frames=int(axis_frames),
                                lookback=int(lookback))
        rows[bn]["rot_mode"] = smp["mode"][bn]
        if bn not in bones:
            rows[bn]["read_only"] = True
    mrow = rows[main]
    amp = float(mrow["amplitude_deg"] or 0.0)
    pk = float(ev["peak_speed"])
    series, trunc = _series(frames, speeds[main], max_points)
    warnings = []
    if pk < 3.0:
        warnings.append(f"主骨 {main} 峰值只有 {pk:.1f}°/帧（<3），动作偏弱，三个写工具效果会很小")
    if o is None:
        warnings.append("没找到 onset（峰值前速度没降到 onset_frac×峰值）：frame_range 往前扩到静止段")
    if s is None:
        warnings.append("没找到 stop（峰值后速度没降到 stop_frac×峰值）：frame_range 往后扩")
    med = float(np.median(speeds[main]))
    if pk > 0 and med / pk > _NOISE_RATIO:
        warnings.append(f"主骨 {main} 速度中位数/峰值={med / pk:.2f}，整段都在动或噪声大，"
                        "起止点可能不准；考虑换 main_bone 或缩窄窗口")
    drift = float(P.qangle_deg(smp["quat"][main][s_eff], smp["quat"][main][T - 1]))
    if s is not None and drift > 3.0:
        warnings.append(f"主骨 stop({a + s}) 之后到窗口末端还动了 {drift:.1f}°：overshoot_deg / "
                        "post_stop_lobes 是相对窗口末端位姿量的，不准——修后复测用 baseline_op")
    ranking = sorted(((bn, float(np.max(speeds[bn]))) for bn in bones),
                     key=lambda kv: -kv[1])[:4]
    blend = _BLEND
    sug = {}
    if o is not None and s is not None:
        of, sf = a + o, a + s
        pins = {"main_bone": main, "onset_frame": of, "stop_frame": sf}
        sug["anticipation"] = {
            "args": {"frame_range": [of - 6 - blend - 2, sf + blend + 2], **pins,
                     "amount": 0.15, "lead": 6, "delay": 2, "blend": blend},
            "expect": {"counter_move_deg": _r(0.15 * amp),
                       "new_onset_frame": of + 2, "unchanged_from_frame": sf},
            "retest": "analyze_motion 同 frame_range + main_bone（不钉 onset/stop）："
                      "main.onset_frame 应 ≈ new_onset_frame，main.counter_move_deg ≈ "
                      "expect，counter_dir_cos ≈ -1"}
        ft_end = sf + 16
        sug["follow_through"] = {
            "args": {"frame_range": [of - blend - 2, ft_end + blend + 2], **pins,
                     "amount": 0.12, "period": 8, "decay": 6, "cycles": 2,
                     "blend": blend},
            "expect": {"first_lobe_deg": _r(0.12 * amp), "stop_frame": sf,
                       "oscillation_end_frame": ft_end},
            "retest": "analyze_motion 同 frame_range + main_bone + stop_frame + "
                      "baseline_op=<op_id>：vs_baseline.bones[主骨].approach_lobes "
                      "第一瓣 ≈ expect、正负交替 ≥2 次变号"}
        ov_end = sf + 2 + 6
        sug["overshoot"] = {
            "args": {"frame_range": [of - blend - 2, ov_end + blend + 2], **pins,
                     "amount": 0.08, "peak_after": 2, "settle": 6, "blend": blend},
            "expect": {"overshoot_deg": _r(0.08 * amp), "overshoot_peak_frame": sf + 2,
                       "settle_end_frame": ov_end},
            "retest": "analyze_motion 同 frame_range + main_bone + stop_frame + "
                      "baseline_op=<op_id>：vs_baseline.bones[主骨].approach_peak_deg "
                      "≈ expect @ overshoot_peak_frame，单瓣"}
    res = {
        "frame_range": [a, b], "main_bone": main,
        "main": {"onset_frame": None if o is None else a + o,
                 "peak_frame": a + p, "peak_speed": _r(pk),
                 "stop_frame": None if s is None else a + s,
                 "amplitude_deg": mrow["amplitude_deg"],
                 "counter_move_deg": mrow["counter_move_deg"],
                 "counter_dir_cos": mrow["counter_dir_cos"],
                 "overshoot_deg": mrow["overshoot_deg"],
                 "overshoot_frame": mrow["overshoot_frame"],
                 "post_stop_sign_changes": mrow["post_stop_sign_changes"],
                 "post_stop_drift_deg": _r(drift),
                 "pinned": ev["pinned"]},
        "counter_move_deg": mrow["counter_move_deg"],
        "speed_series": series, "speed_series_bone": main,
        "speed_ranking": [[bn, _r(v)] for bn, v in ranking],
        "bones": rows, "suggest": sug,
    }
    # ── 抖动闸门（2026-10-06 用户要求：让 analyze_motion 自己报警，别靠人眼）──────
    # 一半是"孤峰"：某帧速度 > 前后两帧均值 ×2 = 这一帧突然抽了一下（不需要修前数据）。
    # 另一半是"修前修后比值"：有 baseline_op 时才算（jitter / peak_speed 比值）。
    gate = {"pass": True, "failed_bones": [], "bones": {},
            "thresholds": {"spike_ratio": _GATE_SPIKE_RATIO,
                           "jitter_ratio": _GATE_JITTER_RATIO,
                           "peak_ratio": _GATE_PEAK_RATIO}}
    for bn in names:
        sp = np.asarray(speeds[bn], dtype=np.float64)
        row = {}
        if len(sp) >= 5:
            nb = (np.concatenate(([sp[0]], sp[:-1])) + np.concatenate((sp[1:], [sp[-1]]))) / 2.0
            sl = slice(1, len(sp) - 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(nb[sl] > 1e-9, sp[sl] / nb[sl], 0.0)
            k = int(np.argmax(ratio)) if ratio.size else 0
            if ratio.size and float(ratio[k]) > _GATE_SPIKE_RATIO:
                row["spike_frame"] = int(a + 1 + k)
                row["spike_ratio"] = _r(float(ratio[k]), 2)
        if row:
            gate["bones"][bn] = row
            gate["failed_bones"].append(bn)
    if gate["failed_bones"]:
        bn = gate["failed_bones"][0]
        g = gate["bones"][bn]
        warnings.append(
            f"抖动闸门：{bn} 在 {g['spike_frame']} 帧出现孤峰（该帧速度是前后两帧均值的 "
            f"{g['spike_ratio']}×，阈值 {_GATE_SPIKE_RATIO}×）——这一帧突然抽了一下。"
            f"别交报告，先调参或 revert（见 tools_io.md §8.1）")

    if baseline_tracks:
        # mask_tracks：该 op 之后叠在同骨同帧上层的修复。"当前 − 该 op 之前"两边都不能
        # 含它们，否则差值里混进别人的修复（四元数骨上还会被共轭搅乱）
        mask = list(mask or [])
        olds = [bool(t.mute) for t in baseline_tracks]
        mask_olds = [bool(t.mute) for t in mask]
        cur = smp
        try:
            if mask:
                for t in mask:
                    t.mute = True
                cur = P.sample_visible(scene, armature, names, frames)
            for t in baseline_tracks:
                t.mute = True
            base = P.sample_visible(scene, armature, names, frames)
        finally:
            for t, m in zip(baseline_tracks, olds):
                t.mute = m
            for t, m in zip(mask, mask_olds):
                t.mute = m
            scene.frame_set(int(scene.frame_current))   # 当前帧按恢复后的轨重算
        bsp = _speed(base["quat"][main], smooth)
        bev = _events(bsp, a, onset_frac=onset_frac, stop_frac=stop_frac,
                      onset_frame=onset_frame, stop_frame=stop_frame)
        bo = bev["onset"] if bev["onset"] is not None else 0
        bs = bev["stop"] if bev["stop"] is not None else T - 1
        res["vs_baseline"] = {
            "baseline_events": {"onset_frame": None if bev["onset"] is None else a + bev["onset"],
                                "peak_frame": a + bev["peak"],
                                "stop_frame": None if bev["stop"] is None else a + bev["stop"]},
            "bones": {bn: _baseline_report(base["quat"][bn], cur["quat"][bn], a,
                                           o=bo, p=bev["peak"], s=bs,
                                           axis_frames=int(axis_frames))
                      for bn in names},
        }
        # 闸门另一半：修后 vs 修前（同一骨、同一窗口）。修前 = 该 op 生效之前。
        ratio_fail = []
        for bn in names:
            bj = _jitter_mean(base["quat"][bn])
            cj = _jitter_mean(cur["quat"][bn])
            bp = float(np.max(_speed(base["quat"][bn], smooth)))
            cp = float(np.max(speeds[bn]))
            row = gate["bones"].setdefault(bn, {})
            row.update({"jitter_before": _r(bj), "jitter_after": _r(cj),
                        "peak_before": _r(bp), "peak_after": _r(cp)})
            if bj > 1e-6:
                row["jitter_ratio"] = _r(cj / bj, 2)
                row["jitter_fail"] = bool(cj > bj * _GATE_JITTER_RATIO)
            if bp > 1e-6:
                row["peak_ratio"] = _r(cp / bp, 2)
                row["peak_fail"] = bool(cp > bp * _GATE_PEAK_RATIO and cp > 3.0)
            if row.get("jitter_fail") or row.get("peak_fail"):
                ratio_fail.append(bn)
                if bn not in gate["failed_bones"]:
                    gate["failed_bones"].append(bn)
        if ratio_fail:
            bn = max(ratio_fail, key=lambda b: gate["bones"][b].get("jitter_ratio") or 0)
            g = gate["bones"][bn]
            warnings.append(
                f"抖动闸门不通过：{', '.join(ratio_fail)}。最差 {bn}：jitter "
                f"{g['jitter_before']}→{g['jitter_after']}°（×{g.get('jitter_ratio')}，"
                f"阈值 ×{_GATE_JITTER_RATIO}）、peak_speed {g['peak_before']}→"
                f"{g['peak_after']}°/帧（×{g.get('peak_ratio')}，阈值 ×{_GATE_PEAK_RATIO}）——"
                f"指标好看不等于画面好看，先调参或 revert（见 tools_io.md §8.1）")
        gate["pass"] = not gate["failed_bones"]
    res["gate"] = gate
    if _ignored(extra):
        warnings.append(f"忽略了未知参数：{_ignored(extra)}")
    return res, warnings, trunc


# ---------------------------------------------------------------------------
# bridge shells


def _names(ctx, args):
    if args.get("chain"):
        return P.chain_preset(args["chain"], ctx["armature"])
    return ctx["resolve_bones"](args.get("bones") or [])


def _norm_args(ctx, args):
    names = _names(ctx, args)
    if not names:
        raise RuntimeError("bones / chain 至少给一个（骨名、角色名或 arm.L 这类链预设）")
    out = {k: v for k, v in args.items() if k not in ("bones", "chain")}
    if out.get("main_bone"):
        out["main_bone"] = ctx["resolve_bones"]([out["main_bone"]])[0]
    return names, out


def _write_shell(fn):
    def _tool(ctx, **args):
        names, rest = _norm_args(ctx, args)
        P.reject_unknown_args(fn.__name__, fn, rest)
        op = fn(ctx["scene"], ctx["armature"], bones=names,
                data_dir=ctx["data_dir"], **rest)
        if not op.get("dry_run"):
            ctx["after_write"](op["frames"])
        return op
    return _tool


def _scope(ctx, args):
    return [(_names(ctx, args),
             (int(args["frame_range"][0]), int(args["frame_range"][1])))]


def _reapply_shell(fn):
    def _reapply(armature, base_action, *, params, frame_range, status, scene,
                 track_name):
        p = dict(params)
        p["frame_range"] = frame_range
        return fn(scene, armature, op_mode=status, data_dir=None,
                  track_name=track_name, record=False, **p)
    return _reapply


def _tool_analyze_motion(ctx, **args):
    """读工具：调 anticipation / follow_through / overshoot 之前第一步，修后复测也用它。"""
    names, rest = _norm_args(ctx, args)
    baseline_op = rest.pop("baseline_op", None)
    brief = bool(rest.pop("brief", False))      # 只要数字：去掉速度序列（~6KB）
    P.reject_unknown_args("analyze_motion", analyze_motion, rest,
                          extra_allowed=("baseline_op", "brief"))
    tracks = None
    if baseline_op:
        op = agent_ops.get_op(ctx["data_dir"], str(baseline_op))
        if op is None:
            raise RuntimeError(f"baseline_op {baseline_op} 不在 op 记录里：先 list_ops 看 op_id")
        track, _strip = agent_ops.find_op_strip(ctx["armature"], op)
        if track is None:
            raise RuntimeError(f"baseline_op {baseline_op} 的 strip 不在场景里（已 revert？）")
        tracks = [track]
        # 该 op 之后、写同骨同帧的上层修复：vs_baseline 两边都把它们静音
        newer = set(agent_ops.newer_same_bone_tracks(
            ctx["armature"], track.name, agent_ops.op_written_bones(ctx["armature"], op),
            [tuple(op["frames"])] if op.get("frames") else [None]))
        mask = [t for t in ctx["armature"].animation_data.nla_tracks if t.name in newer]
        if mask:
            rest["mask_tracks"] = mask
    res, warnings, trunc = analyze_motion(ctx["scene"], ctx["armature"],
                                          bones=names, baseline_tracks=tracks,
                                          **rest)
    who = {"chain": args["chain"]} if args.get("chain") else {"bones": names}
    for s in res["suggest"].values():
        s["args"] = {**who, **s["args"]}
    m = res["main"]
    summary = (f"主骨 {res['main_bone']}：onset {m['onset_frame']} → 峰值 "
               f"{m['peak_frame']}（{m['peak_speed']}°/帧）→ stop {m['stop_frame']}，"
               f"幅度 {m['amplitude_deg']}°，反向位移 {m['counter_move_deg']}°，"
               f"过冲 {m['overshoot_deg']}°")
    if baseline_op:
        res["vs_baseline"]["op_id"] = str(baseline_op)
        if rest.get("mask_tracks"):
            res["vs_baseline"]["muted_newer_tracks"] = [t.name for t in rest["mask_tracks"]]
        vb = res["vs_baseline"]["bones"][res["main_bone"]]
        summary += (f"；vs {baseline_op}：最大差 {vb['max_diff_deg']}°，接近轴峰 "
                    f"{vb['approach_peak_deg']}° @ {vb['approach_peak_frame']}，"
                    f"{vb['approach_sign_changes']} 次变号，改动帧 {vb['changed_frames']}")
    if brief:
        res.pop("speed_series", None)
        res.pop("speed_series_bone", None)
    return {"summary": summary, "data": res, "warnings": warnings,
            "truncated": bool(trunc),
            "hint": "照 data.suggest.<工具>.args 直接调写工具；修后按 retest 复测"}


_FLOAT = "float"
_INT = "int"

TOOLS = {
    "analyze_motion": _tool_analyze_motion,
    "anticipation": _write_shell(anticipation),
    "follow_through": _write_shell(follow_through),
    "overshoot": _write_shell(overshoot),
}

WRITE_SCOPES = {"anticipation": _scope, "follow_through": _scope,
                "overshoot": _scope}

_TAIL = [{"key": "strength", "kind": _FLOAT, "min": 0.0, "max": 2.0},
         {"key": "blend", "kind": _INT, "min": 0, "max": 40},
         {"key": "frame_range", "kind": "range"}]

TUNABLE = {
    "anticipation": [
        {"key": "amount", "kind": _FLOAT, "min": 0.0, "max": 0.5, "label": "反向幅度比例"},
        {"key": "lead", "kind": _INT, "min": 0, "max": 30, "label": "提前帧"},
        {"key": "delay", "kind": _INT, "min": 0, "max": 15, "label": "起点后挪帧"},
        *_TAIL],
    "follow_through": [
        {"key": "amount", "kind": _FLOAT, "min": 0.0, "max": 0.5, "label": "第一瓣比例"},
        {"key": "period", "kind": _FLOAT, "min": 2.0, "max": 40.0, "label": "周期(帧)"},
        {"key": "decay", "kind": _FLOAT, "min": 0.5, "max": 60.0, "label": "衰减(帧)"},
        {"key": "cycles", "kind": _FLOAT, "min": 0.5, "max": 6.0, "label": "周期数"},
        {"key": "propagate", "kind": _FLOAT, "min": 0.0, "max": 10.0, "label": "逐级延迟(帧)"},
        {"key": "gain_per_depth", "kind": _FLOAT, "min": 0.2, "max": 2.0, "label": "逐级增益"},
        *_TAIL],
    "overshoot": [
        {"key": "amount", "kind": _FLOAT, "min": 0.0, "max": 0.5, "label": "过冲比例"},
        {"key": "peak_after", "kind": _INT, "min": 0, "max": 10, "label": "达峰帧"},
        {"key": "settle", "kind": _INT, "min": 1, "max": 40, "label": "回位帧"},
        *_TAIL],
}

REAPPLY = {"anticipation": _reapply_shell(anticipation),
           "follow_through": _reapply_shell(follow_through),
           "overshoot": _reapply_shell(overshoot)}
