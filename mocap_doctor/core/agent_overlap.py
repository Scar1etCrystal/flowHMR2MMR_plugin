"""agent_overlap.py — overlap（骨链逐级错时）/ time_warp（时间重映射）两个写工具
+ chain_lag（链内逐级滞后）读工具。

overlap    子骨的局部旋转 = 自己在 t − lag_b 时刻的可见局部旋转，
           lag_b = min(depth_b · delay, max_delay)，depth 按语义链算（链根 0 不动）。
time_warp  所有骨共用一条时间映射 T(t)：局部旋转 = 原可见旋转在 T(t)；
           map 给分段点 [[t_new, t_old], ...]，或 speed+pivot 便捷模式。
chain_lag  每骨相对链内语义父骨的滞后帧数（角速度曲线互相关 + 抛物线插值求峰；
           默认 mix = 平滑角速度与其差分两条 r(k) 平均，差分让峰更尖、亚帧更准）。

全部在可见姿态上算（读越界允许，写只在 frame_range 内），小数帧用 slerp。
位置通道：骨的局部位置在读窗内确有变化（>0.01mm）时按同一时间映射一起重采样，
否则不写（FK 链骨一般没有位置动画）。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from . import agent_ops, agent_pose as P

_STILL_DEG = 1e-3        # 读窗内旋转变化低于此 = 静止
_STILL_M = 1e-5          # 位置变化低于此（米）= 不写位置通道
_MAX_FRAMES_SHIFT = 30.0  # delay / max_delay 上限（帧）
_LEG_FK = ("thigh_fk", "shin_fk", "foot_fk", "toe_fk")


# ---------------------------------------------------------------------------
# shared helpers


def _scene_or_ctx(scene):
    if scene is not None:
        return scene
    import bpy
    return bpy.context.scene


def _num(x: float, nd: int = 2):
    """JSON 友好：整数值给 int，否则四舍五入的 float。"""
    x = float(x)
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return round(x, nd)


def _members(armature: Any, bones: Sequence[str] | None,
             chain: str | None) -> tuple[list[str], dict]:
    """bones|chain → (骨架上存在的成员 根→梢, skipped {骨: 原因})。

    chain 优先（reapply 时 params 里两者都有，按 chain 重算结果相同）。"""
    skipped: dict = {}
    if chain:
        full = P.chain_preset(str(chain))           # 未知链名在这里抛
        present = [b for b in full if armature.pose.bones.get(b) is not None]
        for b in full:
            if b not in present:
                skipped[b] = "missing"
        if not present:
            raise RuntimeError(
                f"骨链 {chain!r} 的骨在这个骨架上一根都没有（{full}）；"
                "换一条链或直接传 bones 骨名列表")
        return present, skipped
    if not bones:
        raise RuntimeError(
            "需要 bones（骨名/角色名列表）或 chain（arm.L / arm.R / spine_head / "
            "spine / fingers.L / arm_nofingers.L ...）")
    return P.resolve_pose_bones(armature, list(bones)), skipped


def _animated(armature: Any, smp: Mapping[str, Any], bones: Sequence[str]) -> set:
    """有 fcurve（基底动作）或读窗内可见姿态确实在变的骨。"""
    keyed = P.animated_bones(agent_ops.base_action_of(armature))
    out = set()
    for b in bones:
        if b in keyed:
            out.add(b)
            continue
        q = smp["quat"][b]
        loc = smp["loc"][b]
        if len(q) and (float(P.qangle_deg(q[:1], q).max()) > _STILL_DEG
                       or float(np.abs(loc - loc[:1]).max()) > _STILL_M):
            out.add(b)
    return out


def _chain_parent(armature: Any, bone: str, members: Sequence[str]) -> str | None:
    mem = set(members)
    for anc in P.semantic_ancestors(armature, bone):
        if anc in mem:
            return anc
    return None


def _loc_moves(loc: np.ndarray) -> bool:
    return bool(len(loc)) and float(np.abs(loc - loc[:1]).max()) > _STILL_M


def _leg_warnings(armature: Any, bones: Sequence[str]) -> list[str]:
    out = []
    for side in ("L", "R"):
        legs = [b for b in bones if b.startswith(_LEG_FK) and b.endswith("." + side)]
        if not legs:
            continue
        pb = armature.pose.bones.get(f"thigh_parent.{side}")
        try:
            v = pb.get("IK_FK") if pb is not None else None
            v = None if v is None else float(v)
        except Exception:
            v = None
        if v is not None and v < 0.5:
            out.append(f"腿 {side} 是 IK 模式（thigh_parent.{side}[\"IK_FK\"]={v:g}），"
                       f"{legs} 的 FK 改动在视口里看不见；腿请改 foot_ik.{side}")
    return out


def _check_extra(opts: Mapping[str, Any], warnings: list) -> None:
    for k in sorted(opts):
        warnings.append(f"忽略未知参数 {k!r}")


def _deltas(armature, smp, desired_q, desired_l, strength, sl, members):
    """pose_deltas；desired 全空（所有骨都不用动）时给首骨写恒等 delta，
    让 op 照常存在（面板把 delay 拖到 0 不该报错）。"""
    if not desired_q and not desired_l:
        b0 = members[0]
        desired_q = {b0: smp["quat"][b0][sl]}
    return P.pose_deltas(armature, smp, desired_quat=desired_q,
                         desired_loc=desired_l, strength=strength,
                         index_slice=sl)


def _finish(tool, short, armature, smp, desired_q, desired_l, *, params, metrics,
            members, a, b, sl, strength, blend, op_mode, data_dir, track_name,
            dry_run, record):
    scalars, quats, info = _deltas(armature, smp, desired_q, desired_l,
                                   strength, sl, members)
    for bone, cell in info.items():
        if bone in metrics["bones"]:
            metrics["bones"][bone].update(cell)
    if dry_run:
        return {"dry_run": True, "tool": tool, "params": params,
                "metrics": metrics, "frames": [a, b]}
    # NLA auto-blend 的副作用由 agent_ops._write_strip 统一处理（写前快照、写后还原）
    track, strip = P.write_pose(armature, f"agent_{short}_{a}_{b}", a,
                                scalars, quats, blend=blend,
                                track_name=track_name)
    op = agent_ops._new_op(tool, params, (a, b), strip.name, op_mode,
                           metrics, track=track.name)
    return agent_ops._record(data_dir, op) if (data_dir and record) else op


# ---------------------------------------------------------------------------
# overlap


def overlap(scene, armature, *, frame_range, bones=None, chain=None,
            delay=1.0, max_delay=None, depths=None, strength=1.0, blend=None,
            op_mode="preview", data_dir=None, track_name=None,
            dry_run=False, record=True, **opts):
    """骨链错时：desired(b, t) = 可见局部旋转(b, t − lag_b)，
    lag_b = min(depth_b · delay, max_delay)；读 [a − ceil(max_delay) − 1, b]，
    只写 [a, b]（两端 blend 帧 taper）。"""
    scene = _scene_or_ctx(scene)
    a, b, frames = P.strip_window(frame_range)
    delay = float(delay)
    # 用户原话"子骨骼比父骨骼晚 1~3 帧……越往末端越晚"：delay 是**每级**的延迟，
    # 默认不封顶（max_delay=None → 最深一级×delay）。固定封顶 3 帧时 spine_head
    # 的 neck/head 会被钳成同一个延迟，和"越往末端越晚"相反。
    auto_cap = max_delay is None
    max_delay = None if auto_cap else float(max_delay)
    strength = float(strength)
    # blend 默认随最大延迟走（≥2×lag）：taper 区里有效时间 = t − w(t)·lag，其速率
    # ≈ 1 − 1.5·lag/blend——lag=4、blend=4 时为负，链末端会在窗口开头倒放。
    auto_blend = blend is None
    if not (0.0 <= delay <= _MAX_FRAMES_SHIFT):
        raise RuntimeError(f"delay={delay} 不合法：每级延迟帧数要在 0~{_MAX_FRAMES_SHIFT:g}，"
                           "常用 0.5~1.5（可小数）")
    if not auto_cap and not (0.0 <= max_delay <= _MAX_FRAMES_SHIFT):
        raise RuntimeError(f"max_delay={max_delay} 不合法：要在 0~{_MAX_FRAMES_SHIFT:g} 帧，"
                           "常用 2~3")
    members, skipped = _members(armature, bones, chain)
    warnings: list = []
    _check_extra(opts, warnings)
    dmap = {}
    if depths:
        if not isinstance(depths, Mapping):
            raise RuntimeError("depths 要是 {骨名: 深度} 字典（深度 0=链根，可小数），"
                               "或不传/null 用语义链深度")
        for k, v in depths.items():
            if k not in members and k not in skipped:
                raise RuntimeError(
                    f"depths 里的骨 {k!r} 不在这次的骨链里（{members}）；"
                    "只给链内的骨覆盖深度")
            try:
                dv = float(v)
            except (TypeError, ValueError):
                raise RuntimeError(f"depths[{k!r}]={v!r} 不是数字；深度要 ≥0 的数") from None
            if not (0.0 <= dv <= 64.0):
                raise RuntimeError(f"depths[{k!r}]={dv} 越界：深度要在 0~64")
            dmap[str(k)] = dv

    if auto_cap:
        dmax = max([dmap.get(bn, float(P.depth_in(armature, bn, members)))
                    for bn in members] or [0.0])
        max_delay = min(dmax * delay, _MAX_FRAMES_SHIFT)
    if auto_blend:
        blend = max(4, int(math.ceil(2.0 * max_delay)))
    blend = int(blend)
    lo = a - int(math.ceil(max_delay)) - 1
    read = list(range(lo, b + 1))
    smp = P.sample_visible(scene, armature, members, read)
    live = _animated(armature, smp, members)
    for bn in members:
        if bn not in live:
            skipped[bn] = "no_animation"
    work = [bn for bn in members if bn in live]
    if not work:
        raise RuntimeError(f"{members} 在 {lo}-{b} 帧全都没有动画，overlap 无事可做；"
                           "换一段有动作的帧或换骨链")

    sl = slice(a - lo, a - lo + len(frames))
    t_new = np.asarray(frames, dtype=np.float64)
    bones_m: dict = {}
    desired_q: dict = {}
    desired_l: dict = {}
    roots = []
    for bn in work:
        d = dmap[bn] if bn in dmap else float(P.depth_in(armature, bn, work))
        lag = min(d * delay, max_delay)
        par = _chain_parent(armature, bn, work)
        if par is None:
            roots.append(bn)
        bones_m[bn] = {"depth": _num(d), "lag_frames": _num(lag, 3),
                       "parent": par, "rot_mode": smp["mode"][bn],
                       "rot_change_max_deg": 0.0, "rot_change_mean_deg": 0.0}
        if lag <= 1e-9:
            continue
        times = t_new - lag
        desired_q[bn] = P.resample_quats(smp["quat"][bn], lo, times)
        loc = smp["loc"][bn]
        if _loc_moves(loc):
            desired_l[bn] = P.resample_vec(loc, lo, times)
    if not desired_q:
        warnings.append("所有骨的延迟都是 0（delay=0 或只有链根），本 op 不改变姿态")
    if b - a + 1 <= 2 * blend + 2:
        warnings.append(f"窗口只有 {b - a + 1} 帧，两端各 {blend} 帧 taper 后几乎没有生效区；"
                        "把 frame_range 放宽到动作前后各多 blend 帧")
    max_lag_used = max((min((dmap[bn] if bn in dmap else float(P.depth_in(armature, bn, work)))
                            * delay, max_delay) for bn in work), default=0.0)
    if max_lag_used > 0 and blend < 1.5 * max_lag_used:
        warnings.append(f"blend={blend} < 1.5×最大延迟 {max_lag_used:g} 帧：窗口开头链末端会"
                        f"倒放、结尾会快进；建议 blend ≥ {int(math.ceil(2 * max_lag_used))}"
                        "（或不传 blend 用自动值）")
    elif max_lag_used > 0 and blend < 2 * max_lag_used:
        warnings.append(f"blend={blend} < 2×最大延迟：窗口两端链末端会明显变慢/变快；"
                        f"建议 blend ≥ {int(math.ceil(2 * max_lag_used))}")
    warnings += _leg_warnings(armature, work)

    params = {"bones": list(members), "chain": chain, "frame_range": [a, b],
              "delay": delay, "max_delay": None if auto_cap else max_delay,
              "depths": dict(dmap) or None, "strength": strength,
              "blend": None if auto_blend else blend}
    metrics = {"bones": bones_m, "roots": roots, "skipped_bones": skipped,
               "delay": delay, "max_delay": max_delay,
               "max_delay_auto": auto_cap, "blend": blend, "blend_auto": auto_blend,
               "max_lag_frames": _num(max((c["lag_frames"] for c in bones_m.values()),
                                          default=0), 3),
               "read_range": [lo, b], "inner_frames": [a + blend, b - blend],
               "warnings": warnings}
    return _finish("overlap", "overlap", armature, smp, desired_q, desired_l,
                   params=params, metrics=metrics, members=work, a=a, b=b,
                   sl=sl, strength=strength, blend=blend, op_mode=op_mode,
                   data_dir=data_dir, track_name=track_name, dry_run=dry_run,
                   record=record)


# ---------------------------------------------------------------------------
# time_warp


def _knots(a: int, b: int, tmap, speed, pivot, ease: str):
    """map / speed+pivot → 单调结点 (xs=t_new, ys=t_old, slopes|None, info)，含默认端点。

    speed+pivot：[a,pivot] 前慢后快、以 speed 倍速到达 pivot，a / pivot / b 三点恒等。
      linear  结点 (a,a)、(a+D·s/(1+s), a+D/(1+s))、(pivot,pivot)、(b,b)：
              两段匀速 1/s 与 s（D=pivot−a），pivot 之后原速。
      smooth  结点 (a,a)、(pivot,pivot)、(b,b)，斜率 1 / s / 1 的三次 Hermite：
              窗口两端与原时间线速度连续、到达 pivot 恰为 s 倍速；pivot 之后
              从 s 平滑回落到 1（随动段先略超前再收回到 b）。"""
    if tmap is not None and (speed is not None or pivot is not None):
        raise RuntimeError("map 和 speed/pivot 二选一：要自定义分段就只给 map，"
                           "要\"某段加速到达 pivot\"就只给 speed+pivot")
    if tmap is None:
        if speed is None or pivot is None:
            raise RuntimeError(
                "time_warp 需要 map=[[t_new,t_old],...]，或 speed+pivot"
                "（如 speed=1.5, pivot=冲击帧：[a,pivot] 段前慢后快、以 1.5 倍速到达 pivot）")
        s = float(speed)
        pv = float(pivot)
        if not (0.1 <= s <= 10.0):
            raise RuntimeError(f"speed={s} 不合法：要在 0.1~10（1.5=到达 pivot 时 1.5 倍速，"
                               "<1=放慢到达）；smooth 下 >2.8 会被单调限幅")
        if not (a < pv <= b):
            raise RuntimeError(f"pivot={pivot} 越界：要在 ({a}, {b}] 内（窗口起点之后），"
                               "一般给动作的冲击/到位帧")
        D = pv - a
        if ease == "linear":
            xs = [float(a), a + D * s / (1.0 + s), pv]
            ys = [float(a), a + D / (1.0 + s), pv]
            slopes = None
        else:
            xs = [float(a), pv]
            ys = [float(a), pv]
            slopes = [1.0, s]
        if pv < b:
            xs.append(float(b))
            ys.append(float(b))
            if slopes is not None:
                slopes.append(1.0)
        info = {"mode": "speed_pivot", "speed": s, "pivot": _num(pv)}
        if ease == "linear":
            info["segment_speeds"] = [round(1.0 / s, 4), round(s, 4)] + ([1.0] if pv < b else [])
        return np.asarray(xs), np.asarray(ys), slopes, info
    try:
        pts = [(float(p[0]), float(p[1])) for p in tmap]
    except Exception:
        raise RuntimeError("map 要是 [[t_new, t_old], ...] 数对列表，如 "
                           f"[[{a + 10}, {a + 5}], [{b - 10}, {b - 10}]]") from None
    if not pts:
        raise RuntimeError("map 是空的：至少给一个 [t_new, t_old] 点（端点默认 "
                           f"[{a},{a}] 和 [{b},{b}]）")
    for i, (tn, _to) in enumerate(pts):
        if not (a <= tn <= b):
            raise RuntimeError(f"map 第 {i} 点 t_new={tn:g} 越界：t_new 必须在 frame_range "
                               f"[{a}, {b}] 内（只能写窗口内的帧）")
    if pts[0][0] > a:
        pts.insert(0, (float(a), float(a)))
    if pts[-1][0] < b:
        pts.append((float(b), float(b)))
    xs = np.asarray([p[0] for p in pts])
    ys = np.asarray([p[1] for p in pts])
    for i in range(1, len(pts)):
        if xs[i] <= xs[i - 1]:
            raise RuntimeError(
                f"map 非单调：t_new 必须严格递增，第 {i} 点 t_new={xs[i]:g} ≤ 前一点 "
                f"{xs[i - 1]:g}（含默认端点 [{a},{a}]/[{b},{b}]）；按 t_new 从小到大排、别重复")
        if ys[i] < ys[i - 1]:
            raise RuntimeError(
                f"map 非单调：t_old 不能倒退，第 {i} 点 t_old={ys[i]:g} < 前一点 "
                f"{ys[i - 1]:g}（时间倒流）；要定格就让 t_old 相等，要倒放请别用 time_warp")
    return xs, ys, None, {"mode": "map"}


def _hermite_slopes(xs: np.ndarray, ys: np.ndarray, slopes=None) -> np.ndarray:
    """单调三次 Hermite 的结点斜率。

    slopes=None：PCHIP（加权调和平均）；恒等端点（t_old==t_new）取 1，
    从未改动的时间线速度连续地过渡进来。给定 slopes 时用它。
    最后统一做 Fritsch–Carlson 限幅（α²+β²≤9）保证单调。"""
    n = len(xs)
    h = np.diff(xs)
    d = np.diff(ys) / h
    if slopes is not None:
        m = np.asarray(slopes, dtype=np.float64).copy()
    else:
        m = np.zeros(n)
        for k in range(1, n - 1):
            if d[k - 1] * d[k] <= 0.0:
                m[k] = 0.0
            else:
                w1 = 2.0 * h[k] + h[k - 1]
                w2 = h[k] + 2.0 * h[k - 1]
                m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
        m[0] = 1.0 if abs(ys[0] - xs[0]) < 1e-9 else d[0]
        m[-1] = 1.0 if abs(ys[-1] - xs[-1]) < 1e-9 else d[-1]
    for k in range(n - 1):
        if d[k] == 0.0:
            m[k] = 0.0
            m[k + 1] = 0.0
            continue
        al, be = m[k] / d[k], m[k + 1] / d[k]
        if al < 0.0:
            m[k], al = 0.0, 0.0
        if be < 0.0:
            m[k + 1], be = 0.0, 0.0
        s2 = al * al + be * be
        if s2 > 9.0:
            tau = 3.0 / math.sqrt(s2)
            m[k] = tau * al * d[k]
            m[k + 1] = tau * be * d[k]
    return m


def eval_time_map(xs: np.ndarray, ys: np.ndarray, t, ease: str = "smooth",
                  slopes=None) -> np.ndarray:
    """T(t)：linear = 分段线性；smooth = 过同样结点的单调 C1 三次（速度连续）。"""
    t = np.asarray(t, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if ease == "linear" or len(xs) < 2:
        return np.interp(t, xs, ys)
    m = _hermite_slopes(xs, ys, slopes)
    idx = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, len(xs) - 2)
    x0 = xs[idx]
    hh = xs[idx + 1] - x0
    u = np.clip((t - x0) / hh, 0.0, 1.0)
    u2, u3 = u * u, u * u * u
    return ((2 * u3 - 3 * u2 + 1) * ys[idx] + (u3 - 2 * u2 + u) * hh * m[idx]
            + (-2 * u3 + 3 * u2) * ys[idx + 1] + (u3 - u2) * hh * m[idx + 1])


def time_warp(scene, armature, *, frame_range, bones=None, chain=None,
              map=None, speed=None, pivot=None, ease="smooth",  # noqa: A002
              strength=1.0, blend=4, op_mode="preview", data_dir=None,
              track_name=None, dry_run=False, record=True, **opts):
    """时间重映射：所有骨 desired(b, t) = 可见局部旋转(b, T(t))。
    T 过 map 结点（端点默认恒等），ease=smooth 时速度连续。"""
    scene = _scene_or_ctx(scene)
    a, b, frames = P.strip_window(frame_range)
    strength = float(strength)
    blend = int(blend)
    ease = str(ease or "smooth")
    if ease not in ("smooth", "linear"):
        raise RuntimeError(f"ease={ease!r} 不合法：只能是 \"smooth\"（速度连续）或 \"linear\"（分段匀速）")
    xs, ys, slopes, kinfo = _knots(a, b, map, speed, pivot, ease)
    lo_ok = min(a, int(scene.frame_start))
    hi_ok = max(b, int(scene.frame_end))
    if ys.min() < lo_ok or ys.max() > hi_ok:
        raise RuntimeError(f"map 的 t_old 越界：{ys.min():g}~{ys.max():g} 超出可读帧范围 "
                           f"[{lo_ok}, {hi_ok}]（场景帧范围∪窗口）；t_old 要取存在的帧")
    members, skipped = _members(armature, bones, chain)
    warnings: list = []
    _check_extra(opts, warnings)

    t_new = np.asarray(frames, dtype=np.float64)
    T = eval_time_map(xs, ys, t_new, ease, slopes)
    lo = min(a, int(math.floor(float(T.min()))))
    hi = max(b, int(math.ceil(float(T.max()))))
    smp = P.sample_visible(scene, armature, members, list(range(lo, hi + 1)))
    live = _animated(armature, smp, members)
    for bn in members:
        if bn not in live:
            skipped[bn] = "no_animation"
    work = [bn for bn in members if bn in live]
    if not work:
        raise RuntimeError(f"{members} 在 {lo}-{hi} 帧全都没有动画，time_warp 无事可做")
    sl = slice(a - lo, a - lo + len(frames))
    shift = T - t_new
    bones_m: dict = {}
    desired_q: dict = {}
    desired_l: dict = {}
    identity = float(np.abs(shift).max()) < 1e-6
    for bn in work:
        bones_m[bn] = {"rot_mode": smp["mode"][bn], "rot_change_max_deg": 0.0,
                       "rot_change_mean_deg": 0.0}
        if identity:
            continue
        desired_q[bn] = P.resample_quats(smp["quat"][bn], lo, T)
        loc = smp["loc"][bn]
        if _loc_moves(loc):
            desired_l[bn] = P.resample_vec(loc, lo, T)
    if identity:
        warnings.append("时间映射是恒等的（map 全在对角线上或 speed=1），本 op 不改变姿态")
    sp = np.diff(T)
    if len(sp) and float(sp.min()) < -1e-9:
        raise RuntimeError("内部错误：时间映射非单调，请报告 map/speed/pivot 参数")
    inner_lo, inner_hi = a + blend, b - blend
    edge = np.r_[np.abs(shift[:blend]), np.abs(shift[len(shift) - blend:])] if blend else []
    if len(edge) and float(np.max(edge)) > 1.0:
        warnings.append(f"两端 taper 区（各 {blend} 帧）里时间偏移已达 {float(np.max(edge)):.1f} 帧，"
                        "那几帧只部分生效、会有一点顿挫；把 frame_range 两端各放宽几帧，"
                        "或让 map 端点附近保持恒等")
    warnings += _leg_warnings(armature, work)

    step = max(1, len(frames) // 12)
    tm_idx = sorted(set(list(range(0, len(frames), step)) + [len(frames) - 1]))
    metrics = {
        "bones": bones_m, "skipped_bones": skipped, "ease": ease, **kinfo,
        "map_used": [[_num(x, 3), _num(y, 3)] for x, y in zip(xs, ys)],
        **({"knot_speeds": [round(float(v), 3) for v in _hermite_slopes(xs, ys, slopes)]}
           if ease == "smooth" and len(xs) >= 2 else {}),
        "time_map": [[int(frames[i]), round(float(T[i]), 2)] for i in tm_idx],
        "shift_max_frames": round(float(np.abs(shift).max()), 3),
        "speed_min": round(float(sp.min()), 3) if len(sp) else 1.0,
        "speed_max": round(float(sp.max()), 3) if len(sp) else 1.0,
        "read_range": [lo, hi], "inner_frames": [inner_lo, inner_hi],
        "warnings": warnings,
    }
    if kinfo.get("mode") == "speed_pivot":
        pv = int(round(float(pivot)))
        i = pv - a
        if 1 <= i < len(T):
            metrics["speed_into_pivot"] = round(float(T[i] - T[i - 1]), 3)
            metrics["pivot_time_old"] = round(float(T[i]), 3)
    params = {"bones": list(members), "chain": chain, "frame_range": [a, b],
              "map": [list(p) for p in map] if map is not None else None,
              "speed": float(speed) if speed is not None else None,
              "pivot": _num(pivot) if pivot is not None else None,
              "ease": ease, "strength": strength, "blend": blend}
    return _finish("time_warp", "timewarp", armature, smp, desired_q, desired_l,
                   params=params, metrics=metrics, members=work, a=a, b=b,
                   sl=sl, strength=strength, blend=blend, op_mode=op_mode,
                   data_dir=data_dir, track_name=track_name, dry_run=dry_run,
                   record=record)


# ---------------------------------------------------------------------------
# chain_lag (read-only)


_LAG_RELIABLE = 0.4      # corr 低于此（或峰在 ±max_lag 边界）= 不可信
_LAG_SIGNALS = ("mix", "speed", "dspeed")


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if den < 1e-12:
        return None
    return float((x * y).sum()) / den


def lag_signals(q: np.ndarray, smooth: int = 3, signal: str = "mix") -> list:
    """局部旋转 → 互相关用的信号：平滑角速度（度/帧，对称滑窗=零相位）及其
    一阶差分（prewhitening：峰更尖，亚帧更准）。mix = 两者各算一条 r(k) 再平均。"""
    s = P.smooth(P.angular_speed_deg(q), smooth)
    if signal == "speed":
        return [s]
    if signal == "dspeed":
        return [np.gradient(s)]
    return [s, np.gradient(s)]


def xcorr_lag(par_sigs: Sequence[np.ndarray], child_sigs: Sequence[np.ndarray],
              i0: int, i1: int, max_lag: int) -> dict:
    """父骨 [i0,i1] 参考段固定，子骨段平移 k∈[−max_lag, max_lag] 求 Pearson r(k)
    （多条信号的 r(k) 取平均）；峰值 k* 用抛物线插值细化。正值 = 子骨比父骨晚。"""
    ks = np.arange(-max_lag, max_lag + 1)
    curves = []
    for x, y in zip(par_sigs, child_sigs):
        ref = x[i0:i1 + 1]
        r = [_pearson(ref, y[i0 + k:i1 + 1 + k]) for k in ks]
        if all(v is not None for v in r):
            curves.append(r)
    if not curves:
        return {"lag_frames": None, "corr": None, "at_bound": False,
                "reliable": False, "note": "父或子骨在窗口内几乎静止，无法测滞后"}
    r = np.mean(np.asarray(curves, dtype=np.float64), axis=0)
    i = int(np.argmax(r))
    off = 0.0
    peak = float(r[i])
    if 0 < i < len(r) - 1:
        den = r[i - 1] - 2.0 * r[i] + r[i + 1]
        if abs(den) > 1e-12:
            off = float(np.clip(0.5 * (r[i - 1] - r[i + 1]) / den, -0.5, 0.5))
            peak = float(r[i] - 0.25 * (r[i - 1] - r[i + 1]) * off)
    at_bound = i == 0 or i == len(r) - 1
    return {"lag_frames": round(float(ks[i]) + off, 3), "corr": round(peak, 3),
            "at_bound": bool(at_bound),
            "reliable": bool(peak >= _LAG_RELIABLE and not at_bound)}


def chain_lag(scene, armature, *, frame_range, bones=None, chain=None,
              max_lag=6, smooth=3, signal="mix", **opts):
    """每骨相对链内语义父骨的滞后（帧，小数）+ 相关系数。

    参考段 = [a+max_lag, b−max_lag]，子骨段在 ±max_lag 内滑动——比较只用
    frame_range 内的帧（角速度差分/平滑多读两端各 3 帧）。overlap 复测时
    frame_range 用 overlap 的 metrics.inner_frames（避开两端 taper 区），
    前后用同一范围测，比较每级 lag_frames 的增量。"""
    scene = _scene_or_ctx(scene)
    a, b, _frames = P.strip_window(frame_range)
    max_lag = int(max_lag)
    if not (1 <= max_lag <= 30):
        raise RuntimeError(f"max_lag={max_lag} 不合法：要在 1~30 帧（默认 6）")
    smooth = max(1, int(smooth))
    signal = str(signal or "mix")
    if signal not in _LAG_SIGNALS:
        raise RuntimeError(f"signal={signal!r} 不合法：mix（默认）/ speed / dspeed")
    n_ref = (b - a + 1) - 2 * max_lag
    if n_ref < 8:
        raise RuntimeError(f"frame_range 只有 {b - a + 1} 帧，max_lag={max_lag} 时参考段只剩 "
                           f"{n_ref} 帧；至少给 2·max_lag+8 帧，或调小 max_lag")
    members, skipped = _members(armature, bones, chain)
    pad = 3
    lo, hi = a - pad, b + pad
    smp = P.sample_visible(scene, armature, members, list(range(lo, hi + 1)))
    live = _animated(armature, smp, members)
    for bn in members:
        if bn not in live:
            skipped[bn] = "no_animation"
    work = [bn for bn in members if bn in live]
    sigs = {bn: lag_signals(smp["quat"][bn], smooth, signal) for bn in work}
    i0, i1 = a + max_lag - lo, b - max_lag - lo
    out: dict = {}
    levels = []
    for bn in work:
        par = _chain_parent(armature, bn, work)
        sp = P.smooth(P.angular_speed_deg(smp["quat"][bn]), smooth)[a - lo:b - lo + 1]
        cell = {"parent": par, "depth": int(P.depth_in(armature, bn, work)),
                "peak_speed_deg": round(float(sp.max()), 2),
                "mean_speed_deg": round(float(sp.mean()), 2)}
        if par is None:
            cell.update({"lag_frames": None, "corr": None})
        else:
            cell.update(xcorr_lag(sigs[par], sigs[bn], i0, i1, max_lag))
            levels.append({"parent": par, "child": bn,
                           "lag_frames": cell["lag_frames"], "corr": cell["corr"],
                           "reliable": cell["reliable"]})
        out[bn] = cell
    res = {"frame_range": [a, b], "ref_range": [a + max_lag, b - max_lag],
           "max_lag": max_lag, "smooth": smooth, "signal": signal,
           "bones": out, "levels": levels, "skipped_bones": skipped,
           "note": "lag_frames>0 = 子骨比链内父骨晚；reliable=false（corr<0.4 或峰顶在 "
                   "±max_lag 边界）的数别信；验 overlap：前后用同一 frame_range"
                   "（取 overlap 的 metrics.inner_frames）各测一次，每级增量≈该级 lag 之差"}
    if opts:
        res["warnings"] = [f"忽略未知参数 {k!r}" for k in sorted(opts)]
    return res


# ---------------------------------------------------------------------------
# bridge shells


def _need_rig(ctx):
    arm = ctx.get("armature")
    if arm is None:
        raise RuntimeError("没有识别到 RIG 骨架（settings.mmr_rig 为空，场景里也没有 RIG-*）")
    return arm


def _names(ctx, args):            # bones（角色名或骨名）或 chain 预设
    if args.get("chain"):
        return P.chain_preset(args["chain"], ctx["armature"])
    return ctx["resolve_bones"](args.get("bones") or [])


def _split_target(ctx, args):
    """桥层：chain 原样交给求解器（要算 skipped_bones）；bones 把角色名映射成骨名。"""
    args = dict(args)
    chain = args.pop("chain", None)
    bones = args.pop("bones", None)
    if not chain:
        bones = ctx["resolve_bones"](list(bones or []))
    return chain, bones, args


def _tool_overlap(ctx, **args):
    arm = _need_rig(ctx)
    chain, bones, args = _split_target(ctx, args)
    if args.get("depths"):
        args["depths"] = {ctx["resolve_bones"]([k])[0]: v
                          for k, v in dict(args["depths"]).items()}
    P.reject_unknown_args("overlap", overlap, args)
    op = overlap(ctx["scene"], arm, bones=bones, chain=chain,
                 data_dir=ctx["data_dir"], **args)
    if not op.get("dry_run"):
        ctx["after_write"](op["frames"])
    return op


def _tool_time_warp(ctx, **args):
    arm = _need_rig(ctx)
    chain, bones, args = _split_target(ctx, args)
    P.reject_unknown_args("time_warp", time_warp, args)
    op = time_warp(ctx["scene"], arm, bones=bones, chain=chain,
                   data_dir=ctx["data_dir"], **args)
    if not op.get("dry_run"):
        ctx["after_write"](op["frames"])
    return op


def _fmt_level(lv) -> str:
    lag = "?" if lv["lag_frames"] is None else f"{lv['lag_frames']:+.2f}f"
    r = "-" if lv["corr"] is None else f"{lv['corr']:.2f}"
    return f"{lv['child']} {lag}(r{r}{'' if lv['reliable'] else '?'})"


def _tool_chain_lag(ctx, **args):
    arm = _need_rig(ctx)
    chain, bones, args = _split_target(ctx, args)
    P.reject_unknown_args("chain_lag", chain_lag, args)
    res = chain_lag(ctx["scene"], arm, bones=bones, chain=chain, **args)
    fr = res["frame_range"]
    warnings = [f"{lv['parent']}→{lv['child']} 不可信（corr<{_LAG_RELIABLE} 或峰在 ±max_lag 边界）"
                for lv in res["levels"] if not lv["reliable"]]
    return {"summary": f"chain_lag {fr[0]}-{fr[1]}: "
                       + " ".join(_fmt_level(lv) for lv in res["levels"]),
            "data": res, "warnings": warnings, "truncated": False,
            "hint": "验 overlap：frame_range 用 overlap 的 metrics.inner_frames，前后各测一次，"
                    "比较 reliable 级的 lag_frames 增量（≈该级 lag 之差）；带 ? 的级别别信"}


def _scope(ctx, args):            # 并发租约：会写哪些骨 × 哪些帧
    return [(_names(ctx, args),
             (int(args["frame_range"][0]), int(args["frame_range"][1])))]


def _reapply_overlap(armature, base_action, *, params, frame_range, status,
                     scene, track_name):
    p = dict(params)
    p["frame_range"] = frame_range
    return overlap(scene, armature, op_mode=status, data_dir=None,
                   track_name=track_name, record=False, **p)


def _reapply_time_warp(armature, base_action, *, params, frame_range, status,
                       scene, track_name):
    p = dict(params)
    p["frame_range"] = frame_range
    return time_warp(scene, armature, op_mode=status, data_dir=None,
                     track_name=track_name, record=False, **p)


TOOLS = {"overlap": _tool_overlap, "time_warp": _tool_time_warp,
         "chain_lag": _tool_chain_lag}
WRITE_SCOPES = {"overlap": _scope, "time_warp": _scope}
TUNABLE = {
    "overlap": [
        {"key": "delay", "kind": "float", "min": 0.0, "max": 6.0, "label": "每级延迟(帧)"},
        {"key": "max_delay", "kind": "float", "min": 0.0, "max": 12.0, "label": "最大延迟(帧)"},
        {"key": "strength", "kind": "float", "min": 0.0, "max": 2.0},
        {"key": "blend", "kind": "int", "min": 0, "max": 40},
        {"key": "frame_range", "kind": "range"},
    ],
    "time_warp": [
        {"key": "speed", "kind": "float", "min": 0.1, "max": 10.0,
         "when": {"speed": True}, "label": "到达速度(倍)"},
        {"key": "pivot", "kind": "int", "min": 0, "max": 100000,
         "when": {"speed": True}, "label": "到达帧"},
        {"key": "ease", "kind": "choice", "options": ["smooth", "linear"]},
        {"key": "strength", "kind": "float", "min": 0.0, "max": 2.0},
        {"key": "blend", "kind": "int", "min": 0, "max": 40},
        {"key": "frame_range", "kind": "range"},
    ],
}
REAPPLY = {"overlap": _reapply_overlap, "time_warp": _reapply_time_warp}
