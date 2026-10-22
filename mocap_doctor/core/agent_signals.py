"""Named signal registry over baked world-pose samples (pure numpy).

Every signal is a ``(T,)`` float array aligned to ``bake["frames"]``.  Names are
what an LLM asks for - ``foot.L.yaw_rate``, ``pelvis.jerk`` - so each entry
carries a unit and a one-line description that get injected into the prompt.

Interval masks (contact / air / jitter) come from the annotation channels the
wizard already maintains: FOOT_*_EFFECTIVE (planted), AIR (both feet off the
ground), HAND_*_MANUAL (marked bad hand segments).  They arrive as 0/1 masks
and are passed in by the caller, keeping this module Blender-free.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# point ids expected in bake["point_pos"] for each foot side
FOOT_POINTS = {
    "L": {"heel": "foot.L.heel", "ball": "foot.L.ball", "toe": "foot.L.toe"},
    "R": {"heel": "foot.R.heel", "ball": "foot.R.ball", "toe": "foot.R.toe"},
}
PIVOT_SPEED_MIN_M = 0.008   # under this a contact point counts as stationary
HF_WINDOW = 9               # sliding window (frames) for high-frequency loss


def _diff(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    out = np.zeros_like(series)
    if len(series) > 1:
        out[1:] = np.diff(series, axis=0)
    return out


def _masked(stats_values: np.ndarray) -> np.ndarray:
    return np.asarray(stats_values, dtype=np.float64)


def _yaw_deg(pos_toe: np.ndarray, pos_heel: np.ndarray) -> np.ndarray:
    d = pos_toe - pos_heel
    yaw = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return np.unwrap(np.radians(yaw)) * 180.0 / np.pi if len(yaw) else yaw


def _quat_geo_deg(q: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(q[1:] * q[:-1], axis=1))
    out = np.zeros(len(q))
    out[1:] = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
    return out


def _hf_loss(raw_geo: np.ndarray, cur_geo: np.ndarray, window: int) -> np.ndarray:
    """Energy the filtering removed: mean |raw - cur| residual over a window."""
    resid = np.abs(raw_geo - cur_geo)
    out = np.zeros(len(resid))
    half = window // 2
    for i in range(len(resid)):
        lo, hi = max(0, i - half), min(len(resid), i + half + 1)
        out[i] = float(resid[lo:hi].mean())
    return out


def _mask(frames: np.ndarray, ranges) -> np.ndarray:
    """ranges 项支持 [start,end] 元组或 {"start","end"} dict 两种格式。"""
    m = np.zeros(len(frames))
    for item in ranges or ():
        if isinstance(item, dict):
            lo, hi = int(item["start"]), int(item["end"])
        else:
            lo, hi = int(item[0]), int(item[1])
        m[(frames >= lo) & (frames <= hi)] = 1.0
    return m


# ---------------------------------------------------------------------------
# registry

def _reg(signals, name, values, unit, desc, groups):
    signals[name] = {
        "values": _masked(values),
        "unit": unit,
        "desc": desc,
        "groups": tuple(groups),
    }


def compute_signals(
    bake: Mapping[str, Any],
    *,
    floor_z: float = 0.0,
    intervals: Mapping[str, Sequence[Sequence[int]]] | None = None,
    raw_quat: Mapping[str, np.ndarray] | None = None,
) -> dict[str, dict]:
    """Build every registered signal from a bake dict.

    ``bake`` comes from ``agent_bake.bake_bone_samples`` / ``load_bake``.
    ``intervals`` maps kind → list of [start, end] in bake frame numbering.
    ``raw_quat`` supplies the pre-filter reference rotations for hf_loss.
    """

    frames = np.asarray(bake["frames"])
    pos = bake["pos"]
    point = bake["point_pos"]
    quat = bake["quat"]
    intervals = intervals or {}
    signals: dict[str, dict] = {}

    for side in ("L", "R"):
        p = FOOT_POINTS[side]
        if not all(name in point for name in p.values()):
            continue
        heel = point[p["heel"]]
        ball = point[p["ball"]]
        toe = point[p["toe"]]
        pre = f"foot.{side}"

        _reg(signals, f"{pre}.heel_h", heel[:, 2], "m",
             "左脚踝头高度" if side == "L" else "右脚踝头高度", ("ground",))
        _reg(signals, f"{pre}.ball_h", ball[:, 2], "m",
             "前掌(踝尾)高度", ("ground", "pivot"))
        _reg(signals, f"{pre}.toe_h", toe[:, 2], "m",
             "脚尖(足骨尾)高度", ("ground", "pivot"))

        sole = np.minimum(np.minimum(heel[:, 2], ball[:, 2]), toe[:, 2])
        _reg(signals, f"{pre}.sole_h", sole, "m", "脚底最低点高度", ("ground",))
        _reg(signals, f"{pre}.pen", floor_z - sole, "m",
             "穿地深度（正=穿进地面）", ("ground",))
        _reg(signals, f"{pre}.clearance", sole - floor_z, "m",
             "离地间隙（正=悬空）", ("ground",))

        mid = (heel + toe) * 0.5
        speed_xy = np.linalg.norm(_diff(mid[:, :2]), axis=1)
        _reg(signals, f"{pre}.speed_xy", speed_xy, "m/f", "脚水平速度", ("ground", "pivot"))
        _reg(signals, f"{pre}.speed_z", np.abs(_diff(mid[:, 2])), "m/f", "脚垂直速度", ("ground",))

        yaw = _yaw_deg(toe, heel)
        _reg(signals, f"{pre}.yaw", yaw, "deg", "脚朝向（踝→脚尖水平角）", ("pivot",))
        _reg(signals, f"{pre}.yaw_rate", np.abs(_diff(yaw)), "deg/f",
             "脚朝向变化率（碾转信号）", ("pivot",))

        # pivot: which contact point is the still one
        v_heel = np.linalg.norm(_diff(heel[:, :2]), axis=1)
        v_ball = np.linalg.norm(_diff(ball[:, :2]), axis=1)
        v_toe = np.linalg.norm(_diff(toe[:, :2]), axis=1)
        pivot_idx = np.argmin(np.stack([v_heel, v_ball, v_toe]), axis=0).astype(float)
        still = np.minimum(np.minimum(v_heel, v_ball), v_toe) < PIVOT_SPEED_MIN_M
        _reg(signals, f"{pre}.pivot_idx", pivot_idx, "",
             "支点：0=踝头/1=前掌/2=脚尖（该帧最静止的点）", ("pivot",))
        _reg(signals, f"{pre}.pivot_still", still.astype(float), "0/1",
             "支点是否近乎不动（踩出碾转前提）", ("pivot",))
        _reg(signals, f"{pre}.heel_vxy", v_heel, "m/f", "踝头水平速度", ("pivot",))
        _reg(signals, f"{pre}.toe_vxy", v_toe, "m/f", "脚尖水平速度", ("pivot",))

    # --- fingers: curl (三节摆动角之和), dir → up_err/azimuth --------------
    basis = bake.get("basis") or {}
    for side in ("L", "R"):
        for finger in ("index", "middle", "ring", "pinky", "thumb"):
            roles = [f"finger_{side.lower()}_{finger}{i}" for i in (1, 2, 3)]
            if not all(r in basis for r in roles):
                continue
            pre = f"finger.{side}.{finger}"
            from .agent_fx import swing_twist_deg
            curls = [
                np.abs(swing_twist_deg(basis[r])["swing_deg"]) for r in roles
            ]
            curl = np.sum(curls, axis=0)
            _reg(signals, f"{pre}.curl", curl, "deg",
                 f"{'左' if side == 'L' else '右'}手{finger}三节摆动角之和"
                 f"（伸直≈0）", ("finger",))
            for i, r in enumerate(roles, 1):
                ja = swing_twist_deg(basis[r])
                _reg(signals, f"{pre}.flex{i}", np.abs(ja["flex_deg"]), "deg",
                     f"{finger} 第{i}节屈伸", ("finger",))
            root = point.get(f"{pre}.root")
            tip = point.get(f"{pre}.tip")
            if root is not None and tip is not None:
                d = tip - root
                norm = np.linalg.norm(d, axis=1)
                norm[norm < 1e-9] = 1.0
                dn = d / norm[:, None]
                up_err = np.degrees(np.arccos(np.clip(dn[:, 2], -1, 1)))
                _reg(signals, f"{pre}.up_err", up_err, "deg",
                     f"{finger}指向与 +Z 的夹角（0=竖直朝上）", ("finger",))
                _reg(signals, f"{pre}.azimuth",
                     np.unwrap(np.radians(
                         np.degrees(np.arctan2(dn[:, 1], dn[:, 0])))
                     ) * 180 / np.pi,
                     "deg", f"{finger}水平方位角", ("finger",))

    if "hips" in pos:
        h = pos["hips"][:, 2]
        _reg(signals, "pelvis.h", h, "m", "骨盆高度", ("force",))
        sp = np.linalg.norm(_diff(pos["hips"]), axis=1)
        _reg(signals, "pelvis.speed", sp, "m/f", "骨盆速度", ("force",))
        acc = _diff(sp)
        _reg(signals, "pelvis.acc", acc, "m/f^2", "骨盆加速度（发力感）", ("force",))
        _reg(signals, "pelvis.jerk", _diff(acc), "m/f^3",
             "骨盆加加速度（冲击/顿挫感）", ("force",))

    for role, q in quat.items():
        geo = _quat_geo_deg(np.asarray(q))
        _reg(signals, f"{role}.rot_speed", geo, "deg/f",
             f"{role} 旋转速度", ("force", "pivot"))
        if raw_quat is not None and role in raw_quat:
            raw_geo = _quat_geo_deg(np.asarray(raw_quat[role]))
            if len(raw_geo) == len(geo):
                _reg(signals, f"hf_loss.{role}",
                     _hf_loss(raw_geo, geo, HF_WINDOW), "deg/f",
                     f"{role} 滤波削掉的高频能量（力量感损失量）", ("force",))

    for kind, label, sides in (
        ("contact", "脚着地", ("L", "R")),
    ):
        for side in sides:
            key = f"{kind}.{side}"
            _reg(signals, key, _mask(frames, intervals.get(key, ())),
                 "0/1", f"{label}区间掩码", ("ground",))
    for key, label in (("air", "双脚腾空"), ("jitter.L", "左手抖动标注"),
                       ("jitter.R", "右手抖动标注")):
        _reg(signals, key, _mask(frames, intervals.get(key, ())),
             "0/1", f"{label}掩码", ("ground", "force"))

    return signals


GROUP_CHANNELS = {
    "pivot": ("yaw", "yaw_rate", "pivot_idx", "pivot_still", "toe_h", "speed_xy"),
    "force": ("pelvis.acc", "pelvis.jerk", "rot_speed", "hf_loss"),
    "ground": ("sole_h", "pen", "clearance", "contact", "speed_xy"),
    "finger": ("curl", "up_err", "azimuth", "flex"),
}
