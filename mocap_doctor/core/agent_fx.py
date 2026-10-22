"""Pure math for the agent write tools.

Everything here is (T,) or (T,k) numpy in / numpy out - no Blender objects.
The write side stores *deltas* on a Combine NLA strip, so the functions return
delta channels (scalar: desired - base; quat: desired * conj(base)) with the
edges already tapered to identity.  ``undo`` is then just "delete the strip".
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .accent import window_weights


# ---------------------------------------------------------------------------
# generic delta helpers

def delta_scalar(desired: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.asarray(desired, dtype=np.float64) - np.asarray(base, dtype=np.float64)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _qconj(q):
    out = np.asarray(q, dtype=np.float64).copy()
    out[1:] = -out[1:]
    return out


def _qmul_rows(a, b):
    """Row-wise _qmul over (T,4) arrays.  Same expression, same evaluation
    order, IEEE float64 elementwise → bit-identical to looping _qmul (numpy
    ufuncs never fuse mul/sub into FMA), ~100× faster on long series."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)


def delta_quat(desired: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Per-frame relative rotation  conj(base) ⊗ desired  (wxyz, sign-continuous).

    Blender's NLA quaternion Combine right-multiplies:  result = lower ⊗ strip.
    So delta must satisfy ``base ⊗ delta = desired`` → ``delta = conj(base) ⊗
    desired``.  (Left-multiply order yields base ⊗ delta ≠ desired.)
    """
    desired = np.asarray(desired, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    desired = desired / np.linalg.norm(desired, axis=1, keepdims=True)
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    conj = base.copy()
    conj[:, 1:] = -conj[:, 1:]
    out = (_qmul_rows(conj, desired) if len(desired)
           else np.zeros((0, 4)))
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    for i in range(1, len(out)):
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] = -out[i]
    return out


def taper_deltas(deltas: np.ndarray, blend: int = 4) -> np.ndarray:
    """Scale deltas → 0 at both ends so the strip fades in/out smoothly."""
    w = window_weights(len(deltas), blend)
    while deltas.ndim > w.ndim:
        w = w[..., None]
    return np.asarray(deltas, dtype=np.float64) * w


def taper_quat_deltas(d_quat: np.ndarray, blend: int = 4) -> np.ndarray:
    """Taper a rotation delta toward identity at the edges (angle scaling)."""
    from .pkl_hand import aa_to_quat, quat_to_aa

    w = window_weights(len(d_quat), blend)[:, None]
    aa = quat_to_aa(np.asarray(d_quat)) * w
    return aa_to_quat(aa)


# ---------------------------------------------------------------------------
# clean_jitter

def smooth_zero_phase(values: np.ndarray, width: int = 5) -> np.ndarray:
    """Zero-phase smoothing: apply a Hann kernel forward then backward."""
    values = np.asarray(values, dtype=np.float64)
    width = max(3, int(width) | 1)
    x = np.arange(width) - width // 2
    kernel = 0.5 + 0.5 * np.cos(np.pi * x / (width // 2 + 1))
    kernel /= kernel.sum()

    def _pass(v):
        pad = width // 2
        vp = np.pad(v, pad, mode="edge")
        return np.convolve(vp, kernel, mode="valid")[: len(v)]

    return _pass(_pass(values)[::-1])[::-1]


def clean_jitter_values(
    values: np.ndarray,
    *,
    strength: float = 1.0,
    width: int = 5,
    blend: int = 4,
) -> np.ndarray:
    """Zero-phase smooth inside the window only; edges blend back to original."""
    smoothed = smooth_zero_phase(values, width)
    w = window_weights(len(values), blend)
    return values + (smoothed - values) * (w * float(strength))


# ---------------------------------------------------------------------------
# fix_ground

def ground_height_targets(
    sole_h: np.ndarray,
    *,
    floor_z: float,
    rest_clearance: float = 0.0,
    mode: str = "lift",
    blend: int = 4,
) -> np.ndarray:
    """Desired sole heights for a [start..end] window.

    lift  - only pull down frames that float above floor+clearance
    pen   - only push up frames that penetrate the floor
    float - force every frame to floor+clearance
    """
    sole_h = np.asarray(sole_h, dtype=np.float64)
    target = np.full(len(sole_h), float(floor_z) + float(rest_clearance))
    if mode == "lift":
        desired = np.minimum(sole_h, target)
    elif mode == "pen":
        desired = np.maximum(sole_h, target)
    elif mode == "float":
        desired = target.copy()
    else:
        raise ValueError(f"未知 mode：{mode}")
    w = window_weights(len(sole_h), blend)
    return sole_h + (desired - sole_h) * w


# ---------------------------------------------------------------------------
# solve_pelvis

def pelvis_height_corrections(
    pelvis_h: np.ndarray,
    hip_xyz: np.ndarray,
    ankle_xyz: np.ndarray,
    *,
    leg_len: float,
    src_leg_ratio: np.ndarray | float,
    mode: str = "keep_ratio",
    blend: int = 4,
    clamp: float | None = None,
) -> np.ndarray:
    """Pelvis height deltas so the planted leg reaches like the source did.

    ``src_leg_ratio`` = source 髋→踝距离 / 源腿长 (scalar or per-frame).
    Desired model hip height satisfies dist(desired_hip, ankle) = ratio*leg_len;
    solve along Z (hip shifts vertically), keep_ratio applies always,
    reach_only only where the model currently can't reach.
    Returns *deltas* to add to pelvis z (both feet: caller passes each side and
    takes the lower result).
    """
    pelvis_h = np.asarray(pelvis_h, dtype=np.float64)
    ankle = np.asarray(ankle_xyz, dtype=np.float64)
    hip = np.asarray(hip_xyz, dtype=np.float64)
    ratio = np.broadcast_to(np.asarray(src_leg_ratio, dtype=np.float64),
                            pelvis_h.shape)
    want = ratio * float(leg_len)
    # horizontal offset hip→ankle stays; solve vertical component
    d_xy = np.linalg.norm(hip[:, :2] - ankle[:, :2], axis=1)
    need_z2 = np.maximum(want ** 2 - d_xy ** 2, 0.0)
    desired_hip_z = ankle[:, 2] + np.sqrt(need_z2)
    dz = desired_hip_z - hip[:, 2]
    if mode == "reach_only":
        dz = np.minimum(dz, 0.0)     # only sink when it can't reach
    elif mode != "keep_ratio":
        raise ValueError(f"未知 mode：{mode}")
    pelvis_dz = pelvis_h - hip[:, 2] + dz + hip[:, 2] - pelvis_h  # = dz
    delta = np.asarray(pelvis_dz)
    if clamp is not None:
        delta = np.clip(delta, -float(clamp), float(clamp))
    return delta * window_weights(len(delta), blend)


# ---------------------------------------------------------------------------
# exemplar migration

def swing_twist_deg(basis_quats: np.ndarray, axis: int = 1):
    """Decompose per-frame basis quats (wxyz) into swing + twist around a bone
    axis (Blender pose bones: Y = along the bone).

    ``axis`` indexes the rotation *vector* xyz part (1 = Y, the bone axis);
    internally it maps to quaternion column axis+1 since w occupies column 0.
    """
    q = np.asarray(basis_quats, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    # Same rotation, w >= 0 hemisphere: a sign-continuous series that has
    # wrapped past 180 deg (a body that turned around) has w < 0, and the
    # twist came out as 2*arccos(-|w|) → ±(180..360) deg, jumping ~715 deg at
    # the sign change.  Swing / flex / abd are unchanged by the flip.
    q = np.where(q[:, :1] < 0.0, -q, q)
    col = 1 + int(axis)                    # wxyz: component col = axis+1
    # twist = component along `axis`: t = normalize(w, v*axis)
    t = np.zeros_like(q)
    t[:, 0] = q[:, 0]
    t[:, col] = q[:, col]
    norm = np.linalg.norm(t, axis=1)
    norm[norm < 1e-12] = 1.0
    t = t / norm[:, None]
    sign = np.sign(q[:, col])
    sign[sign == 0] = 1.0
    twist_deg = np.degrees(2.0 * np.arccos(np.clip(t[:, 0], -1, 1))) * sign
    # swing = q ⊗ conj(twist)
    tconj = t.copy()
    tconj[:, 1:] = -tconj[:, 1:]
    sw = _qmul_rows(q, tconj) if len(q) else np.zeros((0, 4))
    sw = sw / np.linalg.norm(sw, axis=1, keepdims=True)
    ang = np.degrees(2.0 * np.arccos(np.clip(sw[:, 0], -1, 1)))
    # swing axis = (x,z) plane components (axis component is ~0)
    ax, az = sw[:, 1], sw[:, 3]
    nrm = np.sqrt(ax * ax + az * az)
    nrm[nrm < 1e-9] = 1.0
    return {
        "twist_deg": twist_deg,
        "swing_deg": ang,
        "flex_deg": ang * (ax / nrm),
        "abd_deg": ang * (az / nrm),
    }


def joint_angles(basis_quats: np.ndarray) -> dict:
    """Per-frame flex/abd/twist for one bone's basis quat series."""
    return swing_twist_deg(basis_quats)


def slerp_array(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Single slerp between two wxyz quats."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        out = q0 + float(t) * (q1 - q0)
        return out / np.linalg.norm(out)
    th = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def slerp_series(edge0: np.ndarray, edge1: np.ndarray, n: int) -> np.ndarray:
    """(n,4) slerp bridge for the n frames BETWEEN two good boundary quats.

    t = (k+1)/(n+1): the boundaries are the good neighbours themselves, so the
    bridge must not include them.  (The old i/(n-1) put copies of the good
    frames into the first/last bad frame - a stall then a double step at every
    repaired spike: one bad frame of a 2 deg/frame move came out 8 deg instead
    of 10.)  Same rule as pkl_hand's bridge strategy."""
    out = np.zeros((n, 4))
    for i in range(n):
        out[i] = slerp_array(edge0, edge1, (i + 1) / float(n + 1))
    return out


def quat_to_yaw_deg(quats: np.ndarray) -> np.ndarray:
    """wxyz quats → yaw angle (deg) around world Z, unwrapped."""
    q = np.asarray(quats, dtype=np.float64)
    x, y, z, w = q[:, 1], q[:, 2], q[:, 3], q[:, 0]
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return np.degrees(np.unwrap(np.radians(yaw)))


def exemplar_signature(
    foot_quats: np.ndarray,
    point_pos: np.ndarray,
) -> dict:
    """Context fingerprint for matching: yaw-rate series + duration + step stats."""
    yaw = quat_to_yaw_deg(foot_quats)
    yaw_rate = np.abs(np.diff(yaw))
    speeds = np.linalg.norm(np.diff(point_pos[:, :2], axis=0), axis=1)
    return {
        "duration": int(len(foot_quats)),
        "yaw_rate": [round(float(v), 3) for v in yaw_rate],
        "yaw_sweep": round(float(yaw[-1] - yaw[0]), 2),
        "speed_mean": round(float(speeds.mean()), 5) if len(speeds) else 0.0,
    }


def extract_residual(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    cur_pos: np.ndarray,
    cur_quat: np.ndarray,
    anchor_pos: np.ndarray,
    anchor_yaw_deg: float,
) -> dict:
    """Residual = cur − base expressed in the anchor's local frame.

    anchor = foot state at contact start; rotating deltas by -anchor_yaw makes
    the exemplar re-targetable to different facing directions.
    """
    a = np.radians(-anchor_yaw_deg)
    rot = np.array([[np.cos(a), -np.sin(a), 0],
                    [np.sin(a), np.cos(a), 0],
                    [0, 0, 1]])
    d_pos = (np.asarray(cur_pos) - np.asarray(base_pos)) @ rot.T
    d_quat = delta_quat(np.asarray(cur_quat), np.asarray(base_quat))
    # express orientation delta axes in anchor frame too (yaw only)
    return {"d_pos": d_pos.tolist(), "d_quat": d_quat.tolist(),
            "anchor_yaw_deg": float(anchor_yaw_deg)}


def _warp_idx(n_src: int, n_dst: int) -> np.ndarray:
    """Linear time alignment: output index i → source index."""
    if n_dst <= 1 or n_src <= 1:
        return np.zeros(max(1, n_dst), dtype=int)
    t = np.linspace(0.0, 1.0, max(1, n_dst))
    return np.clip((t * (n_src - 1)).round().astype(int), 0, n_src - 1)


def apply_residual(
    residual: Mapping[str, Any],
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    anchor_yaw_deg: float,
    *,
    yaw_scale: float = 1.0,
    mirror: bool = False,
    blend: int = 4,
) -> dict:
    """Project a stored residual onto a target interval.

    Time-scales to the target length, rotates deltas into the target's anchor
    yaw, optionally mirrors L↔R (negates local X), and returns ABSOLUTE desired
    pos/quat - the caller converts to deltas itself.
    """
    d_pos = np.asarray(residual["d_pos"], dtype=np.float64)
    d_quat = np.asarray(residual["d_quat"], dtype=np.float64)
    n_dst = len(target_pos)
    idx = _warp_idx(len(d_pos), n_dst)
    d_pos = d_pos[idx]
    d_quat = d_quat[idx]

    a = np.radians(float(anchor_yaw_deg))
    rot = np.array([[np.cos(a), -np.sin(a), 0],
                    [np.sin(a), np.cos(a), 0],
                    [0, 0, 1]])
    d_pos = d_pos @ rot                       # rotate local → world
    if mirror:
        d_pos[:, 0] *= -1.0                   # mirror across anchor plane
        d_quat[:, 1] *= -1.0                  # x component flips
    if yaw_scale != 1.0:
        # scale the rotational content by yaw_scale around its axis
        from .pkl_hand import quat_to_aa, aa_to_quat
        aa = quat_to_aa(d_quat) * float(yaw_scale)
        d_quat = aa_to_quat(aa)
    d_pos = taper_deltas(d_pos, blend)
    d_quat = taper_quat_deltas(d_quat, blend)

    desired_pos = np.asarray(target_pos) + d_pos
    desired_quat = np.asarray(
        [_qmul(np.asarray(d_quat[i]), np.asarray(target_quat)[i])
         for i in range(n_dst)]
    )
    desired_quat = desired_quat / np.linalg.norm(desired_quat, axis=1, keepdims=True)
    return {"pos": desired_pos, "quat": desired_quat}


def dtw(a: Sequence[float], b: Sequence[float]) -> float:
    """Small O(nm) DTW distance between two equal-ish sequences."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    if not n or not m:
        return float("inf")
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = abs(a[i - 1] - b[j - 1]) + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    return float(cost[n, m] / (n + m))


def match_signature(sig: Mapping[str, Any], cand: Mapping[str, Any]) -> float:
    """0..1 similarity: DTW on yaw_rate + duration agreement."""
    d = dtw(sig.get("yaw_rate", []), cand.get("yaw_rate", []))
    dur_ratio = min(sig.get("duration", 1), cand.get("duration", 1)) / \
        max(sig.get("duration", 1), cand.get("duration", 1))
    score = np.exp(-d / max(1.0, sig.get("duration", 1))) * dur_ratio
    return round(float(score), 4)
