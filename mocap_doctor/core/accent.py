"""Force-feel restore on animation channels ("修力量感" output side).

All four methods work on *sampled per-frame values* inside a marked window -
endpoints always land on the original values so position/velocity stay
continuous at the boundary, then the caller writes them back as dense keys
(Blender-side) or just reads the metrics.

Methods
-------
hf_reinject  - add back the band-limited residual (raw - current)
ease_reshape - steepen the approach into an impact frame, then overshoot
retime       - local time remap: compress the attack, stretch the recovery
refilter     - redo a weaker smoothing of raw inside the window, crossfade in
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

METHODS = ("hf_reinject", "ease_reshape", "retime", "refilter")


def window_weights(n: int, blend: int = 4) -> np.ndarray:
    """Smoothstep-tapered weights pinned to exactly 0 at both endpoints."""
    n = int(n)
    w = np.ones(n)
    blend = max(0, min(int(blend), n // 2))
    if blend:
        for i in range(blend):
            t = i / float(blend)
            s = t * t * (3.0 - 2.0 * t)      # smoothstep, t=0 at the edge
            w[i] = s
            w[n - 1 - i] = s
    return w


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width <= 1 or len(x) <= 2:
        return np.asarray(x, dtype=np.float64).copy()
    pad = width // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(width) / width
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def hf_reinject(
    cur: np.ndarray,
    raw: np.ndarray,
    *,
    strength: float = 0.5,
    blend: int = 4,
    detrend: int = 5,
) -> np.ndarray:
    """cur + strength * (band-limited residual), tapered at window edges.

    The residual's own moving average is subtracted first so a slow pose
    difference isn't re-added - only the sharp content comes back.
    """
    cur = np.asarray(cur, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    resid = raw - cur
    resid = resid - _moving_average(resid, detrend)
    w = window_weights(len(cur), blend)
    return cur + w * float(strength) * resid


def ease_reshape(
    values: np.ndarray,
    impact_idx: int,
    *,
    pre: int = 8,
    post: int = 8,
    strength: float = 0.5,
    overshoot: float = 0.15,
    blend: int = 4,
) -> np.ndarray:
    """Steepen velocity into `impact_idx`, then let the impact overshoot.

    Works on the velocity profile: approach speed ramps up to
    ``1 + strength`` at the impact, the impact frame's own delta is boosted by
    ``overshoot`` of the approach peak, and a short decay lets it settle.
    Endpoints of the affected span are re-pinned by construction.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = values.copy()
    k = int(impact_idx)
    a = max(1, k - int(pre))
    b = min(n - 1, k + int(post))
    if b - a < 2:
        return out

    seg = values[a:b + 1]
    v = np.diff(seg, prepend=seg[0])
    vp = v.copy()
    for i in range(1, k - a + 1):
        # approach: gain ramps 1 → 1+strength toward the impact
        t = i / max(1, k - a)
        vp[i] = v[i] * (1.0 + strength * t)
    peak = abs(v[max(1, k - a)]) if k - a >= 1 else 0.0
    for i in range(k - a + 1, len(v)):
        # overshoot: extra push right after impact, quadratic decay
        t = (i - (k - a)) / max(1, b - k)
        direction = 1.0 if v[k - a] >= 0 else -1.0
        vp[i] = v[i] + direction * overshoot * peak * (1.0 - t) ** 2
    rebuilt = seg[0] + np.concatenate([[0.0], np.cumsum(vp[1:])])
    # Pin the span end back to the original value.  The pin must live AFTER
    # the impact: spreading it over the whole span cancels the arrival boost
    # (measured on a decelerated rise: gain x2.4 came out as x0.9 - the
    # correction ate exactly what the ramp added).  Absorbed post-impact it
    # reads as overshoot-then-settle, which is the shape we want anyway.
    drift = rebuilt[-1] - seg[-1]
    j0 = k - a                       # arrival index (delta INTO the impact)
    tail = b - k                     # values strictly after the arrival
    if abs(drift) > 1e-12 and tail >= 3:
        rebuilt[j0 + 1:] -= np.linspace(0.0, drift, tail + 1)[1:]
    else:
        rebuilt -= np.linspace(0.0, drift, len(rebuilt))
    w = window_weights(len(seg), blend)
    out[a:b + 1] = seg + (rebuilt - seg) * w
    return out


def retime(
    values: np.ndarray,
    *,
    attack_speed: float = 1.5,
    split: float = 0.35,
    blend: int = 4,
) -> np.ndarray:
    """Compress the attack phase, stretch the recovery; endpoints fixed.

    A monotone piecewise-linear warp φ(t): slope ``attack_speed`` (>1 = attack
    passes faster) on [0, split], then whatever slope closes φ(1)=1.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 4:
        return values.copy()
    m = float(attack_speed)
    s = float(np.clip(split, 0.05, 0.9))
    if m * s >= 1.0:
        raise ValueError("attack_speed*split 必须 < 1，否则回不了端点")
    t = np.linspace(0.0, 1.0, n)
    phi = np.where(
        t <= s,
        t * m,
        s * m + (t - s) * (1.0 - s * m) / (1.0 - s),
    )
    phi = np.clip(phi, 0.0, 1.0)
    resampled = np.interp(phi, t, values)
    w = window_weights(n, blend)
    return values + (resampled - values) * w


def refilter(
    cur: np.ndarray,
    raw: np.ndarray,
    *,
    strength: float = 0.5,
    sigma: float = 1.0,
    blend: int = 4,
) -> np.ndarray:
    """Inside the window, blend toward a *weaker* smoothing of raw.

    ``cur`` is what the aggressive upstream filter produced; ``raw`` is the
    unfiltered source.  A light Gaussian on raw keeps its snap; strength gates
    how far the window moves toward it.
    """
    cur = np.asarray(cur, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    sigma = max(0.2, float(sigma))
    radius = max(1, int(sigma * 3))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    rp = np.pad(raw, radius, mode="edge")
    mild = np.convolve(rp, kernel, mode="valid")[: len(raw)]
    w = window_weights(len(cur), blend)
    return cur + w * float(strength) * (mild - cur)


# ---------------------------------------------------------------------------
# quaternion variants (wxyz, sign-continuous in/out)

def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def hf_reinject_quats(
    cur: np.ndarray,
    raw: np.ndarray,
    *,
    strength: float = 0.5,
    blend: int = 4,
) -> np.ndarray:
    """Quaternion version of hf_reinject: scale the per-frame residual
    rotation raw⊗conj(cur) by strength (geodesic, not component-wise)."""
    from .pkl_hand import aa_to_quat, quat_to_aa

    cur = np.asarray(cur, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)
    raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    raw = np.where((np.sum(raw * cur, axis=1) < 0)[:, None], -raw, raw)
    conj = cur.copy()
    conj[:, 1:] = -conj[:, 1:]
    delta = np.asarray([_quat_mul(raw[i], conj[i]) for i in range(len(cur))])
    aa = quat_to_aa(delta)
    # band-limit the residual: drop its slow drift, keep the sharp content
    aa = aa - np.stack([_moving_average(aa[:, c], 5) for c in range(3)], axis=1)
    w = window_weights(len(cur), blend)[:, None] * float(strength)
    scaled = aa_to_quat(aa * w)
    out = np.asarray([_quat_mul(cur[i], scaled[i]) for i in range(len(cur))])
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    for i in range(1, len(out)):
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] = -out[i]
    return out


def accent_metrics(
    before: np.ndarray,
    after: np.ndarray,
    raw: np.ndarray | None = None,
) -> dict:
    """How much snap came back: peak accel ratio vs raw (target ~0.8-1.0)."""
    def _peak_acc(v):
        v = np.asarray(v, dtype=np.float64)
        if len(v) < 3:
            return 0.0
        return float(np.abs(np.diff(np.diff(v))).max())

    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    out = {
        "peak_acc_before": round(_peak_acc(before), 5),
        "peak_acc_after": round(_peak_acc(after), 5),
    }
    if raw is not None:
        raw = np.asarray(raw, dtype=np.float64)
        peak_raw = _peak_acc(raw)
        out["peak_acc_raw"] = round(peak_raw, 5)
        out["acc_ratio_vs_raw"] = (
            round(_peak_acc(after) / peak_raw, 3) if peak_raw > 1e-12 else None
        )
    jerk = np.abs(np.diff(np.diff(np.diff(after))))
    out["peak_jerk_after"] = round(float(jerk.max()), 5) if len(jerk) else 0.0
    return out


def apply_scalar(
    action: Any,
    data_path: str,
    index: int,
    frame_start: int,
    frame_end: int,
    method: str,
    *,
    strength: float = 0.5,
    raw_values: np.ndarray | None = None,
    impact_frame: int | None = None,
    retime_speed: float = 1.5,
    retime_split: float = 0.35,
    blend: int = 4,
    group: str = "agent",
) -> dict:
    """Sample the channel, apply a method, write dense keys INTO the action.

    Direct-write variant for pkl-level/test paths - the agent flow should use
    ``agent_ops.restore_accent`` which puts the delta on a Combine NLA strip
    instead, keeping the source action untouched and revertable."""
    from . import agent_bake

    cur = agent_bake.sample_fcurve_values(action, data_path, index,
                                        frame_start, frame_end)
    if cur is None:
        raise RuntimeError(f"通道不存在：{data_path}[{index}]")
    n = len(cur)
    method = str(method)
    if method == "hf_reinject":
        if raw_values is None:
            raise RuntimeError("hf_reinject 需要 raw_values（滤波前数据）")
        new = hf_reinject(cur, np.asarray(raw_values)[:n],
                          strength=strength, blend=blend)
    elif method == "ease_reshape":
        k = (impact_frame or (frame_start + frame_end) // 2) - frame_start
        new = ease_reshape(cur, k, pre=n // 3, post=n // 3,
                           strength=strength, blend=blend)
    elif method == "retime":
        new = retime(cur, attack_speed=retime_speed, split=retime_split,
                     blend=blend)
    elif method == "refilter":
        if raw_values is None:
            raise RuntimeError("refilter 需要 raw_values")
        new = refilter(cur, np.asarray(raw_values)[:n],
                       strength=strength, blend=blend)
    else:
        raise RuntimeError(f"未知方式 {method}，可用 {METHODS}")

    agent_bake.write_fcurve_values(action, data_path, index, frame_start, new,
                                   group=group)
    metrics = accent_metrics(cur, new, None if raw_values is None else np.asarray(raw_values)[:n])
    metrics.update({"frames": [int(frame_start), int(frame_end)],
                    "path": data_path, "index": int(index),
                    "method": method})
    return metrics
