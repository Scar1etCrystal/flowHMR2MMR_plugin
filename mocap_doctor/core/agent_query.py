"""Declarative query tools over the baked signal store (pure, JSON in/out).

Every function returns the inner half of the unified envelope
    {"summary", "data", "warnings", "truncated", "hint"}
- the bridge adds {"ok", "tool", "version", "error"}.  Errors raise
``AgentQueryError(code, message, fix)`` with fixed codes:
E_UNKNOWN (name typo, suggestion in fix), E_RANGE, E_SCOPE.

Numbers: lengths rounded to mm (0.001 m), angles to 0.1 deg, ratios 2 digits.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

MAX_SERIES_POINTS = 200
MAX_LIST_ITEMS = 200
_OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


class AgentQueryError(ValueError):
    def __init__(self, message: str, code: str = "E_UNKNOWN", fix: str = ""):
        super().__init__(message)
        self.code = code
        self.fix = fix


def _env(summary: str, data: Any, *, warnings=None, truncated=False,
         hint="") -> dict:
    return {"summary": summary, "data": data,
            "warnings": list(warnings or []), "truncated": bool(truncated),
            "hint": hint}


def _f(v, digits=3):
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return v


@dataclass
class DataStore:
    frames: np.ndarray                       # (T,) work-frame numbers
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    signal_meta: dict[str, dict] = field(default_factory=dict)
    intervals: dict[str, list] = field(default_factory=dict)  # kind → items
    positions: dict[str, np.ndarray] = field(default_factory=dict)
    joint_basis: dict[str, np.ndarray] = field(default_factory=dict)
    bone_roles: dict[str, str] = field(default_factory=dict)  # role → bone
    fps: float = 30.0
    floor_z: float = 0.0

    def frame_bounds(self):
        return int(self.frames[0]), int(self.frames[-1])


def build_store(
    signals: Mapping[str, Mapping[str, Any]],
    frames: Sequence[int],
    *,
    intervals: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    positions: Mapping[str, np.ndarray] | None = None,
    joint_basis: Mapping[str, np.ndarray] | None = None,
    bone_roles: Mapping[str, str] | None = None,
    fps: float = 30.0,
    floor_z: float = 0.0,
) -> DataStore:
    """Assemble a DataStore from ``agent_signals.compute_signals`` output."""
    store = DataStore(
        frames=np.asarray(frames),
        positions=dict(positions or {}),
        joint_basis=dict(joint_basis or {}),
        bone_roles=dict(bone_roles or {}),
        fps=float(fps),
        floor_z=float(floor_z),
    )
    for name, entry in signals.items():
        store.signals[name] = np.asarray(entry["values"], dtype=np.float64)
        store.signal_meta[name] = {
            "unit": entry.get("unit", ""),
            "desc": entry.get("desc", ""),
            "groups": list(entry.get("groups", ())),
        }
    for kind, items in (intervals or {}).items():
        store.intervals[kind] = [dict(item) for item in items]
    return store


def contact_heights(store: DataStore) -> dict:
    """每只脚"正常着地"时脚底点（关节中心）离 floor_z 的高度（米）。

    脚底点是踝/前掌/脚尖骨的头尾，不是鞋底：穿厚底靴的模型（Teto）踩实时它们离地
    7–8 cm，原来按"贴着 floor_z"判断，每段接触都被报成悬空、穿地永远报不出。取每段
    contact 标注里脚底点最低值的中位数作参照；不足 3 段 → 不标定（按 0，旧行为）。
    结果缓存在 store 上（快照不变它就不变）。"""
    cached = getattr(store, "_contact_heights", None)
    if cached is not None:
        return cached
    out = {}
    fr = np.asarray(store.frames)
    for side in ("L", "R"):
        sole = store.signals.get(f"foot.{side}.sole_h")
        if sole is None:
            continue
        mins = []
        for item in store.intervals.get(f"contact.{side}", ()):
            m = (fr >= int(item["start"])) & (fr <= int(item["end"]))
            if m.any():
                mins.append(float(sole[m].min()) - float(store.floor_z))
        if len(mins) >= 3:
            arr = np.asarray(mins) * 1000.0
            out[side] = {"height_m": float(np.median(arr)) / 1000.0, "contacts": len(mins),
                         "p10_mm": round(float(np.percentile(arr, 10)), 1),
                         "p90_mm": round(float(np.percentile(arr, 90)), 1)}
    store._contact_heights = out
    return out


def _channel(store: DataStore, name: str) -> np.ndarray:
    if name in store.signals:
        return store.signals[name]
    close = difflib.get_close_matches(name, store.signals.keys(), n=3,
                                      cutoff=0.5)
    hint = f"是不是想要 {' / '.join(close)}？" if close else \
        "用 get_overview 看全部信号名"
    raise AgentQueryError(f"没有信号 {name}", code="E_UNKNOWN", fix=hint)


def _resolve_range(store: DataStore, target) -> tuple[int, int]:
    """target: 'kind:id', [start, end], or int frame."""
    if isinstance(target, str) and ":" in target:
        kind, _, ident = target.partition(":")
        for item in store.intervals.get(kind, ()):
            if str(item.get("id")) == ident:
                return int(item["start"]), int(item["end"])
        raise AgentQueryError(
            f"找不到区间 {target}", code="E_UNKNOWN",
            fix="用 list_intervals 看有哪些区间")
    if isinstance(target, (list, tuple)) and len(target) == 2:
        a, b = int(target[0]), int(target[1])
        lo, hi = store.frame_bounds()
        if a > b:
            raise AgentQueryError(f"帧范围 {a}>{b} 颠倒", code="E_RANGE")
        if b < lo or a > hi:
            raise AgentQueryError(
                f"帧范围 {a}-{b} 在数据 {lo}-{hi} 之外", code="E_RANGE",
                fix=f"有效范围 {lo}-{hi}")
        return a, b
    if isinstance(target, (int, float)):
        f = int(target)
        return f, f
    raise AgentQueryError(
        f"无法解析 target：{target!r}（支持 'kind:id'、[start,end]、帧号）",
        code="E_UNKNOWN")


def _window(store: DataStore, start: int, end: int):
    return (store.frames >= start) & (store.frames <= end)


def _stats(store: DataStore, name: str, start: int, end: int) -> dict:
    series = _channel(store, name)
    w = _window(store, start, end)
    if not w.any():
        raise AgentQueryError(
            f"帧范围 {start}-{end} 在数据之外", code="E_RANGE")
    vals = series[w]
    fr = store.frames[w]
    i_max, i_min = int(vals.argmax()), int(vals.argmin())
    return {
        "min": _f(vals.min()), "max": _f(vals.max()),
        "mean": _f(vals.mean()), "std": _f(vals.std()),
        "first": _f(vals[0]), "last": _f(vals[-1]),
        "argmax_frame": int(fr[i_max]), "argmin_frame": int(fr[i_min]),
    }


# ---------------------------------------------------------------------------

def get_overview(store: DataStore) -> dict:
    lo, hi = store.frame_bounds()
    n_intervals = sum(len(v) for v in store.intervals.values())
    return _env(
        f"{lo}-{hi} 帧 · {len(store.signals)} 个信号 · {n_intervals} 个区间",
        {
            "frames": [lo, hi],
            "fps": store.fps,
            "floor_z": _f(store.floor_z),
            "convention": "世界坐标 Z 朝上；长度单位米（读数已按 mm 精度修约）；帧号含端点",
            "channels": [
                {"name": n, "unit": m["unit"], "desc": m["desc"]}
                for n, m in sorted(store.signal_meta.items())
            ],
            "intervals": {
                kind: {"count": len(items), "items": items[:MAX_LIST_ITEMS]}
                for kind, items in store.intervals.items()
            },
        })


def list_intervals(
    store: DataStore,
    kind: str | None = None,
    frame_range: Sequence[int] | None = None,
    tag: str | None = None,
) -> dict:
    items = []
    for k, seq in store.intervals.items():
        if kind and k != kind:
            continue
        for item in seq:
            if tag and str(item.get("tag", "")) != tag:
                continue
            if frame_range is not None:
                a, b = int(frame_range[0]), int(frame_range[1])
                if int(item["end"]) < a or int(item["start"]) > b:
                    continue
            items.append({"kind": k, **item})
    shown = items[:MAX_LIST_ITEMS]
    return _env(
        f"{len(items)} 个区间" + (f"（截断 {len(items)-len(shown)}）"
                                if len(items) > len(shown) else ""),
        {"items": shown, "total": len(items), "shown": len(shown)},
        truncated=len(items) > len(shown),
        hint="加 kind / frame_range / tag 缩小范围" if len(items) > len(shown)
        else "")


def list_timeline_markers(
    markers: Sequence[Mapping[str, Any]],
    intervals: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    frame_range: Sequence[int] | None = None,
    name: str | None = None,
    with_intervals: bool = True,
) -> dict:
    """The user's own timeline markers (Blender ``M`` key) as anchor frames.

    A user pointing at "就是这一下" is far more precise with a named marker on
    the timeline than with a frame number typed into a message, and unlike a
    message it survives as long as the file does.  Each marker is reported
    together with the annotation intervals covering that frame, so "出拳 505"
    arrives already joined to "那几帧左脚是 planted、在左手区间里" - the
    agent does not have to re-derive what the user was pointing at.
    """

    a = b = None
    if frame_range is not None:
        if not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2:
            raise AgentQueryError(
                f"frame_range 要写成 [起, 止]，收到 {frame_range!r}",
                code="E_RANGE", fix='例："frame_range":[505,560]')
        a, b = int(frame_range[0]), int(frame_range[1])
        if a > b:
            raise AgentQueryError(f"帧范围 {a}>{b} 颠倒", code="E_RANGE")

    needle = str(name).lower() if name else None
    items: list[dict[str, Any]] = []
    for entry in markers:
        try:
            frame = int(entry.get("frame"))
        except (TypeError, ValueError):
            continue
        label = str(entry.get("name") or "")
        if needle is not None and needle not in label.lower():
            continue
        if a is not None and not (a <= frame <= b):
            continue
        row: dict[str, Any] = {"name": label, "frame": frame}
        if with_intervals and intervals:
            row["covered_by"] = [
                {"kind": kind, "start": int(iv["start"]), "end": int(iv["end"])}
                for kind, seq in intervals.items()
                for iv in seq
                if int(iv["start"]) <= frame <= int(iv["end"])
            ]
        items.append(row)
    items.sort(key=lambda row: row["frame"])
    shown = items[:MAX_LIST_ITEMS]

    if not markers:
        hint = ("时间轴上没有标记。用户按 M 键放命名标记来指「就是这一下」；"
                "没有标记就按他给的帧号用 analyze_motion 在 [N−12, N+25] 上找 onset/stop")
    elif not items:
        hint = "有标记但被 frame_range / name 过滤掉了：去掉过滤再看"
    else:
        hint = ("标记帧 = 用户指的那一下；写入窗用 analyze_motion 在该帧附近取 "
                "onset/stop，别把标记帧直接当窗口端点")
    preview = "、".join(
        f"{row['frame']}{(' ' + row['name']) if row['name'] else ''}"
        for row in shown[:6])
    return _env(
        f"{len(items)} 个时间轴标记" + (f"：{preview}" if items else ""),
        {"items": shown, "total": len(items), "shown": len(shown)},
        warnings=[] if (items or not markers) else ["标记都被 frame_range / name 过滤掉了"],
        truncated=len(items) > len(shown),
        hint=hint,
    )


def _flags(store: DataStore, start: int, end: int) -> list:
    """Pre-judged anomalies for the window - one line each."""
    out = []
    fr = store.frames
    win = _window(store, start, end)
    heights = contact_heights(store)
    for side in ("L", "R"):
        ch = heights.get(side, {}).get("height_m", 0.0)   # 这只脚正常着地的高度
        pen = store.signals.get(f"foot.{side}.pen")
        if pen is not None and win.any():
            rel_pen = pen + ch                              # 比正常着地低多少
            if float(rel_pen[win].max()) > 0.005:
                i = int(np.argmax(rel_pen * win))
                out.append(f"foot.{side} 在 {int(fr[i])} 帧比正常着地低 "
                           f"{_f(rel_pen[i] * 1000, 1)}mm（下沉/穿地）" if ch else
                           f"foot.{side}.pen 在 {int(fr[i])} 帧最深 "
                           f"{_f(pen[i] * 1000, 1)}mm（穿地）")
        sole = store.signals.get(f"foot.{side}.sole_h")
        contact = store.signals.get(f"contact.{side}")
        if sole is not None and contact is not None:
            rel = sole - store.floor_z - ch
            bad = win & (contact > 0.5) & (rel > 0.02)
            if bad.any():
                i = int(np.argmax(rel * bad))
                out.append(f"foot.{side} 着地段 {int(fr[i])} 帧比正常着地高 "
                           f"{_f(rel[i] * 1000, 1)}mm（悬空）" if ch else
                           f"foot.{side} 着地段 {int(fr[i])} 帧还悬空 "
                           f"{_f(rel[i] * 1000, 1)}mm")
        yawr = store.signals.get(f"foot.{side}.yaw_rate")
        still = store.signals.get(f"foot.{side}.pivot_still")
        if yawr is not None and still is not None:
            bad = win & (still > 0.5) & (yawr > 15)
            if bad.any():
                i = int(np.argmax(yawr * bad))
                out.append(f"foot.{side} {int(fr[i])} 帧支点不动但 "
                           f"yaw_rate {_f(yawr[i], 1)}°/f（疑似碾转）")
        for finger in ("index", "middle", "ring", "pinky", "thumb"):
            curl = store.signals.get(f"finger.{side}.{finger}.curl")
            if curl is not None and win.any():
                bad = win & (curl > 15)
                if bad.any():
                    i = int(np.argmax(curl * win))
                    out.append(f"finger.{side}.{finger}.curl 在 {int(fr[i])} "
                               f"帧 {_f(curl[i], 1)}°（弯曲）")
    return out


def describe(
    store: DataStore,
    target,
    channels: Sequence[str] | None = None,
    context: int = 15,
) -> dict:
    start, end = _resolve_range(store, target)
    if channels is None:
        group = None
        if isinstance(target, str) and ":" in target:
            kind, _, ident = target.partition(":")
            for item in store.intervals.get(kind, ()):
                if str(item.get("id")) == ident:
                    group = str(item.get("tag", ""))
                    break
        from .agent_signals import GROUP_CHANNELS
        suffixes = GROUP_CHANNELS.get(group or "", ())
        channels = [
            n for n in store.signals
            if any(n.endswith(suf) or suf in n for suf in suffixes)
        ][:8] or list(store.signals)[:6]
    stats = {}
    lo_bound, hi_bound = store.frame_bounds()
    for name in channels:
        entry = _stats(store, name, start, end)
        entry["unit"] = store.signal_meta.get(name, {}).get("unit", "")
        if context:
            entry["before"] = (
                _stats(store, name, max(start - context, lo_bound), start - 1)
                if start - 1 >= lo_bound else None)
            entry["after"] = (
                _stats(store, name, end + 1, min(end + context, hi_bound))
                if end + 1 <= hi_bound else None)
        stats[name] = entry
    events = []
    for name in channels:
        series = _channel(store, name)
        w = _window(store, start, end)
        vals = series[w]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826 or 1e-9
        hot = store.frames[w][vals > med + 3 * mad]
        if len(hot):
            events.append({"channel": name, "type": "spike",
                           "frames": [int(f) for f in hot[:24]]})
    flags = _flags(store, start, end)
    summary = f"{start}-{end}（{end - start + 1} 帧）"
    if flags:
        summary += " · " + "；".join(flags[:3]) + \
                   ("…" if len(flags) > 3 else "")
    return _env(summary, {
        "target": target,
        "frames": [start, end],
        "length": end - start + 1,
        "stats": stats,
        "events": events,
        "flags": flags,
    })


def get_series(
    store: DataStore,
    channels: Sequence[str],
    frame_range: Sequence[int],
    max_points: int = 60,
    agg: str = "mean",
) -> dict:
    start, end = _resolve_range(store, frame_range)
    max_points = max(2, min(int(max_points), MAX_SERIES_POINTS))
    w = _window(store, start, end)
    fr = store.frames[w]
    n = len(fr)
    step = max(1, int(np.ceil(n / max_points)))
    idx = np.arange(0, n, step)
    fr_out = fr[idx]
    aggs = {"mean": np.mean, "max": np.max, "min": np.min}
    agg_fn = aggs.get(str(agg), np.mean)
    series = {}
    for name in channels:
        vals = _channel(store, name)[w]
        series[name] = [
            _f(agg_fn(vals[i:i + step])) for i in idx
        ]
    return _env(
        f"{n} 帧按 step={step} 聚合（{agg}）",
        {"frames": [int(f) for f in fr_out], "step": int(step),
         "agg": str(agg), "series": series},
        truncated=step > 1,
        hint="缩小 frame_range 或提高 max_points" if step > 1 else "")


def find_events(
    store: DataStore,
    cond: Mapping[str, Any],
    frame_range: Sequence[int] | None = None,
) -> dict:
    """Declarative condition → matching frame ranges.

    cond = {"all": [{"ch": name, "op": "<", "v": x}, ...],
            "any": [...],               # optional OR group
            "min_len": int }
    """
    lo, hi = store.frame_bounds() if frame_range is None else \
        (int(frame_range[0]), int(frame_range[1]))
    mask = _window(store, lo, hi)
    frames = store.frames

    def clause(term):
        name = term.get("ch")
        op = term.get("op")
        if op not in _OPS:
            raise AgentQueryError(
                f"不支持的运算符 {op!r}", code="E_UNKNOWN",
                fix=f"可用运算符：{sorted(_OPS)}")
        series = _channel(store, str(name))
        return _OPS[op](series, float(term.get("v", 0.0)))

    all_terms = cond.get("all") or []
    any_terms = cond.get("any") or []
    if not all_terms and not any_terms:
        raise AgentQueryError("cond 需要 all 或 any 条件列表", code="E_UNKNOWN")
    keep = np.ones(len(frames), dtype=bool)
    for term in all_terms:
        keep &= clause(term)
    if any_terms:
        any_mask = np.zeros(len(frames), dtype=bool)
        for term in any_terms:
            any_mask |= clause(term)
        keep &= any_mask
    keep &= mask

    hits = frames[keep]
    min_len = int(cond.get("min_len", 1))
    matches = []
    if len(hits):
        # also report the peak inside each match for the first "all" channel
        probe = all_terms[0]["ch"] if all_terms else (any_terms[0]["ch"])
        probe_s = store.signals.get(str(probe)) if probe else None
        run_start = prev = int(hits[0])
        runs = []

        def flush(a, b):
            if b - a + 1 >= min_len:
                item = {"frames": [a, b]}
                if probe_s is not None:
                    seg = probe_s[_window(store, a, b)]
                    i = int(np.argmax(np.abs(seg)))
                    item["peak"] = {"frame": int(
                        frames[_window(store, a, b)][i]),
                        "value": _f(seg[i])}
                runs.append(item)

        for f in hits[1:]:
            f = int(f)
            if f == prev + 1:
                prev = f
            else:
                flush(run_start, prev)
                run_start = prev = f
        flush(run_start, prev)
        matches = runs[:MAX_LIST_ITEMS]
    return _env(f"{len(matches)} 段命中（{lo}-{hi} 范围内）",
                {"cond_echo": dict(cond), "matches": matches,
                 "count": len(matches), "frame_range": [lo, hi]})


def compare(store: DataStore, channel: str, a, b) -> dict:
    sa, ea = _resolve_range(store, a)
    sb, eb = _resolve_range(store, b)
    s = _channel(store, channel)
    va = s[_window(store, sa, ea)]
    vb = s[_window(store, sb, eb)]
    qa = np.quantile(va, np.linspace(0, 1, 11)) if len(va) else va
    qb = np.quantile(vb, np.linspace(0, 1, 11)) if len(vb) else vb
    dist = _f(np.abs(qa - qb).mean(), 4) if len(qa) == len(qb) else None
    peak_ratio = round(float(va.max() / vb.max()), 2) \
        if len(vb) and vb.max() else None
    return _env(
        f"{channel}：峰值比 {peak_ratio}，均值差 "
        f"{_f(va.mean() - vb.mean(), 4)}",
        {"channel": channel,
         "a": {"frames": [sa, ea], "stats": _stats(store, channel, sa, ea)},
         "b": {"frames": [sb, eb], "stats": _stats(store, channel, sb, eb)},
         "peak_ratio": peak_ratio,
         "mean_diff": _f(va.mean() - vb.mean(), 4),
         "quantile_l1": dist})


def snapshot(store: DataStore, frame: int,
             roles: Sequence[str] | None = None) -> dict:
    f = int(frame)
    idx = int(np.searchsorted(store.frames, f))
    if idx >= len(store.frames) or int(store.frames[idx]) != f:
        raise AgentQueryError(
            f"帧 {f} 不在数据范围内", code="E_RANGE",
            fix=f"有效范围 {store.frame_bounds()}")
    picked = roles or list(store.positions)
    return _env(
        f"帧 {f} · {len([r for r in picked if r in store.positions])} 个角色",
        {"frame": f,
         "positions": {
             role: [_f(v) for v in store.positions[role][idx]]
             for role in picked if role in store.positions
         }})


def get_joint_angles(
    store: DataStore,
    bones: Sequence[str],
    frame_range: Sequence[int],
    max_points: int = 60,
) -> dict:
    """Per-frame flex/abd/twist (deg) for each bone - LLM never sees quats."""
    from .agent_fx import joint_angles

    start, end = _resolve_range(store, frame_range)
    w = _window(store, start, end)
    fr = store.frames[w]
    n = len(fr)
    step = max(1, int(np.ceil(n / max(2, min(int(max_points),
                                           MAX_SERIES_POINTS)))))
    idx = np.arange(0, n, step)
    out = {}
    for role_or_bone in bones:
        role = role_or_bone
        if role not in store.joint_basis:
            inv = {v: k for k, v in store.bone_roles.items()}
            role = inv.get(role_or_bone, role_or_bone)
        if role not in store.joint_basis:
            close = difflib.get_close_matches(
                str(role_or_bone), list(store.joint_basis), n=3, cutoff=0.5)
            raise AgentQueryError(
                f"没有关节角数据：{role_or_bone}", code="E_UNKNOWN",
                fix=("是不是想要 " + " / ".join(close) + "？") if close
                else "该骨骼不在烘焙集里")
        ja = joint_angles(store.joint_basis[role][w])
        out[store.bone_roles.get(role, role)] = {
            "flex_deg": [_f(ja["flex_deg"][i], 1) for i in idx],
            "abd_deg": [_f(ja["abd_deg"][i], 1) for i in idx],
            "twist_deg": [_f(ja["twist_deg"][i], 1) for i in idx],
        }
    return _env(
        f"{len(out)} 根骨骼 × {len(idx)} 帧关节角",
        {"frames": [int(f) for f in fr[idx]], "step": int(step),
         "bones": out},
        truncated=step > 1,
        hint="缩小 frame_range 或提高 max_points" if step > 1 else "")
