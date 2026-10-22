"""Pre-import hand repair on the merged PoseCapture pkl (Blender-free).

GVHMR + HaMeR hand failures are deterministic across re-runs of the same
video, and confidence channels do not flag them (HaMeR is confident when it
hallucinates).  The reliable fix is on the pkl itself: rewrite
``*_hand_global_orient`` (wrist lives inside ``body_pose`` columns 57:63) and
``*_hand_pose`` (15 joints x 3 axis-angles) inside user-marked ranges, write a
``*_repaired.pkl``, and let PoseCapture import the fixed source.

Detection works on the raw HaMeR pkl when present - its rotation matrices are
the un-smoothed signal (the merged pkl already passed one_euro filtering,
which damps exactly the spikes we look for).

This module is deliberately Blender-free so it can be unit tested.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Wrist axis-angle columns inside body_pose (SMPL joint order, L_Wrist/R_Wrist).
WRIST_COLS = {"left": slice(57, 60), "right": slice(60, 63)}
FINGER_JOINTS = 15
STRATEGIES = ("hold", "bridge", "smooth")
CHANNEL_CHOICES = ("wrist", "fingers", "both")

REF_WINDOW = 5
SMOOTH_MARGIN = 3
# Absolute-degree floors, not bare z-scores: wrist deltas sit at 3-5 deg/frame
# baseline while real failures run 50-150 deg, so anything under ~12 deg is
# ordinary motion no matter how MAD-tight the baseline is.
DETECT_WRIST_MIN_DEG = 15.0
DETECT_FINGER_MIN_DEG = 10.0
DETECT_HF_MIN_DEG = 8.0
DETECT_Z = 4.0
DETECT_HF_Z = 6.0
DETECT_MAX_DEG = 45.0
DETECT_MERGE_GAP = 2
DETECT_MIN_LEN = 3
DETECT_KEEP_PEAK_DEG = 12.0


# --------------------------------------------------------------------------
# pickle I/O

class _NumpyCompatUnpickler(pickle.Unpickler):
    """Read pkls written by numpy >= 2 under Blender 4.5's numpy 1.26.

    numpy 2 pickles (protocol 5, e.g. the user's merged.pkl) reference
    ``numpy._core.numeric._frombuffer``; a clean numpy 1.26 only ships a
    ``numpy._core.multiarray`` stub, so plain ``pickle.load`` failed with
    "No module named 'numpy._core.numeric'" unless PoseCapture's import-time
    shim happened to be loaded.  Fall back to ``numpy.core.*`` for those."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module == "numpy._core" or module.startswith("numpy._core."):
                return super().find_class("numpy.core" + module[len("numpy._core"):], name)
            raise


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return _NumpyCompatUnpickler(handle, encoding="latin1").load()


# --------------------------------------------------------------------------
# rotation helpers (ported from the verified stage2c repair script)

def aa_to_quat(aa):
    aa = np.atleast_2d(np.asarray(aa, dtype=np.float64))
    theta = np.linalg.norm(aa, axis=1)
    half = theta / 2.0
    quat = np.zeros((len(aa), 4))
    quat[:, 0] = np.cos(half)
    small = theta < 1e-9
    quat[:, 1:] = aa * np.where(small, 0.5, np.sin(half) / np.where(small, 1.0, theta))[:, None]
    return quat


def quat_to_aa(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    quat = np.where((quat[:, 0] < 0)[:, None], -quat, quat)
    w = np.clip(quat[:, 0], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    out = np.zeros((len(quat), 3))
    big = s > 1e-9
    out[big] = quat[big, 1:] / s[big, None] * theta[big, None]
    return out


def quat_slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    q1 = np.asarray(q1, dtype=np.float64).copy()
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    theta0 = np.arccos(np.clip(d, -1.0, 1.0))
    q2 = q1 - q0 * d
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta0 * t) + q2 * np.sin(theta0 * t)


def _align(quats, ref):
    return [q if float(np.dot(q, ref)) >= 0.0 else -q for q in quats]


def quat_mean(quats):
    ref = quats[0]
    mean = np.mean(_align(quats, ref), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-12 else ref


def quat_weighted(quats, weights):
    ref = quats[0]
    mean = np.average(_align(quats, ref), axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-12 else ref


def quat_median_filter(quats):
    aligned = _align(quats, quats[0])
    best, best_cost = aligned[0], None
    for candidate in aligned:
        cost = len(aligned) - sum(abs(float(np.dot(candidate, other))) for other in aligned)
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, candidate
    return best


def geodesic_deg(q0, q1):
    return np.degrees(2.0 * np.arccos(np.clip(abs(float(np.dot(q0, q1))), -1.0, 1.0)))


def rotvec_to_mat(rv):
    rv = np.asarray(rv, dtype=np.float64)
    theta = np.linalg.norm(rv, axis=-1, keepdims=True)
    axis = np.where(theta > 1e-12, rv / np.maximum(theta, 1e-12), 0.0)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = np.cos(theta[..., 0]), np.sin(theta[..., 0])
    C = 1.0 - c
    R = np.empty(rv.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    return R


def mat_geo_deg(ra, rb):
    """Geodesic angle between rotation matrix arrays (...,3,3)."""
    tr = np.einsum("...ii->...", np.matmul(np.swapaxes(ra, -1, -2), rb))
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def smooth_aa_path(new_aa, previous_aa):
    """Keep the axis-angle representation continuous with the frame before.

    The alternative representation of a rotation θ·u is (θ − 2π)·u - the SAME
    rotation the long way round.  (−θ·u, used before, is the INVERSE rotation:
    a repair path crossing 180° came out up to 40° wrong.)"""
    for i in range(len(new_aa)):
        prev = previous_aa if i == 0 else new_aa[i - 1]
        if np.linalg.norm(new_aa[i] - prev) > np.pi:
            theta = float(np.linalg.norm(new_aa[i]))
            if theta < 1e-9:
                continue
            alt = new_aa[i] * (1.0 - 2.0 * np.pi / theta)
            if np.linalg.norm(alt - prev) < np.linalg.norm(new_aa[i] - prev):
                new_aa[i] = alt
    return new_aa


# --------------------------------------------------------------------------
# repair

def _reference_frames(n, start, end, exclude):
    """The reference frames on each side of [start, end]: up to REF_WINDOW
    good frames adjacent to the segment, stopping at frames of OTHER marked
    segments (``exclude``).  Before, two bad segments less than REF_WINDOW
    frames apart averaged each other's bad poses into their references
    (measured: 5.6 deg → 44.2 deg error, a 44 deg jump in the gap).  When a
    side has no good frame next to it (segments touching), the nearest good
    frames past the other segment are used - the two then bridge as one.
    With nothing excluded this is exactly the old window."""
    exclude = exclude or ()
    if not exclude:
        return (list(range(max(0, start - REF_WINDOW), start)),
                list(range(end + 1, min(n, end + 1 + REF_WINDOW))))

    def run(frames):
        out = []
        for f in frames:
            if f in exclude or len(out) >= REF_WINDOW:
                break
            out.append(f)
        if not out:                     # touching segments: skip over them
            out = [f for f in frames if f not in exclude][:REF_WINDOW]
        return out

    pre = run(range(start - 1, -1, -1))
    post = run(range(end + 1, n))
    return sorted(pre), post


def repair_quats(quats, start, end, strategy, log, exclude=None):
    """Repair one rotation sequence; returns the new sequence.

    ``exclude``: frames of the other marked segments (never references)."""
    out = np.asarray(quats).copy()
    n = len(quats)
    pre_idx, post_idx = _reference_frames(n, start, end, exclude)
    pre = quats[pre_idx] if pre_idx else quats[0:0]
    post = quats[post_idx] if post_idx else quats[0:0]
    if len(pre) == 0 or end + 1 >= n:
        log["skipped"] = True
        return out
    ref_pre = quat_mean(pre)
    ref_target = quat_mean(post) if len(post) else quats[end + 1]
    log.setdefault("angles", []).append(round(geodesic_deg(ref_pre, ref_target), 1))

    if strategy == "hold":
        span = end - start + 1
        blend = min(6, max(2, span // 3))     # wider tail keeps the exit gentle
        for i in range(start, end + 1):
            pos = i - start
            out[i] = ref_pre if pos < span - blend else quat_slerp(
                ref_pre, ref_target, min(1.0, (pos - (span - blend) + 1) / float(blend)))
    elif strategy == "bridge":
        span = end - start + 1
        for i in range(start, end + 1):
            out[i] = quat_slerp(ref_pre, ref_target, (i - start + 1) / float(span + 1))
    elif strategy == "smooth":
        width = SMOOTH_MARGIN
        kernel = np.exp(-0.5 * (np.arange(-width, width + 1) / max(1.0, width / 1.5)) ** 2)

        def window(i):
            lo, hi = i - width, i + width
            idx = [j for j in range(lo, hi + 1) if 0 <= j < n]
            return idx, np.array([kernel[j - lo] for j in idx])

        med = np.asarray(quats).copy()
        for i in range(start - width, end + width + 1):
            if 0 <= i < n:
                idx, _ = window(i)
                med[i] = quat_median_filter([quats[j] for j in idx])
        for i in range(start, end + 1):
            idx, weights = window(i)
            out[i] = quat_weighted([med[j] for j in idx], weights)
    else:
        raise ValueError(f"未知策略：{strategy}")
    return out


def repair_block(block, start, end, strategy, log, exclude=None):
    """block: (N, 3*k). Repair each 3-axis rotation independently."""
    n_joints = block.shape[1] // 3
    out = block.copy()
    for j in range(n_joints):
        cols = slice(j * 3, j * 3 + 3)
        quats = aa_to_quat(block[:, cols])
        repaired = repair_quats(quats, start, end, strategy, log, exclude=exclude)
        new_aa = quat_to_aa(repaired)[start:end + 1]
        previous = out[start - 1, cols] if start > 0 else new_aa[0]
        out[start:end + 1, cols] = smooth_aa_path(new_aa, previous)
    return out


def _step_scores(block):
    return np.concatenate([[0.0], np.linalg.norm(np.diff(block, axis=0), axis=1)])


def _segment_channels(segment):
    choice = str(segment.get("channel", "both"))
    if choice not in CHANNEL_CHOICES:
        raise ValueError(f"未知通道：{choice}")
    if choice == "wrist":
        return ("wrist",)
    if choice == "fingers":
        return ("finger",)
    return ("wrist", "finger")


def repair_pkl(src_path, dst_path, segments, *, report_path=None):
    """Rewrite hand channels inside marked ranges; everything else byte-stable.

    ``segments``: iterable of mappings with keys
        side     - "left" / "right"
        start    - first pkl frame (0-based)
        end      - last pkl frame, inclusive
        strategy - "hold" | "bridge" | "smooth"
        channel  - "wrist" | "fingers" | "both"   (default "both")

    Returns a report dict.  Writes ``dst_path``; the source is never modified.
    """

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    data = load_pickle(src_path)

    sections = tuple(key for key in data if str(key).startswith("smpl_params_"))
    if not sections:
        raise RuntimeError("pkl 里没有 smpl_params_* 节")
    report = {"source": str(src_path), "output": str(dst_path),
              "segments": [], "checks": {}}

    channel_keys = ("body_pose", "left_hand_pose", "right_hand_pose")
    orig = {
        key: {s: np.array(data[s][key], dtype=np.float64) for s in sections}
        for key in channel_keys
    }
    dtypes = {k: {s: np.asarray(data[s][k]).dtype for s in sections} for k in orig}
    # Everything not repaired must come out identical; snapshot the rest.
    snapshot = {}
    for s in sections:
        for key, value in data[s].items():
            if key in channel_keys:
                continue
            snapshot[(s, key)] = np.asarray(value).copy()

    new = {k: {s: np.array(v[s]) for s in sections} for k, v in orig.items()}
    touched = {"body_pose": {}, "hand_pose": {}}

    segments = list(segments)
    # Frames each (side, channel) has marked bad: a segment's reference window
    # must not average another nearby segment's bad frames in.
    marked: dict = {}
    for segment in segments:
        side_key = "left" if str(segment["side"]).lower().startswith("l") else "right"
        for channel in _segment_channels(segment):
            marked.setdefault((side_key, channel), set()).update(
                range(int(segment["start"]), int(segment["end"]) + 1))

    for segment in segments:
        side = str(segment["side"]).lower()
        side_key = "left" if side.startswith("l") else "right"
        start = int(segment["start"])
        end = int(segment["end"])
        strategy = str(segment.get("strategy", "bridge"))
        if strategy not in STRATEGIES:
            raise ValueError(f"未知策略：{strategy}")
        for channel in _segment_channels(segment):
            if channel == "wrist":
                source_key, channel_block = "body_pose", WRIST_COLS[side_key]
            else:
                source_key = f"{side_key}_hand_pose"
                channel_block = slice(0, None)

            log = {"side": side_key, "channel": channel,
                   "pkl_frames": [start, end], "strategy": strategy}
            base = orig[source_key][sections[0]][:, channel_block]
            before = float(_step_scores(base)[start:end + 1].max())
            others = marked.get((side_key, channel), set()) - set(range(start, end + 1))
            repaired = repair_block(base, start, end, strategy, log, exclude=others)
            for s in sections:
                new[source_key][s][start:end + 1, channel_block] = repaired[start:end + 1]
            after = float(_step_scores(
                new[source_key][sections[0]][:, channel_block])[start:end + 1].max())
            bucket = "body_pose" if channel == "wrist" else "hand_pose"
            touched[bucket].setdefault(side_key, []).append((start, end, channel_block))
            log["step_before"] = round(before, 4)
            log["step_after"] = round(after, 4)
            log["reduction"] = round(1.0 - after / max(before, 1e-9), 3)
            log.pop("angles", None)
            if log.pop("skipped", None):
                log["strategy"] = "skipped_segment_at_edge"
            report["segments"].append(log)

    for s in sections:
        arr = np.array(orig["body_pose"][s], dtype=np.float64)
        for side_key, segs in touched["body_pose"].items():
            for start, end, cols in segs:
                arr[start:end + 1, cols] = new["body_pose"][s][start:end + 1, cols]
        wrist_cols = set()
        for side_key in touched["body_pose"]:
            cols = WRIST_COLS[side_key]
            wrist_cols.update(range(cols.start, cols.stop))
        other_cols = [c for c in range(arr.shape[1]) if c not in wrist_cols]
        # Only wrist columns inside repaired segments may differ.
        if not np.array_equal(arr[:, other_cols], orig["body_pose"][s][:, other_cols]):
            raise RuntimeError(f"{s}: 非手腕 body_pose 列被改动")
        mask = np.ones(len(arr), dtype=bool)
        for segs in touched["body_pose"].values():
            for start, end, _c in segs:
                mask[start:end + 1] = False
        wrist_mask = np.zeros(arr.shape[1], dtype=bool)
        for side_key in touched["body_pose"]:
            cols = WRIST_COLS[side_key]
            wrist_mask[cols.start:cols.stop] = True
        if not np.array_equal(arr[mask][:, wrist_mask],
                              orig["body_pose"][s][mask][:, wrist_mask]):
            raise RuntimeError(f"{s}: 段外手腕列被改动")
        data[s]["body_pose"] = arr.astype(dtypes["body_pose"][s], copy=False)

        for hp_key in ("left_hand_pose", "right_hand_pose"):
            side_key = hp_key.split("_")[0]
            arr = np.array(orig[hp_key][s], dtype=np.float64)
            for start, end, _c in touched["hand_pose"].get(side_key, []):
                arr[start:end + 1] = new[hp_key][s][start:end + 1]
            mask = np.ones(len(arr), dtype=bool)
            for start, end, _c in touched["hand_pose"].get(side_key, []):
                mask[start:end + 1] = False
            if not np.array_equal(arr[mask], orig[hp_key][s][mask]):
                raise RuntimeError(f"{s}/{hp_key} 段外被改动")
            data[s][hp_key] = arr.astype(dtypes[hp_key][s], copy=False)

    for (s, key), original in snapshot.items():
        if not np.array_equal(np.asarray(data[s][key]), original):
            raise RuntimeError(f"{s}/{key} 被改动")

    report["checks"] = {
        "wrist_and_finger_only": True,
        "outside_segments_unchanged": True,
        "other_fields_unchanged": True,
    }
    with dst_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report["written"] = str(dst_path)

    if report_path is not None:
        import json
        with Path(report_path).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1, default=str)
    return report


# --------------------------------------------------------------------------
# detection (candidate segments for a human to review)

def _series_from_raw_hamer(data, hand_index):
    """Per-frame wrist/finger rotation deltas from the raw HaMeR pkl."""
    wrist = np.asarray(data["global_orient"][:, hand_index], dtype=np.float64)   # (T,3,3)
    fingers = np.asarray(data["hand_pose"][:, hand_index], dtype=np.float64)     # (T,15,3,3)
    w_d = np.concatenate([[0.0], mat_geo_deg(wrist[1:], wrist[:-1])])
    f_d = np.concatenate([[0.0], np.mean(mat_geo_deg(fingers[1:], fingers[:-1]), axis=1)])
    return w_d, f_d


def _series_from_merged(data, hand_key):
    """Same deltas from a merged pkl (smoothed rotvec branch)."""
    g = data.get("smpl_params_global") or next(
        v for k, v in data.items() if str(k).startswith("smpl_params_"))
    w = np.asarray(g[f"{hand_key}_hand_global_orient"], dtype=np.float64)
    hp = np.asarray(g[f"{hand_key}_hand_pose"], dtype=np.float64)
    T = len(w)
    w_d = np.concatenate([[0.0], mat_geo_deg(rotvec_to_mat(w[1:]), rotvec_to_mat(w[:-1]))])
    fp = hp.reshape(T, FINGER_JOINTS, 3)
    f_d = np.concatenate([[0.0], np.mean(
        mat_geo_deg(rotvec_to_mat(fp[1:]), rotvec_to_mat(fp[:-1])), axis=1)])
    return w_d, f_d


def _hf_residual(series, width=2):
    """|x[t] - median(x[t-w..t+w])| - the D-class fine-tremor cue."""
    out = np.zeros(len(series))
    for t in range(width, len(series) - width):
        out[t] = abs(series[t] - float(np.median(series[t - width:t + width + 1])))
    return out


def _robust_flags(series, z, floor_deg):
    """Flag frames above a degree threshold that adapts to take noisiness.

    ``med + z*MAD`` rides up on noisy takes and buries mid-size anomalies, so
    the threshold is capped at DETECT_MAX_DEG; ``floor_deg`` keeps calm takes
    from flagging ordinary 2-3 deg transitions.  Candidates are seeds for
    human review - catching the anomaly matters more than its exact edges.
    """
    med = np.median(series[1:])
    mad = np.median(np.abs(series[1:] - med)) * 1.4826
    threshold = min(max(floor_deg, med + z * mad), DETECT_MAX_DEG)
    return series > threshold, med, mad


def _merge_frames(frames, gap=DETECT_MERGE_GAP, min_len=DETECT_MIN_LEN):
    if not len(frames):
        return []
    runs = [[int(frames[0]), int(frames[0])]]
    for f in frames[1:]:
        if f - runs[-1][1] <= gap + 1:
            runs[-1][1] = int(f)
        else:
            runs.append([int(f), int(f)])
    return [(a, b) for a, b in runs if b - a + 1 >= min_len]


def detect_candidates(path):
    """Return per-hand candidate segments with a suggested strategy.

    Prefers the raw HaMeR pkl (un-smoothed rotation matrices); falls back to a
    merged pkl's rotvec channels.  Everything here is a *candidate* - spikes
    (B/C) and tremor (D) show up, pose-wrong-at-normal-rate (A) does not.
    """

    path = Path(path)
    data = load_pickle(path)

    raw = "global_orient" in data and "hand_pose" in data
    result = {}
    for side, hand_key, idx in (("L", "left", 0), ("R", "right", 1)):
        if raw:
            w_d, f_d = _series_from_raw_hamer(data, idx)
        else:
            w_d, f_d = _series_from_merged(data, hand_key)
        hf = _hf_residual(w_d)
        flags_w, _, _ = _robust_flags(w_d, DETECT_Z, DETECT_WRIST_MIN_DEG)
        flags_f, _, _ = _robust_flags(f_d, DETECT_Z, DETECT_FINGER_MIN_DEG)
        flags_hf, _, _ = _robust_flags(hf, DETECT_HF_Z, DETECT_HF_MIN_DEG)
        frames = sorted(set(np.where(flags_w | flags_f | flags_hf)[0].tolist()))

        candidates = []
        for start, end in _merge_frames(frames):
            span = end - start + 1
            peak = float(w_d[start:end + 1].max())
            hf_peak = float(hf[start:end + 1].max())
            if peak < DETECT_KEEP_PEAK_DEG and hf_peak < DETECT_HF_MIN_DEG:
                continue
            if hf_peak >= 15.0 and peak < 40.0:
                kind, strategy = "tremor", "smooth"
            else:
                kind, strategy = "spike", "bridge"
            candidates.append({
                "start": start, "end": end, "kind": kind,
                "strategy": strategy,
                "peak_deg": round(peak, 1),
                "hf_deg": round(hf_peak, 1),
            })
        result[side] = candidates
    result["source_kind"] = "raw_hamer" if raw else "merged"
    return result


# --------------------------------------------------------------------------
# take-dir plumbing

def find_take_pkls(take_dir):
    """Locate the merged pkl (repair target) and raw HaMeR pkl (detect source)."""

    take_dir = Path(take_dir)
    if not take_dir.is_dir():
        return None, None
    merged = take_dir / "hamer" / "merged.pkl"
    if not merged.is_file():
        merged = None
        for cand in sorted(take_dir.glob("*.pkl")):
            if "repaired" not in cand.name:
                merged = cand
                break
    raw = None
    hamer_dir = take_dir / "hamer"
    if hamer_dir.is_dir():
        for cand in sorted(hamer_dir.glob("*_hamer_data.pkl")):
            raw = cand
            break
    return merged, raw
