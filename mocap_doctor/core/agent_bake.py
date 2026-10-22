"""Per-frame world-pose bake for the agent input layer.

The LLM-facing query layer (``agent_query``) works on plain numpy arrays; this
module produces them: every frame, for a fixed bone/point set, the world-space
position and orientation are evaluated through the depsgraph and stored in an
``.npz`` cache keyed by armature name + frame range + bake version.

Blender-dependent; the signal/query layer on top stays pure numpy so it can be
unit tested without a running Blender.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .animation import (
    current_view_layer,
    pose_bone_point_world,
    preserve_scene_frame,
    resolve_frame_range,
    set_scene_frame,
)

BAKE_VERSION = 2   # bump whenever the bake payload schema/bone set changes


def bake_bone_samples(
    scene: Any,
    armature: Any,
    bones: Mapping[str, str],
    points: Mapping[str, tuple[str, str]],
    frame_start: int | None = None,
    frame_end: int | None = None,
    *,
    view_layer: Any | None = None,
) -> dict[str, Any]:
    """Evaluate world pose of every listed bone/point per frame.

    ``bones``  – role → pose-bone name (positions + quaternion baked)
    ``points`` – point id → (bone, "head"|"tail") contact points

    Returns {"frames": (T,) int, "pos": {id: (T,3)}, "quat": {role: (T,4)}}.
    """
    start, end = resolve_frame_range(scene, frame_start, frame_end)
    view_layer = view_layer or current_view_layer()
    count = end - start + 1
    frames = np.arange(start, end + 1, dtype=np.int32)
    pos = {role: np.zeros((count, 3), dtype=np.float64) for role in bones}
    point_pos = {
        name: np.zeros((count, 3), dtype=np.float64) for name in points
    }
    quat = {
        role: np.zeros((count, 4), dtype=np.float64) for role in bones
    }
    basis = {
        role: np.zeros((count, 4), dtype=np.float64) for role in bones
    }
    missing = set()
    missing_roles = set()

    with preserve_scene_frame(scene, view_layer):
        for offset, frame in enumerate(range(start, end + 1)):
            set_scene_frame(scene, frame, view_layer)
            world = armature.matrix_world
            for role, bone_name in bones.items():
                pose_bone = armature.pose.bones.get(bone_name)
                if pose_bone is None:
                    missing.add(bone_name)
                    missing_roles.add(role)
                    continue
                mat = world @ pose_bone.matrix
                pos[role][offset] = mat.translation
                quat[role][offset] = mat.to_quaternion()  # (w, x, y, z)
                basis.setdefault(role, np.zeros((count, 4)))[offset] = \
                    pose_bone.matrix_basis.to_quaternion()
            for name, (bone_name, point) in points.items():
                pose_bone = armature.pose.bones.get(bone_name)
                if pose_bone is None:
                    missing.add(bone_name)
                    continue
                vec = pose_bone.head if point == "head" else pose_bone.tail
                point_pos[name][offset] = world @ vec

    # Quaternion sign continuity: neighbouring quats must share a hemisphere or
    # every downstream velocity signal explodes.
    for role in quat:
        for i in range(1, count):
            if float(np.dot(quat[role][i - 1], quat[role][i])) < 0.0:
                quat[role][i] = -quat[role][i]
    for role in basis:
        for i in range(1, count):
            if float(np.dot(basis[role][i - 1], basis[role][i])) < 0.0:
                basis[role][i] = -basis[role][i]

    for role in missing_roles:
        pos.pop(role, None)
        quat.pop(role, None)
        basis.pop(role, None)
    out = {"frames": frames, "pos": pos, "point_pos": point_pos, "quat": quat,
           "basis": basis,
           "missing_bones": sorted(missing), "frame_start": start,
           "frame_end": end}
    return out


def bake_cache_path(
    data_dir: str | Path,
    armature_name: str,
    frame_start: int,
    frame_end: int,
    tag: str = "",
) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in armature_name)
    suffix = f"_{tag}" if tag else ""
    return (
        Path(data_dir)
        / "agent_cache"
        / f"bake_{safe}_{int(frame_start)}_{int(frame_end)}{suffix}_v{BAKE_VERSION}.npz"
    )


def _hash_action(h: Any, action: Any) -> None:
    if action is None:
        h.update(b"<none>")
        return
    curves = list(getattr(action, "fcurves", None) or ())
    h.update(f"{len(curves)}".encode())
    for fc in curves:
        n = len(fc.keyframe_points)
        co = np.empty(2 * n, dtype=np.float32)
        fc.keyframe_points.foreach_get("co", co)
        h.update(f"{fc.data_path}[{fc.array_index}]:{n}".encode())
        h.update(co.tobytes())


def snapshot_fingerprint(armature: Any) -> str:
    """Content fingerprint of what a snapshot bake depends on - EXCEPT the
    agent delta strips (writes keep the snapshot by design, see agent_bridge
    _STORE_EPOCH): the base action's curves (active action or the mcd_base
    strip's - identical before/after the first write pushes it onto NLA) and
    the animation / static transform of every parent object (global
    correction).  Re-running a wizard step or restoring a checkpoint changes
    it, so describe / validate / get_series / fix_ground stop reading a pose
    that no longer exists.  ~curves×keys bytes hashed, a few ms."""
    h = hashlib.sha1()
    anim = getattr(armature, "animation_data", None)
    base = getattr(anim, "action", None) if anim is not None else None
    if base is None and anim is not None:
        for tr in anim.nla_tracks:
            if tr.name == "mcd_base" and tr.strips:
                base = tr.strips[0].action
                break
    _hash_action(h, base)
    parent = getattr(armature, "parent", None)
    while parent is not None:
        p_anim = getattr(parent, "animation_data", None)
        p_act = getattr(p_anim, "action", None) if p_anim is not None else None
        if p_act is not None:
            _hash_action(h, p_act)
        else:
            h.update(np.asarray(parent.matrix_basis, dtype=np.float32).tobytes())
        parent = getattr(parent, "parent", None)
    return h.hexdigest()[:16]


def save_bake(path: str | Path, bake: Mapping[str, Any],
              fingerprint: str | None = None) -> Path:
    """Atomic: written to a temp file and os.replace'd - a kill mid-write used
    to leave a truncated npz that broke every read tool until deleted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        "frames": bake["frames"],
        "missing_bones": np.asarray(bake.get("missing_bones", []), dtype="U64"),
        "meta": np.asarray([bake["frame_start"], bake["frame_end"]], dtype=np.int64),
    }
    if fingerprint:
        flat["fingerprint"] = np.asarray([str(fingerprint)], dtype="U64")
    for role, arr in bake["quat"].items():
        flat[f"quat::{role}"] = arr
    for role, arr in bake["pos"].items():
        flat[f"pos::{role}"] = arr
    for name, arr in bake["point_pos"].items():
        flat[f"point::{name}"] = arr
    for role, arr in (bake.get("basis") or {}).items():
        flat[f"basis::{role}"] = arr
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **flat)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def load_bake(path: str | Path, fingerprint: str | None = None) -> dict[str, Any] | None:
    """None = bake again: missing, unreadable (moved aside as ``*.corrupt``),
    an old schema, or ``fingerprint`` given and not the one it was baked from."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return _load_bake(path, fingerprint)
    except Exception:
        try:
            path.replace(path.with_name(path.name + ".corrupt"))
        except OSError:
            pass
        return None


def _load_bake(path: Path, fingerprint: str | None) -> dict[str, Any] | None:
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.files)
        # 老缓存没有 basis:: 键 → 手指聚合信号会静默丢失，宁可重烘
        if not any(k.startswith("basis::") for k in keys):
            return None
        if fingerprint is not None:
            stored = str(npz["fingerprint"][0]) if "fingerprint" in keys else ""
            if stored != str(fingerprint):
                return None
        bake = {
            "frames": npz["frames"],
            "missing_bones": [str(v) for v in npz["missing_bones"]],
            "frame_start": int(npz["meta"][0]),
            "frame_end": int(npz["meta"][1]),
            "pos": {},
            "point_pos": {},
            "quat": {},
            "basis": {},
        }
        for key in keys:
            if key.startswith("quat::"):
                bake["quat"][key[6:]] = npz[key]
            elif key.startswith("pos::"):
                bake["pos"][key[5:]] = npz[key]
            elif key.startswith("point::"):
                bake["point_pos"][key[7:]] = npz[key]
            elif key.startswith("basis::"):
                bake["basis"][key[7:]] = npz[key]
    return bake


def sample_fcurve_values(
    action: Any,
    data_path: str,
    index: int,
    frame_start: int,
    frame_end: int,
) -> np.ndarray | None:
    """Evaluate one F-curve channel per frame (no scene playback needed)."""
    from .animation import get_fcurve

    fcurve = get_fcurve(action, data_path, int(index))
    if fcurve is None:
        return None
    frames = np.arange(int(frame_start), int(frame_end) + 1)
    return np.asarray([float(fcurve.evaluate(f)) for f in frames])


def write_fcurve_values(
    action: Any,
    data_path: str,
    index: int,
    frame_start: int,
    values: Sequence[float],
    *,
    group: str = "agent",
) -> int:
    """Dense per-frame write of values into (creating if needed) an F-curve.

    P9: a fresh (empty) curve - every agent delta strip - is filled in one
    keyframe_points.add(n) + foreach_set instead of n RNA inserts (19-bone
    copy: 20-30 ms → ~1 ms); same keys (co, LINEAR), checked by the golden."""
    from .animation import ensure_fcurve, keyframe_map, set_fcurve_value

    fcurve = ensure_fcurve(action, data_path, int(index), group=group)
    n = len(values)
    if n and len(fcurve.keyframe_points) == 0:
        points = fcurve.keyframe_points
        points.add(n)
        co = np.empty(2 * n, dtype=np.float64)
        co[0::2] = int(frame_start) + np.arange(n)
        co[1::2] = np.asarray(values, dtype=np.float64)
        points.foreach_set("co", co)
        for key in points:
            key.interpolation = "LINEAR"
        fcurve.update()
        return n
    cache = keyframe_map(fcurve)
    for offset, value in enumerate(values):
        set_fcurve_value(fcurve, int(frame_start) + offset, float(value), cache=cache)
    fcurve.update()
    return len(values)


def write_quat_values(
    action: Any,
    data_path: str,
    frame_start: int,
    quats: np.ndarray,
    *,
    group: str = "agent",
) -> int:
    """Write (T,4) wxyz quats to a rotation_quaternion channel, sign-continuous."""
    quats = np.asarray(quats, dtype=np.float64)
    out = quats.copy()
    # sign-continue against the previous frame inside the write block
    for i in range(1, len(out)):
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] = -out[i]
    for axis in range(4):
        write_fcurve_values(
            action, data_path, axis, frame_start, out[:, axis], group=group
        )
    return len(out)
