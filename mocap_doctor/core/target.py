"""Validated repairs for the arue Teto model and its MMR control rig."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import statistics
from typing import Any

try:
    import bpy  # type: ignore
    from mathutils import Euler, Matrix, Quaternion, Vector  # type: ignore
except ImportError:  # pragma: no cover - module is executed inside Blender.
    bpy = None
    Euler = None
    Matrix = None
    Quaternion = None
    Vector = None

from .animation import (
    EPSILON,
    _fcurves,
    bone_path,
    add_missing_keys,
    cache_fcurve_values,
    current_view_layer,
    ensure_action,
    ensure_fcurve,
    get_action,
    get_fcurve,
    get_pose_quaternion,
    insert_pose_rotation_key,
    keyframe_map,
    limit_frame_delta,
    pose_bone_point_world,
    pose_bone_world_location,
    preserve_scene_frame,
    resolve_frame_range,
    set_fcurve_value,
    set_linear_for_paths,
    set_pose_quaternion,
    set_scene_frame,
    smooth_frame_values,
    update_action,
)
from .ranges import normalize_ranges
# gvhmr_source_prefix is only still used to keep the retired GVHMR profile
# importable; the fixes themselves read the FlowHMR native bone names.
from ..presets import gvhmr_source_prefix  # noqa: F401  (deprecated compat)


DEFAULT_FOOT_IK = {"L": "foot_ik.L", "R": "foot_ik.R"}
# The MMD ankle bone spans heel to ball of foot, so its head and tail are the
# closest thing to a sole the fixed Teto skeleton offers for measuring where a
# planted foot actually is.  The toe bone is deliberately not used: MMR does not
# drive it before MMD Visual Bake, so it would report a frozen rest position.
DEFAULT_GROUND_POINTS = {
    "L": (("足首.L", "head"), ("足首.L", "tail")),
    "R": (("足首.R", "head"), ("足首.R", "tail")),
}
# Airborne spans are rebuilt ballistically; gravity is what turns the marked
# flight time into a jump height, so it is a physical constant, not a taste knob.
GRAVITY = 9.81
# 0.8 s of flight is already a 78 cm jump (g*T^2/8); anything longer is not a
# jump, so it falls back to interpolating the neighbouring pins.
BALLISTIC_MAX_FRAMES = 24
# A fuse, not a judgement: only a mis-marked range can ask for more than this.
MAX_CORRECTION = 0.5
# Grounded frames whose foot sits higher than this are reported as suspected
# missing airborne marks (informational only - the pin still applies).
SUSPECT_AIRBORNE_HEIGHT = 0.06
# Pelvis reach pass (inside foot_lock, before the ankle lock): re-solve the
# body height per frame so the model keeps the SOURCE's leg reach ratio
# |hip-ankle| / leg_length - which for a two-bone chain is the knee angle by
# geometric identity.  Teto's legs are proportionally longer than the capture
# subject's, so retargeted poses straighten the knee beyond the source and
# lift planted feet off the floor; lowering the torso onto the correct reach
# restores both.  Written onto torso_root (the bone the retarget actually
# animates for body translation; torso/hips/root carry no location fcurves).
PELVIS_CORRECTION_MAX = 0.025  # metres of torso drop/lift per frame
PELVIS_SMOOTH_FRAMES = 9  # low-pass half-window for the correction curve
PELVIS_DEADBAND = 0.002  # corrections below this are noise, skip them
DEFAULT_PELVIS_BONE = "torso_root"
# FlowHMR 原生骨架的腿骨名（Q14：只为 FlowHMR 服务，直接读原生名，不再兼容
# GVHMR 的 f_avg_/m_avg_ 前缀骨架 —— 那个前缀在这条流水线上已经不存在了，
# 硬编码它才是 _settle_pelvis_for_reach 一直静默跳过的原因）。
SOURCE_LEG_BONES = {
    "L": ("L_Hip", "L_Knee", "L_Ankle"),
    "R": ("R_Hip", "R_Knee", "R_Ankle"),
}
MODEL_LEG_BONES = {
    "L": ("足.L", "ひざ.L", "足首.L"),
    "R": ("足.R", "ひざ.R", "足首.R"),
}
# Closed-loop stabilizer converge thresholds for the post-write verification
# measure.  They gate the reported state, not the write: a residual above
# them after one pass is a reachability floor, not a retryable error.
STABILIZE_POS_TOLERANCE = 0.0005  # metres of ankle translation error
STABILIZE_ROT_TOLERANCE_DEG = 0.5  # degrees of ankle rotation error

# The loop measures the MMD-side ankle: the constrained bone the mesh and the
# bake actually follow.
DEFAULT_ANKLE_BONES = {"L": "足首.L", "R": "足首.R"}
DEFAULT_EXCLUDED_MESH_KEYWORDS = (
    "ground",
    "plane",
    "video",
    "VID",
    "rigid",
    "Rigid",
    "joints",
    "Joint",
    "Camera",
    "Light",
)


# Root pivot (silently applied when the retarget milestone is recorded).  The
# ARP bmap maps the source pelvis - location AND rotation - onto torso_root,
# whose head sits at the MMD センター height: on Teto that is ~0.39 m BELOW the
# hip joints, while the SMPL-X pelvis joint is ~0.09 m ABOVE them.  The same
# pelvis tilt about the low pivot swings the hips against their own sway:
# 0001-0999 frames 151-277 measure 0.19 m of hip sway on the source and 0.07 m
# (out of phase) on Teto.  The fix keeps every rotation and shifts torso_root so
# the rotation acts about the hips bone head; child pose = rest + R0 loc +
# R0 R(q) S d, so loc += d - R(q) S d with d = hips head - torso_root head in
# torso_root's rest frame.  Foot IK targets are untouched (separate chains).
ROOT_PIVOT_BONE = "hips"
ROOT_PIVOT_MARK = "mcd_root_pivot_fix"


def fix_root_pivot(
    armature: Any,
    action: Any | None = None,
    *,
    root_bone: str = DEFAULT_PELVIS_BONE,
    pivot_bone: str = ROOT_PIVOT_BONE,
    force: bool = False,
) -> dict[str, Any]:
    """Re-pivot the retarget's baked root rotation about ``pivot_bone``'s head.

    Idempotent: the action is marked and a second call is a no-op unless
    ``force``.  Returns a report; ``skipped`` explains why nothing was written
    (no root animation, unexpected bone layout, already applied).
    """
    action = action if action is not None else get_action(armature, required=False)
    report: dict[str, Any] = {"operation": "fix_root_pivot", "armature": getattr(armature, "name", ""),
                              "root_bone": root_bone, "pivot_bone": pivot_bone}
    if action is None:
        report["skipped"] = "no action"
        return report
    report["action"] = action.name
    previous = action.get(ROOT_PIVOT_MARK)
    if previous and not force:
        report["skipped"] = "already applied"
        report["previous"] = json.loads(previous) if isinstance(previous, str) else str(previous)
        return report
    bones = armature.data.bones
    if root_bone not in bones or pivot_bone not in bones:
        report["skipped"] = "bones missing"
        return report
    root, pivot = bones[root_bone], bones[pivot_bone]
    chain, bone = [], pivot.parent
    while bone is not None and bone.name != root_bone:
        chain.append(bone.name)
        bone = bone.parent
    if bone is None:
        report["skipped"] = f"{pivot_bone} is not under {root_bone}"
        return report
    animated = sorted({name for name in [pivot_bone, *chain]
                       for fcurve in _fcurves(action) if fcurve.data_path.startswith(bone_path(name) + ".")})
    if animated:
        # The fix assumes a rigid offset between the two heads.
        report["skipped"] = f"bones between root and pivot are animated: {animated}"
        return report
    rest = root.matrix_local.to_3x3().normalized()
    offset = rest.transposed() @ (pivot.head_local - root.head_local)
    loc = [get_fcurve(action, bone_path(root_bone, "location"), i) for i in range(3)]
    quat = [get_fcurve(action, bone_path(root_bone, "rotation_quaternion"), i) for i in range(4)]
    scale = [get_fcurve(action, bone_path(root_bone, "scale"), i) for i in range(3)]
    if any(fc is None for fc in quat):
        report["skipped"] = "no quaternion rotation on root"
        return report
    frames = sorted({int(round(key.co.x)) for fc in quat + loc if fc is not None for key in fc.keyframe_points})
    deltas = []
    for frame in frames:
        q = Quaternion([fc.evaluate(frame) for fc in quat]).normalized()
        s = Vector([fc.evaluate(frame) if fc is not None else 1.0 for fc in scale])
        scaled = Vector((offset.x * s.x, offset.y * s.y, offset.z * s.z))
        deltas.append(offset - q.to_matrix() @ scaled)
    pose_root = armature.pose.bones.get(root_bone)
    for axis in range(3):
        fcurve = loc[axis]
        if fcurve is None:
            fcurve = ensure_fcurve(action, bone_path(root_bone, "location"), axis, group=root_bone)
            base = float(pose_root.location[axis]) if pose_root is not None else 0.0
            for frame in frames:
                fcurve.keyframe_points.insert(frame, base, options={"FAST"})
            loc[axis] = fcurve
        before = {frame: float(fcurve.evaluate(frame)) for frame in frames}
        cache = keyframe_map(fcurve)
        for frame, delta in zip(frames, deltas):
            set_fcurve_value(fcurve, frame, before[frame] + float(delta[axis]), cache=cache)
    update_action(action)
    magnitude = [delta.length for delta in deltas]
    report.update({
        "frames": len(frames),
        "pivot_offset_m": [round(float(v), 4) for v in (pivot.head_local - root.head_local)],
        "max_shift_m": round(max(magnitude), 4) if magnitude else 0.0,
        "mean_shift_m": round(sum(magnitude) / len(magnitude), 4) if magnitude else 0.0,
    })
    action[ROOT_PIVOT_MARK] = json.dumps({k: report[k] for k in ("root_bone", "pivot_bone", "frames",
                                                                  "pivot_offset_m", "max_shift_m")})
    return report


# Upper-body follow (silently applied at the retarget milestone, right after
# the root pivot fix).  ARP matches every bone's WORLD orientation (measured on
# 0001-0999: the offset Rs^T Rt drifts 0.00 deg on all spine/neck/arm pairs),
# but the chains differ in where the rotations act.  The source counter-leans
# its hip sway with Pelvis+Spine1 (23 deg peak-to-peak) over 0.23 m of lumbar
# ABOVE its pelvis joint; on Teto, Spine1's orientation lands on spine_fk, which
# swings the pelvis segment BELOW the hips head, and the upper chain starts AT
# the hips head with Spine2's orientation (12 deg).  Frames 151-277: source
# shoulders sway 0.077 m under a 0.191 m hip sway, Teto's 0.149 m.
# The fix swings spine_fk.001 about its head (= hips head; everything above it
# rides along, the hips and legs do not) so Teto's shoulder-mid sits where the
# source puts it relative to the hips-mid: the lateral offset 1:1 - hips and
# feet are mapped 1:1, so "the shoulders stay put" is absolute - and the
# forward offset scaled by the torso length ratio, i.e. the source's forward
# lean angle (absolute would bend Teto's 0.71x torso ~1.4x as far in bows).
# A lateral target scaled like the forward one measures 0.108 m of shoulder
# sway instead of 0.087 m: Teto's torso is too short for the angle alone.
UPPER_BODY_MARK = "mcd_upper_body_follow"
UPPER_BODY_SWING_BONE = "spine_fk.001"
UPPER_BODY_RIG_HIPS = ("ORG-thigh.L", "ORG-thigh.R")
UPPER_BODY_RIG_SHOULDERS = ("ORG-upper_arm.L", "ORG-upper_arm.R")
# FlowHMR native joint names (Q14): read straight off the source rig, no prefix.
UPPER_BODY_SOURCE_HIPS = ("L_Hip", "R_Hip")
UPPER_BODY_SOURCE_SHOULDERS = ("L_Shoulder", "R_Shoulder")
# The bones the source rig must have to count as a FlowHMR native skeleton.
FLOWHMR_SOURCE_BONES = ("Pelvis", "L_Hip", "R_Hip", "L_Shoulder", "R_Shoulder")
# Gaussian sigma in frames: the 151-277 sway has a ~20-frame period, which a
# 1.5-frame sigma keeps at ~90%.
UPPER_BODY_SMOOTH_SIGMA = 1.5
# A fuse, not a target: 0001-0999 peaks at 14 deg.
UPPER_BODY_MAX_SWING_DEG = 20.0
# Source torso lean (hips-mid -> shoulder-mid from vertical) over which the
# correction fades out: floor work and handstands are not hip isolation.
UPPER_BODY_UPRIGHT_FADE_DEG = (45.0, 60.0)


def _rotate_rows(rotvec: Any, vectors: Any) -> Any:
    """Rodrigues: rotate each row of ``vectors`` by the matching rotation vector."""
    import numpy as np

    angle = np.linalg.norm(rotvec, axis=1)
    axis = rotvec / np.maximum(angle, 1e-12)[:, None]
    cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
    along = np.einsum("ij,ij->i", axis, vectors)[:, None]
    return vectors * cos + np.cross(axis, vectors) * sin + axis * along * (1.0 - cos)


def upper_body_swing(
    pivot: Any,
    shoulders: Any,
    hip_l: Any,
    hip_r: Any,
    src_shoulders: Any,
    src_hip_l: Any,
    src_hip_r: Any,
    *,
    torso_ratio: float | None = None,
    lateral_scale: float = 1.0,
    smooth_sigma: float = UPPER_BODY_SMOOTH_SIGMA,
    max_swing_deg: float = UPPER_BODY_MAX_SWING_DEG,
    upright_fade_deg: Sequence[float] = UPPER_BODY_UPRIGHT_FADE_DEG,
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> dict[str, Any]:
    """Per-frame world swing about ``pivot`` that puts the shoulders where the source has them.

    Inputs are (F, 3) world positions per frame: Teto's swing-bone head
    (``pivot``), shoulder-mid and hip joints; the source's shoulder-mid and hip
    joints.  Each skeleton gets its own horizontal lateral axis (its hip line)
    and forward axis (up x lateral).  The target offset of the shoulder-mid
    from the hips-mid is the source's lateral component times
    ``lateral_scale`` and its forward component times ``torso_ratio``
    (Teto/source hips-mid -> shoulder-mid length, measured when None); the
    vertical follows from keeping the shoulder-mid's distance to the pivot.
    The minimal rotation is smoothed (Gaussian, frames), faded out where the
    source torso leans past ``upright_fade_deg``, zeroed where a hip line is
    near vertical, and clamped.  Pure numpy, so it runs without Blender.

    Returns ``rotvec`` ((F, 3) radians, world axes) plus JSON-ready stats.
    """
    import numpy as np

    def rows(values: Any) -> Any:
        return np.asarray(values, dtype=float).reshape(-1, 3)

    def dot(a: Any, b: Any) -> Any:
        return np.einsum("ij,ij->i", a, b)

    piv, sho, hl, hr = rows(pivot), rows(shoulders), rows(hip_l), rows(hip_r)
    src_sho, src_hl, src_hr = rows(src_shoulders), rows(src_hip_l), rows(src_hip_r)
    count = len(piv)
    up_axis = np.asarray(up, dtype=float)
    up_axis = up_axis / np.linalg.norm(up_axis)

    def axes(left: Any, right: Any) -> tuple[Any, Any, Any]:
        line = right - left
        flat = line - np.outer(line @ up_axis, up_axis)
        length = np.linalg.norm(flat, axis=1)
        usable = length > 0.5 * np.maximum(np.linalg.norm(line, axis=1), 1e-9)
        lateral = flat / np.maximum(length, 1e-9)[:, None]
        return lateral, np.cross(up_axis, lateral), usable

    lat_t, fwd_t, ok_t = axes(hl, hr)
    lat_s, fwd_s, ok_s = axes(src_hl, src_hr)
    hip, src_hip = (hl + hr) / 2.0, (src_hl + src_hr) / 2.0
    offset, src_offset = sho - hip, src_sho - src_hip
    if torso_ratio is None:
        torso_ratio = float(np.linalg.norm(offset, axis=1).mean()
                            / max(float(np.linalg.norm(src_offset, axis=1).mean()), 1e-9))
    want_lat = float(lateral_scale) * dot(src_offset, lat_s)
    want_fwd = float(torso_ratio) * dot(src_offset, fwd_s)
    arm = sho - piv
    radius = np.maximum(np.linalg.norm(arm, axis=1), 1e-9)
    base = piv - hip
    u_lat = (want_lat - dot(base, lat_t)) / radius
    u_fwd = (want_fwd - dot(base, fwd_t)) / radius
    reach = np.hypot(u_lat, u_fwd)
    unreachable = reach > 0.98
    shrink = np.where(unreachable, 0.98 / np.maximum(reach, 1e-9), 1.0)
    u_lat, u_fwd = u_lat * shrink, u_fwd * shrink
    u_up = np.sqrt(np.clip(1.0 - u_lat ** 2 - u_fwd ** 2, 0.0, 1.0))
    want = u_lat[:, None] * lat_t + u_fwd[:, None] * fwd_t + u_up[:, None] * up_axis
    current = arm / radius[:, None]
    axis = np.cross(current, want)
    sin = np.linalg.norm(axis, axis=1)
    angle = np.arctan2(sin, dot(current, want))
    rotvec = np.where(sin[:, None] > 1e-12, axis / np.maximum(sin, 1e-12)[:, None] * angle[:, None], 0.0)

    lean = np.degrees(np.arccos(np.clip(
        (src_offset @ up_axis) / np.maximum(np.linalg.norm(src_offset, axis=1), 1e-9), -1.0, 1.0)))
    low, high = (float(v) for v in upright_fade_deg)
    weight = np.clip((high - lean) / max(high - low, 1e-9), 0.0, 1.0)
    weight[~(ok_t & ok_s)] = 0.0
    rotvec = rotvec * weight[:, None]
    if smooth_sigma > 0 and count > 1:
        reach_k = int(math.ceil(3.0 * float(smooth_sigma)))
        kernel = np.exp(-0.5 * (np.arange(-reach_k, reach_k + 1) / float(smooth_sigma)) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(rotvec, ((reach_k, reach_k), (0, 0)), mode="edge")
        rotvec = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(3)], axis=1)
    magnitude = np.linalg.norm(rotvec, axis=1)
    limit = math.radians(float(max_swing_deg))
    clamped = magnitude > limit
    rotvec = rotvec * np.where(clamped, limit / np.maximum(magnitude, 1e-12), 1.0)[:, None]

    new_offset = piv + _rotate_rows(rotvec, arm) - hip

    def spread(values: Any) -> dict[str, float]:
        values = np.abs(values) * 100.0
        return {"p50": round(float(np.median(values)), 2), "p95": round(float(np.percentile(values, 95)), 2)}

    degrees = np.degrees(np.linalg.norm(rotvec, axis=1))
    return {
        "rotvec": rotvec,
        "torso_ratio": round(float(torso_ratio), 4),
        "lateral_scale": round(float(lateral_scale), 4),
        "swing_deg": {"p50": round(float(np.median(degrees)), 2), "p95": round(float(np.percentile(degrees, 95)), 2),
                      "max": round(float(degrees.max()), 2)},
        "clamped_frames": int(clamped.sum()),
        "unreachable_frames": int(unreachable.sum()),
        "faded_frames": int((weight < 1.0).sum()),
        "degenerate_frames": int((~(ok_t & ok_s)).sum()),
        "shoulder_lateral_err_cm": {"before": spread(dot(offset, lat_t) - want_lat),
                                    "after": spread(dot(new_offset, lat_t) - want_lat)},
        "shoulder_forward_err_cm": {"before": spread(dot(offset, fwd_t) - want_fwd),
                                    "after": spread(dot(new_offset, fwd_t) - want_fwd)},
    }


def _is_animated(obj: Any) -> bool:
    """An active action or at least one playing NLA strip."""
    data = getattr(obj, "animation_data", None)
    if data is None:
        return False
    if data.action is not None:
        return True
    return any(not track.mute and any(not strip.mute for strip in track.strips) for track in data.nla_tracks)


def _upper_body_source(armature: Any, source_armature: Any | None) -> tuple[Any | None, str | None]:
    """The wizard's source rig, else the one animated FlowHMR-native armature in the scene."""
    if source_armature is not None:
        return source_armature, None
    scene = getattr(bpy.context, "scene", None)
    objects = scene.objects if scene is not None else bpy.data.objects
    candidates = [obj for obj in objects
                  if obj is not armature and _is_flowhmr_source(obj) and _is_animated(obj)]
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, "no animated FlowHMR source armature (Pelvis/L_Hip/R_Hip/L_Shoulder/R_Shoulder) in the scene"
    return None, f"several FlowHMR source armatures, set the project's source: {sorted(o.name for o in candidates)}"


def _is_flowhmr_source(armature: Any) -> bool:
    """True for the FlowHMR native skeleton (Q14: the only frontend we serve)."""

    if armature is None or getattr(armature, "type", "") != "ARMATURE":
        return False
    names = {bone.name for bone in armature.data.bones}
    return all(name in names for name in FLOWHMR_SOURCE_BONES)


def fix_upper_body_follow(
    armature: Any,
    source_armature: Any | None = None,
    action: Any | None = None,
    *,
    swing_bone: str = UPPER_BODY_SWING_BONE,
    smooth_sigma: float = UPPER_BODY_SMOOTH_SIGMA,
    max_swing_deg: float = UPPER_BODY_MAX_SWING_DEG,
    force: bool = False,
) -> dict[str, Any]:
    """Swing the retarget's upper spine so the shoulders follow the source's position.

    Samples every keyed frame of ``swing_bone`` (Teto and source evaluated
    together), computes ``upper_body_swing`` and writes it into the swing
    bone's quaternion keys as a world rotation about its head:
    q' = q (Rw^T Q Rw).  Only those four curves change; the hips, legs, feet
    and torso_root keys do not.  The sampled pose must be this action's, so
    the action has to be the armature's active one with no NLA strips playing.
    Idempotent via an action mark; returns a report whose ``skipped`` explains
    why nothing was written (missing source/bones, layout, already applied).
    """
    action = action if action is not None else get_action(armature, required=False)
    report: dict[str, Any] = {"operation": "fix_upper_body_follow", "armature": getattr(armature, "name", ""),
                              "swing_bone": swing_bone}
    if action is None:
        report["skipped"] = "no action"
        return report
    report["action"] = action.name
    previous = action.get(UPPER_BODY_MARK)
    if previous and not force:
        report["skipped"] = "already applied"
        report["previous"] = json.loads(previous) if isinstance(previous, str) else str(previous)
        return report
    animation_data = getattr(armature, "animation_data", None)
    if getattr(animation_data, "action", None) is not action:
        report["skipped"] = "action is not the armature's active action (the fix samples the evaluated pose)"
        return report
    playing = [track.name for track in animation_data.nla_tracks
               if not track.mute and any(not strip.mute for strip in track.strips)]
    if playing:
        report["skipped"] = f"NLA tracks play on top of the action: {playing}"
        return report
    source, why = _upper_body_source(armature, source_armature)
    if source is None:
        report["skipped"] = why
        return report
    report["source"] = source.name
    if not _is_flowhmr_source(source):
        report["skipped"] = (
            "source is not the FlowHMR native skeleton (need "
            + "/".join(FLOWHMR_SOURCE_BONES) + ")"
        )
        return report
    if not _is_animated(source):
        report["skipped"] = "source armature has no animation (a static source would freeze the upper body)"
        return report
    src_hips = UPPER_BODY_SOURCE_HIPS
    src_shoulders = UPPER_BODY_SOURCE_SHOULDERS
    missing = [name for name in (*src_hips, *src_shoulders) if name not in source.data.bones]
    if missing:
        report["skipped"] = f"source bones missing: {missing}"
        return report
    bones = armature.data.bones
    missing = [name for name in (swing_bone, *UPPER_BODY_RIG_HIPS, *UPPER_BODY_RIG_SHOULDERS) if name not in bones]
    if missing:
        report["skipped"] = f"bones missing: {missing}"
        return report

    def under(name: str) -> bool:
        bone = bones[name].parent
        while bone is not None:
            if bone.name == swing_bone:
                return True
            bone = bone.parent
        return False

    if not all(under(name) for name in UPPER_BODY_RIG_SHOULDERS) or any(under(name) for name in UPPER_BODY_RIG_HIPS):
        report["skipped"] = f"unexpected layout: shoulders must hang under {swing_bone}, hips must not"
        return report
    pose_bone = armature.pose.bones[swing_bone]
    quat = [get_fcurve(action, bone_path(swing_bone, "rotation_quaternion"), i) for i in range(4)]
    if pose_bone.rotation_mode != "QUATERNION" or any(fc is None for fc in quat):
        report["skipped"] = f"no quaternion rotation keys on {swing_bone}"
        return report
    frames = sorted({int(round(key.co.x)) for fc in quat for key in fc.keyframe_points})
    if not frames:
        report["skipped"] = f"no keys on {swing_bone}"
        return report

    import numpy as np

    scene = bpy.context.scene
    view_layer = current_view_layer()
    (hip_l, hip_r), (sho_l, sho_r) = UPPER_BODY_RIG_HIPS, UPPER_BODY_RIG_SHOULDERS
    samples: dict[str, list[Any]] = {key: [] for key in ("piv", "hl", "hr", "sho", "s_hl", "s_hr", "s_sho")}
    world_rot: list[Any] = []
    with preserve_scene_frame(scene, view_layer):
        for frame in frames:
            set_scene_frame(scene, frame, view_layer)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            rig_eval = armature.evaluated_get(depsgraph)
            src_eval = source.evaluated_get(depsgraph)
            rig_world, src_world = rig_eval.matrix_world, src_eval.matrix_world
            rig_pose, src_pose = rig_eval.pose.bones, src_eval.pose.bones
            swing_matrix = rig_world @ rig_pose[swing_bone].matrix
            world_rot.append(swing_matrix.to_3x3().normalized().to_quaternion())
            samples["piv"].append(tuple(swing_matrix.translation))
            samples["hl"].append(tuple(rig_world @ rig_pose[hip_l].head))
            samples["hr"].append(tuple(rig_world @ rig_pose[hip_r].head))
            samples["sho"].append(tuple((rig_world @ rig_pose[sho_l].head + rig_world @ rig_pose[sho_r].head) / 2.0))
            samples["s_hl"].append(tuple(src_world @ src_pose[src_hips[0]].head))
            samples["s_hr"].append(tuple(src_world @ src_pose[src_hips[1]].head))
            samples["s_sho"].append(tuple((src_world @ src_pose[src_shoulders[0]].head
                                           + src_world @ src_pose[src_shoulders[1]].head) / 2.0))
    arrays = {key: np.array(values, dtype=float) for key, values in samples.items()}
    swing = upper_body_swing(arrays["piv"], arrays["sho"], arrays["hl"], arrays["hr"],
                             arrays["s_sho"], arrays["s_hl"], arrays["s_hr"],
                             smooth_sigma=smooth_sigma, max_swing_deg=max_swing_deg)
    rotvec = swing.pop("rotvec")
    # read every frame before writing any: a frame keyed on only some channels
    # would otherwise interpolate between keys this loop already moved
    originals = [Quaternion([float(fc.evaluate(frame)) for fc in quat]).normalized() for frame in frames]
    caches = [keyframe_map(fc) for fc in quat]
    local_before, local_after = [], []
    for index, frame in enumerate(frames):
        old = originals[index]
        vec = Vector([float(v) for v in rotvec[index]])
        turn = Quaternion(vec.normalized(), vec.length) if vec.length > 1e-12 else Quaternion()
        rw = world_rot[index]
        new = (old @ (rw.conjugated() @ turn @ rw)).normalized()
        if new.dot(old) < 0.0:
            new.negate()
        for axis in range(4):
            set_fcurve_value(quat[axis], frame, float(new[axis]), cache=caches[axis])
        local_before.append(min(math.degrees(old.angle), 360.0 - math.degrees(old.angle)))
        local_after.append(min(math.degrees(new.angle), 360.0 - math.degrees(new.angle)))
    update_action(action)
    report.update({
        "frames": len(frames),
        "frame_range": [frames[0], frames[-1]],
        "source_bones": {"hips": list(src_hips), "shoulders": list(src_shoulders)},
        **swing,
        "swing_bone_local_deg_max": {"before": round(max(local_before), 2), "after": round(max(local_after), 2)},
        "root_pivot_fixed": bool(action.get(ROOT_PIVOT_MARK)),
    })
    action[UPPER_BODY_MARK] = json.dumps({key: report[key] for key in (
        "swing_bone", "source", "frames", "torso_ratio", "swing_deg", "clamped_frames", "shoulder_lateral_err_cm")})
    return report


def ensure_global_correction_empty(
    name: str = "teto_global_correction",
    *,
    collection: Any | None = None,
) -> Any:
    """Find or create the outer correction Empty; no file is saved."""

    if bpy is None:
        raise RuntimeError("global correction must run inside Blender")
    existing = bpy.data.objects.get(name)
    if existing is not None:
        return existing
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.25
    (collection or bpy.context.collection).objects.link(empty)
    return empty


def _parent_keep_world(child: Any, parent: Any) -> None:
    world = child.matrix_world.copy()
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    child.matrix_world = world


def apply_global_correction(
    model_root: Any,
    rig: Any,
    correction: Any,
    *,
    rotation_degrees: Sequence[float] = (-4.2, 3.7, 0.0),
) -> dict[str, Any]:
    """Parent Teto and MMR under one Empty, then apply the tested correction."""

    if len(rotation_degrees) != 3:
        raise ValueError("rotation_degrees must contain X, Y, and Z")
    children = (model_root, rig)
    world_matrices = {child.name: child.matrix_world.copy() for child in children}

    # Establish a neutral parent while preserving each child's current world pose.
    correction.location = (0.0, 0.0, 0.0)
    correction.rotation_euler = (0.0, 0.0, 0.0)
    correction.scale = (1.0, 1.0, 1.0)
    for child in children:
        if child.parent != correction:
            _parent_keep_world(child, correction)
        child.matrix_world = world_matrices[child.name]

    radians = tuple(math.radians(float(value)) for value in rotation_degrees)
    correction.rotation_mode = "XYZ"
    correction.rotation_euler = radians
    return {
        "operation": "apply_teto_global_correction",
        "model_root": model_root.name,
        "rig": rig.name,
        "correction": correction.name,
        "rotation_degrees": [float(value) for value in rotation_degrees],
    }


def _lerp_angle(current: float, target: float, strength: float) -> float:
    difference = (target - current + math.pi) % (2.0 * math.pi) - math.pi
    return current + difference * strength


def _action_rotation_sampler(rig: Any, bone_names: Sequence[str]) -> Any:
    """frame -> pose rotation read straight off the active Action's F-curves,
    or None when that would not equal what a frame_set leaves in the pose
    properties (then the caller keeps its frame sweep).

    Equal when: the Action is evaluated alone at full strength (no tweak mode,
    influence 1, REPLACE, no unmuted NLA strip underneath that could matter),
    no driver touches the channel, and an unanimated (or muted) channel keeps
    its static property value.  Euler and quaternion bones only - the values
    and the mathutils conversions are exactly get_pose_quaternion's."""

    anim = getattr(rig, "animation_data", None)
    action = getattr(anim, "action", None) if anim is not None else None
    if action is None or getattr(anim, "use_tweak_mode", False):
        return None
    if float(getattr(anim, "action_influence", 1.0)) != 1.0:
        return None
    if str(getattr(anim, "action_blend_type", "REPLACE")) != "REPLACE":
        return None
    if getattr(anim, "use_nla", True) and any(
        not track.mute and len(track.strips) > 0 for track in anim.nla_tracks
    ):
        return None
    slots = getattr(action, "slots", None)
    if slots is not None and (
        len(slots) != 1 or getattr(anim, "action_slot", None) is None
    ):
        # legacy action.fcurves is the first slot's channelbag: with several
        # slots (or none bound) it may not be what animates this rig
        return None
    channels: dict[str, tuple[str, list]] = {}
    for name in bone_names:
        pose_bone = rig.pose.bones[name]
        mode = str(pose_bone.rotation_mode)
        if mode == "QUATERNION":
            prop, count = "rotation_quaternion", 4
        elif mode == "AXIS_ANGLE":
            return None
        else:
            prop, count = "rotation_euler", 3
        path = bone_path(name, prop)
        static = list(getattr(pose_bone, prop))
        chans = []
        for index in range(count):
            if anim.drivers.find(path, index=index) is not None:
                return None
            fcurve = get_fcurve(action, path, index)
            group = getattr(fcurve, "group", None) if fcurve is not None else None
            if fcurve is not None and (
                fcurve.mute or not fcurve.is_valid or (group is not None and group.mute)
            ):
                fcurve = None
            chans.append((fcurve, float(static[index])))
        channels[name] = (mode, chans)

    def sample(name: str, frame: int) -> Any:
        mode, chans = channels[name]
        values = [
            float(fcurve.evaluate(frame)) if fcurve is not None else static
            for fcurve, static in chans
        ]
        if mode == "QUATERNION":
            quaternion = Quaternion(values)
        else:
            quaternion = Euler(values, mode).to_quaternion()
        quaternion.normalize()
        return quaternion

    return sample


def damp_foot_ik_tilt(
    scene: Any,
    rig: Any,
    action: Any,
    *,
    foot_bones: Sequence[str] = ("foot_ik.L", "foot_ik.R"),
    frame_start: int | None = None,
    frame_end: int | None = None,
    reference_frame: int | None = None,
    strength: float = 0.65,
    damp_axes: Sequence[bool] = (True, True, False),
) -> dict[str, Any]:
    """Dampen foot pitch/roll toward a reference while preserving yaw."""

    start, end = resolve_frame_range(scene, frame_start, frame_end)
    reference = start if reference_frame is None else int(reference_frame)
    if not start <= reference <= end:
        raise ValueError(f"reference frame {reference} is outside {start}-{end}")
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if len(damp_axes) != 3:
        raise ValueError("damp_axes must contain X, Y, and Z flags")
    if get_action(rig) is not action:
        raise RuntimeError("the supplied Action is not active on the MMR rig")
    if Euler is None:
        raise RuntimeError("foot IK tilt repair must run inside Blender")

    valid = [bone for bone in foot_bones if rig.pose.bones.get(bone) is not None]
    missing = [bone for bone in foot_bones if bone not in valid]
    if not valid:
        raise RuntimeError("no target foot IK bones were found")

    samples: dict[str, dict[int, tuple[float, float, float]]] = {
        bone: {} for bone in valid
    }
    view_layer = current_view_layer()
    # P1: the read pass only needs foot_ik's own rotation channels - straight
    # off the F-curves when that is provably the same value (~1500 whole-scene
    # frame_sets saved, 4.3 s -> ~1 s); otherwise the original sweep.
    sampler = _action_rotation_sampler(rig, valid)
    with preserve_scene_frame(scene, view_layer):
        for frame in range(start, end + 1):
            if sampler is None:
                set_scene_frame(scene, frame, view_layer)
            for bone_name in valid:
                quaternion = (
                    get_pose_quaternion(rig.pose.bones[bone_name])
                    if sampler is None
                    else sampler(bone_name, frame)
                )
                euler = quaternion.to_euler("XYZ")
                samples[bone_name][frame] = (
                    float(euler.x),
                    float(euler.y),
                    float(euler.z),
                )

        # 写入遍不再逐帧 frame_set（任务2）：目标值只由第一遍的采样算出，
        # insert_pose_rotation_key 的帧号是显式的、写入的是刚设的属性值——
        # 逐帧求值没有任何读数依赖它，却在每插一个 key 后逼 Blender 重算
        # 整个场景（并拷贝刚被改脏的 517 条曲线的 Action），约占本步 80% 耗时。
        # 退出 preserve_scene_frame 时会按原帧重求值一次，结束状态不变。
        for frame in range(start, end + 1):
            for bone_name in valid:
                current = samples[bone_name][frame]
                target = samples[bone_name][reference]
                values = tuple(
                    _lerp_angle(current[index], target[index], strength)
                    if bool(damp_axes[index])
                    else current[index]
                    for index in range(3)
                )
                pose_bone = rig.pose.bones[bone_name]
                set_pose_quaternion(
                    pose_bone,
                    Euler(values, "XYZ").to_quaternion(),
                )
                insert_pose_rotation_key(pose_bone, frame)

    for bone_name in valid:
        set_linear_for_paths(
            action,
            (
                bone_path(bone_name, "rotation_quaternion"),
                bone_path(bone_name, "rotation_euler"),
                bone_path(bone_name, "rotation_axis_angle"),
            ),
        )
    return {
        "operation": "damp_teto_foot_ik_tilt",
        "frame_range": [start, end],
        "reference_frame": reference,
        "strength": strength,
        "damp_axes": [bool(value) for value in damp_axes],
        "changed_bones": valid,
        "missing_bones": missing,
        "frames_keyed": (end - start + 1) * len(valid),
    }


def _has_excluded_keyword(obj: Any, keywords: Sequence[str]) -> bool:
    current = obj
    while current is not None:
        if any(keyword in current.name for keyword in keywords):
            return True
        current = current.parent
    return False


def collect_target_meshes(
    model_root: Any,
    *,
    visible_only: bool = True,
    excluded_keywords: Sequence[str] = DEFAULT_EXCLUDED_MESH_KEYWORDS,
) -> list[Any]:
    objects = [model_root, *list(model_root.children_recursive)]
    return [
        obj
        for obj in objects
        if obj.type == "MESH"
        and (not visible_only or obj.visible_get())
        and not _has_excluded_keyword(obj, excluded_keywords)
        and obj.data is not None
        and len(obj.data.vertices) > 0
    ]


def _evaluated_mesh_min_z(
    obj: Any,
    depsgraph: Any,
    vertex_sample_step: int,
) -> float | None:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = None
    try:
        mesh = evaluated.to_mesh()
        if mesh is None or not mesh.vertices:
            return None
        # 任务2：逐顶点 Python 循环（本步 ~10 s、1700 万次 min）→ numpy 批量。
        # 逐位复现 mathutils 的 Matrix @ Vector：每个乘积按 float32 算、在 double
        # 里按列顺序累加、最后转回 float32（18546 次网格求值对拍 0 差异）。
        import numpy as np
        n = len(mesh.vertices)
        co = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(n, 3)[::max(1, int(vertex_sample_step))]
        m = np.array(evaluated.matrix_world, dtype=np.float32)
        z = (co[:, 0] * m[2, 0]).astype(np.float64)
        z = z + (co[:, 1] * m[2, 1]).astype(np.float64)
        z = z + (co[:, 2] * m[2, 2]).astype(np.float64)
        z = z + np.float64(m[2, 3] * np.float32(1.0))
        return float(z.astype(np.float32).min())
    finally:
        if mesh is not None:
            evaluated.to_mesh_clear()


_BOOT_GROUP_TEMPLATES = ("足首D.{s}", "足先EX.{s}")


def boot_vertex_indices(mesh_object: Any, sides: Sequence[str] = ("L", "R")) -> dict[str, list[int]] | None:
    """Per-side vertex indices of the boot (MMD ``足首D`` / ``足先EX``, weight > 0.5).

    The measured truth for "where is this foot's sole" during a planted span.
    Returns None when the mesh is not an MMD-named model or the groups are
    missing, so callers fall back to the bind-pose sole vector instead of
    guessing.
    """

    if mesh_object is None or getattr(mesh_object, "type", "") != "MESH":
        return None
    vertices = getattr(getattr(mesh_object, "data", None), "vertices", None)
    if not vertices:
        return None
    groups = {group.name: group.index for group in mesh_object.vertex_groups}
    out: dict[str, list[int]] = {}
    for side in sides:
        wanted = {
            groups[name.format(s=side)]
            for name in _BOOT_GROUP_TEMPLATES
            if name.format(s=side) in groups
        }
        if not wanted:
            return None
        indices = [
            vertex.index
            for vertex in vertices
            if any(entry.group in wanted and entry.weight > 0.5 for entry in vertex.groups)
        ]
        if not indices:
            return None
        out[side] = indices
    return out


def evaluated_boot_min_z(
    mesh_object: Any,
    depsgraph: Any,
    boot_indices: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    """Lowest evaluated world Z of each side's boot vertices at the current frame."""

    import numpy as np

    evaluated = mesh_object.evaluated_get(depsgraph)
    mesh = None
    try:
        mesh = evaluated.to_mesh()
        if mesh is None or not mesh.vertices:
            return {}
        n = len(mesh.vertices)
        co = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(n, 3)
        m = np.array(evaluated.matrix_world, dtype=np.float32)
        z = (co[:, 0] * m[2, 0]).astype(np.float64)
        z = z + (co[:, 1] * m[2, 1]).astype(np.float64)
        z = z + (co[:, 2] * m[2, 2]).astype(np.float64)
        z = z + np.float64(m[2, 3] * np.float32(1.0))
        z = z.astype(np.float32)
        out: dict[str, float] = {}
        for side, indices in boot_indices.items():
            usable = [i for i in indices if 0 <= i < n]
            if usable:
                out[side] = float(z[usable].min())
        return out
    finally:
        if mesh is not None:
            evaluated.to_mesh_clear()


def repair_mesh_floor_lift_v3_safe(
    scene: Any,
    model_root: Any,
    correction: Any,
    *,
    action: Any | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    floor_z: float = 0.0257,
    target_clearance: float = 0.0015,
    tolerance: float = 0.004,
    max_lift_per_frame: float = 0.035,
    strength: float = 0.55,
    smooth_radius: int = 3,
    max_delta_per_frame: float = 0.0045,
    visible_only: bool = True,
    vertex_sample_step: int = 2,
    reset_existing_z_curve: bool = True,
    excluded_keywords: Sequence[str] = DEFAULT_EXCLUDED_MESH_KEYWORDS,
    worst_sample_limit: int = 100,
) -> dict[str, Any]:
    """Scan evaluated Teto meshes and add a non-accumulating correction Z."""

    start, end = resolve_frame_range(scene, frame_start, frame_end)
    meshes = collect_target_meshes(
        model_root,
        visible_only=visible_only,
        excluded_keywords=excluded_keywords,
    )
    if not meshes:
        raise RuntimeError("no target meshes were found under the Teto root")

    existing_action = get_action(correction, required=False)
    if action is not None and existing_action is not action:
        raise RuntimeError("the supplied correction Action is not active")
    action = action or existing_action
    existing_curve = get_fcurve(action, "location", 2) if action else None

    view_layer = current_view_layer()
    original_z: dict[int, float] = {}
    depsgraph = (
        bpy.context.evaluated_depsgraph_get() if bpy is not None else None
    )
    if depsgraph is None:
        raise RuntimeError("mesh floor repair must run inside Blender")

    # 任务2：原来先单独扫一遍只为读 original_z、删旧 Z 曲线、再扫第二遍量网格
    # （第二遍里又把 Z 显式设回 original_z）。合成一遍：旧曲线还在时 frame_set
    # 给出的 Z 就是 original_z[frame]，求值状态与原来第二遍逐位相同；旧曲线在
    # 循环后再删。省掉 1499 次整场景求值。
    # P2：之后保留的"把同一个值显式设回去 + view_layer.update()"也去掉——值没变，
    # 只是逼整棵子树（Teto 网格）每帧再算一遍（bench 摘要逐位相同）。
    minimum_by_frame: dict[int, float | None] = {}
    mesh_by_frame: dict[int, str | None] = {}
    with preserve_scene_frame(scene, view_layer):
        for frame in range(start, end + 1):
            set_scene_frame(scene, frame, view_layer)
            original_z[frame] = float(correction.location.z)
            minimum: float | None = None
            minimum_mesh: str | None = None
            for mesh in meshes:
                value = _evaluated_mesh_min_z(
                    mesh,
                    depsgraph,
                    vertex_sample_step,
                )
                if value is not None and (minimum is None or value < minimum):
                    minimum = value
                    minimum_mesh = mesh.name
            minimum_by_frame[frame] = minimum
            mesh_by_frame[frame] = minimum_mesh

    removed_curves = 0
    if reset_existing_z_curve and existing_curve is not None:
        action.fcurves.remove(existing_curve)
        existing_curve = None
        removed_curves = 1

    target_z = float(floor_z) + float(target_clearance)
    raw: dict[int, float] = {}
    worst: list[dict[str, Any]] = []
    for frame in range(start, end + 1):
        minimum = minimum_by_frame[frame]
        if minimum is None:
            raw[frame] = 0.0
            continue
        penetration = target_z - minimum
        if penetration > float(tolerance):
            lift = min(penetration, float(max_lift_per_frame)) * float(strength)
            raw[frame] = lift
            worst.append(
                {
                    "frame": frame,
                    "min_z": round(minimum, 6),
                    "penetration": round(penetration, 6),
                    "raw_lift": round(lift, 6),
                    "mesh": mesh_by_frame[frame],
                }
            )
        else:
            raw[frame] = 0.0

    smoothed = smooth_frame_values(raw, start, end, int(smooth_radius))
    lift_by_frame = limit_frame_delta(
        smoothed,
        start,
        end,
        float(max_delta_per_frame),
    )
    action = action or ensure_action(correction, f"{correction.name}_floor_lift_v3")
    z_curve = ensure_fcurve(action, "location", 2)
    cache = keyframe_map(z_curve)
    changed_frames = 0
    max_lift = 0.0
    for frame in range(start, end + 1):
        lift = float(lift_by_frame[frame])
        set_fcurve_value(z_curve, frame, original_z[frame] + lift, cache=cache)
        if abs(lift) > EPSILON:
            changed_frames += 1
            max_lift = max(max_lift, lift)
    z_curve.update()
    worst.sort(key=lambda item: item["penetration"], reverse=True)
    return {
        "operation": "repair_teto_mesh_floor_lift_v3_safe",
        "frame_range": [start, end],
        "mesh_count": len(meshes),
        "meshes": [mesh.name for mesh in meshes],
        "removed_existing_z_curves": removed_curves,
        "changed_frames": changed_frames,
        "max_applied_lift": round(max_lift, 6),
        "params": {
            "floor_z": float(floor_z),
            "target_clearance": float(target_clearance),
            "tolerance": float(tolerance),
            "max_lift_per_frame": float(max_lift_per_frame),
            "strength": float(strength),
            "smooth_radius": int(smooth_radius),
            "max_delta_per_frame": float(max_delta_per_frame),
            "vertex_sample_step": int(vertex_sample_step),
        },
        "penetration_sample_count": len(worst),
        "worst_samples": worst[: int(worst_sample_limit)],
    }


def _normalize_side_ranges(
    values: Mapping[str, Sequence[Sequence[int]]],
    side: str,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    unexpected = set(values) - {"L", "R"}
    if unexpected:
        raise ValueError(f"unsupported side(s): {sorted(unexpected)!r}")
    return normalize_ranges(
        values.get(side, ()), frame_start=start, frame_end=end
    )


def analyze_foot_ik_drift(
    scene: Any,
    rig: Any,
    planted_ranges: Mapping[str, Sequence[Sequence[int]]],
    *,
    foot_bones: Mapping[str, str] = DEFAULT_FOOT_IK,
    frame_start: int | None = None,
    frame_end: int | None = None,
    trim_segment_ends: int = 2,
    min_segment_len: int = 4,
) -> dict[str, Any]:
    """Measure target foot IK world-space drift in effective planted ranges."""

    start, end = resolve_frame_range(scene, frame_start, frame_end)
    results: dict[str, list[dict[str, Any]]] = {"L": [], "R": []}
    missing: list[str] = []
    view_layer = current_view_layer()
    # P3: one frame sweep for both feet (the L and R segments overlap: 1711
    # frame_sets → the 1156 of their union); the per-frame positions and every
    # number below are the same as sweeping segment by segment.
    plan: dict[str, list[tuple[int, int, int, int]]] = {"L": [], "R": []}
    wanted: dict[int, set[str]] = {}
    for side in ("L", "R"):
        bone_name = foot_bones[side]
        if rig.pose.bones.get(bone_name) is None:
            missing.append(bone_name)
            continue
        for raw_start, raw_end in _normalize_side_ranges(
            planted_ranges, side, start, end
        ):
            segment_start = raw_start + int(trim_segment_ends)
            segment_end = raw_end - int(trim_segment_ends)
            if segment_end - segment_start + 1 < int(min_segment_len):
                continue
            plan[side].append((raw_start, raw_end, segment_start, segment_end))
            for frame in range(segment_start, segment_end + 1):
                wanted.setdefault(frame, set()).add(side)
    located: dict[tuple[str, int], Any] = {}
    with preserve_scene_frame(scene, view_layer):
        for frame in sorted(wanted):
            set_scene_frame(scene, frame, view_layer)
            for side in sorted(wanted[frame]):
                location = pose_bone_world_location(rig, foot_bones[side])
                if location is not None:
                    located[(side, frame)] = location
    for side in ("L", "R"):
        for raw_start, raw_end, segment_start, segment_end in plan[side]:
            positions: list[tuple[int, Any]] = [
                (frame, located[(side, frame)])
                for frame in range(segment_start, segment_end + 1)
                if (side, frame) in located
            ]
            if not positions:
                continue
            anchor = positions[0][1]
            maximum = 0.0
            total = 0.0
            previous = anchor
            for _, location in positions[1:]:
                maximum = max(
                    maximum,
                    math.hypot(location.x - anchor.x, location.y - anchor.y),
                )
                total += math.hypot(
                    location.x - previous.x,
                    location.y - previous.y,
                )
                previous = location
            final = positions[-1][1]
            results[side].append(
                {
                    "frames": [segment_start, segment_end],
                    "source_frames": [raw_start, raw_end],
                    "length": segment_end - segment_start + 1,
                    "max_drift_xy_m": round(maximum, 5),
                    "end_drift_xy_m": round(
                        math.hypot(final.x - anchor.x, final.y - anchor.y),
                        5,
                    ),
                    "total_xy_motion_m": round(total, 5),
                }
            )
    for side in ("L", "R"):
        results[side].sort(key=lambda item: item["max_drift_xy_m"], reverse=True)
    return {
        "schema_version": "teto_foot_ik_drift_v2",
        "operation": "analyze_teto_foot_ik_drift",
        "rig": rig.name,
        "frame_range": [start, end],
        "params": {
            "trim_segment_ends": int(trim_segment_ends),
            "min_segment_len": int(min_segment_len),
        },
        "missing_bones": missing,
        "feet": results,
    }


def _choose_anchor(
    values: Mapping[int, float],
    start: int,
    end: int,
    mode: str,
) -> float:
    frames = list(range(start, end + 1))
    if mode == "first":
        return float(values[start])
    if mode == "middle":
        return float(values[frames[len(frames) // 2]])
    if mode == "median":
        return float(statistics.median(values[frame] for frame in frames))
    raise ValueError(f"unsupported anchor mode: {mode!r}")


def _apply_channel_lock(
    original: Mapping[int, float],
    output: dict[int, float],
    start: int,
    end: int,
    anchor: float,
    blend_frames: int,
) -> set[int]:
    touched: set[int] = set()
    for frame in range(start, end + 1):
        output[frame] = anchor
        touched.add(frame)
    for offset in range(1, int(blend_frames) + 1):
        before = start - offset
        after = end + offset
        factor = offset / (int(blend_frames) + 1)
        if before in original:
            output[before] = original[before] * factor + anchor * (1.0 - factor)
            touched.add(before)
        if after in original:
            output[after] = anchor * (1.0 - factor) + original[after] * factor
            touched.add(after)
    return touched


def lock_foot_ik_xy(
    scene: Any,
    rig: Any,
    action: Any,
    planted_ranges: Mapping[str, Sequence[Sequence[int]]],
    *,
    foot_bones: Mapping[str, str] = DEFAULT_FOOT_IK,
    frame_start: int | None = None,
    frame_end: int | None = None,
    trim_segment_ends: int = 2,
    min_segment_len: int = 5,
    blend_frames: int = 2,
    anchor_mode: str = "median",
    min_local_xy_range: float = 0.006,
    lock_x: bool = True,
    lock_y: bool = True,
    lock_z: bool = False,
) -> dict[str, Any]:
    """Lock only selected local location axes in effective planted ranges."""

    start, end = resolve_frame_range(scene, frame_start, frame_end)
    if get_action(rig) is not action:
        raise RuntimeError("the supplied Action is not active on the MMR rig")
    if not any((lock_x, lock_y, lock_z)):
        raise ValueError("at least one lock axis must be enabled")
    if int(blend_frames) < 0:
        raise ValueError("blend_frames must be non-negative")

    repaired: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    missing: list[str] = []
    total_values_written = 0

    for side in ("L", "R"):
        bone_name = foot_bones[side]
        pose_bone = rig.pose.bones.get(bone_name)
        if pose_bone is None:
            missing.append(bone_name)
            continue
        data_path = bone_path(bone_name, "location")
        curves = [get_fcurve(action, data_path, axis) for axis in range(3)]
        originals: list[dict[int, float] | None] = [
            cache_fcurve_values(curve, start, end) if curve is not None else None
            for curve in curves
        ]
        if any(values is None for values in originals):
            missing_axes = [
                axis for axis, values in enumerate(originals) if values is None
            ]
            sampled = {axis: {} for axis in missing_axes}
            view_layer = current_view_layer()
            with preserve_scene_frame(scene, view_layer):
                for frame in range(start, end + 1):
                    set_scene_frame(scene, frame, view_layer)
                    for axis in missing_axes:
                        sampled[axis][frame] = float(pose_bone.location[axis])
            for axis in missing_axes:
                originals[axis] = sampled[axis]
        original_values = [dict(values) for values in originals]
        output_values = [dict(values) for values in original_values]
        touched_by_axis: list[set[int]] = [set(), set(), set()]

        for raw_start, raw_end in _normalize_side_ranges(
            planted_ranges, side, start, end
        ):
            segment_start = raw_start + int(trim_segment_ends)
            segment_end = raw_end - int(trim_segment_ends)
            if segment_end - segment_start + 1 < int(min_segment_len):
                skipped.append(
                    {
                        "side": side,
                        "source_frames": [raw_start, raw_end],
                        "reason": "too_short_after_trim",
                    }
                )
                continue
            x_values = [
                original_values[0][frame]
                for frame in range(segment_start, segment_end + 1)
            ]
            y_values = [
                original_values[1][frame]
                for frame in range(segment_start, segment_end + 1)
            ]
            x_range = max(x_values) - min(x_values)
            y_range = max(y_values) - min(y_values)
            xy_range = math.hypot(x_range, y_range)
            if xy_range < float(min_local_xy_range):
                skipped.append(
                    {
                        "side": side,
                        "frames": [segment_start, segment_end],
                        "reason": "below_threshold",
                        "xy_range": round(xy_range, 6),
                    }
                )
                continue

            anchors = [
                _choose_anchor(
                    original_values[axis],
                    segment_start,
                    segment_end,
                    anchor_mode,
                )
                for axis in range(3)
            ]
            enabled = (bool(lock_x), bool(lock_y), bool(lock_z))
            for axis in range(3):
                if enabled[axis]:
                    touched_by_axis[axis].update(
                        _apply_channel_lock(
                            original_values[axis],
                            output_values[axis],
                            segment_start,
                            segment_end,
                            anchors[axis],
                            int(blend_frames),
                        )
                    )
            repaired.append(
                {
                    "side": side,
                    "source_frames": [raw_start, raw_end],
                    "frames": [segment_start, segment_end],
                    "xy_range": round(xy_range, 6),
                    "anchor": [round(value, 6) for value in anchors],
                }
            )

        for axis in range(3):
            if not touched_by_axis[axis]:
                continue
            curve = curves[axis] or ensure_fcurve(
                action, data_path, axis, group=bone_name
            )
            cache = keyframe_map(curve)
            for frame in sorted(touched_by_axis[axis]):
                set_fcurve_value(
                    curve,
                    frame,
                    output_values[axis][frame],
                    cache=cache,
                )
                total_values_written += 1

    update_action(action)
    return {
        "operation": "lock_teto_foot_ik_xy",
        "frame_range": [start, end],
        "params": {
            "trim_segment_ends": int(trim_segment_ends),
            "min_segment_len": int(min_segment_len),
            "blend_frames": int(blend_frames),
            "anchor_mode": anchor_mode,
            "min_local_xy_range": float(min_local_xy_range),
            "lock_axes": [bool(lock_x), bool(lock_y), bool(lock_z)],
        },
        "repaired_count": len(repaired),
        "skipped_count": len(skipped),
        "repaired_segments": repaired,
        "skipped_segments": skipped,
        "missing_bones": missing,
        "values_written": total_values_written,
    }


def _segment_lock_weights(
    seg_start: int,
    seg_end: int,
    blend_frames: int,
    frame_start: int,
    frame_end: int,
) -> dict[int, float]:
    """Frame -> correction weight for one locked segment.

    Interior frames take the full correction; a ring of ``blend_frames`` frames
    outside the trimmed segment eases it toward zero with weight
    (b + 1 - k)/(b + 1) at ring distance k - the same easing the channel locks
    apply.
    """

    weights: dict[int, float] = {}
    for frame in range(int(seg_start), int(seg_end) + 1):
        if frame_start <= frame <= frame_end:
            weights[frame] = 1.0
    for offset in range(1, int(blend_frames) + 1):
        weight = 1.0 - offset / (int(blend_frames) + 1.0)
        for frame in (seg_start - offset, seg_end + offset):
            if frame_start <= frame <= frame_end:
                weights[frame] = max(weight, weights.get(frame, 0.0))
    return weights


def sole_contact_offsets(
    sample_armature: Any,
    mesh_object: Any,
    *,
    ankle_bones: Mapping[str, str] = DEFAULT_ANKLE_BONES,
    vertex_sample_limit: int = 2000,
) -> dict[str, Any] | None:
    """Per-side ankle-local vector from ankle head to the sole contact point.

    The sole point is the lowest mesh vertex on the ankle's own half of the
    model in bind pose.  With this vector the planted ankle height for ANY
    foot orientation follows geometrically: `ankle_z = floor - (R @ d).z` -
    a flat foot degenerates to `floor + sole_offset` and a toe-stand
    automatically keeps its higher ankle instead of being flattened.

    "Own half" is decided in ARMATURE space (the sign of the ankle bone's
    ``head_local.x``), never world X: the model is not centred on world X=0
    (the fixture and the user's work file sit at X~-2.56), and a world-X
    split put both ankles on the same side, so both searched the whole body
    and picked the same left-boot vertex - the right foot's ankle->sole vector
    came out 15.5 cm sideways and the planted anchor height 8-27 mm wrong at
    3-10 deg of foot roll.

    Returns ``{side: Vector}`` or None when the geometry cannot be measured.
    """

    if sample_armature is None or mesh_object is None:
        return None
    vertices = getattr(getattr(mesh_object, "data", None), "vertices", None)
    if not vertices:
        return None
    mesh_world = mesh_object.matrix_world
    mesh_to_armature = sample_armature.matrix_world.inverted_safe() @ mesh_world
    step = max(1, len(vertices) // max(1, int(vertex_sample_limit)))
    out: dict[str, Any] = {}
    for side, bone_name in ankle_bones.items():
        bone = sample_armature.data.bones.get(bone_name)
        if bone is None:
            continue
        head_w = sample_armature.matrix_world @ bone.head_local
        sign = 1.0 if bone.head_local.x >= 0.0 else -1.0
        lowest = None
        lowest_z = None
        for index in range(0, len(vertices), step):
            co = vertices[index].co
            if (mesh_to_armature @ co).x * sign < 0.0:
                continue
            v = mesh_world @ co
            if lowest_z is None or v.z < lowest_z:
                lowest, lowest_z = v, v.z
        if lowest is None:
            continue
        rest_rot = (
            (sample_armature.matrix_world @ bone.matrix_local)
            .to_quaternion()
            .normalized()
        )
        out[side] = rest_rot.inverted() @ (lowest - head_w)
    return out or None


def _settle_pelvis_for_reach(
    scene: Any,
    source_armature: Any,
    sample_armature: Any,
    rig: Any,
    action: Any,
    segments: Mapping[str, list[dict[str, Any]]],
    usable: Mapping[str, bool],
    *,
    pelvis_bone: str,
    ankle_bones: Mapping[str, str],
    grounded_height_for: Any,
    anchor_z: Mapping[str, Mapping[int, float]] | None = None,
    correction_max: float,
    smooth_frames: int,
    frame_start: int,
    frame_end: int,
    view_layer: Any,
    source_leg_bones: Mapping[str, Sequence[str]] = SOURCE_LEG_BONES,
    model_leg_bones: Mapping[str, Sequence[str]] = MODEL_LEG_BONES,
) -> dict[str, Any]:
    """Re-solve the torso height per frame from the source leg reach ratio.

    For a two-bone leg chain the ratio |hip-ankle| / (thigh+shin) determines
    the knee angle exactly, so holding the model's ratio equal to the source
    skeleton's preserves the captured knee bend.  Teto's legs are
    proportionally longer than the capture subject's: the same retargeted hip
    height leaves the model's knee straighter than intended and - inside
    planted spans - physically unable to reach the floor.  Rather than only
    dropping the pelvis when a planted foot saturates, every frame is solved
    for the hip height that makes the model's reach ratio equal the source's;
    planted frames use the (floor-grounded) anchor as the foot target, and
    when both feet are planted the lower of the two solutions wins so both
    stay reachable.

    The correction is written as a Z offset on torso_root, the bone that
    carries the retargeted body translation on this rig family - its parent
    (root) holds no fcurves and stays static, so the same constant-multiplier
    basis decomposition used for foot_ik applies.  The curve is interpolated
    over unmeasurable frames, low-passed, dead-banded and hard-clamped to
    +-correction_max; clamped frames are counted in the report but never
    warned (user decision).
    """

    report: dict[str, Any] = {"status": "applied", "pelvis_bone": pelvis_bone}

    pelvis_pb = rig.pose.bones.get(pelvis_bone)
    pelvis_rest = rig.data.bones.get(pelvis_bone)
    if pelvis_pb is None or pelvis_rest is None or pelvis_rest.parent is None:
        report.update(
            {"status": "skipped_no_pelvis_bone", "bone": pelvis_bone}
        )
        return report

    needed: list[str] = []
    for side in ("L", "R"):
        if not usable.get(side):
            continue
        for name in model_leg_bones[side]:
            if sample_armature.pose.bones.get(name) is None:
                needed.append(name)
        for name in source_leg_bones[side]:
            if source_armature.pose.bones.get(name) is None:
                needed.append(name)
    if needed:
        report.update(
            {
                "status": "skipped_missing_bones",
                "bones": sorted(set(needed)),
            }
        )
        return report

    planted_frames: dict[str, set[int]] = {"L": set(), "R": set()}
    for side in ("L", "R"):
        for segment in segments[side]:
            for frame in range(
                segment["frames"][0], segment["frames"][1] + 1
            ):
                planted_frames[side].add(frame)

    frames = list(range(int(frame_start), int(frame_end) + 1))
    sweep: dict[int, dict[str, Any]] = {}
    with preserve_scene_frame(scene, view_layer):
        for frame in frames:
            set_scene_frame(scene, frame, view_layer)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            ev_mmd = sample_armature.evaluated_get(depsgraph)
            ev_rig = rig.evaluated_get(depsgraph)
            ev_src = source_armature.evaluated_get(depsgraph)
            entry: dict[str, Any] = {
                "rig_world": ev_rig.matrix_world.copy(),
                "pelvis": (
                    ev_rig.matrix_world @ ev_rig.pose.bones[pelvis_bone].matrix
                ).copy(),
            }
            src_entry: dict[str, Any] = {}
            for side in ("L", "R"):
                if not usable.get(side):
                    continue
                thigh, knee, ankle = model_leg_bones[side]
                ankle_matrix = (
                    ev_mmd.matrix_world
                    @ ev_mmd.pose.bones[ankle_bones[side]].matrix
                ).copy()
                entry[side] = {
                    "hip": (
                        ev_mmd.matrix_world
                        @ ev_mmd.pose.bones[thigh].matrix
                    ).translation.copy(),
                    "knee": (
                        ev_mmd.matrix_world
                        @ ev_mmd.pose.bones[knee].matrix
                    ).translation.copy(),
                    "ankle": ankle_matrix.translation.copy(),
                    "ankle_matrix": ankle_matrix,
                }
                s_hip, s_knee, s_ankle = source_leg_bones[side]
                src_entry[side] = {
                    "hip": (
                        ev_src.matrix_world
                        @ ev_src.pose.bones[s_hip].matrix
                    ).translation.copy(),
                    "knee": (
                        ev_src.matrix_world
                        @ ev_src.pose.bones[s_knee].matrix
                    ).translation.copy(),
                    "ankle": (
                        ev_src.matrix_world
                        @ ev_src.pose.bones[s_ankle].matrix
                    ).translation.copy(),
                }
            entry["src"] = src_entry
            sweep[frame] = entry

    # Leg lengths are rigid - one frame's measurement is enough.
    leg_model: dict[str, float] = {}
    leg_source: dict[str, float] = {}
    for side in ("L", "R"):
        if not usable.get(side):
            continue
        sample = sweep.get(frames[0], {})
        mdl = sample.get(side)
        src = sample.get("src", {}).get(side)
        if mdl is not None:
            leg_model[side] = (mdl["hip"] - mdl["knee"]).length + (
                mdl["knee"] - mdl["ankle"]
            ).length
        if src is not None:
            leg_source[side] = (src["hip"] - src["knee"]).length + (
                src["knee"] - src["ankle"]
            ).length
    if not leg_model or not leg_source:
        report.update({"status": "skipped_no_leg_measure"})
        return report

    # Anchored planted foot targets: mid-segment XY, grounded ankle height.
    # `anchor_z` carries the measured boot-mesh grounding (same source as the
    # ankle lock, so the reach solve and the lock agree); without it the
    # bind-pose sole vector grounds the anchor under its own rotation.
    anchor_target: dict[str, dict[int, Any]] = {"L": {}, "R": {}}
    for side in ("L", "R"):
        for segment in segments[side]:
            anchor_frame = int(segment["anchor_frame"])
            anchor_matrix = sweep[anchor_frame][side]["ankle_matrix"]
            target_pos = anchor_matrix.translation.copy()
            measured_z = (anchor_z or {}).get(side, {}).get(anchor_frame)
            if measured_z is not None:
                target_pos.z = float(measured_z)
            else:
                grounded = grounded_height_for(side, anchor_matrix)
                if grounded is not None:
                    target_pos.z = grounded
            for frame in range(
                segment["frames"][0], segment["frames"][1] + 1
            ):
                anchor_target[side][frame] = target_pos

    raw: dict[int, float] = {}
    for frame in frames:
        entry = sweep.get(frame)
        if entry is None:
            continue
        candidates: list[tuple[float, bool]] = []
        for side in ("L", "R"):
            if not usable.get(side):
                continue
            mdl = entry.get(side)
            src = entry.get("src", {}).get(side)
            if mdl is None or src is None:
                continue
            if leg_source[side] <= EPSILON or leg_model[side] <= EPSILON:
                continue
            ratio = (src["hip"] - src["ankle"]).length / leg_source[side]
            planted = frame in planted_frames[side]
            foot = (
                anchor_target[side].get(frame) if planted else mdl["ankle"]
            )
            if foot is None:
                continue
            required_reach = ratio * leg_model[side]
            d_xy = math.hypot(
                mdl["hip"].x - foot.x, mdl["hip"].y - foot.y
            )
            dz = math.sqrt(max(required_reach * required_reach - d_xy * d_xy, 0.0))
            hip_required_z = foot.z + dz
            candidates.append((hip_required_z - mdl["hip"].z, planted))
        if not candidates:
            continue
        if any(planted for _, planted in candidates):
            # Planted drives; the deeper drop keeps both feet reachable.
            correction = min(c for c, planted in candidates if planted)
        else:
            correction = sum(c for c, _ in candidates) / len(candidates)
        raw[frame] = correction

    series = [raw.get(frame) for frame in frames]
    index = 0
    while index < len(series):
        if series[index] is not None:
            index += 1
            continue
        end_index = index
        while end_index < len(series) and series[end_index] is None:
            end_index += 1
        left = series[index - 1] if index > 0 else None
        right = series[end_index] if end_index < len(series) else None
        for k in range(index, end_index):
            if left is None and right is None:
                series[k] = 0.0
            elif left is None:
                series[k] = right
            elif right is None:
                series[k] = left
            else:
                t = (k - index + 1) / (end_index - index + 1)
                series[k] = left + (right - left) * t
        index = end_index

    half = max(1, int(smooth_frames)) // 2
    smoothed = [
        sum(
            series[k]
            for k in range(
                max(0, i - half), min(len(series), i + half + 1)
            )
        )
        / len(
            range(max(0, i - half), min(len(series), i + half + 1))
        )
        for i in range(len(series))
    ]

    cap = abs(float(correction_max))
    capped = 0
    corrected = []
    for value in smoothed:
        if abs(value) < PELVIS_DEADBAND:
            value = 0.0
        elif abs(value) > cap:
            value = math.copysign(cap, value)
            capped += 1
        corrected.append(value)

    # Constant-multiplier write path - same guarantees as the foot_ik writer:
    # parent (root) verified static on range ends + middle, then an empirical
    # basis-vs-channels check against a live sample.
    parent_rest = pelvis_rest.parent
    sample_frames = sorted({int(frame_start), int(frame_end), (int(frame_start) + int(frame_end)) // 2})
    parent_poses = []
    pose_samples = []
    with preserve_scene_frame(scene, view_layer):
        for sample_frame in sample_frames:
            set_scene_frame(scene, sample_frame, view_layer)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            ev_rig = rig.evaluated_get(depsgraph)
            parent_poses.append(
                ev_rig.pose.bones[parent_rest.name].matrix.copy()
            )
            pose_samples.append(
                ev_rig.pose.bones[pelvis_bone].matrix.copy()
            )
    first = parent_poses[0]
    pos_spread = max(
        (first.translation - p.translation).length for p in parent_poses[1:]
    )
    rot_spread = max(
        abs(
            math.degrees(
                first.to_quaternion()
                .rotation_difference(p.to_quaternion())
                .angle
            )
        )
        for p in parent_poses[1:]
    )
    if len(parent_poses) == 1:
        pos_spread = rot_spread = 0.0
    if pos_spread > 0.0001 or rot_spread > 0.05:
        report.update(
            {
                "status": "skipped_parent_not_static",
                "parent": parent_rest.name,
                "pos_mm": round(pos_spread * 1000.0, 3),
                "rot_deg": round(rot_spread, 3),
            }
        )
        return report
    # The map from location channel to armature-space pose translation is
    # affine (pose_trans = M @ loc + t) for every inherit/local-location flag
    # combination - only (M, t) change.  Instead of assuming the default
    # composition, evaluate both candidate rules and keep whichever
    # reproduces the live channels at the sample frames.  torso_root on this
    # rig family does NOT follow the default rule (it failed the
    # matrix_local-based basis check by ~300 mm, hence this calibration).
    parent_pose = parent_poses[0]
    g = parent_rest.matrix_local.inverted_safe() @ pelvis_rest.matrix_local
    candidates = [
        (
            "default",
            (parent_pose @ g).to_3x3(),
            (parent_pose @ g).translation,
        ),
        (
            "parent_space",
            (parent_pose @ parent_rest.matrix_local.inverted_safe()).to_3x3(),
            (
                parent_pose
                @ parent_rest.matrix_local.inverted_safe()
                @ pelvis_rest.matrix_local
            ).translation,
        ),
    ]
    chosen = None
    best_err = None
    with preserve_scene_frame(scene, view_layer):
        for tag, m_rot, t_vec in candidates:
            worst = 0.0
            for sample_frame in sample_frames:
                set_scene_frame(scene, sample_frame, view_layer)
                depsgraph = bpy.context.evaluated_depsgraph_get()
                ev_rig = rig.evaluated_get(depsgraph)
                pose_trans = ev_rig.pose.bones[pelvis_bone].matrix.translation
                loc_channel = rig.pose.bones[pelvis_bone].location.copy()
                predicted = m_rot @ loc_channel + t_vec
                worst = max(worst, (predicted - pose_trans).length)
            if best_err is None or worst < best_err:
                best_err = worst
                chosen = (tag, m_rot, t_vec)
    if best_err is None or best_err > 0.0001:
        report.update(
            {
                "status": "skipped_transform_mismatch",
                "loc_mm": round((best_err or 0.0) * 1000.0, 3),
            }
        )
        return report
    comp_rule, m_rot, t_vec = chosen
    report["composition"] = comp_rule
    m_inv = m_rot.inverted_safe()

    loc_path = bone_path(pelvis_bone, "location")
    loc_curves = [
        ensure_fcurve(action, loc_path, axis, group=pelvis_bone)
        for axis in range(3)
    ]
    loc_cache = [keyframe_map(curve) for curve in loc_curves]
    needed_frames = {
        frame
        for frame, corr in zip(frames, corrected)
        if abs(corr) > PELVIS_DEADBAND
    }
    for curve, cache in zip(loc_curves, loc_cache):
        # add() reallocates the key array: new keys = the last n, cache rebuilt
        add_missing_keys(curve, needed_frames, cache)

    written = 0
    for frame, corr in zip(frames, corrected):
        if abs(corr) <= PELVIS_DEADBAND:
            continue
        desired = sweep[frame]["pelvis"].copy()
        desired.translation.z += corr
        target_arm = (
            sweep[frame]["rig_world"].inverted_safe() @ desired
        ).translation
        loc = m_inv @ (target_arm - t_vec)
        for axis in range(3):
            key = loc_cache[axis].get(frame)
            if key is None:
                continue
            key.co.y = float(loc[axis])
            key.interpolation = "LINEAR"
        written += 1
    for curve in loc_curves:
        curve.update()
    update_action(action)

    magnitudes = [abs(v) for v in corrected if abs(v) > PELVIS_DEADBAND]
    report.update(
        {
            "frames_corrected": written,
            "capped_frames": capped,
            "max_correction_mm": round(
                (max(magnitudes) if magnitudes else 0.0) * 1000.0, 2
            ),
            "mean_correction_mm": round(
                (sum(magnitudes) / len(magnitudes) if magnitudes else 0.0)
                * 1000.0,
                2,
            ),
            "grounded_ankle_z_mm": sorted(
                {
                    round(p.z * 1000.0, 2)
                    for targets in anchor_target.values()
                    for p in targets.values()
                }
            ),
        }
    )
    return report


def settle_pelvis_for_reach(
    scene: Any,
    source_armature: Any,
    sample_armature: Any,
    rig: Any,
    action: Any,
    planted_ranges: Mapping[str, Sequence[Sequence[int]]],
    *,
    pelvis_bone: str = DEFAULT_PELVIS_BONE,
    correction_max: float = PELVIS_CORRECTION_MAX,
    smooth_frames: int = PELVIS_SMOOTH_FRAMES,
    ankle_bones: Mapping[str, str] = DEFAULT_ANKLE_BONES,
    foot_bones: Mapping[str, str] = DEFAULT_FOOT_IK,
    ground_mesh: Any = None,
    ground_clearance: float = 0.0015,
    floor_z: float = 0.0,
    sole_offset: float = 0.0,
    sole_dirs: Mapping[str, Any] | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    trim_segment_ends: int = 2,
    min_segment_len: int = 5,
) -> dict[str, Any]:
    """Standalone entry to the pelvis reach solve (the wizard's C button).

    ``stabilize_planted_feet`` runs this pass inside the foot-lock chain.  The
    reach problem it fixes - Teto's legs are proportionally longer than the
    capture subject's, so a retargeted hip height leaves the knee straighter
    than the source and planted feet saturate at leg extension - is already
    visible right after the retarget, so the user decision (Q15) is that the
    same solve must be runnable on its own, with the planted spans and their
    editor moved along with it.

    The planted spans come from the SAME source-derived annotation ranges the
    planted detector writes: the detector reads the source armature and that
    stays exactly as it is (planted only means anything if it describes the
    capture, not the retarget).

    Builds the same segment/anchor data ``stabilize_planted_feet`` builds for
    the reach pass, then returns ``_settle_pelvis_for_reach``'s report with the
    planted-segment summary added.  ``status`` is ``applied`` only when the
    solve actually ran; every skip says which bone or segment was missing.
    """

    start, end = resolve_frame_range(scene, frame_start, frame_end)
    if bpy is None or Matrix is None or Quaternion is None:
        raise RuntimeError("pelvis reach solve must run inside Blender")
    if get_action(rig, required=False) is not action:
        raise RuntimeError("the supplied Action is not active on the MMR rig")

    segments: dict[str, list[dict[str, Any]]] = {"L": [], "R": []}
    usable: dict[str, bool] = {}
    skipped: list[dict[str, Any]] = []
    missing: list[str] = []
    for side in ("L", "R"):
        ok = True
        for owner, bone_name in (
            (sample_armature, ankle_bones[side]),
            (rig, foot_bones[side]),
        ):
            if owner.pose.bones.get(bone_name) is None:
                missing.append(bone_name)
                ok = False
        usable[side] = ok
        if not ok:
            continue
        for raw_start, raw_end in _normalize_side_ranges(planted_ranges, side, start, end):
            seg_start = raw_start + int(trim_segment_ends)
            seg_end = raw_end - int(trim_segment_ends)
            if seg_end - seg_start + 1 < int(min_segment_len):
                skipped.append({
                    "side": side,
                    "source_frames": [raw_start, raw_end],
                    "reason": "too_short_after_trim",
                })
                continue
            segments[side].append({
                "frames": [seg_start, seg_end],
                "source_frames": [raw_start, raw_end],
                "anchor_frame": seg_start + (seg_end - seg_start) // 2,
            })

    report: dict[str, Any] = {
        "operation": "settle_pelvis_for_reach",
        "frame_range": [start, end],
        "params": {
            "pelvis_bone": pelvis_bone,
            "correction_max": float(correction_max),
            "smooth_frames": int(smooth_frames),
            "trim_segment_ends": int(trim_segment_ends),
            "min_segment_len": int(min_segment_len),
            "floor_z": float(floor_z),
        },
        "planted_segments": {
            side: [segment["frames"] for segment in segments[side]] for side in ("L", "R")
        },
        "usable": dict(usable),
        "skipped_segments": skipped,
        "missing_bones": sorted(set(missing)),
    }
    if not any(segments.values()):
        report["status"] = "skipped_no_planted_segments"
        report["reason"] = "planted 区间为空（先跑 planted 检测/编辑）"
        return report
    if not any(usable.get(side) for side in ("L", "R")):
        report["status"] = "skipped_no_usable_side"
        report["reason"] = "腿骨缺失：" + ", ".join(sorted(set(missing)))
        return report

    view_layer = current_view_layer()
    # Planted anchors are grounded: same measurement the foot lock uses, so the
    # reach solve and the ankle lock agree on where the floor is.
    ground_target_z = float(floor_z) + float(ground_clearance)
    boot = (
        boot_vertex_indices(ground_mesh, ("L", "R"))
        if ground_mesh is not None and bpy is not None
        else None
    )
    anchor_z: dict[str, dict[int, float]] = {"L": {}, "R": {}}
    if boot:
        with preserve_scene_frame(scene, view_layer):
            for side in ("L", "R"):
                bone_name = ankle_bones.get(side)
                if bone_name is None:
                    continue
                for segment in segments[side]:
                    anchor_frame = int(segment["anchor_frame"])
                    set_scene_frame(scene, anchor_frame, view_layer)
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    minima = evaluated_boot_min_z(ground_mesh, depsgraph, boot)
                    measured_sole = minima.get(side)
                    ankle = pose_bone_point_world(
                        sample_armature.evaluated_get(depsgraph), bone_name, "head"
                    )
                    if measured_sole is None or ankle is None:
                        continue
                    anchor_z[side][anchor_frame] = float(ankle.z) + (
                        ground_target_z - float(measured_sole)
                    )

    def _grounded_height(side, anchor_matrix):
        direction = sole_dirs.get(side) if sole_dirs else None
        if direction is not None:
            sole_down = (
                anchor_matrix.to_quaternion().normalized() @ direction
            ).z
            return float(floor_z) - sole_down
        if float(sole_offset) > EPSILON:
            return float(floor_z) + float(sole_offset)
        return None

    solve = _settle_pelvis_for_reach(
        scene,
        source_armature,
        sample_armature,
        rig,
        action,
        segments,
        usable,
        pelvis_bone=pelvis_bone,
        ankle_bones=ankle_bones,
        grounded_height_for=_grounded_height,
        anchor_z=anchor_z,
        correction_max=float(correction_max),
        smooth_frames=int(smooth_frames),
        frame_start=start,
        frame_end=end,
        view_layer=view_layer,
    )
    report["pelvis_solve"] = solve
    report["status"] = solve.get("status", "skipped")
    report["frames_corrected"] = solve.get("frames_corrected", 0)
    report["max_correction_mm"] = solve.get("max_correction_mm", 0.0)
    return report


def stabilize_planted_feet(
    scene: Any,
    sample_armature: Any,
    rig: Any,
    action: Any,
    planted_ranges: Mapping[str, Sequence[Sequence[int]]],
    *,
    source_armature: Any = None,
    pelvis_bone: str = DEFAULT_PELVIS_BONE,
    pelvis_correction_max: float = PELVIS_CORRECTION_MAX,
    pelvis_smooth_frames: int = PELVIS_SMOOTH_FRAMES,
    sole_offset: float = 0.0,
    sole_dirs: Mapping[str, Any] | None = None,
    ground_mesh: Any = None,
    ground_clearance: float = 0.0015,
    floor_z: float = 0.0,
    ankle_bones: Mapping[str, str] = DEFAULT_ANKLE_BONES,
    foot_bones: Mapping[str, str] = DEFAULT_FOOT_IK,
    frame_start: int | None = None,
    frame_end: int | None = None,
    trim_segment_ends: int = 2,
    min_segment_len: int = 5,
    blend_frames: int = 2,
    pos_tolerance: float = STABILIZE_POS_TOLERANCE,
    rot_tolerance_deg: float = STABILIZE_ROT_TOLERANCE_DEG,
    worst_sample_limit: int = 100,
) -> dict[str, Any]:
    """Freeze each planted foot's world pose by closing the loop on the ankle.

    Locking controller channels on faith failed measurably: foot_ik froze to
    0.1 mm while the MMD ankle - the bone the mesh and the bake actually
    follow - kept swinging ~9-19 deg inside planted spans (measured on
    0001-1499 segment [1,100]).  This pass inverts the direction.  Every
    planted segment gets an anchor: the evaluated 足首 world matrix at its
    middle frame.  Per frame the evaluated ankle is compared against the
    anchor, and foot_ik is moved by the world-space difference - position and
    rotation together, so the foot stops pivoting on its frozen ankle instead
    of merely stopping sliding.

    The ankle chain is rigid for what it can reach: foot_ik -> MCH parent ->
    ORG-foot -> COPY_TRANSFORMS 足首 transfers a world delta verbatim (probe
    measured +50 mm -> +50.0 mm), so one write pass pins every reachable
    frame - a second pass can only re-enter the fcurve write path that
    corrupts Blender 4.5's guarded allocator on large actions.  What does not
    converge after one pass is a reachability floor, not slack: at landing
    and lift-off edges the anchor pose can sit beyond the leg's extension, so
    the IK clamps the ankle short of it (~10 cm on this file).  The verify
    measure reports that honestly instead of hammering saturated frames.

    The anchor keeps the ankle's pose at the anchor frame, except height:
    planted means in contact, so the anchor's Z is replaced by the grounded
    ankle height (floor + sole offset measured off the model's bind pose)
    whenever that offset is available - a foot the pelvis carried into the
    air gets put back on the floor, and the preceding pelvis-reach pass
    (source-leg reach ratio re-solved onto torso_root) is what makes that
    grounding physically reachable instead of pinned in the air at leg
    extension.  A real on-the-spot turn is exempted the existing way: delete
    those frames from the planted annotation so no segment covers them.
    """

    start, end = resolve_frame_range(scene, frame_start, frame_end)
    if get_action(rig) is not action:
        raise RuntimeError("the supplied Action is not active on the MMR rig")
    if bpy is None or Matrix is None or Quaternion is None:
        raise RuntimeError("planted foot stabilization must run inside Blender")
    if int(blend_frames) < 0:
        raise ValueError("blend_frames must be non-negative")

    segments: dict[str, list[dict[str, Any]]] = {"L": [], "R": []}
    skipped: list[dict[str, Any]] = []
    missing: list[str] = []
    usable: dict[str, bool] = {}
    for side in ("L", "R"):
        ok = True
        for owner, bone_name in (
            (sample_armature, ankle_bones[side]),
            (rig, foot_bones[side]),
        ):
            if owner.pose.bones.get(bone_name) is None:
                missing.append(bone_name)
                ok = False
        usable[side] = ok
        if not ok:
            continue
        for raw_start, raw_end in _normalize_side_ranges(
            planted_ranges, side, start, end
        ):
            seg_start = raw_start + int(trim_segment_ends)
            seg_end = raw_end - int(trim_segment_ends)
            if seg_end - seg_start + 1 < int(min_segment_len):
                skipped.append(
                    {
                        "side": side,
                        "source_frames": [raw_start, raw_end],
                        "reason": "too_short_after_trim",
                    }
                )
                continue
            segments[side].append(
                {
                    "frames": [seg_start, seg_end],
                    "source_frames": [raw_start, raw_end],
                    "anchor_frame": seg_start + (seg_end - seg_start) // 2,
                    "weights": _segment_lock_weights(
                        seg_start, seg_end, int(blend_frames), start, end
                    ),
                }
            )
    if missing and not any(usable.values()):
        raise RuntimeError(
            "缺少闭环稳定所需骨骼：" + ", ".join(sorted(set(missing)))
        )

    report: dict[str, Any] = {
        "operation": "stabilize_planted_feet",
        "frame_range": [start, end],
        "params": {
            "trim_segment_ends": int(trim_segment_ends),
            "min_segment_len": int(min_segment_len),
            "blend_frames": int(blend_frames),
            "pos_tolerance": float(pos_tolerance),
            "rot_tolerance_deg": float(rot_tolerance_deg),
            "pelvis_correction_max": float(pelvis_correction_max),
            "pelvis_smooth_frames": int(pelvis_smooth_frames),
            "sole_offset": float(sole_offset),
            "floor_z": float(floor_z),
        },
        "skipped_segments": skipped,
        "missing_bones": missing,
    }

    work_frames = sorted(
        {
            frame
            for side in ("L", "R")
            for segment in segments[side]
            for frame in segment["weights"]
        }
    )
    if not work_frames:
        report.update(
            {
                "verification": "nothing_to_lock",
                "iterations": [],
                "segments": [],
                "stabilized_segments": 0,
                "frames_written": 0,
                "worst_samples_before": [],
            }
        )
        return report

    view_layer = current_view_layer()

    # Planted anchors are grounded: contact means the sole touches the floor, so
    # a foot the retarget left floating has to come down.  How far down is a
    # MEASUREMENT, not a model: `sole_dirs` is one bind-pose vertex per foot, and
    # the boot is skinned across 足首D + 足先EX, so a pitched foot's real sole
    # hangs ~2 cm lower than that single point predicts - grounding from it
    # pushed EVERY planted segment down (measured: all 63 segments, -7..-65 mm,
    # landing the boot 17-24 mm under the floor).
    #
    # Instead, at each segment's anchor frame measure the real boot mesh and
    # solve the ABSOLUTE ankle height that puts that sole on the floor:
    #     anchor_z = ankle_z + (floor + clearance - boot_min)
    # Absolute, not a delta: the reach pass below moves the torso ~4 cm, and a
    # delta measured before it lands 4 cm off at both consumers.
    # Grounded segments get their own height back (no move); floating ones come
    # down by exactly their clearance.  63 mesh evaluations, ~0.2 s.
    ground_target_z = float(floor_z) + float(ground_clearance)
    ground_boot: dict[str, list[int]] | None = None
    if ground_mesh is not None and bpy is not None:
        ground_boot = boot_vertex_indices(ground_mesh, ("L", "R"))

    def _measure_anchor_z():
        """Absolute grounded ankle height per (side, anchor frame), from the mesh.

        Read live rather than cached: the boot's lowest vertex does NOT travel
        1:1 with the ankle (it is skinned across 足首D + 足先EX, and the leg's
        joint angles change when the reach pass lowers the torso), so the
        measurement is only valid for the pose it was taken in.  Called once
        before the reach pass (for its foot targets) and once after it - the
        second one is what the ankle lock grounds to, and it is why the foot
        lands on the floor instead of 4.5 mm under it.
        """

        result: dict[str, dict[int, float]] = {"L": {}, "R": {}}
        if ground_boot is None:
            return result
        with preserve_scene_frame(scene, view_layer):
            for side in ("L", "R"):
                bone_name = ankle_bones.get(side)
                if bone_name is None:
                    continue
                for segment in segments[side]:
                    anchor_frame = int(segment["anchor_frame"])
                    set_scene_frame(scene, anchor_frame, view_layer)
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    minima = evaluated_boot_min_z(
                        ground_mesh, depsgraph, ground_boot
                    )
                    measured_sole = minima.get(side)
                    if measured_sole is None:
                        continue
                    ankle = pose_bone_point_world(
                        sample_armature.evaluated_get(depsgraph), bone_name, "head"
                    )
                    if ankle is None:
                        continue
                    result[side][anchor_frame] = float(ankle.z) + (
                        ground_target_z - float(measured_sole)
                    )
        return result

    anchor_z = _measure_anchor_z()

    # Fallback when the boot cannot be measured (non-MMD naming, no groups):
    # the old bind-pose sole vector, which grounds under the anchor's own
    # rotation.  Without any sole geometry we keep the anchor's own height
    # rather than guess one.
    def _grounded_height(side, anchor_matrix):
        direction = sole_dirs.get(side) if sole_dirs else None
        if direction is not None:
            sole_down = (
                anchor_matrix.to_quaternion().normalized() @ direction
            ).z
            return float(floor_z) - sole_down
        if float(sole_offset) > EPSILON:
            return float(floor_z) + float(sole_offset)
        return None

    def _anchor_pos(anchor_matrix, side, anchor_frame):
        pos = anchor_matrix.translation.copy()
        measured_z = anchor_z[side].get(int(anchor_frame))
        if measured_z is not None:
            pos.z = float(measured_z)
            return pos
        grounded = _grounded_height(side, anchor_matrix)
        if grounded is not None:
            pos.z = grounded
        return pos

    # Reach pass first: the torso height solve gives the legs their margin
    # back, so the ankle lock that follows can actually reach the floor.
    if source_armature is not None and float(pelvis_correction_max) > 0.0:
        report["pelvis_solve"] = _settle_pelvis_for_reach(
            scene,
            source_armature,
            sample_armature,
            rig,
            action,
            segments,
            usable,
            pelvis_bone=str(pelvis_bone),
            ankle_bones=ankle_bones,
            grounded_height_for=_grounded_height,
            anchor_z=anchor_z,
            correction_max=float(pelvis_correction_max),
            smooth_frames=int(pelvis_smooth_frames),
            frame_start=start,
            frame_end=end,
            view_layer=view_layer,
        )
    else:
        report["pelvis_solve"] = {"status": "disabled"}

    # The reach pass just changed the leg's joint angles, so the boot's lowest
    # vertex moved relative to the ankle.  Re-measure now that the pose is
    # final; this reading is what the ankle lock grounds to.  (The write path
    # below can only run ONCE per call - a second pass re-enters the fcurve
    # writes that corrupt Blender 4.5's guarded allocator - so the correction
    # has to be computed right the first time, not iterated.)
    settled_anchor_z = _measure_anchor_z()
    if any(settled_anchor_z[side] for side in ("L", "R")):
        anchor_z = settled_anchor_z

    def _measure(frames_to_scan):
        data: dict[int, dict[str, Any]] = {}
        with preserve_scene_frame(scene, view_layer):
            for frame in frames_to_scan:
                set_scene_frame(scene, frame, view_layer)
                depsgraph = bpy.context.evaluated_depsgraph_get()
                ev_mmd = sample_armature.evaluated_get(depsgraph)
                ev_rig = rig.evaluated_get(depsgraph)
                entry: dict[str, Any] = {
                    "rig_world": ev_rig.matrix_world.copy(),
                }
                for side in ("L", "R"):
                    if not usable[side]:
                        continue
                    ankle_pb = ev_mmd.pose.bones.get(ankle_bones[side])
                    ik_pb = ev_rig.pose.bones.get(foot_bones[side])
                    entry[side] = {
                        "ankle": (
                            (ev_mmd.matrix_world @ ankle_pb.matrix).copy()
                            if ankle_pb is not None
                            else None
                        ),
                        "ik": (
                            (ev_rig.matrix_world @ ik_pb.matrix).copy()
                            if ik_pb is not None
                            else None
                        ),
                    }
                data[frame] = entry
        return data

    def _interior_residuals(measured):
        """Max ankle pose error vs anchor over fully locked (weight 1) frames."""
        max_pos = 0.0
        max_rot = 0.0
        rows: list[dict[str, Any]] = []
        for side in ("L", "R"):
            for segment in segments[side]:
                anchor_matrix = (
                    measured.get(segment["anchor_frame"], {})
                    .get(side, {})
                    .get("ankle")
                )
                if anchor_matrix is None:
                    continue
                anchor_quat = anchor_matrix.to_quaternion()
                anchor_loc = _anchor_pos(
                    anchor_matrix, side, segment["anchor_frame"]
                )
                for frame in range(
                    segment["frames"][0], segment["frames"][1] + 1
                ):
                    sample = (
                        measured.get(frame, {}).get(side, {}).get("ankle")
                    )
                    if sample is None:
                        continue
                    d_pos = float((sample.translation - anchor_loc).length)
                    d_angle = (
                        anchor_quat.rotation_difference(
                            sample.to_quaternion()
                        ).angle
                    )
                    if d_angle > math.pi:
                        d_angle = 2.0 * math.pi - d_angle
                    d_rot = abs(math.degrees(d_angle))
                    max_pos = max(max_pos, d_pos)
                    max_rot = max(max_rot, d_rot)
                    rows.append(
                        {
                            "side": side,
                            "frame": frame,
                            "pos_mm": round(d_pos * 1000.0, 3),
                            "rot_deg": round(d_rot, 3),
                        }
                    )
        return max_pos, max_rot, rows

    def _corrections(measured):
        """frame -> target foot_ik world matrix; strongest claim wins."""
        corrections: dict[str, dict[int, Any]] = {"L": {}, "R": {}}
        claims: dict[str, dict[int, float]] = {"L": {}, "R": {}}
        for side in ("L", "R"):
            for segment in segments[side]:
                anchor_matrix = (
                    measured.get(segment["anchor_frame"], {})
                    .get(side, {})
                    .get("ankle")
                )
                if anchor_matrix is None:
                    continue
                anchor_quat = anchor_matrix.to_quaternion()
                anchor_loc = _anchor_pos(
                    anchor_matrix, side, segment["anchor_frame"]
                )
                for frame, weight in segment["weights"].items():
                    sample = measured.get(frame, {}).get(side, {})
                    ankle_matrix = sample.get("ankle")
                    ik_matrix = sample.get("ik")
                    if ankle_matrix is None or ik_matrix is None:
                        continue
                    if weight >= 1.0:
                        target_quat = anchor_quat.copy()
                        target_loc = anchor_loc.copy()
                    else:
                        current_quat = ankle_matrix.to_quaternion()
                        blended_anchor = anchor_quat.copy()
                        if current_quat.dot(blended_anchor) < 0.0:
                            blended_anchor.negate()
                        target_quat = current_quat.slerp(
                            blended_anchor, weight
                        )
                        target_loc = ankle_matrix.translation.lerp(
                            anchor_loc, weight
                        )
                    delta_rot = (
                        target_quat
                        @ ankle_matrix.to_quaternion().inverted()
                    )
                    delta_pos = target_loc - ankle_matrix.translation
                    target = Matrix.LocRotScale(
                        ik_matrix.translation + delta_pos,
                        delta_rot @ ik_matrix.to_quaternion(),
                        ik_matrix.to_scale(),
                    )
                    previous = claims[side].get(frame)
                    if previous is None or weight >= previous:
                        claims[side][frame] = weight
                        corrections[side][frame] = target
        return corrections

    # Static-parent analytic basis path.  foot_ik's parent chain is rigid on
    # this rig family, so the basis decomposition of a world-space target
    # reduces to a constant left-multiplier:
    #     basis = (B_rest^-1 @ P_rest @ P_pose^-1) @ M_target_armature
    # The write pass then stays pure math plus direct key writes - no pose
    # assignment, no per-frame scene evaluation, and no mid-pass keyframe
    # insertion.  Those allocator-heavy paths corrupted Blender 4.5's guarded
    # memory pools (MEM_dupallocN mismatches, then a tbbmalloc access
    # violation) when hammered across ~2500 writes on the full range.
    channel_setup: dict[str, dict[str, Any]] = {}
    with preserve_scene_frame(scene, view_layer):
        for side in ("L", "R"):
            if not usable[side]:
                continue
            bone_name = foot_bones[side]
            pose_bone = rig.pose.bones[bone_name]
            bone = rig.data.bones.get(bone_name)
            parent_bone = bone.parent if bone is not None else None
            if bone is None or parent_bone is None:
                missing.append(f"{bone_name} (rest bone)")
                usable[side] = False
                continue

            # The multiplier is only constant while the parent holds still;
            # verify on the range ends and middle rather than trusting it.
            # Read through the EVALUATED depsgraph so a constrained parent
            # still reports its true pose.
            sample_frames = sorted(
                {
                    int(frame_start),
                    int(frame_end),
                    (int(frame_start) + int(frame_end)) // 2,
                }
            )
            parent_poses = []
            pose_samples = []
            loc_samples = []
            quat_samples = []
            for sample_frame in sample_frames:
                set_scene_frame(scene, sample_frame, view_layer)
                depsgraph = bpy.context.evaluated_depsgraph_get()
                ev_rig = rig.evaluated_get(depsgraph)
                parent_poses.append(
                    ev_rig.pose.bones[parent_bone.name].matrix.copy()
                )
                pose_samples.append(
                    ev_rig.pose.bones[bone_name].matrix.copy()
                )
                # Channel values belong to the frame we set, not the frame
                # preserve_scene_frame restores afterwards - sample them
                # here or the basis check compares two different frames.
                loc_samples.append(
                    rig.pose.bones[bone_name].location.copy()
                )
                if rig.pose.bones[bone_name].rotation_mode == "QUATERNION":
                    quat_samples.append(
                        get_pose_quaternion(rig.pose.bones[bone_name]).copy()
                    )
            first = parent_poses[0]
            pos_spread = max(
                (first.translation - p.translation).length
                for p in parent_poses[1:]
            )
            rot_spread = max(
                abs(
                    math.degrees(
                        first.to_quaternion()
                        .rotation_difference(p.to_quaternion())
                        .angle
                    )
                )
                for p in parent_poses[1:]
            )
            if len(parent_poses) == 1:
                pos_spread = rot_spread = 0.0
            if pos_spread > 0.0001 or rot_spread > 0.05:
                raise RuntimeError(
                    f"脚 IK {bone_name} 的父骨 {parent_bone.name} 在区间内不静止"
                    f"（位移 {pos_spread * 1000:.2f} mm / "
                    f"旋转 {rot_spread:.2f}°），闭环稳定无法解析写入；请报告"
                )
            multiplier = (
                bone.matrix_local.inverted_safe()
                @ parent_bone.matrix_local
                @ first.inverted_safe()
            )

            # Empirical formula check: decomposing the evaluated pose through
            # the multiplier must reproduce the live channel values.
            check_basis = multiplier @ pose_samples[-1]
            loc_err = (
                check_basis.to_translation() - loc_samples[-1]
            ).length
            if pose_bone.rotation_mode == "QUATERNION":
                rot_err = 1.0 - abs(
                    check_basis.to_quaternion().dot(quat_samples[-1])
                )
            else:
                rot_err = 0.0  # non-quaternion modes fall back below anyway
            if loc_err > 0.0001 or rot_err > 0.0001:
                raise RuntimeError(
                    f"脚 IK {bone_name} 的父子变换结构不符合默认继承模式"
                    f"（位置差 {loc_err * 1000:.3f} mm）；请报告此情形"
                )

            loc_path = bone_path(bone_name, "location")
            if pose_bone.rotation_mode == "QUATERNION":
                rot_prop, rot_count = "rotation_quaternion", 4
            elif pose_bone.rotation_mode == "AXIS_ANGLE":
                rot_prop, rot_count = "rotation_axis_angle", 4
            else:
                rot_prop, rot_count = "rotation_euler", 3
            rot_path = bone_path(bone_name, rot_prop)
            loc_curves = [
                ensure_fcurve(action, loc_path, axis, group=bone_name)
                for axis in range(3)
            ]
            rot_curves = [
                ensure_fcurve(action, rot_path, axis, group=bone_name)
                for axis in range(rot_count)
            ]
            channel_setup[side] = {
                "mode": pose_bone.rotation_mode,
                "multiplier": multiplier,
                "loc_curves": loc_curves,
                "rot_curves": rot_curves,
                "loc_cache": [keyframe_map(curve) for curve in loc_curves],
                "rot_cache": [keyframe_map(curve) for curve in rot_curves],
            }

    written_quats: dict[str, dict[int, Any]] = {"L": {}, "R": {}}

    def _seed_quat(side, frame):
        values = [
            float(curve.evaluate(frame))
            for curve in channel_setup[side]["rot_curves"]
        ]
        quat = Quaternion(values)
        if quat.magnitude < EPSILON:
            return None
        quat.normalize()
        return quat

    def _ensure_keys(setup, frames_needed):
        """Batch-create missing keys once per curve, then rebuild the cache.

        Missing keys are rare (the retarget action is baked dense), but
        inserting them one by one inside the write loop would hammer the
        guarded allocator's realloc path - the suspected crash trigger.
        """

        all_curves = [*setup["loc_curves"], *setup["rot_curves"]]
        all_caches = [*setup["loc_cache"], *setup["rot_cache"]]
        for curve, cache in zip(all_curves, all_caches):
            # add() reallocates the key array and update() re-sorts it: the
            # new keys are the last n, and the cache must be rebuilt (the
            # old as_pointer() diff rewrote the curve's FIRST keys instead)
            add_missing_keys(curve, frames_needed, cache)

    def _write(corrections, measured):
        written = 0
        touched: set[Any] = set()
        for side in ("L", "R"):
            setup = channel_setup.get(side)
            if setup is None:
                continue
            frames_to_write = sorted(corrections[side])
            print(
                f"[STAB] write side {side}: {len(frames_to_write)} frames",
                flush=True,
            )
            _ensure_keys(setup, frames_to_write)
            multiplier = setup["multiplier"]
            for frame in frames_to_write:
                rig_world = measured[frame]["rig_world"]
                m_target = (
                    rig_world.inverted_safe() @ corrections[side][frame]
                )
                basis = multiplier @ m_target
                loc = basis.to_translation()
                for axis in range(3):
                    key = setup["loc_cache"][axis].get(frame)
                    if key is None:
                        continue
                    key.co.y = float(loc[axis])
                    key.interpolation = "LINEAR"
                if setup["mode"] == "QUATERNION":
                    quat = basis.to_quaternion()
                    reference = written_quats[side].get(frame - 1)
                    if reference is None:
                        reference = _seed_quat(side, frame - 1)
                    if reference is not None and quat.dot(reference) < 0.0:
                        quat.negate()
                    for axis, value in enumerate(
                        (quat.w, quat.x, quat.y, quat.z)
                    ):
                        key = setup["rot_cache"][axis].get(frame)
                        if key is None:
                            continue
                        key.co.y = float(value)
                        key.interpolation = "LINEAR"
                    written_quats[side][frame] = quat.copy()
                elif setup["mode"] == "AXIS_ANGLE":
                    values = basis.to_quaternion().to_axis_angle()
                    for axis, value in enumerate(
                        (values[1], values[0].x, values[0].y, values[0].z)
                    ):
                        key = setup["rot_cache"][axis].get(frame)
                        if key is None:
                            continue
                        key.co.y = float(value)
                        key.interpolation = "LINEAR"
                else:
                    values = basis.to_euler(setup["mode"])
                    for axis in range(3):
                        key = setup["rot_cache"][axis].get(frame)
                        if key is None:
                            continue
                        key.co.y = float(values[axis])
                        key.interpolation = "LINEAR"
                written += 1
            touched.update(
                [*setup["loc_curves"], *setup["rot_curves"]]
            )
        for curve in touched:
            curve.update()
        update_action(action)
        return written

    # Single write pass, then a verifying re-measure.  One pass is all the
    # geometry can use: on this rig a commanded foot_ik lands on the ankle
    # exactly, so whatever residual survives is a reachability floor (the leg
    # IK clamps an unreachable anchor at full extension) rather than
    # correction slack - measured residual 104 mm staying identical after a
    # second pass proved pushing further cannot move a saturated ankle.
    # Keeping a second write pass also re-entered the allocator path that
    # kept crashing Blender 4.5 on this file size.
    iterations: list[dict[str, Any]] = []
    after_rows: list[dict[str, Any]] = []
    frames_written = 0

    measured = _measure(work_frames)
    max_pos, max_rot, before_rows = _interior_residuals(measured)
    iterations.append(
        {
            "iteration": 0,
            "max_pos_mm": round(max_pos * 1000.0, 3),
            "max_rot_deg": round(max_rot, 3),
        }
    )
    print(
        f"[STAB] before: {len(work_frames)} frames, "
        f"max residual {max_pos * 1000.0:.2f} mm / {max_rot:.2f} deg",
        flush=True,
    )

    if max_pos <= float(pos_tolerance) and max_rot <= float(rot_tolerance_deg):
        verification = "already_still"
        after_rows = before_rows
    else:
        frames_written = _write(_corrections(measured), measured)
        print(f"[STAB] wrote corrections: {frames_written} frames", flush=True)

        # P3: _interior_residuals reads only the weight-1 interiors (anchors sit
        # inside them) - the blend rings need no second measurement
        post = _measure(sorted({
            frame
            for side in ("L", "R")
            for segment in segments[side]
            for frame in range(segment["frames"][0], segment["frames"][1] + 1)
        }))
        max_pos, max_rot, after_rows = _interior_residuals(post)
        iterations.append(
            {
                "iteration": 1,
                "max_pos_mm": round(max_pos * 1000.0, 3),
                "max_rot_deg": round(max_rot, 3),
            }
        )
        print(
            f"[STAB] after: max residual {max_pos * 1000.0:.2f} mm / "
            f"{max_rot:.2f} deg",
            flush=True,
        )
        verification = (
            "converged"
            if max_pos <= float(pos_tolerance)
            and max_rot <= float(rot_tolerance_deg)
            else "unconverged"
        )

    segment_reports = [
        {
            "side": side,
            "source_frames": segment["source_frames"],
            "frames": segment["frames"],
            "anchor_frame": segment["anchor_frame"],
            "locked_frames": sum(
                1 for w in segment["weights"].values() if w >= 1.0
            ),
        }
        for side in ("L", "R")
        for segment in segments[side]
    ]
    def _severity(row):
        return (row["rot_deg"], row["pos_mm"])

    before_rows.sort(key=_severity, reverse=True)
    after_rows.sort(key=_severity, reverse=True)
    unconverged_frames = [
        row
        for row in after_rows
        if row["pos_mm"] > float(pos_tolerance) * 1000.0
        or row["rot_deg"] > float(rot_tolerance_deg)
    ]
    report.update(
        {
            "verification": verification,
            "iterations": iterations,
            "residual_before": iterations[0] if iterations else None,
            "residual_after": (
                iterations[1] if len(iterations) > 1 else None
            ),
            "segments": segment_reports,
            "stabilized_segments": len(segment_reports),
            "frames_written": frames_written,
            "worst_samples_before": before_rows[: int(worst_sample_limit)],
            "worst_samples_after": after_rows[: int(worst_sample_limit)],
            "unconverged_frames": unconverged_frames[: int(worst_sample_limit)],
            "unconverged_count": len(unconverged_frames),
        }
    )
    return report


def _pin_corrections(lowest_by_frame, frame_start, frame_end, target_z):
    """Signed whole-body shifts that put the lowest foot on the floor.

    Unconditional by design: the annotation only marks where the person is
    AIRBORNE, so every other frame counts as a contact and gets pinned.  A
    frame with no usable sample holds the previous value - a missing sample is
    a data gap, not a reason to move the body.
    """

    corrections: dict[int, float] = {}
    missing: list[int] = []
    previous = 0.0
    for frame in range(int(frame_start), int(frame_end) + 1):
        lowest = lowest_by_frame.get(frame)
        if lowest is None:
            missing.append(frame)
            corrections[frame] = previous
            continue
        previous = float(target_z) - float(lowest)
        corrections[frame] = previous
    return corrections, missing


def _smooth_limit_grounded(
    corrections, airborne_frames, frame_start, frame_end, radius, max_delta
):
    """Smooth and rate-limit inside each grounded stretch, never across a flight.

    The rate limit is what keeps the body from hopping at stretch edges, but a
    real takeoff moves several cm per frame; letting the smoothing or the limit
    see across an airborne span would flatten the jump, so both utilities run
    per stretch instead.
    """

    out = dict(corrections)
    start, end = int(frame_start), int(frame_end)
    frame = start
    while frame <= end:
        if frame in airborne_frames:
            frame += 1
            continue
        run_start = frame
        while frame + 1 <= end and (frame + 1) not in airborne_frames:
            frame += 1
        run_end = frame
        values = {f: float(out.get(f, 0.0)) for f in range(run_start, run_end + 1)}
        if int(radius) > 0:
            values = smooth_frame_values(values, run_start, run_end, int(radius))
        if float(max_delta) > 0.0:
            values = limit_frame_delta(values, run_start, run_end, float(max_delta))
        out.update(values)
        frame += 1
    return out


def _ballistic_z(z0, z1, t0, t1, frames, fps, gravity):
    """Vertical positions of a true ballistic arc through both anchors.

    ``z0``/``z1`` are the body heights at the takeoff and landing anchors and
    ``t1 - t0`` is the flight time.  Gravity plus the flight time decide the
    apex (equal anchors: g*T^2/8) - that is the point of rebuilding instead of
    interpolating: a jump flies at the height its duration implies, so GVHMR's
    crushed arcs are restored and the drift correction rides along for free.
    """

    duration = (int(t1) - int(t0)) / float(fps)
    initial = (float(z1) - float(z0)) / duration + 0.5 * float(gravity) * duration
    arc: dict[int, float] = {}
    for frame in frames:
        t = (int(frame) - int(t0)) / float(fps)
        arc[int(frame)] = (
            float(z0) + initial * t - 0.5 * float(gravity) * t * t
        )
    return arc


def _world_z_shift_in_object_space(obj, delta):
    """Express a world +Z shift in the object's own ``location`` space."""

    if Vector is None:
        raise RuntimeError("foot grounding must run inside Blender")
    parent = obj.parent
    if parent is None:
        return (0.0, 0.0, float(delta))
    local = parent.matrix_world.inverted_safe().to_3x3() @ Vector((0.0, 0.0, float(delta)))
    return (float(local.x), float(local.y), float(local.z))


def _world_z_shift_from_parent_rot(parent_rot_inv, delta):
    """_world_z_shift_in_object_space with the parent's inverted world rotation
    recorded beforehand (None = no parent)."""
    if parent_rot_inv is None:
        return (0.0, 0.0, float(delta))
    local = parent_rot_inv @ Vector((0.0, 0.0, float(delta)))
    return (float(local.x), float(local.y), float(local.z))


def model_sole_offset(
    sample_armature: Any,
    mesh_object: Any,
    *,
    contact_points: Mapping[str, Sequence[tuple[str, str]]] = DEFAULT_GROUND_POINTS,
    vertex_sample_limit: int = 2000,
) -> float:
    """How far the pinned bone rests above the model's sole in its bind pose.

    A PMX model is built standing on the ground, so the ankle's bind height IS
    the ankle-to-sole drop - measured on arue Teto as 5.6 cm (ankle sample at
    +5.3 cm, sole's lowest vertex at -0.3 cm).  Pinning the ankle to the floor
    without this offset buries the shoe by that much; the old pipeline hid the
    problem behind a hand tuned 2.57 cm default, this reads it off the model
    instead.  Returns 0.0 when the geometry cannot be measured, which keeps the
    old bare-bone behaviour rather than guessing a number.
    """

    if sample_armature is None or mesh_object is None:
        return 0.0
    rest = []
    for points in contact_points.values():
        for bone_name, point in points:
            bone = sample_armature.data.bones.get(bone_name)
            if bone is None:
                continue
            local = bone.head_local if point == "head" else bone.tail_local
            rest.append(float((sample_armature.matrix_world @ local).z))
    vertices = getattr(getattr(mesh_object, "data", None), "vertices", None)
    if not rest or not vertices:
        return 0.0
    matrix = mesh_object.matrix_world
    step = max(1, len(vertices) // max(1, int(vertex_sample_limit)))
    sole = min(
        float((matrix @ vertices[index].co).z)
        for index in range(0, len(vertices), step)
    )
    return float(min(rest) - sole)


def ground_feet_outside_airborne(
    scene: Any,
    sample_armature: Any,
    move_object: Any,
    action: Any,
    airborne_ranges: Sequence[Sequence[int]] = (),
    *,
    planted_ranges: Sequence[Sequence[int]] | None = None,
    contact_points: Mapping[str, Sequence[tuple[str, str]]] = DEFAULT_GROUND_POINTS,
    frame_start: int | None = None,
    frame_end: int | None = None,
    floor_z: float = 0.0,
    clearance: float = 0.0015,
    sole_offset: float = 0.0,
    smooth_radius: int = 2,
    max_delta: float = 0.01,
    worst_sample_limit: int = 100,
) -> dict[str, Any]:
    """Pin the lowest foot to the floor everywhere except marked airborne spans.

    The old design pinned the frames a *planted* detection called contacts, and
    it could not be made to work: GVHMR's data does not let anyone tell a float
    from a jump, so a detection band either missed floats entirely (0001-1499
    stopped detecting at frame 566 of 1499) or admitted 16 cm of air as a
    contact (0001-0999 frames 1-11: pinned 19.3 cm, then yanked back - the
    "comic hop").  Both failures share one root: the annotation asked a human to
    mark something invisible in the data.

    This pass inverts it.  A human marks where the person is AIRBORNE - few,
    obvious, visible in the video - and every other frame counts as a contact
    and is pinned unconditionally, PoseCapture's way.  Nothing has to infer
    contact from corrupted data any more.

    Airborne spans are rebuilt as a true ballistic arc anchored at the pinned
    heights on both sides: the body flies at the height its duration implies
    (equal anchors: g*T^2/8), which restores GVHMR's crushed jumps, and the
    drift correction rides through the flight for free - no takeoff boost, no
    landing settle.  A span longer than ``BALLISTIC_MAX_FRAMES`` (0.8 s, i.e. a
    78 cm jump) or missing an anchor falls back to a smooth interpolation
    between the neighbouring pins.

    The rig object moves rather than a bone because a top level object's
    ``location`` is plain world metres, and because the MMD armature follows the
    rig through WORLD-space copy constraints; the correction therefore rides
    into MMD Visual Bake and on into the VMD, and survives a re-bake.

    Sampling reads *sample_armature* (the constrained MMD skeleton, i.e. what
    will be baked) while *move_object* is the rig that gets the correction.
    """

    start, end = resolve_frame_range(scene, frame_start, frame_end)
    if get_action(move_object) is not action:
        raise RuntimeError("the supplied Action is not active on the MMR rig")
    if int(smooth_radius) < 0:
        raise ValueError("smooth_radius must be non-negative")
    if float(max_delta) < 0.0:
        raise ValueError("max_delta must be non-negative")

    prepared: dict[str, list[tuple[str, str]]] = {}
    missing_bones: list[str] = []
    for side in ("L", "R"):
        points = [
            (bone_name, point)
            for bone_name, point in contact_points[side]
            if sample_armature.pose.bones.get(bone_name) is not None
        ]
        if not points:
            missing_bones.extend(name for name, _point in contact_points[side])
        prepared[side] = points
    if not any(prepared.values()):
        raise RuntimeError("目标骨架缺少贴地接触骨：" + ", ".join(missing_bones))

    spans = normalize_ranges(airborne_ranges, frame_start=start, frame_end=end)
    airborne_frames: set[int] = set()
    for raw_start, raw_end in spans:
        airborne_frames.update(range(raw_start, raw_end + 1))

    # planted_ranges = pin ONLY inside these spans (user decision 2026-10-17).
    # Without it the pass keeps its old "pin everything that is not airborne"
    # behaviour, which also lifted the swing/floating frames.
    planted_frames: set[int] | None = None
    pin_frames: set[int] | None = None
    if planted_ranges is not None:
        planted_frames = set()
        for raw_start, raw_end in normalize_ranges(
            planted_ranges, frame_start=start, frame_end=end
        ):
            planted_frames.update(range(raw_start, raw_end + 1))
        pin_frames = planted_frames | airborne_frames

    lowest_by_frame: dict[int, float | None] = {}
    original_location: dict[int, tuple[float, float, float]] = {}
    view_layer = current_view_layer()
    if bpy is None:
        raise RuntimeError("foot grounding must run inside Blender")
    parent_rot_inv: dict[int, Any] = {}     # 写入遍用：父物体世界旋转之逆（逐帧）
    with preserve_scene_frame(scene, view_layer):
        for frame in range(start, end + 1):
            set_scene_frame(scene, frame, view_layer)
            original_location[frame] = tuple(float(value) for value in move_object.location)
            if move_object.parent is not None:
                parent_rot_inv[frame] = \
                    move_object.parent.matrix_world.inverted_safe().to_3x3()
            # The MMD skeleton is driven by constraints, so its world pose only
            # exists on the evaluated copy.  Reading the original object reports
            # the stale pre-constraint pose, which measured feet up to 5 cm away
            # from where the viewer actually sees them.
            evaluated = sample_armature.evaluated_get(
                bpy.context.evaluated_depsgraph_get()
            )
            lowest: float | None = None
            for points in prepared.values():
                for bone_name, point in points:
                    location = pose_bone_point_world(evaluated, bone_name, point)
                    if location is None:
                        continue
                    z = float(location.z)
                    lowest = z if lowest is None else min(lowest, z)
            lowest_by_frame[frame] = lowest

    target_z = float(floor_z) + float(clearance) + float(sole_offset)
    fps_base = float(getattr(scene.render, "fps_base", 1.0)) or 1.0
    fps = float(scene.render.fps) / fps_base

    corrected, missing_samples = _pin_corrections(
        lowest_by_frame, start, end, target_z
    )
    if pin_frames is not None:
        # Outside the planted spans the body is moving/swinging: leave it alone.
        for frame in list(corrected):
            if frame not in pin_frames:
                corrected[frame] = 0.0
    corrected = _smooth_limit_grounded(
        corrected, airborne_frames, start, end, int(smooth_radius), float(max_delta)
    )
    if pin_frames is not None:
        # Smoothing must not bleed a planted correction into the swing frames.
        for frame in list(corrected):
            if frame not in pin_frames:
                corrected[frame] = 0.0

    airborne_stats: list[dict[str, Any]] = []
    for raw_start, raw_end in spans:
        frames = list(range(raw_start, raw_end + 1))
        takeoff = next(
            (f for f in range(raw_start - 1, start - 1, -1) if f not in airborne_frames),
            None,
        )
        landing = next(
            (f for f in range(raw_end + 1, end + 1) if f not in airborne_frames),
            None,
        )
        entry: dict[str, Any] = {
            "frames": [raw_start, raw_end],
            "length": len(frames),
            "takeoff_anchor": takeoff,
            "landing_anchor": landing,
        }
        if (
            takeoff is not None
            and landing is not None
            and (landing - takeoff) <= BALLISTIC_MAX_FRAMES
        ):
            z0 = original_location[takeoff][2] + corrected[takeoff]
            z1 = original_location[landing][2] + corrected[landing]
            arc = _ballistic_z(z0, z1, takeoff, landing, frames, fps, GRAVITY)
            for frame in frames:
                corrected[frame] = arc[frame] - original_location[frame][2]
            entry.update(
                {
                    "mode": "ballistic",
                    "flight_seconds": round((landing - takeoff) / fps, 4),
                    "apex_height": round(max(arc.values()) - min(z0, z1), 4),
                }
            )
        else:
            if takeoff is None and landing is None:
                start_value = end_value = 0.0
            elif takeoff is None:
                start_value = end_value = corrected[landing]
            elif landing is None:
                start_value = end_value = corrected[takeoff]
            else:
                start_value = corrected[takeoff]
                end_value = corrected[landing]
            for offset, frame in enumerate(frames):
                weight = (offset + 1) / (len(frames) + 1.0)
                corrected[frame] = start_value + (end_value - start_value) * weight
            entry["mode"] = (
                "interpolated" if takeoff is not None and landing is not None else "held"
            )
            if entry["mode"] == "interpolated":
                entry["note"] = (
                    f"腾空段 {len(frames)} 帧超过弹道重建上限 "
                    f"{BALLISTIC_MAX_FRAMES} 帧，按两端插值"
                )
        airborne_stats.append(entry)

    fuse_clamped = 0
    for frame in range(start, end + 1):
        if abs(corrected[frame]) > MAX_CORRECTION:
            corrected[frame] = math.copysign(MAX_CORRECTION, corrected[frame])
            fuse_clamped += 1

    curves = [ensure_fcurve(action, "location", axis) for axis in range(3)]
    caches = [keyframe_map(curve) for curve in curves]
    changed_frames = 0
    max_applied = 0.0
    measured: list[dict[str, Any]] = []
    with preserve_scene_frame(scene, view_layer):
        for frame in range(start, end + 1):
            correction = float(corrected.get(frame, 0.0))
            # 不再逐帧 frame_set（任务2）：换算只需要父物体的世界矩阵，而父物体
            # 不受本物体位置 key 的影响——读取遍里已逐帧记下，结果逐位相同。
            delta = _world_z_shift_from_parent_rot(
                parent_rot_inv.get(frame), correction)
            base = original_location[frame]
            for axis in range(3):
                set_fcurve_value(
                    curves[axis],
                    frame,
                    base[axis] + delta[axis],
                    cache=caches[axis],
                )
            if abs(correction) > EPSILON:
                changed_frames += 1
                max_applied = max(max_applied, abs(correction))
            lowest = lowest_by_frame.get(frame)
            if (frame not in airborne_frames
                    and (pin_frames is None or frame in pin_frames)
                    and lowest is not None):
                measured.append(
                    {
                        "frame": frame,
                        "foot_z_before": round(lowest, 6),
                        "correction": round(correction, 6),
                        "foot_z_after": round(lowest + correction, 6),
                    }
                )
    for curve in curves:
        curve.update()
    update_action(action)

    measured.sort(key=lambda item: abs(item["correction"]), reverse=True)
    applied = [item["correction"] for item in measured]

    suspects: list[dict[str, Any]] = []
    current_suspect: dict[str, Any] | None = None
    for frame in range(start, end + 1):
        lowest = lowest_by_frame.get(frame)
        height = None
        if (frame not in airborne_frames
                and (pin_frames is None or frame in pin_frames)
                and lowest is not None):
            height = lowest - target_z
        if height is not None and height > SUSPECT_AIRBORNE_HEIGHT:
            if current_suspect is None:
                current_suspect = {"frames": [frame, frame], "max_height": height}
            else:
                current_suspect["frames"][1] = frame
                current_suspect["max_height"] = max(
                    current_suspect["max_height"], height
                )
        elif current_suspect is not None:
            suspects.append(current_suspect)
            current_suspect = None
    if current_suspect is not None:
        suspects.append(current_suspect)
    suspects.sort(key=lambda item: item["max_height"], reverse=True)
    suspect_report = [
        {
            "frames": item["frames"],
            "length": item["frames"][1] - item["frames"][0] + 1,
            "max_height": round(item["max_height"], 4),
        }
        for item in suspects[: int(worst_sample_limit)]
    ]

    return {
        "operation": "ground_teto_feet_outside_airborne",
        "frame_range": [start, end],
        "sample_armature": sample_armature.name,
        "move_object": move_object.name,
        "params": {
            "floor_z": float(floor_z),
            "clearance": float(clearance),
            "sole_offset": float(sole_offset),
            "smooth_radius": int(smooth_radius),
            "max_delta": float(max_delta),
            "gravity": float(GRAVITY),
            "ballistic_max_frames": int(BALLISTIC_MAX_FRAMES),
        },
        "contact_points": {
            side: [f"{bone_name}:{point}" for bone_name, point in points]
            for side, points in prepared.items()
        },
        "airborne_segments": airborne_stats,
        "pin_mode": "planted" if planted_frames is not None else "non_airborne",
        "planted_frames": (
            len(planted_frames) if planted_frames is not None else None
        ),
        "pinned_frames": (
            len(planted_frames) if planted_frames is not None
            else (end - start + 1) - len(airborne_frames)
        ),
        "changed_frames": changed_frames,
        "max_applied_correction": round(max_applied, 6),
        "correction_range": (
            [round(min(applied), 6), round(max(applied), 6)] if applied else None
        ),
        "fuse_clamped_frames": fuse_clamped,
        "missing_sample_frames": len(missing_samples),
        "unmarked_airborne_suspects": suspect_report,
        "worst_samples": measured[: int(worst_sample_limit)],
    }

