"""Foot-spacing compensation, ported from PoseCapture and re-targeted at IK.

Ported from ``gvhmr_pose_capture.py::_compensate_foot_spacing`` (~4517-4839) with
the MMD controller profile (``_MMD_CONTROLLER_FOOT_SPACING_PROFILE``, ~3762):
``clearance_ratio 0.55``, ``close_fade_ratio 1.50``, ``crossing_fade_frames 6``,
``crossing_detection_frames 24``, ``crossing_min_gap_ratio 0.08``.

What the original did: measure the ankle spacing the retargeter produced, then
pull the desired spacing toward ``clearance_ratio × rest gap`` only while the
source's feet are close (bidirectional - an over-wide result is narrowed too),
fade that out before ordinary standing/walking distances, follow the source's
left/right order, and push each side by half the missing gap.  It then baked the
samples onto the **FK** chain (``thigh_fk/shin_fk/foot_fk``).

What this port changes (the user's decision):

* the bake goes to the **IK** side - exactly the original's own
  ``IK_CONTROL`` branch (its ~4784-4820 block): write the offset onto the
  ``foot_ik.L`` / ``foot_ik.R`` control translation and keyframe its
  ``location``.  In this project the legs are pure IK (``thigh_parent.IK_FK = 0``),
  so an FK bake would be invisible.
* **no rotation is written at all** (not even the original's ankle-rotation
  preservation keys): leaving every rotation curve alone is what keeps the
  ankle orientation exactly as the retargeter baked it.  The report lists every
  data path that changed, which must be the two ``foot_ik.*.location`` paths.
* source bones are FlowHMR's native names (``L_Hip/L_Knee/L_Ankle``, ``R_*``);
  Q14 dropped the ``f_avg_*`` spellings, so there is no prefix search here.
* the spacing is measured on the IK controls themselves - the same bones that
  are written - instead of the FK ankle.  Measuring and driving the same bones
  keeps "gap at the foot" and "gap we correct" the same quantity.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:  # pragma: no cover - imported outside Blender for py_compile only
    import bpy
    from mathutils import Matrix, Vector
except Exception:  # noqa: BLE001
    bpy = None
    Matrix = None
    Vector = None

from .animation import (
    bone_path,
    current_view_layer,
    get_action,
    preserve_scene_frame,
    set_linear_for_paths,
    set_scene_frame,
    update_action,
)

# FlowHMR native leg chain: (hip, knee, ankle).  The first two lengths are the
# two-bone chain used for the source/target scale.
DEFAULT_SOURCE_LEG_BONES = {
    "L": ("L_Hip", "L_Knee", "L_Ankle"),
    "R": ("R_Hip", "R_Knee", "R_Ankle"),
}
DEFAULT_IK_BONES = {"L": "foot_ik.L", "R": "foot_ik.R"}
DEFAULT_CLEARANCE_RATIO = 0.55
DEFAULT_CLOSE_FADE_RATIO = 1.50
DEFAULT_CROSSING_FADE_FRAMES = 6
DEFAULT_CROSSING_DETECTION_FRAMES = 24
DEFAULT_CROSSING_MIN_GAP_RATIO = 0.08


def foot_crossing_clearance_weights(
    signed_gap_by_frame: Mapping[int, float],
    rest_gap: float,
    fade_frames: int = DEFAULT_CROSSING_FADE_FRAMES,
    detection_frames: int = DEFAULT_CROSSING_DETECTION_FRAMES,
    min_gap_ratio: float = DEFAULT_CROSSING_MIN_GAP_RATIO,
) -> dict[int, float]:
    """Fade the minimum-clearance floor around genuine left/right crossings.

    Verbatim port of the original helper: tiny sign jitter while the feet are
    together must not count as a crossing, so a sign change only counts when
    both sides of it reach a meaningful lateral distance inside the detection
    window.
    """

    frames = sorted(signed_gap_by_frame)
    if len(frames) < 2:
        return {frame: 1.0 for frame in frames}

    fade_frames = max(1, int(fade_frames))
    detection_frames = max(fade_frames, int(detection_frames))
    sign_epsilon = max(float(rest_gap) * 1e-4, 1e-8)
    significant_gap = max(float(rest_gap) * float(min_gap_ratio), sign_epsilon)
    centers: list[float] = []
    last_sign = 0
    last_index = None

    for index, frame in enumerate(frames):
        value = float(signed_gap_by_frame[frame])
        sign = 1 if value > sign_epsilon else (-1 if value < -sign_epsilon else 0)
        if sign == 0:
            continue
        if last_sign and sign != last_sign and last_index is not None:
            left_start = max(0, last_index - detection_frames)
            right_end = min(len(frames), index + detection_frames + 1)
            left_peak = max(
                (
                    abs(float(signed_gap_by_frame[frames[i]]))
                    for i in range(left_start, last_index + 1)
                    if float(signed_gap_by_frame[frames[i]]) * last_sign > 0.0
                ),
                default=0.0,
            )
            right_peak = max(
                (
                    abs(float(signed_gap_by_frame[frames[i]]))
                    for i in range(index, right_end)
                    if float(signed_gap_by_frame[frames[i]]) * sign > 0.0
                ),
                default=0.0,
            )
            if left_peak >= significant_gap and right_peak >= significant_gap:
                centers.append((frames[last_index] + frame) * 0.5)
        last_sign = sign
        last_index = index

    weights: dict[int, float] = {}
    for frame in frames:
        if not centers:
            weights[frame] = 1.0
            continue
        distance = min(abs(float(frame) - center) for center in centers)
        t = max(0.0, min(1.0, distance / float(fade_frames)))
        weights[frame] = t * t * (3.0 - 2.0 * t)
    return weights


def _leg_paths(action: Any, markers: Sequence[str]) -> dict[tuple, tuple]:
    """Snapshot of every leg fcurve, keyed by (data path, array index).

    Keying by data path alone would collapse the three ``location`` curves into
    one entry and hide a change that only moved x/y (the z curve would win the
    dict slot) - which is exactly what the x/y spacing correction does.
    """

    snapshot: dict[tuple, tuple] = {}
    for fcurve in getattr(action, "fcurves", ()) or ():
        path = str(getattr(fcurve, "data_path", ""))
        if not any(marker in path for marker in markers):
            continue
        snapshot[(path, int(getattr(fcurve, "array_index", 0)))] = tuple(
            (round(float(key.co.x), 4), round(float(key.co.y), 9))
            for key in fcurve.keyframe_points
        )
    return snapshot


def compensate_foot_spacing_ik(
    scene: Any,
    source_armature: Any,
    rig: Any,
    action: Any,
    *,
    frame_start: int,
    frame_end: int,
    source_bones: Mapping[str, Sequence[str]] = DEFAULT_SOURCE_LEG_BONES,
    ik_bones: Mapping[str, str] = DEFAULT_IK_BONES,
    clearance_ratio: float = DEFAULT_CLEARANCE_RATIO,
    close_fade_ratio: float = DEFAULT_CLOSE_FADE_RATIO,
    crossing_fade_frames: int = DEFAULT_CROSSING_FADE_FRAMES,
    crossing_detection_frames: int = DEFAULT_CROSSING_DETECTION_FRAMES,
    crossing_min_gap_ratio: float = DEFAULT_CROSSING_MIN_GAP_RATIO,
    view_layer: Any | None = None,
) -> dict[str, Any]:
    """Correct the foot-IK spacing toward the source motion.  Returns a report.

    ``status`` is ``applied`` only when keys were written; every skip names the
    missing bone or the bad input.  ``changed_paths`` lists exactly which curves
    the pass touched (``fk_leg_paths_touched`` must stay empty).
    """

    if bpy is None or Vector is None:
        raise RuntimeError("foot spacing compensation must run inside Blender")
    report: dict[str, Any] = {
        "operation": "compensate_foot_spacing_ik",
        "rig": getattr(rig, "name", ""),
        "source": getattr(source_armature, "name", ""),
        "params": {
            "clearance_ratio": float(clearance_ratio),
            "close_fade_ratio": float(close_fade_ratio),
            "crossing_fade_frames": int(crossing_fade_frames),
            "crossing_detection_frames": int(crossing_detection_frames),
            "crossing_min_gap_ratio": float(crossing_min_gap_ratio),
            "frame_range": [int(frame_start), int(frame_end)],
        },
    }
    if source_armature is None or rig is None:
        report["status"] = "skipped_no_armature"
        return report
    if get_action(rig, required=False) is not action:
        report["status"] = "skipped_action_not_active"
        report["reason"] = "the supplied Action is not active on the MMR rig"
        return report

    missing: list[str] = []
    for side in ("L", "R"):
        for name in source_bones[side]:
            if source_armature.pose.bones.get(name) is None:
                missing.append(name)
        if rig.pose.bones.get(ik_bones[side]) is None:
            missing.append(ik_bones[side])
    if missing:
        report["status"] = "skipped_missing_bones"
        report["bones"] = sorted(set(missing))
        return report

    source_scale = abs(float(source_armature.matrix_world.to_scale().x)) or 1.0
    target_scale = abs(float(rig.matrix_world.to_scale().x)) or 1.0
    source_leg = sum(
        float(source_armature.data.bones[name].length) for name in source_bones["L"][:2]
    ) * source_scale
    # The rig's own two-bone chain (thigh + shin) is the honest target length.
    rig_leg = sum(
        float(rig.data.bones[name].length)
        for name in ("thigh_fk.L", "shin_fk.L")
        if rig.data.bones.get(name) is not None
    )
    target_leg = rig_leg * target_scale
    if target_leg <= 1e-8 or source_leg <= 1e-8:
        report["status"] = "skipped_no_leg_measure"
        report["leg_source"] = source_leg
        report["leg_target"] = target_leg
        return report
    spacing_scale = max(0.5, min(2.0, target_leg / source_leg))

    rest_delta = (
        rig.data.bones[ik_bones["L"]].head_local - rig.data.bones[ik_bones["R"]].head_local
    )
    rest_flat = Vector((float(rest_delta.x), float(rest_delta.y), 0.0))
    if rest_flat.length <= 1e-8:
        report["status"] = "skipped_invalid_side_axis"
        return report
    rest_side_axis = rest_flat.normalized()
    target_side_world = (rig.matrix_world.to_3x3() @ rest_side_axis).normalized()
    rest_flat_gap = rest_flat.length * target_scale
    close_clearance = rest_flat_gap * float(clearance_ratio)
    close_fade_end = rest_flat_gap * float(close_fade_ratio)

    view_layer = view_layer if view_layer is not None else current_view_layer()
    frames = list(range(int(frame_start), int(frame_end) + 1))
    samples: list[tuple[int, Any, Any, float]] = []
    marker_paths = ("foot_ik", "thigh", "shin", "foot", "toe", "leg", "足", "ひざ")
    before = _leg_paths(action, marker_paths)

    with preserve_scene_frame(scene, view_layer):
        source_lateral: dict[int, float] = {}
        for frame in frames:
            set_scene_frame(scene, frame, view_layer)
            src_l = source_armature.matrix_world @ source_armature.pose.bones[
                source_bones["L"][2]
            ].head
            src_r = source_armature.matrix_world @ source_armature.pose.bones[
                source_bones["R"][2]
            ].head
            source_lateral[frame] = (src_l - src_r).dot(target_side_world) * spacing_scale

        crossing_weights = foot_crossing_clearance_weights(
            source_lateral,
            rest_flat_gap,
            fade_frames=crossing_fade_frames,
            detection_frames=crossing_detection_frames,
            min_gap_ratio=crossing_min_gap_ratio,
        )
        crossing_fade_frame_count = sum(
            1 for weight in crossing_weights.values() if weight < 0.999999
        )

        for frame in frames:
            set_scene_frame(scene, frame, view_layer)
            drive_l = rig.pose.bones[ik_bones["L"]].matrix.translation.copy()
            drive_r = rig.pose.bones[ik_bones["R"]].matrix.translation.copy()
            target_delta = drive_l - drive_r
            source_lateral_signed = source_lateral[frame]
            source_flat_gap = abs(source_lateral_signed)
            target_lateral_signed = target_delta.dot(rest_side_axis)
            target_flat_gap = abs(target_lateral_signed) * target_scale

            close_weight = 0.0
            if close_fade_end > rest_flat_gap and source_flat_gap < close_fade_end:
                close_weight = (
                    (close_fade_end - source_flat_gap) / (close_fade_end - rest_flat_gap)
                )
                close_weight = max(0.0, min(1.0, close_weight))
                close_weight = close_weight * close_weight * (3.0 - 2.0 * close_weight)
                close_weight *= crossing_weights.get(frame, 1.0)
                source_flat_gap += (close_clearance - source_flat_gap) * close_weight

            if abs(source_lateral_signed) > 1e-8:
                direction_sign = 1.0 if source_lateral_signed >= 0.0 else -1.0
            elif abs(target_lateral_signed) > 1e-8:
                direction_sign = 1.0 if target_lateral_signed >= 0.0 else -1.0
            else:
                direction_sign = 1.0

            if target_lateral_signed * direction_sign >= 0.0:
                gap_error = source_flat_gap - target_flat_gap
                missing_gap = gap_error if gap_error >= 0.0 else gap_error * close_weight
            else:
                missing_gap = source_flat_gap + target_flat_gap
            direction = rest_side_axis * direction_sign
            offset = direction * (missing_gap * 0.5 / max(target_scale, 1e-8))
            samples.append((frame, drive_l + offset, drive_r - offset, missing_gap))

        # --- the actual write: IK control translation only, no rotations -----
        compensated_frames = 0
        max_compensation = 0.0
        for frame, left_target, right_target, missing_gap in samples:
            set_scene_frame(scene, frame, view_layer)
            rig.pose.bones[ik_bones["L"]].matrix.translation = left_target
            rig.pose.bones[ik_bones["R"]].matrix.translation = right_target
            rig.pose.bones[ik_bones["L"]].keyframe_insert(
                data_path="location", frame=frame, group=ik_bones["L"]
            )
            rig.pose.bones[ik_bones["R"]].keyframe_insert(
                data_path="location", frame=frame, group=ik_bones["R"]
            )
            if abs(missing_gap) > 1e-6:
                compensated_frames += 1
                max_compensation = max(max_compensation, abs(missing_gap))

    ik_paths = {
        bone_path(ik_bones["L"], "location"),
        bone_path(ik_bones["R"], "location"),
    }
    set_linear_for_paths(action, ik_paths)
    update_action(action)

    after = _leg_paths(action, marker_paths)
    changed = sorted(
        f"{path}[{axis}]"
        for path, axis in set(before) | set(after)
        if before.get((path, axis)) != after.get((path, axis))
    )
    fk_markers = ("thigh_fk", "shin_fk", "foot_fk", "thigh_ik", "ORG-thigh", "ORG-shin", "ORG-foot")
    fk_touched = [name for name in changed if any(m in name for m in fk_markers)]

    report.update(
        {
            "status": "applied",
            "frames": len(frames),
            "compensated_frames": compensated_frames,
            "spacing_scale": round(spacing_scale, 4),
            "leg_source_m": round(source_leg, 5),
            "leg_target_m": round(target_leg, 5),
            "rest_gap_m": round(rest_flat_gap, 5),
            "clearance_m": round(close_clearance, 5),
            "close_fade_end_m": round(close_fade_end, 5),
            "crossing_fade_frames": int(crossing_fade_frame_count),
            "max_compensation_m": round(max_compensation, 6),
            "mean_compensation_m": round(
                sum(abs(gap) for *_rest, gap in samples) / max(1, len(samples)), 6
            ),
            "changed_paths": changed,
            "fk_leg_paths_touched": fk_touched,
            "ik_bones": dict(ik_bones),
            "rotation_curves_written": 0,
        }
    )
    return report
