"""Repairs and diagnostics for the baked FreeMoCap source armature."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .animation import (
    EPSILON,
    current_view_layer,
    ensure_fcurve,
    keyframe_map,
    limit_frame_delta,
    pose_bone_point_world,
    preserve_scene_frame,
    resolve_frame_range,
    set_fcurve_value,
    set_scene_frame,
    smooth_frame_values,
    update_action,
)


DEFAULT_SOURCE_CONTACT_POINTS = (
    ("foot.L", "head"),
    ("foot.L", "tail"),
    ("foot.R", "head"),
    ("foot.R", "tail"),
    ("heel.02.L", "head"),
    ("heel.02.L", "tail"),
    ("heel.02.R", "head"),
    ("heel.02.R", "tail"),
)



FLOOR_PERCENTILE = 0.05


def _rolling_low_percentile(values, start, end, window, percentile=FLOOR_PERCENTILE):
    """Low percentile of the nearby samples: the ground, not the current step.

    A jump puts both feet up for a moment; the low percentile of a window wide
    enough to contain it still reports the ground the dancer will land on.
    """

    flat = {frame: value for frame, value in values.items() if value is not None}
    out: dict[int, float | None] = {}
    for frame in range(start, end + 1):
        nearby = sorted(
            value for f, value in flat.items() if start <= f <= end and abs(f - frame) <= window
        )
        if not nearby:
            out[frame] = None
            continue
        index = min(int(len(nearby) * percentile), len(nearby) - 1)
        out[frame] = nearby[index]
    # Frames with no usable sample fall back to the nearest estimate.
    known = [f for f in range(start, end + 1) if out[f] is not None]
    if not known:
        return out
    for frame in range(start, end + 1):
        if out[frame] is None:
            nearest = min(known, key=lambda f: abs(f - frame))
            out[frame] = out[nearest]
    return out


def level_source_ground(
    scene: Any,
    armature: Any,
    action: Any,
    *,
    root_bone: str | None = None,
    contact_points: Sequence[tuple[str, str]] = DEFAULT_SOURCE_CONTACT_POINTS,
    frame_start: int | None = None,
    frame_end: int | None = None,
    floor_z: float = 0.02,
    tolerance: float = 0.012,
    target_clearance: float = 0.004,
    max_lift_per_frame: float = 0.6,
    strength: float = 1.0,
    window: int = 30,
    smooth_radius: int = 2,
    max_correction_delta_per_frame: float = 0.018,
    worst_sample_limit: int = 100,
) -> dict[str, Any]:
    """Flatten the source's ground plane and put the feet back on it.

    GVHMR's global trajectory drifts vertically.  On take 0001-1499 the pelvis
    and the feet both climb about 0.4 m across the clip, so a fixed ``floor_z``
    is only right at one end: past roughly frame 500 the feet float 4-17 cm up
    and no frame can satisfy the contact height any more, which is why planted
    detection silently stopped at frame 566 of 1499.

    The ground is estimated from the feet themselves - a rolling low percentile
    of the lowest contact point - and the whole body is moved so that estimate
    lands on ``floor_z + target_clearance``.  The correction is signed, so it
    lowers a floating body as well as raising a sunken one.
    """

    start, end = resolve_frame_range(scene, frame_start, frame_end)
    if root_bone and armature.pose.bones.get(root_bone) is None:
        raise RuntimeError(f"source root bone not found: {root_bone}")
    # Move the armature OBJECT, the way PoseCapture's anti-slide fix moves its
    # root empty: a top level object's ``location`` is plain world metres, so
    # there is no unit conversion and no connected-bone trap.  Writing the
    # pelvis bone instead does nothing at all - the SMPL pelvis is connected to
    # ``f_avg_root`` and Blender ignores the location channel of a connected
    # bone, and the rig is imported at 0.01 scale besides.

    valid_points = [
        (bone_name, point)
        for bone_name, point in contact_points
        if armature.pose.bones.get(bone_name) is not None
    ]
    missing_points = [
        [bone_name, point]
        for bone_name, point in contact_points
        if armature.pose.bones.get(bone_name) is None
    ]
    if not valid_points:
        raise RuntimeError("no valid source foot contact points were found")

    view_layer = current_view_layer()
    min_z_by_frame: dict[int, float | None] = {}
    source_by_frame: dict[int, str | None] = {}
    original_location: dict[int, tuple[float, float, float]] = {}
    with preserve_scene_frame(scene, view_layer):
        for frame in range(start, end + 1):
            set_scene_frame(scene, frame, view_layer)
            original_location[frame] = tuple(float(v) for v in armature.location)
            lowest_z: float | None = None
            lowest_source: str | None = None
            for bone_name, point in valid_points:
                location = pose_bone_point_world(armature, bone_name, point)
                if location is None:
                    continue
                z = float(location.z)
                if lowest_z is None or z < lowest_z:
                    lowest_z = z
                    lowest_source = f"{bone_name}:{point}"
            min_z_by_frame[frame] = lowest_z
            source_by_frame[frame] = lowest_source

    floor_by_frame = _rolling_low_percentile(min_z_by_frame, start, end, int(window))
    target = float(floor_z) + float(target_clearance)
    raw: dict[int, float] = {}
    samples: list[dict[str, Any]] = []
    for frame in range(start, end + 1):
        ground = floor_by_frame[frame]
        if ground is None:
            raw[frame] = 0.0
            continue
        correction = target - float(ground)
        if abs(correction) < float(tolerance):
            correction = 0.0
        elif abs(correction) > float(max_lift_per_frame):
            correction = max_lift_per_frame * (1.0 if correction > 0 else -1.0)
        correction *= float(strength)
        raw[frame] = correction
        if abs(correction) > EPSILON:
            samples.append(
                {
                    "frame": frame,
                    "floor_z": round(float(ground), 6),
                    "correction": round(correction, 6),
                    "source": source_by_frame[frame],
                }
            )

    smoothed = smooth_frame_values(raw, start, end, int(smooth_radius))
    corrected = limit_frame_delta(
        smoothed,
        start,
        end,
        float(max_correction_delta_per_frame),
    )

    # A basis location lives in the bone's own space, and the rig may be
    # scaled: PoseCapture imports the SMPL skeleton at 0.01 with the pelvis
    # bone's axes off the world axes, so one unit of ``location`` is 4.2 mm
    # of world Z, not one metre.  Writing metres straight into that channel is
    # why the old 穿地修复 step never moved anything on a GVHMR file.
    data_path = "location"
    curves = [ensure_fcurve(action, data_path, axis) for axis in range(3)]
    caches = [keyframe_map(curve) for curve in curves]
    changed_frames = 0
    max_correction = 0.0
    for frame in range(start, end + 1):
        correction = float(corrected.get(frame, 0.0))
        base = original_location[frame]
        # World Z is the object's own Z: only that one axis moves.
        set_fcurve_value(
            curves[2], frame, base[2] + correction, cache=caches[2]
        )
        if abs(correction) > EPSILON:
            changed_frames += 1
            max_correction = max(max_correction, abs(correction))
    for curve in curves:
        curve.update()
    update_action(action)

    ground_values = [v for v in floor_by_frame.values() if v is not None]
    samples.sort(key=lambda item: abs(item["correction"]), reverse=True)
    return {
        "operation": "level_source_ground",
        "frame_range": [start, end],
        "root_bone": root_bone,
        "floor_z": float(floor_z),
        "target_clearance": float(target_clearance),
        "window": int(window),
        "floor_before": [round(min(ground_values), 6), round(max(ground_values), 6)]
        if ground_values
        else None,
        "floor_span_before": round(max(ground_values) - min(ground_values), 6)
        if ground_values
        else None,
        "changed_frames": changed_frames,
        "max_correction": round(max_correction, 6),
        "worst_samples": samples[: int(worst_sample_limit)],
        "missing_contact_points": missing_points,
    }

