"""Shared Blender animation helpers used by the core repair algorithms.

This module never saves a ``.blend`` file and never looks objects up by a
hard-coded name.  Callers pass the scene, object, action, and frame window
explicitly so preview/checkpoint handling can remain in the add-on layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
import math
from typing import Any

try:  # Allows ordinary Python to import pure helpers from this package.
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - exercised only outside Blender.
    bpy = None


EPSILON = 1.0e-8


def require_blender() -> None:
    if bpy is None:
        raise RuntimeError("This operation must run inside Blender")


def resolve_frame_range(
    scene: Any,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> tuple[int, int]:
    """Resolve and clip a requested frame window to the scene range."""

    scene_start = int(scene.frame_start)
    scene_end = int(scene.frame_end)
    if scene_start > scene_end:
        raise ValueError(
            f"scene frame range is invalid: {scene_start}-{scene_end}"
        )

    requested_start = scene_start if frame_start is None else int(frame_start)
    requested_end = scene_end if frame_end is None else int(frame_end)
    if requested_start > requested_end:
        raise ValueError(
            f"frame_start ({requested_start}) is after frame_end ({requested_end})"
        )

    start = max(scene_start, requested_start)
    end = min(scene_end, requested_end)
    if start > end:
        raise ValueError(
            f"requested range {requested_start}-{requested_end} does not overlap "
            f"scene range {scene_start}-{scene_end}"
        )
    return start, end


@contextmanager
def preserve_scene_frame(scene: Any, view_layer: Any | None = None):
    """Restore the user's current frame after a sampling or repair pass."""

    original = int(scene.frame_current)
    try:
        yield
    finally:
        scene.frame_set(original)
        if view_layer is not None:
            view_layer.update()


def current_view_layer() -> Any | None:
    if bpy is None:
        return None
    return getattr(bpy.context, "view_layer", None)


def set_scene_frame(
    scene: Any,
    frame: int,
    view_layer: Any | None = None,
) -> None:
    scene.frame_set(int(frame))
    layer = view_layer if view_layer is not None else current_view_layer()
    if layer is not None:
        layer.update()


def get_action(owner: Any, *, required: bool = True) -> Any | None:
    animation_data = getattr(owner, "animation_data", None)
    action = getattr(animation_data, "action", None) if animation_data else None
    if required and action is None:
        raise RuntimeError(f"{getattr(owner, 'name', 'object')} has no active Action")
    return action


def ensure_action_slot(owner: Any) -> None:
    """Bind an Action slot on Blender 4.4+ slotted Actions.

    Assigning a brand new (empty) Action leaves ``animation_data.action_slot``
    unset, and every legacy ``action.fcurves`` write then lands in a channelbag
    the object is not bound to: the curve exists, evaluates, and animates
    nothing.  An Action copied from an existing one carries a bound slot, so
    only owners that start WITHOUT an Action were silently dead - measured on
    the correction Empty of a GVHMR project (全局扶正 never run), where the
    mesh floor repair had been writing a lift that never applied.
    """

    animation_data = getattr(owner, "animation_data", None)
    action = getattr(animation_data, "action", None) if animation_data else None
    if animation_data is None or action is None:
        return
    slots = getattr(action, "slots", None)
    if slots is None:  # Blender < 4.4 keeps the plain legacy behaviour.
        return
    if getattr(animation_data, "action_slot", None) is not None:
        return
    try:
        slot = slots[0] if len(slots) else slots.new(
            id_type=getattr(owner, "id_type", "OBJECT"), name=str(owner.name)
        )
        animation_data.action_slot = slot
    except Exception:  # noqa: BLE001 - binding is best effort, never fatal
        pass


def ensure_action(owner: Any, name: str) -> Any:
    """Return the owner's Action, creating a legacy Action when necessary."""

    require_blender()
    owner.animation_data_create()
    action = owner.animation_data.action
    if action is None:
        action = bpy.data.actions.new(name=name)
        owner.animation_data.action = action
    ensure_action_slot(owner)
    return action


def _fcurves(action: Any) -> Any:
    curves = getattr(action, "fcurves", None)
    if curves is None:
        raise RuntimeError(
            "The Action does not expose legacy F-Curves; convert/bake it to an "
            "editable Action before running MoCap Doctor"
        )
    return curves


def get_fcurve(action: Any, data_path: str, index: int) -> Any | None:
    if action is None:
        return None
    for fcurve in _fcurves(action):
        if fcurve.data_path == data_path and fcurve.array_index == int(index):
            return fcurve
    return None


def ensure_fcurve(
    action: Any,
    data_path: str,
    index: int,
    *,
    group: str | None = None,
) -> Any:
    fcurve = get_fcurve(action, data_path, index)
    if fcurve is not None:
        return fcurve
    kwargs: dict[str, Any] = {"data_path": data_path, "index": int(index)}
    if group:
        kwargs["action_group"] = group
    return _fcurves(action).new(**kwargs)


class KeyframeMap(dict):
    """frame -> Keyframe of one F-curve (built by :func:`keyframe_map`).

    The Keyframe objects hold raw pointers into the curve's key array, and
    ``keyframe_points.insert()`` / ``.add()`` reallocate it - so does
    ``FCurve.update()`` on Blender 4.5 (measured: a write through a key taken
    before update() is lost): after any of them every cached object points at
    freed memory.  ``stale`` marks that:
    the frame SET stays exact (inserted frames are added), only the pointers
    are suspect, so a miss is still a miss and a hit re-reads the curve first.
    """

    stale = False


def keyframe_map(fcurve: Any) -> KeyframeMap:
    """Build a dense-key lookup used to avoid quadratic key searches."""

    return KeyframeMap(
        (int(round(float(key.co.x))), key)
        for key in fcurve.keyframe_points
        if abs(float(key.co.x) - round(float(key.co.x))) < 0.001
    )


def _refresh_keyframe_map(fcurve: Any, cache: dict[int, Any]) -> None:
    cache.clear()
    cache.update(keyframe_map(fcurve))
    if isinstance(cache, KeyframeMap):
        cache.stale = False


def add_missing_keys(
    fcurve: Any,
    frames: Iterable[int],
    cache: dict[int, Any],
    *,
    value: float = 0.0,
    interpolation: str = "LINEAR",
) -> int:
    """Batch-create keys for ``frames`` missing from ``cache`` (a keyframe_map
    of ``fcurve``), then rebuild ``cache`` in place.

    ``keyframe_points.add(n)`` appends n keys at the END of the array (before
    ``update()`` sorts it), so the new keys are the last n entries - not
    "pointers that were not there before": the add reallocates the whole array,
    every old key gets a new address, and the old pointer diff picked the
    curve's FIRST keys and rewrote them to (missing frame, 0).  Returns the
    number of keys added."""

    missing = sorted({int(f) for f in frames} - {f for f in cache if isinstance(f, int)})
    if not missing:
        return 0
    points = fcurve.keyframe_points
    first = len(points)
    points.add(len(missing))
    for offset, frame in enumerate(missing):
        key = points[first + offset]
        key.co.x = float(frame)
        key.co.y = float(value)
        key.interpolation = interpolation
    fcurve.update()
    _refresh_keyframe_map(fcurve, cache)
    return len(missing)


def get_keyframe(
    fcurve: Any,
    frame: int,
    *,
    cache: Mapping[int, Any] | None = None,
) -> Any | None:
    frame = int(frame)
    if cache is not None:
        key = cache.get(frame)
        if key is not None and getattr(cache, "stale", False):
            _refresh_keyframe_map(fcurve, cache)      # see KeyframeMap
            key = cache.get(frame)
        return key
    for key in fcurve.keyframe_points:
        if abs(float(key.co.x) - frame) < 0.001:
            return key
    return None


def set_fcurve_value(
    fcurve: Any,
    frame: int,
    value: float,
    *,
    interpolation: str = "LINEAR",
    cache: dict[int, Any] | None = None,
) -> Any:
    frame = int(frame)
    key = get_keyframe(fcurve, frame, cache=cache)
    if key is None:
        key = fcurve.keyframe_points.insert(
            frame,
            float(value),
            options={"FAST"},
        )
        if cache is not None:
            # insert() reallocated the key array: every OTHER cached key is
            # stale now (writing through one wrote freed memory on sparse
            # curves).  The fresh key and the frame set stay exact.
            if isinstance(cache, KeyframeMap):
                cache[frame] = key
                cache.stale = True
            else:
                _refresh_keyframe_map(fcurve, cache)
    else:
        key.co.y = float(value)
    if interpolation:
        key.interpolation = interpolation
    return key


def cache_fcurve_values(
    fcurve: Any,
    frame_start: int,
    frame_end: int,
) -> dict[int, float]:
    return {
        frame: float(fcurve.evaluate(frame))
        for frame in range(int(frame_start), int(frame_end) + 1)
    }


def update_action(action: Any) -> None:
    if action is None:
        return
    for fcurve in _fcurves(action):
        fcurve.update()


def escape_identifier(value: str) -> str:
    if bpy is not None:
        return bpy.utils.escape_identifier(str(value))
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def bone_path(bone_name: str, property_name: str | None = None) -> str:
    prefix = f'pose.bones["{escape_identifier(bone_name)}"]'
    return prefix if property_name is None else f"{prefix}.{property_name}"


def fcurves_for_bone(action: Any, bone_name: str) -> list[Any]:
    prefix = bone_path(bone_name)
    return [
        fcurve
        for fcurve in _fcurves(action)
        if fcurve.data_path == prefix
        or fcurve.data_path.startswith(prefix + ".")
        or fcurve.data_path.startswith(prefix + "[")
    ]


def remove_fcurves_for_bone(action: Any, bone_name: str) -> int:
    curves = fcurves_for_bone(action, bone_name)
    for fcurve in curves:
        _fcurves(action).remove(fcurve)
    return len(curves)


def set_linear_for_paths(action: Any, data_paths: Iterable[str]) -> int:
    paths = set(data_paths)
    changed = 0
    for fcurve in _fcurves(action):
        if fcurve.data_path not in paths:
            continue
        for key in fcurve.keyframe_points:
            key.interpolation = "LINEAR"
            changed += 1
        fcurve.update()
    return changed


def pose_bone_world_location(armature: Any, bone_name: str) -> Any | None:
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        return None
    return (armature.matrix_world @ pose_bone.matrix).translation.copy()


def pose_bone_point_world(
    armature: Any,
    bone_name: str,
    point: str,
) -> Any | None:
    pose_bone = armature.pose.bones.get(bone_name)
    if pose_bone is None:
        return None
    if point == "head":
        local = pose_bone.head.copy()
    elif point == "tail":
        local = pose_bone.tail.copy()
    else:
        raise ValueError(f"unsupported bone point: {point!r}")
    return armature.matrix_world @ local


def get_pose_quaternion(pose_bone: Any) -> Any:
    mode = pose_bone.rotation_mode
    if mode == "QUATERNION":
        quaternion = pose_bone.rotation_quaternion.copy()
    elif mode == "AXIS_ANGLE":
        quaternion = pose_bone.rotation_axis_angle.to_quaternion()
    else:
        quaternion = pose_bone.rotation_euler.to_quaternion()
    quaternion.normalize()
    return quaternion


def set_pose_quaternion(pose_bone: Any, quaternion: Any) -> None:
    value = quaternion.copy()
    value.normalize()
    mode = pose_bone.rotation_mode
    if mode == "QUATERNION":
        pose_bone.rotation_quaternion = value
    elif mode == "AXIS_ANGLE":
        axis, angle = value.to_axis_angle()
        pose_bone.rotation_axis_angle[0] = angle
        pose_bone.rotation_axis_angle[1] = axis.x
        pose_bone.rotation_axis_angle[2] = axis.y
        pose_bone.rotation_axis_angle[3] = axis.z
    else:
        pose_bone.rotation_euler = value.to_euler(mode)


def insert_pose_rotation_key(pose_bone: Any, frame: int) -> None:
    mode = pose_bone.rotation_mode
    if mode == "QUATERNION":
        data_path = "rotation_quaternion"
    elif mode == "AXIS_ANGLE":
        data_path = "rotation_axis_angle"
    else:
        data_path = "rotation_euler"
    pose_bone.keyframe_insert(data_path=data_path, frame=int(frame))


def normalize_quaternion(values: Sequence[float]) -> list[float]:
    length = math.sqrt(sum(float(value) ** 2 for value in values))
    if length < EPSILON:
        return [float(value) for value in values]
    return [float(value) / length for value in values]


def gaussian_weights(radius: int) -> tuple[list[float], list[int]]:
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return [1.0], [0]

    sigma = max(1.0, radius / 1.5)
    offsets = list(range(-radius, radius + 1))
    weights = [
        math.exp(-(offset * offset) / (2.0 * sigma * sigma))
        for offset in offsets
    ]
    total = sum(weights)
    return [weight / total for weight in weights], offsets


def smooth_frame_values(
    values: Mapping[int, float],
    frame_start: int,
    frame_end: int,
    radius: int,
) -> dict[int, float]:
    weights, offsets = gaussian_weights(radius)
    out: dict[int, float] = {}
    for frame in range(int(frame_start), int(frame_end) + 1):
        weighted = 0.0
        total = 0.0
        for weight, offset in zip(weights, offsets):
            sample_frame = frame + offset
            if sample_frame < frame_start or sample_frame > frame_end:
                continue
            weighted += float(values.get(sample_frame, 0.0)) * weight
            total += weight
        out[frame] = (
            weighted / total if total > EPSILON else float(values.get(frame, 0.0))
        )
    return out


def limit_frame_delta(
    values: Mapping[int, float],
    frame_start: int,
    frame_end: int,
    max_delta: float,
) -> dict[int, float]:
    max_delta = float(max_delta)
    if max_delta < 0.0:
        raise ValueError("max_delta must be non-negative")

    start, end = int(frame_start), int(frame_end)
    out = {frame: float(values.get(frame, 0.0)) for frame in range(start, end + 1)}
    previous = out[start]
    for frame in range(start + 1, end + 1):
        current = min(max(out[frame], previous - max_delta), previous + max_delta)
        out[frame] = current
        previous = current

    previous = out[end]
    for frame in range(end - 1, start - 1, -1):
        current = min(max(out[frame], previous - max_delta), previous + max_delta)
        out[frame] = current
        previous = current
    return out


def clear_object_transform_animation(obj: Any) -> int:
    """Remove object-level transform curves and reset the transform."""

    removed = 0
    action = get_action(obj, required=False)
    if action is not None:
        transform_paths = {
            "location",
            "rotation_euler",
            "rotation_quaternion",
            "rotation_axis_angle",
            "scale",
        }
        for fcurve in list(_fcurves(action)):
            if fcurve.data_path in transform_paths:
                _fcurves(action).remove(fcurve)
                removed += 1
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    return removed
