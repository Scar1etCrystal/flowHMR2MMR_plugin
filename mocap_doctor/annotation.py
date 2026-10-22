"""NLA based interval annotation for MoCap Doctor.

The Scene collection is the source of truth.  Users edit inclusive endpoints
through the dedicated in/out operators.  Whole strips may move vertically
between the two visible tracks to correct left/right assignment; native time
movement and edge scaling are deliberately ignored.

This module deliberately does not register itself.  The package ``__init__``
can register ``CLASSES`` first and then call ``register_properties()``.
"""

from __future__ import annotations

import math
import uuid
from typing import Iterable, Mapping, Sequence

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, PointerProperty, StringProperty


INTERNAL_COLLECTION_PREFIX = ".MCD_Annotations_"
HELPER_OBJECT_PREFIX = ".MCD_Annotation_Timeline_"
DUMMY_ACTION_PREFIX = ".MCD_Annotation_Dummy_"
PENDING_MARKER_PREFIX = ".MCD_IN_"
STRIP_META_SEPARATOR = " ::MCD:: "


# These identifiers are intentionally stable.  Reports and project JSON may
# use them without depending on localized UI labels or NLA track names.
CHANNEL_HAND_L_AUTO = "HAND_L_AUTO"
CHANNEL_HAND_L_MANUAL = "HAND_L_MANUAL"
CHANNEL_HAND_R_AUTO = "HAND_R_AUTO"
CHANNEL_HAND_R_MANUAL = "HAND_R_MANUAL"
CHANNEL_FOOT_L_AUTO = "FOOT_L_AUTO"
CHANNEL_FOOT_L_EFFECTIVE = "FOOT_L_EFFECTIVE"
CHANNEL_FOOT_R_AUTO = "FOOT_R_AUTO"
CHANNEL_FOOT_R_EFFECTIVE = "FOOT_R_EFFECTIVE"
# Airborne spans are one track, not a left/right pair: the exemption is about
# the whole body leaving the floor, which is what the pin cares about.
CHANNEL_AIR_AUTO = "AIR_AUTO"
CHANNEL_AIR = "AIR"


TRACK_DEFINITIONS = (
    {
        "channel": CHANNEL_HAND_L_AUTO,
        "label": "左手 自动提示",
        "track": "01 | 左手 自动提示 [锁定]",
        "editable": False,
        "source": "AUTO",
    },
    {
        "channel": CHANNEL_HAND_L_MANUAL,
        "label": "左手坏区间",
        "track": "01 | 左手坏区间",
        "editable": True,
        "source": "MANUAL",
        "style": "HAND",
    },
    {
        "channel": CHANNEL_HAND_R_AUTO,
        "label": "右手 自动提示",
        "track": "03 | 右手 自动提示 [锁定]",
        "editable": False,
        "source": "AUTO",
    },
    {
        "channel": CHANNEL_HAND_R_MANUAL,
        "label": "右手坏区间",
        "track": "02 | 右手坏区间",
        "editable": True,
        "source": "MANUAL",
        "style": "HAND",
    },
    {
        "channel": CHANNEL_FOOT_L_AUTO,
        "label": "左脚 自动 planted",
        "track": "05 | 左脚 自动 planted [锁定]",
        "editable": False,
        "source": "AUTO",
    },
    {
        "channel": CHANNEL_FOOT_L_EFFECTIVE,
        "label": "左脚 planted",
        "track": "01 | 左脚 planted",
        "editable": True,
        "source": "EFFECTIVE",
        "style": "PLANTED",
    },
    {
        "channel": CHANNEL_FOOT_R_AUTO,
        "label": "右脚 自动 planted",
        "track": "07 | 右脚 自动 planted [锁定]",
        "editable": False,
        "source": "AUTO",
    },
    {
        "channel": CHANNEL_FOOT_R_EFFECTIVE,
        "label": "右脚 planted",
        "track": "02 | 右脚 planted",
        "editable": True,
        "source": "EFFECTIVE",
        "style": "PLANTED",
    },
    {
        "channel": CHANNEL_AIR_AUTO,
        "label": "腾空 自动提示",
        "track": "08 | 腾空 自动提示 [锁定]",
        "editable": False,
        "source": "AUTO",
    },
    {
        "channel": CHANNEL_AIR,
        "label": "腾空区间",
        "track": "08 | 腾空区间",
        "editable": True,
        "source": "EFFECTIVE",
        "style": "PLANTED",
    },
)

TRACK_BY_CHANNEL = {item["channel"]: item for item in TRACK_DEFINITIONS}
CHANNEL_BY_TRACK_NAME = {item["track"]: item["channel"] for item in TRACK_DEFINITIONS}
EDITABLE_CHANNELS = tuple(item["channel"] for item in TRACK_DEFINITIONS if item["editable"])

VISIBLE_CHANNELS_BY_STEP = {
    "hand_ranges": (CHANNEL_HAND_L_MANUAL, CHANNEL_HAND_R_MANUAL),
    "contacts": (CHANNEL_FOOT_L_EFFECTIVE, CHANNEL_FOOT_R_EFFECTIVE),
    "ground_feet": (CHANNEL_AIR,),
}

AUTO_CHANNEL_BY_WORKING = {
    CHANNEL_HAND_L_MANUAL: CHANNEL_HAND_L_AUTO,
    CHANNEL_HAND_R_MANUAL: CHANNEL_HAND_R_AUTO,
    CHANNEL_FOOT_L_EFFECTIVE: CHANNEL_FOOT_L_AUTO,
    CHANNEL_FOOT_R_EFFECTIVE: CHANNEL_FOOT_R_AUTO,
    CHANNEL_AIR: CHANNEL_AIR_AUTO,
}

CHANNEL_ENUM_ITEMS = tuple(
    (item["channel"], item["label"], "MoCap Doctor 区间轨道")
    for item in TRACK_DEFINITIONS
)
EDITABLE_CHANNEL_ENUM_ITEMS = tuple(
    (item["channel"], item["label"], "可编辑的 MoCap Doctor 区间轨道")
    for item in TRACK_DEFINITIONS
    if item["editable"]
)


def _stable_scene_suffix(scene: bpy.types.Scene) -> str:
    """Return a short, name-safe suffix for a scene-owned helper."""

    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in scene.name)
    return cleaned[:32] or "Scene"


def inclusive_to_nla(frame_start: int, frame_end: int) -> tuple[float, float]:
    """Convert inclusive business frames to Blender's half-open strip span."""

    start = int(frame_start)
    end = int(frame_end)
    if end < start:
        start, end = end, start
    return float(start), float(end + 1)


def _round_frame(value: float) -> int:
    # Avoid Python's banker rounding: frame 10.5 should become 11.
    return int(math.floor(float(value) + 0.5))


def nla_to_inclusive(frame_start: float, frame_end: float) -> tuple[int, int] | None:
    """Convert a strip span to inclusive integer frames.

    Native NLA scaling may leave fractional bounds.  Commit snaps each edge to
    the nearest frame and rejects zero/negative length strips.
    """

    start = _round_frame(frame_start)
    exclusive_end = _round_frame(frame_end)
    if exclusive_end <= start:
        return None
    return start, exclusive_end - 1


def normalize_ranges(ranges: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    """Sort ranges and merge overlap or adjacency, preserving inclusive ends."""

    items: list[tuple[int, int]] = []
    for value in ranges:
        if len(value) < 2:
            continue
        start, end = int(value[0]), int(value[1])
        if end < start:
            start, end = end, start
        items.append((start, end))
    items.sort(key=lambda item: (item[0], item[1]))

    merged: list[list[int]] = []
    for start, end in items:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _new_uid() -> str:
    return uuid.uuid4().hex


class MCD_AnnotationRange(bpy.types.PropertyGroup):
    """Persistent interval stored on the Scene (both endpoints inclusive)."""

    uid: StringProperty(name="Stable ID", options={"HIDDEN"})
    channel: EnumProperty(name="Channel", items=CHANNEL_ENUM_ITEMS)
    frame_start: IntProperty(name="Start")
    frame_end: IntProperty(name="End")
    source: EnumProperty(
        name="Source",
        items=(
            ("AUTO", "Automatic", "Created by an automatic detector"),
            ("MANUAL", "Manual", "Created or corrected by the user"),
            ("EFFECTIVE", "Effective", "Final planted interval used downstream"),
        ),
    )


def _record_dict(record: MCD_AnnotationRange) -> dict:
    return {
        "uid": record.uid or _new_uid(),
        "channel": record.channel,
        "frame_start": int(record.frame_start),
        "frame_end": int(record.frame_end),
        "source": record.source,
    }


def _normalise_items_with_ids(
    channel: str,
    values: Iterable,
    *,
    merge_adjacent: bool = True,
) -> list[dict]:
    """Normalize tuples/dicts while retaining the first interval's stable ID."""

    parsed: list[dict] = []
    for value in values:
        if isinstance(value, Mapping):
            start = int(value.get("frame_start", value.get("start_frame", value.get("start", 0))))
            end = int(value.get("frame_end", value.get("end_frame", value.get("end", start))))
            uid = str(value.get("uid", ""))
            source = str(value.get("source", ""))
        else:
            if len(value) < 2:
                continue
            start, end = int(value[0]), int(value[1])
            uid = str(value[2]) if len(value) > 2 and value[2] else ""
            source = ""
        if end < start:
            start, end = end, start
        parsed.append(
            {
                "uid": uid,
                "channel": channel,
                "frame_start": start,
                "frame_end": end,
                "source": source,
            }
        )

    parsed.sort(key=lambda item: (item["frame_start"], item["frame_end"]))
    merged: list[dict] = []
    for item in parsed:
        separate_after = merged[-1]["frame_end"] + (1 if merge_adjacent else 0) if merged else None
        if not merged or item["frame_start"] > separate_after:
            merged.append(item.copy())
            continue
        merged[-1]["frame_end"] = max(merged[-1]["frame_end"], item["frame_end"])
        if not merged[-1]["uid"] and item["uid"]:
            merged[-1]["uid"] = item["uid"]
        if merged[-1]["source"] != item["source"]:
            # A user-created interval touching an automatic one becomes a
            # single edited working interval rather than remaining "auto".
            merged[-1]["source"] = TRACK_BY_CHANNEL[channel]["source"]

    for item in merged:
        item["uid"] = item["uid"] or _new_uid()
        item["source"] = item["source"] or TRACK_BY_CHANNEL[channel]["source"]
    return merged


def _write_all_records(scene: bpy.types.Scene, values: Iterable[Mapping]) -> None:
    order = {item["channel"]: index for index, item in enumerate(TRACK_DEFINITIONS)}
    sorted_values = sorted(
        values,
        key=lambda item: (order.get(item["channel"], 999), int(item["frame_start"]), int(item["frame_end"])),
    )
    scene.mcd_annotation_ranges.clear()
    for value in sorted_values:
        if value["channel"] not in TRACK_BY_CHANNEL:
            continue
        record = scene.mcd_annotation_ranges.add()
        record.uid = str(value.get("uid") or _new_uid())
        record.channel = value["channel"]
        record.frame_start = int(value["frame_start"])
        record.frame_end = int(value["frame_end"])
        record.source = str(value.get("source") or TRACK_BY_CHANNEL[value["channel"]]["source"])


def _replace_channels(
    scene: bpy.types.Scene,
    updates: Mapping[str, Iterable],
    *,
    merge_adjacent: bool = True,
    preserve_incoming_metadata: bool = False,
) -> None:
    """Replace one or more channels in a single authoritative write."""

    update_keys = set(updates)
    retained = [_record_dict(item) for item in scene.mcd_annotation_ranges if item.channel not in update_keys]
    existing_by_channel: dict[str, dict[tuple[int, int], dict]] = {}
    for item in scene.mcd_annotation_ranges:
        existing_by_channel.setdefault(item.channel, {})[
            (int(item.frame_start), int(item.frame_end))
        ] = {"uid": item.uid, "source": item.source}

    replacements: list[dict] = []
    for channel, raw_values in updates.items():
        if channel not in TRACK_BY_CHANNEL:
            raise ValueError(f"Unknown annotation channel: {channel}")
        normalized = _normalise_items_with_ids(channel, raw_values, merge_adjacent=merge_adjacent)
        old_items = existing_by_channel.get(channel, {})
        for item in normalized:
            # Exact unchanged ranges keep their old ID even if the projection
            # lost custom properties during an Blender edit.
            old = old_items.get((item["frame_start"], item["frame_end"]))
            if old:
                if preserve_incoming_metadata:
                    item["uid"] = item["uid"] or old["uid"]
                    item["source"] = item["source"] or old["source"]
                else:
                    item["uid"] = old["uid"] or item["uid"]
                    item["source"] = old["source"] or item["source"]
        replacements.extend(normalized)
    _write_all_records(scene, retained + replacements)


def live_channel_ranges(scene: bpy.types.Scene, channel: str) -> list[tuple[int, int]]:
    """Ranges as they currently look in the NLA strips, not the saved records.

    Strips are the editing surface: the user can split and delete them directly
    in the NLA editor, and only 保存标注 used to write that back.  The viewport
    planted rings read the saved records, so a manual strip edit looked stale
    until a save.  This reads the strips themselves and falls back to the
    records when the helper or its tracks are not available.
    """

    helper = getattr(scene, "mcd_annotation_helper", None)
    if helper is None or not getattr(helper, "animation_data", None):
        return get_channel_ranges(scene, channel)
    found: list[tuple[int, int]] = []
    track_seen = False
    for track in helper.animation_data.nla_tracks:
        if _track_channel(track) != channel:
            continue
        track_seen = True
        for strip in track.strips:
            bounds = nla_to_inclusive(strip.frame_start, strip.frame_end)
            if bounds:
                found.append(bounds)
    if not found:
        # A visible track with every strip deleted means "no planted frames";
        # only fall back when the channel has no track at all.
        return [] if track_seen else get_channel_ranges(scene, channel)
    return normalize_ranges(found)


def get_channel_ranges(scene: bpy.types.Scene, channel: str) -> list[tuple[int, int]]:
    """Return normalized inclusive ranges for one channel."""

    return normalize_ranges(
        (item.frame_start, item.frame_end)
        for item in scene.mcd_annotation_ranges
        if item.channel == channel
    )


def replace_channel_ranges(
    scene: bpy.types.Scene,
    channel: str,
    ranges: Iterable[Sequence[int]],
    *,
    rebuild: bool = True,
) -> None:
    """Public API used by detectors and project loading."""

    _replace_channels(scene, {channel: ranges})
    if rebuild and getattr(scene, "mcd_annotation_helper", None):
        rebuild_projection(scene)


def set_hand_auto_hints(
    scene: bpy.types.Scene,
    side: str,
    ranges: Iterable[Sequence[int]],
    *,
    initialize_manual: bool = True,
    force_manual: bool = False,
    rebuild=True,
) -> None:
    is_left = _side_is_left(side)
    auto_channel = CHANNEL_HAND_L_AUTO if is_left else CHANNEL_HAND_R_AUTO
    manual_channel = CHANNEL_HAND_L_MANUAL if is_left else CHANNEL_HAND_R_MANUAL
    cached = list(ranges)
    updates = {auto_channel: cached}
    initialized_key = f"mcd_{manual_channel.lower()}_initialized"
    already_initialized = bool(scene.get(initialized_key, False)) or bool(
        get_channel_ranges(scene, manual_channel)
    )
    if initialize_manual and (force_manual or not already_initialized):
        updates[manual_channel] = [
            {"frame_start": item[0], "frame_end": item[1], "source": "AUTO"}
            for item in cached
        ]
        scene[initialized_key] = True
    _replace_channels(scene, updates)
    if rebuild and getattr(scene, "mcd_annotation_helper", None):
        rebuild_projection(scene)


def set_hand_manual_ranges(scene: bpy.types.Scene, side: str, ranges: Iterable[Sequence[int]], *, rebuild=True) -> None:
    channel = CHANNEL_HAND_L_MANUAL if _side_is_left(side) else CHANNEL_HAND_R_MANUAL
    scene[f"mcd_{channel.lower()}_initialized"] = True
    replace_channel_ranges(scene, channel, ranges, rebuild=rebuild)


def set_planted_auto_ranges(
    scene: bpy.types.Scene,
    side: str,
    ranges: Iterable[Sequence[int]],
    *,
    initialize_effective: bool = True,
    rebuild: bool = True,
) -> None:
    """Write detector output and optionally seed the user-editable result."""

    is_left = _side_is_left(side)
    source = CHANNEL_FOOT_L_AUTO if is_left else CHANNEL_FOOT_R_AUTO
    effective = CHANNEL_FOOT_L_EFFECTIVE if is_left else CHANNEL_FOOT_R_EFFECTIVE
    cached = list(ranges)
    updates = {source: cached}
    initialized_key = f"mcd_{effective.lower()}_initialized"
    already_initialized = bool(scene.get(initialized_key, False)) or bool(
        get_channel_ranges(scene, effective)
    )
    if initialize_effective and not already_initialized:
        updates[effective] = [
            {"frame_start": item[0], "frame_end": item[1], "source": "AUTO"}
            for item in cached
        ]
        scene[initialized_key] = True
    _replace_channels(scene, updates)
    if rebuild and getattr(scene, "mcd_annotation_helper", None):
        rebuild_projection(scene)


def set_planted_effective_ranges(scene: bpy.types.Scene, side: str, ranges: Iterable[Sequence[int]], *, rebuild=True) -> None:
    channel = CHANNEL_FOOT_L_EFFECTIVE if _side_is_left(side) else CHANNEL_FOOT_R_EFFECTIVE
    scene[f"mcd_{channel.lower()}_initialized"] = True
    replace_channel_ranges(scene, channel, ranges, rebuild=rebuild)


def set_air_auto_ranges(
    scene: bpy.types.Scene,
    ranges: Iterable[Sequence[int]],
    *,
    initialize_effective: bool = True,
    force_effective: bool = False,
    rebuild: bool = True,
) -> None:
    """Write the detector's both-feet-airborne spans and seed the editable track.

    ``force_effective`` overwrites hand edits (used by the explicit "reload
    hints" button); the automatic seeding in the contacts step only fills an
    untouched track.
    """

    cached = list(ranges)
    updates = {CHANNEL_AIR_AUTO: cached}
    initialized_key = f"mcd_{CHANNEL_AIR.lower()}_initialized"
    already_initialized = bool(scene.get(initialized_key, False)) or bool(
        get_channel_ranges(scene, CHANNEL_AIR)
    )
    if force_effective or (initialize_effective and not already_initialized):
        updates[CHANNEL_AIR] = [
            {"frame_start": item[0], "frame_end": item[1], "source": "AUTO"}
            for item in cached
        ]
        scene[initialized_key] = True
    _replace_channels(scene, updates)
    if rebuild and getattr(scene, "mcd_annotation_helper", None):
        rebuild_projection(scene)


def set_air_ranges(
    scene: bpy.types.Scene, ranges: Iterable[Sequence[int]], *, rebuild=True
) -> None:
    """Write the hand-editable airborne track directly."""

    scene[f"mcd_{CHANNEL_AIR.lower()}_initialized"] = True
    replace_channel_ranges(scene, CHANNEL_AIR, ranges, rebuild=rebuild)


def reset_planted_effective_to_auto(scene: bpy.types.Scene, side: str, *, rebuild=True) -> None:
    is_left = _side_is_left(side)
    source = CHANNEL_FOOT_L_AUTO if is_left else CHANNEL_FOOT_R_AUTO
    effective = CHANNEL_FOOT_L_EFFECTIVE if is_left else CHANNEL_FOOT_R_EFFECTIVE
    values = [
        {"frame_start": start, "frame_end": end, "source": "AUTO"}
        for start, end in get_channel_ranges(scene, source)
    ]
    scene[f"mcd_{effective.lower()}_initialized"] = True
    _replace_channels(scene, {effective: values})
    if rebuild and getattr(scene, "mcd_annotation_helper", None):
        rebuild_projection(scene)


def _side_is_left(side: str) -> bool:
    value = str(side).strip().upper()
    if value in {"L", "LEFT", "左"}:
        return True
    if value in {"R", "RIGHT", "右"}:
        return False
    raise ValueError(f"Unknown side {side!r}; expected L/LEFT or R/RIGHT")


def _mark_working_channel_initialized(scene: bpy.types.Scene, channel: str) -> None:
    if channel in EDITABLE_CHANNELS:
        scene[f"mcd_{channel.lower()}_initialized"] = True


def visible_channels(scene: bpy.types.Scene) -> tuple[str, ...]:
    settings = getattr(scene, "mocap_doctor", None)
    step_id = getattr(settings, "annotation_step_id", "") if settings else ""
    channels = VISIBLE_CHANNELS_BY_STEP.get(step_id)
    if channels:
        return channels
    active = getattr(scene, "mcd_annotation_active_channel", CHANNEL_HAND_L_MANUAL)
    if active in VISIBLE_CHANNELS_BY_STEP["contacts"]:
        return VISIBLE_CHANNELS_BY_STEP["contacts"]
    return VISIBLE_CHANNELS_BY_STEP["hand_ranges"]


def visible_track_definitions(scene: bpy.types.Scene) -> tuple[dict, ...]:
    channels = set(visible_channels(scene))
    return tuple(item for item in TRACK_DEFINITIONS if item["channel"] in channels)


def ensure_working_ranges_initialized(scene: bpy.types.Scene, step_id: str) -> None:
    """Migrate old projects to one editable track per side."""

    if step_id == "hand_ranges":
        pairs = (
            (CHANNEL_HAND_L_AUTO, CHANNEL_HAND_L_MANUAL),
            (CHANNEL_HAND_R_AUTO, CHANNEL_HAND_R_MANUAL),
        )
    elif step_id == "contacts":
        pairs = (
            (CHANNEL_FOOT_L_AUTO, CHANNEL_FOOT_L_EFFECTIVE),
            (CHANNEL_FOOT_R_AUTO, CHANNEL_FOOT_R_EFFECTIVE),
        )
    elif step_id == "ground_feet":
        pairs = ((CHANNEL_AIR_AUTO, CHANNEL_AIR),)
    else:
        return

    updates = {}
    for auto_channel, working_channel in pairs:
        key = f"mcd_{working_channel.lower()}_initialized"
        working = get_channel_ranges(scene, working_channel)
        if bool(scene.get(key, False)) or working:
            scene[key] = True
            continue
        updates[working_channel] = [
            {"frame_start": start, "frame_end": end, "source": "AUTO"}
            for start, end in get_channel_ranges(scene, auto_channel)
        ]
        scene[key] = True
    if updates:
        _replace_channels(scene, updates)


def ensure_internal_collection(scene: bpy.types.Scene) -> bpy.types.Collection:
    name = f"{INTERNAL_COLLECTION_PREFIX}{_stable_scene_suffix(scene)}"
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    collection["mcd_annotation_internal"] = True
    collection.hide_render = True
    if collection.name not in scene.collection.children:
        scene.collection.children.link(collection)
    return collection


def _ensure_dummy_action(style: str, uid: str, source: str) -> bpy.types.Action:
    style = str(style or "HAND").upper()
    action_name = f"{DUMMY_ACTION_PREFIX}{style}_{str(uid)[:12]}"
    action = bpy.data.actions.get(action_name)
    if action is None:
        action = bpy.data.actions.new(action_name)
    action.use_fake_user = False
    action["mcd_annotation_dummy"] = True
    action["mcd_uid"] = str(uid)
    action["mcd_source"] = str(source)

    # A real one-frame range makes single-frame business intervals visible as
    # one-frame NLA strips.  Keep this defensive for both legacy and layered
    # Action APIs used around Blender 4.3.
    try:
        action.use_frame_range = True
        action.frame_start = 0.0
        action.frame_end = 1.0
    except (AttributeError, TypeError):
        pass
    try:
        if len(action.fcurves) == 0:
            curve = action.fcurves.new(data_path='["mcd_annotation_dummy"]')
            curve.keyframe_points.insert(0.0, 0.0)
            curve.keyframe_points.insert(1.0, 0.0)
    except (AttributeError, RuntimeError, TypeError):
        # Manual action ranges are sufficient if this Blender build uses only
        # the layered Action API.
        pass
    return action


def _strip_style(definition: Mapping, source: str) -> str:
    return "AUTO" if str(source).upper() == "AUTO" else definition.get("style", "HAND")


def ensure_helper_object(scene: bpy.types.Scene) -> bpy.types.Object:
    collection = ensure_internal_collection(scene)
    helper = getattr(scene, "mcd_annotation_helper", None)
    if helper is None:
        helper = next((obj for obj in collection.objects if obj.get("mcd_annotation_helper")), None)
    if helper is None:
        name = f"{HELPER_OBJECT_PREFIX}{_stable_scene_suffix(scene)}"
        helper = bpy.data.objects.get(name)
        if helper is None or not helper.get("mcd_annotation_helper"):
            helper = bpy.data.objects.new(name, None)
        if helper.name not in collection.objects:
            collection.objects.link(helper)

    helper["mcd_annotation_helper"] = True
    helper["mcd_exclude_export"] = True
    helper["mcd_scene"] = scene.name
    helper["mcd_annotation_dummy"] = 0.0
    helper.empty_display_type = "PLAIN_AXES"
    helper.empty_display_size = 0.0001
    helper.hide_render = True
    helper.animation_data_create()
    scene.mcd_annotation_helper = helper
    return helper


def _track_channel(track: bpy.types.NlaTrack) -> str | None:
    try:
        channel = track.get("mcd_channel")
    except (AttributeError, TypeError):
        channel = None
    return channel if channel in TRACK_BY_CHANNEL else CHANNEL_BY_TRACK_NAME.get(track.name)


def _strip_uid(strip: bpy.types.NlaStrip) -> str:
    try:
        value = str(strip.get("mcd_uid", ""))
        if value:
            return value
    except (AttributeError, TypeError):
        pass
    action = getattr(strip, "action", None)
    if action is not None:
        value = str(action.get("mcd_uid", ""))
        if value:
            return value
    parts = str(strip.name).rsplit(STRIP_META_SEPARATOR, 2)
    return parts[-1] if len(parts) == 3 else ""


def _strip_source(strip: bpy.types.NlaStrip) -> str:
    try:
        value = str(strip.get("mcd_source", ""))
        if value:
            return value
    except (AttributeError, TypeError):
        pass
    action = getattr(strip, "action", None)
    if action is not None:
        value = str(action.get("mcd_source", ""))
        if value:
            return value
    parts = str(strip.name).rsplit(STRIP_META_SEPARATOR, 2)
    return parts[-2] if len(parts) == 3 else ""


def _strip_display_name(value: Mapping) -> str:
    source = str(value.get("source", ""))
    origin = "自动" if source == "AUTO" else "手动"
    readable = f"{value['frame_start']}–{value['frame_end']} [{origin}]"
    return readable


def _uid_is_selected(uid: str, selected_uids: Iterable[str]) -> bool:
    return any(uid.startswith(token) or token.startswith(uid) for token in selected_uids if token)


def _set_id_property(value, key: str, item) -> None:
    try:
        value[key] = item
    except (AttributeError, TypeError):
        pass


def rebuild_projection(
    scene: bpy.types.Scene,
    *,
    active_channel: str | None = None,
    selected_uids: Iterable[str] | None = None,
) -> bpy.types.Object:
    """Recreate the two tracks relevant to the active annotation step."""

    helper = ensure_helper_object(scene)
    animation_data = helper.animation_data_create()
    animation_data.action = None
    remembered_uids = set(selected_uids or ())
    if selected_uids is None:
        for old_track in animation_data.nla_tracks:
            for old_strip in old_track.strips:
                if old_strip.select and _strip_uid(old_strip):
                    remembered_uids.add(_strip_uid(old_strip))
    definitions = visible_track_definitions(scene)
    visible = {item["channel"] for item in definitions}
    active_channel = active_channel or scene.mcd_annotation_active_channel
    if active_channel not in visible:
        active_channel = definitions[0]["channel"]
    scene.mcd_annotation_active_channel = active_channel

    for track in list(animation_data.nla_tracks):
        animation_data.nla_tracks.remove(track)

    by_channel = {
        channel: [_record_dict(item) for item in scene.mcd_annotation_ranges if item.channel == channel]
        for channel in visible
    }

    # Blender displays the newest NLA track at the top, so create the desired
    # visual order in reverse: left first, right second.
    for definition in reversed(definitions):
        track = animation_data.nla_tracks.new()
        track.name = definition["track"]
        track.mute = False
        try:
            # Whole strips may be dragged vertically between the two visible
            # tracks or horizontally in time. Edge scaling is discarded when
            # changes are saved; endpoint commands own interval length edits.
            track.lock = False
        except AttributeError:
            pass
        try:
            track.select = definition["channel"] == active_channel
            track.active = definition["channel"] == active_channel
        except AttributeError:
            pass
        _set_id_property(track, "mcd_channel", definition["channel"])
        _set_id_property(track, "mcd_editable", True)

        # Preserve intentional split boundaries.  "Apply and tidy" is the
        # explicit operation that collapses adjacent intervals again.
        values = _normalise_items_with_ids(
            definition["channel"],
            by_channel[definition["channel"]],
            merge_adjacent=False,
        )
        for value in values:
            nla_start, nla_end = inclusive_to_nla(value["frame_start"], value["frame_end"])
            strip_name = _strip_display_name(value)
            style = _strip_style(definition, value.get("source", ""))
            action = _ensure_dummy_action(
                style,
                value["uid"],
                value.get("source", definition["source"]),
            )
            strip = track.strips.new(strip_name, int(nla_start), action)
            strip.name = strip_name
            try:
                strip.action_frame_start = 0.0
                strip.action_frame_end = 1.0
            except (AttributeError, TypeError):
                pass
            strip.frame_start = nla_start
            strip.frame_end = nla_end
            strip.mute = False
            strip.select = _uid_is_selected(value["uid"], remembered_uids)
            try:
                strip.extrapolation = "NOTHING"
                strip.use_sync_length = False
            except (AttributeError, TypeError):
                pass
            _set_id_property(strip, "mcd_uid", value["uid"])
            _set_id_property(strip, "mcd_channel", definition["channel"])
            _set_id_property(strip, "mcd_source", value.get("source", definition["source"]))
            _set_id_property(strip, "mcd_locked", False)
    for action in list(bpy.data.actions):
        if action.get("mcd_annotation_dummy") and action.users == 0:
            bpy.data.actions.remove(action)
    return helper


def _projection_updates(
    scene: bpy.types.Scene,
    *,
    require_all_tracks: bool = True,
) -> dict[str, list[dict]]:
    helper = getattr(scene, "mcd_annotation_helper", None)
    if helper is None or helper.animation_data is None:
        return {}
    updates = {}
    present_channels = set()
    for track in helper.animation_data.nla_tracks:
        channel = _track_channel(track)
        if channel in visible_channels(scene):
            present_channels.add(channel)
        if channel not in visible_channels(scene):
            continue
        updates.setdefault(channel, [])
        for strip in track.strips:
            bounds = nla_to_inclusive(strip.frame_start, strip.frame_end)
            if bounds is None:
                continue
            source = _strip_source(strip)
            auto_channel = AUTO_CHANNEL_BY_WORKING.get(channel)
            if (
                source == "AUTO"
                and auto_channel
                and bounds not in set(get_channel_ranges(scene, auto_channel))
            ):
                source = TRACK_BY_CHANNEL[channel]["source"]
            updates[channel].append(
                {
                    "uid": _strip_uid(strip),
                    "frame_start": bounds[0],
                    "frame_end": bounds[1],
                    "source": source,
                }
            )
    missing = set(visible_channels(scene)) - present_channels
    if missing and require_all_tracks:
        labels = [TRACK_BY_CHANNEL[channel]["label"] for channel in sorted(missing)]
        raise RuntimeError("标注轨道显示不完整，请保存并关闭标注，然后重新打开：" + "、".join(labels))
    return updates


def commit_projection(
    scene: bpy.types.Scene,
    *,
    rebuild: bool = True,
    require_all_tracks: bool = True,
) -> int:
    """Commit native NLA edits, snapping and merging editable tracks."""

    updates = _projection_updates(scene, require_all_tracks=require_all_tracks)
    if not updates:
        return 0
    _replace_channels(scene, updates)
    if rebuild:
        rebuild_projection(scene)
    return sum(len(get_channel_ranges(scene, channel)) for channel in visible_channels(scene))


def _merge_projection_edit_connections(
    channel: str,
    values: Iterable[Mapping],
    *,
    translated_uids: set[str],
    edited_uids: set[str],
    selected_uids: set[str],
) -> list[dict]:
    """Merge only connections affected by native whole-strip translations."""

    ordered = sorted(
        (dict(value) for value in values),
        key=lambda item: (int(item["frame_start"]), int(item["frame_end"])),
    )

    def merge_group(group: list[dict]) -> dict:
        preferred = next(
            (item for item in group if item.get("uid") in selected_uids),
            None,
        )
        preferred = preferred or next(
            (item for item in group if item.get("uid") in edited_uids),
            group[0],
        )
        result = preferred.copy()
        result["channel"] = channel
        result["frame_start"] = min(int(item["frame_start"]) for item in group)
        result["frame_end"] = max(int(item["frame_end"]) for item in group)
        if any(item.get("uid") in edited_uids for item in group) or len(
            {item.get("source", "") for item in group}
        ) > 1:
            result["source"] = TRACK_BY_CHANNEL[channel]["source"]
        return result

    def overlap_groups(group: list[dict]) -> list[list[dict]]:
        result: list[list[dict]] = []
        for item in group:
            if not result or int(item["frame_start"]) > max(
                int(value["frame_end"]) for value in result[-1]
            ):
                result.append([item])
            else:
                result[-1].append(item)
        return result

    adjacent_groups: list[list[dict]] = []
    for item in ordered:
        if not adjacent_groups or int(item["frame_start"]) > max(
            int(value["frame_end"]) for value in adjacent_groups[-1]
        ) + 1:
            adjacent_groups.append([item])
        else:
            adjacent_groups[-1].append(item)

    merged: list[dict] = []
    for group in adjacent_groups:
        if any(item.get("uid") in translated_uids for item in group):
            merged.append(merge_group(group))
            continue
        merged.extend(merge_group(items) for items in overlap_groups(group))
    return merged


def commit_track_reassignments(
    scene: bpy.types.Scene,
    *,
    rebuild: bool = True,
    active_channel: str | None = None,
) -> int:
    """Save native deletes, track moves, and whole-strip time translations.

    A strip whose projected duration differs from its authoritative inclusive
    duration was edge-scaled, so only its track assignment is accepted.  A
    duration-preserving transform is a whole-strip translation and is snapped
    back to inclusive integer frames.
    """

    helper = getattr(scene, "mcd_annotation_helper", None)
    if helper is None or helper.animation_data is None:
        return 0
    visible = set(visible_channels(scene))
    projections = {}
    selected_uids = set()
    present_channels = set()
    for track in helper.animation_data.nla_tracks:
        channel = _track_channel(track)
        if channel not in visible:
            continue
        present_channels.add(channel)
        for strip in track.strips:
            uid = _strip_uid(strip)
            if not uid:
                continue
            projections[uid] = {
                "channel": channel,
                "bounds": nla_to_inclusive(strip.frame_start, strip.frame_end),
                "duration": float(strip.frame_end) - float(strip.frame_start),
            }
            if strip.select:
                selected_uids.add(uid)

    missing_tracks = visible - present_channels
    if missing_tracks:
        labels = [TRACK_BY_CHANNEL[channel]["label"] for channel in sorted(missing_tracks)]
        raise RuntimeError("标注轨道显示不完整，无法安全保存删除：" + "、".join(labels))

    updates = {channel: [] for channel in visible}
    translated_by_channel = {channel: set() for channel in visible}
    edited_by_channel = {channel: set() for channel in visible}
    changed = 0
    for item in scene.mcd_annotation_ranges:
        if item.channel not in visible:
            continue
        value = _record_dict(item)
        projection = projections.get(item.uid)
        if projection is None:
            # The fixed track still exists, but native Delete/X removed this
            # projected UID. Persist that user decision in the working channel.
            changed += 1
            _mark_working_channel_initialized(scene, item.channel)
            continue
        destination = projection["channel"]
        projected_bounds = projection["bounds"]
        expected_duration = int(item.frame_end) - int(item.frame_start) + 1
        projected_duration = (
            projected_bounds[1] - projected_bounds[0] + 1
            if projected_bounds is not None
            else None
        )
        translated = bool(
            projected_bounds is not None
            and projected_duration == expected_duration
            and math.isclose(
                projection["duration"],
                expected_duration,
                rel_tol=0.0,
                abs_tol=1e-4,
            )
            and projected_bounds != (int(item.frame_start), int(item.frame_end))
        )
        reassigned = destination != item.channel
        if translated:
            value["frame_start"], value["frame_end"] = projected_bounds
            translated_by_channel[destination].add(item.uid)
        if translated or reassigned:
            changed += 1
            value["source"] = TRACK_BY_CHANNEL[destination]["source"]
            edited_by_channel[destination].add(item.uid)
            _mark_working_channel_initialized(scene, item.channel)
            _mark_working_channel_initialized(scene, destination)
        if destination != item.channel:
            value["channel"] = destination
        updates[destination].append(value)
    if changed:
        merged_updates = {
            channel: _merge_projection_edit_connections(
                channel,
                values,
                translated_uids=translated_by_channel[channel],
                edited_uids=edited_by_channel[channel],
                selected_uids=selected_uids,
            )
            for channel, values in updates.items()
        }
        _replace_channels(
            scene,
            merged_updates,
            merge_adjacent=False,
            preserve_incoming_metadata=True,
        )
    if rebuild:
        rebuild_projection(
            scene,
            active_channel=active_channel,
            selected_uids=selected_uids,
        )
    return changed


def configure_nla_area(area: bpy.types.Area, scene: bpy.types.Scene | None = None) -> bool:
    """Keep the helper visible without relying on object selection.

    Blender's NLA editor uses a DopeSheet filter.  Disabling selected-only and
    filtering to the internal collection/object means clicking empty space or
    selecting a mesh does not make annotation rows disappear.
    """

    if area is None or area.type != "NLA_EDITOR":
        return False
    scene = scene or bpy.context.scene
    collection = ensure_internal_collection(scene)
    space = area.spaces.active
    dopesheet = getattr(space, "dopesheet", None)
    if dopesheet is None:
        return False
    if hasattr(dopesheet, "show_only_selected"):
        dopesheet.show_only_selected = False
    if hasattr(dopesheet, "show_hidden"):
        dopesheet.show_hidden = True
    if hasattr(dopesheet, "filter_text"):
        # DopeSheet.filter_text is also applied to child NLA channel names in
        # Blender 4.3.  Filtering by the helper object's name therefore kept
        # the object/<No Action> row but hid tracks named "01 | 左手 ...".
        # The internal Collection already isolates the helper, so keep the
        # text search empty and let every fixed track/strip remain visible.
        dopesheet.filter_text = ""
    if hasattr(dopesheet, "use_filter_invert"):
        dopesheet.use_filter_invert = False
    if hasattr(dopesheet, "filter_collection"):
        try:
            dopesheet.filter_collection = collection
        except (AttributeError, TypeError):
            pass
    return True


def configure_all_nla_areas(scene: bpy.types.Scene) -> None:
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "NLA_EDITOR":
                configure_nla_area(area, scene)


def _set_preview_bounds(scene: bpy.types.Scene, frame_start: int, frame_end: int) -> None:
    """Set preview bounds without losing one edge to Blender's clamping."""

    start, end = sorted((int(frame_start), int(frame_end)))
    if start > scene.frame_preview_end:
        scene.frame_preview_end = end
        scene.frame_preview_start = start
    else:
        scene.frame_preview_start = start
        scene.frame_preview_end = end


def clear_pending_marker(scene: bpy.types.Scene) -> None:
    for marker in list(scene.timeline_markers):
        if marker.name.startswith(PENDING_MARKER_PREFIX):
            scene.timeline_markers.remove(marker)


def set_pending_marker(scene: bpy.types.Scene, channel: str, frame: int) -> None:
    clear_pending_marker(scene)
    label = TRACK_BY_CHANNEL[channel]["label"]
    marker = scene.timeline_markers.new(
        f"{PENDING_MARKER_PREFIX}{label} 入点",
        frame=int(frame),
    )
    marker.select = True


def restore_loop_range(scene: bpy.types.Scene) -> bool:
    if not getattr(scene, "mcd_annotation_loop_active", False):
        return False
    scene.use_preview_range = bool(scene.mcd_annotation_previous_use_preview)
    _set_preview_bounds(
        scene,
        scene.mcd_annotation_previous_preview_start,
        scene.mcd_annotation_previous_preview_end,
    )
    scene.mcd_annotation_loop_active = False
    return True


def enter_annotation_mode(context) -> None:
    scene = context.scene
    if not scene.mcd_annotation_mode:
        scene.mcd_annotation_mode = True
        return
    rebuild_projection(scene)


def exit_annotation_mode(scene: bpy.types.Scene) -> None:
    restore_loop_range(scene)
    clear_pending_marker(scene)
    scene.mcd_annotation_has_pending_in = False
    scene.mcd_annotation_mode = False


def _annotation_mode_update(scene: bpy.types.Scene, context) -> None:
    if scene.mcd_annotation_mode:
        rebuild_projection(scene)
        if scene.mcd_annotation_has_pending_in:
            set_pending_marker(
                scene,
                scene.mcd_annotation_pending_channel,
                scene.mcd_annotation_pending_in,
            )
    else:
        restore_loop_range(scene)
        clear_pending_marker(scene)


def is_annotation_context(context) -> bool:
    return bool(
        context
        and context.scene
        and getattr(context.scene, "mcd_annotation_mode", False)
        and context.area
        and context.area.type == "NLA_EDITOR"
        and (context.region is None or context.region.type in {"WINDOW", "UI"})
    )


def _selected_strips(scene: bpy.types.Scene, *, editable_only: bool = False) -> list[tuple]:
    helper = getattr(scene, "mcd_annotation_helper", None)
    if helper is None or helper.animation_data is None:
        return []
    selected = []
    for track in helper.animation_data.nla_tracks:
        channel = _track_channel(track)
        if channel not in TRACK_BY_CHANNEL:
            continue
        if editable_only and channel not in EDITABLE_CHANNELS:
            continue
        for strip in track.strips:
            if strip.select:
                selected.append((track, strip, channel))
    return selected


def select_only_channel(scene: bpy.types.Scene, channel: str) -> bool:
    if channel not in visible_channels(scene):
        return False
    helper = getattr(scene, "mcd_annotation_helper", None)
    if helper is None or helper.animation_data is None:
        return False
    found = False
    for track in helper.animation_data.nla_tracks:
        selected = _track_channel(track) == channel
        try:
            track.select = selected
            track.active = selected
        except AttributeError:
            pass
        if selected:
            found = True
        for strip in track.strips:
            strip.select = False
    if found:
        scene.mcd_annotation_active_channel = channel
    return found


def _context_track_channel(context) -> str | None:
    helper = getattr(context.scene, "mcd_annotation_helper", None)
    if helper is not None and helper.animation_data is not None:
        selected_tracks = {
            _track_channel(track)
            for track in helper.animation_data.nla_tracks
            if track.select and _track_channel(track) in visible_channels(context.scene)
        }
        if len(selected_tracks) == 1:
            return next(iter(selected_tracks))
    selected = _selected_strips(context.scene)
    channels = {item[2] for item in selected}
    if len(channels) == 1:
        return next(iter(channels))
    active_track = getattr(context, "active_nla_track", None)
    if active_track is not None:
        channel = _track_channel(active_track)
        if channel in visible_channels(context.scene):
            return channel
    return None


def _unique_selected_track_channel(scene: bpy.types.Scene) -> str | None:
    """Return the one selected visible projection track, if it is unique."""

    helper = getattr(scene, "mcd_annotation_helper", None)
    if helper is None or helper.animation_data is None:
        return None
    channels = {
        _track_channel(track)
        for track in helper.animation_data.nla_tracks
        if bool(getattr(track, "select", False))
        and _track_channel(track) in visible_channels(scene)
    }
    return next(iter(channels)) if len(channels) == 1 else None


def _editable_channel(context, operator=None) -> str | None:
    channel = _context_track_channel(context) or context.scene.mcd_annotation_active_channel
    if channel not in visible_channels(context.scene):
        if operator is not None:
            operator.report({"WARNING"}, "请选择当前步骤中的左侧或右侧轨道")
        return None
    context.scene.mcd_annotation_active_channel = channel
    return channel


def _endpoint_channel(context, operator=None) -> str | None:
    """Require one explicitly selected left/right track for endpoint edits."""

    channel = _unique_selected_track_channel(context.scene)
    if channel not in visible_channels(context.scene):
        if operator is not None:
            operator.report({"WARNING"}, "请先只选择一条左侧或右侧轨道")
        return None
    context.scene.mcd_annotation_active_channel = channel
    return channel


def _range_record_at_frame(scene: bpy.types.Scene, channel: str, frame: int):
    matches = [
        item
        for item in scene.mcd_annotation_ranges
        if item.channel == channel
        and int(item.frame_start) <= int(frame) <= int(item.frame_end)
    ]
    return matches[0] if len(matches) == 1 else None


def _overwrite_range_endpoint(
    scene: bpy.types.Scene,
    channel: str,
    record,
    frame: int,
    boundary: str,
) -> tuple[int, int, bool]:
    """Move one inclusive endpoint to a frame already inside the interval."""

    old_start = int(record.frame_start)
    old_end = int(record.frame_end)
    frame = int(frame)
    if not (old_start <= frame <= old_end):
        raise ValueError("Endpoint frame must remain inside the interval")
    new_start = frame if boundary == "IN" else old_start
    new_end = frame if boundary == "OUT" else old_end
    changed = (new_start, new_end) != (old_start, old_end)
    values = []
    for item in scene.mcd_annotation_ranges:
        if item.channel != channel:
            continue
        value = _record_dict(item)
        if item.uid == record.uid:
            value["frame_start"] = new_start
            value["frame_end"] = new_end
            if changed:
                value["source"] = TRACK_BY_CHANNEL[channel]["source"]
        values.append(value)
    _replace_channels(scene, {channel: values}, merge_adjacent=False)
    _mark_working_channel_initialized(scene, channel)
    rebuild_projection(scene, active_channel=channel)
    select_only_channel(scene, channel)
    return new_start, new_end, changed


def _set_selected_range_out(
    scene: bpy.types.Scene,
    channel: str,
    uid: str,
    frame: int,
) -> tuple[int, int, bool, int]:
    """Set one selected interval's out-point and merge forward connections."""

    values = [
        _record_dict(item)
        for item in scene.mcd_annotation_ranges
        if item.channel == channel
    ]
    target = next((item for item in values if item["uid"] == uid), None)
    if target is None:
        raise LookupError("找不到所选区间的权威记录")

    frame = int(frame)
    start = int(target["frame_start"])
    old_end = int(target["frame_end"])
    if frame < start:
        raise ValueError(f"当前帧 {frame} 早于所选区间起点 {start}，不能设置出点")

    changed = frame != old_end
    target["frame_end"] = frame
    if changed:
        target["source"] = TRACK_BY_CHANNEL[channel]["source"]

    merged_end = frame
    consumed = {uid}
    if frame > old_end:
        # Only an out-point extension can absorb following intervals.  Keep
        # unrelated adjacent ranges elsewhere on the track untouched.
        while True:
            connected = [
                item
                for item in values
                if item["uid"] not in consumed
                and int(item["frame_start"]) >= start
                and int(item["frame_start"]) <= merged_end + 1
            ]
            if not connected:
                break
            for item in connected:
                consumed.add(item["uid"])
                merged_end = max(merged_end, int(item["frame_end"]))
        if len(consumed) > 1:
            target["frame_end"] = merged_end
            target["source"] = TRACK_BY_CHANNEL[channel]["source"]

    replacements = [item for item in values if item["uid"] not in consumed or item["uid"] == uid]
    _replace_channels(scene, {channel: replacements}, merge_adjacent=False)
    _mark_working_channel_initialized(scene, channel)
    rebuild_projection(
        scene,
        active_channel=channel,
        selected_uids={uid},
    )
    return start, int(target["frame_end"]), changed, len(consumed) - 1


def _set_selected_range_in(
    scene: bpy.types.Scene,
    channel: str,
    uid: str,
    frame: int,
) -> tuple[int, int, bool, int]:
    """Set one selected interval's in-point and merge backward connections."""

    values = [
        _record_dict(item)
        for item in scene.mcd_annotation_ranges
        if item.channel == channel
    ]
    target = next((item for item in values if item["uid"] == uid), None)
    if target is None:
        raise LookupError("找不到所选区间的权威记录")

    frame = int(frame)
    old_start = int(target["frame_start"])
    end = int(target["frame_end"])
    if frame > end:
        raise ValueError(f"当前帧 {frame} 晚于所选区间终点 {end}，不能设置入点")

    changed = frame != old_start
    target["frame_start"] = frame
    if changed:
        target["source"] = TRACK_BY_CHANNEL[channel]["source"]

    merged_start = frame
    consumed = {uid}
    if frame < old_start:
        while True:
            connected = [
                item
                for item in values
                if item["uid"] not in consumed
                and int(item["frame_end"]) <= end
                and int(item["frame_end"]) >= merged_start - 1
            ]
            if not connected:
                break
            for item in connected:
                consumed.add(item["uid"])
                merged_start = min(merged_start, int(item["frame_start"]))
        if len(consumed) > 1:
            target["frame_start"] = merged_start
            target["source"] = TRACK_BY_CHANNEL[channel]["source"]

    replacements = [item for item in values if item["uid"] not in consumed or item["uid"] == uid]
    _replace_channels(scene, {channel: replacements}, merge_adjacent=False)
    _mark_working_channel_initialized(scene, channel)
    rebuild_projection(
        scene,
        active_channel=channel,
        selected_uids={uid},
    )
    return int(target["frame_start"]), end, changed, len(consumed) - 1


def _append_authoritative_range(scene: bpy.types.Scene, channel: str, start: int, end: int):
    current = [
        {
            "uid": item.uid,
            "frame_start": item.frame_start,
            "frame_end": item.frame_end,
            "source": item.source,
        }
        for item in scene.mcd_annotation_ranges
        if item.channel == channel
    ]
    current.append(
        {
            "frame_start": start,
            "frame_end": end,
            "source": TRACK_BY_CHANNEL[channel]["source"],
        }
    )
    _replace_channels(scene, {channel: current})
    _mark_working_channel_initialized(scene, channel)
    low, high = sorted((int(start), int(end)))
    created = next(
        (
            item
            for item in scene.mcd_annotation_ranges
            if item.channel == channel
            and int(item.frame_start) <= low
            and int(item.frame_end) >= high
        ),
        None,
    )
    selected = {created.uid} if created is not None else set()
    rebuild_projection(scene, active_channel=channel, selected_uids=selected)
    return created


class _MCDAnnotationOperator:
    @classmethod
    def poll(cls, context):
        return is_annotation_context(context)


class MCD_OT_annotation_add_preview(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_add_preview"
    bl_label = "添加 Preview Range"
    bl_description = "把当前 Preview Range 作为 inclusive 区间添加到活动可编辑轨道"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        channel = _editable_channel(context, self)
        if channel is None:
            return {"CANCELLED"}
        if not scene.use_preview_range:
            self.report({"WARNING"}, "请先启用并设置 Preview Range")
            return {"CANCELLED"}
        _append_authoritative_range(scene, channel, scene.frame_preview_start, scene.frame_preview_end)
        self.report({"INFO"}, f"已添加 {scene.frame_preview_start}–{scene.frame_preview_end}")
        return {"FINISHED"}


class MCD_OT_annotation_set_channel(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_set_channel"
    bl_label = "选择标注轨道"
    bl_options = {"INTERNAL"}

    channel: StringProperty()

    def execute(self, context):
        if not select_only_channel(context.scene, self.channel):
            self.report({"WARNING"}, "该轨道不属于当前标注步骤")
            return {"CANCELLED"}
        return {"FINISHED"}


class MCD_OT_annotation_mark_in(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_mark_in"
    bl_label = "标记或覆盖入点"
    bl_description = "选中色块时设置其入点；未选中色块时记录新区间入点"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        selected = _selected_strips(scene, editable_only=True)
        if len(selected) > 1:
            self.report({"WARNING"}, "一次只能修改一个区间；请取消多余选择")
            return {"CANCELLED"}
        if len(selected) == 1:
            _track, strip, selected_channel = selected[0]
            selected_uid = _strip_uid(strip)
            if not selected_uid:
                self.report({"ERROR"}, "所选区间缺少稳定 ID，请重新打开标注编辑器")
                return {"CANCELLED"}
            commit_track_reassignments(
                scene,
                active_channel=selected_channel,
            )
            try:
                start, end, changed, merged_count = _set_selected_range_in(
                    scene,
                    selected_channel,
                    selected_uid,
                    scene.frame_current,
                )
            except (LookupError, ValueError) as exc:
                self.report({"WARNING"}, str(exc))
                return {"CANCELLED"}
            scene.mcd_annotation_has_pending_in = False
            clear_pending_marker(scene)
            if merged_count:
                message = f"入点已扩展并合并 {merged_count} 个相连区间：{start}–{end}"
            elif changed:
                message = f"所选区间入点已改为 {start}（区间 {start}–{end}）"
            else:
                message = f"所选区间入点已经是 {start}"
            self.report({"INFO"}, message)
            return {"FINISHED"}

        channel = _endpoint_channel(context, self)
        if channel is None:
            return {"CANCELLED"}
        commit_track_reassignments(scene, active_channel=channel)
        scene.mcd_annotation_has_pending_in = True
        scene.mcd_annotation_pending_in = scene.frame_current
        scene.mcd_annotation_pending_channel = channel
        select_only_channel(scene, channel)
        set_pending_marker(scene, channel, scene.frame_current)
        self.report({"INFO"}, f"入点：{scene.frame_current}")
        return {"FINISHED"}


class MCD_OT_annotation_cancel_in(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_cancel_in"
    bl_label = "取消入点"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return is_annotation_context(context) and bool(context.scene.mcd_annotation_has_pending_in)

    def execute(self, context):
        context.scene.mcd_annotation_has_pending_in = False
        clear_pending_marker(context.scene)
        return {"FINISHED"}


class MCD_OT_annotation_mark_out(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_mark_out"
    bl_label = "标记或覆盖出点"
    bl_description = "选中色块时设置其出点；未选中色块时完成待创建区间"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        selected = _selected_strips(scene, editable_only=True)
        if len(selected) > 1:
            self.report({"WARNING"}, "一次只能修改一个区间；请取消多余选择")
            return {"CANCELLED"}
        if len(selected) == 1:
            _track, strip, selected_channel = selected[0]
            selected_uid = _strip_uid(strip)
            if not selected_uid:
                self.report({"ERROR"}, "所选区间缺少稳定 ID，请重新打开标注编辑器")
                return {"CANCELLED"}
            commit_track_reassignments(
                scene,
                active_channel=selected_channel,
            )
            try:
                start, end, changed, merged_count = _set_selected_range_out(
                    scene,
                    selected_channel,
                    selected_uid,
                    scene.frame_current,
                )
            except (LookupError, ValueError) as exc:
                self.report({"WARNING"}, str(exc))
                return {"CANCELLED"}
            scene.mcd_annotation_has_pending_in = False
            clear_pending_marker(scene)
            if merged_count:
                message = f"出点已扩展并合并 {merged_count} 个相连区间：{start}–{end}"
            elif changed:
                message = f"所选区间出点已改为 {end}（区间 {start}–{end}）"
            else:
                message = f"所选区间出点已经是 {end}"
            self.report({"INFO"}, message)
            return {"FINISHED"}

        if not scene.mcd_annotation_has_pending_in:
            self.report({"WARNING"}, "请先标记入点")
            return {"CANCELLED"}
        channel = scene.mcd_annotation_pending_channel
        if channel not in visible_channels(scene):
            self.report({"WARNING"}, "待创建区间不属于当前标注步骤；请重新标记入点")
            return {"CANCELLED"}

        commit_track_reassignments(scene, active_channel=channel)
        start, end = scene.mcd_annotation_pending_in, scene.frame_current
        _append_authoritative_range(scene, channel, start, end)
        scene.mcd_annotation_has_pending_in = False
        clear_pending_marker(scene)
        # Creating a range is a batch operation: leave the channel active but
        # nothing selected, so the next mark-in starts the following range
        # without a detour through the empty space and the track header.  This
        # is deliberately only the creation path - revising an existing strip
        # keeps its selection, which is what makes chained edits work there.
        rebuild_projection(scene, active_channel=channel, selected_uids=())
        self.report({"INFO"}, f"已创建 {min(start, end)}–{max(start, end)}")
        return {"FINISHED"}


class MCD_OT_annotation_split_current(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_split_current"
    bl_label = "在当前帧拆分"
    bl_description = "把所选区间 [start, end] 拆成 [start, current-1] 和 [current, end]"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return is_annotation_context(context) and len(_selected_strips(context.scene, editable_only=True)) == 1

    def execute(self, context):
        scene = context.scene
        selected = _selected_strips(scene, editable_only=True)
        if len(selected) != 1:
            self.report({"WARNING"}, "请只选择一个要拆分的区间")
            return {"CANCELLED"}
        _track, strip, channel = selected[0]
        target_uid = _strip_uid(strip)
        if not target_uid:
            self.report({"ERROR"}, "所选区间缺少稳定 ID，请重新打开标注编辑器")
            return {"CANCELLED"}

        commit_track_reassignments(scene, active_channel=channel)
        target = next(
            (
                _record_dict(item)
                for item in scene.mcd_annotation_ranges
                if item.channel == channel and item.uid == target_uid
            ),
            None,
        )
        if target is None:
            self.report({"ERROR"}, "无法定位所选区间的权威记录")
            return {"CANCELLED"}
        start = int(target["frame_start"])
        end = int(target["frame_end"])
        frame = int(scene.frame_current)
        if not (start < frame <= end):
            self.report({"WARNING"}, f"当前帧必须在 {start + 1}–{end} 内")
            return {"CANCELLED"}

        replacement: list[dict] = []
        right_uid = _new_uid()
        for item in scene.mcd_annotation_ranges:
            if item.channel != channel:
                continue
            value = _record_dict(item)
            if item.uid == target_uid:
                replacement.append(
                    {
                        "uid": target_uid,
                        "frame_start": start,
                        "frame_end": frame - 1,
                        "source": TRACK_BY_CHANNEL[channel]["source"],
                    }
                )
                replacement.append(
                    {
                        "uid": right_uid,
                        "frame_start": frame,
                        "frame_end": end,
                        "source": TRACK_BY_CHANNEL[channel]["source"],
                    }
                )
            else:
                replacement.append(value)
        _replace_channels(scene, {channel: replacement}, merge_adjacent=False)
        _mark_working_channel_initialized(scene, channel)
        rebuild_projection(
            scene,
            active_channel=channel,
            selected_uids={target_uid, right_uid},
        )
        self.report({"INFO"}, f"已拆分为 {start}–{frame - 1} 和 {frame}–{end}")
        return {"FINISHED"}


class MCD_OT_annotation_delete(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_delete"
    bl_label = "删除所选区间"
    bl_description = "删除选中的区间色块"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return is_annotation_context(context) and bool(_selected_strips(context.scene, editable_only=True))

    def execute(self, context):
        selected = _selected_strips(context.scene, editable_only=True)
        selected_uids = {_strip_uid(strip) for _track, strip, _channel in selected}
        selected_channels = {channel for _track, _strip, channel in selected}
        active_channel = selected[0][2] if selected else context.scene.mcd_annotation_active_channel
        commit_track_reassignments(context.scene, rebuild=False)
        updates = {channel: [] for channel in visible_channels(context.scene)}
        for item in context.scene.mcd_annotation_ranges:
            if item.channel in updates and item.uid not in selected_uids:
                updates[item.channel].append(_record_dict(item))
        _replace_channels(context.scene, updates, merge_adjacent=False)
        for channel in selected_channels:
            _mark_working_channel_initialized(context.scene, channel)
        rebuild_projection(context.scene, active_channel=active_channel)
        self.report({"INFO"}, f"已删除 {len(selected)} 个区间")
        return {"FINISHED"}


class MCD_OT_annotation_locate(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_locate"
    bl_label = "定位所选区间"
    bl_description = "移动时间指针到区间开头，并在 NLA 中聚焦所选条带"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return is_annotation_context(context) and bool(_selected_strips(context.scene))

    def execute(self, context):
        active_channel = _context_track_channel(context)
        commit_track_reassignments(context.scene, active_channel=active_channel)
        selected = _selected_strips(context.scene)
        bounds = nla_to_inclusive(selected[0][1].frame_start, selected[0][1].frame_end)
        if bounds is None:
            return {"CANCELLED"}
        context.scene.frame_set(bounds[0])
        try:
            bpy.ops.nla.view_selected()
        except RuntimeError:
            self.report({"WARNING"}, "已定位时间指针，但当前区域无法缩放视图")
        return {"FINISHED"}


class MCD_OT_annotation_loop_selected(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_loop_selected"
    bl_label = "循环检查所选区间"
    bl_description = "设置临时 Preview Range；再次执行可恢复原来的范围"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        active_channel = _context_track_channel(context)
        commit_track_reassignments(scene, active_channel=active_channel)
        if scene.mcd_annotation_loop_active:
            restore_loop_range(scene)
            self.report({"INFO"}, "已恢复原 Preview Range")
            return {"FINISHED"}

        selected = _selected_strips(scene)
        if not selected:
            self.report({"WARNING"}, "请先选择一个区间")
            return {"CANCELLED"}
        bounds = nla_to_inclusive(selected[0][1].frame_start, selected[0][1].frame_end)
        if bounds is None:
            return {"CANCELLED"}
        scene.mcd_annotation_previous_use_preview = scene.use_preview_range
        scene.mcd_annotation_previous_preview_start = scene.frame_preview_start
        scene.mcd_annotation_previous_preview_end = scene.frame_preview_end
        scene.use_preview_range = True
        _set_preview_bounds(scene, bounds[0], bounds[1])
        scene.frame_set(bounds[0])
        scene.mcd_annotation_loop_active = True
        self.report({"INFO"}, f"循环范围：{bounds[0]}–{bounds[1]}（按空格播放）")
        return {"FINISHED"}


class MCD_OT_annotation_apply(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_apply"
    bl_label = "应用并整理"
    bl_description = "提交原生 NLA 移动/缩放，取整并合并重叠或相邻区间"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        count = commit_projection(context.scene, rebuild=True)
        self.report({"INFO"}, f"已提交并整理 {count} 个可编辑区间")
        return {"FINISHED"}


class MCD_OT_annotation_rebuild(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_rebuild"
    bl_label = "重建标注轨道"
    bl_description = "丢弃未提交的 NLA 改动，并从 Scene 权威数据重建辅助对象和轨道"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        rebuild_projection(context.scene)
        configure_nla_area(context.area, context.scene)
        self.report({"INFO"}, "标注轨道已重建")
        return {"FINISHED"}


class MCD_OT_annotation_configure_view(_MCDAnnotationOperator, bpy.types.Operator):
    bl_idname = "mcd.annotation_configure_view"
    bl_label = "固定标注视图"
    bl_description = "重新应用内部 Collection 和辅助对象过滤，不依赖当前物体选择"
    bl_options = {"REGISTER"}

    def execute(self, context):
        configure_nla_area(context.area, context.scene)
        return {"FINISHED"}


CLASSES = (
    MCD_AnnotationRange,
    MCD_OT_annotation_set_channel,
    MCD_OT_annotation_mark_in,
    MCD_OT_annotation_cancel_in,
    MCD_OT_annotation_mark_out,
    MCD_OT_annotation_split_current,
    MCD_OT_annotation_delete,
    MCD_OT_annotation_locate,
    MCD_OT_annotation_loop_selected,
)


_SCENE_PROPERTIES = (
    "mcd_annotation_ranges",
    "mcd_annotation_mode",
    "mcd_annotation_active_channel",
    "mcd_annotation_helper",
    "mcd_annotation_has_pending_in",
    "mcd_annotation_pending_in",
    "mcd_annotation_pending_channel",
    "mcd_annotation_loop_active",
    "mcd_annotation_previous_use_preview",
    "mcd_annotation_previous_preview_start",
    "mcd_annotation_previous_preview_end",
)


def register_properties() -> None:
    Scene = bpy.types.Scene
    Scene.mcd_annotation_ranges = CollectionProperty(type=MCD_AnnotationRange)
    Scene.mcd_annotation_mode = BoolProperty(
        name="MoCap Doctor 标注模式",
        description="启用锁定的 NLA 区间投影和端点快捷键",
        default=False,
        update=_annotation_mode_update,
    )
    Scene.mcd_annotation_active_channel = EnumProperty(
        name="编辑轨道",
        items=EDITABLE_CHANNEL_ENUM_ITEMS,
        default=CHANNEL_HAND_L_MANUAL,
    )
    Scene.mcd_annotation_helper = PointerProperty(type=bpy.types.Object, options={"HIDDEN"})
    Scene.mcd_annotation_has_pending_in = BoolProperty(default=False, options={"HIDDEN"})
    Scene.mcd_annotation_pending_in = IntProperty(default=0, options={"HIDDEN"})
    Scene.mcd_annotation_pending_channel = EnumProperty(
        items=EDITABLE_CHANNEL_ENUM_ITEMS,
        default=CHANNEL_HAND_L_MANUAL,
        options={"HIDDEN"},
    )
    Scene.mcd_annotation_loop_active = BoolProperty(default=False, options={"HIDDEN"})
    Scene.mcd_annotation_previous_use_preview = BoolProperty(default=False, options={"HIDDEN"})
    Scene.mcd_annotation_previous_preview_start = IntProperty(default=0, options={"HIDDEN"})
    Scene.mcd_annotation_previous_preview_end = IntProperty(default=0, options={"HIDDEN"})


def unregister_properties() -> None:
    # Extension disable/uninstall can run while Blender exposes _RestrictData.
    for scene in getattr(bpy.data, "scenes", ()):
        if hasattr(scene, "mcd_annotation_loop_active"):
            try:
                restore_loop_range(scene)
            except Exception as exc:
                print(f"[MoCap Doctor] loop-range restore during unregister failed: {exc}")
    for name in reversed(_SCENE_PROPERTIES):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
