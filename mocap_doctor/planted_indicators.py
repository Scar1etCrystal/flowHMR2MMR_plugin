"""Ephemeral viewport-only planted indicators for contact annotation.

The indicators deliberately live outside the NLA helper collection: the NLA
editor filters to that collection, so mixing viewport helpers into it would
add unrelated object rows to the two-track annotation view.
"""

from __future__ import annotations

import math

import bpy
from bpy.app.handlers import persistent

from .presets import SOURCE_BONES, resolve_source_profile


COLLECTION_PREFIX = ".MCD_Planted_Indicators_"
OBJECT_PREFIX = ".MCD_Planted_"
MESH_PREFIX = ".MCD_Planted_Mesh_"
MATERIAL_PREFIX = ".MCD_Planted_Material_"
FOLLOW_CONSTRAINT = "MCD Follow Source Foot"
ASSET_VERSION = 3
EMISSION_STRENGTH = 5.0

MARKER_MAJOR_SEGMENTS = 16
MARKER_MINOR_SEGMENTS = 4
MARKER_MAJOR_RADIUS = 0.80
MARKER_TUBE_RADIUS = 0.16
MARKER_INNER_RADIUS = MARKER_MAJOR_RADIUS - MARKER_TUBE_RADIUS


def _torus_geometry():
    """Return a context-free, low-poly horizontal ring with an open center."""

    vertices = []
    faces = []
    for major_index in range(MARKER_MAJOR_SEGMENTS):
        major_angle = math.tau * major_index / MARKER_MAJOR_SEGMENTS
        cos_major = math.cos(major_angle)
        sin_major = math.sin(major_angle)
        for minor_index in range(MARKER_MINOR_SEGMENTS):
            minor_angle = math.tau * minor_index / MARKER_MINOR_SEGMENTS
            radius = MARKER_MAJOR_RADIUS + MARKER_TUBE_RADIUS * math.cos(minor_angle)
            vertices.append(
                (
                    radius * cos_major,
                    radius * sin_major,
                    MARKER_TUBE_RADIUS * math.sin(minor_angle),
                )
            )
    for major_index in range(MARKER_MAJOR_SEGMENTS):
        next_major = (major_index + 1) % MARKER_MAJOR_SEGMENTS
        for minor_index in range(MARKER_MINOR_SEGMENTS):
            next_minor = (minor_index + 1) % MARKER_MINOR_SEGMENTS
            faces.append(
                (
                    major_index * MARKER_MINOR_SEGMENTS + minor_index,
                    next_major * MARKER_MINOR_SEGMENTS + minor_index,
                    next_major * MARKER_MINOR_SEGMENTS + next_minor,
                    major_index * MARKER_MINOR_SEGMENTS + next_minor,
                )
            )
    return tuple(vertices), tuple(faces)


MARKER_VERTICES, MARKER_FACES = _torus_geometry()

# Annotation channel identifiers are stable project/report IDs.  Keep this
# module independent from annotation.py so its lifecycle can be registered and
# rolled back without introducing an import cycle.
CONFIG = {
    "L": {
        "channel": "FOOT_L_EFFECTIVE",
        "bone": SOURCE_BONES["left_foot"],
        "label": "左脚 PLANTED",
        "color": (0.10, 0.65, 1.00, 1.00),
    },
    "R": {
        "channel": "FOOT_R_EFFECTIVE",
        "bone": SOURCE_BONES["right_foot"],
        "label": "右脚 PLANTED",
        "color": (1.00, 0.25, 0.10, 1.00),
    },
}


_ACTIVE_SCENES: set[int] = set()
_UPDATING = False


def _scene_suffix(scene: bpy.types.Scene) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in scene.name)
    return cleaned[:32] or "Scene"


def _contacts_mode_active(scene: bpy.types.Scene) -> bool:
    settings = getattr(scene, "mocap_doctor", None)
    return bool(
        settings
        and getattr(scene, "mcd_annotation_mode", False)
        and getattr(settings, "annotation_step_id", "") == "contacts"
    )


def source_foot_bones(source: bpy.types.Object) -> dict[str, str]:
    """Resolve the L/R foot bones for whichever frontend produced the rig.

    CONFIG keeps the channel/label/colour metadata; the bone names depend on the
    source profile, so refresh them here before anything reads CONFIG.
    """

    settings = getattr(bpy.context.scene, "mocap_doctor", None)
    requested = getattr(settings, "source_profile", "AUTO") if settings else "AUTO"
    maps = resolve_source_profile(source, requested)
    if maps is None:
        raise RuntimeError("无法识别源骨架，无法显示 planted 脚部标志")
    bones = maps["bones"]
    resolved = {"L": bones["left_foot"], "R": bones["right_foot"]}
    for side, name in resolved.items():
        CONFIG[side]["bone"] = name
    return resolved


def require_source(scene: bpy.types.Scene) -> bpy.types.Object:
    """Return the configured source armature after foot-bone validation."""

    settings = getattr(scene, "mocap_doctor", None)
    source = getattr(settings, "source_armature", None) if settings else None
    if source is None:
        raise RuntimeError("缺少源骨架，无法显示 planted 脚部标志")
    if source.type != "ARMATURE":
        raise RuntimeError(f"{source.name} 不是 Armature，无法显示 planted 脚部标志")
    resolved = source_foot_bones(source)
    missing = [name for name in resolved.values() if name not in source.pose.bones]
    if missing:
        raise RuntimeError("源骨架缺少脚部骨骼：" + "、".join(missing))
    return source


def _indicator_collection(scene: bpy.types.Scene, *, create: bool) -> bpy.types.Collection | None:
    collections = getattr(bpy.data, "collections", None)
    if collections is None:
        return None
    name = f"{COLLECTION_PREFIX}{_scene_suffix(scene)}"
    collection = collections.get(name)
    if collection is None and create:
        collection = collections.new(name)
    if collection is None:
        return None
    collection["mcd_planted_indicator_collection"] = True
    collection["mcd_scene"] = scene.name
    # Material Preview follows render visibility for collections in Blender
    # 4.3.  Keep this ephemeral helper render-visible while annotation is
    # active; export exclusion is handled by its tag and explicit cleanup.
    collection.hide_render = False
    if collection.name not in scene.collection.children:
        scene.collection.children.link(collection)
    return collection


def _owned_objects(scene: bpy.types.Scene | None = None) -> list[bpy.types.Object]:
    objects = getattr(bpy.data, "objects", ())
    result = []
    for obj in tuple(objects):
        try:
            if not bool(obj.get("mcd_planted_indicator")):
                continue
            if scene is not None:
                linked = obj.name in scene.objects
                tagged = str(obj.get("mcd_scene", "")) == scene.name
                if not linked and not tagged:
                    continue
            result.append(obj)
        except (AttributeError, ReferenceError, TypeError):
            continue
    return result


def _owned_data_blocks(collection, scene: bpy.types.Scene | None = None) -> list:
    result = []
    for item in tuple(collection or ()):
        try:
            if not bool(item.get("mcd_planted_indicator_asset")):
                continue
            if scene is not None and str(item.get("mcd_scene", "")) != scene.name:
                continue
            result.append(item)
        except (AttributeError, ReferenceError, TypeError):
            continue
    return result


def _owned_meshes(scene: bpy.types.Scene | None = None) -> list[bpy.types.Mesh]:
    return _owned_data_blocks(getattr(bpy.data, "meshes", ()), scene)


def _owned_materials(scene: bpy.types.Scene | None = None) -> list[bpy.types.Material]:
    return _owned_data_blocks(getattr(bpy.data, "materials", ()), scene)


def _remove_object(obj: bpy.types.Object) -> bool:
    objects = getattr(bpy.data, "objects", None)
    if objects is None:
        return False
    try:
        objects.remove(obj, do_unlink=True)
        return True
    except (ReferenceError, RuntimeError, TypeError, ValueError):
        return False


def cleanup(scene: bpy.types.Scene | None = None) -> int:
    """Remove owned indicators and empty owned collections."""

    if scene is None:
        _ACTIVE_SCENES.clear()
    else:
        try:
            _ACTIVE_SCENES.discard(scene.as_pointer())
        except ReferenceError:
            pass

    owned_objects = _owned_objects(scene)
    referenced_meshes = {
        obj.data
        for obj in owned_objects
        if obj.type == "MESH"
        and obj.data is not None
        and bool(obj.data.get("mcd_planted_indicator_asset"))
    }
    referenced_materials = {
        material
        for mesh in referenced_meshes
        for material in mesh.materials
        if material is not None and bool(material.get("mcd_planted_indicator_asset"))
    }
    removed = sum(1 for obj in owned_objects if _remove_object(obj))
    meshes = getattr(bpy.data, "meshes", None)
    if meshes is not None:
        for mesh in set(_owned_meshes(scene)) | referenced_meshes:
            if mesh.users == 0:
                try:
                    meshes.remove(mesh)
                except (ReferenceError, RuntimeError, TypeError, ValueError):
                    pass
    materials = getattr(bpy.data, "materials", None)
    if materials is not None:
        for material in set(_owned_materials(scene)) | referenced_materials:
            if material.users == 0:
                try:
                    materials.remove(material)
                except (ReferenceError, RuntimeError, TypeError, ValueError):
                    pass

    collections = getattr(bpy.data, "collections", None)
    if collections is None:
        return removed
    for collection in tuple(collections):
        try:
            if not bool(collection.get("mcd_planted_indicator_collection")):
                continue
            if scene is not None:
                linked = collection.name in scene.collection.children
                tagged = str(collection.get("mcd_scene", "")) == scene.name
                if not linked and not tagged:
                    continue
            if len(collection.objects) == 0:
                collections.remove(collection)
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
            continue
    return removed


def _display_size(source: bpy.types.Object, bone_name: str) -> float:
    bone = source.data.bones.get(bone_name)
    if bone is None:
        return 0.10
    try:
        world_scale = max(abs(float(value)) for value in source.matrix_world.to_scale())
    except (AttributeError, TypeError, ValueError):
        world_scale = 1.0
    world_length = float(bone.length) * max(world_scale, 1.0e-6)
    return max(0.06, min(0.20, world_length * 1.25))


def _follow_constraint(obj: bpy.types.Object):
    return next(
        (
            constraint
            for constraint in obj.constraints
            if constraint.type == "COPY_LOCATION" and constraint.name == FOLLOW_CONSTRAINT
        ),
        None,
    )


def _matches(obj: bpy.types.Object, source: bpy.types.Object, side: str) -> bool:
    config = CONFIG[side]
    constraint = _follow_constraint(obj)
    return bool(
        obj.type == "MESH"
        and obj.data
        and bool(obj.data.get("mcd_planted_indicator_asset"))
        and obj.active_material
        and bool(obj.active_material.get("mcd_planted_indicator_asset"))
        and constraint
        and constraint.target == source
        and constraint.subtarget == config["bone"]
        and str(obj.get("mcd_side", "")) == side
    )


def _asset_name(prefix: str, scene: bpy.types.Scene, side: str) -> str:
    return f"{prefix}{side}_{_scene_suffix(scene)}"


def _tag_asset(item, scene: bpy.types.Scene, side: str) -> None:
    item["mcd_planted_indicator_asset"] = True
    item["mcd_scene"] = scene.name
    item["mcd_side"] = side
    item["mcd_asset_version"] = ASSET_VERSION


def _ensure_mesh(scene: bpy.types.Scene, side: str) -> bpy.types.Mesh:
    name = _asset_name(MESH_PREFIX, scene, side)
    mesh = next(
        (
            item
            for item in _owned_meshes(scene)
            if str(item.get("mcd_side", "")) == side
        ),
        None,
    )
    if mesh is None:
        mesh = bpy.data.meshes.new(name)
    _tag_asset(mesh, scene, side)
    if int(mesh.get("mcd_geometry_version", 0)) != ASSET_VERSION or not mesh.polygons:
        mesh.clear_geometry()
        mesh.from_pydata(MARKER_VERTICES, (), MARKER_FACES)
        mesh.update()
        mesh["mcd_geometry_version"] = ASSET_VERSION
    return mesh


def _ensure_material(scene: bpy.types.Scene, side: str) -> bpy.types.Material:
    config = CONFIG[side]
    name = _asset_name(MATERIAL_PREFIX, scene, side)
    material = next(
        (
            item
            for item in _owned_materials(scene)
            if str(item.get("mcd_side", "")) == side
        ),
        None,
    )
    if material is None:
        material = bpy.data.materials.new(name)
    _tag_asset(material, scene, side)
    material.diffuse_color = config["color"]
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = config["color"]
    emission.inputs["Strength"].default_value = EMISSION_STRENGTH
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _configure_object(
    obj: bpy.types.Object,
    scene: bpy.types.Scene,
    source: bpy.types.Object,
    side: str,
) -> None:
    config = CONFIG[side]
    obj["mcd_planted_indicator"] = True
    obj["mcd_exclude_export"] = True
    obj["mcd_scene"] = scene.name
    obj["mcd_side"] = side
    obj["mcd_source_bone"] = config["bone"]
    mesh = _ensure_mesh(scene, side)
    material = _ensure_material(scene, side)
    if obj.data != mesh:
        obj.data = mesh
    mesh.materials.clear()
    mesh.materials.append(material)
    size = _display_size(source, config["bone"])
    obj.scale = (size, size, size)
    obj.color = config["color"]
    obj.display_type = "SOLID"
    obj.show_name = True
    obj.show_in_front = True
    obj.hide_select = True
    obj.hide_render = False
    obj.hide_viewport = True
    display = getattr(obj, "display", None)
    if display is not None and hasattr(display, "show_shadows"):
        display.show_shadows = False
    for attribute in (
        "visible_camera",
        "visible_diffuse",
        "visible_glossy",
        "visible_transmission",
        "visible_volume_scatter",
        "visible_shadow",
    ):
        if hasattr(obj, attribute):
            setattr(obj, attribute, True)
    if hasattr(obj, "show_line"):
        obj.show_line = False
    for constraint in list(obj.constraints):
        obj.constraints.remove(constraint)
    constraint = obj.constraints.new("COPY_LOCATION")
    constraint.name = FOLLOW_CONSTRAINT
    constraint.target = source
    constraint.subtarget = config["bone"]
    if hasattr(constraint, "head_tail"):
        constraint.head_tail = 0.5
    if hasattr(constraint, "target_space"):
        constraint.target_space = "WORLD"
    if hasattr(constraint, "owner_space"):
        constraint.owner_space = "WORLD"


def ensure(scene: bpy.types.Scene) -> dict[str, bpy.types.Object]:
    """Create exactly one owned, bone-following viewport marker per foot."""

    source = require_source(scene)
    collection = _indicator_collection(scene, create=True)
    if collection is None:
        raise RuntimeError("无法创建 planted 指示标志 Collection")

    by_side: dict[str, bpy.types.Object] = {}
    for obj in _owned_objects(scene):
        side = str(obj.get("mcd_side", ""))
        if obj.type != "MESH" or side not in CONFIG or side in by_side:
            _remove_object(obj)
            continue
        by_side[side] = obj

    # Remove assets left behind by an older Empty-based or interrupted session.
    for mesh in _owned_meshes(scene):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in _owned_materials(scene):
        if material.users == 0:
            bpy.data.materials.remove(material)

    objects = getattr(bpy.data, "objects", None)
    if objects is None:
        raise RuntimeError("Blender 当前不允许创建 planted 指示标志")
    for side, config in CONFIG.items():
        obj = by_side.get(side)
        if obj is None:
            mesh = _ensure_mesh(scene, side)
            obj = objects.new(
                f"{OBJECT_PREFIX}{config['label']}_{_scene_suffix(scene)}",
                mesh,
            )
            collection.objects.link(obj)
            by_side[side] = obj
            _configure_object(obj, scene, source, side)
        else:
            if obj.name not in collection.objects:
                collection.objects.link(obj)
            if not _matches(obj, source, side):
                _configure_object(obj, scene, source, side)
            else:
                # P4: only write what differs - every property write tags the
                # object, and refresh() runs on each frame change and inside
                # depsgraph_update_post (a self-triggered re-evaluation loop)
                if obj.hide_render:
                    obj.hide_render = False
                if not obj.hide_select:
                    obj.hide_select = True
                if not obj.show_in_front:
                    obj.show_in_front = True
    return by_side


def _frame_in_channel(scene: bpy.types.Scene, channel: str) -> bool:
    frame = int(scene.frame_current)
    try:
        from . import annotation as annotation_module
        ranges = sorted(annotation_module.live_channel_ranges(scene, channel))
    except Exception:  # noqa: BLE001 - 标注模块不可用时退回已保存区间
        ranges = sorted(
            (
                int(item.frame_start),
                int(item.frame_end),
            )
            for item in getattr(scene, "mcd_annotation_ranges", ())
            if item.channel == channel
        )
    for start, end in ranges:
        if end < start:
            start, end = end, start
        if frame < start:
            return False
        if start <= frame <= end:
            return True
    return False


def refresh(scene: bpy.types.Scene) -> dict[str, bpy.types.Object]:
    """Synchronize current-frame visibility without rebuilding valid helpers."""

    global _UPDATING
    if _UPDATING:
        return {}
    try:
        pointer = scene.as_pointer()
    except ReferenceError:
        return {}
    if pointer not in _ACTIVE_SCENES:
        return {}
    if not _contacts_mode_active(scene):
        cleanup(scene)
        return {}

    _UPDATING = True
    try:
        source = require_source(scene)
        by_side = {
            str(obj.get("mcd_side", "")): obj
            for obj in _owned_objects(scene)
            if str(obj.get("mcd_side", "")) in CONFIG
        }
        if set(by_side) != set(CONFIG) or any(
            not _matches(by_side[side], source, side) for side in CONFIG
        ):
            by_side = ensure(scene)
        for side, config in CONFIG.items():
            obj = by_side[side]
            visible = _frame_in_channel(scene, config["channel"])
            if bool(obj.hide_viewport) == visible:
                obj.hide_viewport = not visible
            if obj.hide_render:              # P4: unconditional write retagged it every frame
                obj.hide_render = False
        return by_side
    finally:
        _UPDATING = False


def activate(scene: bpy.types.Scene) -> dict[str, bpy.types.Object]:
    """Start the contacts-annotation indicator session for one Scene."""

    require_source(scene)
    cleanup(scene)
    _ACTIVE_SCENES.add(scene.as_pointer())
    ensure(scene)
    return refresh(scene)


@persistent
def _frame_change_post(scene, _depsgraph=None):
    try:
        refresh(scene)
    except Exception as exc:
        cleanup(scene)
        print(f"[MoCap Doctor] planted indicator frame refresh failed: {exc}")


@persistent
def _depsgraph_update_post(scene, _depsgraph=None):
    try:
        refresh(scene)
    except Exception as exc:
        cleanup(scene)
        print(f"[MoCap Doctor] planted indicator depsgraph refresh failed: {exc}")


@persistent
def _load_pre(_dummy):
    cleanup()


@persistent
def _load_post(_dummy):
    # Indicators are session UI, not project data.  Remove any helpers that
    # were present in a file saved while annotation mode was open.
    cleanup()


def register() -> None:
    cleanup()
    handlers = (
        (bpy.app.handlers.frame_change_post, _frame_change_post),
        (bpy.app.handlers.depsgraph_update_post, _depsgraph_update_post),
        (bpy.app.handlers.load_pre, _load_pre),
        (bpy.app.handlers.load_post, _load_post),
    )
    for collection, handler in handlers:
        if handler not in collection:
            collection.append(handler)


def unregister() -> None:
    handlers = (
        (bpy.app.handlers.frame_change_post, _frame_change_post),
        (bpy.app.handlers.depsgraph_update_post, _depsgraph_update_post),
        (bpy.app.handlers.load_pre, _load_pre),
        (bpy.app.handlers.load_post, _load_post),
    )
    for collection, handler in handlers:
        if handler in collection:
            collection.remove(handler)
    cleanup()
