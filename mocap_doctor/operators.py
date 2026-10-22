from contextlib import contextmanager
import json
import math
import os
from pathlib import Path

import mathutils

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper

from . import project
from . import annotation
from . import planted_indicators
from .core import animation as core_animation
from .core import contacts as core_contacts
from .core import export as core_export
from .core import foot_contacts as core_foot_contacts
from .core import foot_spacing as core_foot_spacing
from .core import official_import as core_official_import
from .core import pkl_hand as core_pkl_hand
from .core import receiver as core_receiver
from .core import source as core_source
from .core import target as core_target
from .core.ranges import diff_ranges, frames_to_ranges
from .presets import (
    EXPECTED_FPS,
    MMD_ARM_FK_BONES,
    MMD_CENTER_BONE,
    MMD_ARM_TWIST_BONES,
    MMD_ELBOW_BONES,
    MMD_LEG_FK_BONES,
    MMD_ROOT_BONE,
    MMR_COPY_CONSTRAINT_NAME,
    MMR_LEG_CONSTRAINTS,
    OBJECT_NAMES,
    SOURCE_PROFILE_AUTO,
    TARGET_FOOT_IK,
    fixed_object_name_matches,
    profile_is_available,
    resolve_mmd_foot_ik,
    resolve_mmd_hand_bones,
    resolve_mmd_toe_ik,
    resolve_source_profile,
)
from .workflow import STEPS, STEP_INDEX, clamp_step, is_project_free_step, step_at


_CONVERTED_ANNOTATION_AREAS = {}
_PENDING_ANNOTATION_GROUP = ""
_PENDING_ANNOTATION_ATTEMPTS = 0


def _snapshot_nla_filters(area):
    dopesheet = getattr(area.spaces.active, "dopesheet", None)
    if dopesheet is None:
        return {}
    values = {}
    for name in (
        "show_only_selected",
        "show_hidden",
        "filter_text",
        "use_filter_invert",
        "filter_collection",
    ):
        if hasattr(dopesheet, name):
            values[name] = getattr(dopesheet, name)
    return values


def _restore_nla_filters(area, values):
    dopesheet = getattr(area.spaces.active, "dopesheet", None)
    if dopesheet is None:
        return
    for name, value in values.items():
        try:
            setattr(dopesheet, name, value)
        except (AttributeError, ReferenceError, TypeError):
            pass


def _restore_annotation_area(screen):
    if screen is None:
        return False
    snapshot = _CONVERTED_ANNOTATION_AREAS.pop(screen.as_pointer(), None)
    if not snapshot:
        return False
    area = next(
        (item for item in screen.areas if item.as_pointer() == snapshot["pointer"]),
        None,
    )
    if area is None:
        return False
    original_type = snapshot.get("type", "NLA_EDITOR")
    if original_type != "NLA_EDITOR":
        area.type = original_type
        original_ui_type = snapshot.get("ui_type", "")
        if original_ui_type:
            try:
                area.ui_type = original_ui_type
            except (AttributeError, TypeError):
                pass
    else:
        _restore_nla_filters(area, snapshot.get("filters", {}))
    area.tag_redraw()
    return True


def _restore_all_annotation_areas():
    wm = getattr(bpy.context, "window_manager", None)
    screens = []
    if wm is not None:
        for window in wm.windows:
            if window.screen not in screens:
                screens.append(window.screen)
    for screen in screens:
        _restore_annotation_area(screen)
    # A closed workspace/window can invalidate its Area pointer.  It cannot be
    # restored, but it also cannot leave a visible NLA editor behind.
    _CONVERTED_ANNOTATION_AREAS.clear()


def _annotation_step_for_group(channel_group):
    group = str(channel_group).upper()
    if group == "FOOT":
        return "contacts"
    if group == "AIR":
        return "ground_feet"
    return "hand_ranges"


def _activate_annotation_editor_impl(scene, screen, channel_group):
    """Open or reuse one editor without overwriting its original snapshot."""

    requested_step_id = _annotation_step_for_group(channel_group)
    if getattr(scene, "mcd_annotation_mode", False):
        annotation.commit_track_reassignments(scene, rebuild=False)
    if requested_step_id == "contacts":
        planted_indicators.require_source(scene)
    if screen is None:
        raise RuntimeError("当前窗口没有可用工作区")
    key = screen.as_pointer()
    snapshot = _CONVERTED_ANNOTATION_AREAS.get(key)
    area = None
    if snapshot:
        area = next(
            (item for item in screen.areas if item.as_pointer() == snapshot["pointer"]),
            None,
        )
    if area is None:
        timelines = [
            item
            for item in screen.areas
            if item.type == "DOPESHEET_EDITOR" and getattr(item, "ui_type", "") == "TIMELINE"
        ]
        if timelines:
            area = min(timelines, key=lambda item: item.height)
        else:
            area = next((item for item in screen.areas if item.type == "NLA_EDITOR"), None)
        if area is None:
            raise RuntimeError("当前工作区没有 Timeline 或 NLA Editor")
        snapshot = {
            "pointer": area.as_pointer(),
            "type": area.type,
            "ui_type": getattr(area, "ui_type", ""),
            "filters": _snapshot_nla_filters(area) if area.type == "NLA_EDITOR" else {},
        }
        _CONVERTED_ANNOTATION_AREAS[key] = snapshot

    converted = area.type != "NLA_EDITOR"
    if converted:
        area.type = "NLA_EDITOR"

    step_id = _annotation_step_for_group(channel_group)
    if step_id == "contacts":
        active_channel = annotation.CHANNEL_FOOT_L_EFFECTIVE
    else:
        active_channel = annotation.CHANNEL_HAND_L_MANUAL
    settings = scene.mocap_doctor
    settings.annotation_step_id = step_id
    scene.mcd_annotation_active_channel = active_channel
    annotation.ensure_working_ranges_initialized(scene, step_id)
    if scene.mcd_annotation_has_pending_in and (
        scene.mcd_annotation_pending_channel not in annotation.visible_channels(scene)
    ):
        scene.mcd_annotation_has_pending_in = False
        annotation.clear_pending_marker(scene)
    if not scene.mcd_annotation_mode:
        scene.mcd_annotation_mode = True
    else:
        annotation.rebuild_projection(scene, active_channel=active_channel)
    annotation.configure_nla_area(area, scene)
    annotation.select_only_channel(scene, active_channel)
    if step_id == "contacts":
        planted_indicators.activate(scene)
    else:
        planted_indicators.cleanup(scene)
    for item in screen.areas:
        item.tag_redraw()
    return area, converted, step_id


def _activate_annotation_editor(scene, screen, channel_group):
    """Transactional wrapper for editor conversion and group switching."""

    was_open = bool(getattr(scene, "mcd_annotation_mode", False))
    old_step_id = scene.mocap_doctor.annotation_step_id
    old_active_channel = getattr(scene, "mcd_annotation_active_channel", "")
    screen_key = screen.as_pointer() if screen is not None else None
    had_snapshot = screen_key in _CONVERTED_ANNOTATION_AREAS if screen_key is not None else False
    try:
        return _activate_annotation_editor_impl(scene, screen, channel_group)
    except Exception:
        planted_indicators.cleanup(scene)
        scene.mocap_doctor.annotation_step_id = old_step_id
        if old_active_channel:
            scene.mcd_annotation_active_channel = old_active_channel
        if was_open:
            annotation.rebuild_projection(scene, active_channel=old_active_channel or None)
        if not had_snapshot and screen is not None:
            _restore_annotation_area(screen)
        if not was_open and getattr(scene, "mcd_annotation_mode", False):
            annotation.exit_annotation_mode(scene)
            scene.mocap_doctor.annotation_step_id = ""
        raise


def _cancel_pending_annotation_open():
    global _PENDING_ANNOTATION_GROUP, _PENDING_ANNOTATION_ATTEMPTS
    _PENDING_ANNOTATION_GROUP = ""
    _PENDING_ANNOTATION_ATTEMPTS = 0
    if bpy.app.timers.is_registered(_finish_pending_annotation_open):
        try:
            bpy.app.timers.unregister(_finish_pending_annotation_open)
        except (RuntimeError, ValueError):
            pass


def _finish_pending_annotation_open():
    global _PENDING_ANNOTATION_GROUP, _PENDING_ANNOTATION_ATTEMPTS
    if not _PENDING_ANNOTATION_GROUP:
        return None
    wm = getattr(bpy.context, "window_manager", None)
    window = getattr(bpy.context, "window", None)
    if window is None and wm is not None and wm.windows:
        window = wm.windows[0]
    try:
        if window is None or window.screen is None or window.scene is None:
            raise RuntimeError("恢复后窗口尚未就绪")
        group = _PENDING_ANNOTATION_GROUP
        _activate_annotation_editor(window.scene, window.screen, group)
        step_index = STEP_INDEX.get(_annotation_step_for_group(group))
        if step_index is not None:
            window.scene.mocap_doctor.current_step = step_index
        window.scene.mocap_doctor.status_message = "已保留现有标注；请补充或调整手部坏区间"
        _cancel_pending_annotation_open()
        return None
    except Exception as exc:
        _PENDING_ANNOTATION_ATTEMPTS += 1
        if _PENDING_ANNOTATION_ATTEMPTS < 12:
            return 0.25
        if window is not None and window.scene is not None and hasattr(window.scene, "mocap_doctor"):
            window.scene.mocap_doctor.status_message = f"检查点已恢复，但无法自动打开标注编辑器：{exc}"
        _cancel_pending_annotation_open()
        return None


@persistent
def _resume_annotation_after_load(_dummy):
    if _PENDING_ANNOTATION_GROUP and not bpy.app.timers.is_registered(_finish_pending_annotation_open):
        bpy.app.timers.register(_finish_pending_annotation_open, first_interval=0.25)


def cleanup_annotation_sessions():
    _restore_all_annotation_areas()
    # During Extension install/enable Blender temporarily exposes
    # ``bpy.data`` as ``_RestrictData``.  Cleanup is also called from
    # unregister(), so do not assume the main database is available.
    for scene in getattr(bpy.data, "scenes", ()):
        if getattr(scene, "mcd_annotation_mode", False):
            annotation.exit_annotation_mode(scene)
        if hasattr(scene, "mocap_doctor"):
            scene.mocap_doctor.annotation_step_id = ""
        planted_indicators.cleanup(scene)


@persistent
def _cleanup_annotation_before_load(_dummy):
    cleanup_annotation_sessions()


# 步骤标签：留在向导里的读 STEPS；已从 UI 移除、但仍有运行器可以被别的按钮
# 调用的步骤（现在只有 contacts）读这张表。
RETIRED_STEP_LABELS = {"contacts": "Planted 检测与修订"}


def _step_label(step_id):
    index = STEP_INDEX.get(step_id)
    if index is not None:
        return STEPS[index].label
    return RETIRED_STEP_LABELS.get(step_id, str(step_id))


def _require_object(settings, attribute, object_type=None, exact_name=None):
    obj = getattr(settings, attribute, None)
    if obj is None and exact_name:
        # The target objects are fixed presets, and they usually arrive AFTER
        # the project was created: the retarget step is what imports Teto and
        # the MMR rig.  A stale pointer then reported "缺少对象" for an object
        # sitting right there in the scene, so adopt it by name when the name
        # is unambiguous.  「发现对象」 still does the explicit pass.
        matches = [
            item for item in bpy.data.objects if fixed_object_name_matches(item.name, exact_name)
        ]
        if len(matches) == 1:
            obj = matches[0]
            setattr(settings, attribute, obj)
    if obj is None:
        if exact_name:
            raise RuntimeError(
                f"缺少对象：{attribute}（场景里没有名为 {exact_name} 的对象；"
                "导入 Teto/MMR 后请点「发现对象」）"
            )
        raise RuntimeError(f"缺少对象：{attribute}")
    if object_type and obj.type != object_type:
        raise RuntimeError(f"{obj.name} 不是 {object_type} 对象")
    if exact_name and not fixed_object_name_matches(obj.name, exact_name):
        raise RuntimeError(f"只支持固定对象 {exact_name}（允许 Blender 重名后缀 .001/.002），当前为 {obj.name}")
    return obj


def _require_action(owner, label):
    action = getattr(getattr(owner, "animation_data", None), "action", None)
    if action is None:
        raise RuntimeError(f"{label} 没有活动 Action")
    return action


def _require_no_nla(owner, label):
    # Empty tracks animate nothing (the user's RIG carries 103 empty NlaTrack.*,
    # same as the MMD Bake check below): only tracks with strips can stack.
    animation_data = getattr(owner, "animation_data", None)
    if animation_data and any(len(track.strips) > 0 for track in animation_data.nla_tracks):
        raise RuntimeError(f"{label} 存在 NLA Track；请先合并或移除，避免与活动 Action 叠加")


def _leg_ik_toggle_targets(armature, foot_ik):
    """The native MMD leg IK bones an FK driven leg needs switched off."""

    names = [foot_ik.get("L"), foot_ik.get("R")]
    for side in ("L", "R"):
        names.append(resolve_mmd_toe_ik(armature, side))
    return [name for name in names if name]


def _resolve_correction_empty(settings):
    """The outer correction Empty, or ``None`` when 全局扶正 was never run.

    The Empty is created by the "Teto 全局扶正" step and carries its hand tuned
    tilt.  A gravity aligned source (the GVHMR front end) needs no such tilt, so
    skipping that step is a legitimate route: nothing then owns a world level
    transform, and export_prep has nothing to bake into 全ての親.
    """

    correction = getattr(settings, "correction_empty", None)
    if correction is not None:
        return correction
    return bpy.data.objects.get(OBJECT_NAMES["correction_empty"])


def _ensure_correction_empty(settings, model_root, rig):
    """Find the correction Empty, creating and adopting it when absent."""

    correction = _resolve_correction_empty(settings)
    if correction is not None:
        return correction, False
    correction = core_target.ensure_global_correction_empty(
        OBJECT_NAMES["correction_empty"], collection=bpy.context.scene.collection
    )
    settings.correction_empty = correction
    core_target.apply_global_correction(
        model_root, rig, correction, rotation_degrees=(0.0, 0.0, 0.0)
    )
    return correction, True


def _validate_correction_transform(correction):
    if correction.parent is not None:
        raise RuntimeError("全局校正 Empty 必须是顶层对象")
    if any(abs(float(value) - 1.0) > 1.0e-6 for value in correction.scale):
        raise RuntimeError("全局校正 Empty 必须保持单位缩放")
    action = getattr(getattr(correction, "animation_data", None), "action", None)
    if action and any(curve.data_path == "scale" for curve in action.fcurves):
        raise RuntimeError("全局校正 Empty 不能带缩放动画")


def _ensure_no_pending_preview(settings, step_id=None):
    if not settings.preview_step_id:
        return
    if step_id and settings.preview_step_id == step_id and settings.preview_action:
        if not project.rollback_action_preview(bpy.context.scene):
            raise RuntimeError("无法重建已保存的 Action 预览；请先丢弃并恢复检查点")
        return
    raise RuntimeError("请先接受或丢弃当前预览，再运行其他步骤")


def _require_restore_before_rerun(scene, step_id):
    record = project.find_step_record(scene.mocap_doctor, step_id, create=False)
    if record is not None and record.status in {"ACCEPTED", "STALE"}:
        label = _step_label(step_id)
        raise RuntimeError(f"此步骤已经提交；请使用“恢复到‘{label}’执行前”后再运行")


def _begin_structure_preview(scene, step_id):
    settings = scene.mocap_doctor
    _ensure_no_pending_preview(settings, step_id)
    recovery = project.create_checkpoint(scene, f"{step_id}_before")
    settings.preview_step_id = step_id
    settings.preview_restore_checkpoint = str(recovery)
    settings.preview_owner_name = ""
    settings.preview_base_action = ""
    settings.preview_action = ""
    record = project.record_step(scene, step_id, "PREVIEW", "等待人工检查")
    record.before_checkpoint = str(recovery)
    project.save_manifest(scene)
    return recovery


def _record_report(scene, step_id, report, status, message, params=None):
    path = project.write_report(scene, step_id, report)
    project.record_step(
        scene,
        step_id,
        status,
        message,
        artifact_path=str(path),
        parameter_hash=project.parameter_hash(params or report.get("params", {})),
    )
    return path


def _planted_ranges(scene):
    return {
        "L": annotation.get_channel_ranges(scene, annotation.CHANNEL_FOOT_L_EFFECTIVE),
        "R": annotation.get_channel_ranges(scene, annotation.CHANNEL_FOOT_R_EFFECTIVE),
    }


def _validate_mmd_identity(settings):
    root = _require_object(settings, "model_root", "EMPTY", OBJECT_NAMES["model_root"])
    armature = _require_object(settings, "mmd_armature", "ARMATURE", OBJECT_NAMES["mmd_armature"])
    rig = _require_object(settings, "mmr_rig", "ARMATURE", OBJECT_NAMES["mmr_rig"])
    if armature.parent != root:
        raise RuntimeError("原生 MMD Armature 不在固定 Teto MMD Root 下")
    marker = getattr(root, "mmd_type", "")
    if marker not in {None, "", "ROOT"}:
        raise RuntimeError("Teto 根对象不是 mmd_tools ROOT")
    # mmd_root also survives as an ID property in files saved with mmd_tools
    # when the addon itself is not loaded in this Blender session.
    if marker != "ROOT" and not (hasattr(root, "mmd_root") or root.get("mmd_root") is not None):
        raise RuntimeError("无法确认 Teto 的 mmd_tools 根结构")
    missing = [
        name
        for name in (MMD_ROOT_BONE, MMD_CENTER_BONE, *MMD_LEG_FK_BONES)
        if name not in armature.pose.bones
    ]
    if missing:
        raise RuntimeError("MMD 骨骼结构不匹配：" + ", ".join(missing))
    foot_ik = {side: resolve_mmd_foot_ik(armature, side) for side in ("L", "R")}
    if not all(foot_ik.values()):
        raise RuntimeError("无法通过 mmd_tools 元数据确认左右足 IK")
    mesh = _require_object(settings, "target_mesh", "MESH", OBJECT_NAMES["target_mesh"])
    if mesh.parent != armature:
        raise RuntimeError("Teto Mesh 不在固定原生 MMD Armature 下")
    armature_modifiers = [modifier for modifier in mesh.modifiers if modifier.type == "ARMATURE"]
    if not any(getattr(modifier, "object", None) == armature for modifier in armature_modifiers):
        raise RuntimeError("Teto Mesh 的 Armature Modifier 未指向固定 MMD 骨架")
    return root, armature, rig, foot_ik


def _validate_mmd_constraint_mapping(settings):
    _root, armature, rig, foot_ik = _validate_mmd_identity(settings)
    constrained = []
    errors = []
    for bone_name, (constraint_type, subtarget) in MMR_LEG_CONSTRAINTS.items():
        bone = armature.pose.bones.get(bone_name)
        constraint = bone.constraints.get(MMR_COPY_CONSTRAINT_NAME) if bone else None
        valid = (
            constraint is not None
            and constraint.type == constraint_type
            and getattr(constraint, "target", None) == rig
            and getattr(constraint, "subtarget", "") == subtarget
            and not constraint.mute
            and abs(float(constraint.influence) - 1.0) < 1.0e-6
            and getattr(constraint, "owner_space", "WORLD") == "WORLD"
            and getattr(constraint, "target_space", "WORLD") == "WORLD"
            and getattr(constraint, "mix_mode", "REPLACE") == "REPLACE"
            and getattr(constraint, "is_valid", True)
            and subtarget in rig.pose.bones
        )
        if not valid:
            errors.append(bone_name)
    if errors:
        raise RuntimeError("MMR 腿部约束与 arue Teto 基准不一致：" + ", ".join(errors))
    invalid_targets = []
    for bone in armature.pose.bones:
        for constraint in bone.constraints:
            if (
                getattr(constraint, "target", None) != rig
                or constraint.mute
                or float(constraint.influence) <= 0.0
            ):
                continue
            subtarget = getattr(constraint, "subtarget", "")
            if not subtarget or subtarget not in rig.pose.bones or not getattr(constraint, "is_valid", True):
                invalid_targets.append(f"{bone.name} -> {subtarget or '<空>'}")
            constrained.append(bone)
    if invalid_targets:
        raise RuntimeError("MMR 约束目标损坏：" + ", ".join(invalid_targets))
    constrained = list({bone.name: bone for bone in constrained}.values())
    if not constrained:
        raise RuntimeError("没有检测到 MMR 到 MMD 的复制约束")
    return armature, rig, foot_ik, constrained


def _bone_has_range_keys(action, bone_name, frame_start, frame_end):
    prefix = f'pose.bones["{bone_name}"]'
    curves = [curve for curve in action.fcurves if curve.data_path.startswith(prefix)]
    has_start = any(
        abs(float(point.co.x) - int(frame_start)) < 0.001
        for curve in curves
        for point in curve.keyframe_points
    )
    has_end = any(
        abs(float(point.co.x) - int(frame_end)) < 0.001
        for curve in curves
        for point in curve.keyframe_points
    )
    return bool(curves) and has_start and has_end


def _action_has_range_keys(action, frame_start, frame_end):
    points = [point for curve in action.fcurves for point in curve.keyframe_points]
    if not points:
        return False
    minimum = min(float(point.co.x) for point in points)
    maximum = max(float(point.co.x) for point in points)
    return minimum <= int(frame_start) and maximum >= int(frame_end)


def _action_bone_names(action, armature):
    """Return pose bones represented by an Action, independent of constraints."""

    if action is None or armature is None:
        return set()
    return {
        bone.name
        for bone in armature.pose.bones
        if action_has_any_bone_curve(action, [bone.name])
    }


def _validate_mmd_action(
    action,
    foot_ik,
    frame_start,
    frame_end,
    expected_bones=(),
    require_clean_fk=False,
):
    required = [MMD_ROOT_BONE, foot_ik["L"], foot_ik["R"]]
    required.extend(name for name in expected_bones if name not in required)
    missing = [
        name
        for name in required
        if not _bone_has_range_keys(action, name, frame_start, frame_end)
    ]
    if missing:
        preview = ", ".join(missing[:12])
        if len(missing) > 12:
            preview += f" 等 {len(missing)} 根"
        raise RuntimeError("Bake 输出缺少覆盖起止帧的骨骼曲线：" + preview)
    remaining_fk = [name for name in MMD_LEG_FK_BONES if action_has_any_bone_curve(action, [name])]
    if require_clean_fk and remaining_fk:
        raise RuntimeError("仍有腿 FK 曲线：" + ", ".join(remaining_fk))
    return remaining_fk


def _leg_fk_cleanup_plan(armature, foot_ik):
    """Should the six leg FK curves be deleted, and why?

    Only when MMD's own leg IK actually drives the legs - the classic MMD
    workflow, where the FK curves are redundant and would double-solve against
    the IK.  PoseCapture's retarget instead binds the legs with
    COPY_TRANSFORMS onto the FK bones and leaves the native IK at influence 0
    (the IK/FK slider sits near FK), so there the FK curves ARE the animation:
    deleting them freezes the legs at rest.
    """

    for side in ("L", "R"):
        knee = armature.pose.bones.get(f"ひざ.{side}")
        if knee is None:
            return True, "缺少ひざ骨骼，按传统 IK 路径处理"
        active = any(
            constraint.type == 'IK'
            and getattr(constraint, "subtarget", "") == foot_ik[side]
            and not constraint.mute
            and float(constraint.influence) > 0.0
            for constraint in knee.constraints
        )
        if not active:
            return False, f"ひざ.{side} 的 MMD 原生 IK 未启用（腿由 FK 曲线驱动）"
    return True, "MMD 原生腿 IK 驱动中"



def _active_mmr_constraints(armature, rig):
    return [
        constraint
        for bone in armature.pose.bones
        for constraint in bone.constraints
        if getattr(constraint, "target", None) == rig
        and not constraint.mute
        and float(constraint.influence) > 0.0
    ]


def _mmr_driven_bone_names(armature, rig):
    return sorted(
        {
            bone.name
            for bone in armature.pose.bones
            for constraint in bone.constraints
            if getattr(constraint, "target", None) == rig
        }
    )


def _mute_mmr_constraints(armature, rig):
    constraints = _active_mmr_constraints(armature, rig)
    for constraint in constraints:
        constraint.mute = True
    return len(constraints)


def _write_effective_contact_report(scene):
    settings = scene.mocap_doctor
    record = project.find_step_record(settings, "contacts", create=False)
    report_path = (project.resolve_project_file(settings, record.artifact_path, "reports")
                   if record is not None else "")
    if report_path and Path(report_path).is_file():
        with Path(report_path).open('r', encoding='utf-8') as handle:
            raw_report = json.load(handle)
    else:
        raw_report = {'planted': {
            'L': annotation.get_channel_ranges(scene, annotation.CHANNEL_FOOT_L_AUTO),
            'R': annotation.get_channel_ranges(scene, annotation.CHANNEL_FOOT_R_AUTO),
        }}
    additions = {}
    deletions = {}
    effective = _planted_ranges(scene)
    if 'feet' not in raw_report:
        for side in ('L', 'R'):
            changes = diff_ranges(raw_report.get('planted', {}).get(side, ()), effective[side])
            additions[side], deletions[side] = changes['added'], changes['removed']
        raw_report['planted'] = effective
        raw_report['manual_overrides'] = {'additions': additions, 'deletions': deletions}
        raw_report['segment_counts'] = {side: len(effective[side]) for side in ('L', 'R')}
        raw_report['planted_frames'] = {
            side: sum(int(end) - int(start) + 1 for start, end in effective[side])
            for side in ('L', 'R')
        }
        return _record_report(scene, 'contacts', raw_report, 'PENDING', '贴地区间已更新')
    for side in ("L", "R"):
        foot = raw_report["feet"][side]
        raw = foot.get("raw", foot).get("planted_segments", ())
        changes = diff_ranges(raw, effective[side])
        additions[side] = changes["added"]
        deletions[side] = changes["removed"]
    final_report = core_contacts.apply_contact_overrides(
        raw_report,
        additions=additions,
        deletions=deletions,
        merge_gap=0,
        min_segment_len=1,
    )
    return _record_report(
        scene,
        "contacts",
        final_report,
        "PREVIEW",
        "最终 planted 区间等待提交",
        final_report.get("params", {}),
    )


def _settings(context):
    return context.scene.mocap_doctor


def _require_project(context):
    settings = _settings(context)
    if not settings.initialized:
        raise RuntimeError("请先创建 MoCap Doctor 工作项目")
    if context.scene.render.fps != EXPECTED_FPS or context.scene.render.fps_base != 1.0:
        raise RuntimeError("项目必须保持 30fps / fps_base 1.0")
    if settings.mocap_frame_end < settings.mocap_frame_start:
        raise RuntimeError("有效动捕结束帧不能早于开始帧")
    project.apply_project_range(context.scene, settings)
    return settings


def _default_template_path():
    current = Path(bpy.data.filepath) if bpy.data.filepath else None
    candidates = []
    if current:
        candidates.extend(
            [
                current.parent / "arue_teto.blend",
                current.parent / "blends" / "arue_teto.blend",
                current.parent.parent / "blends" / "arue_teto.blend",
            ]
        )
    for path in candidates:
        if path.is_file():
            return str(path)
    return ""


@contextmanager
def _active_armature(context, armature, pose=False):
    view_layer = context.view_layer
    selected = [obj for obj in view_layer.objects if obj.select_get()]
    active = view_layer.objects.active
    old_mode = active.mode if active else "OBJECT"
    try:
        if active and active.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        armature.hide_set(False)
        armature.select_set(True)
        view_layer.objects.active = armature
        if pose:
            bpy.ops.object.mode_set(mode="POSE")
        yield
    finally:
        try:
            if view_layer.objects.active and view_layer.objects.active.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected:
            if obj.name in view_layer.objects:
                obj.select_set(True)
        if active and active.name in view_layer.objects:
            view_layer.objects.active = active
            if old_mode != "OBJECT":
                try:
                    bpy.ops.object.mode_set(mode=old_mode)
                except RuntimeError:
                    pass


def remove_bone_fcurves(action, bone_names):
    prefixes = tuple(f'pose.bones["{name}"]' for name in bone_names)
    removed = []
    for fcurve in list(action.fcurves):
        if fcurve.data_path.startswith(prefixes):
            removed.append((fcurve.data_path, fcurve.array_index))
            action.fcurves.remove(fcurve)
    return removed


def action_has_any_bone_curve(action, candidates):
    prefixes = tuple(f'pose.bones["{name}"]' for name in candidates)
    return any(fcurve.data_path.startswith(prefixes) for fcurve in action.fcurves)


def _pose_bone_vmd_name(pose_bone):
    metadata = getattr(pose_bone, "mmd_bone", None)
    japanese = str(getattr(metadata, "name_j", "") or "") if metadata else ""
    return japanese or pose_bone.name


def _bone_has_exportable_vmd_curve(action, pose_bone):
    prefix = f'pose.bones["{pose_bone.name}"]'
    rotation_path = {
        "QUATERNION": "rotation_quaternion",
        "AXIS_ANGLE": "rotation_axis_angle",
    }.get(pose_bone.rotation_mode, "rotation_euler")
    export_paths = {f"{prefix}.location", f"{prefix}.{rotation_path}"}
    return any(curve.data_path in export_paths for curve in action.fcurves)


def _keyed_vmd_name_collisions(armature, action):
    """Return VMD export names shared by multiple keyed pose bones."""

    by_export_name = {}
    for bone in armature.pose.bones:
        if not _bone_has_exportable_vmd_curve(action, bone):
            continue
        by_export_name.setdefault(_pose_bone_vmd_name(bone), []).append(bone.name)
    return {
        export_name: names
        for export_name, names in by_export_name.items()
        if len(names) > 1
    }


def _curve_covers_frame_range(curve, frame_start, frame_end):
    """Check that an F-curve can be evaluated throughout the motion range.

    MikuMikuRig's Bake may compress a dense result to a few Bezier keys.  A
    key on every frame is not needed for a valid VMD export: every required
    channel only needs keys on both sides of the requested frame range.
    """

    frames = [float(point.co.x) for point in curve.keyframe_points]
    return bool(frames) and min(frames) <= float(frame_start) + 0.001 and max(frames) >= float(frame_end) - 0.001


def _bone_has_complete_arm_animation(action, pose_bone, frame_start, frame_end):
    bone_name = pose_bone.name
    prefix = f'pose.bones["{bone_name}"]'

    def complete(data_path, indices):
        curves = {
            int(curve.array_index): curve
            for curve in action.fcurves
            if curve.data_path == f"{prefix}.{data_path}"
        }
        return all(
            index in curves
            and _curve_covers_frame_range(curves[index], frame_start, frame_end)
            for index in indices
        )

    rotation_path, rotation_indices = {
        "QUATERNION": ("rotation_quaternion", range(4)),
        "AXIS_ANGLE": ("rotation_axis_angle", range(4)),
    }.get(pose_bone.rotation_mode, ("rotation_euler", range(3)))
    rotation_complete = complete(rotation_path, rotation_indices)
    return complete("location", range(3)) and rotation_complete


def _sample_pose_bone_matrices(scene, armature, bone_names, frame_start, frame_end):
    samples = {}
    view_layer = bpy.context.view_layer
    for frame in range(int(frame_start), int(frame_end) + 1):
        scene.frame_set(frame)
        view_layer.update()
        samples[frame] = {
            name: armature.pose.bones[name].matrix.copy()
            for name in bone_names
        }
    return samples


def _maximum_matrix_sample_error(before, after):
    maximum = 0.0
    for frame, bones in before.items():
        for bone_name, matrix in bones.items():
            compared = after[frame][bone_name]
            maximum = max(
                maximum,
                max(
                    abs(float(matrix[row][column] - compared[row][column]))
                    for row in range(4)
                    for column in range(4)
                ),
            )
    return maximum


def _require_complete_arm_fk_animation(action, armature, frame_start, frame_end):
    """Require complete transform channels across the six physical arm bones."""

    missing = []
    for bone_name in MMD_ARM_FK_BONES:
        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None:
            missing.append(f"{bone_name}（骨骼不存在）")
        elif not _bone_has_complete_arm_animation(action, pose_bone, frame_start, frame_end):
            missing.append(bone_name)
    if missing:
        raise RuntimeError("上肢 Bake 不完整，不能导出 VMD：" + ", ".join(missing))


def _visual_bake_teto_arm_fk(scene, armature, action, frame_start, frame_end):
    """Bake the evaluated upper-arm/elbow/wrist result into real FK bones."""

    missing = [name for name in MMD_ARM_FK_BONES if armature.pose.bones.get(name) is None]
    if missing:
        raise RuntimeError("固定 Teto 缺少上肢骨骼：" + ", ".join(missing))
    if armature.animation_data is None or armature.animation_data.action is not action:
        raise RuntimeError("MMD Action 不是原生骨架的活动 Action，无法烘焙上肢")
    with _active_armature(bpy.context, armature, pose=True):
        bpy.ops.pose.select_all(action="DESELECT")
        for bone_name in MMD_ARM_FK_BONES:
            armature.pose.bones[bone_name].bone.select = True
        result = bpy.ops.nla.bake(
            frame_start=int(frame_start), frame_end=int(frame_end), step=1,
            only_selected=True, visual_keying=True, clear_constraints=False,
            clear_parents=False, use_current_action=True, clean_curves=False,
            bake_types={"POSE"},
        )
    if "FINISHED" not in result:
        raise RuntimeError("上肢 FK Visual Bake 没有成功完成")
    _require_complete_arm_fk_animation(action, armature, frame_start, frame_end)
    return {"bones": list(MMD_ARM_FK_BONES), "frame_range": [int(frame_start), int(frame_end)]}


def _resolve_teto_hand_ik_helpers(armature):
    """Return Blender-only MMR hand helpers, keeping real wrists mandatory."""

    helpers = {}
    absent_sides = []
    for side in ("L", "R"):
        wrist, helper = resolve_mmd_hand_bones(armature, side)
        if wrist is None:
            raise RuntimeError(f"固定 Teto 缺少 {side} 侧真实手首骨骼")
        if helper is None:
            absent_sides.append(side)
        else:
            helpers[side] = helper.name
    return helpers, absent_sides


def _remove_mmr_hand_helper_constraints(armature, helper_names):
    """Remove constraints that would point at a helper being deleted."""

    removed = []
    helper_names = set(helper_names)
    for pose_bone in armature.pose.bones:
        for constraint in list(pose_bone.constraints):
            if getattr(constraint, "subtarget", "") not in helper_names:
                continue
            label = f"{pose_bone.name}:{constraint.name}"
            pose_bone.constraints.remove(constraint)
            removed.append(label)
    return removed


_BONE_TRANSFORM_CHANNELS = {
    "QUATERNION": ("location", "rotation_quaternion", "scale"),
    "AXIS_ANGLE": ("location", "rotation_axis_angle", "scale"),
}


def _rekey_bone_world_pose(scene, armature, action, poses, frame_start, frame_end):
    """Rewrite a bone's channels so its world pose survives a parent swap.

    A pose bone stores a basis that is interpreted against its current parent,
    so re-parenting re-interprets the whole bake.  Reassigning
    ``pose_bone.matrix`` asks Blender for the basis that reproduces a given
    armature-space pose under the *new* parent, which is exactly the value MMD
    will need once it walks its own hierarchy.
    """

    view_layer = bpy.context.view_layer
    view_layer.objects.active = armature
    written = {}
    for name, matrices in poses.items():
        pose_bone = armature.pose.bones.get(name)
        if pose_bone is None:
            raise RuntimeError(f"准备重设关键帧的骨骼不存在：{name}")
        channels = _BONE_TRANSFORM_CHANNELS.get(
            pose_bone.rotation_mode, ("location", "rotation_euler", "scale")
        )
        for frame in range(int(frame_start), int(frame_end) + 1):
            scene.frame_set(frame)
            view_layer.update()
            pose_bone.matrix = matrices[frame]
            view_layer.update()
            for channel in channels:
                pose_bone.keyframe_insert(channel, frame=frame, group=name)
        written[name] = int(frame_end) - int(frame_start) + 1
    core_animation.update_action(action)
    return written


def _restore_teto_elbow_parents_and_remove_helpers(
    scene, armature, action, helper_names, frame_start, frame_end
):
    """Restore Teto's arm hierarchy, then delete MMR's temporary hand bones.

    MMR hangs ``ひじ`` directly off ``腕``; Teto itself routes it through the
    ``腕捩`` twist bone.  ``腕捩`` carries a real pose of its own, so the swap
    alone would swing the whole forearm - record the evaluated elbows first and
    re-key them against the restored hierarchy afterwards.
    """

    restored_parents = []
    deleted_helpers = []
    poses = {}
    elbow_names = list(MMD_ELBOW_BONES.values())
    already_routed = all(
        armature.pose.bones[name].parent is not None
        and armature.pose.bones[name].parent.name == MMD_ARM_TWIST_BONES[side]
        for side, name in MMD_ELBOW_BONES.items()
        if name in armature.pose.bones
    )
    if not already_routed:
        poses = _sample_pose_bone_matrices(
            scene, armature, elbow_names, frame_start, frame_end
        )
        poses = {
            name: {frame: bones[name] for frame, bones in poses.items()}
            for name in elbow_names
        }
    with _active_armature(bpy.context, armature, pose=False):
        bpy.ops.object.mode_set(mode="EDIT")
        edit_bones = armature.data.edit_bones
        for side in ("L", "R"):
            elbow_name = MMD_ELBOW_BONES[side]
            twist_name = MMD_ARM_TWIST_BONES[side]
            upper_arm_name = f"腕.{side}"
            elbow = edit_bones.get(elbow_name)
            twist = edit_bones.get(twist_name)
            if elbow is None or twist is None:
                raise RuntimeError(f"固定 Teto 缺少 {side} 侧肘部或腕捩骨骼")
            parent = elbow.parent
            if parent == twist:
                continue
            if parent is None or parent.name != upper_arm_name:
                found = parent.name if parent else "无父级"
                raise RuntimeError(
                    f"{elbow_name} 的父级是 {found}，不是固定 Teto 的 {upper_arm_name} 或 {twist_name}；"
                    "为避免猜测性改骨，已停止导出"
                )
            elbow.parent = twist
            restored_parents.append({"bone": elbow_name, "from": upper_arm_name, "to": twist_name})
        for helper_name in helper_names:
            helper = edit_bones.get(helper_name)
            if helper is None:
                raise RuntimeError(f"准备删除的手 IK 辅助骨不存在：{helper_name}")
            children = [bone.name for bone in edit_bones if bone.parent == helper]
            if children:
                raise RuntimeError(
                    f"{helper_name} 仍有子骨骼（{', '.join(children)}），不能安全删除"
                )
            edit_bones.remove(helper)
            deleted_helpers.append(helper_name)
    rekeyed = {}
    if restored_parents and poses:
        bpy.context.view_layer.update()
        rekeyed = _rekey_bone_world_pose(
            scene,
            armature,
            action,
            poses,
            frame_start,
            frame_end,
        )
    return restored_parents, deleted_helpers, rekeyed


def _prepare_teto_mmr_hand_export_cleanup(
    scene,
    armature,
    action,
    frame_start,
    frame_end,
    *,
    matrix_tolerance=1.0e-4,
):
    """Convert MMR's temporary hand controls back into a portable Teto arm."""

    # MMD Bake already creates the physical arm keys. Re-baking here can
    # overwrite a valid manual Bake under MMR's altered elbow hierarchy.
    _require_complete_arm_fk_animation(action, armature, frame_start, frame_end)
    arm_bake = {
        "operation": "validated_existing_arm_fk_bake",
        "bones": list(MMD_ARM_FK_BONES),
        "frame_range": [int(frame_start), int(frame_end)],
    }
    arm_names = list(MMD_ARM_FK_BONES)
    original_frame = int(scene.frame_current)
    try:
        before = _sample_pose_bone_matrices(scene, armature, arm_names, frame_start, frame_end)
        helpers, absent_helpers = _resolve_teto_hand_ik_helpers(armature)
        helper_names = list(helpers.values())
        removed_constraints = _remove_mmr_hand_helper_constraints(armature, helper_names)
        removed_curves = {
            helper_name: len(remove_bone_fcurves(action, [helper_name]))
            for helper_name in helper_names
        }
        restored_parents, deleted_helpers, rekeyed = (
            _restore_teto_elbow_parents_and_remove_helpers(
                scene, armature, action, helper_names, frame_start, frame_end
            )
        )
        bpy.context.view_layer.update()
        after = _sample_pose_bone_matrices(scene, armature, arm_names, frame_start, frame_end)
        maximum_error = _maximum_matrix_sample_error(before, after)
    finally:
        scene.frame_set(original_frame)
        bpy.context.view_layer.update()

    if maximum_error > float(matrix_tolerance):
        raise RuntimeError(
            "上肢 FK 兼容 Bake 后姿势发生变化，已阻止导出："
            f"最大矩阵误差 {maximum_error:.6g}"
        )
    return {
        "operation": "prepare_teto_mmr_hand_export_cleanup",
        "arm_fk_bake": arm_bake,
        "restored_elbow_parents": restored_parents,
        "rekeyed_elbow_frames": rekeyed,
        "deleted_hand_ik_helpers": deleted_helpers,
        "hand_ik_helper_not_present": absent_helpers,
        "removed_hand_helper_constraints": removed_constraints,
        "removed_helper_fcurves_by_bone": removed_curves,
        "removed_helper_fcurve_count": sum(removed_curves.values()),
        "maximum_arm_matrix_error": maximum_error,
    }


def _require_no_keyed_vmd_name_collisions(armature, action):
    collisions = _keyed_vmd_name_collisions(armature, action)
    if collisions:
        details = "; ".join(
            f"{export_name}: {', '.join(names)}"
            for export_name, names in sorted(collisions.items())
        )
        raise RuntimeError("仍有带关键帧的重复 VMD 骨名：" + details)
    return collisions


def _source_maps(settings, armature):
    """Source bone maps for the frontend that produced this armature."""

    maps = resolve_source_profile(
        armature, getattr(settings, "source_profile", SOURCE_PROFILE_AUTO)
    )
    if maps is None:
        raise RuntimeError(
            "无法识别源骨架：既不是 FreeMoCap 骨架，也没有 GVHMR 的 f_avg/m_avg 骨骼。"
            "请在上一步确认源骨架对象，或在项目设置中选择源数据来源。"
        )
    missing = profile_is_available(maps, armature)
    if missing:
        preview = "、".join(missing[:6])
        raise RuntimeError(
            f"源骨架缺少 {maps['profile']} 结构所需的骨骼：{preview}"
            + ("等" if len(missing) > 6 else "")
        )
    return maps


# DEPRECATED（2026-10-16 从界面移除 FreeMoCap/GVHMR 相关 UI）：执行器保留，界面不可达。
def _run_source_check(context, settings):
    """Validate the imported source animation; replaces the FreeMoCap bake step."""

    scene = context.scene
    _require_restore_before_rerun(scene, "source_check")
    # The source rig is identified by its bones (see _source_maps), not by a
    # fixed object name: GVHMR imports it as "Armature", FreeMoCap used
    # "import_synchronized_videos_rig".
    armature = _require_object(settings, "source_armature", "ARMATURE")
    action = _require_action(armature, "源骨架")
    maps = _source_maps(settings, armature)

    fps = int(scene.render.fps)
    if fps != EXPECTED_FPS:
        raise RuntimeError(f"项目要求 {EXPECTED_FPS} fps，当前场景为 {fps} fps")

    frame_start, frame_end = int(settings.mocap_frame_start), int(settings.mocap_frame_end)
    if frame_end <= frame_start:
        raise RuntimeError("动捕范围无效：结束帧必须大于开始帧")
    start, end = (float(action.frame_range[0]), float(action.frame_range[1]))
    if start > frame_start or end < frame_end:
        raise RuntimeError(
            f"源动作只覆盖 {int(start)}-{int(end)} 帧，动捕范围 {frame_start}-{frame_end} "
            "超出该区间；请调整范围或重新导入"
        )

    bones = list(dict.fromkeys(maps["bones"].values()))
    missing_channels = [
        name for name in bones if not _bone_has_range_keys(action, name, frame_start, frame_end)
    ]
    if missing_channels:
        raise RuntimeError(
            "源动作缺少覆盖动捕范围的骨骼曲线：" + "、".join(missing_channels[:6])
        )

    report = {
        "operation": "validate_source_import",
        "profile": maps["profile"],
        "prefix": maps.get("prefix", ""),
        "armature": armature.name,
        "action": action.name,
        "action_range": [int(start), int(end)],
        "mocap_range": [frame_start, frame_end],
        "fps": fps,
        "checked_bones": bones,
    }
    path = _record_report(
        scene,
        "source_check",
        report,
        "ACCEPTED",
        f"源数据校验通过（{maps['profile']}，{frame_start}-{frame_end} 帧）",
        {"source_profile": maps["profile"]},
    )
    settings.current_step = STEP_INDEX.get("source_floor", settings.current_step)
    settings.status_message = "源数据校验通过"
    project.create_accepted_checkpoint(scene, "source_check", "源数据校验通过")
    return f"源数据校验通过：{path.name}"





def _run_source_floor(context, settings):
    scene = context.scene
    _require_restore_before_rerun(scene, "source_floor")
    armature = _require_object(settings, "source_armature", "ARMATURE")
    _require_action(armature, "源骨架")
    _ensure_no_pending_preview(settings, "source_floor")
    action = project.begin_action_preview(scene, armature, "source_floor")
    params = {
        "floor_z": settings.source_floor_z,
        "tolerance": settings.source_floor_tolerance,
        "target_clearance": settings.source_floor_clearance,
        "max_lift_per_frame": settings.source_floor_max_lift,
        "strength": settings.source_floor_strength,
        "smooth_radius": settings.source_floor_smooth_radius,
        "max_correction_delta_per_frame": settings.source_floor_max_delta,
        "window": settings.source_floor_window,
    }
    maps = _source_maps(settings, armature)
    floor_kwargs = {}
    if maps["contact_points"] is not None:
        floor_kwargs["contact_points"] = maps["contact_points"]
    # The role name differs per frontend, so take it from the profile:
    # hard coding "pelvis" is what made this step fail on every GVHMR file.
    hips_bone = maps["bones"].get("hips")
    if hips_bone:
        floor_kwargs["root_bone"] = hips_bone
    result = core_source.level_source_ground(
        scene,
        armature,
        action,
        frame_start=settings.mocap_frame_start,
        frame_end=settings.mocap_frame_end,
        **params,
        **floor_kwargs,
    )
    _record_report(scene, "source_floor", result, "PREVIEW", "源骨架地面修复等待检查", params)
    span = result.get("floor_span_before")
    detail = f"，整平前地面高低差 {span:.3f} m" if span else ""
    return f"源地面修复影响 {result.get('changed_frames', 0)} 帧" + detail


def _contacts_from_npz(settings):
    """默认贴地来源：FlowHMR 官方后处理第 5 步（npz 的 foot_contacts）。

    官方实现 ``postprocess_utils.find_foot_contact_interval``（阈值 0.7 / 速度
    0.45 / 补充 0.15 / 收缩 3 帧 / 踝 10cm 脚 5cm 拆分 / >7 帧），见
    ``core/foot_contacts.py`` 顶部的 file:line 说明。

    **速度项**：官方喂的是 FK 出来的绝对关键点。路线 (B) 用 ``final.fbx`` 里
    52 根骨的 bone head 当绝对关节（``tools/fbx_joints_dump.py`` 导出，路径见
    ``foot_contacts_joints_npz``／同目录 sidecar）。没有那份文件时退回 npz 的
    ``position_mode`` 近似，报告里 ``velocity_is_guess`` 为真、状态行写「在猜」。

    返回 ``(report, why)``：成功时 why 为空；读不了/没配路径时 report 为 None、
    why 写明原因，调用方据此回退到源骨架几何检测。**同 take 守卫的
    TakeMismatchError 不吞**：把 A 片的分数配 B 片的速度是配置错误，必须挡住。
    """

    path = str(getattr(settings, "foot_contacts_npz", "") or "")
    if not path:
        return None, "未指定 FlowHMR npz"
    explicit = str(getattr(settings, "foot_contacts_joints_npz", "") or "")
    joints_path = core_foot_contacts.resolve_joints_path(path, explicit)
    try:
        report = core_foot_contacts.official_planted_ranges(
            path,
            frame_start=settings.mocap_frame_start,
            frame_end=settings.mocap_frame_end,
            frame_offset=int(getattr(settings, "npz_frame_offset", 1) or 0),
            joints_path=joints_path,
        )
    except core_foot_contacts.TakeMismatchError:
        raise                      # 配对错误必须可见，不能退回几何检测
    except Exception as exc:  # noqa: BLE001 - 回退路径要拿到原因，不能静默
        return None, str(exc)
    return report, ""


def _velocity_status(report) -> str:
    """一行说明速度项来源（(B) 绝对关节 / npz 近似在猜）。"""

    if report.get("velocity_is_guess"):
        return f"速度项在猜（npz {report.get('position_mode', '?')}）"
    guard = report.get("take_guard") or {}
    return (f"速度项 fbx 绝对关节（take {guard.get('take', '?')}，"
            f"{guard.get('joints_frames', 0)} 帧 vs {guard.get('npz_frames', 0)} 帧）")


def _run_contacts(context, settings):
    scene = context.scene
    _ensure_no_pending_preview(settings)
    armature = _require_object(settings, "source_armature", "ARMATURE")

    # 默认来源：npz 的 foot_contacts（用户决定）。读不到就回退到几何检测，
    # 并在状态里说明——回退永远是可见的，不静默。
    requested = str(getattr(settings, "contact_source", "NPZ") or "NPZ").upper()
    npz_report, npz_why = (None, "")
    if requested == "NPZ":
        npz_report, npz_why = _contacts_from_npz(settings)
    if npz_report is not None:
        for side in ("L", "R"):
            annotation.set_planted_auto_ranges(
                scene,
                side,
                npz_report["planted"][side],
                initialize_effective=True,
                rebuild=side == "R",
            )
        # 两只脚都没接地的帧就是腾空区间——同一份 npz 直接给出，ground_feet 用它。
        annotation.set_air_auto_ranges(
            scene, [tuple(item) for item in npz_report["air"]], initialize_effective=True
        )
        npz_report["source"] = "npz_official"
        npz_report["fallback_reason"] = ""
        velocity = _velocity_status(npz_report)
        _record_report(
            scene,
            "contacts",
            npz_report,
            "PREVIEW",
            f"贴地区间来自 FlowHMR 官方第 5 步（npz foot_contacts；{velocity}）",
            {"foot_contacts_npz": npz_report["npz"],
             "frame_offset": npz_report["frame_offset"],
             "velocity_source": npz_report.get("velocity_source", ""),
             "velocity_is_guess": bool(npz_report.get("velocity_is_guess")),
             "joints_path": npz_report.get("joints_path", ""),
             "take_guard": npz_report.get("take_guard"),
             "official": npz_report.get("params", {})},
        )
        counts = npz_report["segment_counts"]
        frames = npz_report["planted_frames"]
        # 状态行只有窄窄一条：只回极短的一句（详细数字留在报告文件里）。
        return f"已检测到 {counts['L'] + counts['R']} 个贴地区间"

    params = {
        "contact_height": settings.contact_height,
        "planted_xy_speed": settings.contact_xy_speed,
        "moving_xy_speed": settings.contact_moving_xy_speed,
        "max_vertical_speed": settings.contact_vertical_speed,
        "min_segment_len": settings.contact_min_segment_len,
        "merge_gap": settings.contact_merge_gap,
        "max_anchor_drift": settings.contact_anchor_drift,
        "penetration_tolerance": settings.contact_penetration_tolerance,
    }
    contacts_kwargs = {}
    foot_points = _source_maps(settings, armature)["foot_points"]
    if foot_points is not None:
        contacts_kwargs["foot_points"] = foot_points

    def _detect(floor_z):
        return core_contacts.detect_contacts_v2(
            scene,
            armature,
            frame_start=settings.mocap_frame_start,
            frame_end=settings.mocap_frame_end,
            floor_z=floor_z,
            **params,
            **contacts_kwargs,
        )

    report = _detect(settings.source_floor_z)
    auto_floor = None
    if not any(report["feet"][side]["raw"]["planted_segments"] for side in ("L", "R")):
        # 源骨架整个悬空时（FlowHMR 原生骨架的脚踝最低点离 Z=0 有十几厘米），
        # 按配置的地面 Z 一个 planted 都找不到——这不是"没有接触"，是地面给错了。
        # 「源骨架地面修复」那一步已经从向导里退休，没人再替它报地面；这里按数据
        # 自己量一次：把这只 take 里最低的脚点当地面（planted 就成了"脚在最低处"，
        # 这正是 planted 要表达的意思），并在报告里写明用了多少。
        lowest = min(
            entry["height_above_floor"]
            for side in ("L", "R")
            for entry in report["feet"][side]["per_frame"].values()
        )
        candidate = round(float(settings.source_floor_z) + float(lowest), 5)
        if candidate > float(settings.source_floor_z) + 1e-6:
            retry = _detect(candidate)
            if any(retry["feet"][side]["raw"]["planted_segments"] for side in ("L", "R")):
                auto_floor = candidate
                report = retry
                report["auto_floor_z"] = candidate
                report["params"]["floor_z"] = candidate
    for side in ("L", "R"):
        raw = report["feet"][side]["raw"]["planted_segments"]
        annotation.set_planted_auto_ranges(
            scene,
            side,
            raw,
            initialize_effective=True,
            rebuild=side == "R",
        )
    # "Both feet off the floor at once" is the one thing the data CAN state
    # reliably - it is the grounding step's exemption list, and a missed jump
    # is what happens when nobody marks it.  Seed it as a hint; the human
    # extends it (GVHMR smears real jumps into near-ground, so it under-finds).
    airborne_frames = [
        frame
        for frame in range(settings.mocap_frame_start, settings.mocap_frame_end + 1)
        if all(
            report["feet"][side]["per_frame"].get(frame, {}).get("state")
            == "airborne_or_lifted"
            for side in ("L", "R")
        )
    ]
    annotation.set_air_auto_ranges(
        scene, frames_to_ranges(airborne_frames), initialize_effective=True
    )
    path = _record_report(
        scene,
        "contacts",
        report,
        "PREVIEW",
        "自动 planted 已更新；请检查最终区间",
        params,
    )
    counts_geom = {
        side: len(report["feet"][side]["raw"]["planted_segments"]) for side in ("L", "R")
    }
    if npz_why:
        settings.status_level = 'WARNING'
        message = '贴地数据不可用，已按骨架推算'
    else:
        message = f"已检测到 {counts_geom['L'] + counts_geom['R']} 个贴地区间"
    return message


def _run_global_correction(context, settings):
    scene = context.scene
    _require_restore_before_rerun(scene, "global_correction")
    model_root = _require_object(
        settings, "model_root", "EMPTY", OBJECT_NAMES["model_root"]
    )
    rig = _require_object(settings, "mmr_rig", "ARMATURE", OBJECT_NAMES["mmr_rig"])
    _begin_structure_preview(scene, "global_correction")
    correction = core_target.ensure_global_correction_empty(
        OBJECT_NAMES["correction_empty"], collection=scene.collection
    )
    settings.correction_empty = correction
    degrees = tuple(
        math.degrees(value)
        for value in (settings.global_rot_x, settings.global_rot_y, settings.global_rot_z)
    )
    result = core_target.apply_global_correction(
        model_root, rig, correction, rotation_degrees=degrees
    )
    _record_report(
        scene,
        "global_correction",
        result,
        "PREVIEW",
        "Teto 全局扶正等待检查",
        {"rotation_degrees": degrees},
    )
    return "Teto 全局扶正预览已生成"


def _run_tilt(context, settings):
    scene = context.scene
    _require_restore_before_rerun(scene, "tilt")
    rig = _require_object(settings, "mmr_rig", "ARMATURE", OBJECT_NAMES["mmr_rig"])
    _require_action(rig, "MMR Rig")
    _ensure_no_pending_preview(settings, "tilt")
    action = project.begin_action_preview(scene, rig, "tilt")
    params = {
        "reference_frame": settings.tilt_reference_frame,
        "strength": settings.tilt_strength,
        "damp_axes": (settings.tilt_damp_x, settings.tilt_damp_y, settings.tilt_damp_z),
    }
    result = core_target.damp_foot_ik_tilt(
        scene,
        rig,
        action,
        foot_bones=tuple(TARGET_FOOT_IK.values()),
        frame_start=settings.mocap_frame_start,
        frame_end=settings.mocap_frame_end,
        **params,
    )
    _record_report(scene, "tilt", result, "PREVIEW", "Foot IK 倾斜修复等待检查", params)
    return "Foot IK 倾斜修复预览已生成"


def _run_target_floor(context, settings):
    scene = context.scene
    _require_restore_before_rerun(scene, "target_floor")
    model_root = _require_object(
        settings, "model_root", "EMPTY", OBJECT_NAMES["model_root"]
    )
    rig = _require_object(settings, "mmr_rig", "ARMATURE", OBJECT_NAMES["mmr_rig"])
    _ensure_no_pending_preview(settings, "target_floor")
    correction, created_correction = _ensure_correction_empty(
        settings, model_root, rig
    )
    _validate_correction_transform(correction)
    action = project.begin_action_preview(scene, correction, "target_floor")
    params = {
        "floor_z": settings.target_floor_z,
        "target_clearance": settings.target_clearance,
        "tolerance": settings.target_floor_tolerance,
        "max_lift_per_frame": settings.target_floor_max_lift,
        "strength": settings.target_floor_strength,
        "smooth_radius": settings.target_floor_smooth_radius,
        "max_delta_per_frame": settings.target_floor_max_delta,
        "visible_only": settings.target_visible_only,
        "vertex_sample_step": settings.target_vertex_sample_step,
        "reset_existing_z_curve": settings.target_reset_existing_z,
    }
    result = core_target.repair_mesh_floor_lift_v3_safe(
        scene,
        model_root,
        correction,
        action=action,
        frame_start=settings.mocap_frame_start,
        frame_end=settings.mocap_frame_end,
        **params,
    )
    result["created_correction_empty"] = created_correction
    _record_report(scene, "target_floor", result, "PREVIEW", "Teto Mesh 穿地修复等待检查", params)
    message = f"Mesh 地面修复影响 {result.get('changed_frames', 0)} 帧"
    if created_correction:
        message += "；已补建全局校正 Empty（未扶正）"
    return message


def _run_foot_lock(context, settings):
    scene = context.scene
    _require_restore_before_rerun(scene, "foot_lock")
    planted = _planted_ranges(scene)
    if not any(planted.values()):
        raise RuntimeError("最终 planted 区间为空，请先完成接触区间修订")
    rig = _require_object(settings, "mmr_rig", "ARMATURE", OBJECT_NAMES["mmr_rig"])
    _require_action(rig, "MMR Rig")
    armature = _require_object(
        settings, "mmd_armature", "ARMATURE", OBJECT_NAMES["mmd_armature"]
    )
    # The closed loop measures the ankle through the MMR copy constraints;
    # with them muted (i.e. after a bake) it would correct against a frozen
    # bone and write garbage.  Fail before touching anything.
    if not _active_mmr_constraints(armature, rig):
        raise RuntimeError(
            "MMR 到 MMD 的约束未激活；闭环稳定靠约束把脚 IK 传给足首，"
            "请在 MMD Visual Bake 之前运行此步骤"
        )
    drift = core_target.analyze_foot_ik_drift(
        scene,
        rig,
        planted,
        foot_bones=TARGET_FOOT_IK,
        frame_start=settings.mocap_frame_start,
        frame_end=settings.mocap_frame_end,
        trim_segment_ends=settings.lock_trim,
        min_segment_len=settings.drift_min_segment_len,
    )
    _ensure_no_pending_preview(settings, "foot_lock")
    action = project.begin_action_preview(scene, rig, "foot_lock")
    params = {
        "trim_segment_ends": settings.lock_trim,
        "min_segment_len": settings.lock_min_segment_len,
        "blend_frames": settings.lock_blend_frames,
        "anchor_mode": settings.lock_anchor_mode,
        "min_local_xy_range": settings.lock_min_xy_range,
        "lock_x": settings.lock_x,
        "lock_y": settings.lock_y,
        "lock_z": settings.lock_z,
    }
    repair = core_target.lock_foot_ik_xy(
        scene,
        rig,
        action,
        planted,
        foot_bones=TARGET_FOOT_IK,
        frame_start=settings.mocap_frame_start,
        frame_end=settings.mocap_frame_end,
        **params,
    )
    # Closed-loop world-space stabilization: measure the evaluated MMD ankle
    # (what the mesh and the bake actually follow) and pull foot_ik until the
    # ankle holds the segment anchor pose.  This replaces the 1.4.0 channel
    # locks that only ever saw the controller side of the constraint chain.
    # The pelvis pass runs inside it first: per-frame the model's leg reach
    # ratio is re-solved onto the source skeleton's (restoring the captured
    # knee bend Teto's longer legs were flattening), which lowers torso_root
    # enough that planted anchors at floor height stay physically reachable.
    stabilized = core_target.stabilize_planted_feet(
        scene,
        armature,
        rig,
        action,
        planted,
        source_armature=settings.source_armature,
        pelvis_correction_max=settings.pelvis_correction_max,
        sole_offset=core_target.model_sole_offset(
            armature, settings.target_mesh
        ),
        sole_dirs=core_target.sole_contact_offsets(
            armature, settings.target_mesh
        ),
        # Grounding reference: the real boot mesh at each segment's anchor
        # frame ("how far off the floor is it, move it that much"), not the
        # bind-pose single-point sole vector, which is ~2 cm low and pushed
        # every planted segment through the floor.
        ground_mesh=settings.target_mesh,
        ground_clearance=settings.target_clearance,
        # Same floor the ground_feet step pins to; without it the anchors
        # were grounded at Z=0 whatever 地面 Z said.
        floor_z=settings.target_floor_z,
        foot_bones=TARGET_FOOT_IK,
        frame_start=settings.mocap_frame_start,
        frame_end=settings.mocap_frame_end,
        trim_segment_ends=settings.lock_trim,
        min_segment_len=settings.lock_min_segment_len,
        blend_frames=settings.lock_blend_frames,
    )
    report = {
        "operation": "analyze_and_lock_teto_foot_ik",
        "drift_before": drift,
        "repair": repair,
        "stabilize": stabilized,
    }
    _record_report(scene, "foot_lock", report, "PREVIEW", "脚滑 XY Lock 等待检查", params)
    message = f"XY Lock 修复 {repair.get('repaired_count', 0)} 个区间"
    residual = (
        stabilized.get("residual_after") or stabilized.get("residual_before") or {}
    )
    message += (
        f"；闭环稳定 {stabilized.get('stabilized_segments', 0)} 段"
        f"（踝残余 {residual.get('max_pos_mm', 0.0):.1f} mm / "
        f"{residual.get('max_rot_deg', 0.0):.2f}°）"
    )
    pelvis = stabilized.get("pelvis_solve") or {}
    if pelvis.get("status") == "applied":
        message += (
            f"；骨盆修约 {pelvis.get('frames_corrected', 0)} 帧"
            f"（最大 {pelvis.get('max_correction_mm', 0.0):.1f} mm）"
        )
    if stabilized.get("verification") == "unconverged":
        message += "；个别段首帧残余未收敛（落地帧腿够程上限），见报告"
    return message


def _air_source_labels(scene):
    """Map each airborne radio to its annotation provenance, for the report."""

    labels = []
    for record in getattr(scene, "mcd_annotation_ranges", ()):
        if record.channel != annotation.CHANNEL_AIR:
            continue
        labels.append(
            {
                "frames": [int(record.frame_start), int(record.frame_end)],
                "source": str(record.source or ""),
            }
        )
    return labels


def _run_ground_feet(context, settings):
    scene = context.scene
    _require_restore_before_rerun(scene, "ground_feet")
    _model_root, armature, rig, _foot_ik = _validate_mmd_identity(settings)
    # The correction rides into MMD Visual Bake through the MMR copy
    # constraints.  Once those are muted (or the MMD armature already has an
    # Action) the bake is done, and moving the rig would no longer reach the
    # model - say so instead of writing a correction that does nothing.
    if not _active_mmr_constraints(armature, rig):
        raise RuntimeError(
            "MMR 到 MMD 的约束已经禁用（多半已完成 MMD Bake）；"
            "贴地锁定必须在 Bake 之前运行，请先恢复到“MMD Visual Bake”执行前"
        )
    animation_data = armature.animation_data
    if animation_data and (
        animation_data.action is not None or len(animation_data.nla_tracks) > 0
    ):
        raise RuntimeError(
            "原生 MMD 骨架已有 Action/NLA（多半已完成 Bake）；"
            "贴地锁定必须在 Bake 之前运行，请先恢复到“MMD Visual Bake”执行前"
        )
    _require_no_nla(rig, "MMR Rig")
    rig_action = _require_action(rig, "MMR Rig")
    if not _action_has_range_keys(
        rig_action, settings.mocap_frame_start, settings.mocap_frame_end
    ):
        raise RuntimeError("MMR Rig Action 没有覆盖完整动捕范围")
    _ensure_no_pending_preview(settings, "ground_feet")
    action = project.begin_action_preview(scene, rig, "ground_feet")
    # 贴地锁定本身不触发任何自动检测：只使用场景里已经存在的腾空区间
    # （不论是自动检测留下的还是手动标记的）。用户没先检测腾空时，
    # 这里就是空列表，于是只锁 planted 区间。
    airborne = annotation.get_channel_ranges(scene, annotation.CHANNEL_AIR)
    planted = _planted_ranges(scene)
    planted_spans = [tuple(item) for side in ("L", "R") for item in planted.get(side, ())]
    if not planted_spans:
        raise RuntimeError("没有贴地区间，请先检测贴地区间")
    # The pin works on ankle bones; the shoe sole hangs below them by the
    # model's own bind-pose drop (~5.6 cm on Teto).  Read it off the model
    # instead of carrying a hand tuned default.
    sole_offset = core_target.model_sole_offset(armature, settings.target_mesh)
    params = {
        "floor_z": settings.target_floor_z,
        "clearance": settings.target_clearance,
        "sole_offset": sole_offset,
        "smooth_radius": settings.ground_smooth_radius,
        "max_delta": settings.ground_max_delta,
    }
    result = core_target.ground_feet_outside_airborne(
        scene,
        armature,
        rig,
        action,
        airborne,
        planted_ranges=planted_spans,
        frame_start=settings.mocap_frame_start,
        frame_end=settings.mocap_frame_end,
        **params,
    )
    result["airborne_sources"] = _air_source_labels(scene)
    result["params"]["pin_mode"] = "planted"
    result["params"]["planted_spans"] = len(planted_spans)
    _record_report(scene, "ground_feet", result, "PREVIEW", "贴地锁定等待检查", params)
    message = (
        f"贴地锁定 {result.get('changed_frames', 0)} 帧"
        f"（仅贴地区间 {result.get('planted_frames', 0)} 帧）"
        f"，最大修正 {result.get('max_applied_correction', 0.0) * 1000:.0f} mm"
    )
    spans = result.get("airborne_segments", ())
    if spans:
        ballistic = sum(1 for item in spans if item.get("mode") == "ballistic")
        message += f"；腾空段 {len(spans)} 个（弹道重建 {ballistic}）"
    else:
        message += "；没有腾空区间——若这段舞有跳跃，请先标注再重跑，否则跳跃会被压平"
    suspects = result.get("unmarked_airborne_suspects", ())
    if suspects:
        worst = suspects[0]
        message += (
            f"；另有 {len(suspects)} 段没标腾空但脚离地很高"
            f"（最高 {worst['max_height'] * 100:.0f} cm，见报告）"
        )
    return message


def _run_export_prep(context, settings):
    scene = context.scene
    if not bool(scene.get("mcd_hand_ik_export_confirmed", False)):
        raise RuntimeError("请先确认 MMR 手部导出清理提醒，再生成 VMD 导出预览")
    _require_restore_before_rerun(scene, "export_prep")
    model_root, armature, rig, foot_ik = _validate_mmd_identity(settings)
    correction = _resolve_correction_empty(settings)
    if correction is not None:
        if model_root.parent != correction or rig.parent != correction:
            raise RuntimeError("Teto MMD Root 与 MMR Rig 必须共同位于全局校正 Empty 下")
        _validate_correction_transform(correction)
    if _active_mmr_constraints(armature, rig):
        raise RuntimeError("仍有活动的 MMR 到 MMD 约束；请先完成并接受 MMD Bake")
    action = _require_action(armature, "原生 MMD Armature")
    _require_no_nla(armature, "原生 MMD Armature")
    _validate_mmd_action(
        action,
        foot_ik,
        settings.mocap_frame_start,
        settings.mocap_frame_end,
    )
    _begin_structure_preview(scene, "export_prep")
    if correction is None:
        # 全局扶正没执行（GVHMR 源本身已对齐重力）：没有外层世界变换要搬到根骨骼，
        # 直接跳过根补偿烘焙。
        baked = {
            "operation": "bake_global_correction_to_all_parent",
            "skipped": True,
            "reason": "没有全局校正 Empty（“Teto 全局扶正”未执行）",
            "frames_keyed": 0,
            "values_written": 0,
        }
    else:
        baked = core_export.bake_global_correction_to_all_parent(
            scene,
            armature,
            correction,
            action=action,
            bone_name=MMD_ROOT_BONE,
            frame_start=settings.mocap_frame_start,
            frame_end=settings.mocap_frame_end,
            reset_correction=True,
        )
    delete_leg_fk, leg_reason = _leg_fk_cleanup_plan(armature, foot_ik)
    if delete_leg_fk:
        cleaned = core_export.remove_teto_leg_fk_curves(
            action, bone_names=MMD_LEG_FK_BONES
        )
        leg_ik = {
            "operation": "key_ik_toggle_state",
            "skipped": True,
            "reason": "腿由 MMD 原生 IK 驱动，VMD 必须保持足ＩＫ开启",
        }
    else:
        cleaned = {
            "operation": "remove_teto_leg_fk_curves",
            "skipped": True,
            "reason": leg_reason,
            "removed_fcurve_count": 0,
        }
        # The legs travel on FK curves, so the receiving model has to leave its
        # own 足ＩＫ / つま先ＩＫ alone; say so inside the VMD instead of asking
        # the user to find the toggle in every player.
        leg_ik = core_export.key_ik_toggle_state(
            armature,
            action,
            _leg_ik_toggle_targets(armature, foot_ik),
            enabled=False,
            frame_start=settings.mocap_frame_start,
            frame_end=settings.mocap_frame_end,
        )
    hand_export_cleanup = _prepare_teto_mmr_hand_export_cleanup(
        scene,
        armature,
        action,
        settings.mocap_frame_start,
        settings.mocap_frame_end,
    )
    _require_no_keyed_vmd_name_collisions(armature, action)
    offset = core_export.apply_vmd_floor_z_offset(
        scene,
        armature,
        action=action,
        bone_name=MMD_ROOT_BONE,
        z_offset=settings.vmd_floor_offset,
        frame_start=settings.mocap_frame_start,
        frame_end=settings.mocap_frame_end,
    )
    _validate_mmd_action(
        action,
        foot_ik,
        settings.mocap_frame_start,
        settings.mocap_frame_end,
        require_clean_fk=delete_leg_fk,
    )
    report = {
        "operation": "prepare_teto_vmd_export",
        "global_correction_bake": baked,
        "leg_fk_cleanup": cleaned,
        "leg_ik_toggle": leg_ik,
        "mmr_hand_export_cleanup": hand_export_cleanup,
        "floor_offset": offset,
        "validated_foot_ik": foot_ik,
    }
    _record_report(
        scene,
        "export_prep",
        report,
        "PREVIEW",
        "VMD 导出准备等待隐藏 MMR 检查",
        {"vmd_floor_offset": settings.vmd_floor_offset},
    )
    message = (
        f"导出预览已生成，删除 {cleaned['removed_fcurve_count']} 条腿 FK 曲线、"
        f"{hand_export_cleanup['removed_helper_fcurve_count']} 条手 IK 辅助曲线"
    )
    if correction is None:
        message += "；未执行全局扶正，已跳过根补偿烘焙"
    if leg_ik.get("bones"):
        message += "；已在 VMD 里关闭足ＩＫ / つま先ＩＫ"
    return message


def _run_foot_spacing(context, settings):
    """脚间距补偿步骤（排在**重定向之后**、贴地锁定之前）。

    只在源两脚靠得很近时把目标脚间距拉向 ``clearance_ratio × rest 间距``（双向），
    左右顺序跟随源、各推一半；写的是 ``foot_ik.L/R`` 的位置关键帧，不写任何 FK 腿
    曲线、也不写任何旋转曲线（踝部旋转保持重定向烘焙的原样）。
    原先它在 index 1（重定向之前），顺序是错的——改成重定向之后。
    """

    scene = context.scene
    _require_restore_before_rerun(scene, "foot_spacing")
    rig, action = _fix_rig_action(settings)
    source = getattr(settings, "source_armature", None)
    if source is None:
        raise RuntimeError("未设置源骨架（脚间距按源骨架的距离和左右顺序修正）")
    _ensure_no_pending_preview(settings, 'foot_spacing')
    action = project.begin_action_preview(scene, rig, 'foot_spacing')
    report = core_foot_spacing.compensate_foot_spacing_ik(
        scene,
        source,
        rig,
        action,
        frame_start=settings.mocap_frame_start,
        frame_end=settings.mocap_frame_end,
        clearance_ratio=settings.foot_spacing_clearance_ratio,
        close_fade_ratio=settings.foot_spacing_close_fade_ratio,
        crossing_fade_frames=settings.foot_spacing_crossing_fade,
        crossing_detection_frames=settings.foot_spacing_crossing_detect,
        crossing_min_gap_ratio=settings.foot_spacing_crossing_min_gap,
    )
    if report.get('status') != 'applied':
        raise RuntimeError('脚间距补偿未完成：' + str(report.get('reason') or report.get('status')))
    _record_report(scene, 'foot_spacing', report, 'PREVIEW', '补偿预览已生成')
    settings.foot_spacing_status = ''
    return '补偿预览已生成'


def _run_import_official(context, settings):
    """第 1 步 · 导入：一键把动作骨架、贴地数据、参考视频拿进场景。

    全自动、静默：三份 body npz 选哪份、一致性校验、绝对关节、官方第 5 步都在
    内部决定；状态栏只出现「已导入 / 未找到动作数据 / 缺少贴地数据，已改用骨架推算」
    这种短句（用户 2026-10-16 要求：底部提示框只有窄窄一条）。
    """

    # 就地写当前文件：先确保文件已落盘，第一次登记时顺手备份未被动过的原件。
    project.prepare_source_file()
    if not settings.initialized:
        project.ensure_project(context.scene)
    info = _official_scan(settings)
    if not info.get("exists") or not info.get("fbx"):
        raise RuntimeError('未找到动作数据，请检查目录')
    # 导入不走“预览/恢复”：先留一份执行前检查点，完成后直接记录并进入下一步。
    _require_restore_before_rerun(context.scene, 'import_official')
    _ensure_no_pending_preview(settings)
    project.create_checkpoint(context.scene, 'import_official_before')
    # 必须把扫描结果落到设置上（fbx / npz / 视频路径），否则后面的视频平面拿不到路径
    _official_apply_scan(settings, info)
    bucket = _official_import_fbx(context, str(settings.official_fbx or info["fbx"]))
    armature = bucket["armature"]
    if armature is None:
        raise RuntimeError('动作文件中没有骨架，请检查导出目录')
    settings.official_reference = armature
    settings.source_armature = armature
    # 导入完成后自动重新检测对象（原先第 0 步那个按钮）。
    project.discover_known_objects(context.scene)

    # 贴地数据：三档解析（动作数据目录 → 同片的官方结果 → 骨架推算），静默
    contacts_path, _text = _official_resolved_contacts(settings)
    settings.foot_contacts_npz = contacts_path or ""
    settings.contact_source = "NPZ" if contacts_path else "GEOMETRY"
    settings.foot_contacts_joints_npz = ""
    degraded = not contacts_path
    if contacts_path:
        sidecar = core_foot_contacts.resolve_joints_path(contacts_path, "")
        if sidecar:
            settings.foot_contacts_joints_npz = sidecar
        else:
            try:
                needs_velocity = "local_joints_positions" not in set(
                    core_official_import.npz_keys(contacts_path)
                )
            except Exception:  # noqa: BLE001
                needs_velocity = False
            if needs_velocity:
                # 路线 B：由代码内部用刚导入的骨架补速度项（用户看不到这一步）；
                # 补不上就退回骨架推算，并且只给一句短 warning。
                try:
                    if not _official_dump_joints_now(context, settings, frame_limit=0):
                        degraded = True
                except Exception:  # noqa: BLE001
                    settings.foot_contacts_joints_npz = ""
                    degraded = True

    if settings.official_ensure_30fps and (
        context.scene.render.fps != EXPECTED_FPS or context.scene.render.fps_base != 1.0
    ):
        project.ensure_fps(context.scene)

    video = str(settings.official_video or '')
    video_missing = not video
    if video:
        try:
            _official_video_plane(context, settings, video)
        except Exception as exc:
            print('[MoCap Doctor] reference video:', exc)
            video_missing = True

    imported_action = getattr(getattr(armature, 'animation_data', None), 'action', None)
    if imported_action:
        start, end = (int(value) for value in imported_action.frame_range)
    else:
        start, end = 1, int(info.get('npz_frames') or context.scene.frame_end)
    settings.mocap_frame_start = max(1, start)
    settings.mocap_frame_end = max(settings.mocap_frame_start, end)
    context.scene.frame_start = settings.mocap_frame_start
    context.scene.frame_end = settings.mocap_frame_end
    notes = []
    if degraded:
        notes.append('贴地区间改用骨架推算')
    if video_missing:
        notes.append('参考视频未载入')
    project.create_accepted_checkpoint(context.scene, 'import_official', '导入完成')
    settings.current_step = clamp_step(STEP_INDEX['import_official'] + 1)
    project.save_workfile(context.scene)
    if notes:
        settings.status_level = 'WARNING'
        return '导入完成，' + '；'.join(notes)
    settings.status_level = 'INFO'
    return '导入完成'


def _run_import_official_step(context, settings):
    """RUN_STEP_HANDLERS 用的薄包装（保持执行器签名一致）。"""

    return _run_import_official(context, settings)


def _retarget_objects(settings):
    """Validate the source/target pair and the mapping; returns (source, target)."""
    from .core import flowhmr_retarget as RT

    source = settings.source_armature
    target = settings.mmr_rig
    if source is None or source.type != 'ARMATURE':
        raise RuntimeError('请选择源骨架')
    if target is None or target.type != 'ARMATURE':
        raise RuntimeError('请选择目标骨架')
    if source == target:
        raise RuntimeError('源骨架与目标骨架不能相同')
    if not all(name in source.data.bones for name in ('Pelvis', 'L_Wrist', 'R_Wrist')):
        raise RuntimeError('源骨架不符合动作数据格式')
    RT._resolve_bmap(settings.retarget_bmap_choice, settings.retarget_bmap_custom)
    RT.find_arp()
    return source, target


def _run_retarget(context, settings):
    """Create a structural preview: ARP changes both rigs, not just one Action."""
    from .core import flowhmr_retarget as RT

    scene = context.scene
    source, target = _retarget_objects(settings)
    _require_restore_before_rerun(scene, 'retarget')
    _ensure_no_pending_preview(settings)
    if not any(_planted_ranges(scene).values()):
        _run_contacts(context, settings)
    if not any(_planted_ranges(scene).values()):
        raise RuntimeError('未检测到贴地区间，请检查源动作')
    _begin_structure_preview(scene, 'retarget')
    preset = str(getattr(settings, 'retarget_preset', '') or '')
    if not preset:
        from .core import rest_pose_preset as rest_presets
        preset = rest_presets.find_matching(getattr(target, 'name', ''))
    report = RT.retarget(
        source, target, settings.mocap_frame_start, settings.mocap_frame_end,
        settings.retarget_bmap_choice, settings.retarget_bmap_custom,
        rest_preset=preset,
    )
    report['skeleton_fixes'] = _apply_skeleton_fixes(context, settings)
    report['planted_at_retarget'] = {
        side: len(spans) for side, spans in _planted_ranges(context.scene).items()
    }
    # 手指校正并入重定向：静默执行，不额外开预览。失败不影响重定结果，
    # 只把原因记进报告。
    try:
        from .core.finger_align import fit_finger_masters
        rig_action = getattr(getattr(target, "animation_data", None), "action", None)
        if rig_action is None:
            report['finger_fit'] = {'status': 'skipped', 'reason': '目标骨架没有动作'}
        else:
            report['finger_fit'] = fit_finger_masters(
                context.scene, target, rig_action,
                frame_start=settings.mocap_frame_start,
                frame_end=settings.mocap_frame_end)
    except Exception as exc:  # noqa: BLE001
        report['finger_fit'] = {'status': 'skipped', 'reason': str(exc)}
    _record_report(scene, 'retarget', report, 'PREVIEW', '重定向预览已生成')
    settings.retarget_status = ''
    settings.status_level = 'INFO'
    return '重定向预览已生成'


class MD_OT_DeleteCheckpoints(Operator):
    """Delete every checkpoint file of this project (confirmation dialog)."""

    bl_idname = 'mocap_doctor.delete_checkpoints'
    bl_label = '删除所有检查点'

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        settings = _settings(context)
        try:
            _require_project(context)
            removed = project.delete_checkpoints(context.scene)
            settings.status_level = 'INFO'
            settings.status_message = f'已删除 {removed} 个检查点'
            self.report({'INFO'}, settings.status_message)
            return {'FINISHED'}
        except Exception as exc:
            settings.status_level = 'ERROR'
            settings.status_message = _short_reason(str(exc), limit=64)
            self.report({'ERROR'}, settings.status_message)
            return {'CANCELLED'}


class MD_OT_SaveRestPreset(Operator):
    """Freeze the current ARP rest snapshot as a reusable preset.

    Works after any successful retarget (the geometric alignment stores its
    result in the same scene snapshot ARP's Apply writes), so a model only has
    to be set up once.
    """

    bl_idname = 'mocap_doctor.save_rest_preset'
    bl_label = '保存 rest 预设'
    bl_description = '把当前静置姿态存成预设，之后同一模型可以直接复用'

    def execute(self, context):
        settings = _settings(context)
        try:
            _require_project(context)
            from .core import rest_pose_preset as rest_presets
            target = settings.mmr_rig
            if target is None:
                raise RuntimeError('请选择目标骨架')
            if not rest_presets.snapshot_exists(context.scene):
                raise RuntimeError('还没有静置姿态；先完成一次重定向')
            path = rest_presets.export_snapshot(context.scene, target.name, target.name)
            settings.retarget_preset = str(path)
            settings.status_level = 'INFO'
            settings.status_message = '已保存静置姿态预设'
            self.report({'INFO'}, settings.status_message)
            return {'FINISHED'}
        except Exception as exc:
            settings.status_level = 'ERROR'
            settings.status_message = _short_reason(str(exc), limit=64)
            self.report({'ERROR'}, settings.status_message)
            return {'CANCELLED'}


class MD_OT_MarkRetargetDone(Operator):
    """Manual mode: the user retargeted outside the wizard.

    Nothing is validated beyond "the target rig actually has an Action" — a
    missing Action means the retarget really did not happen, and continuing
    would fail everywhere downstream.  That case only warns; it never blocks.
    """

    bl_idname = 'mocap_doctor.mark_retarget_done'
    bl_label = '重定向已完成'
    bl_description = '手动模式：确认已在外部完成重定向，记录本步骤并进入下一步'

    def execute(self, context):
        settings = _settings(context)
        scene = context.scene
        try:
            _require_project(context)
            target = settings.mmr_rig
            if target is None or target.type != 'ARMATURE':
                raise RuntimeError('请选择目标骨架')
            action = getattr(getattr(target, 'animation_data', None), 'action', None)
            _require_restore_before_rerun(scene, 'retarget')
            project.create_accepted_checkpoint(scene, 'retarget', '手动重定向已完成')
            settings.current_step = clamp_step(STEP_INDEX['retarget'] + 1)
            project.save_workfile(scene)
            if action is None:
                settings.status_level = 'WARNING'
                settings.status_message = '已记录；但目标骨架没有动作，请确认重定向已完成'
            else:
                settings.status_level = 'INFO'
                settings.status_message = '已记录手动重定向'
            self.report({'INFO'}, settings.status_message)
            return {'FINISHED'}
        except Exception as exc:
            settings.status_level = 'ERROR'
            settings.status_message = _short_reason(str(exc), limit=64)
            self.report({'ERROR'}, settings.status_message)
            return {'CANCELLED'}


class MD_OT_RetargetPrepare(Operator):
    """Prepare ARP for a MANUAL retarget (left click in ARP's own panel).

    Runs exactly what the one-click path runs before ARP's own retarget: unbind,
    IK/FK, T-pose on frame 0, mapping import and the geometric rest alignment of
    arms/hands/fingers.  It then leaves the scene untouched so the user can do
    Redefine Rest Pose / Save Pose Rest / Retarget by hand - which is what makes
    other MMR skeletons (and other mappings) reachable.  Everything is inside a
    discardable preview, so a wrong manual attempt can be rolled back.
    """

    bl_idname = 'mocap_doctor.retarget_prepare'
    bl_label = '准备手动 ARP 重定向'
    bl_description = (
        '对齐静置姿态并载入映射，然后由你在 ARP 面板里手动执行'
        ' Redefine Rest Pose → Save Pose Rest → Retarget；完成后点“接受”记录本步骤'
    )

    def execute(self, context):
        settings = _settings(context)
        from .core import flowhmr_retarget as RT

        try:
            scene = context.scene
            source, target = _retarget_objects(settings)
            _require_restore_before_rerun(scene, 'retarget')
            _ensure_no_pending_preview(settings)
            if not any(_planted_ranges(scene).values()):
                _run_contacts(context, settings)
            _begin_structure_preview(scene, 'retarget')
            report, _state = RT.prepare(
                source, target, settings.mocap_frame_start, settings.mocap_frame_end,
                settings.retarget_bmap_choice, settings.retarget_bmap_custom,
            )
            _record_report(scene, 'retarget', report, 'PREVIEW', '已准备手动 ARP 重定向')
            settings.status_message = '已准备：在 ARP 里手动重定向后点接受'
            settings.status_level = 'INFO'
            self.report({'INFO'}, settings.status_message)
            return {'FINISHED'}
        except Exception as exc:
            settings.status_level = 'ERROR'
            settings.status_message = _short_reason(str(exc), limit=64)
            self.report({'ERROR'}, settings.status_message)
            return {'CANCELLED'}


def _apply_skeleton_fixes(context, settings):
    """Run A/B/C at the end of the retarget preview.

    A and B always apply.  C needs planted intervals, so a missing/empty
    detection is recorded as "skipped" instead of failing the whole retarget —
    the user can detect intervals later and rerun the step.
    """
    reports = {}
    for letter, runner in _TARGET_FIXES.items():
        try:
            report, _details = runner(context, settings)
        except Exception as exc:  # noqa: BLE001
            reason = str(exc)
            reports[letter] = {'operation': letter, 'skipped': reason}
            if letter == 'C' and 'planted' in reason:
                continue
            raise RuntimeError('骨架修复未完成：' + _short_reason(reason))
        reports[letter] = report
        reason = _skipped_reason(report)
        if reason and reason != 'already applied':
            if letter == 'C' and 'planted' in str(reason):
                continue
            if report.get('bones') or report.get('missing_bones'):
                raise RuntimeError('骨架修复未完成：模型缺少所需骨骼')
            raise RuntimeError('骨架修复未完成：' + _short_reason(reason))
    return reports


def _run_target_skeleton_fix(context, settings):
    """DEPRECATED: retained for old scripts; normal workflow runs this in retarget."""
    scene = context.scene
    _require_restore_before_rerun(scene, 'target_skeleton_fix')
    _ensure_no_pending_preview(settings)
    rig, _action = _fix_rig_action(settings)
    # Imported contact data is prepared automatically; the editor remains
    # available for manual corrections before creating this preview.
    if not any(_planted_ranges(scene).values()):
        _run_contacts(context, settings)
    if not any(_planted_ranges(scene).values()):
        raise RuntimeError('请先检测贴地区间')
    project.begin_action_preview(scene, rig, 'target_skeleton_fix')
    reports = {}
    for letter, runner in _TARGET_FIXES.items():
        report, _details = runner(context, settings)
        reports[letter] = report
        reason = _skipped_reason(report)
        if reason and reason != 'already applied':
            _record_report(scene, 'target_skeleton_fix', reports, 'FAILED', '骨架修复未完成')
            if report.get('bones'):
                raise RuntimeError('骨架修复未完成：模型缺少所需骨骼')
            raise RuntimeError('骨架修复未完成：' + _short_reason(reason))
    _record_report(scene, 'target_skeleton_fix', reports, 'PREVIEW', '修复预览已生成')
    settings.fix_status = ''
    return '修复预览已生成'


def _run_hand_spacing(context, settings):
    from .core.hand_spacing import compensate_hand_spacing_fk
    scene = context.scene
    _require_restore_before_rerun(scene, 'hand_spacing')
    _ensure_no_pending_preview(settings)
    rig, _base = _fix_rig_action(settings)
    source = settings.source_armature
    if source is None:
        raise RuntimeError('请选择源骨架')
    action = project.begin_action_preview(scene, rig, 'hand_spacing')
    report = compensate_hand_spacing_fk(
        scene, source, rig, action, frame_start=settings.mocap_frame_start,
        frame_end=settings.mocap_frame_end, strength=settings.hand_spacing_strength)
    _record_report(scene, 'hand_spacing', report, 'PREVIEW', '手间距预览已生成')
    return '手间距预览已生成'
# 一页多操作的步骤：接受预览后停在本页，不能被自动带到下一步。
MULTI_OP_STEPS = {'hand_spacing'}


class MD_OT_FingerAlign(Operator):
    """Remove the sideways kink FlowHMR leaves in finger joints."""

    bl_idname = 'mocap_doctor.finger_align'
    bl_label = '手指校正'
    bl_description = (
        '把每根手指后两节的弯折拉回同一平面，去掉源数据里的侧向歪折；'
        '只写手指旋转，弯折角度和张开角度不变。接受后保留，丢弃可还原'
    )

    def execute(self, context):
        settings = _settings(context)
        from .core.finger_align import fit_finger_masters

        try:
            scene = context.scene
            _require_project(context)
            _ensure_no_pending_preview(settings)
            rig, _base = _fix_rig_action(settings)
            action = project.begin_action_preview(scene, rig, 'hand_spacing')
            # 拟合到 fingers 控制器：整根手指刚性旋转指向最佳方向，指节拉直
            # （放弃逐指节弯曲这一自由度，换“绝不可能歪”的形态）。
            report = fit_finger_masters(
                scene, rig, action, frame_start=settings.mocap_frame_start,
                frame_end=settings.mocap_frame_end)
            _record_report(scene, 'hand_spacing', report, 'PREVIEW', '手指校正预览已生成')
            settings.status_level = 'INFO'
            settings.status_message = (
                f'手指校正预览已生成（整根拟合：'
                f'{report.get("masters_written")} 个控制器、'
                f'{report.get("details_flat")} 根指节已拉直）'
            )
            self.report({'INFO'}, settings.status_message)
            return {'FINISHED'}
        except Exception as exc:  # noqa: BLE001
            scene = context.scene
            if settings.preview_step_id == 'hand_spacing':
                project.record_step(scene, 'hand_spacing', 'FAILED', str(exc))
            settings.status_level = 'ERROR'
            settings.status_message = _short_reason(str(exc), limit=64)
            self.report({'ERROR'}, settings.status_message)
            return {'CANCELLED'}


RUN_STEP_HANDLERS = {
    'hand_spacing': _run_hand_spacing,
    "target_skeleton_fix": _run_target_skeleton_fix,
    "retarget": _run_retarget,
    "import_official": _run_import_official_step,
    "foot_spacing": _run_foot_spacing,
    "source_check": _run_source_check,
    "source_floor": _run_source_floor,
    "contacts": _run_contacts,
    "global_correction": _run_global_correction,
    "tilt": _run_tilt,
    "target_floor": _run_target_floor,
    "foot_lock": _run_foot_lock,
    "ground_feet": _run_ground_feet,
    "export_prep": _run_export_prep,
}


class MD_OT_PrepareVMDExport(Operator):
    bl_idname = "mocap_doctor.prepare_vmd_export"
    bl_label = "确认并生成 VMD 导出预览"
    bl_description = "将手 IK 驱动的可见上肢动作烘到腕、肘、手首后再生成导出预览"

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, context):
        layout = self.layout
        layout.label(text="将清理此工作文件中 MMR 创建的手部编辑控制。", icon="INFO")
        layout.label(text="会先确认上肢已烘到普通的“腕 / 肘 / 手首”骨骼，")
        layout.label(text="再恢复肘部层级，并删除只供 Blender 编辑使用的手 IK 骨。")
        layout.separator()
        layout.label(text="不会修改 PMX，也不会修改原始 Teto 模板。", icon="CHECKMARK")
        layout.label(text="会先创建检查点；丢弃预览即可完整恢复。", icon="LOOP_BACK")

    def execute(self, context):
        scene = context.scene
        scene["mcd_hand_ik_export_confirmed"] = True
        try:
            return bpy.ops.mocap_doctor.run_step("EXEC_DEFAULT", step_id="export_prep")
        finally:
            scene.pop("mcd_hand_ik_export_confirmed", None)


class MD_OT_RunStep(Operator):
    bl_idname = "mocap_doctor.run_step"
    bl_label = "运行当前 MoCap Doctor 步骤"
    bl_description = "生成预览供检查；接受后保留结果，丢弃则恢复执行前状态"

    step_id: StringProperty()

    @classmethod
    def description(cls, context, properties):
        help_text = {
            'retarget': '将源骨架的动作重定向到目标骨架，并校准手臂与手指的静置姿态。生成预览后，可接受或丢弃恢复。',
            'import_official': '从所选动作目录导入骨架、贴地区间和参考视频。生成预览后，可接受或丢弃恢复。',
            'foot_spacing': '仅在源骨架双脚靠近时调整目标脚间距，保持踝部旋转；可检查预览后接受或丢弃。',
            'hand_spacing': '双手靠近时，按源动作恢复手腕间距并解算FK手臂；保持手腕方向和手指动作。接受后保留，丢弃可还原。',
            'contacts': '从导入动作生成左右脚的贴地区间；已有人工修订会保留，可通过编辑按钮调整。',
        }
        return help_text.get(properties.step_id, cls.bl_description)

    def execute(self, context):
        settings = _settings(context)
        old_current_step = settings.current_step
        handler = RUN_STEP_HANDLERS.get(self.step_id)
        if handler is None:
            self.report({"ERROR"}, f"步骤没有自动执行器：{self.step_id}")
            return {"CANCELLED"}
        try:
            # 导入步骤在建项目之前就要能跑（它是第 0 步）
            if not is_project_free_step(self.step_id):
                _require_project(context)
            settings.busy = True
            settings.status_level = 'INFO'
            settings.status_message = '正在执行'
            message = handler(context, settings)
            settings.busy = False
            settings.status_message = message
            if self.step_id == "contacts" and project.current_file_is_workfile(settings):
                # 顺手存一下只允许发生在"打开的就是这个项目的工作文件"时：
                # 这一步现在从重定向页的按钮调用，面板里的自动保存绝不能落到
                # 一个不是工作文件的 .blend 上（原始文件/别人的文件）。
                project.save_workfile(context.scene)
            self.report({"INFO"}, message)
            return {"FINISHED"}
        except Exception as exc:
            import traceback
            traceback.print_exc()
            settings.current_step = old_current_step
            message = str(exc).splitlines()[0] or '操作未完成'
            settings.status_message = _short_reason(message, limit=64)
            settings.status_level = 'ERROR'
            record = project.find_step_record(settings, self.step_id, create=False)
            if record is None or record.status not in {'ACCEPTED', 'STALE'}:
                project.record_step(context.scene, self.step_id, 'FAILED', str(exc))
            self.report({'ERROR'}, settings.status_message)
            return {'CANCELLED'}
        finally:
            settings.busy = False


class MD_OT_CreateProject(Operator, ExportHelper):
    bl_idname = "mocap_doctor.create_project"
    bl_label = "创建 MoCap Doctor 工作副本"
    bl_description = "保留原文件，创建固定工作文件和基线检查点"

    filename_ext = ".blend"
    filter_glob: StringProperty(default="*.blend", options={"HIDDEN"})

    def invoke(self, context, event):
        if not bpy.data.filepath:
            self.report({"ERROR"}, "请先保存或打开一个源 .blend 文件")
            return {"CANCELLED"}
        source = Path(bpy.data.filepath)
        self.filepath = str(source.with_name(source.stem + "_mocap_doctor_work.blend"))
        return super().invoke(context, event)

    def execute(self, context):
        old_step = context.scene.mocap_doctor.current_step
        try:
            baseline = project.initialize_project(
                context.scene,
                self.filepath,
                target_template_path=_default_template_path(),
            )
            self.report({"INFO"}, f"工作项目已创建，基线：{baseline.name}")
            return {"FINISHED"}
        except Exception as exc:
            context.scene.mocap_doctor.current_step = old_step
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MD_OT_DiscoverObjects(Operator):
    bl_idname = "mocap_doctor.discover_objects"
    bl_label = "重新检测对象"

    def execute(self, context):
        found = project.discover_known_objects(context.scene)
        names = [key for key, value in found.items() if value]
        self.report({"INFO"}, "已检测：" + (", ".join(names) if names else "没有匹配对象"))
        settings = context.scene.mocap_doctor
        if settings.initialized:
            project.save_workfile(context.scene)
        return {"FINISHED"}


class MD_OT_UseSceneRange(Operator):
    """Copy the timeline range into the wizard's fixed test range."""

    bl_idname = "mocap_doctor.use_scene_range"
    bl_label = "用场景帧范围"
    bl_description = "把时间轴当前起止帧作为这一步要处理的区间（测试时只跑这一段）"

    def execute(self, context):
        settings = _settings(context)
        scene = context.scene
        start, end = int(scene.frame_start), int(scene.frame_end)
        if end < start:
            start, end = end, start
        settings.mocap_frame_start = start
        settings.mocap_frame_end = end
        self.report({"INFO"}, f"区间 {start}–{end}")
        return {"FINISHED"}


class MD_OT_EnsureFPS(Operator):
    bl_idname = "mocap_doctor.ensure_fps"
    bl_label = "设为 30fps"

    def execute(self, context):
        changed = project.ensure_fps(context.scene)
        if hasattr(context.scene, "mocap_doctor") and context.scene.mocap_doctor.initialized:
            project.save_workfile(context.scene)
        self.report({"INFO"}, "已设为 30fps" if changed else "场景已经是 30fps")
        return {"FINISHED"}


class MD_OT_SetRangeBoundary(Operator):
    bl_idname = "mocap_doctor.set_range_boundary"
    bl_label = "设置动捕范围边界"

    boundary: StringProperty(default="START")

    def execute(self, context):
        settings = _settings(context)
        frame = int(context.scene.frame_current)
        if self.boundary == "START":
            settings.mocap_frame_start = frame
            context.scene.frame_start = frame
        else:
            settings.mocap_frame_end = frame
            context.scene.frame_end = frame
        if settings.mocap_frame_end < settings.mocap_frame_start:
            self.report({"WARNING"}, "结束帧早于开始帧，请继续设置另一端")
        if settings.initialized:
            project.save_workfile(context.scene)
        return {"FINISHED"}


class MD_OT_SyncRangeFromScene(Operator):
    bl_idname = "mocap_doctor.sync_range_from_scene"
    bl_label = "读取场景范围"

    def execute(self, context):
        project.sync_scene_range(context.scene, _settings(context))
        if context.scene.mocap_doctor.initialized:
            project.save_workfile(context.scene)
        return {"FINISHED"}


def _navigation_status_message(settings, step_id):
    record = project.find_step_record(settings, step_id, create=False)
    if record is not None and record.status in {"ACCEPTED", "STALE", "FAILED"}:
        return '浏览已执行步骤，动作未回退'
    return ""


class MD_OT_NavigateStep(Operator):
    bl_idname = "mocap_doctor.navigate_step"
    bl_label = "仅浏览步骤页面"
    bl_description = "只切换面板页面，不恢复文件、撤销结果或改变步骤状态"

    delta: IntProperty(default=0)
    target: IntProperty(default=-1)

    def execute(self, context):
        settings = _settings(context)
        if settings.preview_step_id:
            self.report({"ERROR"}, "请先接受或丢弃当前预览")
            return {"CANCELLED"}
        settings.current_step = clamp_step(self.target if self.target >= 0 else settings.current_step + self.delta)
        # Future/unexecuted pages are harmless previews of their controls;
        # avoid showing a rollback warning when there is nothing to undo.
        settings.status_message = _navigation_status_message(
            settings,
            step_at(settings.current_step).id,
        )
        settings.status_level = 'INFO'
        # Browsing a page is UI state: no 190 MB .blend save per click (0.3 s
        # here, longer on Windows).  The page index goes to the manifest; the
        # .blend is marked dirty and saved with the next real change.
        project.save_manifest(context.scene)
        return {"FINISHED"}


class MD_OT_RestoreBeforeStep(Operator):
    bl_idname = "mocap_doctor.restore_before_step"
    bl_label = "恢复并重新执行此步骤"
    bl_description = "恢复此步骤执行前的检查点，恢复完成后停留在此步骤；不会仅切换页面"

    step_id: StringProperty()

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        settings = _settings(context)
        try:
            _require_project(context)
            target_index = STEP_INDEX.get(self.step_id)
            if target_index is None:
                raise RuntimeError("未知工作流步骤")
            candidates = []
            for record in settings.steps:
                index = STEP_INDEX.get(record.step_id, -1)
                if index < target_index and record.status == "ACCEPTED" and record.checkpoint:
                    path = Path(project.resolve_project_file(
                        settings, record.checkpoint, "checkpoints"))
                    if path.is_file():
                        candidates.append((index, path))
            record = project.find_step_record(settings, self.step_id, create=False)
            before = project.resolve_project_file(settings, record.before_checkpoint, 'checkpoints') if record and record.before_checkpoint else ''
            if before and Path(before).is_file():
                candidates.append((target_index, Path(before)))
            if not candidates:
                raise RuntimeError("找不到此步骤执行前的检查点")
            _index, checkpoint = max(candidates, key=lambda item: item[0])
            label = step_at(target_index).label
            project.restore_checkpoint(
                checkpoint,
                project.resolve_work_filepath(settings),
                resume_step_id=self.step_id,
                reset_step_id=self.step_id,
                message=f"已恢复到“{label}”执行前，可以重新运行",
            )
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MD_OT_ReloadAirHints(Operator):
    bl_idname = "mocap_doctor.reload_air_hints"
    bl_description = (
        "从 planted 检测结果里取「双脚同时离地」的帧作为腾空提示初稿。"
        "只在腾空轨道为空时写入；已有标注不会被覆盖（标错的跳跃比漏标贵得多）"
    )
    bl_label = "载入自动腾空提示"

    def execute(self, context):
        settings = _settings(context)
        try:
            _require_project(context)
            scene = context.scene
            if annotation.get_channel_ranges(scene, annotation.CHANNEL_AIR):
                raise RuntimeError(
                    "腾空轨道里已经有标注，这次载入不改动它；"
                    "要重新载入请先在标注编辑器里清空腾空区间"
                )
            record = project.find_step_record(settings, "contacts", create=False)
            report_path = (
                project.resolve_project_file(settings, record.artifact_path, "reports")
                if record is not None
                else ""
            )
            if not report_path or not Path(report_path).is_file():
                raise RuntimeError(
                    "还没有 planted 检测结果；请先运行「Planted 检测与修订」"
                )
            with Path(report_path).open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            frames = []
            for frame in range(
                int(settings.mocap_frame_start), int(settings.mocap_frame_end) + 1
            ):
                states = []
                for side in ("L", "R"):
                    per_frame = report.get("feet", {}).get(side, {}).get("per_frame", {})
                    entry = per_frame.get(str(frame)) or per_frame.get(frame) or {}
                    states.append(entry.get("state"))
                if states[0] == "airborne_or_lifted" and states[1] == "airborne_or_lifted":
                    frames.append(frame)
            if not frames:
                # GVHMR smears real jumps into near-ground, so this take's data
                # may never show both feet off the floor at once.
                raise RuntimeError(
                    "检测结果里没有「双脚同时离地」的帧——这段数据里跳跃被抹平了，"
                    "自动提示帮不上忙，请在视频里手动标注腾空区间"
                )
            # The track was checked empty above, but its "initialized" flag
            # survives a manual clear - without force_effective the reload
            # filled only the hidden auto channel and said "已载入 N 帧".
            annotation.set_air_auto_ranges(
                scene, frames_to_ranges(frames), force_effective=True
            )
            settings.status_message = f"已载入 {len(frames)} 帧腾空提示初稿"
            self.report({"INFO"}, settings.status_message)
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001 - surface the reason in the UI
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


def _pkl_hand_source(context):
    """Resolve the take dir + locate merged/raw pkls; raise a readable error."""

    settings = _settings(context)
    take_dir = str(getattr(settings, "gvhmr_take_dir", "") or "").strip()
    if not take_dir:
        raise RuntimeError("请先选择 GVHMR 输出目录（含 hamer/merged.pkl）")
    merged, raw = core_pkl_hand.find_take_pkls(Path(bpy.path.abspath(take_dir)))
    if merged is None:
        raise RuntimeError("目录里找不到 hamer/merged.pkl（或顶层 *.pkl）")
    return settings, merged, raw


class MD_OT_PklHandDetect(Operator):
    bl_idname = "mocap_doctor.pkl_hand_detect"
    bl_label = "检测手部候选段"
    bl_description = (
        "读原始 HaMeR 数据（没有则读 merged pkl）的腕/指旋转帧差与高频残余，"
        "把候选坏段写入手部自动轨道供人工筛改。"
        "手型错但跳变速率正常的段算法看不见，仍需人标"
    )

    def execute(self, context):
        settings = _settings(context)
        try:
            settings, merged, raw = _pkl_hand_source(context)
            detect_src = raw if raw is not None else merged
            result = core_pkl_hand.detect_candidates(detect_src)
            first_work = int(getattr(settings, "source_pkl_frame_start", 1) or 1)
            shown = {"L": [], "R": []}
            for side in ("L", "R"):
                candidates = result[side]
                auto_ranges = [
                    (int(c["start"]) + first_work, int(c["end"]) + first_work)
                    for c in candidates
                ]
                annotation.set_hand_auto_hints(
                    context.scene,
                    side,
                    auto_ranges,
                    initialize_manual=True,
                    force_manual=False,
                    rebuild=side == "R",
                )
                shown[side] = [
                    {
                        "frames": [int(c["start"]) + first_work,
                                   int(c["end"]) + first_work],
                        "kind": str(c["kind"]),
                        "strategy": str(c["strategy"]),
                        "peak": float(c["peak_deg"]),
                    }
                    for c in candidates
                ]
            context.scene["mcd_pkl_hand_candidates"] = shown
            kind_note = "按原始 HaMeR 矩阵检测" if raw is not None else "按 merged pkl 检测"
            settings.status_message = (
                f"{kind_note}：L {len(shown['L'])} 段 / R {len(shown['R'])} 段"
                "已写入手部自动轨道"
            )
            self.report({"INFO"}, settings.status_message)
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MD_OT_PklHandRepair(Operator):
    bl_idname = "mocap_doctor.pkl_hand_repair"
    bl_label = "写出修复 pkl"
    bl_description = (
        "按手部手动轨道的坏段重写 merged pkl 的腕部与手指旋转，"
        "输出 *_repaired.pkl；源 pkl 不改，需用修复文件重新导入"
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        settings = _settings(context)
        try:
            settings, merged, _raw = _pkl_hand_source(context)
            scene = context.scene
            if scene.mcd_annotation_mode:
                annotation.commit_track_reassignments(scene, rebuild=False)
            first_work = int(getattr(settings, "source_pkl_frame_start", 1) or 1)
            strategy_prop = str(getattr(settings, "hand_pkl_strategy", "auto"))
            channel_prop = str(getattr(settings, "hand_pkl_channel", "both"))
            candidates = scene.get("mcd_pkl_hand_candidates") or {}

            segments = []
            used_fallback = False
            for side in ("L", "R"):
                manual = annotation.get_channel_ranges(
                    scene, annotation.CHANNEL_HAND_L_MANUAL if side == "L"
                    else annotation.CHANNEL_HAND_R_MANUAL)
                source_ranges = list(manual)
                suggested = {}
                if not source_ranges:
                    auto = annotation.get_channel_ranges(
                        scene, annotation.CHANNEL_HAND_L_AUTO if side == "L"
                        else annotation.CHANNEL_HAND_R_AUTO)
                    source_ranges = list(auto)
                    if source_ranges:
                        used_fallback = True
                        for item in candidates.get(side, ()):
                            frames = item.get("frames") or (0, 0)
                            suggested[(int(frames[0]), int(frames[1]))] = item.get(
                                "strategy", "bridge")
                for start, end in source_ranges:
                    p_start = int(start) - first_work
                    p_end = int(end) - first_work
                    if p_end < 0:
                        continue
                    p_start = max(0, p_start)
                    if strategy_prop != "auto":
                        strategy = strategy_prop
                    else:
                        strategy = suggested.get((int(start), int(end)), "bridge")
                    segments.append({
                        "side": side,
                        "start": p_start,
                        "end": p_end,
                        "strategy": strategy,
                        "channel": channel_prop,
                    })
            if not segments:
                raise RuntimeError(
                    "没有可用的手部坏段：先在自动检测或标注编辑器里标出区间"
                )
            dst = merged.with_name(merged.stem + "_repaired.pkl")
            report_path = merged.with_name(merged.stem + "_repaired_report.json")
            report = core_pkl_hand.repair_pkl(
                merged, dst, segments, report_path=report_path
            )
            done = sum(
                1 for item in report["segments"]
                if item["strategy"] != "skipped_segment_at_edge"
            )
            settings.status_message = (
                f"已写出 {dst.name}（{done} 段）；"
                "请用修复后的 pkl 重新导入，再开始向导"
            )
            if used_fallback:
                settings.status_message += "（手动轨道为空，用了自动候选段）"
            self.report({"INFO"}, settings.status_message)
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


def _silent_root_pivot_fix(context, settings):
    """重定向里程碑时静默修正骨盆旋转支点（core_target.fix_root_pivot）。

    ARP 映射把源骨盆的旋转放在 torso_root（センター高度，比髋关节低 ~0.39 m）上转，胯部左右摆被反向抵掉；
    这里只平移 torso_root、让旋转绕 hips 头生效，旋转与脚 IK 都不动。找不到 Rig/动作就跳过，不阻断记录。
    """
    try:
        rig = _require_object(settings, "mmr_rig", "ARMATURE", OBJECT_NAMES["mmr_rig"])
    except RuntimeError:
        return "（未找到 MMR Rig，未做骨盆支点修正）"
    action = getattr(getattr(rig, "animation_data", None), "action", None)
    if action is None:
        return "（MMR Rig 没有活动 Action，未做骨盆支点修正）"
    report = core_target.fix_root_pivot(rig, action)
    if settings.initialized:
        project.write_report(context.scene, "retarget_root_pivot", report)
    if report.get("skipped") == "already applied":
        return "（骨盆支点已修正过）"
    if report.get("skipped"):
        return f"（未做骨盆支点修正：{report['skipped']}）"
    return f"；已修正骨盆旋转支点（{report['frames']} 帧，最大平移 {report['max_shift_m'] * 100:.1f} cm）"


def _silent_upper_body_fix(context, settings):
    """重定向里程碑时静默把上半身位置对齐源（core_target.fix_upper_body_follow），紧接在骨盆支点修正之后。

    ARP 按世界朝向逐骨映射，但源靠骨盆关节以上 0.23 m 的腰段侧倾来抵消胯摆，Teto 的上半身链从 hips 头才开始、
    拿的是 Spine2 的朝向，于是胯摆时肩跟着晃（0001-0999 第 151–277 帧：源 0.077 m，Teto 0.149 m）。
    这里逐帧绕 hips 头转 spine_fk.001：肩中相对髋中的左右偏移照抄源（1:1，与髋、脚一致），前后偏移按躯干长度比缩放；
    髋、腿、脚和 torso_root 都不动。找不到 Rig/动作/源骨架就跳过，出错也不阻断记录。
    """
    try:
        rig = _require_object(settings, "mmr_rig", "ARMATURE", OBJECT_NAMES["mmr_rig"])
    except RuntimeError:
        return "（未找到 MMR Rig，未做上半身对齐）"
    action = getattr(getattr(rig, "animation_data", None), "action", None)
    if action is None:
        return "（MMR Rig 没有活动 Action，未做上半身对齐）"
    try:
        report = core_target.fix_upper_body_follow(rig, getattr(settings, "source_armature", None), action)
    except Exception as exc:  # noqa: BLE001 - a milestone must still be recordable
        report = {"operation": "fix_upper_body_follow", "skipped": f"error: {exc}"}
    if settings.initialized:
        project.write_report(context.scene, "retarget_upper_body", report)
    if report.get("skipped") == "already applied":
        return "（上半身已对齐过）"
    if report.get("skipped"):
        return f"（未做上半身对齐：{report['skipped']}）"
    err = report["shoulder_lateral_err_cm"]
    return (f"；上半身位置已对齐源（{report['frames']} 帧，肩相对源横向偏差 P95 "
            f"{err['before']['p95']:.1f}→{err['after']['p95']:.1f} cm）")


# ---------------------------------------------------------------------------
# A / B / C：三个目标骨架修复按钮
# Q15：原来的“记录里程碑”操作本身不要了，那一页改成三个按钮分别执行 A/B/C。
# Q17：结果就在按钮下面，每题一行短句（那块地方显示不了多少字）。
# A 只平移 torso_root，B 只写 spine_fk.001 的四元数，C 只写 torso_root 的高度，
# 三者互不覆盖；C 需要 planted 区间（检测器读源骨架数据，保持不变）。
# 三个按钮都不存盘：它们不改当前打开的文件路径，工作文件的落盘仍走向导原有
# 的保存路径（save_workfile 只认打开的那个文件，绝不对别的路径另存）。
# ---------------------------------------------------------------------------


def _short_reason(text, limit=26):
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _skipped_reason(report):
    if report.get("skipped"):
        return str(report["skipped"])
    status = str(report.get("status") or "")
    if status.startswith("skipped"):
        return str(report.get("reason") or status)
    return ""


def _fix_line(letter, report, applied):
    why = _skipped_reason(report)
    if why:
        return f"{letter} 跳过：{_short_reason(why)}"
    return f"{letter} 已应用 · {applied}"


def _fix_rig_action(settings):
    rig = _require_object(settings, "mmr_rig", "ARMATURE", OBJECT_NAMES["mmr_rig"])
    action = getattr(getattr(rig, "animation_data", None), "action", None)
    if action is None:
        raise RuntimeError("MMR Rig 还没有活动 Action（先完成重定向）")
    return rig, action


def _target_fix_root_pivot(context, settings):
    """A：把重定向烤进 torso_root 的骨盆旋转改回绕 hips 头转。"""

    rig, action = _fix_rig_action(settings)
    report = core_target.fix_root_pivot(rig, action)
    if settings.initialized:
        project.write_report(context.scene, "retarget_root_pivot", report)
    applied = (
        f"{report.get('frames', 0)} 帧 · 最大 {report.get('max_shift_m', 0.0) * 100:.1f} cm"
    )
    return report, applied


def _target_fix_upper_body(context, settings):
    """B：让上半身位置跟着源骨架（FlowHMR 原生名，不再找 f_avg_ 前缀）。"""

    rig, action = _fix_rig_action(settings)
    try:
        report = core_target.fix_upper_body_follow(
            rig, getattr(settings, "source_armature", None), action
        )
    except Exception as exc:  # noqa: BLE001 - 报告里要写清楚，不能静默
        report = {"operation": "fix_upper_body_follow", "skipped": f"error: {exc}"}
    if settings.initialized:
        project.write_report(context.scene, "retarget_upper_body", report)
    err = report.get("shoulder_lateral_err_cm") or {}
    before = (err.get("before") or {}).get("p95")
    after = (err.get("after") or {}).get("p95")
    if before is None or after is None:
        applied = f"{report.get('frames', 0)} 帧"
    else:
        applied = f"{report.get('frames', 0)} 帧 · 肩差 {before:.1f}→{after:.1f} cm"
    return report, applied


def _target_fix_pelvis_reach(context, settings):
    """C：按源骨架腿长比重解骨盆高度（planted 区间来自源骨架的检测结果）。"""

    scene = context.scene
    rig, action = _fix_rig_action(settings)
    armature = _require_object(
        settings, "mmd_armature", "ARMATURE", OBJECT_NAMES["mmd_armature"]
    )
    source = getattr(settings, "source_armature", None)
    if source is None:
        raise RuntimeError("未设置源骨架（planted 区间和腿长比重都从它量）")
    planted = _planted_ranges(scene)
    if not any(planted.values()):
        raise RuntimeError("planted 区间为空：先跑「检测 planted 区间」")
    report = core_target.settle_pelvis_for_reach(
        scene,
        source,
        armature,
        rig,
        action,
        planted,
        correction_max=settings.pelvis_correction_max,
        ground_mesh=settings.target_mesh,
        ground_clearance=settings.target_clearance,
        floor_z=settings.target_floor_z,
        sole_offset=core_target.model_sole_offset(armature, settings.target_mesh),
        sole_dirs=core_target.sole_contact_offsets(armature, settings.target_mesh),
        frame_start=settings.mocap_frame_start,
        frame_end=settings.mocap_frame_end,
        trim_segment_ends=settings.lock_trim,
        min_segment_len=settings.lock_min_segment_len,
    )
    if settings.initialized:
        project.write_report(scene, "retarget_pelvis_reach", report)
    applied = (
        f"{report.get('frames_corrected', 0)} 帧 · "
        f"最大 {report.get('max_correction_mm', 0.0):.1f} mm"
    )
    return report, applied


_TARGET_FIXES = {
    "A": _target_fix_root_pivot,
    "B": _target_fix_upper_body,
    "C": _target_fix_pelvis_reach,
}


class _MD_OT_TargetFix(Operator):
    """A/B/C 三个按钮的公共执行体；子类只给 bl_idname / 标签 / 说明。"""

    bl_options = {"REGISTER", "UNDO"}
    which = ""
    status_attr = ""

    def execute(self, context):
        settings = _settings(context)
        runner = _TARGET_FIXES.get(self.which)
        if runner is None:
            self.report({"ERROR"}, f"未知修复：{self.which}")
            return {"CANCELLED"}
        try:
            _require_project(context)
            report, applied = runner(context, settings)
            line = _fix_line(self.which, report, applied)
            setattr(settings, self.status_attr, line)
            settings.status_message = line
            self.report({"INFO"}, line)
            return {"FINISHED"}
        except Exception as exc:  # noqa: BLE001 - 失败也要在面板上留一行
            line = f"{self.which} 失败：{_short_reason(exc)}"
            setattr(settings, self.status_attr, line)
            settings.status_message = line
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MD_OT_TargetFixA(_MD_OT_TargetFix):
    bl_idname = "mocap_doctor.target_fix_a"
    bl_label = "A 骨盆支点"
    bl_description = (
        "把重定向烤进 torso_root 的骨盆旋转改回绕 hips 头转：只平移 torso_root，"
        "旋转和脚 IK 都不动。不读源骨名，任何时候都能跑。"
    )
    which = "A"
    status_attr = "fix_a_status"


class MD_OT_TargetFixB(_MD_OT_TargetFix):
    bl_idname = "mocap_doctor.target_fix_b"
    bl_label = "B 上半身跟随"
    bl_description = (
        "让上半身位置跟着源骨架（读 FlowHMR 原生名 Pelvis / L_Hip / R_Hip / "
        "L_Shoulder / R_Shoulder）：逐帧绕 hips 头转 spine_fk.001，髋、腿、脚和 "
        "torso_root 都不动。需要源骨架有动画。"
    )
    which = "B"
    status_attr = "fix_b_status"


class MD_OT_TargetFixC(_MD_OT_TargetFix):
    bl_idname = "mocap_doctor.target_fix_c"
    bl_label = "C 骨盆够程"
    bl_description = (
        "按源骨架的腿长比重逐帧解骨盆高度（写 torso_root）：Teto 的腿比拍摄者长，"
        "同一个髋高会让膝盖比源更直、踩实的脚够不到地面。planted 区间读的是源骨架"
        "的检测结果，需要先跑「检测 planted 区间」。"
    )
    which = "C"
    status_attr = "fix_c_status"


class MD_OT_TargetFixAll(Operator):
    """一个按钮跑完 A/B/C（用户 2026-10-16：三个按钮合成一个）。

    依次执行既有的三个修复（各自逻辑不变），把三句结果压成**一行**写进
    ``settings.fix_status``：只保留每个修复的判定词（已应用 / 跳过 / 失败…）。
    """

    bl_idname = "mocap_doctor.target_fix_all"
    bl_label = "修复骨架"
    bl_description = "生成骨架修复预览；检查后接受，或丢弃以恢复执行前的动作"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        return bpy.ops.mocap_doctor.run_step('EXEC_DEFAULT', step_id='target_skeleton_fix')


class MD_OT_CreateCheckpoint(Operator):
    bl_idname = "mocap_doctor.create_checkpoint"
    bl_label = "记录当前状态"

    label: StringProperty(default="manual_milestone")
    step_id: StringProperty(default="")

    def execute(self, context):
        old_step = context.scene.mocap_doctor.current_step
        try:
            settings = _require_project(context)
            note = ""
            if self.step_id:
                _require_restore_before_rerun(context.scene, self.step_id)
                if self.step_id == "retarget":
                    note = _silent_root_pivot_fix(context, settings)
                    note += _silent_upper_body_fix(context, settings)
                settings.current_step = clamp_step(STEP_INDEX.get(self.step_id, settings.current_step) + 1)
                path = project.create_accepted_checkpoint(
                    context.scene,
                    self.step_id,
                    "人工步骤已记录",
                    label=self.label,
                )
            else:
                path = project.create_checkpoint(context.scene, self.label)
            project.save_workfile(context.scene)
            settings.status_message = f"已记录检查点：{path.name}{note}"
            self.report({"INFO"}, settings.status_message)
            return {"FINISHED"}
        except Exception as exc:
            context.scene.mocap_doctor.current_step = old_step
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MD_OT_RestoreLastCheckpoint(Operator):
    bl_idname = "mocap_doctor.restore_last_checkpoint"
    bl_label = "恢复最近检查点"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        settings = _settings(context)
        try:
            _require_project(context)
            project.restore_checkpoint(settings.last_checkpoint, project.resolve_work_filepath(settings))
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MD_OT_EnterAnnotationMode(Operator):
    bl_idname = "mocap_doctor.enter_annotation_mode"
    bl_label = "编辑区间"

    channel_group: StringProperty(default="HAND")

    def execute(self, context):
        try:
            if self.channel_group == "HAND":
                # Hand bad-range marking is a pre-project tool (pkl repair);
                # no step record is needed to open the editor.
                pass
            else:
                _require_project(context)
            if self.channel_group == "FOOT":
                # 检测按钮现在是重定向/修复页上的普通工具，不再走向导的步骤记录，
                # 所以这里不能要求"步骤未提交"（用户 2026-10-16：按过检测后编辑按钮失效）。
                record = project.find_step_record(
                    context.scene.mocap_doctor, "contacts", create=False
                )
                report_path = (
                    project.resolve_project_file(
                        context.scene.mocap_doctor, record.artifact_path, "reports"
                    )
                    if record is not None
                    else ""
                )
                if (not report_path or not Path(report_path).is_file()) and not any(_planted_ranges(context.scene).values()):
                    raise RuntimeError('请先检测贴地区间')
            elif self.channel_group == "AIR":
                # Airborne spans are read off the video, not off the data, so
                # this editor has no prerequisite - the auto hints are optional.
                _require_restore_before_rerun(context.scene, "ground_feet")
            was_open = bool(context.scene.mcd_annotation_mode)
            _area, converted, step_id = _activate_annotation_editor(
                context.scene,
                context.screen,
                self.channel_group,
            )
            action = '已切换到' if was_open else '已打开'
            label = {'contacts': '贴地区间', 'ground_feet': '腾空区间'}.get(step_id, '手部区间')
            settings = _settings(context)
            settings.status_message = action + label + '编辑'
            settings.status_level = 'INFO'
            self.report({'INFO'}, settings.status_message)
            return {'FINISHED'}
        except Exception as exc:
            settings = _settings(context)
            settings.status_message = _short_reason(str(exc), limit=64)
            settings.status_level = 'ERROR'
            self.report({'ERROR'}, settings.status_message)
            return {'CANCELLED'}


class MD_OT_EnterAnnotationAir(Operator):
    """腾空区间编辑器的薄包装：同一个实现，只是把说明做成按钮的悬停提示。

    Q9：面板上的长提示改成 tooltip。这一段（原来占 5 行 label）讲的就是这个按钮
    旁边那块东西，放这里最合适。
    """

    bl_idname = "mocap_doctor.enter_annotation_air"
    bl_label = "编辑腾空区间"
    bl_description = (
        "腾空区间 = 两只脚都不在地面的帧：从最后一只脚离地之后起，到第一只脚触地之前止。\n"
        "一只脚在地面的过渡帧不用标；漏标会把跳跃压平。\n"
        "非腾空帧一律把最低的脚钉到地面；跳高由区间时长决定（g·T²/8）。\n"
        "必须在 MMD Visual Bake 之前运行。"
    )
    execute = MD_OT_EnterAnnotationMode.execute
    channel_group: StringProperty(default="AIR")


class MD_OT_ExitAnnotationMode(Operator):
    bl_idname = "mocap_doctor.exit_annotation_mode"
    bl_label = "保存标注并返回时间轴"

    def execute(self, context):
        scene = context.scene
        old_current_step = scene.mocap_doctor.current_step
        try:
            settings = scene.mocap_doctor
            step_id = settings.annotation_step_id
            annotation.commit_track_reassignments(scene, rebuild=False)
            if step_id == "contacts":
                _write_effective_contact_report(scene)
            # Discard any native strip transform attempted after manually
            # unlocking a projection track; Scene ranges remain authoritative.
            annotation.rebuild_projection(scene)
            annotation.exit_annotation_mode(scene)
            planted_indicators.cleanup(scene)
            _restore_all_annotation_areas()
            settings.annotation_step_id = ""
            if step_id == 'contacts':
                settings.current_step = STEP_INDEX['contacts']
                project.mark_downstream_stale(scene, 'retarget')
                project.create_checkpoint(scene, 'contacts_edited')
            settings.status_message = '区间标注已保存'
            settings.status_level = 'INFO'
            if settings.initialized:
                project.save_workfile(scene)
            self.report({"INFO"}, settings.status_message)
            return {"FINISHED"}
        except Exception as exc:
            scene.mocap_doctor.current_step = old_current_step
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MD_OT_ResetEffectiveContacts(Operator):
    bl_idname = "mocap_doctor.reset_effective_contacts"
    bl_label = "最终区间重置为自动结果"
    bl_description = "明确丢弃左右脚的人工 planted 修订，重新复制锁定的自动检测区间"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        try:
            settings = _require_project(context)
            _ensure_no_pending_preview(settings)
            annotation.reset_planted_effective_to_auto(context.scene, "L", rebuild=False)
            annotation.reset_planted_effective_to_auto(context.scene, "R", rebuild=True)
            settings.status_message = '贴地区间已重置'
            settings.status_level = 'INFO'
            project.save_workfile(context.scene)
            self.report({'INFO'}, settings.status_message)
            return {'FINISHED'}
        except Exception as exc:
            settings = _settings(context)
            settings.status_message = _short_reason(str(exc), limit=64)
            settings.status_level = 'ERROR'
            self.report({'ERROR'}, settings.status_message)
            return {'CANCELLED'}


class MD_OT_MMDBake(Operator):
    bl_idname = "mocap_doctor.mmd_bake"
    bl_label = "检测并运行 MMD Visual Bake"

    def execute(self, context):
        settings = _settings(context)
        try:
            _require_project(context)
            if settings.mmd_bake_mode == "MANUAL":
                raise RuntimeError("当前策略为手动 Bake；完成后请点“检查手工 Bake 并清理 FK”")
            _ensure_no_pending_preview(settings, "mmd_bake")
            armature, rig, foot_ik, constrained = _validate_mmd_constraint_mapping(settings)
            rig_animation = rig.animation_data
            # agent 修复轨 = 有意的修正，visual_keying 求值 NLA 栈时会被烘进去，
            # 所以放行；只拦不认识的轨。
            # 空轨不动画任何东西：用户工作文件的 RIG 上有 103 条空 NlaTrack.*，
            # 把它们算进来默认的自动 Bake 直接报错。只拦有 strip 的陌生轨。
            foreign_tracks = []
            if rig_animation:
                from .core import agent_ops
                foreign_tracks = [
                    t.name for t in rig_animation.nla_tracks
                    if len(t.strips) > 0
                    and not agent_ops.is_agent_track_name(t.name)
                    and t.name != agent_ops.BASE_TRACK
                ]
            if foreign_tracks:
                raise RuntimeError(
                    f"MMR Rig 存在陌生的 NLA Track {foreign_tracks}；"
                    "自动 Bake 只接受单一活动 Action")
            from .core import agent_bridge
            rig_action = agent_bridge._base_action(rig)
            if rig_action is None:
                raise RuntimeError("MMR Rig 没有可烘焙的 Action（也没有 mcd_base 轨）")
            if not _action_has_range_keys(
                rig_action, settings.mocap_frame_start, settings.mocap_frame_end
            ):
                raise RuntimeError("MMR Rig Action 没有覆盖完整动捕范围")
            animation_data = armature.animation_data
            if animation_data and (
                animation_data.action is not None or len(animation_data.nla_tracks) > 0
            ):
                raise RuntimeError("原生 MMD 骨架已有 Action/NLA；为避免混写，请改用手动 Bake 回退")
            before = _begin_structure_preview(context.scene, "mmd_bake")
            settings.busy = True
            settings.status_message = "正在生成 MMD Visual Bake 预览"
            with _active_armature(context, armature, pose=True):
                bpy.ops.pose.select_all(action="DESELECT")
                selected_names = {bone.name for bone in constrained}
                selected_names.update((MMD_ROOT_BONE, foot_ik["L"], foot_ik["R"]))
                for bone_name in selected_names:
                    armature.pose.bones[bone_name].bone.select = True
                result = bpy.ops.nla.bake(
                    frame_start=settings.mocap_frame_start,
                    frame_end=settings.mocap_frame_end,
                    step=1,
                    only_selected=True,
                    visual_keying=True,
                    clear_constraints=False,
                    clear_parents=False,
                    use_current_action=False,
                    clean_curves=False,
                    bake_types={"POSE"},
                )
            if "FINISHED" not in result:
                raise RuntimeError("MMD Visual Bake 没有成功完成")
            action = armature.animation_data.action if armature.animation_data else None
            if not action:
                raise RuntimeError("MMD Bake 后没有生成 Action")
            _validate_mmd_action(
                action,
                foot_ik,
                settings.mocap_frame_start,
                settings.mocap_frame_end,
                expected_bones=set(selected_names) | set(MMD_ARM_FK_BONES),
            )
            delete_leg_fk, leg_reason = _leg_fk_cleanup_plan(armature, foot_ik)
            removed = remove_bone_fcurves(action, MMD_LEG_FK_BONES) if delete_leg_fk else []
            _validate_mmd_action(
                action,
                foot_ik,
                settings.mocap_frame_start,
                settings.mocap_frame_end,
                expected_bones=(set(selected_names) - set(MMD_LEG_FK_BONES)) | set(MMD_ARM_FK_BONES),
                require_clean_fk=delete_leg_fk,
            )
            muted_count = _mute_mmr_constraints(armature, rig)
            if muted_count == 0:
                raise RuntimeError("Bake 后没有可禁用的 MMR 约束")
            project.record_step(
                context.scene,
                "mmd_bake",
                "PREVIEW",
                f"Bake 完成、禁用 {muted_count} 个 MMR 约束；{leg_reason}；删除 {len(removed)} 条腿 FK 曲线",
                str(before),
            )
            settings.status_message = f"MMD Bake 预览完成；{leg_reason}；已删 {len(removed)} 条腿 FK 曲线"
            return {"FINISHED"}
        except Exception as exc:
            settings.status_message = str(exc)
            if settings.preview_step_id == "mmd_bake":
                project.record_step(context.scene, "mmd_bake", "FAILED", str(exc))
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            settings.busy = False


class MD_OT_ValidateManualMMDBake(Operator):
    bl_idname = "mocap_doctor.validate_manual_mmd_bake"
    bl_label = "检查手工 Bake 并清理 FK"
    bl_description = "验证全ての親和左右足 IK 曲线，并自动删除六根腿 FK 曲线"

    def execute(self, context):
        settings = _settings(context)
        try:
            _require_project(context)
            _root, armature, rig, foot_ik = _validate_mmd_identity(settings)
            source_action = _require_action(armature, "原生 MMD Armature")
            _require_no_nla(armature, "原生 MMD Armature")
            driven_bones = _mmr_driven_bone_names(armature, rig)
            if not driven_bones:
                # MMR's own manual Bake command may remove its constraints
                # after writing the MMD Action.  In that case the Action is
                # the authoritative evidence of which MMD bones were baked.
                driven_bones = _action_bone_names(source_action, armature)
                required_only = {
                    MMD_ROOT_BONE,
                    MMD_CENTER_BONE,
                    foot_ik["L"],
                    foot_ik["R"],
                }
                if not (set(driven_bones) - set(MMD_LEG_FK_BONES) - required_only):
                    raise RuntimeError("固定 Teto 骨架上没有 MMR 驱动骨，也没有可验证的 MMD Bake 骨骼曲线")
            _validate_mmd_action(
                source_action,
                foot_ik,
                settings.mocap_frame_start,
                settings.mocap_frame_end,
                expected_bones=(set(driven_bones) - set(MMD_LEG_FK_BONES)) | set(MMD_ARM_FK_BONES),
            )
            _ensure_no_pending_preview(settings, "mmd_bake")
            _begin_structure_preview(context.scene, "mmd_bake")
            action = source_action
            delete_leg_fk, leg_reason = _leg_fk_cleanup_plan(armature, foot_ik)
            removed = remove_bone_fcurves(action, MMD_LEG_FK_BONES) if delete_leg_fk else []
            muted_count = _mute_mmr_constraints(armature, rig)
            _validate_mmd_action(
                action,
                foot_ik,
                settings.mocap_frame_start,
                settings.mocap_frame_end,
                expected_bones=(set(driven_bones) - set(MMD_LEG_FK_BONES)) | set(MMD_ARM_FK_BONES),
                require_clean_fk=delete_leg_fk,
            )
            report = {
                "operation": "validate_manual_mmd_bake",
                "action": action.name,
                "validated_bones": [MMD_ROOT_BONE, foot_ik["L"], foot_ik["R"], *MMD_ARM_FK_BONES],
                "removed_leg_fk_fcurves": len(removed),
                "leg_drive": leg_reason,
                "muted_mmr_constraints": muted_count,
                "constraint_mapping_present": bool(_mmr_driven_bone_names(armature, rig)),
            }
            _record_report(
                context.scene,
                "mmd_bake",
                report,
                "PREVIEW",
                f"手工 Bake 已验证（{leg_reason}）；删除 {len(removed)} 条腿 FK 曲线，"
                "MMR→MMD 约束已禁用；请隐藏 MMR Rig 检查",
            )
            settings.status_message = "手工 MMD Bake 验证通过；MMR→MMD 约束已禁用，等待检查"
            return {"FINISHED"}
        except Exception as exc:
            if settings.preview_step_id == "mmd_bake":
                project.record_step(context.scene, "mmd_bake", "FAILED", str(exc))
            settings.status_message = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MD_OT_CleanupLegFK(Operator):
    bl_idname = "mocap_doctor.cleanup_leg_fk"
    bl_label = "一键删除六个腿 FK 曲线"

    def execute(self, context):
        settings = _settings(context)
        try:
            _require_project(context)
            _root, armature, rig, foot_ik = _validate_mmd_identity(settings)
            source_action = _require_action(armature, "原生 MMD Armature")
            _require_no_nla(armature, "原生 MMD Armature")
            driven_bones = _mmr_driven_bone_names(armature, rig)
            _validate_mmd_action(
                source_action,
                foot_ik,
                settings.mocap_frame_start,
                settings.mocap_frame_end,
                expected_bones=set(driven_bones) - set(MMD_LEG_FK_BONES),
            )
            _ensure_no_pending_preview(settings, "mmd_bake")
            _begin_structure_preview(context.scene, "mmd_bake")
            action = source_action
            delete_leg_fk, leg_reason = _leg_fk_cleanup_plan(armature, foot_ik)
            if not delete_leg_fk:
                raise RuntimeError("不能删除腿 FK 曲线：" + leg_reason + "；删除会让腿失去动画")
            removed = remove_bone_fcurves(action, MMD_LEG_FK_BONES)
            _mute_mmr_constraints(armature, rig)
            _validate_mmd_action(
                action,
                foot_ik,
                settings.mocap_frame_start,
                settings.mocap_frame_end,
                expected_bones=set(driven_bones) - set(MMD_LEG_FK_BONES),
                require_clean_fk=True,
            )
            settings.status_message = f"预览中：删除了 {len(removed)} 条腿 FK 曲线"
            self.report({"INFO"}, settings.status_message)
            return {"FINISHED"}
        except Exception as exc:
            if settings.preview_step_id == "mmd_bake":
                project.record_step(context.scene, "mmd_bake", "FAILED", str(exc))
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MD_OT_AcceptPreview(Operator):
    bl_idname = "mocap_doctor.accept_preview"
    bl_label = "接受并进入下一步"

    def execute(self, context):
        settings = _settings(context)
        step_id = settings.preview_step_id
        if not step_id:
            self.report({"ERROR"}, "当前没有等待接受的预览")
            return {"CANCELLED"}
        try:
            old_step = settings.current_step
            record = project.find_step_record(settings, step_id, create=False)
            if record is not None and record.status == "FAILED":
                raise RuntimeError("此预览执行失败，只能丢弃并恢复检查点")
            if step_id == "mmd_bake":
                _root, armature, rig, _foot_ik = _validate_mmd_identity(settings)
                if _active_mmr_constraints(armature, rig):
                    raise RuntimeError("MMR→MMD 约束仍在生效，不能接受 Bake；请重新检查手工 Bake")
            next_step = clamp_step(STEP_INDEX.get(step_id, old_step) + 1)
            # 同一页有多个操作时不能跳步：否则接受第一个操作就被带离本页，
            # 还得退回来重跑第二个。
            if step_id in MULTI_OP_STEPS:
                settings.current_step = old_step
                settings.status_message = "步骤已接受；本页还有其它操作可继续"
            else:
                settings.current_step = next_step
                settings.status_message = "步骤已接受"
            settings.status_level = 'INFO'
            if settings.preview_action:
                project.accept_action_preview(context.scene)
            else:
                preview_fields = {
                    name: getattr(settings, name)
                    for name in (
                        "preview_step_id",
                        "preview_owner_name",
                        "preview_base_action",
                        "preview_action",
                        "preview_restore_checkpoint",
                    )
                }
                settings.preview_step_id = ""
                settings.preview_restore_checkpoint = ""
                settings.preview_owner_name = ""
                settings.preview_base_action = ""
                settings.preview_action = ""
                try:
                    project.create_accepted_checkpoint(context.scene, step_id, "预览已接受")
                except Exception:
                    for name, value in preview_fields.items():
                        setattr(settings, name, value)
                    raise
                project.save_workfile(context.scene)
            return {"FINISHED"}
        except Exception as exc:
            settings.current_step = old_step
            settings.status_level = 'ERROR'
            settings.status_message = '接受失败：' + _short_reason(str(exc), limit=48)
            self.report({'ERROR'}, settings.status_message)
            return {'CANCELLED'}


class MD_OT_DiscardPreview(Operator):
    bl_idname = "mocap_doctor.discard_preview"
    bl_label = "丢弃预览"

    def execute(self, context):
        settings = _settings(context)
        step_id = settings.preview_step_id
        if not step_id:
            self.report({"ERROR"}, "当前没有预览")
            return {"CANCELLED"}
        checkpoint = settings.preview_restore_checkpoint
        if settings.preview_action:
            restored = project.rollback_action_preview(context.scene)
            if restored:
                project.record_step(context.scene, step_id, "PENDING", "预览已丢弃")
                settings.status_message = "已恢复执行前动作"
                settings.status_level = 'INFO'
                project.save_workfile(context.scene)
                self.report({'INFO'}, settings.status_message)
                return {"FINISHED"}
            if checkpoint:
                try:
                    project.restore_checkpoint(checkpoint, project.resolve_work_filepath(settings))
                    return {"FINISHED"}
                except Exception as exc:
                    self.report({"ERROR"}, str(exc))
                    return {"CANCELLED"}
        if not checkpoint:
            self.report({"ERROR"}, "找不到结构步骤的恢复检查点")
            return {"CANCELLED"}
        try:
            project.restore_checkpoint(
                checkpoint, project.workfile_save_target(settings),
                resume_step_id=step_id, reset_step_id=step_id,
                message='已恢复执行前状态',
            )
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MD_OT_ExportVMD(Operator):
    bl_idname = "mocap_doctor.export_vmd"
    bl_label = "导出 VMD"

    def execute(self, context):
        settings = _settings(context)
        try:
            _require_project(context)
            record = project.find_step_record(settings, "export_prep", create=False)
            if record is None or record.status != "ACCEPTED":
                raise RuntimeError("请先接受 VMD 导出准备预览")
            root, armature, rig, foot_ik = _validate_mmd_identity(settings)
            if _active_mmr_constraints(armature, rig):
                raise RuntimeError("仍有活动的 MMR 到 MMD 约束，不能导出")
            action = _require_action(armature, "原生 MMD Armature")
            _require_no_nla(armature, "原生 MMD Armature")
            delete_leg_fk, _leg_reason = _leg_fk_cleanup_plan(armature, foot_ik)
            _validate_mmd_action(
                action,
                foot_ik,
                settings.mocap_frame_start,
                settings.mocap_frame_end,
                require_clean_fk=delete_leg_fk,
            )
            _require_no_keyed_vmd_name_collisions(armature, action)
            raw_path = str(settings.vmd_export_path or "").strip()
            if not raw_path:
                raise RuntimeError("请先指定 VMD 文件保存位置")
            filepath = Path(bpy.path.abspath(raw_path))
            if filepath.is_dir():
                # The field takes a file name, but picking a folder is the
                # natural mistake; name the file after the work file instead
                # of failing or writing "<folder>.vmd" next to it.
                stem = Path(bpy.data.filepath).stem if bpy.data.filepath else "mocap"
                filepath = filepath / f"{stem}.vmd"
            elif not filepath.name:
                raise RuntimeError(f"请在保存位置里写上文件名（例如 {raw_path}我的动作.vmd）")
            elif filepath.suffix.lower() != ".vmd":
                filepath = filepath.with_suffix(".vmd")
            if not filepath.parent.is_dir():
                raise RuntimeError(f"VMD 保存目录不存在：{filepath.parent}")
            previous_file = None
            if filepath.is_file():
                stat = filepath.stat()
                previous_file = (stat.st_mtime_ns, stat.st_size)

            # mmd_tools reads context.active_object inside execute().  Its file
            # browser can lose that object and then dereference None. Execute
            # directly with the validated MMD Root active instead.
            with _active_armature(context, root, pose=False):
                if not bpy.ops.mmd_tools.export_vmd.poll():
                    raise RuntimeError("mmd_tools 未启用，或 Teto 根对象不能导出 VMD")
                result = bpy.ops.mmd_tools.export_vmd(
                    "EXEC_DEFAULT",
                    filepath=str(filepath),
                    use_frame_range=True,
                )
            current_file = None
            if filepath.is_file():
                stat = filepath.stat()
                current_file = (stat.st_mtime_ns, stat.st_size)
            if "FINISHED" not in result or current_file is None or current_file == previous_file:
                raise RuntimeError("mmd_tools 未生成 VMD 文件；请查看系统控制台中的导出错误")
            settings.vmd_export_path = str(filepath)
            settings.status_message = f"VMD 已导出：{filepath}"
            project.save_workfile(context.scene)
            self.report({"INFO"}, settings.status_message)
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MD_OT_PrepareReceiverTemplate(Operator):
    bl_idname = "mocap_doctor.prepare_receiver_template"
    bl_label = "准备 VMD 接收模板"
    bl_description = (
        "删除 MMR 手臂控制绑定（控制骨架、手 IK 辅助骨骼与驱动约束），恢复ひじ父级为腕捩，"
        "关闭手臂 IK，并清理旧的 VMD 动作；之后导入 VMD 时手腕与手肘才能按导出姿态正确旋转"
    )
    bl_options = {"REGISTER", "UNDO"}

    @staticmethod
    def _find_mmd_root(context):
        obj = context.view_layer.objects.active
        while obj is not None:
            if getattr(obj, "mmd_type", "") == "ROOT":
                return obj
            obj = obj.parent
        for obj in bpy.data.objects:
            if getattr(obj, "mmd_type", "") == "ROOT":
                return obj
        return None

    @staticmethod
    def _find_mmd_armature(root):
        for child in root.children:
            if child.type == "ARMATURE" and not child.name.startswith("RIG-"):
                return child
        return None

    def execute(self, context):
        root = self._find_mmd_root(context)
        if root is None:
            self.report({"ERROR"}, "未找到 MMD 根对象（mmd_type == ROOT）；请打开含 Teto 模型的接收模板")
            return {"CANCELLED"}
        armature = self._find_mmd_armature(root)
        if armature is None:
            self.report({"ERROR"}, "MMD 根对象下没有原生 Armature，无法准备模板")
            return {"CANCELLED"}

        # ---- build plan data from the live scene ----
        bones = {}
        for pb in armature.pose.bones:
            mmd_bone = getattr(pb, "mmd_bone", None)
            name_j = str(getattr(mmd_bone, "name_j", "") or "") if mmd_bone else ""
            bones[pb.name] = {
                "name": pb.name,
                "name_j": name_j,
                "parent": pb.parent.name if pb.parent else None,
                "has_ik_toggle": hasattr(pb, "mmd_ik_toggle"),
                "ik_toggle": bool(getattr(pb, "mmd_ik_toggle", False)),
                "has_children": any(
                    bone.parent is not None and bone.parent.name == pb.name
                    for bone in armature.data.bones
                ),
                "constraints": [
                    {
                        "name": constraint.name,
                        "type": constraint.type,
                        "subtarget": getattr(constraint, "subtarget", ""),
                    }
                    for constraint in pb.constraints
                ],
            }
        constraint_plan = core_receiver.plan_constraints(bones.values())
        toggle_off = core_receiver.plan_arm_ik_toggle_off(bones)
        helper_plan = core_receiver.plan_helper_bone_deletion(bones.values())
        reparent = core_receiver.plan_elbow_reparent(bones)
        rig_plan = core_receiver.plan_rig_deletion(
            [(obj.name, obj.type) for obj in bpy.data.objects if obj is not armature],
            [text.name for text in bpy.data.texts],
        )
        stale_actions = core_receiver.plan_stale_actions(
            [
                {
                    "name": action.name,
                    "has_pose_bone_fcurves": any(
                        fc.data_path.startswith('pose.bones["') for fc in action.fcurves
                    ),
                }
                for action in bpy.data.actions
            ]
        )

        # ---- apply ----
        remove_keys = {
            (item["bone"], item["name"]) for item in constraint_plan["remove"]
        }
        for pb in armature.pose.bones:
            if pb.name in toggle_off and hasattr(pb, "mmd_ik_toggle"):
                pb.mmd_ik_toggle = False
            for constraint in list(pb.constraints):
                if (pb.name, constraint.name) in remove_keys:
                    pb.constraints.remove(constraint)

        for name in rig_plan["rig_objects"]:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)
        for name in rig_plan["rig_texts"]:
            text = bpy.data.texts.get(name)
            if text is not None:
                bpy.data.texts.remove(text)
        for name in stale_actions:
            action = bpy.data.actions.get(name)
            if action is not None:
                bpy.data.actions.remove(action)
        if armature.animation_data:
            armature.animation_data.action = None

        # mmd_tools' VMD importer starts the motion at the CURRENT frame
        # (core/vmd/importer.py: frame = vmd_frame + scene.frame_current), so a
        # template parked at frame 145 silently pushes the whole take there and
        # holds a frozen pose over everything before it.  Park it at the start.
        scene = context.scene
        rewound = int(scene.frame_current) != 1 or int(scene.frame_start) != 1
        scene.frame_start = 1
        scene.frame_current = 1

        with _active_armature(context, armature, pose=False):
            bpy.ops.object.mode_set(mode="EDIT")
            edit_bones = armature.data.edit_bones
            for bone_name, new_parent in reparent:
                bone = edit_bones.get(bone_name)
                parent = edit_bones.get(new_parent)
                if bone is not None and parent is not None:
                    bone.parent = parent
            for name in helper_plan["delete"]:
                bone = edit_bones.get(name)
                if bone is not None:
                    edit_bones.remove(bone)
            bpy.ops.object.mode_set(mode="OBJECT")

        report = (
            f"模板已准备：删除约束 {len(constraint_plan['remove'])} 个，"
            f"保留原生 IK {len(constraint_plan['keep'])} 个；"
            f"删除辅助骨骼 {helper_plan['delete']}；"
            f"恢复父级 {reparent}；"
            f"关闭手臂 IK {toggle_off}；"
            f"删除控制骨架 {rig_plan['rig_objects']}；"
            f"清理旧动作 {stale_actions}。请保存模板后再导入 VMD。"
        )
        if rewound:
            report += " 当前帧已归到 1（mmd_tools 从当前帧起放置导入的动作）。"
        if helper_plan["skipped"]:
            report += f" 注意：有子级的辅助骨骼已跳过 {helper_plan['skipped']}。"
        self.report({"INFO"}, report)
        return {"FINISHED"}


class MD_OT_AgentServerToggle(Operator):
    """启动/停止 Agent 工具服务器（本地 socket + 主线程队列执行）"""
    bl_idname = "mocap_doctor.agent_server_toggle"
    bl_label = "Agent 服务开关"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        from .core import agent_bridge
        if agent_bridge.is_running():
            agent_bridge.stop_server()
            self.report({"INFO"}, "Agent 工具服务器已停止")
        else:
            info = agent_bridge.start_server()
            self.report({"INFO"},
                        f"Agent 工具服务器已启动 127.0.0.1:{info['port']}")
        return {"FINISHED"}


class MD_OT_AgentPreviewToggle(Operator):
    """A/B：静音/取消静音所有 agent 轨（每条修复各一轨），对比前后"""
    bl_idname = "mocap_doctor.agent_ab_toggle"
    bl_label = "A/B 预览对比"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        from .core import agent_bridge, agent_ops
        settings = context.scene.mocap_doctor
        armature = agent_bridge._rig_armature(settings, context.scene)
        if armature is None:
            self.report({"ERROR"}, "没有识别到 RIG 骨架")
            return {"CANCELLED"}
        anim = getattr(armature, "animation_data", None)
        agent_tracks = [t for t in (anim.nla_tracks if anim else ())
                        if agent_ops.is_agent_track_name(t.name)]
        if not agent_tracks:
            self.report({"INFO"}, "还没有 agent 轨（尚无预览/提交 op）")
            return {"CANCELLED"}
        # 有任何一个还响着 → 全部静音（看修复前）；全静音 → 全部放响（看修复后）
        new_state = any(not t.mute for t in agent_tracks)
        for t in agent_tracks:
            t.mute = new_state
        for item in settings.agent_fixes:
            item.muted = new_state
        agent_bridge._redraw()
        self.report({"INFO"},
                    "agent 修复已隐藏" if new_state else "agent 修复可见")
        return {"FINISHED"}


class _AgentFixOp(Operator):
    """Base: resolve the row the button belongs to."""

    bl_options = {"INTERNAL"}

    op_id: StringProperty(default="")
    track: StringProperty(default="")
    strip: StringProperty(default="")
    index: IntProperty(default=-1)

    def _item(self, context):
        settings = context.scene.mocap_doctor
        for item in settings.agent_fixes:
            if self.op_id and item.op_id == self.op_id:
                return settings, item
        if self.track:
            for item in settings.agent_fixes:
                if item.track == self.track and (
                        not self.strip or item.strip == self.strip):
                    return settings, item
        return settings, None


class MD_OT_AgentFixMute(_AgentFixOp):
    """单独静音/取消这条修复"""

    bl_idname = "mocap_doctor.agent_fix_mute"
    bl_label = "单条静音"

    def execute(self, context):
        from .core import agent_bridge
        settings, item = self._item(context)
        if item is None:
            self.report({"ERROR"}, "找不到这条修复")
            return {"CANCELLED"}
        item.muted = not item.muted      # update 回调负责写轨
        agent_bridge.sync_fixes_list(settings, context.scene)
        return {"FINISHED"}


def _fix_targets(settings):
    """批量目标：勾选了任何行就用勾选集，否则退回当前选中行。"""
    sel = [i for i in settings.agent_fixes if i.selected]
    if sel:
        return sel
    idx = settings.agent_fix_index
    if 0 <= idx < len(settings.agent_fixes):
        return [settings.agent_fixes[idx]]
    return []


def _sync_batch_display(settings):
    """选择操作后把批量滑块/眼睛的显示值对齐当前行（静默，不回写）。"""
    from . import properties as _props
    idx = settings.agent_fix_index
    if 0 <= idx < len(settings.agent_fixes):
        it = settings.agent_fixes[idx]
        _props.set_quietly(settings, "agent_batch_exponent", it.exponent)
        _props.set_quietly(settings, "agent_batch_muted", it.muted)


class MD_OT_AgentFixSelAll(Operator):
    """全选 / 清空勾选"""

    bl_idname = "mocap_doctor.agent_fix_sel_all"
    bl_label = "全选修复"
    bl_options = {"INTERNAL"}

    select: BoolProperty(default=True)

    def execute(self, context):
        settings = context.scene.mocap_doctor
        for item in settings.agent_fixes:
            item.selected = bool(self.select)
        if self.select:
            _sync_batch_display(settings)
        return {"FINISHED"}


class MD_OT_AgentFixSelWindow(Operator):
    """选中与当前行同一窗口（同帧段）的所有修复——一次点中一个发力窗"""

    bl_idname = "mocap_doctor.agent_fix_sel_window"
    bl_label = "选中同窗"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        settings = context.scene.mocap_doctor
        idx = settings.agent_fix_index
        if not (0 <= idx < len(settings.agent_fixes)):
            self.report({"ERROR"}, "先点一行")
            return {"CANCELLED"}
        frames = settings.agent_fixes[idx].frames
        n = 0
        for item in settings.agent_fixes:
            item.selected = (item.frames == frames and frames != "")
            n += int(item.selected)
        _sync_batch_display(settings)
        self.report({"INFO"}, f"选中 {n} 条")
        return {"FINISHED"}


class MD_OT_AgentFixCommit(_AgentFixOp):
    """提交修复（勾选了多条就批量；不搬轨，仍可调力度/静音）"""

    bl_idname = "mocap_doctor.agent_fix_commit"
    bl_label = "提交修复"

    def execute(self, context):
        from .core import agent_bridge, agent_ops
        settings = context.scene.mocap_doctor
        targets = [i for i in _fix_targets(settings) if i.op_id]
        if not targets:
            self.report({"ERROR"}, "没有可提交的记录")
            return {"CANCELLED"}
        for item in targets:
            agent_ops.commit(agent_bridge._data_dir(settings), item.op_id)
        agent_bridge._bump_ops_rev_from(settings)
        agent_bridge.sync_fixes_list(settings, context.scene)
        agent_bridge._redraw()
        self.report({"INFO"}, f"已提交 {len(targets)} 条（仍可继续调力度）")
        return {"FINISHED"}


class MD_OT_AgentFixRevert(_AgentFixOp):
    """撤销修复：删 strip + action + 空轨（勾选了多条就批量）"""

    bl_idname = "mocap_doctor.agent_fix_revert"
    bl_label = "撤销修复"

    def execute(self, context):
        from .core import agent_bridge, agent_ops
        settings = context.scene.mocap_doctor
        targets = _fix_targets(settings)
        if not targets:
            self.report({"ERROR"}, "没有可撤销的修复")
            return {"CANCELLED"}
        rig = agent_bridge._rig_armature(settings, context.scene)
        n = 0
        for item in targets:
            if item.op_id:
                agent_ops.revert(rig, agent_bridge._data_dir(settings),
                                 item.op_id)
            elif item.strip and rig is not None:   # 孤儿行：直接删 strip
                agent_ops.delete_op_strip(rig, {"strip": item.strip,
                                                "track": item.track})
            else:
                continue
            n += 1
        agent_bridge._bump_ops_rev_from(settings)
        agent_bridge.sync_fixes_list(settings, context.scene)
        agent_bridge._redraw()
        self.report({"INFO"}, f"已撤销 {n} 条")
        return {"FINISHED"}


class MD_OT_AgentFixForget(_AgentFixOp):
    """清掉记录：丢失的删 log 条目，孤儿 strip 直接删（勾选批量）"""

    bl_idname = "mocap_doctor.agent_fix_forget"
    bl_label = "清除记录"

    def execute(self, context):
        from .core import agent_bridge, agent_ops
        settings = context.scene.mocap_doctor
        targets = _fix_targets(settings)
        if not targets:
            self.report({"ERROR"}, "没有可清除的记录")
            return {"CANCELLED"}
        rig = agent_bridge._rig_armature(settings, context.scene)
        data_dir = agent_bridge._data_dir(settings)
        drop_ids = set()
        n = 0
        for item in targets:
            if item.op_id:
                op = None
                for cand in agent_ops.list_ops(data_dir):
                    if cand["id"] == item.op_id:
                        op = cand
                        break
                if op is not None:
                    if rig is not None:
                        agent_ops.delete_op_strip(rig, op)
                    drop_ids.add(item.op_id)
                    n += 1
            elif rig is not None and item.strip:
                agent_ops.delete_op_strip(rig, {"strip": item.strip,
                                                "track": item.track})
                n += 1
        if drop_ids:
            ops = [o for o in agent_ops.list_ops(data_dir)
                   if o["id"] not in drop_ids]
            agent_ops._save_oplog(data_dir, ops)
        agent_bridge._bump_ops_rev_from(settings)
        agent_bridge.sync_fixes_list(settings, context.scene)
        agent_bridge._redraw()
        self.report({"INFO"}, f"已清除 {n} 条")
        return {"FINISHED"}


class MD_OT_AgentFixRefresh(Operator):
    """重新对账：迁移旧轨 + op log ↔ 场景 NLA（定时器失灵时的手动兜底）"""

    bl_idname = "mocap_doctor.agent_fix_refresh"
    bl_label = "刷新修复列表"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        from .core import agent_bridge
        settings = context.scene.mocap_doctor
        res = agent_bridge.ensure_layout(settings, context.scene)
        agent_bridge._redraw()
        self.report({"INFO"},
                    f"{res.get('rows', 0)} 条修复"
                    + (f"，迁移 {res.get('migrated')} 条" if res.get("migrated") else ""))
        return {"FINISHED"}


class MD_OT_AgentParamChoice(Operator):
    """choice 型参数按钮：写值 → update 回调排队防抖重写"""

    bl_idname = "mocap_doctor.agent_param_choice"
    bl_label = "选参数值"
    bl_options = {"INTERNAL"}

    pkey: StringProperty(default="")
    value: StringProperty(default="")

    def execute(self, context):
        settings = context.scene.mocap_doctor
        item = next((p for p in settings.agent_params
                     if p.pkey == self.pkey), None)
        if item is None:
            self.report({"ERROR"}, "参数条目不在了（刷新后再试）")
            return {"CANCELLED"}
        item.sval = self.value          # update 回调 → schedule_param_apply
        return {"FINISHED"}


class MD_OT_AgentDirCreate(Operator):
    """新建方向空物体：arrow=单箭头(平行于箭头轴 +Z)，aim=指向点(骨指向它)。
    可 k 动画做随帧变化的方向目标。assign 指定参数条目时自动绑定。"""

    bl_idname = "mocap_doctor.agent_dir_create"
    bl_label = "新建方向物体"
    bl_options = {"INTERNAL"}

    kind: StringProperty(default="arrow")   # arrow / aim
    assign: StringProperty(default="")      # agent_params 条目的 pkey

    def execute(self, context):
        scene = context.scene
        base = "mcd_aim" if self.kind == "aim" else "mcd_dir"
        obj = bpy.data.objects.new(base, None)
        obj.empty_display_type = "SPHERE" if self.kind == "aim" else "SINGLE_ARROW"
        obj.empty_display_size = 0.05 if self.kind == "aim" else 0.15
        obj.location = scene.cursor.location.copy()
        scene.collection.objects.link(obj)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj
        if self.assign:
            settings = scene.mocap_doctor
            item = next((p for p in settings.agent_params
                         if p.pkey == self.assign), None)
            if item is not None:
                item.obj = obj          # update 回调 → 排队重写
        self.report({"INFO"},
                    f"已建 {obj.name}（"
                    + ("摆到指向位置" if self.kind == "aim"
                       else "旋转它，箭头轴即方向") + "）")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# 导入步骤的内部实现（core/official_import.py）
#
# 第 0 步「导入」= 一个按钮：take 目录 → 52 骨 Reference + 贴地数据 + 参考视频平面。
# 三份 body npz 选哪份、接触分数走哪一档、绝对关节怎么补、官方第 5 步怎么接，全是
# 内部自动决定；界面上没有对应的开关/按钮/术语（2026-10-16 用户要求做减法）。


def _official_source_path(settings) -> str:
    """扫描入口：优先 take 目录，其次 fbx / npz 路径（取父目录），最后空。"""

    return str(
        settings.official_take_dir or settings.official_fbx or settings.official_npz or ""
    ).strip()


def _official_scan(settings) -> dict:
    """跑一次 scan_take_dir：带上面板选择器，并尊重手填的非标准 npz 路径。"""

    source = _official_source_path(settings)
    choice = str(getattr(settings, "official_npz_choice", "") or "AUTO")
    info = core_official_import.scan_take_dir(
        source,
        contacts_explicit=str(settings.official_contacts_npz or "").strip(),
        npz_choice=choice,
    )
    manual = str(settings.official_npz or "").strip()
    if (
        manual
        and os.path.isfile(manual)
        and os.path.basename(manual) not in core_official_import.BODY_NPZ_NAMES
    ):
        # 用户手填了别的 npz：不要被自动解析覆盖（接触分数仍按三档规则走）
        info["npz"] = manual
        info["npz_name"] = os.path.basename(manual)
        info["npz_frames"] = core_official_import.npz_frame_count(manual)
        info["manual_npz"] = True
    return info


def _official_apply_scan(settings, info: dict) -> None:
    """把 scan 结果写进 settings（**内部状态，不产生任何界面文字**）。

    三份 npz 选哪份、一致性校验、视频提示等都留在 info 里给内部逻辑用；
    用户界面上只有「已导入 / 未找到动作数据」。
    """

    settings.official_fbx = str(info.get('fbx') or '')
    settings.official_npz = str(info.get('npz') or '')
    settings.official_video = str(info.get('video') or '')

    contacts = info.get("contacts") or {}
    explicit = str(settings.official_contacts_npz or "").strip()
    resolved = str(contacts.get("path") or "")
    settings.official_contacts_resolved = resolved if not explicit else ""


def _official_resolved_contacts(settings) -> tuple[str, str]:
    """(接触分数 npz 路径, 状态文字)。手填优先，其次自动解析结果。"""

    explicit = str(settings.official_contacts_npz or "").strip()
    if explicit:
        if os.path.isfile(explicit) and core_official_import.npz_has_foot_contacts(explicit):
            return explicit, f"接触分数：手动指定（{explicit}）"
        if os.path.isfile(explicit):
            return "", (f"接触分数：手动指定的 npz 没有 foot_contacts（{explicit}）→ "
                        "无接触分数，planted 由几何推断")
        return "", (f"接触分数：手动指定的文件不存在（{explicit}）→ "
                    "无接触分数，planted 由几何推断")
    resolved = str(settings.official_contacts_resolved or "").strip()
    if resolved and os.path.isfile(resolved):
        source = (core_official_import.find_contacts(
            core_official_import.resolve_take_dir(_official_source_path(settings)),
            choice=str(getattr(settings, "official_npz_choice", "") or "AUTO"),
        ) or {})
        what = source.get("status") or f"接触分数：{resolved}"
        return resolved, what
    return "", "接触分数：无接触分数，planted 由几何推断"


def _official_video_plane(context, settings, video_path: str) -> dict:
    """建/更新参考视频平面。

    变换顺序**按用户规格**：RZ(rot_z) → +Y(y) → ×scale → **再用缩放后的包围盒把底边
    对齐到 Z=0**。返回证据字典（含缩放后未对齐 / 对齐之后的 min_z、世界法向、尺寸）。
    """

    import math
    import mathutils

    scene = context.scene
    name = core_official_import.PLANE_OBJECT_NAME
    take_dir = core_official_import.resolve_take_dir(
        settings.official_take_dir or settings.official_npz or settings.official_fbx
    )
    info = core_official_import.read_video_info(take_dir)
    width_px = float(info.get("width") or 0.0)
    height_px = float(info.get("height") or 0.0)
    video_frames = int(info.get("frames") or 0)
    video_fps = float(info.get("fps") or 0.0)
    image = None
    image_note = ""
    if video_path and os.path.isfile(video_path):
        try:
            image = bpy.data.images.load(video_path, check_existing=True)
            try:
                image.source = "MOVIE"
            except Exception:  # noqa: BLE001 - 已经是 MOVIE 或该 build 不认
                pass
            size = tuple(int(v) for v in image.size)
            if size[0] > 0 and size[1] > 0:
                width_px, height_px = float(size[0]), float(size[1])
            frames = int(getattr(image, "frame_duration", 0) or 0)
            if frames > 0:
                video_frames = frames
            fps = float(getattr(image, "fps", 0.0) or 0.0)
            if fps > 0:
                video_fps = fps
        except Exception as exc:  # noqa: BLE001 - 视频坏了不该让导入失败
            image_note = f"视频加载失败（{exc}）"
    elif video_path:
        image_note = "视频文件不存在"
    else:
        image_note = "没有视频路径（跳过平面；可手动指定参考视频后重建）"

    aspect = (width_px / height_px) if (width_px > 0 and height_px > 0) else (16.0 / 9.0)
    base_w, base_h = core_official_import.plane_size_for_aspect(
        aspect, float(settings.video_plane_height)
    )
    scale = float(settings.video_plane_scale)
    rot_z = float(settings.video_plane_rot_z)
    y_pos = float(settings.video_plane_y)
    expected = core_official_import.plane_transform(
        base_w, base_h, y=y_pos, rot_z_deg=rot_z, scale=scale
    )

    # --- 网格：局部 YZ 平面、中心在原点、法向 +X（RZ −90° 后 = 世界 −Y）----
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type != "MESH":
        obj = bpy.data.objects.get(name + "_video")   # 名字被别的类型占了，另起一个
    if obj is None:
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
    else:
        mesh = obj.data
        mesh.clear_geometry()
    mesh.from_pydata([tuple(v) for v in expected["local"]], [], [(0, 1, 2, 3)])
    mesh.update()
    mesh.validate()
    uv_layer = mesh.uv_layers[0] if len(mesh.uv_layers) else mesh.uv_layers.new(name="UVMap")
    for index, uv in enumerate(core_official_import.PLANE_UVS):
        uv_layer.data[index].uv = uv

    # 材质（Emission + Image Texture；视频好不好看都照建）
    material = bpy.data.materials.get(core_official_import.PLANE_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(core_official_import.PLANE_MATERIAL_NAME)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    out_node = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.extension = "EXTEND"
    emit.inputs["Strength"].default_value = 1.0
    links.new(tex.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out_node.inputs["Surface"])
    if len(obj.data.materials):
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
    obj.display_type = "TEXTURED"

    # --- 变换：先 旋转/平移/缩放，再对齐地面（顺序按用户规格，不能换）-----
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, math.radians(rot_z))
    obj.scale = (scale, scale, scale)
    obj.location = (0.0, y_pos, 0.0)
    context.view_layer.update()
    world_before = [obj.matrix_world @ mathutils.Vector(v) for v in expected["local"]]
    min_z_before = min(v[2] for v in world_before)
    obj.location.z = -min_z_before              # 缩放之后才对齐地面
    context.view_layer.update()
    world_after = [obj.matrix_world @ mathutils.Vector(v) for v in expected["local"]]
    min_z_after = min(v[2] for v in world_after)

    normal_world = obj.matrix_world.to_3x3() @ mathutils.Vector(
        core_official_import.PLANE_LOCAL_NORMAL
    )
    normal_world.normalize()
    xs = [v[0] for v in world_after]
    zs = [v[2] for v in world_after]
    ys = [v[1] for v in world_after]

    frame_start = 1 - int(settings.video_plane_frame_offset)
    if image is not None:
        user = tex.image_user
        user.frame_start = frame_start
        user.frame_duration = max(int(video_frames), 1)
        user.use_auto_refresh = True

    return {
        "object": obj.name,
        "location": [round(v, 6) for v in obj.location],
        "rotation_z_deg": rot_z,
        "scale": scale,
        "base_size_m": [round(base_w, 4), round(base_h, 4)],
        "world_width": round(max(xs) - min(xs), 6),
        "world_height": round(max(zs) - min(zs), 6),
        "expected_world_width": round(expected["expected_width"], 6),
        "expected_world_height": round(expected["expected_height"], 6),
        "aspect": round(aspect, 6),
        "y": y_pos,
        "min_z_before_align": round(min_z_before, 6),
        "min_z_after_align": round(min_z_after, 6),
        "align_dz": round(-min_z_before, 6),
        "center_x": round((max(xs) + min(xs)) * 0.5, 6),
        "normal": [round(v, 6) for v in normal_world],
        "normal_dot_minus_y": round(float(normal_world.dot(mathutils.Vector((0.0, -1.0, 0.0)))), 6),
        "image": image.name if image is not None else "",
        "image_px": [int(width_px), int(height_px)],
        "image_frames": int(video_frames),
        "image_fps": round(video_fps, 3),
        "image_user_frame_start": frame_start,
        "note": image_note,
        "video": video_path,
    }


def _official_import_fbx(context, fbx_path: str) -> dict:
    """导入 FBX；返回新对象清单 + 骨架 + 重名提示（绝不改用户已有对象）。"""

    before = {obj.name for obj in bpy.data.objects}
    existing_reference = "Reference" in before
    bpy.ops.import_scene.fbx(filepath=fbx_path, automatic_bone_orientation=True)
    new_objects = [obj for obj in bpy.data.objects if obj.name not in before]
    armatures = [obj for obj in new_objects if obj.type == "ARMATURE"]
    armature = None
    if armatures:
        armature = max(armatures, key=lambda item: len(item.pose.bones))
    notes = []
    if armature is not None and armature.name != "Reference":
        if "Reference" not in {obj.name for obj in bpy.data.objects if obj not in new_objects}:
            armature.name = "Reference"
            armature.data.name = "Reference"
        else:
            notes.append(
                f"场景里已有 Reference，导入的骨架保留自动名 {armature.name}（没动已有对象）"
            )
    elif armature is not None and existing_reference:
        notes.append(
            f"场景里本来就有 Reference，导入得到 {armature.name}（没动已有对象）"
        )
    return {
        "new_objects": [obj.name for obj in new_objects],
        "armature": armature,
        "armature_name": armature.name if armature is not None else "",
        "bones": len(armature.pose.bones) if armature is not None else 0,
        "notes": notes,
    }


def _official_dump_joints_now(context, settings, frame_limit: int = 0) -> bool:
    """内部用（不面向用户）：把导入骨架的绝对关节写进工作目录。

    官方第 5 步的速度项按关节名取 L_Ankle/L_Foot/R_Ankle/R_Foot；骨名对不上就
    不设来源，让链路自己退回几何检测。返回是否设成了来源。
    """

    import numpy as np

    armature = settings.official_reference or settings.source_armature
    if armature is None or len(armature.pose.bones) < 20:
        raise RuntimeError("没有可导出的骨架")
    action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        raise RuntimeError("骨架没有动画")
    take_dir = core_official_import.resolve_take_dir(
        settings.official_take_dir or settings.official_fbx
    )
    take_label = os.path.basename(take_dir.rstrip("\\/")) or "take"
    if settings.initialized:
        out_dir = project.ensure_project_directories(settings) / 'cache' / 'foot_joints'
    else:
        import tempfile
        out_dir = Path(tempfile.gettempdir()) / 'mocap_doctor' / 'foot_joints'
    out = out_dir / f"{take_label}_fbx_joints.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = context.scene
    pbs = list(armature.pose.bones)
    names = [pb.name for pb in pbs]
    parents = [pb.parent.name if pb.parent else "" for pb in pbs]
    f0 = int(round(action.frame_range[0]))
    f1 = int(round(action.frame_range[1]))
    if frame_limit:
        f1 = min(f1, f0 + int(frame_limit) - 1)
    frames = list(range(f0, f1 + 1))
    pos = np.zeros((len(frames), len(names), 3), dtype=np.float64)
    for index, frame in enumerate(frames):
        scene.frame_set(frame)
        for j, pb in enumerate(pbs):
            pos[index, j] = np.array(
                (armature.matrix_world @ pb.matrix).translation, dtype=np.float64
            )
    np.savez(
        str(out), positions=pos, names=np.array(names, dtype=object),
        parents=np.array(parents, dtype=object),
        frames=np.array(frames, dtype=np.int64),
        arm_matrix=np.array(armature.matrix_world, dtype=np.float64),
        source_fbx=str(settings.official_fbx or ""),
    )
    missing = [n for n in ("L_Ankle", "L_Foot", "R_Ankle", "R_Foot") if n not in names]
    if missing:
        settings.foot_contacts_joints_npz = ""
        return False
    settings.foot_contacts_joints_npz = str(out)
    return True


CLASSES = (
    MD_OT_PrepareVMDExport,
    MD_OT_RunStep,
    MD_OT_CreateProject,
    MD_OT_DiscoverObjects,
    MD_OT_UseSceneRange,
    MD_OT_EnsureFPS,
    MD_OT_SetRangeBoundary,
    MD_OT_SyncRangeFromScene,
    MD_OT_NavigateStep,
    MD_OT_RestoreBeforeStep,
    MD_OT_PklHandDetect,
    MD_OT_PklHandRepair,
    MD_OT_CreateCheckpoint,
    MD_OT_TargetFixA,
    MD_OT_TargetFixB,
    MD_OT_TargetFixC,
    MD_OT_TargetFixAll,
    MD_OT_RetargetPrepare,
    MD_OT_MarkRetargetDone,
    MD_OT_SaveRestPreset,
    MD_OT_FingerAlign,
    MD_OT_DeleteCheckpoints,
    MD_OT_RestoreLastCheckpoint,
    MD_OT_EnterAnnotationMode,
    MD_OT_EnterAnnotationAir,
    MD_OT_ExitAnnotationMode,
    MD_OT_ResetEffectiveContacts,
    MD_OT_ReloadAirHints,
    MD_OT_MMDBake,
    MD_OT_ValidateManualMMDBake,
    MD_OT_CleanupLegFK,
    MD_OT_AcceptPreview,
    MD_OT_DiscardPreview,
    MD_OT_ExportVMD,
    MD_OT_PrepareReceiverTemplate,
    MD_OT_AgentServerToggle,
    MD_OT_AgentPreviewToggle,
    MD_OT_AgentFixMute,
    MD_OT_AgentFixSelAll,
    MD_OT_AgentFixSelWindow,
    MD_OT_AgentFixCommit,
    MD_OT_AgentFixRevert,
    MD_OT_AgentFixForget,
    MD_OT_AgentFixRefresh,
    MD_OT_AgentParamChoice,
    MD_OT_AgentDirCreate,
)


def register_operators():
    registered = []
    try:
        for cls in CLASSES:
            bpy.utils.register_class(cls)
            registered.append(cls)
        if _cleanup_annotation_before_load not in bpy.app.handlers.load_pre:
            bpy.app.handlers.load_pre.append(_cleanup_annotation_before_load)
        if _resume_annotation_after_load not in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.append(_resume_annotation_after_load)
    except Exception:
        if _cleanup_annotation_before_load in bpy.app.handlers.load_pre:
            bpy.app.handlers.load_pre.remove(_cleanup_annotation_before_load)
        if _resume_annotation_after_load in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(_resume_annotation_after_load)
        for cls in reversed(registered):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
        raise


def unregister_operators():
    _cancel_pending_annotation_open()
    if _cleanup_annotation_before_load in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_cleanup_annotation_before_load)
    if _resume_annotation_after_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_resume_annotation_after_load)
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
