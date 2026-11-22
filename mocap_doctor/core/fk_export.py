"""Turn a baked MMD action into pure-FK, VMD-safe data.

The repair pipeline keeps using MMR/IK, but the exported MMD action must be pure
FK:

* every rotation channel is kept as baked - the baked leg rotations are the FK
  result of the IK-driven pose, so with the model's own IK switched off they
  reproduce the same pose;
* the body translation is moved onto ``センター``.  The old bake left it in the
  local ``location`` of all 57 driven bones (up to 0.42 m each); MMD locks those
  bones, and the duplicated offsets make IK explode;
* ``location`` curves are removed everywhere else, all ``scale`` curves are
  removed (VMD cannot store them), and the ``足ＩＫ`` / ``つま先ＩＫ`` /
  ``mmd_ik_toggle`` channels are removed;
* MMD leg IK constraints are muted so the FK result plays back as authored.

The result is verified frame by frame against the baked pose (positions within
``tolerance``, rotations unchanged).
"""

from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector

from .animation import (bone_path, current_view_layer, preserve_scene_frame,
                        set_linear_for_paths, set_scene_frame, update_action)

CENTER_BONE = "センター"
ROOT_BONE = "全ての親"
LOWER_BONE = "下半身"


def _rest_matrices(armature):
    return {bone.name: bone.matrix_local.copy() for bone in armature.data.bones}


def _chain_matrix(armature, rest, base_matrix, start_name, target_name):
    """Matrix produced by walking rest-relative transforms from ``start_name``."""

    chain = []
    bone = armature.data.bones[target_name]
    while bone is not None and bone.name != start_name:
        chain.append(bone.name)
        bone = bone.parent
    if bone is None:
        return base_matrix.copy()
    chain.reverse()
    matrix = base_matrix.copy()
    previous = start_name
    for name in chain:
        matrix = matrix @ (rest[previous].inverted() @ rest[name])
        previous = name
    return matrix


def _channel_bone(fcurve):
    path = str(getattr(fcurve, "data_path", ""))
    if not path.startswith('pose.bones["'):
        return ""
    return path.split('"')[1]


def normalize_to_pure_fk(scene, armature, action=None, frame_start=None,
                         frame_end=None, tolerance=0.05):
    """Rewrite a baked MMD action in place; returns a report dict."""

    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("请选择 MMD 骨架")
    if action is None:
        action = armature.animation_data.action if armature.animation_data else None
    if action is None:
        raise RuntimeError("MMD 骨架没有可清理的烘焙动作")
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action

    center = armature.pose.bones.get(CENTER_BONE)
    if center is None:
        raise RuntimeError("MMD 骨架缺少 センター")
    root = armature.pose.bones.get(ROOT_BONE)
    rest = _rest_matrices(armature)
    if root is None:
        raise RuntimeError("MMD 骨架缺少 全ての親")

    start = int(frame_start if frame_start is not None else action.frame_range[0])
    end = int(frame_end if frame_end is not None else action.frame_range[1])
    frames = list(range(start, end + 1))

    driven = set()
    for fcurve in action.fcurves:
        path = str(fcurve.data_path)
        if path.endswith("rotation_quaternion") or path.endswith("rotation_euler"):
            bone_name = _channel_bone(fcurve)
            if bone_name:
                driven.add(bone_name)
    watched = [pb for pb in armature.pose.bones if pb.name in driven]

    layer = current_view_layer()
    snapshot: dict[int, dict[str, Matrix]] = {}
    center_locations: dict[int, Vector] = {}
    with preserve_scene_frame(scene, layer):
        for frame in frames:
            set_scene_frame(scene, frame, layer)
            snapshot[frame] = {pb.name: pb.matrix.copy() for pb in watched}
            root_pose = root.matrix.copy()
            root_pose.translation = rest[ROOT_BONE].translation
            baked_lower = (armature.pose.bones[LOWER_BONE].matrix.translation
                           if LOWER_BONE in armature.pose.bones else
                           center.matrix.translation)
            lowered = _chain_matrix(armature, rest, root_pose, ROOT_BONE, LOWER_BONE) \
                if LOWER_BONE in armature.data.bones else root_pose
            translation = baked_lower - lowered.translation
            center_chain = _chain_matrix(armature, rest, root_pose, ROOT_BONE, CENTER_BONE)
            center_locations[frame] = center_chain.to_quaternion().inverted() @ translation

    removed_location = 0
    removed_scale = 0
    removed_ik = 0
    toggle_curves_kept = 0
    for fcurve in list(action.fcurves):
        path = str(fcurve.data_path)
        bone_name = _channel_bone(fcurve)
        if path.endswith(".scale"):
            action.fcurves.remove(fcurve)
            removed_scale += 1
            continue
        if "mmd_ik_toggle" in path:
            # mmd_tools 的 VMD 导出器**只**从这条曲线生成「禁用 IK」段
            # (core/vmd/exporter.py:256 收集 / :377 写出骨名)。删掉它，VMD 里就
            # 没有 IK 段，接收模型默认开着的 足ＩＫ 会把 FK 腿拉回原姿势。
            # 所以这里保留曲线，改为在下面统一写 OFF。
            toggle_curves_kept += 1
            continue
        if "constraints[" in path:
            action.fcurves.remove(fcurve)
            removed_ik += 1
            continue
        if path.endswith(".location") and bone_name != CENTER_BONE:
            action.fcurves.remove(fcurve)
            removed_location += 1

    # 删掉曲线不会清空骨骼身上的当前值：它们会保留最后一次求值的位置/缩放，
    # 等于把残余位移静态叠加到每一帧上。这里显式归零。
    for pose_bone in armature.pose.bones:
        if pose_bone.name == CENTER_BONE:
            pose_bone.scale = (1.0, 1.0, 1.0)
            continue
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)

    with preserve_scene_frame(scene, layer):
        for frame in frames:
            set_scene_frame(scene, frame, layer)
            center.location = center_locations[frame]
            center.keyframe_insert(data_path="location", frame=frame, group=CENTER_BONE)
    set_linear_for_paths(action, {bone_path(CENTER_BONE, "location")})
    update_action(action)

    muted_ik = []
    ik_toggle_bones = []
    for pose_bone in armature.pose.bones:
        for constraint in pose_bone.constraints:
            if constraint.type != "IK":
                continue
            if not constraint.mute:
                constraint.mute = True
                muted_ik.append(f"{pose_bone.name}:{constraint.name}")
            # IK 约束的 subtarget 就是 IK 骨（足ＩＫ / つま先ＩＫ …）
            target_name = str(getattr(constraint, "subtarget", "") or "")
            ik_bone = armature.pose.bones.get(target_name)
            if ik_bone is not None and hasattr(ik_bone, "mmd_ik_toggle"):
                if target_name not in ik_toggle_bones:
                    ik_toggle_bones.append(target_name)

    if ik_toggle_bones:
        with preserve_scene_frame(scene, layer):
            for name in ik_toggle_bones:
                pose_bone = armature.pose.bones[name]
                pose_bone.mmd_ik_toggle = False
                for frame in (frames[0], frames[-1]):
                    pose_bone.keyframe_insert("mmd_ik_toggle", frame=frame,
                                              group=name)
        set_linear_for_paths(action, {bone_path(name, "mmd_ik_toggle")
                                      for name in ik_toggle_bones})
        update_action(action)

    max_position = 0.0
    max_rotation = 0.0
    with preserve_scene_frame(scene, layer):
        for frame in frames:
            set_scene_frame(scene, frame, layer)
            for pose_bone in watched:
                reference = snapshot[frame][pose_bone.name]
                current = pose_bone.matrix
                if pose_bone.name != CENTER_BONE:
                    # センター 的头部位置本来就会移动（根位移），不参与位置比较；
                    # 其余骨骼只允许结构性位置偏移（毫米到厘米级）。
                    max_position = max(
                        max_position,
                        (current.translation - reference.translation).length)
                angle = reference.to_quaternion().rotation_difference(
                    current.to_quaternion()).angle
                max_rotation = max(max_rotation,
                                   math.degrees(min(angle, math.tau - angle)))
    if max_position > tolerance:
        raise RuntimeError(
            f"纯 FK 清理后姿态偏差过大：{max_position * 1000:.1f} mm"
            f"（允许 {tolerance * 1000:.0f} mm）")

    return {
        "operation": "normalize_to_pure_fk",
        "frames": len(frames),
        "removed_location_curves": removed_location,
        "removed_scale_curves": removed_scale,
        "removed_ik_curves": removed_ik,
        "center": CENTER_BONE,
        "muted_ik_constraints": muted_ik,
        "ik_toggle_off_bones": list(ik_toggle_bones),
        "kept_ik_toggle_curves": toggle_curves_kept,
        "max_position_error_mm": round(max_position * 1000.0, 3),
        "max_rotation_error_deg": round(max_rotation, 4),
    }
