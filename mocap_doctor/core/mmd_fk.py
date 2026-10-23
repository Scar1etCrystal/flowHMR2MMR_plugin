"""Direct FlowHMR -> MMD FK retargeting (no MMR, no ARP).

Per frame:

1. **Absolute rest-relative transfer** – every mapped bone gets
   ``desired = source_pose @ (source_rest^-1 @ target_rest)``, which reproduces
   the target's own rest pose whenever the source is at rest and carries the
   motion rigidly (roll/twist correct by construction).
2. **Joint-angle correction** – per-bone absolute transfer lets joint angles
   drift (source and MMD rest bone axes differ); every mapped parent/child pair
   is rebuilt around its own bend axis with the *source* relative angle.
3. **Ankle placement** – knees are re-solved as a two-link chain so the ankle
   lands where the source leg puts it (scaled by leg-length ratio), instead of
   trusting transfer alone.  Feet therefore stay where the source put them.
4. **Forearm/wrist twist split** – the twist part of 腕→ひじ is moved onto the
   MMD twist bones (``腕捩``/``手捩``) while the hand keeps its orientation, so
   forearm roll looks right without moving the hand.

Legs are pure FK (``ひざ``/``足首`` IK constraints are muted).  ``センター``
carries translation, ``下半身`` the pelvis rotation.  Keys are written with
``keyframe_insert`` and read back as proof.
"""

from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Quaternion, Vector

from .animation import (current_view_layer, preserve_scene_frame,
                        set_linear_for_paths, set_scene_frame, update_action)
from .hand_spacing import solve_two_link
from .mmd_fk_map import (BONE_TABLE, build_lookup, mute_blocking_constraints,
                         resolve)

SPINE_MERGE_WEIGHT = 0.5
ROOT_BONE = "センター"          # 源 Pelvis = 根部（位移 + 整体旋转）
WRITE_EXCLUDE = {"下半身"}      # 根部职责归 センター，下半身保持静止
ANGLE_FIX_EPS = math.radians(0.02)
TWIST_SHARE = 0.5          # 前臂/手腕扭转分配给 腕捩/手捩 的比例
LEG_SIDES = ('L', 'R')


def _rest_world(rig):
    return {bone.name: bone.matrix_local.copy() for bone in rig.data.bones}


def _group_sources():
    grouped: dict[str, list[str]] = {}
    for source_joint, japanese, _required in BONE_TABLE:
        grouped.setdefault(japanese, []).append(source_joint)
    return grouped


def _basis_from_desired(target_pose_world, parent_pose_world, rest_parent, rest_bone):
    if parent_pose_world is None:
        return rest_bone.inverted() @ target_pose_world
    return (parent_pose_world
            @ (rest_parent.inverted() @ rest_bone)).inverted() @ target_pose_world


def _blend_pose(poses, weight):
    first = poses[0]
    rotation = first.to_quaternion()
    for other in poses[1:]:
        rotation = rotation.slerp(other.to_quaternion(), weight)
    return Matrix.LocRotScale(first.translation, rotation, first.to_scale())


def _body_scale(source, target, source_rest_world, target_rest_world):
    """World height ratio; source and target armatures have different object scales."""

    source_height = (source_rest_world['Head'].translation
                     - source_rest_world['Pelvis'].translation).length
    target_height = (target_rest_world['頭'].translation
                     - target_rest_world['センター'].translation).length
    if source_height < 1e-6 or target_height < 1e-6:
        return 1.0
    return target_height / source_height


def _depth(bone):
    depth = 0
    while bone.parent is not None:
        bone = bone.parent
        depth += 1
    return depth


def _source_joint_of(bones):
    mapping = {}
    for joint, pose_bone in bones.items():
        mapping.setdefault(pose_bone.name, joint)
    return mapping


def _swing_twist(quaternion, axis):
    """Split a rotation into (swing, twist) around ``axis``."""

    axis = axis.normalized()
    projected = Vector((quaternion.x, quaternion.y, quaternion.z)).dot(axis)
    twist = Quaternion(axis, 2.0 * math.atan2(projected, quaternion.w))
    swing = quaternion @ twist.inverted()
    return swing, twist


def _correct_rotations(target, source, desired, source_joint_of,
                       source_pose_world, source_rest_world, target_rest_world):
    """Rebuild relative rotations with the source joint angles."""

    corrected: dict[str, Quaternion] = {}
    for name in sorted(desired, key=lambda item: _depth(target.data.bones[item])):
        rotation = desired[name].to_quaternion()
        bone = target.data.bones[name]
        parent = bone.parent
        if parent is not None and parent.name in corrected:
            source_parent = source_joint_of.get(parent.name)
            source_child = source_joint_of.get(name)
            source_parent_bone = (source.data.bones.get(source_parent)
                                  if source_parent else None)
            if (source_child and source_parent
                    and source_parent_bone is not None
                    and source.data.bones[source_child].parent is source_parent_bone):
                s_rel = ((source_pose_world[source_parent].inverted()
                          @ source_pose_world[source_child]).to_quaternion())
                s_rel_rest = ((source_rest_world[source_parent].inverted()
                               @ source_rest_world[source_child]).to_quaternion())
                source_delta = s_rel_rest.inverted() @ s_rel
                t_rel_rest = ((target_rest_world[parent.name].inverted()
                               @ target_rest_world[name]).to_quaternion())
                target_rel = corrected[parent.name].inverted() @ rotation
                target_delta = t_rel_rest.inverted() @ target_rel
                if (target_delta.angle > 1e-6
                        and abs(source_delta.angle - target_delta.angle) > ANGLE_FIX_EPS):
                    fixed = t_rel_rest @ Quaternion(target_delta.axis, source_delta.angle)
                    rotation = corrected[parent.name] @ fixed
        corrected[name] = rotation
    return corrected


def _target_leg_names(target, side):
    lookup = build_lookup(target)
    suffix = f'.{side}'
    return {
        'thigh': getattr(lookup.get(f'足{suffix}'), 'name', None),
        'shin': getattr(lookup.get(f'ひざ{suffix}'), 'name', None),
        'ankle': getattr(lookup.get(f'足首{suffix}'), 'name', None),
    }


def _leg_lengths(source, target, side, source_pose_world, target_rest_world):
    """Source and target leg lengths, both in world space (object scale included)."""

    src = (('L_Hip' if side == 'L' else 'R_Hip'),
           ('L_Knee' if side == 'L' else 'R_Knee'),
           ('L_Ankle' if side == 'L' else 'R_Ankle'))
    thigh = (source_pose_world[src[0]].translation
             - source_pose_world[src[1]].translation).length
    shin = (source_pose_world[src[1]].translation
            - source_pose_world[src[2]].translation).length
    names = _target_leg_names(target, side)
    if not all(names.values()):
        return src, 0.0, 0.0
    target_thigh = (target_rest_world[names['thigh']].translation
                    - target_rest_world[names['shin']].translation).length
    target_shin = (target_rest_world[names['shin']].translation
                   - target_rest_world[names['ankle']].translation).length
    return src, thigh + shin, target_thigh + target_shin


def _apply_leg_fix(target, source, final, source_pose_world, target_rest_world):
    """Re-solve knee/ankle so the ankle sits where the source puts it."""

    errors = 0.0
    for side in LEG_SIDES:
        names = _target_leg_names(target, side)
        if not all(names.values()):
            continue
        thigh = final.get(names['thigh'])
        shin = final.get(names['shin'])
        ankle = final.get(names['ankle'])
        if thigh is None or shin is None or ankle is None:
            continue
        src, source_length, target_length = _leg_lengths(
            source, target, side, source_pose_world, target_rest_world)
        if source_length < 1e-6 or target_length < 1e-6:
            continue
        scale = target_length / source_length
        source_hip = source_pose_world[src[0]]
        source_ankle = source_pose_world[src[2]].translation
        hip_position = thigh.translation.copy()
        hip_rotation = thigh.to_quaternion()
        offset = (source_ankle - source_hip.translation)
        offset = hip_rotation @ (source_hip.to_quaternion().inverted() @ offset)
        desired_ankle = hip_position + offset * scale
        solved, error = solve_two_link(
            [thigh, shin, ankle], desired_ankle)
        errors = max(errors, error)
        final[names['thigh']] = solved[0]
        final[names['shin']] = solved[1]
        # 足首保留原旋转（脚掌朝向），只随小腿更新位置
        delta = (shin.to_quaternion().inverted()
                 @ solved[1].to_quaternion())
        final[names['ankle']] = Matrix.LocRotScale(
            ankle.translation, (delta @ ankle.to_quaternion()), ankle.to_scale())
    return errors


def _twist_bone(target, lookup, japanese):
    pose_bone = lookup.get(japanese)
    return pose_bone.name if pose_bone is not None else None


def _apply_twist_split(target, final, lookup, rest_world):
    """Move 腕→ひじ / ひじ→手首 twist onto the MMD twist bones.

    The hand/forearm keeps its orientation: the twist bone takes part of the
    relative twist and the child's basis absorbs the difference automatically
    because parents are written before children.
    """

    applied = 0
    for side in ('L', 'R'):
        suffix = f'.{side}'
        upper = _twist_bone(target, lookup, f'腕{suffix}')
        twist = _twist_bone(target, lookup, f'腕捩{suffix}')
        fore = _twist_bone(target, lookup, f'ひじ{suffix}')
        wrist_twist = _twist_bone(target, lookup, f'手捩{suffix}')
        hand = _twist_bone(target, lookup, f'手首{suffix}')
        if twist and upper in final and fore in final and twist in rest_world:
            upper_rotation = final[upper].to_quaternion()
            total = upper_rotation.inverted() @ final[fore].to_quaternion()
            _swing, twist_rotation = _swing_twist(total, Vector((0.0, 1.0, 0.0)))
            if twist_rotation.angle > 1e-5:
                rest_rel = (rest_world[upper].inverted()
                            @ rest_world[twist]).to_quaternion()
                translation = (final[upper] @ rest_world[upper].inverted()
                               @ rest_world[twist]).translation
                rotation = (upper_rotation @ rest_rel
                            @ Quaternion(Vector((0.0, 1.0, 0.0)),
                                         twist_rotation.angle * TWIST_SHARE))
                final[twist] = Matrix.LocRotScale(translation, rotation, (1.0, 1.0, 1.0))
                applied += 1
        if wrist_twist and fore in final and hand in final and wrist_twist in rest_world:
            fore_rotation = final[fore].to_quaternion()
            total = fore_rotation.inverted() @ final[hand].to_quaternion()
            _swing, twist_rotation = _swing_twist(total, Vector((0.0, 1.0, 0.0)))
            if twist_rotation.angle > 1e-5:
                rest_rel = (rest_world[fore].inverted()
                            @ rest_world[wrist_twist]).to_quaternion()
                translation = (final[fore] @ rest_world[fore].inverted()
                               @ rest_world[wrist_twist]).translation
                rotation = (fore_rotation @ rest_rel
                            @ Quaternion(Vector((0.0, 1.0, 0.0)),
                                         twist_rotation.angle * TWIST_SHARE))
                final[wrist_twist] = Matrix.LocRotScale(translation, rotation,
                                                        (1.0, 1.0, 1.0))
                applied += 1
    return applied


def _fresh_action(target):
    if target.animation_data is None:
        target.animation_data_create()
    name = f'{target.name}_MMD_FK'
    previous = bpy.data.actions.get(name)
    if previous is not None:
        bpy.data.actions.remove(previous)
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    target.animation_data.action = action
    return action


def retarget_to_mmd_fk(scene, source, target, *, frame_start, frame_end,
                       spine_weight: float = SPINE_MERGE_WEIGHT,
                       verify_step: int = 10):
    """Write FlowHMR motion onto a clean MMD armature as FK keys."""

    if source is None or source.type != 'ARMATURE':
        raise RuntimeError('请选择源骨架')
    if target is None or target.type != 'ARMATURE':
        raise RuntimeError('请选择 MMD 骨架')
    if source == target:
        raise RuntimeError('源骨架与 MMD 骨架不能相同')

    mapping = resolve(target)
    if mapping['missing']:
        raise RuntimeError('MMD 骨架缺少必需骨骼：' + '、'.join(mapping['missing']))
    bones = mapping['bones']
    if not bones:
        raise RuntimeError('映射为空，无法重定向')

    source_rest = _rest_world(source)
    target_rest = _rest_world(target)
    missing_source = [joint for joint in bones if joint not in source.data.bones]
    if missing_source:
        raise RuntimeError('源骨架缺少关节：' + '、'.join(missing_source))

    source_matrix_world = source.matrix_world.copy()
    target_matrix_world = target.matrix_world.copy()
    source_rest_world = {joint: source_matrix_world @ source_rest[joint] for joint in bones}
    target_rest_world = {name: target_matrix_world @ matrix
                         for name, matrix in target_rest.items()}

    body_scale = _body_scale(source, target, source_rest_world, target_rest_world)

    grouped = _group_sources()
    source_joint_of = _source_joint_of(bones)
    lookup = build_lookup(target)
    muted = mute_blocking_constraints(target, bones)
    frames = list(range(int(frame_start), int(frame_end) + 1))
    if not frames:
        raise RuntimeError('帧范围为空')

    action = _fresh_action(target)
    layer = current_view_layer()
    wanted_rotation: dict[str, dict[int, Quaternion]] = {}
    wanted_location: dict[str, dict[int, object]] = {}
    max_leg_error = 0.0
    twist_frames = 0

    with preserve_scene_frame(scene, layer):
        # 位移基准 = 源动画首帧，而不是骨架 rest：
        # 源 FBX 的 rest 骨盆在 z=-0.25m，动画却在 z=0.95m，用 rest 当零点
        # 会凭空多出 1.2m 位移（实测整个人被抬高 2.5m）。
        set_scene_frame(scene, frames[0], layer)
        source_baseline = {
            joint: source_matrix_world @ source.pose.bones[joint].matrix
            for joint in bones
        }
        for index, frame in enumerate(frames):
            set_scene_frame(scene, frame, layer)
            if index % 200 == 0:
                print(f'[MMD-FK] {frame}/{frame_end}', flush=True)

            source_pose_world = {joint: source_matrix_world @ source.pose.bones[joint].matrix
                                 for joint in bones}
            desired: dict[str, Matrix] = {}
            for japanese, joints in grouped.items():
                if japanese in WRITE_EXCLUDE:
                    continue
                pose_bone = next((bones[j] for j in joints if j in bones), None)
                if pose_bone is None:
                    continue
                target_rest_pose = target_rest_world[pose_bone.name]
                rotations = []
                translation = None
                for joint in joints:
                    if joint not in bones:
                        continue
                    rest_s = source_rest_world[joint]
                    pose_s = source_pose_world[joint]
                    # 正确的 rest 相对搬运：S_pose @ (S_rest^-1 T_rest)。
                    # 直接套 T_rest @ (S_rest^-1 S_pose) 会丢共轭项，源骨轴会被
                    # 当成目标骨轴，姿态整体错乱（实测四肢乱飞）。
                    offset_rotation = (rest_s.to_quaternion().inverted()
                                       @ target_rest_pose.to_quaternion())
                    rotations.append(pose_s.to_quaternion() @ offset_rotation)
                    if translation is None:
                        baseline = source_baseline[joint]
                        translation = (target_rest_pose.translation
                                       + body_scale * (pose_s.translation
                                                       - baseline.translation))
                if not rotations or translation is None:
                    continue
                rotation = rotations[0]
                for extra in rotations[1:]:
                    rotation = rotation.slerp(extra, spine_weight)
                desired[pose_bone.name] = Matrix.LocRotScale(translation, rotation,
                                                             (1.0, 1.0, 1.0))

            corrected = _correct_rotations(
                target, source, desired, source_joint_of,
                source_pose_world, source_rest_world, target_rest_world)
            final: dict[str, Matrix] = {}
            for name, pose in desired.items():
                rotation = corrected[name]
                final[name] = Matrix.LocRotScale(pose.translation, rotation,
                                                 pose.to_scale())

            max_leg_error = max(max_leg_error,
                                _apply_leg_fix(target, source, final,
                                               source_pose_world, target_rest_world))
            if _apply_twist_split(target, final, lookup, target_rest_world):
                twist_frames += 1

            for name in sorted(final, key=lambda item: _depth(target.data.bones[item])):
                target_pose = final[name]
                bone = target.data.bones.get(name)
                if bone is None:
                    continue
                parent = bone.parent
                if parent is None:
                    parent_pose = None
                    rest_parent = None
                elif parent.name in final:
                    parent_pose = final[parent.name]
                    rest_parent = target_rest[parent.name]
                else:
                    parent_pose = target_rest_world[parent.name]
                    rest_parent = target_rest[parent.name]
                basis = _basis_from_desired(target_pose, parent_pose, rest_parent,
                                            target_rest[name])
                write_location = name == ROOT_BONE
                location = basis.translation.copy() if write_location else None
                rotation = basis.to_quaternion()
                pose_bone = target.pose.bones[name]
                if pose_bone.rotation_mode == 'QUATERNION':
                    pose_bone.rotation_quaternion = rotation
                    pose_bone.keyframe_insert('rotation_quaternion', frame=frame,
                                              group=name)
                else:
                    pose_bone.rotation_euler = rotation.to_euler(pose_bone.rotation_mode)
                    pose_bone.keyframe_insert('rotation_euler', frame=frame, group=name)
                if write_location:
                    pose_bone.location = location
                    pose_bone.keyframe_insert('location', frame=frame, group=name)
                    wanted_location.setdefault(name, {})[frame] = location.copy()
                wanted_rotation.setdefault(name, {})[frame] = rotation.copy()

    set_linear_for_paths(action, {curve.data_path for curve in action.fcurves})
    update_action(action)

    check_frames = sorted(set(frames[::max(1, int(verify_step))] + [frames[-1]]))
    max_pose_error = 0.0
    with preserve_scene_frame(scene, layer):
        for frame in check_frames:
            set_scene_frame(scene, frame, layer)
            for name, rotations in wanted_rotation.items():
                wanted = rotations.get(frame)
                if wanted is None:
                    continue
                pose_bone = target.pose.bones[name]
                if pose_bone.rotation_mode == 'QUATERNION':
                    evaluated = pose_bone.rotation_quaternion.copy()
                else:
                    evaluated = pose_bone.rotation_euler.to_quaternion()
                angle = evaluated.rotation_difference(wanted).angle
                max_pose_error = max(max_pose_error,
                                     math.degrees(min(angle, math.tau - angle)))
                locations = wanted_location.get(name)
                if locations and frame in locations:
                    max_pose_error = max(
                        max_pose_error,
                        (pose_bone.location - locations[frame]).length * 1000.0)
    if max_pose_error > 0.5:
        raise RuntimeError(f'FK 写入校验失败：最大偏差 {max_pose_error:.2f}')

    return {
        'operation': 'retarget_mmd_fk',
        'status': 'applied',
        'frames': len(frames),
        'bones_written': len(wanted_rotation),
        'action': action.name,
        'source': source.name,
        'target': target.name,
        'mapping': {
            'resolved': len(bones),
            'missing': mapping['missing'],
            'optional_missing': mapping['optional_missing'],
            'fallbacks': mapping['used_fallbacks'],
            'name_source': mapping['name_source'],
        },
        'muted_constraints': [f'{bone}:{con}' for bone, con in muted],
        'spine_merge_weight': spine_weight,
        'body_scale': round(body_scale, 6),
        'joint_angle_correction': True,
        'leg_ankle_fix_max_error_m': round(max_leg_error, 6),
        'twist_split_frames': twist_frames,
        'max_pose_error_deg': round(max_pose_error, 3),
        'verified_frames': len(check_frames),
    }
