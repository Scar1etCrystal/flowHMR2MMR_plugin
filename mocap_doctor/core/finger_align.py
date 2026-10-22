"""Finger correction for the MMR controllers.

Facts this file is built on (all measured on the user's rig):

* ``f_*.01_master.*`` is a free bone (parent ``ORG-hand``, no constraints, no
  children) whose *rotation* aims a whole finger and whose *uniform scale*
  bends it - the same thing the user does with S in the viewport.
* The digit chain already has a small rest curvature: the armature itself
  stores 2.9-15.1 degrees between consecutive finger bones, so "straight" is a
  property of the data, not of the pose.
* The amount of curl per unit of scale differs per finger, so each finger is
  calibrated on its own.

The correction fits each finger onto its controller: rotation aims it along the
source chord, uniform scale reproduces the source curl *total*, and the joint
details are flattened so no per-joint kink can survive.  Afterwards the achieved
curl is measured from the evaluated pose, never assumed.
"""
from __future__ import annotations

import math

from mathutils import Matrix, Quaternion

from .animation import (bone_path, current_view_layer, get_action,
                        preserve_scene_frame, set_linear_for_paths,
                        set_scene_frame, update_action)

FIT_STEMS = ('f_index', 'f_middle', 'f_ring', 'f_pinky', 'thumb')


def _fit_names(stem, side):
    master = f'{stem}.01_master.{side}'
    digits = [f'{stem}.0{i}.{side}' for i in (1, 2, 3)]
    return master, digits


def _bone_dir(rig, name):
    pb = rig.pose.bones[name]
    vector = pb.tail - pb.head
    return vector.normalized() if vector.length > 1e-8 else None


def _local_quaternion(rig, name):
    pb = rig.pose.bones[name]
    return (pb.rotation_quaternion.copy() if pb.rotation_mode == 'QUATERNION'
            else pb.rotation_euler.to_quaternion())


def _pose_rotation(bone, quaternion):
    return Matrix.LocRotScale(bone.matrix.translation, quaternion,
                              bone.matrix.to_scale())


def _basis_for(rig, name, pose_matrix):
    """matrix_basis for a desired pose-space matrix."""

    bone = rig.data.bones[name]
    parent = bone.parent
    if parent is None:
        return bone.matrix_local.inverted() @ pose_matrix
    parent_pose = rig.pose.bones[parent.name].matrix
    return (parent_pose @ parent.matrix_local.inverted()
            @ bone.matrix_local).inverted() @ pose_matrix


def _finger_total_bend(rig, digits):
    directions = [_bone_dir(rig, name) for name in digits]
    if any(direction is None for direction in directions):
        return None
    return sum(math.degrees(directions[0].angle(direction))
               for direction in directions[1:])


def _rest_total_bend(rig, digits):
    """Base bend straight from the armature data (no pose, no evaluation)."""

    directions = []
    for name in digits:
        bone = rig.data.bones[name]
        vector = bone.tail_local - bone.head_local
        if vector.length < 1e-9:
            return 0.0
        directions.append(vector.normalized())
    return sum(math.degrees(directions[0].angle(direction))
               for direction in directions[1:])


def _write_points(rig, per_bone):
    """Key rotation channels only, with Euler/quaternion continuity."""

    paths = set()
    for name, values in per_bone.items():
        pb = rig.pose.bones[name]
        channel = ('rotation_quaternion' if pb.rotation_mode == 'QUATERNION'
                   else 'rotation_euler')
        previous = None
        for frame, quaternion in values:
            quat = quaternion.normalized()
            if channel == 'rotation_quaternion':
                if previous is not None:
                    quat.make_compatible(previous)
                previous = quat.copy()
                pb.rotation_quaternion = quat
            else:
                euler = quat.to_euler(pb.rotation_mode)
                if previous is not None:
                    euler.make_compatible(previous)
                previous = euler.copy()
                pb.rotation_euler = euler
            pb.keyframe_insert(data_path=channel, frame=frame, group=pb.name)
        paths.add(bone_path(name, channel))
    return paths


def _write_scale_points(rig, per_bone):
    """Key the uniform scale of the controllers (all three axes)."""

    paths = set()
    for name, values in per_bone.items():
        pb = rig.pose.bones[name]
        for axis in range(3):
            for frame, scale in values:
                pb.scale = scale
                pb.keyframe_insert(data_path='scale', index=axis, frame=frame,
                                   group=pb.name)
        paths.add(f'pose.bones["{name}"].scale')
    return paths


def _flat_details(rig, names):
    for name in names:
        pb = rig.pose.bones[name]
        if pb.rotation_mode == 'QUATERNION':
            pb.rotation_quaternion = (1, 0, 0, 0)
        else:
            pb.rotation_euler = (0, 0, 0)


def calibrate_finger_curl(rig, layer):
    """Base bend from rest data; per-unit curl measured with no action linked."""

    result = {}
    animation = rig.animation_data
    saved_action = animation.action if animation else None
    saved = {}
    for side in ('L', 'R'):
        for stem in FIT_STEMS:
            master_name, digits = _fit_names(stem, side)
            if master_name not in rig.pose.bones:
                continue
            if any(name not in rig.data.bones for name in digits):
                continue
            saved[(side, stem)] = (
                rig.pose.bones[master_name].scale.copy(),
                [_local_quaternion(rig, name) for name in digits],
            )
    if animation:
        animation.action = None
        layer.update()
    try:
        for key in list(saved):
            side, stem = key
            master_name, digits = _fit_names(stem, side)
            _flat_details(rig, digits)
            master = rig.pose.bones[master_name]
            master.scale = (1.0, 1.0, 1.0)
            layer.update()
            measured = _finger_total_bend(rig, digits)
            base = _rest_total_bend(rig, digits)
            if measured is not None and abs(measured - base) > 1.0:
                # 求值与 rest 不一致时以实测为准（约束可能改变了链条）。
                base = measured
            master.scale = (0.85, 0.85, 0.85)
            layer.update()
            bent = _finger_total_bend(rig, digits)
            if bent is None or bent <= base:
                result[key] = (base, 360.0)
            else:
                result[key] = (base, (bent - base) / 0.15)
    finally:
        for key, (scale, quaternions) in saved.items():
            side, stem = key
            master_name, digits = _fit_names(stem, side)
            rig.pose.bones[master_name].scale = scale
            for name, quaternion in zip(digits, quaternions):
                pb = rig.pose.bones[name]
                if pb.rotation_mode == 'QUATERNION':
                    pb.rotation_quaternion = quaternion
                else:
                    pb.rotation_euler = quaternion.to_euler(pb.rotation_mode)
        layer.update()
        if animation:
            animation.action = saved_action
            layer.update()
    return result


def fit_finger_masters(scene, rig, action, *, frame_start, frame_end, strength=1.0):
    """Aim + curl each finger through its controller; flatten joint details."""

    if get_action(rig) is not action:
        raise RuntimeError('手指校正需要当前动作')
    tracks = [t.name for t in rig.animation_data.nla_tracks
              if not t.mute and any(not s.mute for s in t.strips)]
    if tracks:
        raise RuntimeError('请先停用叠加动作再校正手指')
    strength = max(0., min(1., float(strength)))
    frames = list(range(int(frame_start), int(frame_end) + 1))
    layer = current_view_layer()
    targets = {}
    master_points = {}
    scale_points = {}
    detail_points = {}
    aim_errors = []
    paths = set()
    max_curl_error = 0.0
    limited_count = 0
    worst = None
    samples = []
    calibration = {}
    with preserve_scene_frame(scene, layer):
        set_scene_frame(scene, frames[0], layer)
        calibration = calibrate_finger_curl(rig, layer)
        for index, frame in enumerate(frames):
            set_scene_frame(scene, frame, layer)
            if index % 400 == 0:
                print(f'[FINGER-FIT] {frame}/{frame_end}', flush=True)
            for side in ('L', 'R'):
                for stem in FIT_STEMS:
                    key = (side, stem)
                    master_name, digits = _fit_names(stem, side)
                    if key not in calibration:
                        continue
                    master = rig.pose.bones[master_name]
                    current = [_bone_dir(rig, name) for name in digits]
                    if any(direction is None for direction in current):
                        continue
                    target_bend = sum(math.degrees(current[0].angle(direction))
                                      for direction in current[1:])
                    base, per_unit = calibration[key]
                    if target_bend <= base + 0.5:
                        scale_y = 1.0
                    else:
                        scale_y = 1.0 - (target_bend - base) / max(per_unit, 1.0)
                        scale_y = max(0.05, min(1.0, scale_y))
                    desired = (rig.pose.bones[digits[2]].tail
                               - rig.pose.bones[digits[0]].head)
                    if desired.length < 1e-8:
                        continue
                    desired.normalize()
                    delta = current[0].rotation_difference(desired)
                    if strength < 1.0:
                        delta = Quaternion().slerp(delta, strength)
                    wanted_rotation = delta @ master.matrix.to_quaternion()
                    basis = _basis_for(
                        rig, master_name,
                        _pose_rotation(master, wanted_rotation)).to_quaternion()
                    master_points.setdefault(master_name, []).append((frame, basis))
                    scale_points.setdefault(master_name, []).append((frame, (scale_y,) * 3))
                    for name in digits:
                        detail_points.setdefault(name, []).append(
                            (frame, Quaternion((1, 0, 0, 0))))
                    aim_errors.append(math.degrees(current[0].angle(desired)))
                    targets[(side, stem, frame)] = target_bend
        paths = _write_points(rig, master_points)
        paths |= _write_points(rig, detail_points)
        paths |= _write_scale_points(rig, scale_points)
        set_linear_for_paths(action, paths)
        update_action(action)
        check_frames = sorted({frame for (_s, _st, frame) in targets})
        if len(check_frames) > 60:
            step = max(1, len(check_frames) // 60)
            check_frames = check_frames[::step] + [check_frames[-1]]
        for frame in check_frames:
            set_scene_frame(scene, frame, layer)
            for side in ('L', 'R'):
                for stem in FIT_STEMS:
                    if (side, stem, frame) not in targets:
                        continue
                    _master_name, digits = _fit_names(stem, side)
                    achieved = _finger_total_bend(rig, digits)
                    if achieved is None:
                        continue
                    expected = targets[(side, stem, frame)]
                    base = calibration.get((side, stem), (0.0, 360.0))[0]
                    limited = expected <= base + 0.5
                    if limited:
                        limited_count += 1
                    error = 0.0 if limited else abs(achieved - expected)
                    samples.append([frame, side, stem, round(expected, 2),
                                    round(achieved, 2), bool(limited)])
                    if error > max_curl_error:
                        max_curl_error = error
                        worst = samples[-1]
    return {
        'operation': 'finger_fit_masters',
        'status': 'applied' if master_points else 'nothing_to_do',
        'frames': len(frames),
        'masters_written': len(master_points),
        'details_flat': len(detail_points),
        'scales_written': len(scale_points),
        'aim_error_max_deg': round(max(aim_errors), 2) if aim_errors else None,
        'curl_error_measured_max_deg': round(max_curl_error, 2),
        'curl_limited_frames': limited_count,
        'curl_worst_sample': worst,
        'curl_samples': samples[:16],
        'calibration': {f'{side}_{stem}': [round(base, 1), round(per_unit, 1)]
                        for (side, stem), (base, per_unit) in calibration.items()},
        'rotation_paths_written': sorted(paths),
        'location_curves_written': 0,
        'scale_curves_written': len(scale_points) * 3,
    }
