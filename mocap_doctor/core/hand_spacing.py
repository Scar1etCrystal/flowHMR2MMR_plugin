"""Restore inter-hand spacing on FK arms, using PoseCapture's two-link solve.

The source shoulder axis determines signed wrist separation in a body-relative
frame. Distances are scaled by arm length. Only lost lateral separation (and
incorrect crossing) is repaired; wrist midpoint and components perpendicular
to the shoulder axis stay fixed.
Unlike the foot clearance floor, there is no minimum hand gap: source claps and
intentional hand contact remain possible. Fingers are never keyed.

The geometric two-link solve follows PoseCapture's
_two_bone_fk_spacing_matrices (gvhmr_pose_capture.py:4321). This MMR adapter
converts desired pose rotations through each *actual* parent after dependency
updates: hand_fk is parented through MCH-hand_fk, not directly to forearm_fk.
Only rotation channels are keyed, with Euler/quaternion continuity preserved.
"""
from __future__ import annotations
import math
import bpy
from mathutils import Matrix, Vector
from .animation import (get_action, preserve_scene_frame, set_scene_frame,
                        current_view_layer, bone_path, update_action, set_linear_for_paths)

CHAINS = {s: ('upper_arm_fk.'+s, 'forearm_fk.'+s, 'hand_fk.'+s) for s in ('L','R')}
SOURCE = {s: (s+'_Shoulder', s+'_Elbow', s+'_Wrist') for s in ('L','R')}


def solve_two_link(matrices, target):
    """Small pose-space endpoint move with the original elbow bend direction."""
    upper_m, lower_m, wrist_m = matrices
    shoulder, elbow, wrist = (m.translation.copy() for m in matrices)
    upper, lower = elbow-shoulder, wrist-elbow
    a,b = upper.length,lower.length
    line=target-shoulder
    if min(a,b,line.length)<1e-8:
        raise RuntimeError('手臂长度或目标位置无效')
    distance=max(abs(a-b)+1e-7,min(a+b-1e-7,line.length))
    axis=line.normalized()
    endpoint=shoulder+axis*distance
    along=(a*a-b*b+distance*distance)/(2*distance)
    height=math.sqrt(max(0.0,a*a-along*along))
    old_axis=(wrist-shoulder).normalized()
    bend=(elbow-shoulder)-old_axis*(elbow-shoulder).dot(old_axis)
    bend-=axis*bend.dot(axis)
    if bend.length<1e-8:
        normal=upper.cross(lower)
        bend=normal.normalized().cross(axis) if normal.length>1e-8 else axis.orthogonal()
    bend.normalize()
    new_elbow=shoulder+axis*along+bend*height
    q1=upper.normalized().rotation_difference((new_elbow-shoulder).normalized())
    q2=lower.normalized().rotation_difference((endpoint-new_elbow).normalized())
    return (
        Matrix.LocRotScale(shoulder,q1@upper_m.to_quaternion(),upper_m.to_scale()),
        Matrix.LocRotScale(new_elbow,q2@lower_m.to_quaternion(),lower_m.to_scale()),
        Matrix.LocRotScale(endpoint,wrist_m.to_quaternion(),wrist_m.to_scale()),
    ), (endpoint-target).length


def _key_rotation(rig,pb,matrix,frame,previous):
    basis=rig.convert_space(pose_bone=pb,matrix=matrix,from_space='POSE',to_space='LOCAL')
    rotation=basis.to_quaternion().normalized()
    if pb.rotation_mode=='QUATERNION':
        if previous is not None:rotation.make_compatible(previous)
        pb.rotation_quaternion=rotation
        channel='rotation_quaternion';current=rotation.copy()
    elif pb.rotation_mode=='AXIS_ANGLE':
        if previous is not None:rotation.make_compatible(previous)
        axis,angle=rotation.to_axis_angle()
        pb.rotation_axis_angle=(angle,*axis)
        channel='rotation_axis_angle';current=rotation.copy()
    else:
        value=rotation.to_euler(pb.rotation_mode)
        if previous is not None:value.make_compatible(previous)
        pb.rotation_euler=value
        channel='rotation_euler';current=value.copy()
    pb.keyframe_insert(data_path=channel,frame=frame,group=pb.name)
    return current,channel


def compensate_hand_spacing_fk_reference(scene,source,rig,action,*,frame_start,frame_end,strength=1.0):
    """Reference implementation: keeps per-bone depsgraph updates as the oracle."""
    if get_action(rig) is not action:
        raise RuntimeError('手间距补偿需要当前动作')
    tracks=[t.name for t in rig.animation_data.nla_tracks if not t.mute and any(not s.mute for s in t.strips)]
    if tracks:raise RuntimeError('请先停用叠加动作再补偿手间距')
    missing=[n for ns in SOURCE.values() for n in ns if n not in source.pose.bones]
    missing += [n for ns in CHAINS.values() for n in ns if n not in rig.pose.bones]
    if missing:raise RuntimeError('骨架缺少手臂关节：'+', '.join(missing))
    strength=max(0.,min(1.,float(strength)))
    frames=list(range(int(frame_start),int(frame_end)+1))
    layer=current_view_layer()
    source_rest=sum((source.data.bones[a].head_local-source.data.bones[b].head_local).length
                    for ns in SOURCE.values() for a,b in zip(ns,ns[1:]))/2
    target_rest=sum((rig.data.bones[a].head_local-rig.data.bones[b].head_local).length
                    for ns in CHAINS.values() for a,b in zip(ns,ns[1:]))/2
    source_scale=abs(source.matrix_world.to_scale().x)
    target_scale=abs(rig.matrix_world.to_scale().x)
    if min(source_rest,source_scale,target_scale)<1e-8:raise RuntimeError('骨架缩放或臂长无效')
    ratio=target_rest*target_scale/(source_rest*source_scale)
    samples=[];active=0;max_move=0.;clamped=0
    previous={};paths=set();max_error=0.;max_rotation=0.;audit={}
    with preserve_scene_frame(scene,layer):
        # Snapshot *all* evaluated poses before inserting any key.
        for f in frames:
            set_scene_frame(scene,f,layer)
            for side in ('L','R'):
                parent=rig.pose.bones.get('upper_arm_parent.'+side)
                if parent and float(parent.get('IK_FK',1.))<.99:
                    raise RuntimeError('手臂需要处于FK模式')
            sm={s:[source.matrix_world@source.pose.bones[n].head for n in SOURCE[s]] for s in ('L','R')}
            tm={s:[rig.pose.bones[n].matrix.copy() for n in CHAINS[s]] for s in ('L','R')}
            sa=sm['L'][0]-sm['R'][0]
            shoulder_width=sa.length
            wrist_distance=(sm['L'][2]-sm['R'][2]).length
            ta=tm['L'][0].translation-tm['R'][0].translation
            if min(sa.length,ta.length)<1e-8:raise RuntimeError('无法确定肩部左右方向')
            sa.normalize();ta.normalize()
            desired=(sm['L'][2]-sm['R'][2]).dot(sa)*ratio/target_scale
            actual=(tm['L'][2].translation-tm['R'][2].translation).dot(ta)
            # Do not narrow already wider gestures. Correct reversed order too.
            sign=1. if desired>=0 else -1.
            error=desired-actual if actual*sign<0 or abs(actual)<abs(desired) else 0.
            close=max(0.,min(1.,(shoulder_width-wrist_distance)/(shoulder_width*.5)))
            close=close*close*(3.-2.*close)
            offset=ta*(error*.5*strength*close)
            reachable=1.0
            for side,direction in (('L',offset),('R',-offset)):
                shoulder,elbow,wrist=(m.translation for m in tm[side])
                low=abs((elbow-shoulder).length-(wrist-elbow).length)
                high=(elbow-shoulder).length+(wrist-elbow).length
                if not low <= (wrist+direction-shoulder).length <= high:
                    left,right=0.0,1.0
                    for _ in range(24):
                        middle=(left+right)*.5
                        distance=(wrist+direction*middle-shoulder).length
                        if low <= distance <= high:
                            left=middle
                        else:
                            right=middle
                    reachable=min(reachable,left)
            offset*=reachable
            if offset.length>1e-7:active+=1
            max_move=max(max_move,offset.length*target_scale)
            if (f-frame_start)%500==0:
                print(f'[HAND-SPACING] 扫描参考姿态 {f}/{frame_end}',flush=True)
            samples.append((f,tm,offset,desired,actual))
        # Every frame is keyed again (as originally): the local rotation of the
        # forearm/hand is computed with convert_space, which needs the parent's
        # freshly evaluated pose, so writing keys on a subset silently corrupted
        # the solve (verified by the endpoint check failing).  The only safe
        # speedup is skipping this whole pass when nothing needs compensating.
        corrected=[f for f,_,off,_,_ in samples if off.length>1e-7]
        print(f'[HAND-SPACING] 需要补偿 {len(corrected)} 帧；'
              f'{"" if corrected else "全部帧都无需补偿，跳过写入"}',flush=True)
        for index,(f,tm,offset,desired,actual) in enumerate(samples if corrected else ()):
            if index%200==0:
                print(f'[HAND-SPACING] 写入 {f}/{frame_end}',flush=True)
            set_scene_frame(scene,f,layer)
            solved={}
            for side,delta in (('L',offset),('R',-offset)):
                solved[side],reach_error=solve_two_link(tm[side],tm[side][2].translation+delta)
                if reach_error>1e-5:clamped+=1
            for side in ('L','R'):
                for name,matrix in zip(CHAINS[side],solved[side]):
                    pb=rig.pose.bones[name]
                    oldloc=pb.location.copy();oldscale=pb.scale.copy()
                    previous[name],channel=_key_rotation(rig,pb,matrix,f,previous.get(name))
                    paths.add(bone_path(name,channel))
                    assert (pb.location-oldloc).length<1e-8 and (pb.scale-oldscale).length<1e-8
                    # Parent pose must be re-evaluated before the child converts
                    # its matrix (was removed once; that broke the solve).
                    layer.update()
            # Check actual evaluated endpoints and wrist orientations; writing
            # the intended keys alone is not proof of a correct FK solve.
            actual_gap=(rig.pose.bones[CHAINS['L'][2]].head-rig.pose.bones[CHAINS['R'][2]].head).length*target_scale
            for side in ('L','R'):
                wrist=rig.pose.bones[CHAINS[side][2]]
                max_error=max(max_error,(wrist.head-solved[side][2].translation).length*target_scale)
                angle=wrist.matrix.to_quaternion().rotation_difference(tm[side][2].to_quaternion()).angle
                max_rotation=max(max_rotation,min(angle,abs(2*math.pi-angle)))
            if f in (1,560,570,577,585,600):
                audit[str(f)]={'before_distance_m':(tm['L'][2].translation-tm['R'][2].translation).length*target_scale,
                              'after_distance_m':actual_gap,'source_scaled_signed_m':desired*target_scale,
                              'before_signed_m':actual*target_scale,'each_offset_m':offset.length*target_scale}
    set_linear_for_paths(action,paths);update_action(action)
    if max_error>.001 or math.degrees(max_rotation)>.1:
        raise RuntimeError('手臂骨架不支持当前补偿，请丢弃预览')
    return {'operation':'hand_spacing_fk','status':'applied','frames':len(frames),
            'compensated_frames':active,'arm_scale':ratio,'max_each_hand_offset_m':max_move,
            'reach_clamped_sides':clamped,'max_wrist_position_error_m':max_error,
            'max_wrist_rotation_error_deg':math.degrees(max_rotation),'frames_audit':audit,
            'rotation_paths_written':sorted(paths),'finger_curves_written':0,
            'location_curves_written':0,'scale_curves_written':0}


# ---------------------------------------------------------------------------
# Fast path: identical maths, no per-bone dependency-graph round trips.
#
# The reference implementation calls rig.convert_space() for every bone and had
# to re-evaluate the depsgraph after each write (a child's local basis depends
# on the parent's evaluated pose) - that is the ~50 ms/frame cost.  matrix_basis
# can be derived directly:
#     basis = (parent_pose @ parent_rest^-1 @ bone_rest)^-1 @ target_pose
# so the whole pass becomes matrix maths plus one bulk fcurve write, and only a
# sampled verification pass needs the depsgraph at all.
# ---------------------------------------------------------------------------

FAST_STATS = {}


def _basis_for(target_pose, bone_rest, parent_pose, parent_rest):
    if parent_pose is None or parent_rest is None:
        return bone_rest.inverted() @ target_pose
    return (parent_pose @ parent_rest.inverted() @ bone_rest).inverted() @ target_pose


def _write_rotation_channels(rig, action, per_bone, frames, frame_start, frame_end):
    """Bulk-write rotation channels; only rotation is ever touched."""

    paths = set()
    for name, quats in per_bone.items():
        pb = rig.pose.bones[name]
        if pb.rotation_mode == 'QUATERNION':
            channel, width = 'rotation_quaternion', 4
        else:
            channel, width = 'rotation_euler', 3
        prefix = f'pose.bones["{name}"].{channel}'
        columns = [[] for _ in range(width)]
        previous = None
        for quat in quats:
            q = quat.normalized()
            if pb.rotation_mode == 'QUATERNION':
                if previous is not None:
                    q.make_compatible(previous)
                previous = q.copy()
                coords = (q.w, q.x, q.y, q.z)
            else:
                euler = q.to_euler(pb.rotation_mode)
                if previous is not None:
                    euler.make_compatible(previous)
                previous = euler.copy()
                coords = (euler.x, euler.y, euler.z)
            for k in range(width):
                columns[k].append(coords[k])
        for k in range(width):
            curve = action.fcurves.find(prefix, index=k)
            if curve is None:
                curve = action.fcurves.new(prefix, index=k, action_group=name)
            # Remove by index in reverse, rescanning until clean: holding
            # references while removing invalidates them ("Keyframe not in
            # F-Curve"), and retarget actions can hold duplicate keys per frame.
            while True:
                stale = [index for index, point in enumerate(curve.keyframe_points)
                         if frame_start - 0.5 <= point.co[0] <= frame_end + 0.5]
                if not stale:
                    break
                for index in reversed(stale):
                    curve.keyframe_points.remove(curve.keyframe_points[index])
            inserted = len(curve.keyframe_points)
            curve.keyframe_points.add(len(frames))
            for index, frame in enumerate(frames):
                point = curve.keyframe_points[inserted + index]
                point.co = (float(frame), float(columns[k][index]))
                point.interpolation = 'LINEAR'
            curve.update()
            paths.add(prefix)
    return paths


def compensate_hand_spacing_fk_fast(scene, source, rig, action, *, frame_start,
                                    frame_end, strength=1.0, verify_step=25):
    """Same result as compensate_hand_spacing_fk, without per-bone updates."""

    if get_action(rig) is not action:
        raise RuntimeError('手间距补偿需要当前动作')
    tracks = [t.name for t in rig.animation_data.nla_tracks
              if not t.mute and any(not s.mute for s in t.strips)]
    if tracks:
        raise RuntimeError('请先停用叠加动作再补偿手间距')
    strength = max(0., min(1., float(strength)))
    frames = list(range(int(frame_start), int(frame_end) + 1))
    layer = current_view_layer()
    source_rest = sum((source.data.bones[a].head_local - source.data.bones[b].head_local).length
                      for ns in SOURCE.values() for a, b in zip(ns, ns[1:])) / 2
    target_rest = sum((rig.data.bones[a].head_local - rig.data.bones[b].head_local).length
                      for ns in CHAINS.values() for a, b in zip(ns, ns[1:])) / 2
    source_scale = abs(source.matrix_world.to_scale().x)
    target_scale = abs(rig.matrix_world.to_scale().x)
    if min(source_rest, source_scale, target_scale) < 1e-8:
        raise RuntimeError('骨架缩放或臂长无效')
    ratio = target_rest * target_scale / (source_rest * source_scale)
    rests = {}
    for side in ('L', 'R'):
        first, middle, last = CHAINS[side]
        bones = {key: rig.data.bones[name] for key, name in
                 (('b0', first), ('b1', middle), ('b2', last))}
        parent_of_first = bones['b0'].parent
        parent_of_last = bones['b2'].parent
        rests[side] = {
            'b0': bones['b0'].matrix_local.copy(),
            'b1': bones['b1'].matrix_local.copy(),
            'b2': bones['b2'].matrix_local.copy(),
            'p0': parent_of_first.matrix_local.copy() if parent_of_first else None,
            'p2': parent_of_last.matrix_local.copy() if parent_of_last else None,
        }
    samples = []
    parents0 = []
    relative2 = []
    active = 0
    max_move = 0.
    clamped = 0
    audit = {}
    with preserve_scene_frame(scene, layer):
        for f in frames:
            set_scene_frame(scene, f, layer)
            for side in ('L', 'R'):
                parent = rig.pose.bones.get('upper_arm_parent.' + side)
                if parent and float(parent.get('IK_FK', 1.)) < .99:
                    raise RuntimeError('手臂需要处于FK模式')
            sm = {s: [source.matrix_world @ source.pose.bones[n].head for n in SOURCE[s]]
                  for s in ('L', 'R')}
            tm = {s: [rig.pose.bones[n].matrix.copy() for n in CHAINS[s]] for s in ('L', 'R')}
            p0 = {}
            rl2 = {}
            for side in ('L', 'R'):
                first = rig.pose.bones[CHAINS[side][0]]
                last = rig.pose.bones[CHAINS[side][2]]
                p0[side] = first.parent.matrix.copy() if first.parent else None
                rl2[side] = (rig.pose.bones[CHAINS[side][1]].matrix.inverted() @ last.parent.matrix
                             if last.parent else None)
            sa = sm['L'][0] - sm['R'][0]
            shoulder_width = sa.length
            wrist_distance = (sm['L'][2] - sm['R'][2]).length
            ta = tm['L'][0].translation - tm['R'][0].translation
            if min(sa.length, ta.length) < 1e-8:
                raise RuntimeError('无法确定肩部左右方向')
            sa.normalize()
            ta.normalize()
            desired = (sm['L'][2] - sm['R'][2]).dot(sa) * ratio / target_scale
            actual = (tm['L'][2].translation - tm['R'][2].translation).dot(ta)
            sign = 1. if desired >= 0 else -1.
            error = desired - actual if actual * sign < 0 or abs(actual) < abs(desired) else 0.
            close = max(0., min(1., (shoulder_width - wrist_distance) / (shoulder_width * .5)))
            close = close * close * (3. - 2. * close)
            offset = ta * (error * .5 * strength * close)
            reachable = 1.0
            for side, direction in (('L', offset), ('R', -offset)):
                shoulder, elbow, wrist = (m.translation for m in tm[side])
                low = abs((elbow - shoulder).length - (wrist - elbow).length)
                high = (elbow - shoulder).length + (wrist - elbow).length
                if not low <= (wrist + direction - shoulder).length <= high:
                    left, right = 0.0, 1.0
                    for _ in range(24):
                        middle = (left + right) * .5
                        distance = (wrist + direction * middle - shoulder).length
                        if low <= distance <= high:
                            left = middle
                        else:
                            right = middle
                    reachable = min(reachable, left)
            offset *= reachable
            if offset.length > 1e-7:
                active += 1
            max_move = max(max_move, offset.length * target_scale)
            samples.append((f, tm, offset, desired, actual))
            parents0.append(p0)
            relative2.append(rl2)
            if (f - frame_start) % 500 == 0:
                print(f'[HAND-SPACING] 扫描参考姿态 {f}/{frame_end}', flush=True)
        per_bone = {}
        solved_all = []
        for index, (f, tm, offset, desired, actual) in enumerate(samples):
            solved = {}
            for side, delta in (('L', offset), ('R', -offset)):
                solved[side], reach_error = solve_two_link(
                    tm[side], tm[side][2].translation + delta)
                if reach_error > 1e-5:
                    clamped += 1
            solved_all.append(solved)
            for side in ('L', 'R'):
                first, middle, last = CHAINS[side]
                rest = rests[side]
                basis0 = _basis_for(solved[side][0], rest['b0'], parents0[index][side], rest['p0'])
                basis1 = _basis_for(solved[side][1], rest['b1'], solved[side][0], rest['b0'])
                parent2 = (None if relative2[index][side] is None
                           else solved[side][1] @ relative2[index][side])
                basis2 = _basis_for(solved[side][2], rest['b2'], parent2, rest['p2'])
                per_bone.setdefault(first, []).append(basis0.to_quaternion())
                per_bone.setdefault(middle, []).append(basis1.to_quaternion())
                per_bone.setdefault(last, []).append(basis2.to_quaternion())
        print(f'[HAND-SPACING] 需要补偿 {active} 帧；批量写入 6 根臂骨旋转', flush=True)
        paths = _write_rotation_channels(rig, action, per_bone, frames,
                                         int(frame_start), int(frame_end))
        set_linear_for_paths(action, paths)
        update_action(action)
        # Sampled verification: writing the intended keys is not proof.
        corrected = [i for i, (f, tm, off, d, a) in enumerate(samples) if off.length > 1e-7]
        check = set(corrected)
        check.update({min(max(i - 1, 0), len(frames) - 1) for i in corrected})
        check.update({min(i + 1, len(frames) - 1) for i in corrected})
        check.update(range(0, len(frames), max(1, int(verify_step))))
        max_error = 0.
        max_rotation = 0.
        for index in sorted(check):
            f = frames[index]
            set_scene_frame(scene, f, layer)
            for side in ('L', 'R'):
                wrist = rig.pose.bones[CHAINS[side][2]]
                if wrist.rotation_mode != 'QUATERNION':
                    layer.update()
                max_error = max(max_error, (wrist.head - solved_all[index][side][2].translation
                                            ).length * target_scale)
                angle = wrist.matrix.to_quaternion().rotation_difference(
                    samples[index][1][side][2].to_quaternion()).angle
                max_rotation = max(max_rotation, min(angle, abs(2 * math.pi - angle)))
            if f in (1, 560, 570, 577, 585, 600):
                tm = samples[index][1]
                audit[str(f)] = {
                    'before_distance_m': (tm['L'][2].translation - tm['R'][2].translation).length * target_scale,
                    'after_distance_m': (rig.pose.bones[CHAINS['L'][2]].head
                                         - rig.pose.bones[CHAINS['R'][2]].head).length * target_scale,
                    'source_scaled_signed_m': samples[index][3] * target_scale,
                    'before_signed_m': samples[index][4] * target_scale,
                    'each_offset_m': samples[index][2].length * target_scale,
                }
    if max_error > .001 or math.degrees(max_rotation) > .1:
        raise RuntimeError('手臂骨架不支持当前补偿，请丢弃预览')
    FAST_STATS.update({'verified_frames': len(check), 'frames': len(frames)})
    return {'operation': 'hand_spacing_fk', 'status': 'applied', 'frames': len(frames),
            'compensated_frames': active, 'arm_scale': ratio,
            'max_each_hand_offset_m': max_move, 'reach_clamped_sides': clamped,
            'max_wrist_position_error_m': max_error,
            'max_wrist_rotation_error_deg': math.degrees(max_rotation),
            'frames_audit': audit, 'rotation_paths_written': sorted(paths),
            'finger_curves_written': 0, 'location_curves_written': 0,
            'scale_curves_written': 0, 'verified_frames': len(check),
            'fast_path': True}




def compensate_hand_spacing_fk(scene, source, rig, action, *, frame_start, frame_end,
                               strength=1.0):
    """Public entry point: batched matrix implementation.

    The reference implementation (per-bone convert_space + depsgraph updates) is
    kept as ``compensate_hand_spacing_fk_reference``; it is the oracle the fast
    path is checked against.
    """
    return compensate_hand_spacing_fk_fast(
        scene, source, rig, action, frame_start=frame_start, frame_end=frame_end,
        strength=strength)
