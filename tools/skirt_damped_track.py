"""裙摆 demo v2：有时序的弹簧阻尼目标点 + Damped Track。
Blender Text Editor 打开后 Alt+P；apply 预计算，remove 可撤销。
不使用刚体、不改原动作、不装回调。不含碰撞，修改人体动作后须重新 apply。
"""
import bpy
import json
import math
import time
from mathutils import Vector, Matrix

MODE = 'apply'                         # apply / remove / report
ARM_NAME = 'arue式重音テトver 2.01_arm'
START_FRAME = None                      # None = scene.frame_start
END_FRAME = None                        # None = scene.frame_end
FREQUENCY_HZ = {0: 2.8, 1: 2.3, 2: 1.9}
DAMPING_RATIO = 0.3                    # <1 有回摆；接近1少回摆
INERTIA_GAIN = 0.3                   # 平移加速度带来的裙摆偏转
MAX_SWING_DEG = 28.0
ACCEL_SMOOTH_SECONDS = 0.09
MAX_ANCHOR_ACCEL = 35.0                 # 保护动捕尖峰，m/s²
SUBSTEPS = 8
COLLECTION = 'MCD_SkirtSpring_Demo'
CONSTRAINT = 'MCD_SKIRT_SPRING_TRACK'
TAG = 'mcd_skirt_spring_demo_v2'
BACKUP = 'mcd_skirt_spring_previous_constraints'
OLD_NAMES = {'MCD_follow', 'MCD_gravity'}

# --------------------------- capsule collision (target-point push-out) ------
CAPSULE_ENABLE = True
# 候选名按当前 MMD 骨架填写；脚本会取第一个存在的名字。
# 你的模型若叫别的名字，只改这里，不动算法。
CAPSULE_BONES = [
    (('左足', 'left_thigh', 'thigh_ik.L'), 0.115),
    (('右足', 'right_thigh', 'thigh_ik.R'), 0.115),
    # (('左ひざ', 'left_shin', 'shin_ik.L'), 0.095),
    # (('右ひざ', 'right_shin', 'shin_ik.R'), 0.095),
]
CAPSULE_MARGIN = 0.012       # 胶囊外留出的裙摆间隙（米）
CAPSULE_ITERATIONS = 3        # 多个胶囊/角落重叠时的推出轮数
CAPSULE_REPORT_LIMIT = 12


def names_for(arm):
    names = [b.name for b in arm.pose.bones if b.name.startswith('スカート_')]
    def depth(name):
        n, b = 0, arm.data.bones[name].parent
        while b:
            n += 1
            b = b.parent
        return n
    if not names:
        raise RuntimeError('找不到裙子骨')
    return sorted(names, key=lambda n: (depth(n), n))


def _resolve_capsules(arm):
    """Return [(bone_name, radius)] for configured capsule bones."""
    out = []
    for candidates, radius in CAPSULE_BONES:
        name = next((n for n in candidates if n in arm.pose.bones), None)
        if name:
            out.append((name, float(radius)))
    return out


def _closest_on_segment(point, a, b):
    ab = b - a
    den = ab.length_squared
    t = 0.0 if den < 1e-12 else max(0.0, min(1.0, (point-a).dot(ab) / den))
    return a + ab * t


def _push_capsules(point, capsules, margin):
    """Push a target point outside all world-space bone capsules.

    Capsule = current bone segment + radius. At an exact centerline/zero-distance
    case, use a stable horizontal normal, so no NaN or random huge kick occurs.
    Returns (point, number_of_pushes, max_penetration_m).
    """
    p = point.copy()
    count = 0
    max_pen = 0.0
    for _ in range(max(1, int(CAPSULE_ITERATIONS))):
        moved = False
        for a, b, radius in capsules:
            q = _closest_on_segment(p, a, b)
            delta = p - q
            dist = delta.length
            limit = radius + float(margin)
            if dist >= limit:
                continue
            if dist < 1e-8:
                axis = (b-a).normalized() if (b-a).length > 1e-8 else Vector((0, 0, 1))
                n = axis.cross(Vector((0, 0, 1)))
                if n.length < 1e-8:
                    n = axis.cross(Vector((0, 1, 0)))
                n.normalize()
            else:
                n = delta / dist
            pen = limit - dist
            p += n * pen
            max_pen = max(max_pen, pen)
            count += 1
            moved = True
        if not moved:
            break
    return p, count, max_pen


def _world_capsules(arm, names):
    """Sample capsule bone segments from the current evaluated pose."""
    out = []
    for name, radius in _resolve_capsules(arm):
        pb = arm.pose.bones[name]
        a = arm.matrix_world @ pb.head
        b = arm.matrix_world @ pb.tail
        out.append((a, b, radius))
    return out


def backup_old(arm, names):
    rows = []
    for name in names:
        for c in arm.pose.bones[name].constraints:
            if c.name not in OLD_NAMES:
                continue
            d = dict(bone=name, name=c.name, type=c.type, mute=c.mute,
                     influence=c.influence, target=c.target.name if c.target else None)
            for attr in ('subtarget', 'owner_space', 'target_space', 'track_axis',
                         'mix_mode', 'use_x', 'use_y', 'use_z', 'invert_x',
                         'invert_y', 'invert_z', 'use_offset'):
                if hasattr(c, attr):
                    d[attr] = getattr(c, attr)
            rows.append(d)
    return rows


def cleanup(arm, names):
    for n in names:
        pb = arm.pose.bones[n]
        for c in list(pb.constraints):
            if c.name == CONSTRAINT:
                pb.constraints.remove(c)
    col = bpy.data.collections.get(COLLECTION)
    if col and col.get(TAG):
        for ob in list(col.objects):
            if not ob.get(TAG):
                continue
            ad = ob.animation_data
            act = ad.action if ad else None
            bpy.data.objects.remove(ob, do_unlink=True)
            if act and act.get(TAG) and act.users == 0:
                bpy.data.actions.remove(act)
        if not col.objects:
            bpy.data.collections.remove(col)


def remove(arm_name=ARM_NAME):
    arm = bpy.data.objects[arm_name]
    cleanup(arm, names_for(arm))
    if BACKUP in arm:
        for d in json.loads(arm[BACKUP]):
            pb = arm.pose.bones.get(d['bone'])
            if not pb or any(c.name == d['name'] for c in pb.constraints):
                continue
            c = pb.constraints.new(d['type'])
            c.name = d['name']
            c.target = bpy.data.objects.get(d['target']) if d['target'] else None
            for k, v in d.items():
                if k not in ('bone', 'name', 'type', 'target') and hasattr(c, k):
                    setattr(c, k, v)
        del arm[BACKUP]
    bpy.context.view_layer.update()
    return {'removed': True, 'restored_previous_constraints': True}


def cone(d, goal, limit):
    angle = math.acos(max(-1., min(1., d.dot(goal))))
    if angle <= limit:
        return d, False
    tangent = d - goal * d.dot(goal)
    if tangent.length < 1e-8:
        tangent = goal.orthogonal()
    tangent.normalize()
    return (goal * math.cos(limit) + tangent * math.sin(limit)).normalized(), True


def key_target(ob, points, start):
    act = bpy.data.actions.new('MCD_SpringTargets_' + ob.name)
    act[TAG] = True
    for axis in range(3):
        fc = act.fcurves.new(data_path='location', index=axis)
        fc.keyframe_points.add(len(points))
        co = []
        for i, p in enumerate(points):
            co.extend((start + i, float(p[axis])))
        fc.keyframe_points.foreach_set('co', co)
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
        fc.update()
    ob.animation_data_create().action = act


def run(overrides=None):
    cfg = dict(arm_name=ARM_NAME, start=START_FRAME, end=END_FRAME,
               frequency=FREQUENCY_HZ, damping=DAMPING_RATIO, inertia=INERTIA_GAIN,
               max_swing=MAX_SWING_DEG, accel_tau=ACCEL_SMOOTH_SECONDS,
               max_accel=MAX_ANCHOR_ACCEL, substeps=SUBSTEPS)
    cfg.update(overrides or {})
    scene = bpy.context.scene
    arm = bpy.data.objects[cfg['arm_name']]
    names = names_for(arm)
    a = scene.frame_start if cfg['start'] is None else int(cfg['start'])
    b = scene.frame_end if cfg['end'] is None else int(cfg['end'])
    dt = scene.render.fps_base / scene.render.fps
    h = dt / cfg['substeps']
    if b < a+2 or not 0 < cfg['damping'] <= 2 or not 0 < cfg['max_swing'] < 80:
        raise ValueError('参数范围错误')
    col = bpy.data.collections.get(COLLECTION)
    if col and not col.get(TAG):
        raise RuntimeError('同名 collection 已被其他内容占用')
    for n in names:
        freq = float(cfg['frequency'].get(int(n.split('_')[1]), 2.0))
        if freq <= 0 or 2 * math.pi * freq * h > .5:
            raise ValueError('频率过高；增加 SUBSTEPS')
    frame, subframe = scene.frame_current, scene.frame_subframe
    muted, base, capsule_frames = [], [], []
    started = time.time()
    try:
        # 采样时只排除本脚本的旧约束和已有弹簧；其他动画/约束原样保留。
        for n in names:
            for c in arm.pose.bones[n].constraints:
                if c.name in OLD_NAMES or c.name == CONSTRAINT:
                    muted.append((c, c.mute))
                    c.mute = True
        bpy.context.view_layer.update()
        for f in range(a, b+1):
            scene.frame_set(f)
            base.append({n: (arm.matrix_world @ arm.pose.bones[n].matrix).copy()
                         for n in names})
            capsule_frames.append(_world_capsules(arm, names) if CAPSULE_ENABLE else [])
            if (f-a) % 250 == 0:
                print('[skirt spring] sampled', f, '/', b, flush=True)
    finally:
        for c, mute in muted:
            c.mute = mute
        scene.frame_set(frame, subframe=subframe)

    states, targets, swings = {}, {n: [] for n in names}, []
    clamps = 0
    collision_pushes = 0
    max_penetration = 0.0
    collision_frames = set()
    limit = math.radians(cfg['max_swing'])
    alpha = 1 - math.exp(-dt / max(cfg['accel_tau'], 1e-6))
    for i, mats in enumerate(base):
        corrections = {}
        for n in names:
            pb = arm.pose.bones[n]
            parent = pb.parent.name if pb.parent else None
            inherited = corrections.get(parent, Matrix.Identity(4)) @ mats[n]
            anchor = inherited.translation.copy()
            goal = inherited.to_3x3().col[1].normalized()
            # 每根骨实际的 +Y（长度轴），不按世界Z猜正负，保留末级反向骨原型。
            length = max((inherited.to_3x3() @ Vector((0, pb.bone.length, 0))).length, .02)
            omega = 2 * math.pi * float(cfg['frequency'].get(int(n.split('_')[1]), 2.0))
            st = states.get(n)
            if st is None:
                st = states[n] = dict(d=goal.copy(), v=Vector((0, 0, 0)),
                                     anchor=anchor.copy(), av=Vector((0, 0, 0)),
                                     acc=Vector((0, 0, 0)), goal=goal.copy())
            av = (anchor-st['anchor'])/dt if i else Vector((0, 0, 0))
            acc = (av-st['av'])/dt if i > 1 else Vector((0, 0, 0))
            if acc.length > cfg['max_accel']:
                acc *= cfg['max_accel']/acc.length
            st['acc'] = st['acc'].lerp(acc, alpha)
            d, v = st['d'].copy(), st['v'].copy()
            for step in range(cfg['substeps']):
                g = st['goal'].lerp(goal, (step+1)/cfg['substeps']).normalized()
                force = omega*omega*(g-d) - (2*cfg['damping']*omega)*v
                inert = -st['acc']*(cfg['inertia']/length)
                force += inert-d*inert.dot(d)
                v += h*force
                v -= d*v.dot(d)
                d = (d+h*v).normalized()
                d, clipped = cone(d, g, limit)
                if clipped:
                    clamps += 1
                    tangent = (d-g*d.dot(g)).normalized()
                    v -= tangent*max(0., v.dot(tangent))
                v -= d*v.dot(d)
            q = goal.rotation_difference(d)
            desired = q.to_matrix().to_4x4() @ inherited
            desired.translation = anchor
            corrections[n] = desired @ mats[n].inverted_safe()
            target_point = anchor + d * length
            if CAPSULE_ENABLE and capsule_frames[i]:
                target_point, pushes, pen = _push_capsules(
                    target_point, capsule_frames[i], CAPSULE_MARGIN)
                if pushes:
                    collision_pushes += pushes
                    max_penetration = max(max_penetration, pen)
                    collision_frames.add(a + i)
            targets[n].append(target_point)
            swings.append(math.degrees(goal.angle(d)))
            st.update(d=d, v=v, anchor=anchor, av=av, goal=goal.copy())

    if BACKUP not in arm:
        arm[BACKUP] = json.dumps(backup_old(arm, names), ensure_ascii=False)
    cleanup(arm, names)
    for n in names:
        for c in list(arm.pose.bones[n].constraints):
            if c.name in OLD_NAMES:
                arm.pose.bones[n].constraints.remove(c)
    col = bpy.data.collections.new(COLLECTION)
    scene.collection.children.link(col)
    col[TAG] = True
    for n in names:
        ob = bpy.data.objects.new('MCD_SpringTarget_'+n, None)
        col.objects.link(ob)
        ob[TAG] = True
        ob.empty_display_type = 'PLAIN_AXES'
        ob.empty_display_size = .025
        ob.hide_render = True
        key_target(ob, targets[n], a)
        c = arm.pose.bones[n].constraints.new('DAMPED_TRACK')
        c.name = CONSTRAINT
        c.target = ob
        c.track_axis = 'TRACK_Y'
        c.influence = 1.
    col['settings'] = json.dumps(cfg, ensure_ascii=False)
    col['frame_range'] = [a, b]
    scene.frame_set(frame, subframe=subframe)
    bpy.context.view_layer.update()
    result = dict(bones=len(names), frames=[a, b],
                  max_swing_deg=round(max(swings), 3),
                  mean_swing_deg=round(sum(swings)/len(swings), 3),
                  clamped_substeps=clamps, elapsed_s=round(time.time()-started, 2),
                  temporal_spring=True, collision=bool(CAPSULE_ENABLE),
                  capsule_bones=[n for n, _ in _resolve_capsules(arm)],
                  capsule_pushes=collision_pushes,
                  capsule_max_penetration_m=round(max_penetration, 5),
                  capsule_frames=len(collision_frames))
    print('[skirt spring]', result, flush=True)
    return result


def report():
    col = bpy.data.collections.get(COLLECTION)
    return dict(targets=len(col.objects) if col else 0,
                settings=col.get('settings') if col else None)


if __name__ == '__main__':
    if MODE == 'remove':
        print(remove())
    elif MODE == 'report':
        print(report())
    else:
        run()
