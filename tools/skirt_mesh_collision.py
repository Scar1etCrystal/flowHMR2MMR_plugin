"""裙摆 demo v3：弹簧阻尼目标点 + Damped Track + **真实 mesh 碰撞体**。

和 v2 的区别：
  * 不再用「骨段 + 半径」估胶囊，改用你摆好、绑到大腿上的 mesh（默认 Mball*）：
    在网格本地空间建一次 BVH 树，每帧把目标点变换到本地空间做最近点查询，
    再沿表面法线推到外面。所以你摆成什么形状，碰撞就是什么形状。
  * 多了一个 SAG_GAIN：把每根裙骨的静止目标往「正下方」偏，
    用来解决 v2 的灯笼感（原型往外翻，弹簧只在原型附近晃）。

不改骨骼静止姿态、不改原动作、不用刚体、不装回调。apply / remove 可逆。
修改人体动作后需要重新 apply（目标点是打好的关键帧）。
"""

import bpy
import json
import math
import time
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

# ----------------------------------------------------------------- CONFIG
MODE = 'apply'                  # apply / remove / report
ARM_NAME = 'arue式重音テトver 2.01_arm'
START_FRAME = None              # None = scene.frame_start
END_FRAME = None                # None = scene.frame_end

FREQUENCY_HZ = {0: 2.8, 1: 2.3, 2: 1.9}
DAMPING_RATIO = 0.48
INERTIA_GAIN = 0.22
MAX_SWING_DEG = 28.0
ACCEL_SMOOTH_SECONDS = 0.09
MAX_ANCHOR_ACCEL = 35.0
SUBSTEPS = 8

# 往下垂多少：0 = 保持原型（灯笼感），0.25 ≈ 明显收拢，0.5 很塌
SAG_GAIN = 0.25

# 碰撞体：留空 = 自动找名字以这些前缀开头的网格（你绑在腿上的 Mball*）
COLLIDER_PREFIXES = ('Mball',)
COLLIDER_MESHES = ()            # 想只用某几个就写具体名字，例如 ('Mball.001',)
COLLIDER_MARGIN = 0.012         # 目标点离碰撞体表面至少留这么多米
COLLIDER_ITERATIONS = 3         # 多个碰撞体互相顶住时反复推
MAX_COLLISION_PUSH = 0.20       # 单次最大推出；超过时分步推出，不再直接丢弃
RAY_EPSILON = 1e-5
RAY_MAX_HITS = 64

COLLECTION = 'MCD_SkirtSpring_Demo'
CONSTRAINT = 'MCD_SKIRT_SPRING_TRACK'
TAG = 'mcd_skirt_spring_demo_v3'
BACKUP = 'mcd_skirt_spring_previous_constraints'
OLD_NAMES = {'MCD_follow', 'MCD_gravity', 'MCD_SKIRT_SPRING_TRACK'}
# ------------------------------------------------------------------------


def log(*a):
    print('[skirt v3]', *a)


# --------------------------------------------------------------- colliders

def collider_objects():
    if COLLIDER_MESHES:
        out = []
        for n in COLLIDER_MESHES:
            ob = bpy.data.objects.get(n)
            if ob is None or ob.type != 'MESH':
                raise RuntimeError(f'碰撞体 {n!r} 不存在或不是网格')
            out.append(ob)
        return out
    out = [o for o in bpy.data.objects if o.type == 'MESH'
           and any(o.name.startswith(p) for p in COLLIDER_PREFIXES)]
    if not out:
        raise RuntimeError('没找到碰撞体网格（名字前缀 '
                           + str(COLLIDER_PREFIXES) + '）—— 请填 COLLIDER_MESHES')
    return sorted(out, key=lambda o: o.name)


def build_colliders():
    """每个碰撞体建一次本地空间 BVH（网格无修改器，故只建一次即可）。"""
    out = []
    for ob in collider_objects():
        mesh = ob.data
        verts = [v.co.copy() for v in mesh.vertices]
        polys = [tuple(p.vertices) for p in mesh.polygons]
        bvh = BVHTree.FromPolygons(verts, polys, all_triangles=False)
        parent = ob.parent_bone if ob.parent_type == 'BONE' else (
            ob.parent.name if ob.parent else None)
        out.append(dict(obj=ob, bvh=bvh, parent=parent,
                        verts=len(verts), polys=len(polys)))
        log(f'碰撞体 {ob.name}：{len(verts)} 顶点 / {len(polys)} 面，父级 {parent}')
    return out


def _ray_inside(bvh, point, direction):
    """闭合网格射线奇偶 inside test。三个轴向多数表决，减少切边误判。"""
    d = direction.normalized()
    votes = []
    for axis in (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))):
        # 轻微偏移，避免射线恰好穿过三角形边/顶点。
        o = point + axis * (RAY_EPSILON * 17.0)
        count = 0
        for _ in range(RAY_MAX_HITS):
            hit = bvh.ray_cast(o, axis, 1000.0)
            loc = hit[0]
            if loc is None:
                break
            count += 1
            o = loc + axis * (RAY_EPSILON * 101.0)
        votes.append((count & 1) == 1)
    return sum(votes) >= 2


def _push_meshes(point_world, frame_colliders, margin, iterations):
    """把目标点推到 Mball 闭合网格外；拒绝异常深穿透以保护弹簧目标。"""
    p = point_world.copy()
    pushes = 0
    inside_hits = 0
    deep_rejects = 0
    max_pen = 0.0
    for _ in range(max(1, int(iterations))):
        moved = False
        for mw, mw_inv, bvh in frame_colliders:
            local = mw_inv @ p
            loc, nrm, _idx, dist = bvh.find_nearest(local)
            if loc is None:
                continue
            inside = _ray_inside(bvh, local, Vector((1, 0, 0)))
            if inside:
                inside_hits += 1
                max_pen = max(max_pen, dist)
                # 深穿透也不直接丢弃：从当前点沿表面法线分步推出，
                # 每轮最多 MAX_COLLISION_PUSH，避免一次把弹簧目标弹飞。
                step = min(dist + margin, MAX_COLLISION_PUSH)
                p = p + mw.to_3x3() @ (nrm * step)
                pushes += 1
                moved = True
                continue
            if dist >= margin:
                continue
            max_pen = max(max_pen, margin - dist)
            p = mw @ (loc + nrm * margin)
            pushes += 1
            moved = True
        if not moved:
            break
    return p, pushes, max_pen, inside_hits, deep_rejects


# ------------------------------------------------------------------ spring

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


def backup_old(arm, names):
    rows = []
    for name in names:
        for c in arm.pose.bones[name].constraints:
            if c.name not in OLD_NAMES:
                continue
            d = dict(bone=name, name=c.name, type=c.type, mute=c.mute,
                     influence=c.influence,
                     target=c.target.name if c.target else None)
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


# --------------------------------------------------------------------- run

def run(overrides=None):
    cfg = dict(arm_name=ARM_NAME, start=START_FRAME, end=END_FRAME,
               frequency=FREQUENCY_HZ, damping=DAMPING_RATIO, inertia=INERTIA_GAIN,
               max_swing=MAX_SWING_DEG, accel_tau=ACCEL_SMOOTH_SECONDS,
               max_accel=MAX_ANCHOR_ACCEL, substeps=SUBSTEPS,
               sag=SAG_GAIN, margin=COLLIDER_MARGIN)
    cfg.update(overrides or {})
    scene = bpy.context.scene
    arm = bpy.data.objects[cfg['arm_name']]
    names = names_for(arm)
    a = scene.frame_start if cfg['start'] is None else int(cfg['start'])
    b = scene.frame_end if cfg['end'] is None else int(cfg['end'])
    dt = scene.render.fps_base / scene.render.fps
    h = dt / cfg['substeps']
    if b < a + 2 or not 0 < cfg['damping'] <= 2 or not 0 < cfg['max_swing'] < 80:
        raise ValueError('参数范围错误')
    if not 0.0 <= cfg['sag'] < 0.95:
        raise ValueError('SAG_GAIN 要在 0~0.95 之间')
    col = bpy.data.collections.get(COLLECTION)
    if col and col.get(TAG):
        # v3 重跑必须先拆掉旧 targets/constraints，否则 Blender 会创建 .001，
        # 视口可能仍看着旧目标，造成"碰撞没生效"的假象。
        cleanup(arm, names)
        col = None
    elif col and not col.get(TAG):
        raise RuntimeError('同名 collection 已被其他内容占用')
    for n in names:
        freq = float(cfg['frequency'].get(int(n.split('_')[1]), 2.0))
        if freq <= 0 or 2 * math.pi * freq * h > .5:
            raise ValueError('频率过高；增加 SUBSTEPS')
    colliders = build_colliders()

    frame, subframe = scene.frame_current, scene.frame_subframe
    muted, base, mats_per_frame = [], [], []
    started = time.time()
    try:
        for n in names:
            for c in arm.pose.bones[n].constraints:
                if c.name in OLD_NAMES:
                    muted.append((c, c.mute))
                    c.mute = True
        bpy.context.view_layer.update()
        for f in range(a, b + 1):
            scene.frame_set(f)
            bpy.context.view_layer.update()          # 骨骼父级物体要刷新
            base.append({n: (arm.matrix_world @ arm.pose.bones[n].matrix).copy()
                         for n in names})
            mats_per_frame.append({c['obj'].name: c['obj'].matrix_world.copy()
                                   for c in colliders})
            if (f - a) % 250 == 0:
                log('sampled', f, '/', b)
    finally:
        for c, mute in muted:
            c.mute = mute
        scene.frame_set(frame, subframe=subframe)

    down = Vector((0.0, 0.0, -1.0))
    # 塌陷方向的符号：按每根骨的**静止朝向**定。
    # 末级裙骨常常是"头在下、尾在上"的反向骨 —— 一律往世界下方偏会让它翻 180°，
    # 网格直接翻掉。反向骨要往世界"上"偏，才是它自己语义上的"垂下来"。
    sag_sign = {}
    for n in names:
        db = arm.data.bones[n]
        sag_sign[n] = 1.0 if db.tail_local.z < db.head_local.z else -1.0
    states, targets, swings = {}, {n: [] for n in names}, []
    clamps = pushes = inside_hits = deep_rejects = 0
    max_pen = 0.0
    limit = math.radians(cfg['max_swing'])
    alpha = 1 - math.exp(-dt / max(cfg['accel_tau'], 1e-6))
    for i, mats in enumerate(base):
        # 当帧碰撞体矩阵（求解循环里没有 frame_set，必须用采样值）
        frame_colliders = []
        for c in colliders:
            mw = mats_per_frame[i].get(c['obj'].name)
            if mw is None:
                continue
            frame_colliders.append((mw, mw.inverted_safe(), c['bvh']))
        corrections = {}
        for n in names:
            pb = arm.pose.bones[n]
            parent = pb.parent.name if pb.parent else None
            inherited = corrections.get(parent, Matrix.Identity(4)) @ mats[n]
            anchor = inherited.translation.copy()
            goal = inherited.to_3x3().col[1].normalized()
            if cfg['sag'] > 0:                       # 往该骨自己的"下方"偏 => 塌下去
                sag_dir = down * sag_sign[n]
                goal = (goal * (1.0 - cfg['sag']) + sag_dir * cfg['sag']).normalized()
            length = max((inherited.to_3x3()
                          @ Vector((0, pb.bone.length, 0))).length, .02)
            omega = 2 * math.pi * float(cfg['frequency'].get(int(n.split('_')[1]), 2.0))
            st = states.get(n)
            if st is None:
                st = states[n] = dict(d=goal.copy(), v=Vector((0, 0, 0)),
                                      anchor=anchor.copy(), av=Vector((0, 0, 0)),
                                      acc=Vector((0, 0, 0)), goal=goal.copy())
            av = (anchor - st['anchor']) / dt if i else Vector((0, 0, 0))
            acc = (av - st['av']) / dt if i > 1 else Vector((0, 0, 0))
            if acc.length > cfg['max_accel']:
                acc *= cfg['max_accel'] / acc.length
            st['acc'] = st['acc'].lerp(acc, alpha)
            d, v = st['d'].copy(), st['v'].copy()
            for step in range(cfg['substeps']):
                g = st['goal'].lerp(goal, (step + 1) / cfg['substeps']).normalized()
                force = omega * omega * (g - d) - (2 * cfg['damping'] * omega) * v
                inert = -st['acc'] * (cfg['inertia'] / length)
                force += inert - d * inert.dot(d)
                v += h * force
                v -= d * v.dot(d)
                d = (d + h * v).normalized()
                d, clipped = cone(d, g, limit)
                if clipped:
                    clamps += 1
                    tangent = (d - g * d.dot(g)).normalized()
                    v -= tangent * max(0., v.dot(tangent))
                v -= d * v.dot(d)
            q = goal.rotation_difference(d)
            desired = q.to_matrix().to_4x4() @ inherited
            desired.translation = anchor
            corrections[n] = desired @ mats[n].inverted_safe()
            point = anchor + d * length
            p2, npush, pen, inside, rejected = _push_meshes(
                point, frame_colliders, cfg['margin'], COLLIDER_ITERATIONS)
            if npush:
                pushes += npush
                inside_hits += inside
                max_pen = max(max_pen, pen)
            deep_rejects += rejected
            if npush:
                point = p2
            targets[n].append(point)
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
        ob = bpy.data.objects.new('MCD_SpringTarget_' + n, None)
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
    col['collision_stats'] = json.dumps({
        'colliders': [c['obj'].name for c in colliders],
        'pushes': collision_pushes if 'collision_pushes' in locals() else pushes,
        'inside_hits': inside_hits,
        'max_penetration_m': max_pen,
        'deep_rejects': deep_rejects,
        'margin_m': float(cfg['margin']),
        'frames': [a, b],
    }, ensure_ascii=False)
    scene.frame_set(frame, subframe=subframe)
    bpy.context.view_layer.update()
    result = dict(bones=len(names), frames=[a, b],
                  max_swing_deg=round(max(swings), 3),
                  mean_swing_deg=round(sum(swings) / len(swings), 3),
                  sag=cfg['sag'], clamped_substeps=clamps,
                  colliders=[c['obj'].name for c in colliders],
                  collider_verts=sum(c['verts'] for c in colliders),
                  collision_pushes=pushes, inside_hits=inside_hits,
                  deep_rejects=deep_rejects,
                  max_penetration_m=round(max_pen, 5),
                  elapsed_s=round(time.time() - started, 2))
    log('完成：', json.dumps(result, ensure_ascii=False))
    return result


def report():
    cs = build_colliders()
    col = bpy.data.collections.get(COLLECTION)
    return dict(colliders=[(c['obj'].name, c['verts'], c['polys'], c['parent'])
                           for c in cs],
                targets=len(col.objects) if col else 0,
                settings=col.get('settings') if col else None,
                collision_stats=col.get('collision_stats') if col else None)


if __name__ == '__main__':
    if MODE == 'remove':
        print(remove())
    elif MODE == 'report':
        print(json.dumps(report(), ensure_ascii=False, default=str))
    else:
        print(json.dumps(run(), ensure_ascii=False))
