"""阶段1：准备到 Redefine Rest Pose 为止，然后停下让你手动摆姿态。

用法：Blender 里打开你的工程 → Scripting → 打开本文件 → Run Script。
      或在命令行：blender --background --python 本文件.py

做完本阶段后：
  - 源骨架的 action 被摘除，姿态冻结，所以你在视口里怎么摆都不会被关键帧覆盖；
  - 这时手动修手/腕/手臂的 rest 姿态；
  - 满意后运行阶段2（retarget_stage2_finish.py）。

注意：本阶段结束后【不要】拖时间轴、不要重新指定 action。
"""
import bpy, sys, math, json
from mathutils import Matrix
from pathlib import Path

# ---------------- 参数 ----------------
BLEND = ''                 # '' = 用当前已打开的工程；否则填 .blend 绝对路径
SAVE_AS = ''               # '' = 另存为 <工程名>_stage1_rest.blend
BMAP = r'F:\mocap_ai_doctor\flowhmr_retarget\flowhmr_native_mmd_fix_A_foot_ik.bmap'
SOURCE = 'Reference'
TARGET = 'RIG-arue式重音テトver 2.01_arm'
FRAME_START = 1
FRAME_END = 300
AUTO_ALIGN = True          # True = 先按真实关节连线自动对齐手臂+腕+手指（给你一个起点）
LEG_FK = 0.95              # 与 PoseCapture 一致；None = 不动这个属性
ARM_FK = 1.0               # 与 PoseCapture 一致；None = 不动

FINGERS = (('Index', 'f_index'), ('Middle', 'f_middle'), ('Ring', 'f_ring'),
           ('Pinky', 'f_pinky'), ('Thumb', 'thumb'))
# 实测：IK_FK 属性在四肢 parent 控制骨上，不在 shin_fk/upper_arm_fk 上
IKFK_BONES = ('upper_arm_parent.L', 'upper_arm_parent.R',
              'thigh_parent.L', 'thigh_parent.R')
ARM_BONES = ('upper_arm_parent.L', 'upper_arm_parent.R')
LEG_BONES = ('thigh_parent.L', 'thigh_parent.R')


def log(x):
    print('[STAGE1]', x, flush=True)


def unit(v):
    return v.normalized() if v.length > 1e-8 else None


def head(o, n, rest=False):
    b = o.data.bones[n] if rest else o.pose.bones[n]
    return o.matrix_world @ (b.head_local if rest else b.head)


def direction(o, a, b, rest=False):
    """真实关节连线（bone head），不用 FBX 的 +Y / tail。"""
    return unit(head(o, b, rest) - head(o, a, rest))


def hand_frame(o, names, rest=False):
    w, mid, idx, pk = (head(o, n, rest) for n in names)
    y = unit(mid - w)
    z = unit(y.cross(idx - pk))
    return Matrix((unit(y.cross(z)), y, z)).transposed()


def src_hand(side):
    return (side + '_Wrist', side + '_Middle1', side + '_Index1', side + '_Pinky1')


def tgt_hand(side):
    return ('hand_fk.' + side, 'f_middle.01.' + side,
            'f_index.01.' + side, 'f_pinky.01.' + side)


def tgt_bone_dir(tgt, name):
    b = tgt.data.bones[name]
    return unit(tgt.matrix_world.to_3x3() @ (b.tail_local - b.head_local))


def set_world_rot(pb, rot, obj):
    R = rot.to_4x4() if len(rot) == 3 else rot
    M = obj.matrix_world @ pb.matrix
    M2 = R @ M
    M2.translation = M.translation
    pb.matrix = obj.matrix_world.inverted() @ M2
    bpy.context.view_layer.update()


def find_arp():
    for n, m in list(sys.modules.items()):
        if 'auto_rig_pro' in n and hasattr(m, '_redefine_rest_pose'):
            return m
    raise RuntimeError('找不到 ARP 的 auto_rig_remap 模块；确认 Auto-Rig Pro 已启用')


def ikfk_prop(pb):
    for k in list(pb.keys()):
        norm = k.replace(' ', '').replace('_', '').replace('-', '').replace('>', '').lower()
        if norm in ('ikfk', 'iktofk', 'ikfkblend'):
            return k
    return None


def read_ikfk(tgt):
    """记录原始 IK/FK 值，供阶段2精确还原。必须在任何修改之前调用。"""
    out = {}
    for name in IKFK_BONES:
        pb = tgt.pose.bones.get(name)
        if not pb:
            continue
        k = ikfk_prop(pb)
        if k:
            out[name] = {'key': k, 'value': float(pb[k])}
    return out


def write_ikfk(tgt, names, value):
    if value is None:
        return
    for name in names:
        pb = tgt.pose.bones.get(name)
        if not pb:
            continue
        k = ikfk_prop(pb)
        if k:
            pb[k] = value
            log(f'IK_FK: {name} -> {value}')


def prepare_frame0(src):
    """只清旋转，保留动作第1帧的位移/缩放（清位移会让骨架沉到地板）。"""
    sc = bpy.context.scene
    act = src.animation_data.action
    for fc in list(act.fcurves):
        for kp in list(fc.keyframe_points):
            if abs(kp.co[0]) < 1e-6:
                fc.keyframe_points.remove(kp)
    sc.frame_set(FRAME_START)
    bpy.context.view_layer.update()
    ref = {pb.name: (pb.location.copy(), pb.scale.copy()) for pb in src.pose.bones}
    sc.frame_set(0)
    for pb in src.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.location, pb.scale = ref[pb.name]
        pb.keyframe_insert('location', frame=0)
        pb.keyframe_insert('rotation_quaternion', frame=0)
        pb.keyframe_insert('scale', frame=0)
    sc.frame_set(0)
    bpy.context.view_layer.update()
    log('frame0 = T-pose（仅清旋转，位移/缩放保留）')


def auto_align(src, tgt):
    """按真实关节连线对齐：上臂、前臂、腕掌面、手指近/中节。"""
    for side in ('L', 'R'):
        for sa, sb, tb in ((side + '_Shoulder', side + '_Elbow', 'upper_arm_fk.' + side),
                           (side + '_Elbow', side + '_Wrist', 'forearm_fk.' + side)):
            td = tgt_bone_dir(tgt, tb)
            sd = direction(src, sa, sb)
            set_world_rot(src.pose.bones[sa],
                          sd.rotation_difference(td).to_matrix().to_4x4(), src)
            log(f'{sa}: {math.degrees(sd.angle(td)):.2f}° -> '
                f'{math.degrees(direction(src, sa, sb).angle(td)):.3f}°')
        S = hand_frame(src, src_hand(side), False)
        T = hand_frame(tgt, tgt_hand(side), True)
        set_world_rot(src.pose.bones[side + '_Wrist'], T @ S.inverted(), src)
        for fg, stem in FINGERS:
            for i in (1, 2):
                sd = direction(src, f'{side}_{fg}{i}', f'{side}_{fg}{i+1}')
                td = direction(tgt, f'{stem}.{i:02}.{side}', f'{stem}.{i+1:02}.{side}', True)
                set_world_rot(src.pose.bones[f'{side}_{fg}{i}'],
                              sd.rotation_difference(td).to_matrix().to_4x4(), src)
        log(f'{side}: 腕掌面 + 手指近/中节已对齐')


def main():
    if BLEND:
        bpy.ops.wm.open_mainfile(filepath=BLEND)
    sc = bpy.context.scene
    src = bpy.data.objects.get(SOURCE)
    tgt = bpy.data.objects.get(TARGET)
    if src is None or tgt is None:
        raise RuntimeError(f'找不到 {SOURCE!r} 或 {TARGET!r}')
    sc.source_rig, sc.target_rig = SOURCE, TARGET
    sc.frame_start, sc.frame_end = FRAME_START, FRAME_END

    if 'arp_retarget_bound' in tgt.keys():
        log('目标已绑定 -> 先 unbind')
        bpy.ops.arp.retarget('EXEC_DEFAULT', unbind=True)

    # 先记录原始 IK/FK（必须在 PoseCapture 预处理改动之前读）
    backup = read_ikfk(tgt)
    sc['mocap_doctor_ikfk_backup'] = json.dumps(backup)
    log('原始 IK/FK（改动前）: ' + json.dumps(backup, ensure_ascii=False))

    # 打开 MMD 骨骼集合（方便你之后在视口里选中手指/手臂控制骨）
    try:
        import gvhmr_pose_capture as pc
        pc._prepare_mmd_controller_before_remap(tgt, bpy.context, stage='stage1')
    except Exception as exc:
        log(f'PoseCapture 预处理未调用：{exc}')

    write_ikfk(tgt, ARM_BONES, ARM_FK)
    write_ikfk(tgt, LEG_BONES, LEG_FK)

    prepare_frame0(src)

    bpy.context.view_layer.objects.active = src
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    src.hide_set(False)
    src.select_set(True)
    bpy.ops.arp.build_bones_list()
    bpy.ops.arp.import_config(filepath=BMAP, clear_current=True)
    log(f'bones_map_v2 = {len(sc.bones_map_v2)}')

    if AUTO_ALIGN:
        auto_align(src, tgt)

    arp = find_arp()

    class Op:
        preserve = True
        rest_pose = 'CURRENT'
        is_arp_armature = False

    op = Op()
    if 'arp_updated' in src.data.keys():
        op.is_arp_armature = True
    arp._redefine_rest_pose(op, bpy.context)
    log('_redefine_rest_pose(CURRENT, preserve=True) 完成 -> 源 action 已摘除，姿态冻结')

    target = SAVE_AS or (str(Path(bpy.data.filepath).with_name(
        Path(bpy.data.filepath).stem + '_stage1_rest.blend')) if bpy.data.filepath else '')
    if target:
        bpy.ops.wm.save_as_mainfile(filepath=target)
        log('SAVED ' + target)
    print("""
============================================================
【停在这里，现在手动摆姿态】
  1. 选中源骨架 Reference -> Pose Mode
  2. 摆正手/腕/手臂的 rest 姿态（Alt+R 只清选中骨的旋转）
  3. 不要拖时间轴，不要重新指定 action
  4. 摆好后运行阶段2：retarget_stage2_finish.py
============================================================
""")


if __name__ == '__main__':
    main()
