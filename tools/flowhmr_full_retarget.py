"""FlowHMR 源骨架 -> ARP -> MMD/MMR 全自动重定向（生成 test.blend 23:08 那版所用的完整流程）。

流程（顺序不可随意调换）：
  1) unbind（若已绑定）—— 否则会沿用旧绑定与旧 rest 快照
  2) 记录原始 IK/FK 值
  3) MMD 控制器预处理（打开骨骼集合 + 设置 IK/FK；PoseCapture 可用则调它）
  4) frame0 = T-pose：**只清旋转，保留第 1 帧的位移/缩放**
     （清位移会让骨架半截沉到地板，重定向后模型飞在空中）
  5) build_bones_list + 导入 bmap
  6) 几何 rest 对齐（都在 redefine 之前）：
       上臂  Shoulder->Elbow   -> upper_arm_fk.*  rest 朝向
       前臂  Elbow->Wrist      -> forearm_fk.*    rest 朝向
       手腕  掌面坐标系         -> hand_fk.*
       手指  真实指节连线       -> f_*.01/02.*（近节、中节）
     全部使用 bone head 的真实关节连线，**绝不使用 FBX 的 +Y / tail**，
     因此**不调用 Copy Bone Rest**（它用 DAMPED_TRACK 跟 +Y，实测偏差 88~93°）。
  7) _redefine_rest_pose(CURRENT, preserve=True)  —— 把对齐后的姿态存为 ARP 参考
  8) _save_pose_rest
  9) arp.retarget
 10) 可选：还原 IK/FK（本脚本已把腿设为完全 IK=0.0，手臂 FK=1.0）
 11) 保存 + 写 JSON 报告

用法：
  A) Blender 里打开工程 -> Scripting -> 打开本文件 -> Run（BLEND 留空即用当前文件）
  B) 命令行：blender --background --python tools/flowhmr_full_retarget.py
     （此时把 BLEND 填成工程绝对路径）

注意：
  - 手指第三节没有第四个关节，脚本**不猜**它的滚转（猜过会出接近 180° 翻转）。
  - 若目标骨架已绑定，务必让脚本自己 unbind，不要手动跳过第 1 步。
"""
import bpy, sys, math, json
from mathutils import Matrix
from pathlib import Path

# ----------------------------- 参数 -----------------------------
# 强烈建议：先手动打开你要处理的工程，这里留空 —— 脚本就地处理当前文件。
# 若填了路径：当它和当前已打开的工程不同时，脚本会【替换整个工程】，
# 未保存的修改会丢失（Blender 的 open_mainfile 不询问）。因此默认禁止，需显式放行。
BLEND = ''                       # '' = 用当前已打开的工程（推荐）
ALLOW_DISCARD_UNSAVED = False    # True = 允许 BLEND 强制替换当前工程（会丢未保存内容）
SAVE_AS = ''                     # '' = 保存回当前工程；建议填新路径以保留原件
BACKUP_BEFORE_SAVE = True        # 保存前先把当前工程复制成 <名>_before_retarget.blend
BMAP = r'F:\mocap_ai_doctor\flowhmr_retarget\flowhmr_native_mmd_fix_A_foot_ik.bmap'
SOURCE = 'Reference'
TARGET = 'RIG-arue式重音テトver 2.01_arm'
START, END = 1, 300                         # 整段可改，例如 1, 3800
REPORT = r'F:\mocap_ai_doctor\agent_reports\full_retarget_report.json'

ALIGN_ARMS = True          # 上臂/前臂真实关节连线对齐
ALIGN_HANDS = True         # 腕掌面 + 手指近/中节对齐

# 腿用 IK 完全控制权：IK_FK 在四肢 parent 控制骨上，0 = IK，1 = FK（阈值 <0.5 判 IK）。
# 我们这张 bmap 的腿走 IK 通道：foot_ik.L <- L_Ankle (location=False, ik=True, pole=thigh_ik_target.L)
# 所以腿必须待在 IK 侧(0)，由 foot_ik + 极向量驱动。
# PoseCapture 给腿 0.95（FK 侧），那是给它自己那套 shin_fk<-L_Knee 的纯 FK 表用的；
# 套到我们这张表上会把 IK 驱动几乎屏蔽掉，所以这里显式覆盖为 0.0。
# 手臂是 FK 映射（upper_arm_fk<-L_Shoulder、forearm_fk<-L_Elbow），保持 FK(1.0)。
# 注意：本脚本先调 PoseCapture 预处理（它会把腿设成 0.95），随后立刻用下面的值覆盖。
RETARGET_ARM_IKFK = 1.0
RETARGET_LEG_IKFK = 0.0    # 腿完全 IK
RESTORE_ARM_IKFK = None    # 手臂本来就 1.0，无需还原
RESTORE_LEG_IKFK = None    # 腿已经是 0.0，无需还原

# IK_FK 属性在四肢 parent 控制骨上（实测），不在 shin_fk/upper_arm_fk 上
ARM_BONES = ('upper_arm_parent.L', 'upper_arm_parent.R')
LEG_BONES = ('thigh_parent.L', 'thigh_parent.R')
FINGERS = (('Index', 'f_index'), ('Middle', 'f_middle'), ('Ring', 'f_ring'),
           ('Pinky', 'f_pinky'), ('Thumb', 'thumb'))
# -----------------------------------------------------------------

TAG = '[FLOWHMR-RETARGET]'


def log(x):
    print(TAG, x, flush=True)


def unit(v):
    return v.normalized() if v.length > 1e-8 else None


def head(o, n, rest=False):
    b = o.data.bones[n] if rest else o.pose.bones[n]
    return o.matrix_world @ (b.head_local if rest else b.head)


def direction(o, a, b, rest=False):
    """真实关节连线（bone head）。绝不使用 FBX 的 tail / +Y。"""
    return unit(head(o, b, rest) - head(o, a, rest))


def hand_frame(o, names, rest=False):
    w, mid, idx, pk = (head(o, n, rest) for n in names)
    y = unit(mid - w)
    z = unit(y.cross(idx - pk))
    return Matrix((unit(y.cross(z)), y, z)).transposed()


def src_hand_names(side):
    return (side + '_Wrist', side + '_Middle1', side + '_Index1', side + '_Pinky1')


def tgt_hand_names(side):
    return ('hand_fk.' + side, 'f_middle.01.' + side,
            'f_index.01.' + side, 'f_pinky.01.' + side)


def tgt_bone_dir(tgt, name):
    b = tgt.data.bones[name]
    return unit(tgt.matrix_world.to_3x3() @ (b.tail_local - b.head_local))


def set_world_rot(pb, rot, obj):
    """在保持 head 位置的前提下，把骨的姿态旋转设为给定世界旋转。"""
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


def ensure_view(*objects):
    """取消集合排除/隐藏，保证对象能被选中（否则 select_set 会报 not in View Layer）。"""
    vl = bpy.context.view_layer
    stack = [vl.layer_collection]
    while stack:
        lc = stack.pop()
        lc.exclude = False
        lc.hide_viewport = False
        stack.extend(lc.children)
    for ob in objects:
        if ob is not None:
            ob.hide_viewport = False
    vl.update()


def activate(ob):
    """把 ob 设为唯一选中且活动的对象。ARP 算子 poll 要求 context.active_object 非空。"""
    ensure_view(ob)
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ob.hide_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.context.view_layer.update()


def ikfk_prop(pb):
    for k in list(pb.keys()):
        norm = k.replace(' ', '').replace('_', '').replace('-', '').replace('>', '').lower()
        if norm in ('ikfk', 'iktofk', 'ikfkblend'):
            return k
    return None


def read_ikfk(tgt):
    out = {}
    for name in ARM_BONES + LEG_BONES:
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
            log(f'IK_FK 跳过（无此骨）: {name}')
            continue
        k = ikfk_prop(pb)
        if k:
            pb[k] = value
            log(f'IK_FK: {name} -> {value}')


def prepare_frame0(src):
    """只清旋转；位移/缩放取 START 帧（清位移会让骨架沉到地板）。"""
    sc = bpy.context.scene
    act = src.animation_data.action
    for fc in list(act.fcurves):
        for kp in list(fc.keyframe_points):
            if abs(kp.co[0]) < 1e-6:
                fc.keyframe_points.remove(kp)
    sc.frame_set(START)
    bpy.context.view_layer.update()
    ref = {pb.name: (pb.location.copy(), pb.scale.copy()) for pb in src.pose.bones}
    object_world = src.matrix_world.copy()
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
    dev = max(max(abs(pb.location[i] - ref[pb.name][0][i]) for i in range(3))
              for pb in src.pose.bones)
    if dev > 1e-5:
        raise RuntimeError(f'frame0 位移未按参考帧保留（偏差 {dev:.2e}）')
    if max(abs(src.matrix_world[i][j] - object_world[i][j])
           for i in range(4) for j in range(4)) > 1e-5:
        raise RuntimeError('源骨架对象变换被改变，停止')
    log(f'frame0 = T-pose（只清旋转；位移/缩放保留自第 {START} 帧）')


def prep_controller(tgt):
    try:
        import gvhmr_pose_capture as pc
        pc._prepare_mmd_controller_before_remap(tgt, bpy.context, stage='full-retarget')
        log('PoseCapture 控制器预处理完成（骨骼集合 + IK/FK）')
        return
    except Exception as exc:
        log(f'PoseCapture 预处理不可用（{exc}），改用最小 IK/FK 设置')
    write_ikfk(tgt, ARM_BONES, RETARGET_ARM_IKFK)
    write_ikfk(tgt, LEG_BONES, RETARGET_LEG_IKFK)


def calibrate_arms(src, tgt):
    before, after = [], []
    for side in ('L', 'R'):
        for sa, sb, tb in ((side + '_Shoulder', side + '_Elbow', 'upper_arm_fk.' + side),
                           (side + '_Elbow', side + '_Wrist', 'forearm_fk.' + side)):
            td = tgt_bone_dir(tgt, tb)
            sd = direction(src, sa, sb)
            before.append({'side': side, 'bone': sa, 'target': tb,
                           'gap_deg': round(math.degrees(sd.angle(td)), 3)})
            set_world_rot(src.pose.bones[sa],
                          sd.rotation_difference(td).to_matrix().to_4x4(), src)
            after.append({'side': side, 'bone': sa, 'target': tb,
                          'gap_deg': round(math.degrees(direction(src, sa, sb).angle(td)), 4)})
        log(f'{side} 上臂/前臂已对齐')
    return before, after


def measure_finger_gaps(src, tgt):
    rows = []
    for side in ('L', 'R'):
        for fg, stem in FINGERS:
            for i in (1, 2):
                sd = direction(src, f'{side}_{fg}{i}', f'{side}_{fg}{i+1}')
                td = direction(tgt, f'{stem}.{i:02}.{side}', f'{stem}.{i+1:02}.{side}', True)
                rows.append({'side': side, 'finger': fg, 'segment': i,
                             'gap_deg': math.degrees(sd.angle(td))})
    return rows


def align_hands(src, tgt):
    for side in ('L', 'R'):
        S = hand_frame(src, src_hand_names(side), False)
        T = hand_frame(tgt, tgt_hand_names(side), True)
        set_world_rot(src.pose.bones[side + '_Wrist'], T @ S.inverted(), src)
        for fg, stem in FINGERS:
            for i in (1, 2):
                sd = direction(src, f'{side}_{fg}{i}', f'{side}_{fg}{i+1}')
                td = direction(tgt, f'{stem}.{i:02}.{side}', f'{stem}.{i+1:02}.{side}', True)
                set_world_rot(src.pose.bones[f'{side}_{fg}{i}'],
                              sd.rotation_difference(td).to_matrix().to_4x4(), src)
        log(f'{side} 腕掌面 + 手指近/中节已对齐（远节不猜）')


def main():
    cur = bpy.data.filepath
    if BLEND:
        want = str(Path(BLEND).resolve())
        if cur and Path(cur).resolve() == want:
            log(f'BLEND 与当前工程相同，就地处理（不重新载入，保留未保存修改）')
        else:
            if cur and bpy.data.is_dirty and not ALLOW_DISCARD_UNSAVED:
                raise RuntimeError(
                    f'当前工程 {cur!r} 有未保存修改，而 BLEND 指向 {BLEND!r}。\n'
                    f'  脚本不会替你丢弃它们。请先手动保存/另存，\n'
                    f'  或把 BLEND 留空，或确要丢弃时设 ALLOW_DISCARD_UNSAVED = True。')
            log(f'载入 BLEND = {BLEND}')
            bpy.ops.wm.open_mainfile(filepath=BLEND)
    if not bpy.data.filepath:
        raise RuntimeError('当前没有已打开的工程文件。请先手动打开工程，或设置 BLEND（并注意会替换当前工程）。')
    log(f'当前工程: {bpy.data.filepath}')
    sc = bpy.context.scene
    src = bpy.data.objects.get(SOURCE)
    tgt = bpy.data.objects.get(TARGET)
    if src is None or tgt is None:
        raise RuntimeError(f'找不到 {SOURCE!r} 或 {TARGET!r}')
    sc.source_rig, sc.target_rig = SOURCE, TARGET
    sc.frame_start, sc.frame_end = START, END
    log(f'源={SOURCE!r} 目标={TARGET!r} 帧={START}..{END}')

    if 'arp_retarget_bound' in tgt.keys():
        log('目标已绑定 -> 先 unbind（重建全新绑定与 rest 快照）')
        activate(tgt)
        bpy.ops.arp.retarget('EXEC_DEFAULT', unbind=True)
        log('unbind 完成')

    ikfk_backup = read_ikfk(tgt)                      # 必须在任何修改之前
    sc['mocap_doctor_ikfk_backup'] = json.dumps(ikfk_backup)
    log('原始 IK/FK: ' + json.dumps(ikfk_backup, ensure_ascii=False))

    prep_controller(tgt)
    write_ikfk(tgt, ARM_BONES, RETARGET_ARM_IKFK)
    write_ikfk(tgt, LEG_BONES, RETARGET_LEG_IKFK)

    prepare_frame0(src)

    activate(src)
    bpy.ops.arp.build_bones_list()
    bpy.ops.arp.import_config(filepath=BMAP, clear_current=True)
    log(f'bones_map_v2 = {len(sc.bones_map_v2)}')
    if len(sc.bones_map_v2) == 0:
        raise RuntimeError('bmap 未载入')

    arms_before, arms_after = ([], [])
    if ALIGN_ARMS:
        arms_before, arms_after = calibrate_arms(src, tgt)
        log('上臂/前臂 gap: before=%s after=%s'
            % ([r['gap_deg'] for r in arms_before], [r['gap_deg'] for r in arms_after]))

    finger_before = finger_after = []
    if ALIGN_HANDS:
        finger_before = measure_finger_gaps(src, tgt)
        log('手指 gap before 最大 = %.3f°'
            % max(r['gap_deg'] for r in finger_before))
        align_hands(src, tgt)
        finger_after = measure_finger_gaps(src, tgt)
        log('手指 gap after  最大 = %.6f°'
            % max(r['gap_deg'] for r in finger_after))

    arp = find_arp()

    class Op:
        preserve = True
        rest_pose = 'CURRENT'
        is_arp_armature = False

    op = Op()
    if 'arp_updated' in src.data.keys():
        op.is_arp_armature = True
    arp._redefine_rest_pose(op, bpy.context)
    log('_redefine_rest_pose(CURRENT, preserve=True) 完成（未调用 Copy Bone Rest）')
    arp._save_pose_rest(op)
    log('_save_pose_rest 完成')

    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    activate(tgt)
    bpy.ops.arp.retarget('EXEC_DEFAULT', frame_start=START, frame_end=END)
    log(f'arp.retarget {START}..{END} 完成')

    if RESTORE_LEG_IKFK is not None:
        write_ikfk(tgt, LEG_BONES, RESTORE_LEG_IKFK)
    if RESTORE_ARM_IKFK is not None:
        write_ikfk(tgt, ARM_BONES, RESTORE_ARM_IKFK)

    out = SAVE_AS or bpy.data.filepath
    if not out:
        raise RuntimeError('没有可保存的路径；请设置 SAVE_AS')
    if BACKUP_BEFORE_SAVE and bpy.data.filepath:
        try:
            import shutil
            bak = str(Path(bpy.data.filepath).with_name(
                Path(bpy.data.filepath).stem + '_before_retarget.blend'))
            if not Path(bak).exists():
                shutil.copy2(bpy.data.filepath, bak)
                log('已备份原工程: ' + bak)
            else:
                log('备份已存在，未覆盖: ' + bak)
        except Exception as exc:
            log(f'备份失败（继续保存）: {exc}')
    bpy.ops.wm.save_as_mainfile(filepath=out)
    log('SAVED ' + out)

    try:
        Path(REPORT).write_text(json.dumps({
            'blend': out, 'source': SOURCE, 'target': TARGET,
            'frames': [START, END], 'bones_map_v2': len(sc.bones_map_v2),
            'copy_bone_rest_used': False,
            'ikfk_original': ikfk_backup,
            'ikfk_set_for_retarget': {'arms': RETARGET_ARM_IKFK, 'legs': RETARGET_LEG_IKFK},
            'ikfk_restored': {'arms': RESTORE_ARM_IKFK, 'legs': RESTORE_LEG_IKFK},
            'arm_before': arms_before, 'arm_after': arms_after,
            'finger_before_max_gap_deg': max((r['gap_deg'] for r in finger_before), default=None),
            'finger_after_max_gap_deg': max((r['gap_deg'] for r in finger_after), default=None),
            'finger_before': finger_before, 'finger_after': finger_after,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        log('报告: ' + REPORT)
    except Exception as exc:
        log(f'报告写入失败（不影响结果）: {exc}')


if __name__ == '__main__':
    main()
