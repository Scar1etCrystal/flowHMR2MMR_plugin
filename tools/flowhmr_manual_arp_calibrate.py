"""FlowHMR 原生骨架：真实关节连线 rest 校准 + ARP 短段重定向。

请在干净 execution_native_bmap_short_1_300.blend 或官方 execution.blend 副本上运行。
不使用 ARP copy_bone_rest：它跟踪源骨 +Y，而 FlowHMR FBX +Y 并非肢体方向。
用 Shoulder->Elbow、Elbow->Wrist 的真实连线校准上臂/前臂，
保留其他源骨绝对参考矩阵，再保存 ARP rest snapshot 并烘焙。
默认执行 1..300；不自动保存。已在干净场景真实烘焙 1..100 验证。
"""
import bpy
import sys
from pathlib import Path

# ------------------------- 可改参数 -------------------------
SOURCE = 'Reference'
TARGET = 'RIG-arue式重音テトver 2.01_arm'
BMAP = r'F:\mocap_ai_doctor\flowhmr_retarget\flowhmr_native_mmd_fix_A_foot_ik.bmap'
FRAME_START = 1
FRAME_END = 300
DO_RETARGET = True       # False = 只做 rest 校准，不跑 retarget


def log(s):
    print('[FlowHMR ARP] ' + str(s))


def walk_layer(lc, out):
    out.append(lc)
    for c in lc.children:
        walk_layer(c, out)


def ensure_view(*objects):
    vl = bpy.context.view_layer
    all_lc = []
    walk_layer(vl.layer_collection, all_lc)
    for lc in all_lc:
        lc.exclude = False
        lc.hide_viewport = False
    for ob in objects:
        if ob is None:
            continue
        ob.hide_viewport = False
    vl.update()


def select_object(ob):
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    ensure_view(ob)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ob.hide_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.context.view_layer.update()


def find_arp_remap_module():
    for name, mod in list(sys.modules.items()):
        if 'auto_rig_pro' in name and hasattr(mod, '_redefine_rest_pose'):
            return mod
    raise RuntimeError('找不到 ARP auto_rig_remap 模块；确认 ARP 已启用')


def bake_frame0_rest(src):
    """仅清旋转；位移/缩放取 FRAME_START，避免旧0帧清零值污染身高。

    FlowHMR 骨盆 bind 原点不在地板上。location 清零会丢掉站立高度。
    参考帧默认为动作第1帧；对象matrix_world保持不变。
    """
    scn = bpy.context.scene
    act = src.animation_data.action
    select_object(src)
    # 先取真实动作帧，不能取旧脚本已经清过的 frame 0。
    scn.frame_set(FRAME_START)
    bpy.context.view_layer.update()
    reference = {pb.name: (pb.location.copy(), pb.scale.copy()) for pb in src.pose.bones}
    object_world = src.matrix_world.copy()
    root = next((pb for pb in src.pose.bones if pb.parent is None), None)
    root_z_before = (src.matrix_world @ root.head).z if root else None
    for fc in list(act.fcurves):
        for kp in list(fc.keyframe_points):
            if abs(kp.co[0]) < 1e-6:
                fc.keyframe_points.remove(kp)
    scn.frame_set(0)
    # 不调用 transforms_clear。只给旋转单位值；location/scale显式恢复参考值。
    for pb in src.pose.bones:
        pb.location, pb.scale = reference[pb.name]
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pb.keyframe_insert(data_path='location', frame=0)
        pb.keyframe_insert(data_path='scale', frame=0)
        if pb.rotation_mode == 'QUATERNION':
            pb.keyframe_insert(data_path='rotation_quaternion', frame=0)
        elif pb.rotation_mode == 'AXIS_ANGLE':
            pb.keyframe_insert(data_path='rotation_axis_angle', frame=0)
        else:
            pb.keyframe_insert(data_path='rotation_euler', frame=0)
    scn.frame_set(0)
    bpy.context.view_layer.update()
    if any((pb.location - reference[pb.name][0]).length > 1e-5 or
           (pb.scale - reference[pb.name][1]).length > 1e-5 for pb in src.pose.bones):
        raise RuntimeError('第0帧位移/缩放未按参考帧保留，停止校准')
    if max(abs(src.matrix_world[i][j] - object_world[i][j]) for i in range(4) for j in range(4)) > 1e-5:
        raise RuntimeError('源骨架对象变换改变，停止校准')
    root_z_after = (src.matrix_world @ root.head).z if root else None
    log(f'frame 0 仅清旋转；位移/缩放保留动作第{FRAME_START}帧')
    log(f'源根高度：参考帧={root_z_before}，旋转校准后={root_z_after}')


def calibrate_anatomical_arms(src, tgt):
    """不使用 ARP Copy Bone Rest：它把骨 +Y 当成解剖方向，FBX 不满足。

    这里只校准 Shoulder->Elbow、Elbow->Wrist，保留原骨架/rest/action。
    把真实关节连线转到目标骨 rest 方向；其余源骨的绝对参考矩阵保持不变，
    避免父级校准把 Wrist/手指连带旋转。矩阵先全部计算再按父子顺序赋值。
    """
    from mathutils import Matrix
    refs = {pb.name: pb.matrix.copy() for pb in src.pose.bones}
    desired = {name: matrix.copy() for name, matrix in refs.items()}
    sw = src.matrix_world.copy()
    tw = tgt.matrix_world.copy()
    for side in ('L', 'R'):
        for source, child, target in [
            (f'{side}_Shoulder', f'{side}_Elbow', f'upper_arm_fk.{side}'),
            (f'{side}_Elbow', f'{side}_Wrist', f'forearm_fk.{side}'),
        ]:
            source_dir = (sw.to_3x3() @ (refs[child].translation - refs[source].translation)).normalized()
            tb = tgt.data.bones[target]
            target_dir = (tw.to_3x3() @ (tb.tail_local - tb.head_local)).normalized()
            delta = source_dir.rotation_difference(target_dir).to_matrix().to_4x4()
            world = sw @ refs[source]
            corrected = delta @ world
            corrected.translation = world.translation
            desired[source] = sw.inverted() @ corrected
            import math
            log(f'{source} 解剖方向 rest 校准 {math.degrees(source_dir.angle(target_dir)):.2f}°')
    def depth(pb):
        return len(pb.parent_recursive)
    for pb in sorted(src.pose.bones, key=depth):
        pb.matrix = desired[pb.name]
        bpy.context.view_layer.update()
    log('真实关节连线校准完成；未使用 Damped Track，未修改源骨轴/骨名')


def main():
    src = bpy.data.objects.get(SOURCE)
    tgt = bpy.data.objects.get(TARGET)
    if src is None or src.type != 'ARMATURE':
        raise RuntimeError(f'找不到源骨架：{SOURCE}')
    if tgt is None or tgt.type != 'ARMATURE':
        raise RuntimeError(f'找不到目标骨架：{TARGET}')
    if not src.animation_data or not src.animation_data.action:
        raise RuntimeError('源骨架没有 Action')
    if not Path(BMAP).is_file():
        raise RuntimeError(f'找不到 bmap：{BMAP}')

    ensure_view(src, tgt)
    scene = bpy.context.scene
    scene.source_rig = src.name
    scene.target_rig = tgt.name
    log(f'source={src.name} target={tgt.name}')

    # MMD RIG 控制器准备：尽可能调用 PoseCapture 已验证的函数
    try:
        import gvhmr_pose_capture as pc
        pc._prepare_mmd_controller_before_remap(tgt, bpy.context, stage='flowhmr-manual-script')
        log('MMD controller IK/FK 预处理完成')
    except Exception as exc:
        log(f'MMD controller 预处理未调用：{exc}')
        log('继续执行；如果手臂仍有问题，再手动确认 upper_arm FK=1.0、shin FK=0.95')

    select_object(src)
    bpy.ops.arp.build_bones_list()
    bpy.ops.arp.import_config(filepath=BMAP, clear_current=True)
    if len(scene.bones_map_v2) != 50:
        raise RuntimeError(f'bmap 映射数异常：{len(scene.bones_map_v2)}，预期 50')
    log('原生 FlowHMR bmap 已载入 50 条')

    # ARP rest pose redefine
    bake_frame0_rest(src)
    arp = find_arp_remap_module()

    class MockOp:
        preserve = True
        rest_pose = 'CURRENT'
        is_arp_armature = False

    mock = MockOp()
    if 'arp_updated' in src.data.keys():
        mock.is_arp_armature = True
    arp._redefine_rest_pose(mock, bpy.context)
    log('ARP redefine_rest_pose(CURRENT, Preserve=True) 完成')

    calibrate_anatomical_arms(src, tgt)
    arp._save_pose_rest(mock)
    log('ARP save_pose_rest 完成')

    if DO_RETARGET:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        scene.frame_start = FRAME_START
        scene.frame_end = FRAME_END
        log(f'开始 ARP retarget：{FRAME_START}..{FRAME_END}')
        bpy.ops.arp.retarget('EXEC_DEFAULT',
                             frame_start=FRAME_START,
                             frame_end=FRAME_END)
        log('ARP retarget 完成')
    else:
        log('DO_RETARGET=False：只完成 rest 校准，没有跑 retarget')

    log('完成；没有自动保存 blend。请检查后另存为新文件。')


if __name__ == '__main__':
    main()
