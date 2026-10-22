"""阶段2：把你手动摆好的 rest 姿态存成 ARP 参考，然后重定向，并把 IK/FK 调回去。

用法：接着阶段1的结果（同一个 Blender 会话，或重新打开阶段1另存的文件）运行本文件。

它会：
  1. 校验当前确实处于「已 Redefine、未 Apply」状态；
  2. arp.save_pose_rest -> 把你摆的姿态存进 rest_transf_offset；
  3. 把 IK/FK 还原成阶段1记录的原值（腿默认还原为纯 IK 0.0，除非你改 RESTORE 参数）；
  4. arp.retarget 烘焙；
  5. 另存结果文件。
"""
import bpy, sys, json
from pathlib import Path

BLEND = ''                 # '' = 用当前已打开的工程
SAVE_AS = ''               # '' = <工程名>_retargeted.blend
FRAME_START = 1
FRAME_END = 300
LEG_RESTORE = 0.0          # 你只用 IK -> 腿还原成 0.0；填 None = 用阶段1记录的原值
ARM_RESTORE = 0.0          # 手臂同样还原成 0.0；填 None = 用阶段1记录的原值
SOURCE = 'Reference'
TARGET = 'RIG-arue式重音テトver 2.01_arm'


def log(x):
    print('[STAGE2]', x, flush=True)


def ikfk_prop(pb):
    for k in list(pb.keys()):
        norm = k.replace(' ', '').replace('_', '').replace('-', '').replace('>', '').lower()
        if norm in ('ikfk', 'iktofk', 'ikfkblend'):
            return k
    return None


def set_ikfk(tgt, name, value):
    if value is None:
        return None
    pb = tgt.pose.bones.get(name)
    if not pb:
        return None
    k = ikfk_prop(pb)
    if not k:
        return None
    old = float(pb[k])
    pb[k] = value
    log(f'IK_FK: {name} {old} -> {value}')
    return old


def find_arp():
    for n, m in list(sys.modules.items()):
        if 'auto_rig_pro' in n and hasattr(m, '_save_pose_rest'):
            return m
    raise RuntimeError('找不到 ARP 的 auto_rig_remap 模块')


def main():
    if BLEND:
        bpy.ops.wm.open_mainfile(filepath=BLEND)
    sc = bpy.context.scene
    src = bpy.data.objects.get(SOURCE)
    tgt = bpy.data.objects.get(TARGET)
    if src is None or tgt is None:
        raise RuntimeError(f'找不到 {SOURCE!r} 或 {TARGET!r}')

    if 'remap_redefine_rest_pose' not in src.keys():
        raise RuntimeError(
            '当前不是「已 Redefine、未 Apply」状态。\n'
            '  请先运行阶段1，并在摆好姿态后立刻运行本脚本（不要 Apply、不要重开未保存的文件）。')

    arp = find_arp()

    class Op:
        preserve = True
        rest_pose = 'CURRENT'
        is_arp_armature = False

    op = Op()
    if 'arp_updated' in src.data.keys():
        op.is_arp_armature = True

    # 1) 把你手动摆的姿态存成 ARP 参考 rest
    arp._save_pose_rest(op)
    log('_save_pose_rest 完成 -> 你的手动姿态已成为 ARP 参考')

    # 2) IK/FK 调回去（先记原值，再按参数还原）
    backup = {}
    if 'mocap_doctor_ikfk_backup' in sc.keys():
        try:
            backup = json.loads(sc['mocap_doctor_ikfk_backup'])
        except Exception:
            backup = {}
        log('阶段1记录的原始 IK/FK: ' + json.dumps(backup, ensure_ascii=False))
    for name in ('thigh_parent.L', 'thigh_parent.R'):
        want = LEG_RESTORE
        if want is None and name in backup:
            want = backup[name]['value']
        set_ikfk(tgt, name, want)
    for name in ('upper_arm_parent.L', 'upper_arm_parent.R'):
        want = ARM_RESTORE
        if want is None and name in backup:
            want = backup[name]['value']
        set_ikfk(tgt, name, want)

    # 3) 重定向
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.arp.retarget('EXEC_DEFAULT', frame_start=FRAME_START, frame_end=FRAME_END)
    log(f'arp.retarget {FRAME_START}..{FRAME_END} 完成')

    target = SAVE_AS or (str(Path(bpy.data.filepath).with_name(
        Path(bpy.data.filepath).stem + '_retargeted.blend')) if bpy.data.filepath else '')
    if target:
        bpy.ops.wm.save_as_mainfile(filepath=target)
        log('SAVED ' + target)
    log('完成。别忘了之后按需把 IK/FK 再调一次（腿 IK=0 时 FK 曲线不可见）。')


if __name__ == '__main__':
    main()
