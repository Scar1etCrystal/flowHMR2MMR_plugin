"""FlowHMR 源骨架 → ARP → MMD/MMR RIG 重定向

背景：PoseCapture 的 gvhmr_pose_capture.py 里的 _auto_remap_and_hide() 就是这套流程。
本脚本复刻它，并补上一个 FlowHMR 特有的、必须的前置步骤（见下）。

★ 关键前置：第 0 帧必须是 T-pose
  PoseCapture 的注释写 "T-pose already set during import" —— 它的导入流程会把源骨架
  摆成 T-pose。而 FlowHMR 的 FBX 第 1 帧就是视频第 1 帧的舞蹈姿势，而 fcurve 默认
  常数外推会让【第 0 帧 = 第 1 帧】—— 实测偏离 rest 达 102.57（@f_avg_Pelvis）。
  直接用会让 redefine rest pose 把"舞蹈姿势"当成 rest，结果整体错位。

  所以本脚本先做 PREPARE_FRAME0_REST：
      第 0 帧 → 清空所有 pose 变换（回到 rest = T-pose，实测上臂离水平 2.4°）
              → 按每根骨的 rotation_mode 打 location/rotation 关键帧
              → 重新 frame_set(0) 复验
  烘进 action 之后，PoseCapture 内部再 frame_set(0) 也会稳定取到 T-pose。

两种模式（改 MODE）：
  'posecapture'  推荐。烘好第 0 帧后调用 PoseCapture 的 _auto_remap_and_hide()。
                 保真度 100%，含脚距补偿。
  'standalone'   不依赖 PoseCapture，自己走 7 步；跳过脚距补偿（跑完可能要后处理）。

用法：在【MMD 模板 + 已 append FlowHMR 源骨架】的 Blender 里，
      Scripting 工作区打开本文件 → Run Script。
      第一次务必 DRY_RUN=True 先看检查输出。
"""
import bpy
import sys

# ----------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------
MODE = 'posecapture'          # 'posecapture' | 'standalone'
SOURCE_ARM = 'Reference'      # FlowHMR 源骨架对象名
TARGET_ARM = ''               # MMD/MMR RIG；留空 = 自动挑带 IK/FK 属性的骨架
BMAP_PATH = r'F:\mocap_ai_doctor\mocap_doctor\presets\bmap\gvhmr_mmd_fix_A_foot_ik.bmap'
SRC_PREFIX = 'f_avg_'         # 女体 f_avg_ / 男体 m_avg_
CONTROLLER = 'MMD_RIG'        # 'MMD_RIG' 或 'MMD_ARMATURE'
GENDER = 'FEMALE'
FRAME_START = 0
FRAME_END = None              # None = 用源 action 最后一帧
PREPARE_FRAME0_REST = True    # ★ 必须 True，除非你的源骨架第 0 帧本来就是 T-pose
DRY_RUN = True                # True = 只检查/准备，不执行 redefine + retarget

MMD_LAYERS = [
    "Face", "Torso", "Torso (Tweak)", "Spine (FK)", "Fingers (Detail)",
    "Arm.L (FK)", "Arm.R (FK)", "Leg.L (IK)", "Leg.R (IK)",
    "Leg.L (FK)", "Leg.R (FK)", "Root",
]
IKFK_SPECS = [
    ("upper_arm_fk.L", ("IK->FK (hand.L)", "IK-FK (hand.L)", "IK_FK hand.L", "IK_FK_hand.L", "IK_FK"), 1.0),
    ("upper_arm_fk.R", ("IK->FK (hand.R)", "IK-FK (hand.R)", "IK_FK hand.R", "IK_FK_hand.R", "IK_FK"), 1.0),
    ("shin_fk.L", ("IK->FK (foot.L)", "IK-FK (foot.L)", "IK_FK foot.L", "IK_FK_foot.L", "IK_FK"), 0.95),
    ("shin_fk.R", ("IK->FK (foot.R)", "IK-FK (foot.R)", "IK_FK foot.R", "IK_FK_foot.R", "IK_FK"), 0.95),
]


def P(msg):
    print(msg)


def prepare_compatible_bmap(path, source_names, target_names, output_path=None):
    """只移除缺失且未映射的五行记录；有效映射逐字保留，原表不改。"""
    from pathlib import Path
    import tempfile
    original = Path(path).resolve()
    lines = original.read_text(encoding='utf-8-sig').splitlines(keepends=True)
    if not lines or len(lines) % 5:
        raise RuntimeError(f'BMAP 不是完整五行记录格式：{path} ({len(lines)} 行)')
    source_names, target_names = set(source_names), set(target_names)
    kept, removed, errors = [], [], []
    for i in range(0, len(lines), 5):
        block = lines[i:i + 5]
        target = block[0].rstrip('\r\n').split('%')[0].strip()
        source = block[1].strip()
        unmapped = target in {'', 'None'}
        if source not in source_names:
            if unmapped:
                removed.append(source)
                continue
            errors.append(f'有效源骨缺失：{source} -> {target}')
        if not unmapped and target not in target_names:
            errors.append(f'目标骨缺失：{source} -> {target}')
        if not unmapped and block[3].strip() == 'True':
            pole = block[4].strip()
            if pole not in {'', 'None'} and pole not in target_names:
                errors.append(f'IK pole 缺失：{source} -> {pole}')
        kept.extend(block)
    if errors:
        raise RuntimeError('有效映射校验失败：\n' + '\n'.join(errors))
    if not kept:
        raise RuntimeError('兼容表没有剩余记录，拒绝重定向')
    if output_path is None:
        output = Path(tempfile.mkdtemp(prefix='flowhmr_bmap_')) / (original.stem + '_flowhmr.bmap')
    else:
        output = Path(output_path).resolve()
    if output.resolve() == original:
        raise RuntimeError('兼容表不能覆盖原表')
    output.write_text(''.join(kept), encoding='utf-8', newline='')
    P(f'  BMAP 兼容副本：{len(lines)//5} -> {len(kept)//5} 条；仅移除 {removed}')
    P(f'  有效映射参数原样保留：{output}')
    return str(output)


def pose_dev_from_rest(arm):
    """返回 (最大偏差, 骨名)：当前 pose 相对 rest 偏离多少。"""
    worst, wb = 0.0, ''
    for pb in arm.pose.bones:
        b = pb.matrix_basis
        d = max(abs(b[i][j] - (1.0 if i == j else 0.0))
                for i in range(4) for j in range(4))
        if d > worst:
            worst, wb = d, pb.name
    return worst, wb


def arm_angle(arm, side):
    """上臂相对水平的夹角（度）：判断 T-pose / A-pose。"""
    import math
    mw = arm.matrix_world
    S = mw @ arm.pose.bones[f'{SRC_PREFIX}{side}_Shoulder'].head
    E = mw @ arm.pose.bones[f'{SRC_PREFIX}{side}_Elbow'].head
    v = E - S
    return math.degrees(math.atan2(abs(v.z), math.hypot(v.x, v.y)))


def _walk_layer_collections(lc, out):
    out.append(lc)
    for child in lc.children:
        _walk_layer_collections(child, out)


def ensure_in_view_layer(*objects):
    """确保对象处在活动 View Layer 里，可被选中。

    常见原因：对象所在集合在 View Layer 里被 exclude（Outliner 的集合勾选关掉了），
    此时 select_set() 直接报 "not in View Layer 'ViewLayer'"。
    """
    vl = bpy.context.view_layer
    fixed = []
    objs = [o for o in objects if o is not None]

    lcs = []
    _walk_layer_collections(vl.layer_collection, lcs)
    for lc in lcs:
        if lc.exclude:
            lc.exclude = False
            fixed.append(f'取消 exclude: {lc.name}')
        if lc.hide_viewport:
            lc.hide_viewport = False
            fixed.append(f'取消 hide_viewport: {lc.name}')
    vl.update()

    for ob in objs:
        if ob.hide_viewport:
            ob.hide_viewport = False
            fixed.append(f'{ob.name}: hide_viewport → False')

    for ob in objs:
        if ob.name not in vl.objects:
            try:
                bpy.context.scene.collection.objects.link(ob)
                fixed.append(f'{ob.name}: 重新挂到场景集合')
            except Exception as e:
                fixed.append(f'{ob.name}: 挂载失败 {e}')
    vl.update()

    for f in fixed:
        P(f'    {f}')
    for ob in objs:
        P(f'    {ob.name}: in_view_layer={ob.name in vl.objects}  '
          f'集合={[c.name for c in ob.users_collection]}')
    return fixed


def select_only(ob):
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    ensure_in_view_layer(ob)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ob.hide_set(False)
    try:
        ob.select_set(True)
    except RuntimeError as e:
        raise RuntimeError(
            f'{ob.name!r} 仍无法选中：{e}\n'
            f'   它可能在别的 View Layer，或是被 Link（而非 Append）进来的。\n'
            f'   处理办法：在 Outliner 里找到它，右键 → 确认所在集合没有被排除；\n'
            f'   或改用自己的骨架名填 SOURCE_ARM。')
    bpy.context.view_layer.objects.active = ob
    bpy.context.view_layer.update()


def prepare_frame0_rest(src):
    """把源骨架第 0 帧烘成 rest(T-pose)。返回 (before, after) 偏差。"""
    scn = bpy.context.scene
    act = src.animation_data.action

    # 已有的第 0 帧关键帧先清掉，避免和清除后的姿态打架
    removed = 0
    for fc in list(act.fcurves):
        for kp in list(fc.keyframe_points):
            if abs(kp.co[0]) < 1e-6:
                fc.keyframe_points.remove(kp)
                removed += 1
    if removed:
        P(f'    清掉第0帧旧关键帧 {removed} 个')

    scn.frame_set(0)
    bpy.context.view_layer.update()
    before, bb = pose_dev_from_rest(src)
    P(f'    准备前：第0帧偏离 rest = {before:.6f} @ {bb}')

    select_only(src)
    bpy.ops.object.mode_set(mode='POSE')
    for pb in src.pose.bones:
        pb.bone.select = True
    if src.data.bones:
        src.data.bones.active = src.data.bones[0]
    bpy.context.view_layer.update()
    bpy.ops.pose.transforms_clear()          # 清空 loc/rot/scale → 回到 rest
    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode='OBJECT')

    # 按每根骨自己的 rotation_mode 打帧
    n_loc = n_rot = 0
    for pb in src.pose.bones:
        pb.keyframe_insert(data_path='location', frame=0)
        n_loc += 1
        if pb.rotation_mode == 'QUATERNION':
            pb.keyframe_insert(data_path='rotation_quaternion', frame=0)
        elif pb.rotation_mode == 'AXIS_ANGLE':
            pb.keyframe_insert(data_path='rotation_axis_angle', frame=0)
        else:
            pb.keyframe_insert(data_path='rotation_euler', frame=0)
        n_rot += 1
    P(f'    打帧：location {n_loc} 根，rotation {n_rot} 根')

    # 复验：重新求值后第 0 帧必须等于 rest（且是 T-pose）
    scn.frame_set(0)
    bpy.context.view_layer.update()
    after, ab = pose_dev_from_rest(src)
    aL, aR = arm_angle(src, 'L'), arm_angle(src, 'R')
    P(f'    准备后：第0帧偏离 rest = {after:.6f} @ {ab}')
    P(f'    第0帧手臂离水平：L={aL:.1f}°  R={aR:.1f}°  '
      f'({"✓ T-pose" if max(aL, aR) < 15 else "✗ 不是 T-pose，请人工检查"})')
    if after > 1e-4:
        raise RuntimeError(
            f'第0帧仍偏离 rest（{after:.6f}@{ab}）。'
            f'可能该骨用了约束/驱动，或 rotation_mode 未被覆盖。请人工检查后再跑。')
    return before, after


def find_target():
    global TARGET_ARM
    if TARGET_ARM:
        ob = bpy.data.objects.get(TARGET_ARM)
        if ob is None:
            raise RuntimeError(f'找不到目标骨架 {TARGET_ARM!r}')
        return ob
    cands = []
    for ob in bpy.data.objects:
        if ob.type != 'ARMATURE':
            continue
        has = any('IK_FK' in k or 'IK->FK' in k or 'IK-FK' in k for k in ob.keys())
        for pb in ob.pose.bones:
            if any('IK_FK' in k or 'IK->FK' in k for k in pb.keys()):
                has = True
                break
        cands.append((1 if has else 0, len(ob.pose.bones), ob.name))
    cands.sort(reverse=True)
    P('  候选骨架 (含IK/FK, 骨数, 名字):')
    for c in cands[:6]:
        P(f'    {c}')
    if not cands or not cands[0][0]:
        raise RuntimeError('没找到带 IK/FK 属性的目标骨架，请手动填 TARGET_ARM')
    TARGET_ARM = cands[0][2]
    P(f'  → 自动选中 {TARGET_ARM!r}')
    return bpy.data.objects[TARGET_ARM]


def prep_mmd_controller(tgt):
    hit = 0
    for name in MMD_LAYERS:
        bc = tgt.data.collections.get(name)
        if bc is not None:
            bc.is_visible = True
            hit += 1
    P(f'  骨骼集合打开 {hit}/{len(MMD_LAYERS)}')
    tgt.data.pose_position = 'POSE'
    for bone_name, labels, value in IKFK_SPECS:
        pb = tgt.pose.bones.get(bone_name)
        if pb is None:
            P(f'  IK/FK: {bone_name} 不存在，跳过')
            continue
        found = None
        for label, owner in (('骨', pb), ('对象', tgt), ('数据', tgt.data)):
            for key in list(owner.keys()):
                norm = key.replace(' ', '').replace('_', '').replace('>', '').replace('-', '').lower()
                if norm in ('ikfk', 'iktofk', 'ikfkblend') or key in labels:
                    found = (label, owner, key)
                    break
            if found:
                break
        if not found:
            P(f'  IK/FK: {bone_name} 没找到属性（请手动设为 {value}）')
            continue
        label, owner, key = found
        old = owner[key]
        try:
            owner[key] = value
            P(f'  IK/FK: {bone_name} [{label}.{key}] {old} → {value}')
        except Exception as e:
            P(f'  IK/FK: {bone_name} 设置失败 {e}')


def find_arp_module():
    for _name, mod in list(sys.modules.items()):
        if 'auto_rig_pro' in _name and hasattr(mod, '_redefine_rest_pose'):
            return mod
    return None


def copy_rotation_source_bones():
    bones = set()
    for side in ('L', 'R'):
        for b in ('Shoulder', 'Elbow', 'Hand'):
            bones.add(f'{SRC_PREFIX}{side}_{b}')
        if CONTROLLER.upper() not in {'MIXAMO', 'MMD', 'MMD_RIG', 'MMD_ARMATURE'}:
            bones.add(f'{SRC_PREFIX}{side}_Wrist')
        for finger in ('Index', 'Middle', 'Pinky', 'Ring', 'Thumb'):
            for idx in (1, 2, 3):
                bones.add(f'{SRC_PREFIX}{side}_{finger}{idx}')
    return bones


def select_copy_rotation_bones(src, tgt):
    scn = bpy.context.scene
    wanted = copy_rotation_source_bones()
    mapped = set()
    for item in getattr(scn, 'bones_map_v2', []):
        s = getattr(item, 'source_bone', '')
        t = getattr(item, 'name', '')
        if not s or not t:
            continue
        if t not in tgt.pose.bones:
            continue
        mapped.add(s)
    P(f'  bmap 映射到目标骨的源骨 {len(mapped)} 个')
    select_only(src)
    bpy.ops.object.mode_set(mode='POSE')
    n, active = 0, None
    for pb in src.pose.bones:
        use = pb.name in wanted and pb.name in mapped
        pb.bone.select = use
        if use:
            n += 1
            if active is None:
                active = pb.bone
    if active:
        src.data.bones.active = active
    P(f'  选中待 copy_bone_rest 的源骨 {n} 个')
    return n


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
# 供回归测试用：只加载上面这些辅助函数，不执行主流程
if globals().get('FLOWHMR_HELPERS_ONLY', False):
    raise SystemExit('helpers only, main flow skipped')

P('=' * 72)
P(f'FlowHMR → ARP → MMD 重定向    MODE={MODE}   DRY_RUN={DRY_RUN}')
P('=' * 72)

src = bpy.data.objects.get(SOURCE_ARM)
if src is None or src.type != 'ARMATURE':
    raise RuntimeError(f'找不到源骨架 {SOURCE_ARM!r}；确认已 append src_*.blend')

if not src.animation_data or not src.animation_data.action:
    raise RuntimeError(f'{src.name!r} 没有 action，无法重定向')
act = src.animation_data.action
end = FRAME_END if FRAME_END is not None else int(act.frame_range[1])

P(f'源骨架 {src.name!r}  骨数={len(src.pose.bones)}  '
  f'scale={tuple(round(v,5) for v in src.scale)}')
P(f'action={act.name!r}  帧范围={tuple(act.frame_range)}  '
  f'→ retarget {FRAME_START}..{end}')

if not hasattr(bpy.types, 'ARP_OT_retarget'):
    raise RuntimeError('ARP (auto_rig_pro) 未启用')

P('\n[0] 确保对象在 View Layer 里可选')
ensure_in_view_layer(src)

P('\n[1] 目标骨架')
tgt = find_target()
P(f'  {tgt.name!r}  骨数={len(tgt.pose.bones)}')
ensure_in_view_layer(tgt)

P('\n[映射表] 过滤仅用于旧 SMPL-X 的缺失未映射记录')
COMPAT_BMAP_PATH = prepare_compatible_bmap(
    BMAP_PATH, src.pose.bones.keys(), tgt.pose.bones.keys())

if PREPARE_FRAME0_REST:
    P('\n[2] ★ 把第0帧烘成 rest(T-pose)')
    prepare_frame0_rest(src)
else:
    P('\n[2] 跳过第0帧准备（PREPARE_FRAME0_REST=False）')
    d, b = pose_dev_from_rest(src)
    P(f'    当前第0帧偏离 rest = {d:.6f} @ {b}')
    if d > 1e-4:
        raise RuntimeError('第0帧不是 rest，会导致 redefine rest pose 出错')

if MODE == 'posecapture':
    P('\n[3] 调用 PoseCapture 的 _auto_remap_and_hide()')
    pc = None
    for name, mod in list(sys.modules.items()):
        if name.endswith('gvhmr_pose_capture'):
            pc = mod
            break
    if pc is None:
        try:
            sys.path.insert(0, r'F:\gvhmr\pack2\PoseCapture_Pack\blender_addon')
            import gvhmr_pose_capture as pc  # noqa
        except Exception as e:
            raise RuntimeError(f'无法加载 PoseCapture: {e}\n'
                               f'   请在 Preferences 启用它，或检查路径')
    P(f'  已加载 {pc.__name__}')
    props = bpy.context.scene.pose_capture_settings
    P(f'  原设置: gender={props.gender}  target_rig={getattr(bpy.context.scene, "target_rig", "")}')
    props.gender = GENDER
    props.target_armature = tgt.name
    props.remap_controller_type = CONTROLLER
    props.remap_custom_bmap = BMAP_PATH
    bpy.context.scene.target_rig = tgt.name
    import os
    if not os.path.isfile(BMAP_PATH):
        raise RuntimeError(f'找不到已验证的映射表：{BMAP_PATH}')
    resolved_target, resolved_type = pc._resolve_remap_target(
        bpy.context, props, allow_default_mixamo=False)
    if resolved_target != tgt or resolved_type != CONTROLLER:
        raise RuntimeError('PoseCapture 的目标/控制器设置未按预期生效')
    P(f'  已确认 PoseCapture 目标={tgt.name}  控制器={CONTROLLER}')
    P(f'  强制使用已验证 bmap：{BMAP_PATH}')
    if DRY_RUN:
        P('\n【只做准备，尚未重定向】DRY_RUN=True。实际运行请改成 False。')
    else:
        # MMD_RIG 模式会忽略 remap_custom_bmap，改用内置表。
        # 临时覆盖本次表路径，保留 MMD_RIG 的准备/rest-copy 行为。
        original_get_bmap = pc._get_bmap_path
        def use_verified_bmap(gender, controller_type='ARP', custom_path=''):
            if controller_type.upper() != CONTROLLER.upper():
                raise RuntimeError(f'意外的控制器类型：{controller_type}')
            print(f'[FlowHMR] 使用原表派生的兼容 BMAP: {COMPAT_BMAP_PATH}')
            return COMPAT_BMAP_PATH
        # PoseCapture 最后会 child.hide_set(True)，要求每个直接子网格
        # 同样存在于当前 View Layer；只准备源对象不足以保证这点。
        source_meshes = [child for child in src.children if child.type == 'MESH']
        ensure_in_view_layer(src, *source_meshes)
        pc._get_bmap_path = use_verified_bmap
        try:
            P('\n【正式执行】开始 rest-pose 校准与 ARP 重定向；可能暂时无响应。')
            ok = pc._auto_remap_and_hide(src)
        finally:
            pc._get_bmap_path = original_get_bmap
        P(f'\n_auto_remap_and_hide 返回 {ok}')
        if not ok:
            raise RuntimeError('PoseCapture 未完成重定向，请查看上面的 Auto-remap/错误日志')
else:
    P('\n[3] standalone：复刻 PoseCapture 的 7 步')
    scn = bpy.context.scene
    scn.source_rig = src.name
    scn.target_rig = tgt.name
    P(f'  scn.source_rig={scn.source_rig!r}  scn.target_rig={scn.target_rig!r}')

    P('\n  [1/7] MMD 控制器预处理')
    prep_mmd_controller(tgt)

    P('\n  [2/7] build_bones_list + import bmap')
    select_only(src)
    bpy.ops.arp.build_bones_list()
    P(f'  bmap: {COMPAT_BMAP_PATH}')
    bpy.ops.arp.import_config(filepath=COMPAT_BMAP_PATH, clear_current=True)
    mapped_n = len(getattr(scn, 'bones_map_v2', []))
    P(f'  读入映射 {mapped_n} 条')
    if mapped_n == 0:
        raise RuntimeError('bmap 没读进来（bones_map_v2 为空）')

    P('\n  [3/7] fcurve 全设 LINEAR')
    cnt = 0
    for fc in act.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
            cnt += 1
    P(f'    {cnt} 个关键帧')

    P('\n  [4/7] 删除 scale 曲线')
    rm = [fc for fc in act.fcurves if 'scale' in fc.data_path]
    for fc in rm:
        act.fcurves.remove(fc)
    P(f'    删除 {len(rm)} 条')

    P('\n  [5/7] 第0帧关键帧（已在 [2] 烘好，此处仅复验）')
    scn.frame_set(0)
    bpy.context.view_layer.update()
    d, b = pose_dev_from_rest(src)
    P(f'    第0帧偏离 rest = {d:.6f} @ {b}')

    if DRY_RUN:
        P('\n  [6/7] redefine rest pose —— DRY_RUN 跳过')
        P('  [7/7] arp.retarget        —— DRY_RUN 跳过')
        P('\nDRY_RUN=True → 到此为止。')
    else:
        P('\n  [6/7] redefine rest pose(CURRENT) + copy_bone_rest + save_pose_rest')
        arp_mod = find_arp_module()
        if arp_mod is None:
            raise RuntimeError('找不到 ARP 的 _redefine_rest_pose 模块')

        class _MockOp:
            preserve = True
            rest_pose = 'CURRENT'
            is_arp_armature = False
        mock = _MockOp()
        if 'arp_updated' in src.data.keys():
            mock.is_arp_armature = True
        arp_mod._redefine_rest_pose(mock, bpy.context)
        P('    _redefine_rest_pose ✓')
        n = select_copy_rotation_bones(src, tgt)
        if n:
            try:
                bpy.ops.arp.copy_bone_rest('EXEC_DEFAULT')
            except TypeError:
                bpy.ops.arp.copy_bone_rest()
            P('    copy_bone_rest ✓')
        arp_mod._save_pose_rest(mock)
        P('    _save_pose_rest ✓')

        P(f'\n  [7/7] arp.retarget({FRAME_START}..{end})')
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.arp.retarget('EXEC_DEFAULT',
                             frame_start=FRAME_START, frame_end=end)
        P('    retarget ✓')
        P('\n⚠ standalone 跳过了脚距补偿 _compensate_retargeted_foot_spacing，'
          '跑完请检查两脚间距。')

P('\n' + '=' * 72)
P('结束。完整日志见 Blender 控制台。')
