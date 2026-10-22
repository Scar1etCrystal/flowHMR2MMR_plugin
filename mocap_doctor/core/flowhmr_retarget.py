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
 11) 返回报告；由向导管理预览、接受、丢弃和检查点

插件入口：retarget(source, target, frame_start, frame_end)。
本模块只修改当前场景，不打开工程、不保存文件。

注意：
  - 手指第三节没有第四个关节，脚本**不猜**它的滚转（猜过会出接近 180° 翻转）。
  - 若目标骨架已绑定，务必让脚本自己 unbind，不要手动跳过第 1 步。
"""
import bpy, sys, math, json
from mathutils import Matrix
from pathlib import Path

# 参数由 retarget() 从当前界面的源、目标和帧范围传入。
# 插件入口只处理当前场景；检查点和保存由向导的预览事务管理。
BMAP = str(Path(__file__).resolve().parents[1] / 'presets' / 'bmap' /
           'flowhmr_native_mmd_fix_A_foot_ik.bmap')
SOURCE = ''
TARGET = ''
START, END = 1, 1

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


# The retarget is ARP's own math: the plugin drives ARP instead of reimplementing
# it.  That is deliberate (it follows every ARP fix), but it couples us to ARP's
# API - including two PRIVATE functions.  These are the exact couplings; the
# probe below turns a silent behaviour change into a clear stop / warning.
ARP_VERIFIED_VERSION = '3.75.14'
ARP_PRIVATE_CALLS = ('_redefine_rest_pose', '_save_pose_rest')
ARP_OPERATORS = ('build_bones_list', 'import_config', 'retarget')
ARP_SCENE_PROPS = ('source_rig', 'target_rig', 'bones_map_v2')


def _arp_version(module):
    """Version from the extension manifest next to auto_rig_remap.py, if any."""

    try:
        start = Path(getattr(module, '__file__', '')).resolve().parent
    except (TypeError, ValueError, OSError):
        return ''
    for folder in (start, *start.parents[:5]):
        manifest = folder / 'blender_manifest.toml'
        if not manifest.is_file():
            continue
        try:
            for line in manifest.read_text(encoding='utf-8').splitlines():
                key, _sep, value = line.partition('=')
                if key.strip() == 'version':
                    return value.strip().strip('"\'')
        except OSError:
            return ''
    return ''


def arp_capabilities(module=None):
    """Which pieces of ARP this module needs, and whether they are present.

    Versions differ between users, so nothing here compares version numbers:
    only "is the interface I call there" matters.  The version is recorded in
    the report for troubleshooting, never shown as a warning.
    """

    module = module or find_arp()
    scene = bpy.context.scene
    operators = set(dir(bpy.ops.arp)) if hasattr(bpy.ops, 'arp') else set()
    missing = [name for name in ARP_PRIVATE_CALLS
               if not callable(getattr(module, name, None))]
    missing += [f'arp.{name}' for name in ARP_OPERATORS if name not in operators]
    missing += [f'scene.{name}' for name in ARP_SCENE_PROPS if not hasattr(scene, name)]
    return {
        'version': _arp_version(module),
        'module': getattr(module, '__name__', ''),
        'public_operators': [name for name in ('redefine_rest_pose', 'save_pose_rest')
                             if name in operators],
        'missing': missing,
        'usable': not missing,
    }


def _redefine_rest_pose(module, src, rest_pose='CURRENT'):
    """ARP's own Redefine Rest Pose, preferring its public operator.

    The operator is the exact path the ARP panel's button takes (so it follows
    every ARP update); the private function is only a fallback for the rare
    case the operator's sanity check silently refuses to run.
    """
    # Private call first: this is the path that produced correct placement
    # before.  The public operator runs ARP's sanity_check and a mode switch
    # first, which can behave differently; it is only the fallback now.
    module._redefine_rest_pose(
        _ArpOp(preserve=True, rest_pose=rest_pose,
               is_arp_armature=bool(getattr(src.data, 'keys', None)
                                    and 'arp_updated' in src.data.keys())),
        bpy.context)
    # ARP's save step deletes this tag; make sure it exists even if the private
    # path bailed early, otherwise ARP raises KeyError on 'property not found'.
    if src.get('remap_redefine_rest_pose') is None:
        src['remap_redefine_rest_pose'] = True
    return 'private_function'


class _ArpOp:
    """Minimal stand-in for the properties ARP's private call reads."""

    preserve = True
    rest_pose = 'CURRENT'
    is_arp_armature = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _save_pose_rest(module, src):
    """ARP's Save Pose Rest (the panel's Apply button), with a safe fallback.

    ``_save_pose_rest`` unconditionally deletes ``remap_redefine_rest_pose`` from
    the source rig, so the tag must exist or ARP raises a KeyError.  The tag is
    re-created when missing and the call retried once.
    """
    if src.get('remap_redefine_rest_pose') is None:
        src['remap_redefine_rest_pose'] = True
    try:
        module._save_pose_rest(_ArpOp())
    except KeyError:
        src['remap_redefine_rest_pose'] = True
        module._save_pose_rest(_ArpOp())
    return 'private_function'


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


LEG_CHAIN = (
    ('L_Hip', 'L_Knee', 'L_Ankle'),
    ('R_Hip', 'R_Knee', 'R_Ankle'),
)


def _straightest_leg_frame(src):
    """Frame whose legs are straightest: the physical "standing" reference.

    ARP's preserve mode treats the root translation of the *current* pose as the
    reference, so a clip that starts squatting used to lift the whole result by
    the squat->stand pelvis travel.  Straightness is measured as the vertical
    hip-to-ankle distance, which peak-frames the moments the legs are extended
    (a squat shortens it), independent of where the clip happens to start.
    """
    sc = bpy.context.scene
    hips = [name for name in ('L_Hip', 'R_Hip') if name in src.pose.bones]
    ankles = [name for name in ('L_Ankle', 'R_Ankle') if name in src.pose.bones]
    if not hips or not ankles:
        return START
    step = max(1, (END - START) // 60)
    best_frame, best_score = START, None
    original = sc.frame_current
    for frame in range(START, END + 1, step):
        sc.frame_set(frame)
        bpy.context.view_layer.update()
        score = min(
            (src.matrix_world @ src.pose.bones[hip].head
             - src.matrix_world @ src.pose.bones[ankle].head).length
            for hip, ankle in zip(hips, ankles)
        )
        if best_score is None or score > best_score:
            best_frame, best_score = frame, score
    sc.frame_set(original)
    log(f'rest 参考帧 = {best_frame}（腿最直，髋踝距离 {best_score:.4f}）')
    return best_frame


def prepare_frame0(src):
    """只清旋转；位移/缩放取“腿最直”的那一帧（清位移会让骨架沉到地板）。

    取帧不能固定用 START：动作若从蹲姿开始，第一帧骨盆本来就低，ARP 会把它
    当基准，整段动画就被抬高“蹲→站”那么多（用户实测模型骑着原骨架脑袋）。
    """
    sc = bpy.context.scene
    act = src.animation_data.action
    for fc in list(act.fcurves):
        for kp in list(fc.keyframe_points):
            if abs(kp.co[0]) < 1e-6:
                fc.keyframe_points.remove(kp)
    reference = _straightest_leg_frame(src)
    sc.frame_set(reference)
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
    log(f'frame0 = T-pose（只清旋转；位移/缩放保留自第 {reference} 帧）')


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


def _resolve_bmap(choice, custom):
    """Bundled FlowHMR mapping, or the user's own .bmap when chosen."""
    if str(choice or '').upper() == 'CUSTOM':
        path = Path(bpy.path.abspath(str(custom or '')))
        if not str(custom or '').strip():
            raise RuntimeError('请选择自定义映射文件')
        if not path.is_file():
            raise RuntimeError('找不到自定义映射文件')
        return str(path)
    if not Path(BMAP).is_file():
        raise RuntimeError('缺少内置映射，请重新安装插件')
    return BMAP


def _begin(source, target, frame_start, frame_end, choice, custom):
    global SOURCE, TARGET, START, END
    SOURCE, TARGET = source.name, target.name
    START, END = int(frame_start), int(frame_end)
    module = find_arp()
    caps = arp_capabilities(module)
    if not caps['usable']:
        raise RuntimeError('ARP 缺少重定向所需接口：' + ', '.join(caps['missing']))
    sc = bpy.context.scene
    src = bpy.data.objects.get(SOURCE)
    tgt = bpy.data.objects.get(TARGET)
    if src is None or tgt is None:
        raise RuntimeError(f'找不到 {SOURCE!r} 或 {TARGET!r}')
    sc.source_rig, sc.target_rig = SOURCE, TARGET
    sc.frame_start, sc.frame_end = START, END
    log(f'源={SOURCE!r} 目标={TARGET!r} 帧={START}..{END}  ARP={caps["version"] or "?"}')
    return sc, src, tgt, _resolve_bmap(choice, custom), caps


def prepare(source, target, frame_start, frame_end, choice='FLOWHMR', custom='',
            rest_preset=''):
    """Steps 1-6: unbind, IK/FK, T-pose, mapping import, rest alignment.

    With ``rest_preset`` the saved rest snapshot is loaded instead of aligning
    geometrically (ARP's own ``rest_pose='SAVED'`` path), which is how one
    hand-tuned rest pose is reused for every take of the same model.
    Returns ``(report, state)``.
    """
    sc, src, tgt, bmap_path, caps = _begin(
        source, target, frame_start, frame_end, choice, custom)
    preset_info = None
    preset_error = ''
    if rest_preset:
        from . import rest_pose_preset as presets
        try:
            preset_info = presets.apply_to_scene(sc, rest_preset, src)
            log(f'rest 预设: {preset_info["path"]}（{preset_info["bones"]} 根骨）')
        except Exception as exc:  # noqa: BLE001 - 坏预设不能挡住重定向
            preset_error = str(exc)
            preset_info = None
            log(f'rest 预设不可用（{preset_error}），改用几何自动对齐')

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

    # A preset already describes the rest pose; the T-pose frame and the
    # geometric alignment only exist to *produce* that description.
    if not preset_info:
        prepare_frame0(src)
    root = src.pose.bones.get('Pelvis')
    if root is not None:
        snap_entry = (sc.get('rest_transf_offset') or {}).get('Pelvis')
        log('位移诊断: frame=%d 当前帧=%d Pelvis.location=%s 世界Z=%.4f 快照位移=%s'
            % (bpy.context.scene.frame_current, bpy.context.scene.frame_current,
               tuple(round(v, 4) for v in root.location),
               (src.matrix_world @ root.head).z,
               None if not snap_entry else [round(v, 4) for v in snap_entry[0]]))

    activate(src)
    bpy.ops.arp.build_bones_list()
    bpy.ops.arp.import_config(filepath=bmap_path, clear_current=True)
    log(f'bones_map_v2 = {len(sc.bones_map_v2)}  映射={bmap_path}')
    if len(sc.bones_map_v2) == 0:
        raise RuntimeError('映射未载入')

    arms_before, arms_after = ([], [])
    finger_before = finger_after = []
    if not preset_info:
        if ALIGN_ARMS:
            arms_before, arms_after = calibrate_arms(src, tgt)
            log('上臂/前臂 gap: before=%s after=%s'
                % ([r['gap_deg'] for r in arms_before], [r['gap_deg'] for r in arms_after]))
        if ALIGN_HANDS:
            finger_before = measure_finger_gaps(src, tgt)
            log('手指 gap before 最大 = %.3f°'
                % max(r['gap_deg'] for r in finger_before))
            align_hands(src, tgt)
            finger_after = measure_finger_gaps(src, tgt)
            log('手指 gap after  最大 = %.6f°'
                % max(r['gap_deg'] for r in finger_after))
    else:
        log('使用 rest 预设，跳过几何对齐')

    report = {
        'stage': 'prepare',
        'source': SOURCE, 'target': TARGET,
        'frames': [START, END], 'bmap': bmap_path,
        'arp': caps,
        'rest_preset': preset_info,
        'rest_preset_error': preset_error,
        'bones_map_v2': len(sc.bones_map_v2),
        'copy_bone_rest_used': False,
        'ikfk_original': ikfk_backup,
        'ikfk_set_for_retarget': {'arms': RETARGET_ARM_IKFK, 'legs': RETARGET_LEG_IKFK},
        'arm_before': arms_before, 'arm_after': arms_after,
        'finger_before_max_gap_deg': max((r['gap_deg'] for r in finger_before), default=None),
        'finger_after_max_gap_deg': max((r['gap_deg'] for r in finger_after), default=None),
    }
    return report, (sc, src, tgt, ikfk_backup, bool(preset_info))


def finish(state):
    """Steps 7-9: Redefine Rest Pose, Save Pose Rest, ARP retarget.

    SAVED restores the snapshot written by the preset (ARP's own option);
    CURRENT snapshots the pose the geometric alignment produced.
    """
    sc, src, tgt, ikfk_backup, from_preset = state
    arp = find_arp()

    if 'arp_updated' in src.data.keys():
        src['remap_is_arp_armature'] = True
    rest_pose = 'SAVED' if from_preset else 'CURRENT'
    redefine_via = _redefine_rest_pose(arp, src, rest_pose)
    log(f'Redefine Rest Pose({rest_pose}, preserve=True) 完成（{redefine_via}）')
    save_via = _save_pose_rest(arp, src)
    log(f'Save Pose Rest 完成（{save_via}）')

    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    activate(tgt)
    bpy.ops.arp.retarget('EXEC_DEFAULT', frame_start=START, frame_end=END)
    log(f'arp.retarget {START}..{END} 完成')

    if RESTORE_LEG_IKFK is not None:
        write_ikfk(tgt, LEG_BONES, RESTORE_LEG_IKFK)
    if RESTORE_ARM_IKFK is not None:
        write_ikfk(tgt, ARM_BONES, RESTORE_ARM_IKFK)

    return {
        'stage': 'finished',
        'arp_calls': {'redefine_rest_pose': redefine_via, 'save_pose_rest': save_via},
        'ikfk_restored': {'arms': RESTORE_ARM_IKFK, 'legs': RESTORE_LEG_IKFK},
    }


def retarget(source, target, frame_start, frame_end, choice='FLOWHMR', custom='',
             rest_preset=''):
    """Run the proven calibration in the current scene; never open or save a file.

    The wizard owns the structural checkpoint because ARP also changes the
    source rest pose, constraints, actions and controller properties.
    """
    report, state = prepare(source, target, frame_start, frame_end, choice, custom,
                            rest_preset)
    report.update({k: v for k, v in finish(state).items() if k != 'stage'})
    return report


def main():
    """Standalone-script entry (the globals above act as its parameters)."""
    source = bpy.data.objects.get(SOURCE)
    target = bpy.data.objects.get(TARGET)
    if source is None or target is None:
        raise RuntimeError(f'找不到 {SOURCE!r} 或 {TARGET!r}')
    return retarget(source, target, START, END, 'CUSTOM', BMAP)


if __name__ == '__main__':
    main()
