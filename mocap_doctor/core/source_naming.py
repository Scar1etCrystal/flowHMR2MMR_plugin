"""把导入的规范骨架（Mixamo 命名）改写成 FlowHMR 原生命名。

FlowHMR 的 ``FlowHMR FBX`` 导出用的是"规范骨架"：SMPL-H 的 52 个关节 + 少量末端骨，
但骨骼名沿用 Mixamo 那一套（``Hips`` / ``LeftUpLeg`` / ``LeftHandIndex1`` …）。

本插件下游全部按 FlowHMR 原生命名写死（bmap 的源骨名、脚部几何检测、
FIX A/B/C、手/脚间距），所以导入时**一次性**把它们改回原生名。
改名同时处理蒙皮顶点组与动作曲线路径，否则网格会失去绑定、动作会丢通道。

映射（规范名 → FlowHMR 原生名）与 SMPL-H 关节顺序一一对应：

    Hips→Pelvis  LeftUpLeg→L_Hip  LeftLeg→L_Knee  LeftFoot→L_Ankle
    LeftToeBase→L_Foot  Spine→Spine1  Spine1→Spine2  Spine2→Spine3
    Neck→Neck  Head→Head  LeftShoulder→L_Collar  LeftArm→L_Shoulder
    LeftForeArm→L_Elbow  LeftHand→L_Wrist  手指 LeftHandIndex1→L_Index1 …

右半身同理（``Right…`` → ``R_…``）。
"""

from __future__ import annotations

import bpy

# 规范骨架（Mixamo 命名）→ FlowHMR 原生命名
CANONICAL_TO_NATIVE: dict[str, str] = {
    "Hips": "Pelvis",
    "LeftUpLeg": "L_Hip",
    "RightUpLeg": "R_Hip",
    "LeftLeg": "L_Knee",
    "RightLeg": "R_Knee",
    "LeftFoot": "L_Ankle",
    "RightFoot": "R_Ankle",
    "LeftToeBase": "L_Foot",
    "RightToeBase": "R_Foot",
    "Spine": "Spine1",
    "Spine1": "Spine2",
    "Spine2": "Spine3",
    "Neck": "Neck",
    "Head": "Head",
    "LeftShoulder": "L_Collar",
    "RightShoulder": "R_Collar",
    "LeftArm": "L_Shoulder",
    "RightArm": "R_Shoulder",
    "LeftForeArm": "L_Elbow",
    "RightForeArm": "R_Elbow",
    "LeftHand": "L_Wrist",
    "RightHand": "R_Wrist",
}

_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
_NATIVE_FINGERS = {
    "Thumb": "Thumb",
    "Index": "Index",
    "Middle": "Middle",
    "Ring": "Ring",
    "Pinky": "Pinky",
}
for _side_out, _side_in in (("Left", "L"), ("Right", "R")):
    for _finger in _FINGERS:
        for _index in ("1", "2", "3"):
            CANONICAL_TO_NATIVE[f"{_side_out}Hand{_finger}{_index}"] = (
                f"{_side_in}_{_NATIVE_FINGERS[_finger]}{_index}"
            )

# 判定"这是规范骨架"的最小证据：至少命中这些高阶骨
_DETECT_KEYS = ("Hips", "LeftUpLeg", "LeftArm", "LeftHand")

_TEMP_PREFIX = "__mcd_tmp__"


def looks_canonical(armature) -> bool:
    """骨架是否使用规范（Mixamo）命名。"""

    if armature is None or armature.type != "ARMATURE":
        return False
    names = {bone.name for bone in armature.data.bones}
    return all(key in names for key in _DETECT_KEYS)


def _rename_sequence(old_names) -> list[tuple[str, str]]:
    """生成 (当前名, 目标名) 序列，冲突的目标名跳过。"""

    pairs: list[tuple[str, str]] = []
    for old in old_names:
        target = CANONICAL_TO_NATIVE.get(old)
        if target and target != old:
            pairs.append((old, target))
    return pairs


def _rename_bones(armature, pairs) -> tuple[list[tuple[str, str]], list[str]]:
    """两段式改名（先临时名再目标名），避免相互覆盖。返回 (成功对, 冲突目标)。"""

    renamed: list[tuple[str, str]] = []
    conflicts: list[str] = []
    bones = armature.data.bones
    taken = {bone.name for bone in bones}
    staged: list[tuple[str, str]] = []
    for old, new in pairs:
        if new in taken and new not in {pair[0] for pair in pairs}:
            conflicts.append(new)
            continue
        staged.append((old, new))
    for old, _new in staged:
        if old in bones:
            bones[old].name = _TEMP_PREFIX + old
    for old, new in staged:
        temp = _TEMP_PREFIX + old
        if temp in bones and new not in bones:
            bones[temp].name = new
            renamed.append((old, new))
    return renamed, conflicts


def _rename_vertex_groups(armature, pairs) -> int:
    """把蒙皮到该骨架的网格顶点组同步改名。返回改动的组数。"""

    lookup = dict(pairs)
    touched = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not _is_skinned_to(obj, armature):
            continue
        for group in obj.vertex_groups:
            target = lookup.get(group.name)
            if target and target not in obj.vertex_groups:
                group.name = target
                touched += 1
    return touched


def _is_skinned_to(obj, armature) -> bool:
    if obj.parent is armature:
        return True
    return any(
        modifier.type == "ARMATURE" and modifier.object is armature
        for modifier in obj.modifiers
    )


def _rewrite_actions(armature, pairs) -> int:
    """动作曲线里的 ``pose.bones["旧名"]`` 改成新名。返回改动的曲线数。"""

    if not pairs:
        return 0
    lookup = dict(pairs)
    actions = set()
    animation = armature.animation_data
    if animation is not None and animation.action is not None:
        actions.add(animation.action)
    for track in (animation.nla_tracks if animation is not None else ()):
        for strip in track.strips:
            if strip.action is not None:
                actions.add(strip.action)
    changed = 0
    for action in actions:
        for fcurve in action.fcurves:
            path = fcurve.data_path
            if not path.startswith('pose.bones["'):
                continue
            end = path.find('"]', len('pose.bones["'))
            if end < 0:
                continue
            current = path[len('pose.bones["'):end]
            target = lookup.get(current)
            if target:
                fcurve.data_path = f'pose.bones["{target}"]{path[end + 2:]}'
                changed += 1
    return changed


def canonicalize_armature(armature) -> dict:
    """把规范命名的骨架改写成 FlowHMR 原生名（骨、顶点组、曲线三处同步）。

    返回报告：``{"applied", "renamed", "conflicts", "vertex_groups", "fcurves"}``。
    非规范骨架原样返回 ``applied=False``（不报错，交给下游按原生名处理）。
    """

    report = {
        "applied": False,
        "renamed": 0,
        "conflicts": [],
        "vertex_groups": 0,
        "fcurves": 0,
    }
    if not looks_canonical(armature):
        return report
    pairs = _rename_sequence([bone.name for bone in armature.data.bones])
    renamed, conflicts = _rename_bones(armature, pairs)
    report["renamed"] = len(renamed)
    report["conflicts"] = conflicts
    report["vertex_groups"] = _rename_vertex_groups(armature, renamed)
    report["fcurves"] = _rewrite_actions(armature, renamed)
    report["applied"] = bool(renamed)
    return report
