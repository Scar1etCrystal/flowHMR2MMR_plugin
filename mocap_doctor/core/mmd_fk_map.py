"""FlowHMR -> MMD native FK bone mapping.

The table is written with MMD's *Japanese standard names* (what mmd_tools calls
``name_j``).  Blender bone names carry the side as a ``.L``/``.R`` suffix, so
``左肩`` is the same bone as ``肩.L``; :func:`normalize_name` converts between
the two forms and the resolver matches either one.

Verified on a real model (Tda 式初音ミク, 252 bones):
    センター / 下半身 / 上半身 / 上半身2 / 首 / 頭 preserved
    肩.L 腕.L 腕捩.L ひじ.L 手首.L / 足.L ひざ.L 足首.L
    親指０.L 親指１.L 親指２.L 人指１.L … 小指３.L
    左つま先 missing -> 足先EX.L used instead
    上半身3 exists only on Genshin-like models and is always ignored
"""

from __future__ import annotations

import bpy

# 源关节 -> (日文名, 是否必需, 备注)
# 「必需」= 正常 MMD 模型一定有；缺失就报错并列出清单。
BONE_TABLE: tuple[tuple[str, str, bool], ...] = (
    # 躯干（センター 由位移使用，下半身 承担骨盆旋转）
    ("Pelvis", "センター", True),
    ("Pelvis", "下半身", True),
    ("Spine1", "上半身", True),
    ("Spine2", "上半身2", True),
    ("Spine3", "上半身2", True),          # 与 Spine2 合并到同一根骨
    ("Neck", "首", True),
    ("Head", "頭", True),
    # 左臂
    ("L_Collar", "左肩", True),
    ("L_Shoulder", "左腕", True),
    ("L_Elbow", "左ひじ", True),
    ("L_Wrist", "左手首", True),
    ("L_Thumb1", "左親指０", True),
    ("L_Thumb2", "左親指１", True),
    ("L_Thumb3", "左親指２", True),
    ("L_Index1", "左人指１", True),
    ("L_Index2", "左人指２", True),
    ("L_Index3", "左人指３", True),
    ("L_Middle1", "左中指１", True),
    ("L_Middle2", "左中指２", True),
    ("L_Middle3", "左中指３", True),
    ("L_Ring1", "左薬指１", True),
    ("L_Ring2", "左薬指２", True),
    ("L_Ring3", "左薬指３", True),
    ("L_Pinky1", "左小指１", True),
    ("L_Pinky2", "左小指２", True),
    ("L_Pinky3", "左小指３", True),
    # 右臂
    ("R_Collar", "右肩", True),
    ("R_Shoulder", "右腕", True),
    ("R_Elbow", "右ひじ", True),
    ("R_Wrist", "右手首", True),
    ("R_Thumb1", "右親指０", True),
    ("R_Thumb2", "右親指１", True),
    ("R_Thumb3", "右親指２", True),
    ("R_Index1", "右人指１", True),
    ("R_Index2", "右人指２", True),
    ("R_Index3", "右人指３", True),
    ("R_Middle1", "右中指１", True),
    ("R_Middle2", "右中指２", True),
    ("R_Middle3", "右中指３", True),
    ("R_Ring1", "右薬指１", True),
    ("R_Ring2", "右薬指２", True),
    ("R_Ring3", "右薬指３", True),
    ("R_Pinky1", "右小指１", True),
    ("R_Pinky2", "右小指２", True),
    ("R_Pinky3", "右小指３", True),
    # 腿（纯 FK：足=大腿、ひざ=小腿、足首=脚）
    ("L_Hip", "左足", True),
    ("L_Knee", "左ひざ", True),
    ("L_Ankle", "左足首", True),
    ("R_Hip", "右足", True),
    ("R_Knee", "右ひざ", True),
    ("R_Ankle", "右足首", True),
    # 脚趾：可选；左つま先 常常没有，退到 足先EX
    ("L_Foot", "左つま先", False),
    ("R_Foot", "右つま先", False),
)

# 可选骨的第二候选（存在就用，都不存在就忽略）
FALLBACKS: dict[str, tuple[str, ...]] = {
    "左つま先": ("左足先EX", "左足先ex", "左足先"),
    "右つま先": ("右足先EX", "右足先ex", "右足先"),
}

# 这些骨即使存在也不参与解算
IGNORED = ("上半身3", "上半身４", "上半身4")

SIDE_PREFIX = {"左": "L", "右": "R"}
SIDE_SUFFIX = {"L": "左", "R": "右"}


def normalize_name(name: str) -> str:
    """``左肩``/``肩.L`` -> ``肩.L``; centre bones unchanged.

    Both spellings point at the same bone; the canonical form used for lookup is
    the Blender ``.L``/``.R`` suffix form because that is what bone names use.
    """

    text = str(name or "").strip()
    for jp, side in SIDE_PREFIX.items():
        if text.startswith(jp):
            return f"{text[1:]}.{side}"
    if text.endswith(".L") or text.endswith(".R"):
        return text
    for side, jp in SIDE_SUFFIX.items():
        if text.endswith(f".{side}"):
            return text
    return text


def bone_japanese_name(pose_bone) -> str:
    """``name_j`` when mmd_tools is loaded, else the Blender bone name."""

    try:
        name_j = pose_bone.mmd_bone.name_j
        if name_j:
            return str(name_j)
    except Exception:  # noqa: BLE001 - mmd_tools 未启用
        pass
    return pose_bone.name


def build_lookup(armature) -> dict[str, object]:
    """normalized name (JP and Blender forms) -> pose bone."""

    lookup: dict[str, object] = {}
    for pose_bone in armature.pose.bones:
        candidates = {pose_bone.name, bone_japanese_name(pose_bone)}
        for candidate in candidates:
            lookup.setdefault(normalize_name(candidate), pose_bone)
            lookup.setdefault(str(candidate), pose_bone)
    return lookup


def resolve(armature) -> dict[str, object]:
    """Map every source joint of BONE_TABLE onto an MMD pose bone.

    Returns::

        {
          "bones": {source_joint: pose_bone, ...},
          "missing": [japanese names that are required but absent],
          "optional_missing": [...],
          "ignored": [present bones that are intentionally skipped],
          "used_fallbacks": {wanted: actual},
          "name_source": "mmd_tools" | "blender",
        }
    """

    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("请选择 MMD 骨架")
    lookup = build_lookup(armature)
    name_source = "mmd_tools" if any(
        hasattr(pose_bone, "mmd_bone") for pose_bone in armature.pose.bones
    ) else "blender"

    bones: dict[str, object] = {}
    missing: list[str] = []
    optional_missing: list[str] = []
    used_fallbacks: dict[str, str] = {}
    for source_joint, japanese, required in BONE_TABLE:
        target = lookup.get(japanese) or lookup.get(normalize_name(japanese))
        if target is None:
            for fallback in FALLBACKS.get(japanese, ()):
                target = lookup.get(fallback) or lookup.get(normalize_name(fallback))
                if target is not None:
                    used_fallbacks[japanese] = fallback
                    break
        if target is None:
            (missing if required else optional_missing).append(japanese)
            continue
        bones.setdefault(source_joint, target)

    ignored = [name for name in IGNORED
               if lookup.get(name) is not None or lookup.get(normalize_name(name)) is not None]
    return {
        "bones": bones,
        "missing": missing,
        "optional_missing": optional_missing,
        "ignored": ignored,
        "used_fallbacks": used_fallbacks,
        "name_source": name_source,
    }


def describe(armature) -> str:
    """One-line human summary for status messages."""

    report = resolve(armature)
    parts = [f"已解析 {len(report['bones'])} 根"]
    if report["used_fallbacks"]:
        parts.append("回退 " + "、".join(
            f"{k}→{v}" for k, v in report["used_fallbacks"].items()))
    if report["optional_missing"]:
        parts.append("可选缺失 " + "、".join(report["optional_missing"]))
    if report["missing"]:
        parts.append("缺必需骨 " + "、".join(report["missing"]))
    return "；".join(parts)


def mute_blocking_constraints(armature, bones) -> list[tuple[str, str]]:
    """Mute constraints that would override FK keys on the mapped bones.

    MMD models drive the legs with IK (``ひざ``/``足首``), and an MMR-bound
    armature additionally carries COPY_* constraints from ``ORG-*`` bones.  A
    pure-FK path has to switch both off, otherwise the written rotations are
    silently discarded.  Returns ``[(bone, constraint), ...]`` for restoring.
    """

    muted: list[tuple[str, str]] = []
    names = {pose_bone.name for pose_bone in bones.values()}
    for pose_bone in armature.pose.bones:
        if pose_bone.name not in names:
            continue
        for constraint in pose_bone.constraints:
            target = str(getattr(constraint, "subtarget", "") or "")
            blocking = constraint.type == "IK" or (
                constraint.type in {"COPY_ROTATION", "COPY_TRANSFORMS", "COPY_SCALE"}
                and target.startswith("ORG-")
            )
            if blocking and not constraint.mute:
                constraint.mute = True
                muted.append((pose_bone.name, constraint.name))
    return muted
