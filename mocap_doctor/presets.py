"""Fixed FreeMoCap and arue Teto/MMR preset for Blender 4.3."""

import math
import re


EXPECTED_FPS = 30

OBJECT_NAMES = {
    "source_armature": "import_synchronized_videos_rig",
    "model_root": "arue式重音テトver 2.01",
    "mmr_rig": "RIG-arue式重音テトver 2.01_arm",
    "mmd_armature": "arue式重音テトver 2.01_arm",
    "target_mesh": "arue式重音テトver 2.01_mesh",
    "correction_empty": "teto_global_correction",
}


def fixed_object_name_matches(actual: str, expected: str) -> bool:
    """Accept Blender's duplicate suffix without weakening the fixed preset.

    Importing an additional copy of the same Teto model makes Blender rename
    datablocks to ``.001``, ``.002`` and so on.  Those suffixes are naming
    disambiguators, not structural changes to the rig.
    """

    return bool(
        re.fullmatch(rf"{re.escape(str(expected))}(?:\.\d{{3}})?", str(actual))
    )

SOURCE_BONES = {
    "hips": "pelvis",
    "left_hand": "hand.L",
    "right_hand": "hand.R",
    "left_foot": "foot.L",
    "right_foot": "foot.R",
    "left_heel": "heel.02.L",
    "right_heel": "heel.02.R",
}

ARM_CHAINS = {
    "L": ["shoulder.L", "upper_arm.L", "forearm.L", "hand.L"],
    "R": ["shoulder.R", "upper_arm.R", "forearm.R", "hand.R"],
}

# --- 源骨架配置档 ----------------------------------------------------------
# 前端决定源骨架的骨骼命名：FreeMoCap 用 pelvis / hand.L / heel.02.L，
# GVHMR 经 PoseCapture 导入的是 SMPL 骨架（f_avg_ / m_avg_ 前缀）。
# 算法层只认"角色"（骨盆/手/脚……），命名差异全部在这里吸收。
SOURCE_PROFILE_FREEMOCAP = "FREEMOCAP"
SOURCE_PROFILE_GVHMR = "GVHMR"
SOURCE_PROFILE_AUTO = "AUTO"

GVHMR_PREFIX_CANDIDATES = ("f_avg", "m_avg")

# SMPL 没有独立的脚跟骨：用 Ankle（踝→前掌）与 Foot（前掌→脚尖）两个关节近似
# FreeMoCap 的 foot / heel 两个接触点。
GVHMR_BONE_SUFFIXES = {
    "hips": "Pelvis",
    "left_foot": "L_Ankle",
    "right_foot": "R_Ankle",
    "left_heel": "L_Foot",
    "right_heel": "R_Foot",
    "left_hand": "L_Wrist",
    "right_hand": "R_Wrist",
}

# Keyed by role, resolved to the rig's own bone names in _gvhmr_maps.  A hand
# repair must only touch what is actually unreliable: the wrist is HaMeR's
# prediction, while the collar, shoulder and elbow come from GVHMR's body model
# and are clean.  Interpolating those three would freeze a good arm for the
# length of every hand repair.
GVHMR_ARM_CHAIN_SUFFIXES = {
    "left_hand": ("L_Wrist",),
    "right_hand": ("R_Wrist",),
}


def gvhmr_source_prefix(armature):
    """Return the SMPL bone prefix (``f_avg`` / ``m_avg``) present on the rig.

    Deprecated (Q14): the pipeline now imports FlowHMR's native skeleton, whose
    bones have no prefix (see flowhmr_source_matches).  Kept so old projects can
    still be opened.
    """

    if armature is None or getattr(armature, "type", "") != "ARMATURE":
        return None
    names = {bone.name for bone in armature.data.bones}
    for prefix in GVHMR_PREFIX_CANDIDATES:
        if f"{prefix}_Pelvis" in names:
            return prefix
    return None


# --- FlowHMR 原生骨架（Q14：现在只服务这一种） -------------------------------
# 现在的源骨架是 FlowHMR 的原生命名（Pelvis / L_Hip / L_Ankle / …），前缀骨架
# （f_avg_/m_avg_）和 FreeMoCap 命名都不再出现在这条流水线上。旧实现里
# gvhmr_source_prefix 找不到 f_avg_Pelvis 就返回 None，于是上半身对齐（FIX B）
# 和骨盆够程（FIX C）一路静默跳过——直接读原生名才是对的那条路。
SOURCE_PROFILE_FLOWHMR = "FLOWHMR"

FLOWHMR_BONES = {
    "hips": "Pelvis",
    "left_foot": "L_Ankle",
    "right_foot": "R_Ankle",
    "left_heel": "L_Foot",
    "right_heel": "R_Foot",
    "left_hand": "L_Wrist",
    "right_hand": "R_Wrist",
}

# 认骨架必须有的骨（角色骨，不含手指）。
FLOWHMR_REQUIRED_BONES = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "L_Shoulder",
    "R_Shoulder",
    "L_Ankle",
    "R_Ankle",
    "L_Wrist",
    "R_Wrist",
)


def flowhmr_source_matches(armature):
    """True when the rig carries the FlowHMR native skeleton's joint names."""

    if armature is None or getattr(armature, "type", "") != "ARMATURE":
        return False
    names = {bone.name for bone in armature.data.bones}
    return all(name in names for name in FLOWHMR_REQUIRED_BONES)


def _flowhmr_maps():
    """FlowHMR native source maps (the live frontend)."""

    contact_points = (
        ("L_Ankle", "head"),
        ("L_Ankle", "tail"),
        ("L_Foot", "tail"),
        ("R_Ankle", "head"),
        ("R_Ankle", "tail"),
        ("R_Foot", "tail"),
    )
    foot_points = {
        "L": (("L_Ankle", "head"), ("L_Ankle", "tail"), ("L_Foot", "tail")),
        "R": (("R_Ankle", "head"), ("R_Ankle", "tail"), ("R_Foot", "tail")),
    }
    return {
        "profile": SOURCE_PROFILE_FLOWHMR,
        "prefix": "",
        "bones": dict(FLOWHMR_BONES),
        "arm_chains": {
            FLOWHMR_BONES["left_hand"]: (FLOWHMR_BONES["left_hand"],),
            FLOWHMR_BONES["right_hand"]: (FLOWHMR_BONES["right_hand"],),
        },
        "contact_points": contact_points,
        "foot_points": foot_points,
    }


def _gvhmr_maps(prefix):
    bones = {role: f"{prefix}_{suffix}" for role, suffix in GVHMR_BONE_SUFFIXES.items()}
    # The chain map is looked up by the rig's own bone name, so the role keys
    # have to be resolved here - keying them literally leaves lookups empty.
    arm_chains = {
        bones[role]: tuple(f"{prefix}_{suffix}" for suffix in suffixes)
        for role, suffixes in GVHMR_ARM_CHAIN_SUFFIXES.items()
    }
    contact_points = (
        (f"{prefix}_L_Ankle", "head"),
        (f"{prefix}_L_Ankle", "tail"),
        (f"{prefix}_L_Foot", "tail"),
        (f"{prefix}_R_Ankle", "head"),
        (f"{prefix}_R_Ankle", "tail"),
        (f"{prefix}_R_Foot", "tail"),
    )
    foot_points = {
        "L": (
            (f"{prefix}_L_Ankle", "head"),
            (f"{prefix}_L_Ankle", "tail"),
            (f"{prefix}_L_Foot", "tail"),
        ),
        "R": (
            (f"{prefix}_R_Ankle", "head"),
            (f"{prefix}_R_Ankle", "tail"),
            (f"{prefix}_R_Foot", "tail"),
        ),
    }
    return {
        "profile": SOURCE_PROFILE_GVHMR,
        "prefix": prefix,
        "bones": bones,
        "arm_chains": arm_chains,
        "contact_points": contact_points,
        "foot_points": foot_points,
    }


def _freemocap_maps(armature):
    """FreeMoCap source maps (the legacy frontend)."""

    return {
        "profile": SOURCE_PROFILE_FREEMOCAP,
        "prefix": "",
        "bones": dict(SOURCE_BONES),
        "arm_chains": {
            SOURCE_BONES["left_hand"]: tuple(ARM_CHAINS["L"]),
            SOURCE_BONES["right_hand"]: tuple(ARM_CHAINS["R"]),
        },
        "contact_points": None,
        "foot_points": None,
    }


def resolve_source_profile(armature, requested=SOURCE_PROFILE_AUTO):
    """Pick the source profile and return its bone maps.

    ``requested`` is the project setting; ``AUTO`` detects from the rig's own
    bone names so a mis-set project cannot silently analyse the wrong skeleton.
    FlowHMR native naming is checked first (Q14); the GVHMR / FreeMoCap branches
    below are deprecated compatibility for projects imported before the大改.
    Returns ``None`` when the rig matches neither profile.
    """

    requested = (requested or SOURCE_PROFILE_AUTO).upper()
    if requested in (SOURCE_PROFILE_AUTO, SOURCE_PROFILE_FLOWHMR):
        if flowhmr_source_matches(armature):
            return _flowhmr_maps()
        if requested == SOURCE_PROFILE_FLOWHMR:
            return None

    # --- deprecated: 旧前端（f_avg_/m_avg_ 前缀 GVHMR、FreeMoCap 命名） --------
    prefix = gvhmr_source_prefix(armature)
    if requested == SOURCE_PROFILE_GVHMR:
        return _gvhmr_maps(prefix) if prefix else None
    if requested == SOURCE_PROFILE_FREEMOCAP:
        return _freemocap_maps(armature)
    if prefix:
        return _gvhmr_maps(prefix)
    return _freemocap_maps(armature)


def profile_is_available(maps, armature):
    """Which of the profile's bones are missing on this rig."""

    if maps is None or armature is None:
        return ["<未识别源骨架>"]
    needed = list(dict.fromkeys(maps["bones"].values()))
    for chain in maps["arm_chains"].values():
        needed.extend(chain)
    return [name for name in needed if armature.pose.bones.get(name) is None]

TARGET_FOOT_IK = {"L": "foot_ik.L", "R": "foot_ik.R"}

MMD_ROOT_BONE = "全ての親"
MMD_CENTER_BONE = "センター"
# These are the real Blender bone names in blends/arue_teto.blend.
MMD_LEG_FK_BONES = ["足.L", "ひざ.L", "足首.L", "足.R", "ひざ.R", "足首.R"]
# The physical MMD arm chain. A portable VMD must contain its visible pose.
MMD_ARM_FK_BONES = ["腕.L", "ひじ.L", "手首.L", "腕.R", "ひじ.R", "手首.R"]
MMD_WRIST_BONES = {"L": "手首.L", "R": "手首.R"}
MMD_HAND_IK_BONES = {"L": "手IK.L", "R": "手IK.R"}
# MikuMikuRig temporarily reparents the elbow directly under the upper arm.
# A normal Teto MMD hierarchy keeps it below the upper-arm twist bone.
MMD_ELBOW_BONES = {"L": "ひじ.L", "R": "ひじ.R"}
MMD_ARM_TWIST_BONES = {"L": "腕捩.L", "R": "腕捩.R"}
MMD_WRIST_NAME_CANDIDATES = {
    "L": ["手首.L", "手首L", "左手首"],
    "R": ["手首.R", "手首R", "右手首"],
}
MMD_HAND_IK_NAME_CANDIDATES = {
    "L": ["手IK.L", "手ＩＫ.L", "手IKL", "手ＩＫL", "左手IK", "左手ＩＫ"],
    "R": ["手IK.R", "手ＩＫ.R", "手IKR", "手ＩＫR", "右手IK", "右手ＩＫ"],
}
MMD_FOOT_IK_JAPANESE_NAMES = {"L": "左足ＩＫ", "R": "右足ＩＫ"}
MMD_FOOT_IK_NAME_CANDIDATES = {
    "L": ["足ＩＫ.L", "足IK.L", "左足ＩＫ", "左足IK"],
    "R": ["足ＩＫ.R", "足IK.R", "右足ＩＫ", "右足IK"],
}
MMD_TOE_IK_JAPANESE_NAMES = {"L": "左つま先ＩＫ", "R": "右つま先ＩＫ"}
MMD_TOE_IK_NAME_CANDIDATES = {
    "L": ["つま先ＩＫ.L", "つま先IK.L", "左つま先ＩＫ", "左つま先IK"],
    "R": ["つま先ＩＫ.R", "つま先IK.R", "右つま先ＩＫ", "右つま先IK"],
}

MMR_COPY_CONSTRAINT_NAME = "MMR_复制变换"
MMR_LEG_CONSTRAINTS = {
    "足.L": ("COPY_TRANSFORMS", "ORG-thigh.L_parent"),
    "ひざ.L": ("COPY_TRANSFORMS", "ORG-shin.L_parent"),
    "足首.L": ("COPY_TRANSFORMS", "ORG-foot.L_parent"),
    "足.R": ("COPY_TRANSFORMS", "ORG-thigh.R_parent"),
    "ひざ.R": ("COPY_TRANSFORMS", "ORG-shin.R_parent"),
    "足首.R": ("COPY_TRANSFORMS", "ORG-foot.R_parent"),
}

DEFAULTS = {
    "source_floor_z": 0.0,
    "source_floor_tolerance": 0.012,
    "source_floor_clearance": 0.004,
    "source_floor_max_lift": 0.6,
    "source_floor_strength": 1.0,
    "source_floor_window": 30,
    "source_floor_smooth_radius": 2,
    "source_floor_max_delta": 0.018,
    "contact_height": 0.030,
    "contact_xy_speed": 0.012,
    "contact_moving_xy_speed": 0.012,
    "contact_vertical_speed": 0.010,
    "contact_min_segment_len": 4,
    "contact_merge_gap": 1,
    "contact_anchor_drift": 0.025,
    "contact_penetration_tolerance": 0.008,
    # FreeMoCap 时代手调的倾斜，那时前端输出是歪的。GVHMR 的骨架本身已对齐重力，
    # 再套一个固定角度只会把正确的动作掰歪，所以默认归零。
    "global_rot_x": 0.0,
    "global_rot_y": 0.0,
    "global_rot_z": 0.0,
    "tilt_strength": 0.65,
    "target_floor_z": 0.0,
    "target_clearance": 0.0015,
    "target_floor_tolerance": 0.004,
    "target_floor_max_lift": 0.035,
    "target_floor_strength": 1.0,
    "target_floor_smooth_radius": 0,
    "target_floor_max_delta": 0.01,
    "target_vertex_sample_step": 2,
    "lock_trim": 2,
    "pelvis_correction_max": 0.04,
    "drift_min_segment_len": 4,
    "lock_min_segment_len": 5,
    "lock_blend_frames": 2,
    "lock_min_xy_range": 0.006,
    "ground_smooth_radius": 2,
    "ground_max_delta": 0.01,
    "vmd_floor_offset": 0.0,
    "source_pkl_frame_start": 1,
}


def find_mmd_bone_by_japanese_name(armature, japanese_name):
    """Resolve a Blender bone through mmd_tools metadata without guessing."""

    for pose_bone in armature.pose.bones:
        mmd_bone = getattr(pose_bone, "mmd_bone", None)
        if mmd_bone and getattr(mmd_bone, "name_j", "") == japanese_name:
            return pose_bone.name
    return None


def resolve_mmd_foot_ik(armature, side):
    name = find_mmd_bone_by_japanese_name(armature, MMD_FOOT_IK_JAPANESE_NAMES[side])
    if name:
        return name
    for candidate in MMD_FOOT_IK_NAME_CANDIDATES[side]:
        if candidate in armature.pose.bones:
            return candidate
    return None


def resolve_mmd_toe_ik(armature, side):
    """Resolve the つま先ＩＫ bone the same way as the foot IK target."""

    name = find_mmd_bone_by_japanese_name(armature, MMD_TOE_IK_JAPANESE_NAMES[side])
    if name:
        return name
    for candidate in MMD_TOE_IK_NAME_CANDIDATES[side]:
        if candidate in armature.pose.bones:
            return candidate
    return None


def _mmd_japanese_name(pose_bone):
    metadata = getattr(pose_bone, "mmd_bone", None)
    return str(getattr(metadata, "name_j", "") or "") if metadata else ""


def _resolve_hand_bone(armature, side, *, helper):
    """Resolve one fixed Teto hand bone without accepting ambiguous guesses.

    Blender bone names can acquire suffixes or differ in full-width ``ＩＫ``
    spelling after import.  mmd_tools' ``name_j`` is the authoritative MMD
    label, so use it as a fallback while requiring exactly one candidate.
    """

    candidates = (
        MMD_HAND_IK_NAME_CANDIDATES if helper else MMD_WRIST_NAME_CANDIDATES
    )[side]
    by_blender = []
    for bone in armature.pose.bones:
        if any(
            re.fullmatch(rf"{re.escape(candidate)}(?:\.\d{{3}})?", bone.name)
            for candidate in candidates
        ):
            by_blender.append(bone)
    if len(by_blender) == 1:
        return by_blender[0]
    if len(by_blender) > 1:
        raise RuntimeError(
            f"固定 Teto {side} 侧存在多个候选{'手 IK' if helper else '手首'}骨骼："
            + ", ".join(bone.name for bone in by_blender)
        )

    by_mmd = [
        bone
        for bone in armature.pose.bones
        if _mmd_japanese_name(bone) in candidates
    ]
    if len(by_mmd) == 1:
        return by_mmd[0]
    if len(by_mmd) > 1:
        raise RuntimeError(
            f"固定 Teto {side} 侧 mmd_tools 手部映射不唯一："
            + ", ".join(bone.name for bone in by_mmd)
        )
    return None


def _side_matches_name(name, side):
    value = str(name)
    if side == "L":
        return ".L" in value or "左" in value
    return ".R" in value or "右" in value


def _resolve_hand_ik_from_constraints(armature, side):
    """Find a non-standard hand IK helper from the fixed Teto elbow IK rig.

    The helper is not a normal VMD bone: it is the ``subtarget`` used by the
    elbow's IK constraint.  This is a structural fallback for imported Teto
    files where the helper's Blender name changed beyond a simple ``.001``
    suffix.  It deliberately accepts only one side-matching target.
    """

    # This fallback runs after the fixed physical arm chain was validated and
    # baked, so the exact elbow name is known.  Looking at every IK constraint
    # would also find the leg IK target, which must never be treated as a hand.
    elbow = armature.pose.bones.get(f"ひじ.{side}")
    if elbow is None:
        return None
    candidates = {}
    for constraint in getattr(elbow, "constraints", ()):
        if getattr(constraint, "type", "") != "IK":
            continue
        subtarget = str(getattr(constraint, "subtarget", "") or "")
        helper = armature.pose.bones.get(subtarget) if subtarget else None
        if helper is not None:
            candidates[helper.name] = helper
    values = list(candidates.values())
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        raise RuntimeError(
            f"固定 Teto {side} 侧 IK 约束指向多个候选手 IK 骨骼："
            + ", ".join(sorted(bone.name for bone in values))
        )
    return None


def resolve_mmd_hand_bones(armature, side):
    """Return ``(wrist, hand_ik)`` pose bones for one side of fixed Teto."""

    wrist = _resolve_hand_bone(armature, side, helper=False)
    helper = _resolve_hand_bone(armature, side, helper=True)
    if helper is None:
        helper = _resolve_hand_ik_from_constraints(armature, side)
    return wrist, helper
