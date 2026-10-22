"""Glue: live scene + annotation channels + bake cache → DataStore.

This is the only layer that knows about Blender objects, the fixed source
profiles, and the annotation channel constants.  Everything downstream
(signals, queries, accent methods) consumes plain dicts / numpy arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .. import annotation
from ..presets import resolve_source_profile
from . import agent_bake, agent_query, agent_signals

# extra roles sampled when the rig carries them (SMPL names shown; the role
# list is what's stable for signal names, the suffix differs per frontend)
_EXTRA_ROLES = (
    ("left_hip", "L_Hip"),
    ("right_hip", "R_Hip"),
    ("left_knee", "L_Knee"),
    ("right_knee", "R_Knee"),
    ("spine1", "Spine1"),
    ("spine2", "Spine2"),
    ("spine3", "Spine3"),
    ("neck", "Neck"),
    ("head", "Head"),
)

# 手指三节指骨（SMPL 后缀 L_Index1..3 等），信号层聚合成 finger.L.index.*
_FINGERS = ("Index", "Middle", "Ring", "Pinky", "Thumb")
_FINGER_ROLES = tuple(
    (f"finger_{side.lower()}_{name.lower()}{i}", f"{side}_{name}{i}")
    for side in ("L", "R") for name in _FINGERS for i in (1, 2, 3)
)


def source_bake_spec(
    armature: Any,
    settings: Any,
) -> dict[str, Any]:
    """Resolve this source rig into a bake spec {bones, points} for agent_bake."""
    maps = resolve_source_profile(
        armature, getattr(settings, "source_profile", "AUTO")
    )
    if maps is None:
        raise RuntimeError("无法识别源骨架（既不是 FreeMoCap 也没有 GVHMR 前缀）")
    bones = dict(maps["bones"])
    prefix = maps.get("prefix")
    existing = {b.name for b in armature.data.bones}
    if prefix:
        for role, suffix in _EXTRA_ROLES + _FINGER_ROLES:
            name = f"{prefix}_{suffix}"
            if name in existing:
                bones[role] = name
    points = {}
    # foot_points per side: (heel-ish, ball-ish, toe-ish) in order
    labels = ("heel", "ball", "toe")
    for side in ("L", "R"):
        foot_pts = maps.get("foot_points", {}).get(side, ())
        for idx, (bone_name, point) in enumerate(foot_pts[:3]):
            points[f"foot.{side}.{labels[idx]}"] = (bone_name, point)
    # fingertip points for direction/up_err signals
    for side in ("L", "R"):
        for name in _FINGERS:
            for i in (1, 3):   # 指根(joint1 head) 与 指尖(joint3 tail)
                role = f"finger_{side.lower()}_{name.lower()}{i}"
                if role in bones:
                    label = "root" if i == 1 else "tip"
                    points[f"finger.{side}.{name.lower()}.{label}"] = (
                        bones[role], "head" if i == 1 else "tail")
    return {"bones": bones, "points": points, "maps": maps}


# ---------------------------------------------------------------------------
# MMR RIG spec（agent 协作层的目标骨架）
# 角色名与源骨架 spec 完全一致（signals/查询层不变），骨名换成 MMR 控制骨。
# RIG-* 骨架是 Rigify 风格：控制骨无前缀、机制骨 MCH-、形变骨 DEF-、原始 ORG-。
_RIG_ROLES = {
    "hips": "torso_root",
    "root": "root",
    "spine1": "spine_fk",
    "spine2": "spine_fk.001",
    "spine3": "spine_fk.003",
    "neck": "neck",
    "head": "head",
    "left_shoulder": "shoulder.L",
    "right_shoulder": "shoulder.R",
    "left_upper_arm": "upper_arm_fk.L",
    "right_upper_arm": "upper_arm_fk.R",
    "left_forearm": "forearm_fk.L",
    "right_forearm": "forearm_fk.R",
    "left_hand": "hand_fk.L",
    "right_hand": "hand_fk.R",
    "left_hip": "thigh_fk.L",
    "right_hip": "thigh_fk.R",
    "left_knee": "shin_fk.L",
    "right_knee": "shin_fk.R",
    "left_ankle": "shin_fk.L",     # foot_ik 是控制柄；踝位置近似用小腿尾
    "right_ankle": "shin_fk.R",
    "left_foot": "foot_ik.L",
    "right_foot": "foot_ik.R",
    "left_heel": "DEF-foot.L",
    "right_heel": "DEF-foot.R",
}
# MMR 手指控制骨名：index/middle/ring/pinky 带 f_ 前缀，thumb 不带
_RIG_FINGER_STEMS = {"index": "f_index", "middle": "f_middle",
                     "ring": "f_ring", "pinky": "f_pinky", "thumb": "thumb"}


def rig_bake_spec(armature: Any) -> dict[str, Any]:
    """RIG（MMR 控制架）的 bake spec：{bones, points, maps}，角色名不变。"""
    existing = {b.name for b in armature.data.bones}
    bones = {role: name for role, name in _RIG_ROLES.items()
             if name in existing}
    for side in ("L", "R"):
        for finger, stem in _RIG_FINGER_STEMS.items():
            for i in (1, 2, 3):
                name = f"{stem}.0{i}.{side}"
                if name in existing:
                    bones[f"finger_{side.lower()}_{finger}{i}"] = name
    points = {}
    # 足底三点：DEF-foot 头/尾（踝头/前掌）+ DEF-toe 尾（脚尖）——与 SMPL 的
    # Ankle-head/Ankle-tail/Foot-tail 三点同义
    for side in ("L", "R"):
        foot = f"DEF-foot.{side}"
        if foot in existing:
            points[f"foot.{side}.heel"] = (foot, "head")
            points[f"foot.{side}.ball"] = (foot, "tail")
        toe = f"DEF-toe.{side}"
        if toe in existing:
            points[f"foot.{side}.toe"] = (toe, "tail")
        for finger, stem in _RIG_FINGER_STEMS.items():
            r1 = f"finger_{side.lower()}_{finger}1"
            r3 = f"finger_{side.lower()}_{finger}3"
            if r1 in bones and r3 in bones:
                points[f"finger.{side}.{finger}.root"] = (bones[r1], "head")
                points[f"finger.{side}.{finger}.tip"] = (bones[r3], "tail")
    return {"bones": bones, "points": points,
            "maps": {"profile": "MMR_RIG", "prefix": None, "bones": bones,
                     "foot_points": {s: [points[f"foot.{s}.heel"],
                                         points[f"foot.{s}.ball"],
                                         points[f"foot.{s}.toe"]]
                                     for s in ("L", "R")
                                     if f"foot.{s}.toe" in points}}}


def scene_intervals(scene: Any) -> dict[str, list]:
    """Pull the wizard's reliable ranges into agent interval items."""

    def _items(channel):
        return [
            {"id": i, "start": a, "end": b}
            for i, (a, b) in enumerate(
                annotation.get_channel_ranges(scene, channel)
            )
        ]

    return {
        "contact.L": _items(annotation.CHANNEL_FOOT_L_EFFECTIVE),
        "contact.R": _items(annotation.CHANNEL_FOOT_R_EFFECTIVE),
        "air": _items(annotation.CHANNEL_AIR),
        "jitter.L": _items(annotation.CHANNEL_HAND_L_MANUAL),
        "jitter.R": _items(annotation.CHANNEL_HAND_R_MANUAL),
    }


def build_store_for_scene(
    scene: Any,
    armature: Any,
    settings: Any,
    *,
    data_dir: str | Path | None = None,
    use_cache: bool = True,
    tag: str = "",
    raw_quat: Mapping[str, Any] | None = None,
    spec: dict[str, Any] | None = None,
) -> agent_query.DataStore:
    """Bake (or load cached) the rig and assemble the query store."""

    if spec is None:
        spec = source_bake_spec(armature, settings)
    start = int(getattr(settings, "mocap_frame_start", 0) or 0) or None
    end = int(getattr(settings, "mocap_frame_end", 0) or 0) or None

    bake = None
    path = None
    fingerprint = None
    if data_dir and use_cache:
        path = agent_bake.bake_cache_path(
            data_dir, getattr(armature, "name", "armature"),
            start or scene.frame_start, end or scene.frame_end, tag=tag,
        )
        # 缓存键只有骨架名 + 帧范围 + 版本：重跑向导步骤 / 恢复检查点后同一个
        # npz 还会被读出来——加基底内容指纹，不符就重烘
        fingerprint = agent_bake.snapshot_fingerprint(armature)
        bake = agent_bake.load_bake(path, fingerprint=fingerprint)
    if bake is None:
        bake = agent_bake.bake_bone_samples(
            scene, armature, spec["bones"], spec["points"], start, end
        )
        if path is not None:
            agent_bake.save_bake(path, bake, fingerprint=fingerprint)

    floor_z = float(getattr(settings, "source_floor_z", 0.0) or 0.0)
    interval_masks = scene_intervals(scene)
    signals = agent_signals.compute_signals(
        bake, floor_z=floor_z, intervals=interval_masks, raw_quat=raw_quat
    )
    # query-side intervals = the same items (signal masks mirror them)
    store = agent_query.build_store(
        signals,
        bake["frames"],
        intervals=interval_masks,
        positions=bake["pos"],
        joint_basis=bake.get("basis"),
        bone_roles=dict(spec["bones"]),
        fps=float(scene.render.fps) / float(scene.render.fps_base or 1.0),
        floor_z=floor_z,
    )
    store.bake_missing = bake.get("missing_bones", [])
    store.bake = bake          # 写工具（apply_exemplar/world_dir）要原始矩阵
    store.spec = spec
    return store
