"""接触驱动的脚部后处理（替换旧的 XY 通道锁语义）。

对齐参考实现的四个语义：

* **软接触**：``foot_contacts`` 是 0–1 分数，触地段 = 分数 ≥ 0.75 且连续 ≥ 3 帧，
  中间 ≤ 6 帧的空洞桥接成同一段（参考实现 ``plants()`` 的做法）。
* **脚掌压平**：每帧实测鞋底 heel / toe 两组顶点的世界高度，按两者高差把脚
  摆平（写 ``足先EX`` / ``つま先`` 这类脚掌骨，不动 ``足首``——它上面挂着 MMD 的
  ``mmd_ik_target_override``，改它会连带重解整条腿）。
* **水平锚点**：沿用 :func:`mocap_doctor.core.target.stabilize_planted_feet` 的闭环
  世界空间稳定（比参考实现的定值锁更强：它测的是真实 MMD 踝）。
* **根高度**：支撑段内按鞋底最低点回填，落地/离地各留 ``blend`` 帧过渡。

这里只做「分段 + 压平 + 度量」；锚点交给既有闭环，避免两套求解互相覆盖。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from . import target as core_target
from .animation import (current_view_layer, get_action, preserve_scene_frame,
                        set_linear_for_paths, set_scene_frame, update_action)

# 参考实现的默认值
CONTACT_THRESHOLD = 0.75
MIN_STANCE_FRAMES = 3
BRIDGE_GAP_FRAMES = 6

# 接触分数通道顺序：left_ankle, left_toe, right_ankle, right_toe
CHANNEL_SIDES = (("L", 0), ("L", 1), ("R", 2), ("R", 3))

MMR_TOE_CANDIDATES = {
    "L": ("toe.L", "DEF-toe.L", "ORG-toe.L"),
    "R": ("toe.R", "DEF-toe.R", "ORG-toe.R"),
}

TOE_BONE_CANDIDATES = {
    "L": ("足先EX.L", "左足先EX", "つま先.L", "左つま先", "足先.L", "左足先"),
    "R": ("足先EX.R", "右足先EX", "つま先.R", "右つま先", "足先.R", "右足先"),
}


def contact_weights(path: str, frame_count: int) -> dict[str, np.ndarray] | None:
    """从 ``foot_contacts`` npz 读每帧每侧的接触权重（踝/趾取最大）。"""

    if not path:
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            if "foot_contacts" not in data.files:
                return None
            scores = np.asarray(data["foot_contacts"], dtype=np.float64)
    except Exception:  # noqa: BLE001 - 读不了就当没有分数
        return None
    if scores.ndim != 2 or scores.shape[1] < 4:
        return None
    total = int(scores.shape[0])
    weights = {
        "L": np.maximum(scores[:, 0], scores[:, 1]),
        "R": np.maximum(scores[:, 2], scores[:, 3]),
    }
    if frame_count and total != int(frame_count):
        # 分数与动作帧数不一致时按最短长度截断，宁可少判也不越界
        limit = min(total, int(frame_count))
        weights = {side: values[:limit] for side, values in weights.items()}
    return weights


def weights_from_ranges(ranges: Mapping[str, Sequence[Sequence[int]]],
                        frame_start: int, frame_end: int) -> dict[str, np.ndarray]:
    """没有分数时：把二值区间铺成权重（内部 1、外部 0），供同一套分段逻辑使用。"""

    start, end = int(frame_start), int(frame_end)
    length = max(0, end - start + 1)
    weights = {}
    for side in ("L", "R"):
        values = np.zeros(length, dtype=np.float64)
        for item in ranges.get(side, ()) or ():
            lo, hi = int(item[0]), int(item[1])
            lo = max(lo, start)
            hi = min(hi, end)
            if hi >= lo:
                values[lo - start: hi - start + 1] = 1.0
        weights[side] = values
    return weights


def _runs(values: np.ndarray, threshold: float):
    active = values >= float(threshold)
    runs = []
    start = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))
    return runs


def stance_segments(weights: Mapping[str, np.ndarray], frame_start: int,
                    *, threshold: float = CONTACT_THRESHOLD,
                    min_frames: int = MIN_STANCE_FRAMES,
                    bridge_gap: int = BRIDGE_GAP_FRAMES,
                    trim: int = 0) -> dict[str, list[tuple[int, int]]]:
    """权重 → 支撑段（桥接短空洞、丢掉过短段、按帧号换算回 Blender 帧）。"""

    start = int(frame_start)
    out: dict[str, list[tuple[int, int]]] = {}
    for side in ("L", "R"):
        values = weights.get(side)
        if values is None or len(values) == 0:
            out[side] = []
            continue
        runs = _runs(np.asarray(values, dtype=np.float64), threshold)
        merged = []
        for lo, hi in runs:
            if merged and lo - merged[-1][1] - 1 <= int(bridge_gap):
                merged[-1] = (merged[-1][0], hi)
            else:
                merged.append((lo, hi))
        segments = []
        for lo, hi in merged:
            if hi - lo + 1 < int(min_frames):
                continue
            lo_trim = lo + int(trim)
            hi_trim = hi - int(trim)
            if hi_trim < lo_trim:
                continue
            segments.append((start + lo_trim, start + hi_trim))
        out[side] = segments
    return out


def _side_vertices(mesh_object: Any, side: str) -> list[int]:
    indices = core_target.boot_vertex_indices(mesh_object, (side,)) or {}
    return list(indices.get(side, ()) or ())


def sole_split(mesh_object: Any, side: str) -> tuple[list[int], list[int]]:
    """把一侧鞋底顶点按静止姿态的前后分成 heel / toe 两组。"""

    indices = _side_vertices(mesh_object, side)
    if not indices:
        return [], []
    mesh = mesh_object.data
    coordinates = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coordinates)
    coordinates = coordinates.reshape(-1, 3)[indices]
    # MMD 模型脚掌朝 -Y；取鞋底顶点里最靠前/最后的两半
    forward = coordinates[:, 1]
    split = float(np.median(forward))
    toe = [indices[i] for i in np.nonzero(forward <= split)[0]]
    heel = [indices[i] for i in np.nonzero(forward > split)[0]]
    return heel, toe


def measure_sole(mesh_object: Any, heel: Sequence[int], toe: Sequence[int]) -> tuple[float, float, float]:
    """当前帧 heel / toe 的最低世界 Z 与两者的水平跨度（m）。"""

    if not heel or not toe:
        return 0.0, 0.0, 0.0
    depsgraph = get_clock_depsgraph()
    evaluated = mesh_object.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        count = len(mesh.vertices)
        coordinates = np.empty(count * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coordinates)
        coordinates = coordinates.reshape(-1, 3)
        matrix = np.array(evaluated.matrix_world, dtype=np.float64)
        world = coordinates @ matrix[:3, :3].T + matrix[:3, 3]
        heel_points = world[list(heel)]
        toe_points = world[list(toe)]
        heel_low = heel_points[np.argmin(heel_points[:, 2])]
        toe_low = toe_points[np.argmin(toe_points[:, 2])]
        span = float(np.linalg.norm((toe_low - heel_low)[:2]))
        return float(heel_low[2]), float(toe_low[2]), span
    finally:
        evaluated.to_mesh_clear()


def get_clock_depsgraph():
    import bpy

    return bpy.context.evaluated_depsgraph_get()


def find_toe_bone(armature: Any, side: str) -> str:
    """MMD 骨架上的脚掌骨（仅用于回报，写入见 :func:`find_mmr_toe`）。"""

    for name in TOE_BONE_CANDIDATES[side]:
        if name in armature.pose.bones:
            return name
    return ""


def find_mmr_toe(rig: Any, side: str) -> str:
    """MMR 骨架上的脚掌控制骨——真正要写的那根。

    MMD 的 ``足先EX`` 上挂着活动的 ``COPY_ROTATION``（来自 MMR 的
    ``ORG-toe.L_parent``），直接写 MMD 骨的旋转会被约束覆盖，实测脚掌角度不会
    有任何变化；写 MMR 控制器才会传下去，和 ``foot_ik`` 的做法一致。
    """

    for name in MMR_TOE_CANDIDATES[side]:
        if rig is not None and name in rig.pose.bones:
            return name
    return ""


def level_feet(scene: Any, armature: Any, rig: Any, mesh_object: Any, action: Any,
               segments: Mapping[str, Sequence[Sequence[int]]], *,
               strength: float = 0.6, max_deg: float = 12.0,
               blend: int = 3, floor_z: float = 0.0) -> dict[str, Any]:
    """支撑段内把脚掌摆平（按实测 heel/toe 高差），只写脚掌骨。"""

    layer = current_view_layer()
    report: dict[str, Any] = {"operation": "level_feet", "sides": {}, "applied": 0}
    if float(strength) <= 1e-6:
        report["skipped"] = "脚掌压平已关闭（强度 0）"
        return report
    if mesh_object is None:
        report["skipped"] = "没有目标网格"
        return report

    points = {}
    for side in ("L", "R"):
        heel, toe = sole_split(mesh_object, side)
        bone = find_mmr_toe(rig, side)
        if not heel or not toe or not bone:
            report["sides"][side] = {
                "skipped": "缺少鞋底顶点或 MMR 脚掌控制骨",
                "mmd_bone": find_toe_bone(armature, side),
            }
            continue
        points[side] = (heel, toe, bone)

    if not points:
        report["skipped"] = "两侧都不可用"
        return report

    frames_by_bone: dict[str, dict[int, float]] = {}
    spread_before, spread_after = [], []
    with preserve_scene_frame(scene, layer):
        for side, (heel, toe, bone) in points.items():
            pose_bone = rig.pose.bones[bone]
            mode = pose_bone.rotation_mode
            original = {}
            samples = {}
            for lo, hi in segments.get(side, ()) or ():
                for frame in range(int(lo), int(hi) + 1):
                    set_scene_frame(scene, frame, layer)
                    heel_z, toe_z, span = measure_sole(mesh_object, heel, toe)
                    if span <= 1e-4:
                        continue
                    spread_before.append(abs(toe_z - heel_z))
                    angle = float(np.degrees(np.arctan2(toe_z - heel_z, span)))
                    angle = max(-max_deg, min(max_deg, angle * float(strength)))
                    samples[frame] = angle
            if not samples:
                report["sides"][side] = {"bone": bone, "skipped": "没有可用的支撑帧"}
                continue

            # 相邻帧平滑，避免逐帧抖动写进曲线
            keys = sorted(samples)
            smoothed = {}
            window = max(1, int(blend))
            for frame in keys:
                near = [samples[f] for f in keys
                        if abs(f - frame) <= window]
                smoothed[frame] = float(np.mean(near))

            if mode != "QUATERNION":
                pose_bone.rotation_mode = "QUATERNION"
            for frame, angle in smoothed.items():
                set_scene_frame(scene, frame, layer)
                # 绕脚掌的横向轴（骨骼局部 X）微调；只动脚掌，不动 足首
                from mathutils import Quaternion

                pose_bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0),
                                                           np.radians(angle))
                pose_bone.keyframe_insert("rotation_quaternion", frame=frame,
                                          group=bone)
            for frame in smoothed:
                set_scene_frame(scene, frame, layer)
                heel_z, toe_z, span = measure_sole(mesh_object, heel, toe)
                if span > 1e-4:
                    spread_after.append(abs(toe_z - heel_z))
            report["sides"][side] = {
                "bone": bone,
                "frames": len(smoothed),
                "max_tilt_deg": round(max(abs(v) for v in smoothed.values()), 2),
            }
            report["applied"] += 1
            frames_by_bone[bone] = smoothed

    if frames_by_bone:
        paths = {f'pose.bones["{bone}"].rotation_quaternion' for bone in frames_by_bone}
        set_linear_for_paths(action, paths)
        update_action(action)

    if spread_before:
        report["heel_toe_spread_mm_before"] = round(
            float(np.median(spread_before)) * 1000.0, 2)
    if spread_after:
        report["heel_toe_spread_mm_after"] = round(
            float(np.median(spread_after)) * 1000.0, 2)
    return report

def ground_root(scene: Any, move_object: Any, mesh_object: Any, action: Any,
                segments: Mapping[str, Sequence[Sequence[int]]], *,
                frame_start: int, frame_end: int, floor_z: float = 0.0,
                clearance: float = 0.0015, max_step: float = 0.05,
                smooth_radius: int = 2) -> dict[str, Any]:
    """按鞋底最低点回填根高度——**不做弹道重建**。

    只在真正触地的帧（``segments`` 覆盖的帧）测量并回填：

    * 触地帧：``delta = floor + clearance - min(sole_z)``；
    * 其余帧：偏移在前后两个触地偏移之间线性过渡（保持动作本身的腾空弧线，
      不另造抛物线，也不把脚吸在锚点上再猛拽起来）。

    这替换了旧的 ``ground_feet_outside_airborne``：那套会按腾空区间重造弹道，
    并把每个「非腾空」帧无条件钉在地面，等于我们自己的作用盖过了动作细节。
    """

    layer = current_view_layer()
    report: dict[str, Any] = {"operation": "ground_root", "applied": 0}
    if mesh_object is None or move_object is None:
        report["skipped"] = "缺少网格或根物体"
        return report

    indices = core_target.boot_vertex_indices(mesh_object, ("L", "R")) or {}
    if not indices:
        report["skipped"] = "模型没有 MMD 脚部顶点组"
        return report

    start, end = int(frame_start), int(frame_end)
    frames = list(range(start, end + 1))
    grounded = set()
    for side in ("L", "R"):
        for lo, hi in segments.get(side, ()) or ():
            for frame in range(max(int(lo), start), min(int(hi), end) + 1):
                grounded.add(frame)

    before = {}
    deltas = {}
    depsgraph = get_clock_depsgraph()
    with preserve_scene_frame(scene, layer):
        for frame in frames:
            set_scene_frame(scene, frame, layer)
            lowest = core_target.evaluated_boot_min_z(mesh_object, depsgraph, indices)
            if not lowest:
                continue
            minimum = min(lowest.values())
            before[frame] = minimum
            if frame in grounded:
                deltas[frame] = (float(floor_z) + float(clearance)) - float(minimum)

    if not deltas:
        report["skipped"] = "没有任何触地帧用于回填"
        return report

    # 非触地帧：在两个触地偏移之间线性过渡（端点外保持最近的偏移）
    keys = sorted(deltas)
    filled = {}
    for frame in frames:
        if frame in deltas:
            filled[frame] = deltas[frame]
            continue
        earlier = [k for k in keys if k < frame]
        later = [k for k in keys if k > frame]
        if earlier and later:
            lo, hi = earlier[-1], later[0]
            span = float(hi - lo)
            ratio = 0.0 if span <= 0 else (frame - lo) / span
            filled[frame] = deltas[lo] * (1.0 - ratio) + deltas[hi] * ratio
        elif earlier:
            filled[frame] = deltas[earlier[-1]]
        elif later:
            filled[frame] = deltas[later[0]]

    # 相邻帧变化量封顶 + 平滑，避免抖动写进曲线
    ordered = [filled.get(frame, 0.0) for frame in frames]
    limited = []
    previous = ordered[0] if ordered else 0.0
    for value in ordered:
        step = max(-float(max_step), min(float(max_step), value - previous))
        previous = previous + step
        limited.append(previous)
    radius = max(0, int(smooth_radius))
    smoothed = []
    for index in range(len(limited)):
        lo = max(0, index - radius)
        hi = min(len(limited), index + radius + 1)
        window = limited[lo:hi]
        smoothed.append(sum(window) / float(len(window)))

    # 先记录原始根位置，再写偏移（写的时候会改求值结果）
    originals = {}
    with preserve_scene_frame(scene, layer):
        for index, frame in enumerate(frames):
            set_scene_frame(scene, frame, layer)
            originals[frame] = move_object.location.copy()
        for index, frame in enumerate(frames):
            shift = core_target._world_z_shift_in_object_space(move_object, smoothed[index])
            set_scene_frame(scene, frame, layer)
            base = originals[frame]
            move_object.location = (base.x + shift[0], base.y + shift[1], base.z + shift[2])
            move_object.keyframe_insert("location", frame=frame)

    after = {}
    with preserve_scene_frame(scene, layer):
        for frame in sorted(grounded):
            set_scene_frame(scene, frame, layer)
            lowest = core_target.evaluated_boot_min_z(mesh_object, depsgraph, indices)
            if lowest:
                after[frame] = min(lowest.values()) - float(floor_z)

    def _stats(values):
        if not values:
            return None
        data = np.asarray(values, dtype=np.float64)
        return {
            "median_mm": round(float(np.median(data)) * 1000.0, 2),
            "mean_abs_mm": round(float(np.mean(np.abs(data))) * 1000.0, 2),
            "max_abs_mm": round(float(np.max(np.abs(data))) * 1000.0, 2),
        }

    report.update({
        "applied": len(frames),
        "grounded_frames": len(grounded),
        "measured_frames": len(before),
        "delta_range_mm": [round(min(smoothed) * 1000.0, 2),
                           round(max(smoothed) * 1000.0, 2)],
        "sole_before": _stats([value - float(floor_z) for value in before.values()]),
        "sole_after": _stats([after[frame] for frame in sorted(after)]),
        "ballistic": False,
    })
    update_action(action)
    return report
