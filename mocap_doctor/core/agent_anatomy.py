"""语义解剖探头：从实时场景几何推导解剖学方向，替代"猜局部轴"。

为什么需要这一层：修复意图是语义的（"手心朝前"），而执行器吃的是骨骼局部
轴。以前哪根局部轴对应掌心、符号是正还是负全靠猜，验证又用同一个假设
自证——两次掌心修复失败都是这个模式。本模块从几何事实推导：

- 掌心：优先用**目标网格**标定（见 palm_calibration：掌心皮肤相对 hand_fk 刚性，
  量一次得到固定局部向量，逐帧只需转一下）；没有网格时退回手指几何——四指指根连线
  × 手指指向 定掌平面，各指末节弯曲向量定号（手指一弯这个平面就跟着转，实测与可见
  掌心差中位数 45°，所以只是兜底；evidence.palm_source 标明用的是哪种）。
- 脚底：脚跟/前掌/脚尖三点定平面，小腿在脚背侧定号（空中同样成立）。
- 膝/肘前：大小两段骨的夹角方向即关节凸出方向。
- 身体前方：脚尖水平投影为主，肩线×竖直轴、相机方向交叉验证。
- 每个返回值附带 owner 骨上的**精确局部向量**（喂给 hold_pose.world_axis）
  与次轴（喂给 secondary_axis 双轴解算），以及数值证据和跨帧一致性。

所有方向都在世界空间推导，再反算成控制骨局部向量——骨架约定（Rigify
轴向、左右手镜像）完全不用关心。
"""

from __future__ import annotations

import contextlib
from typing import Any, Mapping, Sequence

import bpy
import numpy as np
from mathutils import Vector

from .animation import preserve_scene_frame, set_scene_frame

# MMR 手指控制骨名（与 agent_io._RIG_FINGER_STEMS 保持一致，本模块独立副本
# 以免探头被 io 层结构绑死）
_FINGER_STEMS = {"index": "f_index", "middle": "f_middle",
                 "ring": "f_ring", "pinky": "f_pinky", "thumb": "thumb"}
_PALM_FINGERS = ("index", "middle", "ring", "pinky")   # 拇指另行处理

_EPS = 1e-6


def _pw(armature: Any, bone: str, end: str = "head") -> Vector | None:
    """世界空间的 pose 骨端点（评估后姿态，包含所有 NLA strip）。"""
    pb = armature.pose.bones.get(bone)
    if pb is None:
        return None
    return armature.matrix_world @ (pb.head if end == "head" else pb.tail)


def _norm(v: Vector) -> Vector | None:
    return v.normalized() if v.length > _EPS else None


def _angle_deg(a: Vector, b: Vector) -> float:
    d = max(-1.0, min(1.0, float(a.normalized() @ b.normalized())))
    return float(np.degrees(np.arccos(d)))


def _local_of(armature: Any, bone: str, world_dir: Vector) -> Vector | None:
    """world_dir 在指定骨上的当前局部表达（喂给 world_axis/secondary_axis）。"""
    pb = armature.pose.bones.get(bone)
    if pb is None:
        return None
    rot = (armature.matrix_world @ pb.matrix).to_quaternion()
    return rot.inverted() @ world_dir


def _sample_frames(scene: Any, frame_range: Sequence[int] | None,
                   max_n: int) -> list[int]:
    if frame_range:
        a = int(frame_range[0])
        b = int(frame_range[1]) if len(frame_range) > 1 else a
        if b <= a:
            return [a]
        n = max(2, min(int(max_n), b - a + 1))
        return sorted({int(round(a + i * (b - a) / (n - 1)))
                       for i in range(n)})
    return [int(scene.frame_current)]


# ---------------------------------------------------------------------------
# 掌心网格标定：让 agent 读到的"掌心"= 用户在视口里看到的掌心网格
#
# 旧法（手指几何）把"手指指向"当掌平面的一条边，手指一弯它就倒向掌心侧，掌平面跟着
# 转：arue 式重音テト 1499 帧实测，与可见掌心网格法线的夹角中位数 45°、最大 168°，
# 手指伸直时还会定错号（置信度 ≥0.5 的帧里 13% 符号相反）。而掌心皮肤本身相对
# hand_fk 是刚性的（权重 94% 在 手首，法线在骨局部坐标里 150 帧散布 ≤1.5°），所以
# 掌心 = hand_fk 上一个固定局部向量，从目标网格量一次就够：
#   1) 手骨几何定掌平面：指根连线 × (腕→指根中心)，符号取拇指根所在侧（拇指根在
#      手掌侧，实测离手中面 12.8 mm；再与手指弯曲定号交叉验证，不一致只做标记）；
#   2) 在手骨坐标系里框出手掌皮肤：腕→指根 25%–95%、横向 ±0.6 指根跨度、掌侧
#      0–40 mm（这件衣服的袖口在 60 mm 外；手背皮肤在另一侧），顶点权重须主要落在
#      本侧 手首/手指/手捩 组（排除贴近的另一只手臂）；
#   3) 取其中朝掌侧的面，面积加权求法线，反算到 hand_fk 局部并缓存。
# 没有目标网格 / 不是 MMD 命名（无 手首.L/R 顶点组）→ 退回手指几何，
# evidence.palm_source = "fingers"（bridge 会给 warning）。

_MESH_HAND_GROUP = "手首.{s}"
_MESH_FINGER_GROUPS = tuple(f"{a}{b}" for a in ("親指", "人指", "中指", "薬指", "小指")
                            for b in ("０", "１", "２", "３", "先"))
_MESH_TWIST_GROUPS = ("手捩", "手捩1", "手捩2", "手捩3")
_PALM_BOX = {"t_min": 0.25, "t_max": 0.95, "lat": 0.6, "h_max": 0.04}
# 缓存（审查 M21）：只缓存**成功**的标定（刚性局部向量，与帧无关）；失败按 (键, 帧) 记，换帧就重试——
# 以前失败结果整个会话都缓存着（比如网格在视口里被禁用时量到的是静止姿态 → 退回手指几何、误差 45°），
# 而 reset 没人调用。键里带物体指针 + 网格指纹，读文件（load_post）时整体清空。
_PALM_CAL: dict[tuple, dict] = {}
_PALM_FAIL: dict[tuple, dict] = {}
_WEIGHT_CACHE: dict[tuple, dict] = {}


def _target_mesh(scene: Any = None) -> Any | None:
    """向导里设的目标模型网格（settings.target_mesh），不是 MESH 时返回 None。"""
    scene = scene if scene is not None else getattr(bpy.context, "scene", None)
    settings = getattr(scene, "mocap_doctor", None)
    obj = getattr(settings, "target_mesh", None)
    return obj if obj is not None and getattr(obj, "type", "") == "MESH" else None


def reset_palm_calibration() -> None:
    _PALM_CAL.clear()
    _PALM_FAIL.clear()
    _WEIGHT_CACHE.clear()


def reset_caches() -> None:
    """读新文件时调（agent_bridge._on_file_loaded / load_post）：标定都是"这个文件、这套骨架"的。"""
    reset_palm_calibration()
    _HINGE_CAL.clear()


def _mesh_key(mesh: Any) -> tuple:
    """物体身份 + 网格指纹：同名物体换了文件 / 换了网格数据 / 顶点组改名增删都会换键。"""
    me = mesh.data
    return (mesh.name, mesh.as_pointer(), me.name, me.as_pointer(), len(me.vertices), len(me.polygons),
            tuple(g.name for g in mesh.vertex_groups))


def _layer_collection_paths(obj: Any) -> list:
    """视图层集合树里，直接装着 obj 的每个 LayerCollection 的路径（根除外）。"""
    out = []

    def walk(lc, path):
        if obj.name in lc.collection.objects:
            out.append(path)
        for ch in lc.children:
            walk(ch, path + [ch])
    vl = bpy.context.view_layer
    walk(vl.layer_collection, [])
    return out


def _needs_unhide(obj: Any) -> bool:
    """网格当帧没被骨架变形求值？——"在视口中禁用"的物体 evaluated_get().is_evaluated 照样是 True，
    但几何不求值（实测：掌心标定退回手指几何、靴底高度变成静止姿态的 +102 mm），所以看可见性。"""
    try:
        if not obj.visible_get():
            return True
    except RuntimeError:
        return True
    if not obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).is_evaluated:
        return True
    return any(m.type == "ARMATURE" and not m.show_viewport for m in getattr(obj, "modifiers", ()))


@contextlib.contextmanager
def evaluable(obj: Any):
    """让 obj 当帧按姿态被求值（审查 M21）。

    "在视口中禁用"（hide_viewport）/ 隐藏 / 所在集合被排除或禁用 / 骨架修改器在视口里关掉的网格，
    evaluated 网格是**静止姿态**——掌心标定量的是 T 姿态、ground_report 的靴底高度是静止高度。
    这里临时打开、求值，退出时原样还原（用户的显示状态不变）。"""
    restore = []
    try:
        if obj is not None and _needs_unhide(obj):
            if getattr(obj, "hide_viewport", False):
                obj.hide_viewport = False
                restore.append((obj, "hide_viewport", True))
            try:
                if obj.hide_get():
                    obj.hide_set(False)
                    restore.append((obj, "hide_set", True))
            except RuntimeError:
                pass
            for m in getattr(obj, "modifiers", ()):
                if m.type == "ARMATURE" and not m.show_viewport:
                    m.show_viewport = True
                    restore.append((m, "show_viewport", False))
            bpy.context.view_layer.update()
            if _needs_unhide(obj):
                for path in _layer_collection_paths(obj):
                    for lc in path:
                        if lc.exclude:
                            lc.exclude = False
                            restore.append((lc, "exclude", True))
                        if lc.hide_viewport:
                            lc.hide_viewport = False
                            restore.append((lc, "hide_viewport", True))
                        if lc.collection.hide_viewport:
                            lc.collection.hide_viewport = False
                            restore.append((lc.collection, "hide_viewport", True))
                bpy.context.view_layer.update()
        yield
    finally:
        for target, attr, val in reversed(restore):
            try:
                if attr == "hide_set":
                    target.hide_set(val)
                else:
                    setattr(target, attr, val)
            except Exception:  # noqa: BLE001 - 还原尽力而为
                pass
        if restore:
            bpy.context.view_layer.update()


def _mesh_group_weights(mesh: Any, groups: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]:
    """{key: (V,) 顶点在该组名单上的权重之和}，一遍扫完所有顶点（MMD 权重已归一）。与帧无关，按网格指纹缓存。"""
    ck = (_mesh_key(mesh), tuple(sorted((k, tuple(v)) for k, v in groups.items())))
    hit = _WEIGHT_CACHE.get(ck)
    if hit is not None:
        return hit
    out = _mesh_group_weights_uncached(mesh, groups)
    _WEIGHT_CACHE[ck] = out
    return out


def _mesh_group_weights_uncached(mesh: Any, groups: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]:
    by_name = {g.name: g.index for g in mesh.vertex_groups}
    ids = {k: {by_name[n] for n in names if n in by_name} for k, names in groups.items()}
    n = len(mesh.data.vertices)
    out = {k: np.zeros(n) for k in groups}
    lookup = {gi: k for k, s in ids.items() for gi in s}
    if not lookup:
        return out
    for v in mesh.data.vertices:
        for ge in v.groups:
            k = lookup.get(ge.group)
            if k is not None:
                out[k][v.index] += ge.weight
    return out


def _evaluated_mesh_arrays(mesh: Any):
    """评估后网格（含骨架变形）的世界空间数组：顶点、面法线、面积、面→顶点索引。
    调用方要在 evaluable(mesh) 里调；网格没被依赖图求值（is_evaluated=False）就报错，绝不拿静止网格当姿态。"""
    if _needs_unhide(mesh):
        raise RuntimeError(f"mesh_not_evaluated: {mesh.name} 没按当帧姿态求值（被禁用/隐藏/集合被排除/骨架修改器关了），"
                           "读到的会是静止网格")
    ev = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    m = ev.to_mesh()
    try:
        nv, nf = len(m.vertices), len(m.polygons)
        co = np.empty(nv * 3, dtype=np.float32)
        m.vertices.foreach_get("co", co)
        pn = np.empty(nf * 3, dtype=np.float32)
        m.polygons.foreach_get("normal", pn)
        pa = np.empty(nf, dtype=np.float32)
        m.polygons.foreach_get("area", pa)
        ls = np.empty(nf, dtype=np.int64)
        m.polygons.foreach_get("loop_start", ls)
        lt = np.empty(nf, dtype=np.int64)
        m.polygons.foreach_get("loop_total", lt)
        lv = np.empty(len(m.loops), dtype=np.int64)
        m.loops.foreach_get("vertex_index", lv)
    finally:
        ev.to_mesh_clear()
    mat = np.asarray(ev.matrix_world, dtype=np.float64)
    rot = mat[:3, :3]
    co = co.reshape(nv, 3).astype(np.float64) @ rot.T + mat[:3, 3]
    pn = pn.reshape(nf, 3).astype(np.float64) @ np.linalg.inv(rot)      # 法线用逆转置
    pn /= np.linalg.norm(pn, axis=1, keepdims=True) + 1e-12
    return co, pn, pa.astype(np.float64), ls, lt, lv


def _calibrate_palm(armature: Any, side: str, mesh: Any) -> dict | None:
    """当前帧量一次：掌心皮肤法线在 hand_fk 局部的表达。失败返回 {"reason": ...}。"""
    pb = armature.pose.bones
    need = [f"hand_fk.{side}", f"thumb.01.{side}"] + \
        [f"{_FINGER_STEMS[f]}.01.{side}" for f in _PALM_FINGERS]
    if any(pb.get(n) is None for n in need):
        return {"reason": "missing_hand_bones"}
    mw = armature.matrix_world

    def head(n):
        return np.asarray(mw @ pb[n].head, dtype=np.float64)
    wrist = head(f"hand_fk.{side}")
    roots = [head(f"{_FINGER_STEMS[f]}.01.{side}") for f in _PALM_FINGERS]
    kc = np.mean(roots, axis=0)
    axis, lat = kc - wrist, roots[3] - roots[0]
    hand_len, span = float(np.linalg.norm(axis)), float(np.linalg.norm(lat))
    if hand_len < 1e-4 or span < 1e-4:
        return {"reason": "degenerate_hand"}
    axis, lat = axis / hand_len, lat / span
    n0 = np.cross(lat, axis)
    if np.linalg.norm(n0) < _EPS:
        return {"reason": "degenerate_hand"}
    n0 /= np.linalg.norm(n0)
    d_thumb = float((head(f"thumb.01.{side}") - 0.5 * (wrist + kc)) @ n0)
    if abs(d_thumb) < 0.003:                       # 拇指根几乎在手中面上：定不了号
        return {"reason": "thumb_on_midplane", "thumb_mm": round(d_thumb * 1000, 1)}
    n_geo = n0 if d_thumb > 0 else -n0

    co, pn, pa, ls, lt, lv = _evaluated_mesh_arrays(mesh)
    if len(co) != len(mesh.data.vertices):
        return {"reason": "mesh_topology_changed"}
    w = _mesh_group_weights(mesh, {
        "hand": [_MESH_HAND_GROUP.format(s=side)],
        "fingers": [f"{g}.{side}" for g in _MESH_FINGER_GROUPS],
        "twist": [f"{g}.{side}" for g in _MESH_TWIST_GROUPS]})
    if not w["hand"].any():
        return {"reason": "no_hand_vertex_group"}
    rel = co - wrist
    t, h = rel @ axis, rel @ n_geo
    l = rel @ lat - float((kc - wrist) @ lat)
    box = _PALM_BOX
    verts = ((w["hand"] + w["fingers"] + w["twist"]) > 0.9) & (w["fingers"] < 0.6) \
        & (t > box["t_min"] * hand_len) & (t < box["t_max"] * hand_len) \
        & (np.abs(l) < box["lat"] * span) & (h > 0.0) & (h < box["h_max"])
    if int(verts.sum()) < 6:
        return {"reason": "too_few_palm_vertices", "n": int(verts.sum())}
    face_all = np.minimum.reduceat(verts[lv].astype(np.int8), ls) == 1   # 面的顶点全部入选
    faces = face_all & (pn @ n_geo > 0.5)
    if int(faces.sum()) < 4:
        return {"reason": "too_few_palm_faces", "n": int(faces.sum())}
    n = (pn[faces] * pa[faces, None]).sum(axis=0)
    if np.linalg.norm(n) < _EPS:
        return {"reason": "degenerate_palm_patch"}
    n /= np.linalg.norm(n)
    fingers = _hand_frame_fingers(armature, side)
    conflict = bool(fingers and fingers["confidence"] >= 0.5
                    and float(np.asarray(fingers["palm"]) @ n) < 0)
    hand_mat = mw @ pb[f"hand_fk.{side}"].matrix
    rot = hand_mat.to_quaternion()
    local = (rot.inverted() @ Vector(n.tolist())).normalized()
    fc = np.add.reduceat(co[lv], ls, axis=0) / lt[:, None]          # 面中心
    centre = (fc[faces] * pa[faces, None]).sum(axis=0) / pa[faces].sum()
    centre_local = hand_mat.inverted() @ Vector(centre.tolist())
    return {"local": local, "centre_local": centre_local,
            "faces": int(faces.sum()), "verts": int(verts.sum()),
            "thumb_mm": round(d_thumb * 1000, 1),
            "geo_vs_mesh_deg": round(_angle_deg(Vector(n_geo.tolist()), Vector(n.tolist())), 1),
            "frame": int(getattr(bpy.context.scene, "frame_current", 0)),
            "sign_conflict": conflict, "mesh": mesh.name}


def palm_calibration(armature: Any, side: str, scene: Any = None) -> dict | None:
    """缓存的掌心标定（键：网格名 + 骨架名 + 侧 + 顶点数）；没有目标网格 → None。"""
    mesh = _target_mesh(scene)
    if mesh is None:
        return None
    key = (_mesh_key(mesh), armature.name, armature.as_pointer(), side)
    hit = _PALM_CAL.get(key)
    if hit is not None:
        return hit
    scene = scene if scene is not None else bpy.context.scene
    fkey = (key, int(scene.frame_current))
    if fkey in _PALM_FAIL:                 # 同一帧已经失败过：别每次 probe 都重算
        return _PALM_FAIL[fkey]
    try:
        with evaluable(mesh):
            res = _calibrate_palm(armature, side, mesh)
    except Exception as exc:  # noqa: BLE001 - 标定只是增强，失败退回手指几何
        res = {"reason": f"error: {exc}"}
    if res and res.get("local") is not None:
        _PALM_CAL[key] = res
    else:
        _PALM_FAIL[fkey] = res
    return res


# ---------------------------------------------------------------------------
# 标记箭头（markers）：用户/工具绑在骨上的 SINGLE_ARROW 空物体 MCD_<part>.<side>，+Z = 方向。
# 有它就以它为准（palm_source / sole_source = "marker"）——用户看到的、agent 算的是同一支箭头。
# 只认骨骼父级的（刚性跟随）；没父级的箭头是静止的世界方向，当定义必错，忽略并提醒。
# 烘焙出来的（mcd_marker="baked"，逐帧 K 帧的显示用箭头）不读回。

MARKER_PREFIX = "MCD_"
# 绑定标记（骨骼父级，一帧对准全程有效）：掌心/脚底（2026-10-04）+ 膝/肘/脸/胸/骨盆（2026-10-04 §16）
MARKER_PARTS = ("palm", "sole", "knee", "elbow", "face", "chest", "pelvis")
SIDED_MARKERS = ("palm", "sole", "knee", "elbow")
# 标记部位 → probe 的 part（标记的初值/对照用这个 part 的几何定义）
MARKER_PROBE_PART = {"palm": "palm", "sole": "sole", "knee": "knee_front", "elbow": "elbow_front",
                     "face": "face", "chest": "chest", "pelvis": "pelvis"}
# probe 的 part → 读哪支标记
PROBE_MARKER_PART = {"palm": "palm", "back_of_hand": "palm", "sole": "sole", "instep": "sole",
                     "knee_front": "knee", "elbow_front": "elbow",
                     "face": "face", "chest": "chest", "pelvis": "pelvis"}
_IGNORE_MARKERS = [False]


@contextlib.contextmanager
def ignore_markers():
    """临时不读标记（markers 工具重建标记、list 对比几何估计时用）。"""
    prev = _IGNORE_MARKERS[0]
    _IGNORE_MARKERS[0] = True
    try:
        yield
    finally:
        _IGNORE_MARKERS[0] = prev


def marker_name(part: str, side: str | None) -> str:
    return f"{MARKER_PREFIX}{part}.{side}" if side else f"{MARKER_PREFIX}{part}"


def bound_marker(part: str, side: str) -> tuple[Any | None, str | None]:
    """(合格的标记物体, None) / (None, 原因)；原因 None = 根本没有这支标记。"""
    if _IGNORE_MARKERS[0]:
        return None, "ignored"
    obj = bpy.data.objects.get(marker_name(part, side))
    if obj is None:
        return None, None
    if obj.get("mcd_marker") == "baked":
        return None, "baked"
    par = getattr(obj, "parent", None)
    if (par is None or getattr(par, "type", "") != "ARMATURE"
            or obj.parent_type != "BONE" or not obj.parent_bone):
        return None, "unbound"
    # 自己带关键帧 / 驱动 / 约束的箭头在骨骼父级之上还在动：不是刚性定义（审查 M20：在同名物体上 bake 过、
    # 又 create overwrite 复用了它，旧关键帧还在——别的帧指错，却被当成 conf=1.0 的标记读回）
    ad = getattr(obj, "animation_data", None)
    if ad is not None and (ad.action is not None or len(ad.drivers) or len(ad.nla_tracks)):
        return None, "keyed"
    if any(not c.mute and c.influence > 0.0 for c in obj.constraints):
        return None, "constrained"
    return obj, None


MARKER_IGNORE_REASONS = {
    "unbound": "没有骨骼父级（静止的世界方向）",
    "keyed": "自己带关键帧/驱动（在骨骼父级之上还在动，不是刚性定义）",
    "constrained": "带约束（Child Of 等：滚转不受控）",
}


def marker_dir(obj: Any) -> Vector:
    return (obj.matrix_world.to_quaternion() @ Vector((0.0, 0.0, 1.0))).normalized()


# ---------------------------------------------------------------------------
# 逐帧几何推导（每帧一个 dict，聚合在 probe() 里做）

def _hand_frame(armature: Any, side: str) -> dict | None:
    """掌心/手背/手指方向。掌心优先用网格标定（刚性局部向量），否则手指几何。"""
    d = _hand_frame_fingers(armature, side)
    if d is None:
        return None
    ev = d["evidence"]
    cal = palm_calibration(armature, side)
    pb = armature.pose.bones.get(f"hand_fk.{side}")
    mesh_palm = None
    if cal and cal.get("local") is not None and pb is not None:
        rot = (armature.matrix_world @ pb.matrix).to_quaternion()
        mesh_palm = (rot @ cal["local"]).normalized()
        ev["finger_palm_vs_mesh_deg"] = round(_angle_deg(d["palm"], mesh_palm), 1)
    mk, why = bound_marker("palm", side)
    if mk is not None:
        palm = marker_dir(mk)
        ev.update({"palm_source": "marker", "marker": mk.name,
                   "marker_bone": f"{mk.parent.name}/{mk.parent_bone}",
                   "sign_ambiguous": False})
        if mesh_palm is not None:
            ev["marker_vs_mesh_deg"] = round(_angle_deg(palm, mesh_palm), 1)
        d["palm"], d["back"] = palm, -palm
        d["confidence"], d["alternatives"] = 1.0, []
        return d
    if why in MARKER_IGNORE_REASONS:
        ev["marker_ignored"] = marker_name("palm", side)
        ev["marker_ignored_why"] = why
    if mesh_palm is not None:
        d["palm"], d["back"] = mesh_palm, -mesh_palm
        d["confidence"] = 0.6 if cal.get("sign_conflict") else 1.0
        d["alternatives"] = []
        ev.update({"palm_source": "mesh", "mesh_faces": cal["faces"],
                   "thumb_side_mm": cal["thumb_mm"], "sign_ambiguous": False})
        if cal.get("sign_conflict"):
            ev["sign_conflict"] = True
        return d
    ev["palm_source"] = "fingers"
    if cal and cal.get("reason"):
        ev["mesh_calibration"] = cal["reason"]
    return d


def _hand_frame_fingers(armature: Any, side: str) -> dict | None:
    """手指几何版：手指弯曲向量只能指向掌心侧 → 定号，无需分左右（网格标定的兜底）。"""
    seg = {}   # finger -> (root, j1, j2, tip) 世界点
    for finger, stem in _FINGER_STEMS.items():
        b1 = armature.pose.bones.get(f"{stem}.01.{side}")
        b2 = armature.pose.bones.get(f"{stem}.02.{side}")
        b3 = armature.pose.bones.get(f"{stem}.03.{side}")
        if b1 is None or b2 is None or b3 is None:
            continue
        seg[finger] = (
            armature.matrix_world @ b1.head,
            armature.matrix_world @ b1.tail,
            armature.matrix_world @ b2.tail,
            armature.matrix_world @ b3.tail,
        )
    avail = [f for f in _PALM_FINGERS if f in seg]
    if len(avail) < 2 or "index" not in seg or "pinky" not in seg:
        return None

    finger_dirs = [_norm(seg[f][3] - seg[f][0]) for f in avail]
    finger_dirs = [d for d in finger_dirs if d is not None]
    if not finger_dirs:
        return None
    finger_dir = _norm(sum(finger_dirs, Vector((0.0, 0.0, 0.0)))) 
    knuckle = _norm(seg["pinky"][0] - seg["index"][0])
    if finger_dir is None or knuckle is None:
        return None
    # 腕→指根中心：相对 hand_fk 刚性，与掌心法线近乎垂直（实测 7°）。掌心修复的次轴用它：
    # finger_dir 在手攥紧时会倒向掌心法线（拳头上只差 24°），双轴解算的滚转就没了依据。
    wrist = _pw(armature, f"hand_fk.{side}", "head")
    roots_c = sum((seg[f][0] for f in avail), Vector((0.0, 0.0, 0.0))) / len(avail)
    hand_axis = _norm(roots_c - wrist) if wrist is not None else None
    if hand_axis is None:
        hand_axis = finger_dir
    n0 = knuckle.cross(finger_dir)
    if n0.length < _EPS:
        return None                        # 指根连线 ∥ 手指方向，病态
    n0.normalize()

    # 弯曲向量：末节指向 − 根节指向；手指铰链只能向掌心侧弯 → 指向掌心。
    # 拇指同样只朝掌心侧收拢，权重减半（它的弯平面斜一些）。
    curl = Vector((0.0, 0.0, 0.0))
    for f in avail:
        s_first = _norm(seg[f][1] - seg[f][0])
        s_last = _norm(seg[f][3] - seg[f][2])
        if s_first is not None and s_last is not None:
            curl += s_last - s_first
    thumb_curl = Vector((0.0, 0.0, 0.0))
    if "thumb" in seg:
        t_first = _norm(seg["thumb"][1] - seg["thumb"][0])
        t_last = _norm(seg["thumb"][3] - seg["thumb"][2])
        if t_first is not None and t_last is not None:
            thumb_curl = t_last - t_first
    curl_all = curl + 0.5 * thumb_curl
    # 只要垂直于手指方向的分量参与定号
    curl_perp = curl_all - finger_dir * curl_all.dot(finger_dir)
    curl_mag = curl_perp.length
    palm = Vector(n0)
    if curl_mag > 0.03:
        if palm.dot(curl_perp) < 0:
            palm = -palm
        ambiguous = False
    else:
        ambiguous = True                   # 手指全直 → 符号两边都可能

    confidence = min(1.0, curl_mag / (0.30 * max(1, len(avail))))
    if ambiguous:
        confidence = min(confidence, 0.3)
    return {
        "palm": palm, "back": -palm, "finger_dir": finger_dir,
        "knuckle": knuckle, "hand_axis": hand_axis,
        "evidence": {
            "curl_mag": round(float(curl_mag), 3),
            "fingers_used": len(avail),
            "thumb_used": "thumb" in seg,
            "sign_ambiguous": ambiguous,
        },
        "confidence": confidence,
        "alternatives": [[*(-n0)], [*n0]] if ambiguous else [],
    }


def _foot_frame(armature: Any, side: str) -> dict | None:
    """脚底法线/脚尖方向。三点定平面，小腿在脚背侧定号——空中也成立。"""
    heel = _pw(armature, f"DEF-foot.{side}", "head")
    ball = _pw(armature, f"DEF-foot.{side}", "tail")
    toe = _pw(armature, f"DEF-toe.{side}", "tail")
    knee = _pw(armature, f"shin_fk.{side}", "head")
    if heel is None or ball is None or toe is None or knee is None:
        return None
    n0 = (ball - heel).cross(toe - ball)
    if n0.length < _EPS:
        return None
    n0.normalize()
    leg = _norm(heel - knee)               # 从膝指向踝 ≈ 小腿向下
    sole = Vector(n0)
    sign_evidence = None
    if leg is not None:
        d = sole.dot(leg)                  # 脚底背离小腿 → 法线与小腿同向为正
        if abs(d) > 0.05:
            if d < 0:
                sole = -sole
            sign_evidence = abs(d)
    toe_dir = _norm(toe - heel)
    if toe_dir is None:
        return None
    confidence = 0.9 if sign_evidence else 0.4
    out = {
        "sole": sole, "instep": -sole, "toe": toe_dir,
        "evidence": {"leg_sign": (round(sign_evidence, 3)
                                 if sign_evidence else None),
                     "sole_source": "bones"},
        "confidence": confidence,
        "alternatives": [] if sign_evidence else [[*(-n0)], [*n0]],
    }
    # 已知局限（2026-10-06 与用户确认，**不自动标定**）：三点法把 `DEF-foot` 的
    # head（**踝关节**）当脚后跟。厚底靴上踝比鞋底高 ~55 mm，三点几乎共面于脚的
    # 矢状面 → 法线偏到侧面（实测踩地帧仰角 −12.7°，应 ≈ −90°，偏 79°）。
    # 试过用靴子网格面法线聚类标定，实测仍差 60–70°，用户拍板"摆正让人来"：
    # **方向类修复请摆一支 `MCD_sole.{s}` 箭头并由用户肉眼确认**，以箭头为准
    # （见 markers/adopt；`markers check` 报"与几何定义差 N°"是预期的，那是让你拍板）。
    out["evidence"]["sole_bones_caveat"] = (
        "三点法（踝当脚跟）在厚底靴上可偏几十度；摆 MCD_sole 箭头并以它为准")
    mk, why = bound_marker("sole", side)
    if mk is not None:
        msole = marker_dir(mk)
        out["evidence"].update({"sole_source": "marker", "marker": mk.name,
                                "marker_bone": f"{mk.parent.name}/{mk.parent_bone}",
                                "marker_vs_bones_deg": round(_angle_deg(msole, sole), 1)})
        out["sole"], out["instep"] = msole, -msole
        out["confidence"], out["alternatives"] = 1.0, []
    elif why in MARKER_IGNORE_REASONS:
        out["evidence"]["marker_ignored"] = marker_name("sole", side)
        out["evidence"]["marker_ignored_why"] = why
    return out


def _finger_frame(armature: Any, side: str, finger: str) -> dict | None:
    """单根手指的指向（修手指用）：owner = 该指 01 节骨。"""
    stem = _FINGER_STEMS.get(str(finger).lower())
    if stem is None:
        raise RuntimeError(
            f"未知手指 {finger!r}，可用：{sorted(_FINGER_STEMS)}")
    b1 = armature.pose.bones.get(f"{stem}.01.{side}")
    b3 = armature.pose.bones.get(f"{stem}.03.{side}")
    if b1 is None or b3 is None:
        return None
    root = armature.matrix_world @ b1.head
    tip = armature.matrix_world @ b3.tail
    d = _norm(tip - root)
    if d is None:
        return None
    return {"finger_dir": d, "_owner": f"{stem}.01.{side}",
            "confidence": 1.0, "evidence": {"finger": finger},
            "alternatives": []}


# ---------------------------------------------------------------------------
# 膝 / 肘：铰链。2026-10-04 实测（arue 式重音テト 1499 帧，每 4 帧采样）：
#
# - 旧版读 thigh_fk/shin_fk——腿是 IK（thigh_parent["IK_FK"]=0），FK 骨"有 key 但看不见"：
#   膝位置与看得见的膝（ORG/DEF/MMD ひざ，三者 0.0 mm 一致）差中位数 40–46 mm、最大 236 mm，
#   凸出方向差中位数 13–18°、最大 49°。agent 量的膝根本不是用户看到的膝。→ 改读形变链 ORG-*。
# - 铰链轴（大小两段的弯曲平面法线）在哪根骨上是刚性的（弯 >20° 的帧，偏离均值的角度）：
#     膝：大腿 ORG-thigh 中位数 4.0/4.3°（p90 8°），小腿 ORG-shin 3.3/3.5°（p90 6.5°、最大 9°）
#     肘：上臂 ORG-upper_arm 16.5/18.9°（p90 31–34°，MMD 腕 24–28°——动捕的前臂会出平面摆），
#         前臂 ORG-forearm 5.4/8.6°（p90 13–19°）
#   所以朝向的刚性参照系取**远端骨**（小腿 / 前臂；肘尖鹰嘴本来就长在尺骨上）。
# - 朝向 = 远端骨上与铰链轴、远端骨都垂直、指向外凸侧的方向（膝盖骨 / 肘尖）：
#   g = s2 × n（n = s1 × s2 的标定均值）。直腿/直臂时同样有定义；与旧的"凸出角平分线"在弯曲
#   平面内恒差半个弯角（弯 30° 差 15°），投影到"根→梢连线的垂面"上则完全相同——
#   swivel（绕连线转）和 err_swivel 都只看这个投影。
# 来源优先级：标记箭头 MCD_knee.{s}/MCD_elbow.{s}（用户绑的，以它为准）> hinge（全片运动标定）
# > bend（当帧凸出角平分线；直腿时没有定义）。evidence.knee_source / elbow_source 标明。

_JOINTS = {
    "knee": {"pts": (("ORG-thigh.{s}", "head"), ("ORG-shin.{s}", "head"), ("ORG-foot.{s}", "head")),
             "fk_pts": (("thigh_fk.{s}", "head"), ("shin_fk.{s}", "head"), ("shin_fk.{s}", "tail")),
             "anchor": ("ORG-shin.{s}", "DEF-shin.{s}", "shin_fk.{s}"),
             "switch": "thigh_parent.{s}", "ik_ctrl": "thigh_ik.{s}", "fk_ctrl": "thigh_fk.{s}",
             "ik_end": "foot_ik.{s}", "fk_end": "foot_fk.{s}",
             "mmd_hinge": "ひざ.{s}"},
    "elbow": {"pts": (("ORG-upper_arm.{s}", "head"), ("ORG-forearm.{s}", "head"), ("ORG-hand.{s}", "head")),
              "fk_pts": (("upper_arm_fk.{s}", "head"), ("forearm_fk.{s}", "head"), ("forearm_fk.{s}", "tail")),
              "anchor": ("ORG-forearm.{s}", "DEF-forearm.{s}", "forearm_fk.{s}"),
              "switch": "upper_arm_parent.{s}", "ik_ctrl": "upper_arm_ik.{s}", "fk_ctrl": "upper_arm_fk.{s}",
              "ik_end": "hand_ik.{s}", "fk_end": "hand_fk.{s}",
              "mmd_hinge": None},
}
_HINGE_CAL: dict[tuple, dict] = {}
_HINGE_MIN_BEND, _HINGE_MAX_BEND = 20.0, 170.0


def reset_hinge_calibration() -> None:
    _HINGE_CAL.clear()


def limb_is_ik(armature: Any, kind: str, side: str) -> bool:
    """Rigify 的 IK/FK 开关（thigh_parent / upper_arm_parent 的 "IK_FK"：0 = IK，1 = FK）。"""
    pb = armature.pose.bones.get(_JOINTS[kind]["switch"].format(s=side))
    if pb is None or "IK_FK" not in pb.keys():
        return False
    try:
        return float(pb["IK_FK"]) < 0.5
    except (TypeError, ValueError):
        return False


def limb_control(armature: Any, kind: str, side: str) -> str | None:
    """能让膝/肘绕连线转的控制骨：IK 肢 = thigh_ik/upper_arm_ik（只有 Y 旋转有效），FK 肢 = thigh_fk/upper_arm_fk。"""
    spec = _JOINTS[kind]
    name = (spec["ik_ctrl"] if limb_is_ik(armature, kind, side) else spec["fk_ctrl"]).format(s=side)
    return name if armature.pose.bones.get(name) is not None else None


def limb_points(armature: Any, kind: str, side: str):
    """(根, 关节, 梢, 链名)：优先形变链 ORG-*（= 网格跟的骨），没有才退 FK。"""
    spec = _JOINTS[kind]
    for label, key in (("ORG", "pts"), ("FK", "fk_pts")):
        pts = [_pw(armature, b.format(s=side), e) for b, e in spec[key]]
        if all(p is not None for p in pts):
            return pts[0], pts[1], pts[2], label
    return None, None, None, None


def _limb_anchor(armature: Any, kind: str, side: str) -> str | None:
    for b in _JOINTS[kind]["anchor"]:
        if armature.pose.bones.get(b.format(s=side)) is not None:
            return b.format(s=side)
    return None


def _calibrate_hinge(armature: Any, side: str, kind: str, scene: Any) -> dict:
    anchor = _limb_anchor(armature, kind, side)
    if anchor is None:
        return {"reason": "missing_limb_bones"}
    f0, f1 = int(scene.frame_start), int(scene.frame_end)
    step = max(1, (f1 - f0 + 1) // 300)
    rows = []
    mmd_hinge = _JOINTS[kind]["mmd_hinge"]
    mmd = getattr(getattr(scene, "mocap_doctor", None), "mmd_armature", None)
    mmd_pb = (mmd.pose.bones.get(mmd_hinge.format(s=side))
              if (mmd is not None and mmd_hinge) else None)
    with preserve_scene_frame(scene):
        for f in range(f0, f1 + 1, step):
            set_scene_frame(scene, f)
            p0, p1, p2, _chain = limb_points(armature, kind, side)
            if p0 is None:
                continue
            s1, s2 = _norm(p1 - p0), _norm(p2 - p1)
            if s1 is None or s2 is None:
                continue
            bend = _angle_deg(s1, s2)
            q = (armature.matrix_world @ armature.pose.bones[anchor].matrix).to_quaternion()
            n = s1.cross(s2)
            mx = None
            if mmd_pb is not None:
                mq = (mmd.matrix_world @ mmd_pb.matrix).to_quaternion()
                mx = mq @ Vector((1.0, 0.0, 0.0))
            rows.append({"bend": bend, "q": q, "s1": s1, "s2": s2,
                         "n": n.normalized() if n.length > _EPS else None,
                         "s2_local": q.inverted() @ s2, "mmd_x": mx})
    sel = [r for r in rows if _HINGE_MIN_BEND <= r["bend"] <= _HINGE_MAX_BEND and r["n"] is not None]
    if len(sel) < 8:
        return {"reason": "too_few_bending_frames", "n": len(sel), "anchor": anchor}
    acc = Vector()
    for r in sel:
        r["n_local"] = (r["q"].inverted() @ r["n"]).normalized()
        acc += r["n_local"] * float(np.sin(np.radians(r["bend"])))
    n_cal = acc.normalized()
    devs = np.array([_angle_deg(n_cal, r["n_local"]) for r in sel])
    s2m = _norm(sum((r["s2_local"] for r in rows), Vector()))
    front_local = _norm(s2m.cross(n_cal))
    if front_local is None:
        return {"reason": "degenerate_hinge", "anchor": anchor}
    signs, halves, mmd_dev = [], [], []
    for r in sel:
        fw = r["q"] @ front_local
        bis = _norm(r["s1"] - r["s2"])
        if bis is None:
            continue
        signs.append(float(fw @ bis) > 0.0)
        halves.append(abs(_angle_deg(fw, bis) - 0.5 * r["bend"]))
        if r["mmd_x"] is not None:
            nw = r["q"] @ n_cal
            mmd_dev.append(min(_angle_deg(nw, r["mmd_x"]), _angle_deg(nw, -r["mmd_x"])))
    out = {"front_local": front_local, "axis_local": n_cal, "anchor": anchor,
           "frames": len(rows), "bend_frames": len(sel), "step": step,
           "spread_med_deg": round(float(np.median(devs)), 1),
           "spread_p90_deg": round(float(np.percentile(devs, 90)), 1),
           "spread_max_deg": round(float(devs.max()), 1),
           "sign_ok_frac": round(float(np.mean(signs)), 3) if signs else None,
           "vs_bisector_minus_half_med_deg": round(float(np.median(halves)), 1) if halves else None}
    if mmd_dev:
        out["vs_mmd_hinge_axis_med_deg"] = round(float(np.median(mmd_dev)), 1)
    return out


def _base_action_name(armature: Any) -> str:
    anim = getattr(armature, "animation_data", None)
    if anim is None:
        return ""
    if anim.action is not None:
        return anim.action.name
    for tr in anim.nla_tracks:
        if tr.name == "mcd_base" and tr.strips:
            act = tr.strips[0].action
            return act.name if act is not None else ""
    return ""


def hinge_calibration(armature: Any, side: str, kind: str, scene: Any = None) -> dict:
    """膝/肘铰链的全片运动标定（缓存）：远端骨局部坐标里的铰链轴 + 朝向向量 + 刚性统计。

    键 = 骨架身份 + 基底动作 + 帧段（换文件 / 换动作就重标；读文件时 reset_caches 清空）。
    结构性失败（缺骨、这段动作里弯得不够）照样缓存（重算也一样，且要 0.8 s）；异常不缓存。"""
    scene = scene if scene is not None else bpy.context.scene
    key = (armature.name, armature.as_pointer(), _base_action_name(armature), side, kind,
           int(scene.frame_start), int(scene.frame_end))
    hit = _HINGE_CAL.get(key)
    if hit is not None:
        return hit
    try:
        res = _calibrate_hinge(armature, side, kind, scene)
    except Exception as exc:  # noqa: BLE001 - 标定只是增强，失败退回当帧弯曲方向
        return {"reason": f"error: {exc}"}
    _HINGE_CAL[key] = res
    return res


def hinge_public(cal: dict | None) -> dict | None:
    """标定结果里能放进 JSON 的部分。"""
    if not cal:
        return None
    out = {k: v for k, v in cal.items() if not isinstance(v, Vector)}
    for k in ("front_local", "axis_local"):
        if isinstance(cal.get(k), Vector):
            out[k] = [round(x, 4) for x in cal[k]]
    return out


def _joint_frame(armature: Any, side: str, kind: str) -> dict | None:
    """膝/肘朝向（膝盖骨 / 肘尖）。kind='knee'|'elbow'。见上面的实测说明。"""
    p0, p1, p2, chain = limb_points(armature, kind, side)
    if p0 is None:
        return None
    s1, s2 = _norm(p1 - p0), _norm(p2 - p1)
    chord = _norm(p2 - p0)
    if s1 is None or s2 is None or chord is None:
        return None
    bend = _angle_deg(s1, s2)
    owner = limb_control(armature, kind, side)
    ev: dict = {"bend_deg": round(bend, 1), "chain": chain,
                "limb": "IK" if limb_is_ik(armature, kind, side) else "FK"}
    bis = _norm(s1 - s2) if bend > 3.0 else None
    mk, why = bound_marker(kind, side)
    cal = hinge_calibration(armature, side, kind)
    hinge_front = hinge_rigid = None
    follow = 0.0
    if cal and isinstance(cal.get("front_local"), Vector):
        pb = armature.pose.bones.get(cal["anchor"])
        if pb is not None:
            q = (armature.matrix_world @ pb.matrix).to_quaternion()
            hinge_rigid = (q @ cal["front_local"]).normalized()
            hinge_front = hinge_rigid
            # 明显弯着的帧（>15° 起渐变，35° 全跟）用当帧的弯曲平面：动捕的肘不是理想铰链（前臂坐标里
            # 铰链轴 p90 散布 14–20°），用户看到的肘尖/膝盖 = 当帧那个角；直的时候用刚性标定（当帧没有定义）
            n_cal_w = (q @ cal["axis_local"]).normalized()
            n_f = _norm(s1.cross(s2))
            if n_f is not None and float(n_f @ n_cal_w) > 0.0:
                x = max(0.0, min(1.0, (bend - 15.0) / 20.0))
                follow = x * x * (3.0 - 2.0 * x)
            if follow > 0.0:
                n_b = _norm(n_cal_w * (1.0 - follow) + n_f * follow) or n_cal_w
                hinge_front = _norm(s2.cross(n_b)) or hinge_rigid
    if mk is not None:
        front, src, conf = marker_dir(mk), "marker", 1.0
        ev.update({"marker": mk.name, "marker_bone": f"{mk.parent.name}/{mk.parent_bone}"})
        if hinge_front is not None:
            ev["marker_vs_hinge_deg"] = round(_angle_deg(front, hinge_front), 1)
    elif hinge_front is not None:
        front, src = hinge_front, "hinge"
        conf = 1.0 if (cal.get("spread_p90_deg") or 99) < 20.0 else 0.7
        ev["follow_bend"] = round(follow, 2)
    elif bis is not None:
        front, src, conf = bis, "bend", min(1.0, bend / 30.0)
        if cal and cal.get("reason"):
            ev["hinge_calibration"] = cal["reason"]
    else:
        ev[f"{kind}_source"] = "none"
        return {"front": None, "chord": chord, "s1": s1, "s2": s2, "joint": p1,
                "confidence": 0.0, "evidence": ev, "alternatives": [], "owner": owner}
    if why in MARKER_IGNORE_REASONS:
        ev["marker_ignored"] = marker_name(kind, side)
        ev["marker_ignored_why"] = why
    ev[f"{kind}_source"] = src
    if bis is not None and src != "bend":
        ev["vs_bisector_deg"] = round(_angle_deg(front, bis), 1)
    off = (p1 - p0) - chord * float((p1 - p0) @ chord)
    pole = off.normalized() if off.length > 0.01 else None      # 关节离连线 > 1 cm 才算有方向
    return {"front": front, "front_rigid": hinge_rigid or front, "chord": chord, "s1": s1, "s2": s2,
            "joint": p1, "pole": pole, "confidence": conf, "evidence": ev, "alternatives": [], "owner": owner}


def swivel_error_deg(front: Vector | None, target: Vector | None, chord: Vector | None,
                     min_perp: float = 0.2, pole: Vector | None = None):
    """绕"根→梢连线"还差多少度（有符号：正 = 绕连线按右手定则转）；退化（目标/朝向几乎平行于
    连线，夹角 < ~11.5°）返回 None。人说"膝盖朝前"只关心这一个自由度。

    pole：关节偏离连线的方向（弯曲平面内、⊥连线、外凸侧）。肘弯到 120°+ 时"⊥前臂"的朝向几乎沿连线，
    投影太短；这时用 pole（同一个弯曲平面，方向完全一致）。"""
    if front is None or target is None or chord is None:
        return None
    p = front - chord * float(front @ chord)
    q = target - chord * float(target @ chord)
    if p.length < min_perp and pole is not None:
        p2 = pole - chord * float(pole @ chord)
        if p2.length > _EPS and (p.length < _EPS or float(p2 @ p) >= 0.0):
            p = p2.normalized()
    if p.length < min_perp or q.length < min_perp:
        return None
    p.normalize()
    q.normalize()
    return float(np.degrees(np.arctan2(float(chord @ p.cross(q)), float(p @ q))))


# ---------------------------------------------------------------------------
# 头 / 胸 / 骨盆：单根骨上的"前方"（静止姿态里角色朝 −Y，随骨转；MMD 头前方与两眼骨方向实测差 1.6°）

_BODY = {"face": {"rig": ("ORG-spine.006", "DEF-spine.006"), "mmd": "頭", "owner": ("head",)},
         "chest": {"rig": ("ORG-spine.003", "DEF-spine.003"), "mmd": "上半身2",
                   "owner": ("spine_fk.003", "chest")},
         "pelvis": {"rig": ("ORG-spine", "DEF-spine"), "mmd": "下半身",
                    "owner": ("torso_root", "hips", "torso")}}


def _body_part_frame(armature: Any, part: str) -> dict | None:
    from . import agent_view
    spec = _BODY[part]
    bone = next((b for b in spec["rig"] if armature.pose.bones.get(b) is not None), None)
    if bone is None:
        return None
    pb = armature.pose.bones[bone]
    q = (armature.matrix_world @ pb.matrix).to_quaternion()
    front = (q @ agent_view.rest_front_local(armature, bone)).normalized()
    up = (q @ agent_view.rest_up_local(armature, bone)).normalized()
    owner = next((b for b in spec["owner"] if armature.pose.bones.get(b) is not None), bone)
    ev = {f"{part}_source": "rest", "bone": bone}
    mk, why = bound_marker(part, None)
    if mk is not None:
        m = marker_dir(mk)
        ev.update({f"{part}_source": "marker", "marker": mk.name,
                   "marker_bone": f"{mk.parent.name}/{mk.parent_bone}",
                   "marker_vs_rest_deg": round(_angle_deg(m, front), 1)})
        front = m
    elif why in MARKER_IGNORE_REASONS:
        ev["marker_ignored"] = marker_name(part, None)
        ev["marker_ignored_why"] = why
    return {part: front, "up_axis": up, "confidence": 1.0, "evidence": ev,
            "alternatives": [], "_owner": owner}


def _body_forward(armature: Any, scene: Any) -> dict | None:
    """身体前方 = 角色躯干朝向（骨盆 + 胸，水平）；脚尖方向只作证据。

    2026-10-04 改：以前以脚尖为主——脚常外八，与躯干朝向差中位数 8°、最大 42°（fixture），
    而人说"朝前"指躯干。旧版还把相机**视线**方向当"前方"的兜底证据（相机在正面时恰好反了）。"""
    from . import agent_view
    mmd = getattr(getattr(scene, "mocap_doctor", None), "mmd_armature", None)
    cf = agent_view.char_frame(armature, mmd=mmd)
    old = _body_forward_feet(armature, scene)
    if cf is None:
        return old
    ev = dict(cf["evidence"])
    ev["source"] = "torso"
    conf = 0.9
    if ev.get("pelvis_vs_chest_deg", 0.0) > 60.0:
        conf = 0.5
    if old is not None:
        if old["evidence"].get("feet_count"):
            feet = old.get("feet_forward")
            if feet is not None:
                ev["feet_forward"] = [round(v, 3) for v in feet]
                ev["feet_vs_torso_deg"] = round(_angle_deg(feet, cf["forward"]), 1)
    return {"forward": Vector(cf["forward"]), "confidence": conf, "evidence": ev,
            "alternatives": []}


def _body_forward_feet(armature: Any, scene: Any) -> dict | None:
    """旧版身体前方（脚尖为主，肩线交叉证据）：现在只给 _body_forward 当证据。"""
    up = Vector((0, 0, 1))
    feet = []
    for side in ("L", "R"):
        heel = _pw(armature, f"DEF-foot.{side}", "head")
        toe = _pw(armature, f"DEF-toe.{side}", "tail")
        if heel is None or toe is None:
            continue
        d = toe - heel
        d.z = 0.0
        d = _norm(d)
        if d is not None:
            feet.append(d)
    evidence = {}
    fwd = None
    if feet:
        fwd = _norm(sum(feet, Vector((0.0, 0.0, 0.0))))
        evidence["feet_count"] = len(feet)
        if len(feet) == 2:
            evidence["feet_spread_deg"] = round(_angle_deg(feet[0], feet[1]), 1)
    # 肩线交叉：角色右肩−左肩 = 右手侧；up × (R−L) = 前方（朝向 −Y、左肩 +X 时）
    sl = _pw(armature, "shoulder.L", "head")
    sr = _pw(armature, "shoulder.R", "head")
    fwd_sh = None
    if sl is not None and sr is not None:
        s_vec = _norm(sr - sl)
        if s_vec is not None:
            fwd_sh = _norm(up.cross(s_vec))
            if fwd_sh is not None:
                evidence["shoulder_fwd"] = [round(v, 3) for v in fwd_sh]
    feet_fwd = fwd
    if fwd is None:
        fwd = fwd_sh
    if fwd is None:
        return None
    # 脚尖与肩线矛盾（例如转身瞬间）→ 降置信度
    conf = 0.9
    if fwd_sh is not None:
        disagree = _angle_deg(fwd, fwd_sh)
        evidence["toe_vs_shoulder_deg"] = round(disagree, 1)
        if disagree > 60:
            conf = 0.4
    if len(feet) == 2 and evidence.get("feet_spread_deg", 0) > 60:
        conf = min(conf, 0.5)
    return {"forward": fwd, "confidence": conf, "evidence": evidence,
            "alternatives": [], "feet_forward": feet_fwd}


# ---------------------------------------------------------------------------
# 聚合与对外入口

def _aggregate(frames_data: list[dict], key: str) -> dict:
    """跨帧聚合一个方向：均值向量 + 最大散布角。"""
    dirs = [d[key] for d in frames_data if d.get(key) is not None]
    if not dirs:
        return {}
    mean = _norm(sum(dirs, Vector((0.0, 0.0, 0.0))))
    spread = max(_angle_deg(mean, d) for d in dirs)
    return {"world_dir": [round(v, 4) for v in mean],
            "spread_deg": round(spread, 1), "n": len(dirs)}


def _resolve_toward(scene: Any, armature: Any, toward: Any, origin: Vector | None,
                    view: str = "camera", side: str | None = None) -> tuple[Vector | None, str]:
    """把 toward 解析成世界向量：三元组 / 方向词 / 物体名（agent_view.resolve_direction）。

    2026-10-04 改：`"camera"` 以前返回相机的**视线**方向（压平）= 背对镜头，正好反了；现在是从部位
    指向镜头（逐帧、透视）。`"forward"` 以前以脚尖为主，现在是躯干朝向。新增 char_left/char_right/back、
    screen_*、viewer、away。裸 "left"/"right" 有歧义，直接报错（让调用者说清楚是角色左还是画面左）。"""
    if toward is None:
        return None, ""
    from . import agent_view
    mmd = getattr(getattr(scene, "mocap_doctor", None), "mmd_armature", None)
    return agent_view.resolve_direction(toward, scene=scene, armature=armature, origin=origin,
                                        view=view, mmd=mmd, side=side)


_HAND_PARTS = ("palm", "back_of_hand", "finger_dir", "knuckle", "hand_axis")
_FOOT_PARTS = ("sole", "instep", "toe")
_JOINT_PARTS = ("knee_front", "elbow_front")
_BODY_PARTS = ("face", "chest", "pelvis")
_PARTS = _HAND_PARTS + _FOOT_PARTS + _JOINT_PARTS + _BODY_PARTS + ("body_forward", "bone_axis")
PART_ALIASES = {"knee": "knee_front", "elbow": "elbow_front", "head": "face",
                "torso": "body_forward", "hips": "pelvis"}


def canonical_part(part: str) -> str:
    p = str(part).strip().lower()
    return PART_ALIASES.get(p, p)


def part_anchor(armature: Any, part: str, side: str | None = None) -> Vector | None:
    """部位在世界里的位置（箭头放哪、"朝镜头"从哪儿量）：掌心皮肤中心 / 脚底三点中心 / 膝肘关节 /
    头中部 / 胸 / 骨盆 / 任意骨头部。"""
    part = canonical_part(part)
    if part in _HAND_PARTS or part == "finger":
        if side not in ("L", "R"):
            return None
        cal = palm_calibration(armature, side)
        pb = armature.pose.bones.get(f"hand_fk.{side}")
        if cal and cal.get("centre_local") is not None and pb is not None:
            return (armature.matrix_world @ pb.matrix) @ cal["centre_local"]
        roots = [_pw(armature, f"{_FINGER_STEMS[f]}.01.{side}") for f in _PALM_FINGERS]
        roots = [r for r in roots if r is not None]
        wrist = _pw(armature, f"hand_fk.{side}")
        if roots and wrist is not None:
            return (wrist + sum(roots, Vector()) / len(roots)) * 0.5
        return wrist
    if part in _FOOT_PARTS:
        pts = [_pw(armature, f"DEF-foot.{side}", "head"), _pw(armature, f"DEF-foot.{side}", "tail"),
               _pw(armature, f"DEF-toe.{side}", "tail")]
        pts = [q for q in pts if q is not None]
        return sum(pts, Vector()) / len(pts) if pts else None
    if part in _JOINT_PARTS:
        _p0, p1, _p2, _c = limb_points(armature, "knee" if part == "knee_front" else "elbow", side)
        return p1
    if part in _BODY_PARTS:
        spec = _BODY[part]
        bone = next((b for b in spec["rig"] if armature.pose.bones.get(b) is not None), None)
        if bone is None:
            return None
        h, t = _pw(armature, bone, "head"), _pw(armature, bone, "tail")
        if part == "face":
            return h + (t - h) * 0.45
        if part == "chest":
            return h + (t - h) * 0.5
        return h
    if part == "body_forward":
        return _pw(armature, "torso_root") or _pw(armature, "ORG-spine")
    return None


def _eval_part(scene: Any, armature: Any, part: str, side_n: str | None,
               finger: str | None, bone: str | None) -> dict | None:
    """一帧的解剖方向（调用前已 set_scene_frame）。"""
    d = None
    if part in _HAND_PARTS:
        if side_n not in ("L", "R"):
            raise RuntimeError(f"{part} 需要 side='L'/'R'")
        if part == "finger_dir" and finger:
            d = _finger_frame(armature, side_n, finger)
        else:
            d = _hand_frame(armature, side_n)
        if d is not None and "_owner" not in d:
            d["_owner"] = f"hand_fk.{side_n}"
    elif part in _FOOT_PARTS:
        if side_n not in ("L", "R"):
            raise RuntimeError(f"{part} 需要 side='L'/'R'")
        d = _foot_frame(armature, side_n)
        if d is not None:
            d["_owner"] = f"foot_ik.{side_n}"
    elif part in _JOINT_PARTS:
        if side_n not in ("L", "R"):
            raise RuntimeError(f"{part} 需要 side='L'/'R'")
        d = _joint_frame(armature, side_n, "knee" if part == "knee_front" else "elbow")
        if d is not None and d.get("owner"):
            d["_owner"] = d["owner"]
    elif part in _BODY_PARTS:
        d = _body_part_frame(armature, part)
    elif part == "body_forward":
        d = _body_forward(armature, scene)
    elif part == "bone_axis":
        if not bone:
            raise RuntimeError("bone_axis 需要 bone=<骨名>")
        pb = armature.pose.bones.get(bone)
        if pb is None:
            raise RuntimeError(f"骨不存在：{bone}")
        rot = (armature.matrix_world @ pb.matrix).to_quaternion()
        d = {"_axes": {k: rot @ Vector(v) for k, v in _AXIS_SET.items()}, "_owner": bone}
    return d


_KEY_MAP = {"palm": "palm", "back_of_hand": "back",
            "finger_dir": "finger_dir", "knuckle": "knuckle", "hand_axis": "hand_axis",
            "sole": "sole", "instep": "instep", "toe": "toe",
            "knee_front": "front", "elbow_front": "front",
            "face": "face", "chest": "chest", "pelvis": "pelvis",
            "body_forward": "forward"}
_SEC_KEY = {"palm": "hand_axis", "back_of_hand": "hand_axis", "hand_axis": "palm",
            "sole": "toe", "instep": "toe",
            "knee_front": "s1", "elbow_front": "s1",
            "face": "up_axis", "chest": "up_axis", "pelvis": "up_axis"}


def _err_one(part: str, d: dict, key: str, t: Vector):
    """一帧的朝向误差（度）。膝/肘 = 绕根→梢连线的转角误差（swivel 平面）；退化帧 None。"""
    if part in _JOINT_PARTS:
        e = swivel_error_deg(d[key], t, d.get("chord"), pole=d.get("pole"))
        return None if e is None else abs(e)
    return _angle_deg(d[key], t)


def _fill_toward(out: dict, part: str, frames_data: list, targets: list, key: str,
                 tv: Vector | None, how: str, toward: Any) -> None:
    out["toward_resolved"] = [round(v, 4) for v in tv] if tv else None
    out["toward_how"] = how
    if isinstance(toward, str):
        out["toward_per_frame"] = True
    if tv is None:
        return
    pairs, degen = [], 0
    for (f, d), t2 in zip(frames_data, targets):
        if d.get(key) is None:
            continue
        if isinstance(t2, Exception):
            raise t2
        if t2 is None:
            continue
        e = _err_one(part, d, key, t2)
        if e is None:
            degen += 1
            continue
        pairs.append((int(f), e))
    if part in _JOINT_PARTS:
        out["err_metric"] = "swivel"
        out["err_metric_note"] = ("膝/肘误差 = 投影到 根→梢 连线垂面上的夹角（swivel 能修的唯一自由度；"
                                  "目标几乎平行于连线的帧不计）")
        out["swivel_degenerate_frames"] = degen
    if not pairs:
        return
    errs = [e for _f, e in pairs]
    out["err_max_deg"] = round(max(errs), 1)
    out["err_mean_deg"] = round(float(np.mean(errs)), 1)
    out["err_per_frame"] = [[f, round(e, 1)] for f, e in pairs]
    inner = out["err_per_frame"][1:-1] if len(out["err_per_frame"]) >= 5 else out["err_per_frame"]
    if inner:
        out["err_inner_deg"] = round(max(e for _, e in inner), 1)


def _fill_common(out: dict, part: str, side_n, owner, frames_data: list, extra: list,
                 key, sec_key, finger, toward) -> None:
    if part == "bone_axis":
        last = frames_data[-1][1]["_axes"]
        out["axes"] = {k: [round(x, 4) for x in v] for k, v in last.items()}
        out["confidence"] = 1.0
    else:
        agg = _aggregate([d for _, d in frames_data], key)
        if not agg:
            raise RuntimeError(f"{part} 在所采帧上推不出方向")
        out.update(agg)
        confs = [d.get("confidence", 0.0) for _, d in frames_data]
        out["confidence"] = round(min(confs), 2)
        out["evidence"] = dict(frames_data[0][1].get("evidence", {}))
        if out["spread_deg"] > 45:
            out["evidence"]["spread_note"] = \
                "方向随帧变化大，修复会逐帧对齐，属正常"
        alts = [d.get("alternatives") for _, d in frames_data
                if d.get("alternatives")]
        if alts:
            out["alternatives"] = alts[0]
    if owner and key:
        locals_ = [e[1] for (_f, d), e in zip(frames_data, extra)
                   if d.get(key) is not None and e[1] is not None]
        if locals_:
            lm = _norm(sum(locals_, Vector((0.0, 0.0, 0.0))))
            out["local_axis"] = [round(v, 4) for v in lm]
            out["local_spread_deg"] = round(
                max(_angle_deg(lm, v) for v in locals_), 1)
            if sec_key:
                secs = [e[2] for (_f, d), e in zip(frames_data, extra)
                        if d.get(sec_key) is not None and e[2] is not None]
                if secs:
                    sm = _norm(sum(secs, Vector((0.0, 0.0, 0.0))))
                    out["secondary_axis"] = [round(v, 4) for v in sm]
                    out["secondary_name"] = sec_key
            hp = _hold_pose_args(part, side_n, owner, out, finger)
            if hp is not None:
                out["hold_pose_args"] = hp
    if part in _JOINT_PARTS:
        kind = "knee" if part == "knee_front" else "elbow"
        args = {"joint": kind, "side": side_n}
        if toward is not None:
            args["toward"] = toward
        out["swivel_args"] = args


def probe(scene: Any, armature: Any, *, part: str, side: str | None = None,
          bone: str | None = None, finger: str | None = None,
          frame_range: Sequence[int] | None = None, toward: Any = None,
          max_frames: int = 9, view: str = "camera") -> dict:
    """解剖探头入口。返回世界方向、owner 骨局部向量、置信度、证据。

    part:
      palm / back_of_hand / finger_dir / knuckle / hand_axis   —— 需 side（hand_axis = 腕→指根，刚性）
      sole / instep / toe                          —— 需 side
      knee_front / elbow_front（别名 knee / elbow） —— 需 side；膝盖骨 / 肘尖朝向（远端骨上的刚性方向，
                                                       见 _joint_frame）；误差按 swivel 平面算
      face / chest / pelvis（别名 head / hips）     —— 脸 / 胸 / 骨盆的前方
      body_forward                                 —— 全身（躯干朝向，水平）
      bone_axis                                    —— 需 bone（任意骨六根主轴）
      finger（finger_dir + finger 名）              —— 单根手指指向
    toward 可选：解析目标方向并附 err_deg（修后复测就是同一调用加 toward）。方向词见 agent_view.WORD_TABLE。
    view："camera"（默认，scene.camera）/ "viewer"（用户的 3D 视口）——screen_* / away 按它解释。

    任务2：原版对同一组采样帧扫 4 遍（取方向 / 局部主轴 / 次轴 / toward 误差），
    每遍都 frame_set。每帧求值是确定的，所以这里一遍扫帧里把 4 样都算掉
    （≈4× 少 frame_set），输出与 _probe_slow 逐位相同（bench golden 校验）。
    唯一不能合并的情形——owner 骨在采样帧间变化——回落到 _probe_slow。
    """
    part = canonical_part(part)
    if part not in _PARTS and part != "finger":
        raise RuntimeError(f"未知 part {part!r}，可用：{_PARTS} + finger（别名：{PART_ALIASES}）")
    part_in = part
    if part == "finger":
        part = "finger_dir"
    side_n = (side or "").upper() or None
    frames = _sample_frames(scene, frame_range, max_frames)
    key = None if part == "bone_axis" else _KEY_MAP[part]
    sec_key = _SEC_KEY.get(part)
    hcal = None
    if part in _JOINT_PARTS and side_n in ("L", "R"):
        hcal = hinge_calibration(armature, side_n, "knee" if part == "knee_front" else "elbow", scene)
    str_toward = isinstance(toward, str)

    frames_data = []
    owner = None
    extra = []          # per frame: (owner_used, local_key, local_sec, t2|exc)
    with preserve_scene_frame(scene):
        for f in frames:
            set_scene_frame(scene, f)
            d = _eval_part(scene, armature, part, side_n, finger, bone)
            if d:
                frames_data.append((f, d))
                if d.get("_owner"):
                    owner = d["_owner"]
                # 同一帧顺手算后三遍要的量（用"到目前为止的 owner"，事后核对）
                lk = ls = t2 = None
                if owner and key:
                    if d.get(key) is not None:
                        lk = _local_of(armature, owner, d[key])
                    if sec_key and d.get(sec_key) is not None:
                        ls = _local_of(armature, owner, d[sec_key])
                if toward is not None and key and d.get(key) is not None:
                    try:
                        origin = (part_anchor(armature, part, side_n) if str_toward
                                  else (_pw(armature, owner, "head") if owner else None))
                        t2, _ = _resolve_toward(scene, armature, toward, origin, view, side_n)
                    except Exception as exc:  # noqa: BLE001 - replayed below
                        t2 = exc
                extra.append((owner, lk, ls, t2))

    if not frames_data:
        raise RuntimeError(f"{part} 所需骨骼在场景里不存在（side={side_n}）")
    if any(e[0] != owner for e in extra):
        return _probe_slow(scene, armature, part=part_in, side=side,
                           bone=bone, finger=finger, frame_range=frame_range,
                           toward=toward, max_frames=max_frames, view=view)

    out: dict[str, Any] = {"part": part, "side": side_n,
                           "frames_sampled": frames,
                           "owner_bone": owner}
    _fill_common(out, part, side_n, owner, frames_data, extra, key, sec_key, finger, toward)
    if hcal is not None:
        out["hinge_calibration"] = hinge_public(hcal)
    if toward is not None and key:
        origin = (part_anchor(armature, part, side_n) if str_toward
                  else (_pw(armature, owner, "head") if owner else None))
        tv, how = _resolve_toward(scene, armature, toward, origin, view, side_n)
        _fill_toward(out, part, frames_data, [e[3] for e in extra], key, tv, how, toward)
    return out


def _probe_slow(scene: Any, armature: Any, *, part: str, side: str | None = None,
                bone: str | None = None, finger: str | None = None,
                frame_range: Sequence[int] | None = None, toward: Any = None,
                max_frames: int = 9, view: str = "camera") -> dict:
    """原版（多遍扫帧）实现：owner 在采样帧间不一致时的兜底，也是逐位对照基准。
    每帧的解剖方向与快路径共用 _eval_part；局部轴 / toward 误差逐帧重新 frame_set 求值。"""
    part = canonical_part(part)
    if part not in _PARTS and part != "finger":
        raise RuntimeError(f"未知 part {part!r}，可用：{_PARTS} + finger")
    if part == "finger":
        part = "finger_dir"
    side = (side or "").upper() or None
    frames = _sample_frames(scene, frame_range, max_frames)
    key = None if part == "bone_axis" else _KEY_MAP[part]
    sec_key = _SEC_KEY.get(part)
    str_toward = isinstance(toward, str)

    frames_data = []
    owner = None
    with preserve_scene_frame(scene):
        for f in frames:
            set_scene_frame(scene, f)
            d = _eval_part(scene, armature, part, side, finger, bone)
            if d:
                frames_data.append((f, d))
                if d.get("_owner"):
                    owner = d["_owner"]
    if not frames_data:
        raise RuntimeError(f"{part} 所需骨骼在场景里不存在（side={side}）")

    extra = []
    targets = []
    with preserve_scene_frame(scene):
        for f, d in frames_data:
            set_scene_frame(scene, f)
            lk = _local_of(armature, owner, d[key]) if (owner and key and d.get(key) is not None) else None
            ls = (_local_of(armature, owner, d[sec_key])
                  if (owner and key and sec_key and d.get(sec_key) is not None) else None)
            extra.append((owner, lk, ls, None))
            t2 = None
            if toward is not None and key and d.get(key) is not None:
                origin = (part_anchor(armature, part, side) if str_toward
                          else (_pw(armature, owner, "head") if owner else None))
                t2, _ = _resolve_toward(scene, armature, toward, origin, view, side)
            targets.append(t2)

    out: dict[str, Any] = {"part": part, "side": side,
                           "frames_sampled": frames,
                           "owner_bone": owner}
    _fill_common(out, part, side, owner, frames_data, extra, key, sec_key, finger, toward)
    if part in _JOINT_PARTS and side in ("L", "R"):
        out["hinge_calibration"] = hinge_public(
            hinge_calibration(armature, side, "knee" if part == "knee_front" else "elbow", scene))
    if toward is not None and key:
        origin = (part_anchor(armature, part, side) if str_toward
                  else (_pw(armature, owner, "head") if owner else None))
        tv, how = _resolve_toward(scene, armature, toward, origin, view, side)
        _fill_toward(out, part, frames_data, targets, key, tv, how, toward)
    return out


_AXIS_SET = {"+X": (1, 0, 0), "-X": (-1, 0, 0),
             "+Y": (0, 1, 0), "-Y": (0, -1, 0),
             "+Z": (0, 0, 1), "-Z": (0, 0, -1)}

_FRAME_KEYS = dict(_KEY_MAP)


# probe 返回的 hold_pose_args = 推荐写法（与剧本 30 的表一致）：逐帧 probe 主轴 + 次轴
# 掌心/手背的次轴是 hand_axis（腕→指根，刚性、⊥掌心）而不是 finger_dir：手攥紧时 finger_dir
# 倒向掌心法线，双轴解算的滚转失去依据（2026-10-04 掌心网格标定后发现）。
# 膝/肘不给 hold_pose_args，给 swivel_args（§16：hold_pose 把一根骨的局部轴对准目标会顺带改大腿/上臂的
# 指向，而且 IK 腿的 thigh_fk 写了看不见）。
_HOLD_AXES = {"palm": ("palm", "hand_axis"), "back_of_hand": ("back_of_hand", "hand_axis"),
              "finger_dir": ("finger_dir", "palm"), "knuckle": ("knuckle", "palm"),
              "hand_axis": ("hand_axis", "palm"),
              "sole": ("sole", "toe"), "instep": ("instep", "toe"), "toe": ("toe", "sole")}


def _hold_pose_args(part, side, owner, out, finger=None) -> dict | None:
    """"可直接展开给 hold_pose"的参数必须是推荐写法：逐帧 "probe:<part>.<side>" 轴。

    以前给的是全段平均的固定局部轴——解剖方向相对控制骨随帧变（手指实测散布 ~70°），
    均值轴对齐后每帧留几十度残差；照抄返回值的弱模型会踩这个坑（sonnet 第三轮指出）。
    单根手指（finger=）/ bone_axis / body_forward / face / chest / pelvis 没有逐帧 probe 轴需要
    （后三者相对 owner 刚性），给均值轴。膝/肘不给（用 swivel_args）。"""
    if part in _JOINT_PARTS:
        return None
    spec = _HOLD_AXES.get(part) if (side in ("L", "R") and not finger) else None
    if spec is None:
        args = {"bones": [owner], "world_axis": out["local_axis"]}
        if out.get("secondary_axis"):
            args["secondary_axis"] = out["secondary_axis"]
        return args
    main, sec = spec
    args = {"bones": [owner], "world_axis": f"probe:{main}.{side}"}
    if sec:
        args["secondary_axis"] = f"probe:{sec}.{side}"
    elif out.get("secondary_axis"):
        args["secondary_axis"] = out["secondary_axis"]
    return args


def frame_probe_fn(part: str, side: str | None = None):
    """逐帧解剖方向函数：set_scene_frame 之后调，返回该帧的世界 Vector。

    hold_pose 的 "probe:<part>.<side>" 轴走这里——解剖方向相对控制骨随帧
    变化（手指有自己的动画），均值轴会留几十度残差，必须逐帧现推。
    无侧部位写 "probe:face"（agent_ops 按 rpartition 拆出 part=""、side="face" 也认）。
    """
    part = canonical_part(part)
    if not part and side and str(side).lower() in _BODY_PARTS + ("body_forward",):
        part, side = str(side).lower(), None
    if part not in _FRAME_KEYS:
        raise RuntimeError(
            f"probe 轴不支持 {part!r}，可用：{sorted(_FRAME_KEYS)}")
    key = _FRAME_KEYS[part]
    side = (side or "").upper() or None

    def fn(armature: Any, scene: Any) -> Vector | None:
        if part in _HAND_PARTS:
            d = _hand_frame(armature, side)
        elif part in _FOOT_PARTS:
            d = _foot_frame(armature, side)
        elif part == "knee_front":
            d = _joint_frame(armature, side, "knee")
        elif part == "elbow_front":
            d = _joint_frame(armature, side, "elbow")
        elif part in _BODY_PARTS:
            d = _body_part_frame(armature, part)
        else:
            d = _body_forward(armature, scene)
        return d.get(key) if d else None
    return fn


def frame_joint_fn(kind: str, side: str):
    """逐帧膝/肘的完整几何（朝向 / 连线 / 关节点 / 来源）：swivel 用。"""
    def fn(armature: Any, scene: Any) -> dict | None:
        return _joint_frame(armature, side, kind)
    return fn
