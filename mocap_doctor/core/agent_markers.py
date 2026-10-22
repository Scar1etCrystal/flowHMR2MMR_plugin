"""标记箭头（markers）：把"用户看到的方向"和"agent 算的方向"钉成同一支箭头。

用户的流程：拿到一个模型，先给掌心、脚底、膝、肘绑好空物体箭头，之后下指令、看结果都
以这些箭头为准。难点是"箭头怎么绑才一直跟着网格"：
  - 掌心/脚底这类部位的网格相对带动它的骨是刚性的（掌心皮肤法线在手骨局部坐标里 150 帧
    散布 ≤1.5°），所以**骨骼父级**就是正确绑法——一帧对准，全程有效。本工具替用户做这步：
    建 SINGLE_ARROW 空物体 MCD_<part>.<side>，父级绑到真正带动网格的骨（MMD 骨架的
    手首/足首，没有时用 RIG 的 DEF 骨），初值 = 网格标定 / 骨几何算出的方向。用户在视口里
    过一眼，不对就直接旋转箭头；probe / hold_pose 读到这支箭头就以它为准
    （evidence.palm_source = "marker"）。
  - 膝/肘（2026-10-04 §16）：铰链轴在**远端骨**上是刚性的（小腿 3.3–3.5°、前臂 5.4–8.6°，中位数；
    上臂 16.5–18.9° 不刚性），所以 MCD_knee.{s} 绑 ひざD/ひざ（小腿），MCD_elbow.{s} 绑 ひじ（前臂），
    方向 = 与远端骨、铰链轴都垂直、指向外凸侧（膝盖骨 / 肘尖），直腿直臂时也有定义。
    初值 = 全片运动标定（agent_anatomy.hinge_calibration）。
  - 脸/胸/骨盆：MCD_face / MCD_chest / MCD_pelvis 绑 頭 / 上半身2 / 下半身，方向 = 静止时的正前方随骨转。
  - bake：逐帧把 agent 算出的方向 K 到一支箭头上（只用来看，不读回）——膝/肘想看"当帧凸出角平分线"时用。
  - Child Of + 顶点组 不行：只跟平均顶点法线，绕法线的滚转不受控（实测漂 8–35°）。
  - check：查绑定错误——没父级、绑错侧（掌心事件：顶点父级的三个顶点属于另一只手）、
    绑错骨、跟着骨转但不刚性、与几何定义差太多、放在另一侧。

动作：create（默认）/ adopt（收编用户自己绑好骨骼父级的箭头）/ check / bake / list / remove。
不碰 RIG 动画数据，不需要 claim。
"""
from __future__ import annotations

from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import agent_anatomy as A
from . import agent_pose as P
from .animation import preserve_scene_frame, set_scene_frame

# 绑定骨：优先 MMD 骨架（真正带动网格的骨），其次 RIG 的 DEF 骨。腿的网格权重在 D 骨（足D/ひざD/足首D）上，
# ひざD 与 ひざ 的旋转差 ≤1.1°（实测），有 D 骨就绑 D 骨。
_BIND_BONES = {"palm": ("手首.{s}", "DEF-hand.{s}"),
               "sole": ("足首.{s}", "DEF-foot.{s}"),
               "knee": ("ひざD.{s}", "ひざ.{s}", "DEF-shin.{s}"),
               "elbow": ("ひじ.{s}", "DEF-forearm.{s}"),
               "face": ("頭", "DEF-spine.006"),
               "chest": ("上半身2", "DEF-spine.003"),
               "pelvis": ("下半身", "DEF-spine")}
# 合法承载某部位标记的骨（任一骨架；用来判"绑错骨"）
_PART_BONES = {"palm": ("手首.{s}", "DEF-hand.{s}", "hand_fk.{s}", "ORG-hand.{s}", "hand_ik.{s}"),
               "sole": ("足首.{s}", "足首D.{s}", "DEF-foot.{s}", "ORG-foot.{s}", "foot_ik.{s}", "foot_fk.{s}"),
               "knee": ("ひざ.{s}", "ひざD.{s}", "DEF-shin.{s}", "ORG-shin.{s}", "shin_fk.{s}"),
               "elbow": ("ひじ.{s}", "DEF-forearm.{s}", "ORG-forearm.{s}", "forearm_fk.{s}"),
               "face": ("頭", "DEF-spine.006", "ORG-spine.006", "head"),
               "chest": ("上半身2", "DEF-spine.003", "ORG-spine.003", "spine_fk.003", "chest"),
               "pelvis": ("下半身", "DEF-spine", "ORG-spine", "torso_root", "hips", "torso")}
# 刚性参照骨（RIG）：标记应当相对它不动
_REF_BONES = {"palm": ("hand_fk.{s}", "ORG-hand.{s}"), "sole": ("ORG-foot.{s}", "DEF-foot.{s}"),
              "knee": ("ORG-shin.{s}", "DEF-shin.{s}"), "elbow": ("ORG-forearm.{s}", "DEF-forearm.{s}"),
              "face": ("ORG-spine.006", "head"), "chest": ("ORG-spine.003", "DEF-spine.003"),
              "pelvis": ("ORG-spine", "DEF-spine")}
_SIDED = ("palm", "back_of_hand", "finger_dir", "knuckle", "hand_axis",
          "sole", "instep", "toe", "knee_front", "elbow_front")
_ACTIONS = ("create", "adopt", "check", "bake", "list", "remove")
# 烘焙（逐帧 K 帧、只用来看）的箭头用独立的名字，永远不和绑定标记 MCD_<part>.<side> 同名（审查 M20：
# 以前 bake palm 会写到 MCD_palm.R 上，再 create overwrite:true 复用这个物体时旧关键帧还在）
BAKE_PREFIX = "MCD_bake_"
_PART_CN = {"palm": "掌心", "sole": "脚底", "knee": "膝盖", "elbow": "肘尖", "face": "脸", "chest": "胸",
            "pelvis": "骨盆"}
_SIDE_CN = {"L": "左（角色自己的左）", "R": "右（角色自己的右）", None: ""}


def _settings(ctx):
    return getattr(ctx["scene"], "mocap_doctor", None)


def _mmd(ctx):
    return getattr(_settings(ctx), "mmd_armature", None)


def _bind_target(ctx, armature, part, side):
    """(父级骨架, 骨名)：MMD 骨架上的 手首/足首/ひざD/ひじ/頭…，没有就 RIG 的 DEF 骨。"""
    mmd = _mmd(ctx)
    for pat in _BIND_BONES[part]:
        name = pat.format(s=side) if side else pat
        if name.isascii():
            if armature.pose.bones.get(name) is not None:
                return armature, name
        elif mmd is not None and mmd.pose.bones.get(name) is not None:
            return mmd, name
    raise RuntimeError(f"找不到可绑定的骨：{[p.format(s=side) if side else p for p in _BIND_BONES[part]]}")


def _clean_reused(obj):
    """复用已有物体前清干净：关键帧/驱动、约束、旧的烘焙标签——绑定标记必须只靠骨骼父级动。"""
    if obj.animation_data is not None:
        obj.animation_data_clear()
    for c in list(obj.constraints):
        obj.constraints.remove(c)
    for k in ("mcd_baked_range",):
        if k in obj.keys():
            del obj[k]


def _get_or_make(name, length):
    obj = bpy.data.objects.get(name)
    created = obj is None
    if created:
        obj = bpy.data.objects.new(name, None)
    if obj.type != "EMPTY":
        raise RuntimeError(f"{name} 已存在但不是空物体，换个名字或先删掉它")
    if obj.name not in bpy.context.scene.collection.objects and not obj.users_collection:
        bpy.context.scene.collection.objects.link(obj)
    obj.empty_display_type = "SINGLE_ARROW"
    obj.empty_display_size = float(length)
    return obj, created


def _aim_matrix(anchor: Vector, direction: Vector) -> Matrix:
    m = direction.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    m.translation = anchor
    return m


def _world_dir_now(scene, armature, part, side):
    fn = A.frame_probe_fn(part, side)
    return fn(armature, scene)


def _definition_now(scene, armature, probe_part, side):
    """标记要对照的"定义"：膝/肘用刚性标定方向（标记本身是刚性的；弯着的帧 probe 跟当帧弯曲平面，
    两者差的是动捕铰链不理想的那部分，不算标记绑错）；其它部位 = probe 的几何/网格定义。"""
    if probe_part in ("knee_front", "elbow_front"):
        jd = A._joint_frame(armature, side, "knee" if probe_part == "knee_front" else "elbow")
        return None if jd is None else (jd.get("front_rigid") or jd.get("front"))
    return _world_dir_now(scene, armature, probe_part, side)


def _skin_point(ctx, origin: Vector, direction: Vector, *, max_dist=0.30, radius=0.025,
                default=0.05) -> tuple[Vector, float | None]:
    """从 origin 沿 direction 找到皮肤表面（目标网格里离这条射线 radius 内、最远的顶点）——
    箭头放在皮肤外面，用户才看得见。没有网格时退回 default 米。"""
    mesh = A._target_mesh(ctx["scene"])
    d = direction.normalized()
    if mesh is not None:
        try:
            with A.evaluable(mesh):
                co = A._evaluated_mesh_arrays(mesh)[0]
            o = np.asarray(origin, dtype=np.float64)
            dv = np.asarray(d, dtype=np.float64)
            rel = co - o
            t = rel @ dv
            perp = np.linalg.norm(rel - np.outer(t, dv), axis=1)
            sel = (t > 0.0) & (t < max_dist) & (perp < radius)
            if sel.any():
                # 取从关节往外的第一簇顶点的最远处 = 本肢体的皮肤；再往外隔开 >2 cm 的是别的东西
                # （膝盖朝前的射线会扎到另一条腿：实测第 230 帧左膝前 18 cm 是右腿）
                ts = np.sort(t[sel])
                dist = float(ts[0])
                for x in ts[1:]:
                    if float(x) - dist > 0.02:
                        break
                    dist = float(x)
                return origin + d * (dist + 0.01), round(dist * 1000.0, 1)
        except Exception:  # noqa: BLE001 - 只影响箭头摆放位置
            pass
    return origin + d * default, None


def _anchor(ctx, armature, part, side, direction):
    """箭头的位置（世界）：掌心皮肤中心 / 脚底三点中心 / 膝肘关节外的皮肤 / 脸胸骨盆的皮肤。"""
    probe_part = A.MARKER_PROBE_PART.get(part, part)
    base = A.part_anchor(armature, probe_part, side)
    if base is None:
        return None, None
    if part in ("knee", "elbow", "face", "chest", "pelvis"):
        return _skin_point(ctx, base, direction)
    return base, None


def _create(ctx, parts, sides, length, overwrite, frame):
    scene, armature = ctx["scene"], ctx["armature"]
    rows, warnings = [], []
    with preserve_scene_frame(scene):
        set_scene_frame(scene, frame)
        for part in parts:
            if part not in A.MARKER_PARTS:
                raise RuntimeError(f"create 只支持 {A.MARKER_PARTS}（逐帧显示用 action=bake）；收到 {part!r}")
            for side in (sides if part in A.SIDED_MARKERS else [None]):
                name = A.marker_name(part, side)
                existing = bpy.data.objects.get(name)
                if existing is not None and not overwrite:
                    mk, why = A.bound_marker(part, side)
                    rows.append({"name": name, "part": part, "side": side, "status": "kept",
                                 "bound": mk is not None, "why": why})
                    if mk is None:
                        warnings.append(f"{name} 已存在但{ '没有骨骼父级' if why == 'unbound' else why }，"
                                        f"没动它；要重建加 overwrite=true")
                    continue
                probe_part = A.MARKER_PROBE_PART[part]
                with A.ignore_markers():
                    res = A.probe(scene, armature, part=probe_part, side=side, frame_range=[frame, frame])
                    d = Vector(res["world_dir"])
                    if part in ("knee", "elbow"):
                        # 刚性箭头用全片标定的刚性方向（不是这一帧的弯曲平面）
                        jd = A._joint_frame(armature, side, part)
                        if jd and jd.get("front_rigid") is not None:
                            d = Vector(jd["front_rigid"])
                anchor, skin_mm = _anchor(ctx, armature, part, side, d)
                if anchor is None:
                    raise RuntimeError(f"{part}{'.' + side if side else ''} 找不到锚点骨")
                par, bone = _bind_target(ctx, armature, part, side)
                obj, created = _get_or_make(name, length)
                _clean_reused(obj)
                obj.parent = par
                obj.parent_type = "BONE"
                obj.parent_bone = bone
                obj.matrix_parent_inverse = Matrix.Identity(4)
                obj["mcd_marker"] = "bound"
                obj["mcd_part"], obj["mcd_side"] = part, side or ""
                bpy.context.view_layer.update()
                obj.matrix_world = _aim_matrix(anchor, d)
                bpy.context.view_layer.update()
                got = A.marker_dir(obj)
                err = A._angle_deg(got, d)
                ev = res.get("evidence", {})
                src = ev.get(f"{part}_source") or ev.get(f"{probe_part}_source")
                row = {"name": name, "part": part, "side": side,
                       "status": "created" if created else "updated",
                       "parent": par.name, "bone": bone, "source": src,
                       "world_dir": [round(v, 4) for v in got],
                       "place_err_deg": round(err, 2)}
                if skin_mm is not None:
                    row["skin_offset_mm"] = skin_mm
                if part in ("knee", "elbow"):
                    row["hinge_calibration"] = res.get("hinge_calibration")
                rows.append(row)
                if err > 0.5:
                    warnings.append(f"{name} 放置后读回方向偏 {err:.1f}°（父级矩阵异常？）")
                if part == "palm" and src == "fingers":
                    warnings.append(f"{name} 的初值来自手指几何（没有网格标定：{ev.get('mesh_calibration')}），"
                                    "可信度低——请在视口里把它转到垂直于掌心")
                if part in ("knee", "elbow") and src == "bend":
                    warnings.append(f"{name} 的初值是当帧弯曲方向（全片标定失败：{ev.get('hinge_calibration')}），"
                                    "这一帧要是直腿/直臂就不可信——请在视口里转到膝盖骨/肘尖方向")
    return rows, warnings


def _adopt(ctx, name, part, side, overwrite, frame_range):
    """把用户自己绑好骨骼父级的箭头收编为标记：改名 MCD_<part>.<side>、打 bound 标签、报它与几何估计的差。"""
    scene, armature = ctx["scene"], ctx["armature"]
    if part not in A.MARKER_PARTS:
        raise RuntimeError(f"adopt 只支持 {A.MARKER_PARTS}；收到 {part!r}")
    if part not in A.SIDED_MARKERS:
        side = None
    obj = bpy.data.objects.get(str(name))
    if obj is None or obj.type != "EMPTY":
        raise RuntimeError(f"找不到空物体 {name!r}")
    par = obj.parent
    ad = obj.animation_data
    if ad is not None and (ad.action is not None or len(ad.drivers) or len(ad.nla_tracks)):
        raise RuntimeError(f"{name} 自己带关键帧/驱动：在骨骼父级之上还会动，不能当刚性定义。先在 Blender 里清掉它的动画"
                           "（物体 → 动画 → 清除关键帧），或用 action=create 让工具建")
    if any(not c.mute and c.influence > 0.0 for c in obj.constraints):
        raise RuntimeError(f"{name} 带约束（{[c.type for c in obj.constraints]}）：Child Of 之类的绑定滚转不受控。"
                           "删掉约束后 Ctrl+P→骨骼，或用 action=create")
    if par is None or getattr(par, "type", "") != "ARMATURE" or obj.parent_type != "BONE" or not obj.parent_bone:
        raise RuntimeError(f"{name} 没有骨骼父级（parent_type={obj.parent_type}）：先在 Blender 里 Ctrl+P→骨骼 绑到 "
                           f"{[p.format(s=side) if side else p for p in _BIND_BONES[part]][0]}，"
                           "或直接用 action=create 让工具建")
    bside = _bone_side(obj.parent_bone)
    if side and bside and bside != side:
        raise RuntimeError(f"{name} 绑在 {obj.parent_bone}（{bside} 侧）上，你要收编成 {side} 侧的 {part}——"
                           "绑错侧了：先在 Blender 里重绑到正确一侧的骨")
    target = A.marker_name(part, side)
    existing = bpy.data.objects.get(target)
    if existing is not None and existing is not obj:
        if not overwrite:
            raise RuntimeError(f"{target} 已存在；要用 {name} 顶替加 overwrite=true")
        bpy.data.objects.remove(existing, do_unlink=True)
    obj.name = target
    obj.empty_display_type = "SINGLE_ARROW"
    obj["mcd_marker"] = "bound"
    obj["mcd_part"], obj["mcd_side"] = part, side or ""
    frames = [int(scene.frame_current)]
    if frame_range is not None:
        a, b = int(frame_range[0]), int(frame_range[1])
        n = min(9, b - a + 1)
        frames = sorted({int(round(a + i * (b - a) / max(1, n - 1))) for i in range(n)})
    worst = 0.0
    probe_part = A.MARKER_PROBE_PART[part]
    with preserve_scene_frame(scene):
        for f in frames:
            set_scene_frame(scene, f)
            with A.ignore_markers():
                g = _definition_now(scene, armature, probe_part, side)
            if g is not None:
                worst = max(worst, A._angle_deg(A.marker_dir(obj), g))
    return {"name": obj.name, "was": str(name), "part": part, "side": side,
            "parent": par.name, "bone": obj.parent_bone, "vs_geometry_max_deg": round(worst, 1), "frames": frames}


def _bake(ctx, part, side, frame_range, length):
    scene, armature = ctx["scene"], ctx["armature"]
    if frame_range is None:
        raise RuntimeError("bake 需要 frame_range=[A,B]")
    part = A.canonical_part(part)
    if part in _SIDED and side not in ("L", "R"):
        raise RuntimeError(f"{part} 需要 side='L'/'R'")
    a, b = int(frame_range[0]), int(frame_range[1])
    name = bake_name(part, side)
    if bpy.data.objects.get(name) is not None and bpy.data.objects[name].get("mcd_marker") == "bound":
        raise RuntimeError(f"{name} 是绑定标记（骨骼父级），不往上烘焙；要看逐帧方向就先 remove 它")
    obj, created = _get_or_make(name, length)
    for c in list(obj.constraints):
        obj.constraints.remove(c)
    obj.parent = None
    obj.rotation_mode = "QUATERNION"
    obj["mcd_marker"] = "baked"
    obj["mcd_part"], obj["mcd_side"] = part, side or ""
    obj["mcd_baked_range"] = [a, b]
    if obj.animation_data and obj.animation_data.action:
        obj.animation_data_clear()
    fallback = 0
    prev = None
    joint = part in ("knee_front", "elbow_front")
    with preserve_scene_frame(scene):
        for f in range(a, b + 1):
            set_scene_frame(scene, f)
            if joint:
                # 膝/肘的 bake 显示"当帧凸出角平分线"（精确的逐帧形状，直腿时沿用上一帧）
                p0, p1, p2, _c = A.limb_points(armature, "knee" if part == "knee_front" else "elbow", side)
                d = None
                if p0 is not None:
                    s1, s2 = (p1 - p0).normalized(), (p2 - p1).normalized()
                    if A._angle_deg(s1, s2) > 3.0:
                        d = (s1 - s2).normalized()
            else:
                d = _world_dir_now(scene, armature, part, side)
            if d is None or d.length < 1e-6:
                if prev is None:
                    continue
                d = prev
                fallback += 1
            prev = d
            anchor = A.part_anchor(armature, part, side) or Vector((0.0, 0.0, 0.0))
            m = _aim_matrix(anchor, d)
            obj.location = m.translation
            obj.rotation_quaternion = m.to_quaternion()
            obj.keyframe_insert("location", frame=f)
            obj.keyframe_insert("rotation_quaternion", frame=f)
    if prev is None:
        raise RuntimeError(f"{part}.{side} 在 {a}–{b} 推不出方向")
    return {"name": name, "part": part, "side": side, "status": "created" if created else "rebaked",
            "frame_range": [a, b], "fallback_frames": fallback,
            "shows": "当帧凸出角平分线（膝/肘）" if joint else "当帧几何方向",
            "note": "烘焙箭头只用来看，probe 不读回；姿态改了（修复后）要重新 bake。膝/肘的默认标记是 create 建的"
                    "刚性箭头 MCD_knee/MCD_elbow（绑在小腿/前臂上，直腿也有定义）"}


def bake_name(part, side=None) -> str:
    return f"{BAKE_PREFIX}{part}.{side}" if side else f"{BAKE_PREFIX}{part}"


def _sample(frame_range, scene, n=9):
    if frame_range is None:
        a, b = int(scene.frame_start), int(scene.frame_end)
    else:
        a, b = int(frame_range[0]), int(frame_range[1])
    n = min(n, b - a + 1)
    return sorted({int(round(a + i * (b - a) / max(1, n - 1))) for i in range(n)})


def _list(ctx, frame_range):
    scene, armature = ctx["scene"], ctx["armature"]
    rows = []
    frames = None
    if frame_range is not None:
        frames = _sample(frame_range, scene)
    for obj in bpy.data.objects:
        if not obj.name.startswith(A.MARKER_PREFIX) or obj.type != "EMPTY":
            continue
        kind = obj.get("mcd_marker") or "unknown"
        part, side = obj.get("mcd_part"), obj.get("mcd_side") or None
        row = {"name": obj.name, "kind": kind, "part": part, "side": side,
               "parent": obj.parent.name if obj.parent else None,
               "bone": obj.parent_bone if obj.parent_type == "BONE" else None}
        if kind == "bound" and part in A.MARKER_PARTS:
            mk, why = A.bound_marker(part, side)
            row["valid"] = mk is not None
            if mk is None:
                row["why"] = why
            if frames and mk is not None:
                worst = 0.0
                with preserve_scene_frame(scene):
                    for f in frames:
                        set_scene_frame(scene, f)
                        with A.ignore_markers():
                            g = _definition_now(scene, armature, A.MARKER_PROBE_PART[part], side)
                        if g is not None:
                            worst = max(worst, A._angle_deg(A.marker_dir(mk), g))
                row["vs_geometry_max_deg"] = round(worst, 1)
                row["frames"] = frames
        rows.append(row)
    return rows


def _remove(parts, sides, all_markers):
    gone = []
    if all_markers:
        names = [o.name for o in bpy.data.objects if o.name.startswith(A.MARKER_PREFIX) and o.type == "EMPTY"]
    else:
        names = []
        for p in parts:
            sided = p in A.SIDED_MARKERS or p in _SIDED
            for s in (sides if sided else [None]):
                names += [A.marker_name(p, s), bake_name(p, s)]
                pp = A.MARKER_PROBE_PART.get(p)
                if pp and pp != p:
                    names.append(bake_name(pp, s))       # 同一部位的逐帧显示箭头一起删
    for n in names:
        obj = bpy.data.objects.get(n)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
            gone.append(n)
    return gone


# ---------------------------------------------------------------------------
# check：绑定体检（掌心事件那一类错误：箭头"有时跟、一转就掉队"）

def _bone_side(name: str | None) -> str | None:
    if not name:
        return None
    n = str(name)
    for suf, s in ((".L", "L"), (".R", "R"), ("_L", "L"), ("_R", "R"), (".l", "L"), (".r", "R")):
        if n.endswith(suf):
            return s
    if n.startswith("左"):
        return "L"
    if n.startswith("右"):
        return "R"
    return None


def _bone_part(name: str | None) -> str | None:
    if not name:
        return None
    for part, pats in _PART_BONES.items():
        for pat in pats:
            for s in ("L", "R"):
                if pat.format(s=s) == name:
                    return part
            if "{s}" not in pat and pat == name:
                return part
    return None


def _ref_bone(armature, part, side):
    for pat in _REF_BONES.get(part, ()):
        n = pat.format(s=side) if side else pat
        if armature.pose.bones.get(n) is not None:
            return n
    return None


def _binding(obj) -> dict:
    """箭头怎么绑的：骨骼父级 / 顶点父级（哪些顶点组、哪一侧）/ Child Of / 没绑。"""
    out = {"type": "NONE"}
    par = obj.parent
    if par is not None:
        pt = obj.parent_type
        out = {"type": pt, "parent": par.name}
        if pt == "BONE":
            out["bone"] = obj.parent_bone
            out["bone_side"] = _bone_side(obj.parent_bone)
            out["bone_part"] = _bone_part(obj.parent_bone)
        elif pt in ("VERTEX", "VERTEX_3") and getattr(par, "type", "") == "MESH":
            idx = [int(i) for i in obj.parent_vertices][: (1 if pt == "VERTEX" else 3)]
            names = {g.index: g.name for g in par.vertex_groups}
            groups = {}
            for i in idx:
                if i >= len(par.data.vertices):
                    continue
                for ge in par.data.vertices[i].groups:
                    nm = names.get(ge.group, "?")
                    if nm.startswith("mmd_"):
                        continue
                    groups[nm] = groups.get(nm, 0.0) + float(ge.weight)
            tops = sorted(groups.items(), key=lambda kv: -kv[1])[:3]
            out["vertices"] = idx
            out["vertex_groups"] = [[n, round(w / max(1, len(idx)), 2)] for n, w in tops]
            sides = {_bone_side(n) for n, _w in tops if _bone_side(n)}
            out["bone_side"] = sides.pop() if len(sides) == 1 else (None if not sides else "mixed")
            out["bone_part"] = _bone_part(tops[0][0]) if tops else None
        elif pt == "OBJECT":
            out["note"] = "父级是物体本身（不跟骨）"
    for c in getattr(obj, "constraints", ()):
        if c.type == "CHILD_OF" and not c.mute and c.influence > 0:
            sub = getattr(c, "subtarget", "")
            out = {"type": "CHILD_OF", "parent": getattr(c.target, "name", None), "subtarget": sub,
                   "bone_side": _bone_side(sub), "bone_part": _bone_part(sub)}
            break
    return out


def _nearest_part(armature, pos: Vector):
    best = (None, None, 1e9)
    for part in A.MARKER_PARTS:
        for side in (("L", "R") if part in A.SIDED_MARKERS else (None,)):
            p = A.part_anchor(armature, A.MARKER_PROBE_PART[part], side)
            if p is None:
                continue
            dist = (p - pos).length
            if dist < best[2]:
                best = (part, side, dist)
    return best


def _nearest_over_frames(scene, armature, obj, frames):
    """静止（没父级）的箭头：在哪一帧离哪个部位最近——用户多半是在那一帧对着那个部位摆的。"""
    best = (None, None, 1e9, None)
    with preserve_scene_frame(scene):
        for f in frames:
            set_scene_frame(scene, f)
            part, side, d = _nearest_part(armature, obj.matrix_world.translation.copy())
            if d < best[2]:
                best = (part, side, d, f)
    return best


def _check_one(ctx, obj, frames, dense_frames) -> dict:
    scene, armature = ctx["scene"], ctx["armature"]
    name = obj.name
    kind = obj.get("mcd_marker") or ("user" if not name.startswith(A.MARKER_PREFIX) else "unknown")
    part = obj.get("mcd_part") or None
    side = obj.get("mcd_side") or None
    bind = _binding(obj)
    t = bind["type"]
    pos = obj.matrix_world.translation.copy()
    near_part, near_side, near_d = _nearest_part(armature, pos)
    near_frame = None
    if t in ("NONE", "OBJECT"):
        near_part, near_side, near_d, near_frame = _nearest_over_frames(scene, armature, obj, dense_frames)
    body_bound = bool(bind.get("bone_part"))      # 绑在认得出的身体骨 / 顶点组上
    if part not in A.MARKER_PARTS:          # 用户自己的箭头：骨骼父级按骨猜部位，顶点父级/没父级按位置
        if t in ("BONE", "CHILD_OF") and bind.get("bone_part"):
            part, side = bind["bone_part"], bind.get("bone_side") if bind["bone_part"] in A.SIDED_MARKERS else None
        elif t in ("VERTEX", "VERTEX_3") and near_d > 0.30 and bind.get("bone_part"):
            bs = bind.get("bone_side")
            part, side = bind["bone_part"], (bs if bs in ("L", "R") else None)
        else:
            part, side = near_part, near_side
    row = {"name": name, "kind": kind, "part": part, "side": side, "binding": bind}
    if kind == "baked":
        row.update({"status": "ok", "problems": [], "note": "逐帧烘焙的显示箭头（不读回）"})
        return row
    if part is None or (kind == "user" and near_d > 0.30 and not body_bound):
        row.update({"status": "unrelated", "problems": [],
                    "note": f"离任何身体部位都 > 30 cm（最近 {near_part} {near_d * 100:.0f} cm），不当身体标记"})
        return row
    cn = f"{_SIDE_CN.get(side, '')}{_PART_CN.get(part, part)}"
    own = A.part_anchor(armature, A.MARKER_PROBE_PART[part], side)
    other_side = {"L": "R", "R": "L"}.get(side)
    oth = A.part_anchor(armature, A.MARKER_PROBE_PART[part], other_side) if other_side else None
    row["position"] = {"nearest": f"{near_part}{'.' + near_side if near_side else ''}",
                       "nearest_cm": round(near_d * 100.0, 1),
                       "to_own_cm": round((own - pos).length * 100.0, 1) if own is not None else None}
    if near_frame is not None:
        row["position"]["nearest_at_frame"] = near_frame
    if oth is not None:
        row["position"]["to_other_side_cm"] = round((oth - pos).length * 100.0, 1)
    problems, fixes = [], []          # problems: (severity, text)
    bone0 = _BIND_BONES[part][0].format(s=side or "")
    ad = obj.animation_data
    if ad is not None and (ad.action is not None or len(ad.drivers) or len(ad.nla_tracks)):
        problems.append(("error", "箭头自己带关键帧/驱动：在骨骼父级之上还会动，各帧指向不一致；probe 不读它"))
        fixes.append("markers create（overwrite:true）重建（会清掉旧关键帧），或在 Blender 里清除它的动画")
    # ---- 绑定方式
    if t == "NONE":
        problems.append(("error", "没有父级：箭头是静止的世界方向，人一动就错，probe 会忽略它"
                         + (f"（它在第 {near_frame} 帧离{cn} {near_d * 100:.1f} cm——多半是那一帧摆的）"
                            if near_frame is not None else "")))
        fixes.append(f"markers create（overwrite:true）重建，或在那一帧 Ctrl+P→骨骼 绑到 {bone0}")
    elif t == "CHILD_OF":
        problems.append(("error", "用 Child Of 约束绑定：只跟目标的平均方向，绕法线的滚转不受控（掌心实测漂 8–35°）"))
        fixes.append("改成骨骼父级（Ctrl+P→骨骼）或 markers create")
    elif t == "OBJECT":
        problems.append(("error", f"父级是物体 {bind.get('parent')} 本身，不跟骨骼动"))
        fixes.append("markers create（overwrite:true）重建")
    bside = bind.get("bone_side")
    if side and bside in ("L", "R") and bside != side:
        if t in ("VERTEX", "VERTEX_3"):
            problems.append(("error", f"顶点父级的顶点属于 {bind.get('vertex_groups')}（{bside} 侧），箭头却在{cn}"
                             f"（离它 {row['position']['to_own_cm']} cm）：另一侧动它才动，这一侧单独转时不跟——"
                             "就是 2026-10-04 掌心箭头'有时跟、一转就跟不上'的原因"))
        else:
            problems.append(("error", f"绑在 {bind.get('bone') or bind.get('subtarget')}（{bside} 侧）上，"
                             f"名字/位置却是{cn}"))
        fixes.append(f"重绑到 {side} 侧（markers create overwrite:true，或 Ctrl+P→骨骼 选 {bone0}）")
    elif bside == "mixed":
        problems.append(("error", f"顶点父级的三个顶点分属两侧 {bind.get('vertex_groups')}：两边一起拖着它"))
        fixes.append("改成骨骼父级（markers create）")
    bpart = bind.get("bone_part")
    if t in ("BONE", "CHILD_OF") and bpart is not None and bpart != part:
        problems.append(("error", f"绑在 {bind.get('bone') or bind.get('subtarget')}（{_PART_CN.get(bpart, bpart)}的骨）上，"
                         f"不是带动{_PART_CN.get(part, part)}的骨"))
        fixes.append(f"重绑到 {bone0}")
    if t not in ("NONE", "OBJECT") and oth is not None and own is not None \
            and (oth - pos).length + 0.02 < (own - pos).length:
        problems.append(("error", f"箭头的位置离另一侧（{other_side}）更近：{row['position']['to_other_side_cm']} cm vs "
                         f"本侧 {row['position']['to_own_cm']} cm——名字和位置对不上"))
    # ---- 刚性 + 与定义的一致性（采样帧）
    ref = _ref_bone(armature, part, side)
    probe_part = A.MARKER_PROBE_PART[part]
    locs, diffs = [], []
    with preserve_scene_frame(scene):
        for f in frames:
            set_scene_frame(scene, f)
            d = A.marker_dir(obj)
            if ref is not None:
                q = (armature.matrix_world @ armature.pose.bones[ref].matrix).to_quaternion()
                locs.append(q.inverted() @ d)
            with A.ignore_markers():
                g = _definition_now(scene, armature, probe_part, side)
            if g is not None:
                diffs.append(A._angle_deg(d, g))
    if len(locs) >= 2:
        m = sum(locs, Vector()).normalized()
        spread = max(A._angle_deg(m, v) for v in locs)
        row["rigid_spread_deg"] = round(spread, 1)
        row["rigid_ref_bone"] = ref
        if spread > 5.0 and t not in ("NONE", "OBJECT"):
            problems.append(("error", f"在 {ref} 的局部坐标里散布 {spread:.1f}°（>5°）：它不跟{cn}刚性转动，"
                             "某些帧会指错"))
            if not fixes:
                fixes.append("改成骨骼父级（markers create overwrite:true）")
    if diffs:
        row["vs_definition_max_deg"] = round(max(diffs), 1)
        row["vs_definition_med_deg"] = round(float(np.median(diffs)), 1)
        if max(diffs) > 20.0 and not problems:
            problems.append(("warn", f"与几何/网格定义最大差 {max(diffs):.1f}°：可能是有意转过（以箭头为准），"
                             "也可能没对准——请在视口里确认一帧"))
    if near_d > 0.25 and t not in ("NONE", "OBJECT"):
        problems.append(("warn", f"离最近的部位（{row['position']['nearest']}）也有 {near_d * 100:.0f} cm：放错地方了？"))
    sev = [s_ for s_, _t in problems]
    row["status"] = "error" if "error" in sev else ("warn" if sev else "ok")
    row["problems"] = [t_ for _s, t_ in problems]
    if fixes:
        row["fix"] = "；".join(dict.fromkeys(fixes))
    if kind == "user" and row["status"] == "ok" and t == "BONE":
        row["fix"] = f"绑得对：可以 markers adopt name={name} part={part}" + (f" side={side}" if side else "") + \
            " 收编，之后 probe 以它为准"
    row["frames"] = frames
    return row


def _check(ctx, frame_range, name=None):
    scene = ctx["scene"]
    frames = _sample(frame_range, scene)
    dense = _sample(frame_range, scene, n=120)
    rows = []
    for obj in bpy.data.objects:
        if obj.type != "EMPTY":
            continue
        if name and obj.name != name:
            continue
        mine = obj.name.startswith(A.MARKER_PREFIX)
        if not mine and not name:
            # 用户自己摆的方向箭头都是 SINGLE_ARROW；mmd_tools 的刚体/关节空物体（mmd_type≠NONE）不是标记
            if obj.empty_display_type != "SINGLE_ARROW" or getattr(obj, "mmd_type", "NONE") not in ("NONE", ""):
                continue
        if not mine and obj.name.startswith("mcd_dir"):
            continue            # agent 自己的方向物体（世界方向，不是身体标记）
        rows.append(_check_one(ctx, obj, frames, dense))
    return rows


def _tool_markers(ctx, action="create", parts=None, sides=None, part=None, side=None,
                  frame_range=None, length=0.15, overwrite=False, frame=None, all=False,
                  name=None, **unknown):
    P.reject_unknown_args("markers", _tool_markers, unknown)
    action = str(action).lower()
    if action not in _ACTIONS:
        raise RuntimeError(f"未知 action {action!r}，可用：{_ACTIONS}")
    if ctx["armature"] is None:
        raise RuntimeError("没有识别到 RIG 骨架")
    scene = ctx["scene"]
    default_parts = ["palm", "sole", "knee", "elbow"] if action == "create" else list(A.MARKER_PARTS)
    parts = [str(p).lower() for p in (parts or ([part] if part else default_parts))]
    parts = [{"knee_front": "knee", "elbow_front": "elbow", "head": "face", "hips": "pelvis"}.get(p, p)
             if action != "bake" else p for p in parts]
    sides = [str(s).upper() for s in (sides or ([side] if side else ["L", "R"]))]
    if action == "create":
        f = int(frame) if frame is not None else int(scene.frame_current)
        rows, warnings = _create(ctx, parts, sides, float(length), bool(overwrite), f)
        made = [r["name"] for r in rows if r["status"] != "kept"]
        return {"summary": f"markers: {len(made)} 支已绑（{', '.join(made) or '无'}），{len(rows) - len(made)} 支保留",
                "data": {"markers": rows, "frame": f}, "warnings": warnings, "truncated": False,
                "hint": "在视口里检查：箭头应从掌心/脚底垂直指出、从膝盖骨/肘尖指出。不对就直接旋转箭头（它跟着骨，"
                        "一帧对准全程有效）；之后 probe_anatomy 的 evidence.<部位>_source 会是 marker。"
                        "绑完可以 markers check 体检"}
    if action == "adopt":
        if not name or len(parts) != 1 or len(sides) != 1:
            raise RuntimeError("adopt 需要 name=<你的箭头名> part=palm|sole|knee|elbow|face|chest|pelvis side=L|R")
        res = _adopt(ctx, name, parts[0], sides[0], bool(overwrite), frame_range)
        warn = ([f"{res['name']} 与网格/骨几何估计差 {res['vs_geometry_max_deg']}°（>15°）——确认它是对准的，"
                 "还是 action=create 重建"] if res["vs_geometry_max_deg"] > 15 else [])
        return {"summary": f"adopt {res['was']} → {res['name']}（父级 {res['parent']}/{res['bone']}，与几何估计差 {res['vs_geometry_max_deg']}°）",
                "data": res, "warnings": warn, "truncated": False,
                "hint": "之后 probe_anatomy 的 evidence.<部位>_source = marker，以这支箭头为准"}
    if action == "check":
        rows = _check(ctx, frame_range, name=name)
        bad = [r["name"] for r in rows if r.get("status") == "error"]
        warn = [r["name"] for r in rows if r.get("status") == "warn"]
        unrel = [r["name"] for r in rows if r.get("status") == "unrelated"]
        warnings = [f"{r['name']}：{'；'.join(r['problems'])}" for r in rows if r.get("status") == "error"]
        ok = len(rows) - len(bad) - len(warn) - len(unrel)
        return {"summary": f"{len(rows) - len(unrel)} 支身体箭头：{ok} ok，{len(warn)} 提醒，{len(bad)} 错误"
                           + (f"（{bad}）" if bad else "") + (f"；另 {len(unrel)} 支离身体太远，忽略" if unrel else ""),
                "data": {"arrows": rows}, "warnings": warnings, "truncated": False,
                "hint": "error 的箭头按 fix 处理后再用；status=ok 的用户箭头（骨骼父级、绑对侧）可以 adopt"}
    if action == "bake":
        if len(parts) != 1:
            raise RuntimeError("bake 一次一个 part（膝/肘/手指等逐帧方向）")
        p0 = A.canonical_part(parts[0])
        res = _bake(ctx, p0, (sides[0] if (side or sides) else None) if p0 in _SIDED else None,
                    frame_range, float(length))
        return {"summary": f"bake {res['name']} @{res['frame_range']}（{res['status']}，fallback {res['fallback_frames']} 帧）",
                "data": res, "warnings": [], "truncated": False,
                "hint": "播放这段看箭头；修复后要重新 bake"}
    if action == "list":
        rows = _list(ctx, frame_range)
        bad = [r["name"] for r in rows if r.get("valid") is False]
        return {"summary": f"{len(rows)} 支标记" + (f"，{len(bad)} 支无效：{bad}" if bad else ""),
                "data": {"markers": rows}, "warnings": [f"{n} 没有骨骼父级，probe 会忽略它" for n in bad],
                "truncated": False, "hint": "vs_geometry_max_deg = 标记与网格/骨几何估计的最大夹角（给了 frame_range 才算）"}
    gone = _remove(parts, sides, bool(all))
    return {"summary": f"removed {gone}", "data": {"removed": gone}, "warnings": [],
            "truncated": False, "hint": ""}


TOOLS = {"markers": _tool_markers}
