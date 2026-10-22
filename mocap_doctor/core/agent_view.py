"""人看到的方向 ↔ agent 算的方向：镜头 / 视口 / 角色 三套参照系，逐帧解析成世界向量。

为什么要这一层（2026-10-04 掌心事件之后的系统排查，汇总 §16）：

- "朝镜头"：旧 ``toward:"camera"`` 返回的是相机的**视线方向**（从相机看出去，再压平），
  也就是"背对镜头"——正好反了。而且 fixture 的 scene.camera 是默认相机（7.4,−6.9,5.0），
  画面里根本没框住 Teto；用户看的是 3D 视口（Layout 屏，在 Teto 前方偏左 2.8 m）。
- "朝前"：剧本写死世界 −Y；这段舞里角色整体转向，第 1 帧躯干朝 (0.91,−0.41)（与 −Y 差 66°），
  第 151 帧 (−0.55,−0.84)（差 33°）。脚尖方向与躯干朝向也差（中位数 8°，最大 42°）。
- "左"：角色自己的左（.L 那侧）与画面左，在正面镜头里正好相反。

本模块：
- ``view_basis``：镜头（scene.camera）或视口（用户正在看/存盘时的 3D 视口）的眼睛位置、视线、
  画面右/上，以及世界点 → 画面坐标（0..1，左下为原点）的投影。
- ``char_frame``：角色自己的前/左/上（躯干：骨盆 + 胸，静止姿态 −Y 随骨转；水平）。
- ``resolve_direction``：方向词 → 每帧世界单位向量（工具在每帧 frame_set 之后调用）。
- ``describe_dir``：任意世界方向 → 人话（相对镜头/视口 + 相对角色 + 画面上的 2D 指向）。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import bpy
from mathutils import Vector

UP = Vector((0.0, 0.0, 1.0))
_EPS = 1e-6

# ---------------------------------------------------------------------------
# 方向词表（工具文档与报错都从这里取）

CHAR_WORDS = {
    "forward": "char_forward", "front": "char_forward", "char_forward": "char_forward",
    "char_front": "char_forward",
    "back": "char_back", "backward": "char_back", "char_back": "char_back",
    "char_left": "char_left", "char_right": "char_right",
}
VIEW_WORDS = ("camera", "to_camera", "toward_camera", "viewer", "to_viewer", "away",
              "away_from_camera", "screen_left", "screen_right", "screen_up", "screen_down")
# 跟着身体某个部位走的目标：toes = 同侧脚尖方向（"膝盖对着脚尖"——人说"膝盖内扣/别内扣/朝外"多半是这个意思，
# 2026-10-04 Haiku 实测：把"膝盖有点内扣、朝外"照字面写成 char_right，膝盖被拧到正侧面 90°）
SIDE_WORDS = ("toes", "over_toes")
PART_PREFIX = "part:"
WORLD_WORDS = {"up": (0, 0, 1), "+z": (0, 0, 1), "down": (0, 0, -1), "-z": (0, 0, -1),
               "+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}
AMBIGUOUS = {"left": "左有两种意思：角色自己的左用 \"char_left\"（.L 那一侧），画面左用 \"screen_left\"",
             "right": "右有两种意思：角色自己的右用 \"char_right\"（.R 那一侧），画面右用 \"screen_right\"",
             "toward": "toward 不是方向词", "front_camera": "用 \"camera\"（朝镜头）"}
WORD_TABLE = (
    ("up / down", "世界上 / 下（+Z / −Z）"),
    ("forward / back", "角色自己的前 / 后（躯干朝向，水平，逐帧）；= char_forward / char_back"),
    ("char_left / char_right", "角色自己的左 / 右（.L / .R 那一侧，逐帧）"),
    ("camera", "朝镜头：从这个部位指向 scene.camera（逐帧，透视）"),
    ("viewer", "朝用户的 3D 视口（GUI 里正在看的那个；headless 时是存盘时的视口）"),
    ("away", "背对镜头（= 镜头的视线方向）"),
    ("screen_left / screen_right / screen_up / screen_down", "画面上的左/右/上/下（镜头坐标轴；view:\"viewer\" 时按视口）"),
    ("toes", "同侧脚尖的方向（膝盖对着脚尖：'膝盖内扣/别内扣/朝外'用它）；需要知道侧（膝/脚的 side）"),
    ("part:<部位>.<侧>", "跟着另一个部位的方向走，例 part:toe.L、part:chest"),
    ("[x, y, z]", "世界向量（前方不一定是 −Y：角色会转身，用 forward）"),
    ("<箭头空物体名>", "该箭头的 +Z 方向（逐帧，可 K 帧）"),
)


def _norm(v: Vector) -> Vector | None:
    return v.normalized() if v.length > _EPS else None


def _hz(v: Vector) -> Vector | None:
    return _norm(Vector((v.x, v.y, 0.0)))


def _deg(x: float) -> float:
    return round(math.degrees(x), 1)


def _angle(a: Vector, b: Vector) -> float:
    d = max(-1.0, min(1.0, float(a.normalized() @ b.normalized())))
    return math.degrees(math.acos(d))


# ---------------------------------------------------------------------------
# 镜头 / 视口

def camera_basis(scene: Any, cam: Any = None) -> dict | None:
    """scene.camera（或指定相机）的眼睛、视线、画面右/上与投影函数。"""
    cam = cam if cam is not None else getattr(scene, "camera", None)
    if cam is None or getattr(cam, "type", "") != "CAMERA":
        return None
    mw = cam.matrix_world
    q = mw.to_quaternion()
    persp = cam.data.type != "ORTHO"

    def proj(co):
        from bpy_extras.object_utils import world_to_camera_view
        v = world_to_camera_view(scene, cam, Vector(co))
        return (float(v.x), float(v.y), float(v.z))

    r = scene.render
    aspect = (r.resolution_x * r.pixel_aspect_x) / max(_EPS, r.resolution_y * r.pixel_aspect_y)
    return {"kind": "camera", "name": cam.name, "eye": mw.translation.copy() if persp else None,
            "forward": (q @ Vector((0.0, 0.0, -1.0))).normalized(),
            "right": (q @ Vector((1.0, 0.0, 0.0))).normalized(),
            "up": (q @ Vector((0.0, 1.0, 0.0))).normalized(),
            "persp": persp, "project": proj, "aspect": aspect}


def _view3d_areas() -> list:
    """(面积, 窗口, 区域) 按面积从大到小：GUI 里是用户正在看的；headless 时是存盘时的布局。"""
    wm = getattr(bpy.context, "window_manager", None)
    out = []
    for win in (wm.windows if wm is not None else ()):
        scr = getattr(win, "screen", None)
        for area in (scr.areas if scr is not None else ()):
            if area.type == "VIEW_3D":
                out.append((int(area.width) * int(area.height), win, area))
    return sorted(out, key=lambda t: -t[0])


def viewer_basis(scene: Any) -> dict | None:
    """用户的 3D 视口（最大的那个）。视口正在"透过相机看"时就是那台相机。"""
    areas = _view3d_areas()
    if not areas:
        return None
    _sz, win, area = areas[0]
    sp = area.spaces.active
    r3 = getattr(sp, "region_3d", None)
    if r3 is None:
        return None
    if r3.view_perspective == "CAMERA":
        b = camera_basis(scene, getattr(sp, "camera", None) or getattr(scene, "camera", None))
        if b is not None:
            b = dict(b)
            b["kind"] = "viewer"
            b["name"] = f"3D 视口（透过相机 {b['name']}）"
            return b
    inv = r3.view_matrix.inverted()
    q = inv.to_quaternion()
    persp = bool(r3.is_perspective)
    pm = r3.perspective_matrix.copy()

    def proj(co):
        v = pm @ Vector((co[0], co[1], co[2], 1.0))
        w = float(v.w) if abs(float(v.w)) > 1e-9 else 1e-9
        return ((float(v.x) / w + 1.0) * 0.5, (float(v.y) / w + 1.0) * 0.5,
                float(v.w) if persp else 1.0)

    return {"kind": "viewer", "name": f"3D 视口（{win.screen.name} 屏）",
            "eye": inv.translation.copy() if persp else None,
            "forward": (q @ Vector((0.0, 0.0, -1.0))).normalized(),
            "right": (q @ Vector((1.0, 0.0, 0.0))).normalized(),
            "up": (q @ Vector((0.0, 1.0, 0.0))).normalized(),
            "persp": persp, "project": proj,
            "aspect": float(area.width) / max(1.0, float(area.height)),
            "pivot": r3.view_location.copy(), "distance": float(r3.view_distance)}


def view_basis(scene: Any, view: str = "camera") -> dict:
    v = str(view or "camera").strip().lower()
    if v in ("camera", "cam", "镜头"):
        b = camera_basis(scene)
        if b is None:
            raise RuntimeError("场景没有相机（scene.camera 为空）：用 view:\"viewer\"（用户的 3D 视口）或直接给世界向量")
        return b
    if v in ("viewer", "viewport", "view", "视口"):
        b = viewer_basis(scene)
        if b is None:
            raise RuntimeError("找不到 3D 视口（没有窗口/没有 VIEW_3D 区域）：用 view:\"camera\" 或世界向量")
        return b
    raise RuntimeError(f"未知 view {view!r}：只有 \"camera\"（scene.camera）和 \"viewer\"（用户的 3D 视口）")


# ---------------------------------------------------------------------------
# 角色自己的参照系（躯干朝向）

_PELVIS = ("ORG-spine", "DEF-spine")
_CHEST = ("ORG-spine.003", "DEF-spine.003")
_MMD_PELVIS, _MMD_CHEST = "下半身", "上半身2"
_REST_SIGN: dict[str, float] = {}


def rest_front_sign(armature: Any) -> float:
    """静止姿态里角色朝 −Y（MMD/Rigify 惯例）→ +1；朝 +Y → −1。用静止脚尖方向判。"""
    key = armature.name
    if key not in _REST_SIGN:
        sign = 1.0
        db = armature.data.bones
        heel, toe = db.get("DEF-foot.L"), db.get("DEF-toe.L")
        if heel is not None and toe is not None:
            if float(toe.tail_local.y - heel.head_local.y) > 0.0:
                sign = -1.0
        _REST_SIGN[key] = sign
    return _REST_SIGN[key]


def rest_front_local(armature: Any, bone: str) -> Vector:
    """静止时的"角色正前方"在这根骨静止坐标系里的表达（随骨转 = 这块身体的前方）。"""
    db = armature.data.bones[bone]
    front = Vector((0.0, -rest_front_sign(armature), 0.0))
    return (db.matrix_local.to_quaternion().inverted() @ front).normalized()


def rest_up_local(armature: Any, bone: str) -> Vector:
    db = armature.data.bones[bone]
    return (db.matrix_local.to_quaternion().inverted() @ UP).normalized()


def _first_bone(armature: Any, names: Sequence[str]) -> str | None:
    for n in names:
        if armature.pose.bones.get(n) is not None:
            return n
    return None


def bone_front(armature: Any, bone: str) -> Vector:
    pb = armature.pose.bones[bone]
    q = (armature.matrix_world @ pb.matrix).to_quaternion()
    return (q @ rest_front_local(armature, bone)).normalized()


def char_frame(armature: Any, mmd: Any = None) -> dict | None:
    """角色自己的 前/左/上（水平）：骨盆前方 + 胸前方的水平平均。

    为什么不用脚尖：脚常外八（fixture 中位数差 8°、最大 42°）；"朝前"指的是躯干朝向。"""
    fronts, ev = [], {}
    for label, rig_names, mmd_name in (("pelvis", _PELVIS, _MMD_PELVIS), ("chest", _CHEST, _MMD_CHEST)):
        b = _first_bone(armature, rig_names)
        arm = armature
        if b is None and mmd is not None and mmd.pose.bones.get(mmd_name) is not None:
            arm, b = mmd, mmd_name
        if b is None:
            continue
        f = _hz(bone_front(arm, b))
        if f is not None:
            fronts.append(f)
            ev[f"{label}_forward"] = [round(v, 3) for v in f]
    if not fronts:
        return None
    fwd = _norm(sum(fronts, Vector()))
    if fwd is None:
        fwd = fronts[0]
    if len(fronts) == 2:
        ev["pelvis_vs_chest_deg"] = round(_angle(fronts[0], fronts[1]), 1)
    left = UP.cross(fwd).normalized()
    pel = _first_bone(armature, _PELVIS)
    centre = (armature.matrix_world @ armature.pose.bones[pel].head) if pel else None
    return {"forward": fwd, "left": left, "right": -left, "back": -fwd, "up": UP.copy(),
            "centre": centre, "evidence": ev}


def yaw_words(v: Vector) -> str:
    """世界水平方向 → "世界 −Y 偏 +X 33°" 这类说法（给 conventions 用）。"""
    h = _hz(v)
    if h is None:
        return "竖直"
    ang = math.degrees(math.atan2(h.x, -h.y))     # 0 = −Y，正 = 偏 +X
    if abs(ang) < 1.0:
        return "世界 −Y"
    return f"世界 −Y 偏 {'+X' if ang > 0 else '−X'} {abs(ang):.0f}°"


# ---------------------------------------------------------------------------
# 方向词 → 世界向量

def is_direction_word(spec: Any) -> bool:
    if not isinstance(spec, str):
        return False
    w = spec.strip().lower()
    return (w in CHAR_WORDS or w in VIEW_WORDS or w in WORLD_WORDS or w in AMBIGUOUS
            or w in SIDE_WORDS or w.startswith(PART_PREFIX))


def _part_direction(armature: Any, scene: Any, part: str, side: str | None) -> Vector:
    from . import agent_anatomy
    v = agent_anatomy.frame_probe_fn(part, side)(armature, scene)
    if v is None or v.length < _EPS:
        raise RuntimeError(f"这一帧推不出 {part}{'.' + side if side else ''} 的方向")
    return Vector(v).normalized()


def _resolve_raw(spec: Any, *, scene: Any, armature: Any, origin: Vector | None = None,
                 view: str = "camera", mmd: Any = None, side: str | None = None) -> tuple[Vector, str]:
    """方向说法 → (世界单位向量, 解析说明)。每帧 frame_set 之后调用（角色/相机都可能在动）。

    spec：[x,y,z] / 方向词（见 WORD_TABLE）/ 物体名（SINGLE_ARROW 空物体 = +Z；其它物体 = 从 origin 指向它）。
    origin：部位所在位置（"camera"/"viewer" 是从这里指向眼睛）。"""
    if spec is None:
        raise RuntimeError("没有给方向")
    if not isinstance(spec, str):
        try:
            v = Vector([float(x) for x in spec])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"无法解析方向 {spec!r}：要 [x,y,z] 或方向词") from exc
        if len(v) != 3 or v.length < _EPS:
            raise RuntimeError(f"方向向量 {list(spec)} 长度为零或不是三元组")
        return v.normalized(), "vec"
    raw = spec.strip()
    w = raw.lower()
    if w in AMBIGUOUS:
        raise RuntimeError(f"方向词 {raw!r} 有歧义：{AMBIGUOUS[w]}")
    if w in WORLD_WORDS:
        return Vector(WORLD_WORDS[w]).normalized(), w
    if w in SIDE_WORDS:
        sd = str(side or "").upper()
        if sd not in ("L", "R"):
            raise RuntimeError(f"方向词 {raw!r} 要知道是哪一侧（跟着修的膝/脚的 side）；或者写 \"part:toe.L\"")
        return _part_direction(armature, scene, "toe", sd), f"toes.{sd}"
    if w.startswith(PART_PREFIX):
        spec2 = raw[len(PART_PREFIX):].strip()
        part, _, sd = spec2.rpartition(".")
        if not part:
            part, sd = spec2, ""
        sd = sd.upper() if sd.upper() in ("L", "R") else ""
        return _part_direction(armature, scene, part.lower(), sd or None), f"part:{part}{'.' + sd if sd else ''}"
    if w in CHAR_WORDS:
        cf = char_frame(armature, mmd=mmd)
        if cf is None:
            raise RuntimeError("推不出角色朝向（RIG 上没有 ORG-spine / ORG-spine.003）：给世界向量")
        key = CHAR_WORDS[w].split("_", 1)[1]
        return Vector(cf[key]), f"{CHAR_WORDS[w]}(躯干)"
    if w in VIEW_WORDS:
        if w in ("viewer", "to_viewer"):
            b, w2 = view_basis(scene, "viewer"), "camera"
        elif w in ("camera", "to_camera", "toward_camera"):
            b, w2 = view_basis(scene, "camera"), "camera"
        else:
            b, w2 = view_basis(scene, view), w
        if w2 == "camera":
            if b["eye"] is not None and origin is not None:
                d = _norm(b["eye"] - Vector(origin))
                if d is not None:
                    return d, f"toward {b['kind']}:{b['name']}"
            return -b["forward"], f"toward {b['kind']}:{b['name']}(平行投影)"
        if w2 in ("away", "away_from_camera"):
            if b["eye"] is not None and origin is not None:
                d = _norm(Vector(origin) - b["eye"])
                if d is not None:
                    return d, f"away from {b['kind']}:{b['name']}"
            return Vector(b["forward"]), f"away from {b['kind']}:{b['name']}"
        axis = {"screen_right": b["right"], "screen_left": -b["right"],
                "screen_up": b["up"], "screen_down": -b["up"]}[w2]
        return Vector(axis), f"{w2}({b['kind']}:{b['name']})"
    obj = bpy.data.objects.get(raw)
    if obj is not None:
        if getattr(obj, "type", "") == "EMPTY" and obj.empty_display_type in ("SINGLE_ARROW", "ARROWS"):
            return (obj.matrix_world.to_quaternion() @ Vector((0.0, 0.0, 1.0))).normalized(), \
                f"object:{raw}(arrow +Z)"
        if origin is not None:
            d = _norm(obj.matrix_world.translation - Vector(origin))
            if d is not None:
                return d, f"object:{raw}(aim)"
        d = _norm(obj.matrix_world.translation)
        if d is not None:
            return d, f"object:{raw}(pos)"
    words = ", ".join(sorted(set(list(CHAR_WORDS) + list(VIEW_WORDS) + list(WORLD_WORDS))))
    raise RuntimeError(f"无法解析方向 {raw!r}：不是方向词，也不是物体名。方向词：{words}；或 [x,y,z]")


# axis → 保留哪两个分量（投影到 ⊥ 该轴的平面）。用户 2026-10-06："膝盖朝空物体，
# 注意是 Z 方向的旋转，就是膝盖和空物体可以有高低差"——只转 Z 时把瞄准方向投影到
# 水平面，空物体比膝盖高/低都不会把膝盖带得仰起来。
_AXIS_KEEP = {"z": (0, 1), "x": (1, 2), "y": (0, 2)}


def resolve_direction(spec: Any, *, scene: Any, armature: Any, origin: Vector | None = None,
                      view: str = "camera", mmd: Any = None, side: str | None = None,
                      axis: str | None = None) -> tuple[Vector, str]:
    """`_resolve_raw` + 可选的"只转某个轴"投影。

    axis="z"：把瞄准方向投影到 ⊥Z 的平面（只用水平方向，忽略高低差）；
    axis="x" / "y" 同理（⊥X = YZ 平面，⊥Y = XZ 平面）。
    axis=None（默认）= 原样三维瞄准，行为和以前逐位一致。
    """
    vec, why = _resolve_raw(spec, scene=scene, armature=armature, origin=origin,
                            view=view, mmd=mmd, side=side)
    if axis is None or axis == "":
        return vec, why
    key = str(axis).strip().lower()
    keep = _AXIS_KEEP.get(key)
    if keep is None:
        raise RuntimeError(
            f'axis 只能是 "x" / "y" / "z"（只转该轴、忽略其余分量），收到 {axis!r}；'
            "不给 axis = 原样三维瞄准")
    comps = [0.0, 0.0, 0.0]
    comps[keep[0]], comps[keep[1]] = float(vec[keep[0]]), float(vec[keep[1]])
    proj = _norm(Vector(comps))
    if proj is None:
        raise RuntimeError(
            f"方向「{why}」投影到『只转 {key.upper()}』的平面上长度为零"
            f"（目标和部位正好差在该轴上，比如空物体在正上方）——挪一下空物体或换 axis")
    return proj, f"{why}｜只转{key.upper()}"


# ---------------------------------------------------------------------------
# 世界方向 → 人话

def _side_word(x: float, pos: str, neg: str) -> str:
    return pos if x >= 0 else neg


def _view_frame(b: dict, origin: Vector | None):
    """(朝向眼睛 t, 画面右 r, 画面上 u)：正交，t 从部位指向眼睛。"""
    if b["eye"] is not None and origin is not None:
        t = _norm(b["eye"] - Vector(origin)) or -b["forward"]
    else:
        t = -b["forward"]
    r = _norm(b["right"] - t * float(b["right"] @ t)) or b["right"]
    u = t.cross(r).normalized()
    return t, r, u


def screen_point(b: dict, co: Vector) -> dict:
    x, y, depth = b["project"](co)
    inside = depth > 0 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    if depth <= 0:
        where = "在镜头后面（画面里看不到）"
    elif not inside:
        parts = []
        if x < 0:
            parts.append("左")
        elif x > 1:
            parts.append("右")
        if y < 0:
            parts.append("下")
        elif y > 1:
            parts.append("上")
        where = "在画面外（偏" + "".join(parts) + "）"
    else:
        hx = "左侧" if x < 0.4 else ("右侧" if x > 0.6 else "中间")
        hy = "下部" if y < 0.35 else ("上部" if y > 0.65 else "中部")
        where = f"画面{hx}{hy}（x={x:.2f}, y={y:.2f}）"
    return {"x": round(x, 3), "y": round(y, 3), "depth": round(depth, 3), "in_frame": bool(inside), "where": where}


def screen_arrow(b: dict, origin: Vector, d: Vector, length: float = 0.15) -> dict | None:
    """世界箭头（origin → origin+length·d）在画面上的 2D 指向：0° = 画面右，90° = 画面上。"""
    x0, y0, z0 = b["project"](origin)
    x1, y1, z1 = b["project"](Vector(origin) + Vector(d) * length)
    if z0 <= 0 or z1 <= 0:
        return None
    dx, dy = (x1 - x0) * b["aspect"], (y1 - y0)
    n = math.hypot(dx, dy)
    t, _r, _u = _view_frame(b, origin)
    # 透视下 length 在画面上的期望长度：用垂直于视线的同长箭头比
    side = (b["right"] * length)
    xs, ys, _zs = b["project"](Vector(origin) + side)
    ref = math.hypot((xs - x0) * b["aspect"], ys - y0)
    ratio = n / ref if ref > 1e-9 else 0.0
    if n < 1e-9:
        return {"angle_deg": None, "clock": None, "words": "箭头正对/背对画面（看起来是一个点）",
                "foreshorten": round(ratio, 2)}
    ang = math.degrees(math.atan2(dy, dx))
    clock = int(round((90.0 - ang) / 30.0)) % 12
    clock = 12 if clock == 0 else clock
    names = ["右", "右上", "上", "左上", "左", "左下", "下", "右下"]
    name = names[int(round(((ang % 360.0) / 45.0))) % 8]
    toward = float(Vector(d) @ t)
    extra = ""
    if ratio < 0.35:
        extra = "（几乎沿视线：" + ("朝着屏幕外指向你" if toward > 0 else "朝屏幕里") + "，画面上很短）"
    return {"angle_deg": round(ang, 1), "clock": clock, "foreshorten": round(ratio, 2),
            "words": f"画面上指向{name}（{clock} 点钟）{extra}"}


def view_words(d: Vector, b: dict, origin: Vector | None, label: str) -> dict:
    """方向相对镜头/视口：与"朝{label}"的夹角、偏画面左右/上下多少度、人话。"""
    t, r, u = _view_frame(b, origin)
    d = Vector(d).normalized()
    ct, cr, cu = float(d @ t), float(d @ r), float(d @ u)
    a = _angle(d, t)
    az = math.degrees(math.atan2(cr, ct))
    el = math.degrees(math.atan2(cu, math.hypot(ct, cr)))
    lr = _side_word(cr, "右", "左")
    ud = _side_word(cu, "上", "下")
    bits = []
    if a <= 12.0:
        text = f"正对{label}"
    elif a < 70.0:
        if abs(cr) > 0.08:
            bits.append(f"偏画面{lr} {abs(math.degrees(math.asin(max(-1, min(1, cr))))):.0f}°")
        if abs(cu) > 0.08:
            bits.append(f"偏{ud} {abs(math.degrees(math.asin(max(-1, min(1, cu))))):.0f}°")
        text = f"朝{label}" + ("，" + "、".join(bits) if bits else "") + f"（与正对差 {a:.0f}°）"
    elif a <= 110.0:
        h = math.degrees(math.atan2(cu, cr))
        names = ["画面右", "画面右上", "画面上", "画面左上", "画面左", "画面左下", "画面下", "画面右下"]
        text = f"侧对{label}：指向{names[int(round((h % 360.0) / 45.0)) % 8]}（与正对差 {a:.0f}°）"
    else:
        if abs(cr) > 0.15:
            bits.append(f"斜向画面{lr}")
        if abs(cu) > 0.15:
            bits.append(f"偏{ud}")
        text = f"背对{label}（指向画面里）" + ("，" + "、".join(bits) if bits else "") + f"（与正对差 {a:.0f}°）"
    return {"angle_to_view_deg": round(a, 1), "screen_right": round(cr, 3), "screen_up": round(cu, 3),
            "toward_view": round(ct, 3), "azimuth_deg": round(az, 1), "elevation_deg": round(el, 1),
            "words": text}


def char_words(d: Vector, cf: dict) -> dict:
    """方向相对角色自己：前/左（角色自己的左）/上。"""
    d = Vector(d).normalized()
    f, l = cf["forward"], cf["left"]
    cfw, cl, cu = float(d @ f), float(d @ l), float(d @ UP)
    az = math.degrees(math.atan2(cl, cfw))           # + = 偏角色左
    el = math.degrees(math.asin(max(-1.0, min(1.0, cu))))
    horiz = math.hypot(cfw, cl)
    if abs(el) >= 60.0:
        base = "朝上" if el > 0 else "朝下"
        if horiz > 0.17:
            base += "，水平分量" + _yaw_name(az)
        text = f"{base}（仰角 {el:+.0f}°）"
    else:
        text = _yaw_name(az)
        if abs(el) >= 10.0:
            text += f"，{'偏上' if el > 0 else '偏下'} {abs(el):.0f}°"
    return {"azimuth_deg": round(az, 1), "elevation_deg": round(el, 1), "words": text,
            "note": "azimuth：0 = 角色正前方，+ = 偏角色自己的左（.L 侧），− = 偏右"}


def _yaw_name(az: float) -> str:
    a = abs(az)
    side = "左" if az > 0 else "右"
    if a <= 15.0:
        return "朝角色正前方" + (f"（偏{side} {a:.0f}°）" if a >= 3 else "")
    if a < 70.0:
        return f"朝角色前方偏{side} {a:.0f}°"
    if a <= 110.0:
        return f"朝角色{side}侧（{a:.0f}°）"
    if a < 165.0:
        return f"朝角色后方偏{side}（{a:.0f}°）"
    return "朝角色正后方"


def lr_mapping(cf: dict, b: dict) -> dict:
    """角色左右 ↔ 画面左右：正面镜头里角色的左在画面右（镜像）。"""
    s = float(cf["left"] @ b["right"])
    if s > 0.35:
        txt = "镜像：角色自己的左（.L）在画面右边——镜头在角色前方"
        kind = "mirrored"
    elif s < -0.35:
        txt = "同向：角色自己的左（.L）在画面左边——镜头在角色后方"
        kind = "same"
    else:
        txt = "侧面：角色左右大致沿视线方向，画面左右分不清角色左右——按 .L/.R 说"
        kind = "side"
    return {"char_left_dot_screen_right": round(s, 3), "kind": kind, "words": txt}


def view_from_character(cf: dict, b: dict) -> dict | None:
    """镜头/视口在角色的哪个方位（以角色躯干为原点）。"""
    if b["eye"] is None or cf.get("centre") is None:
        d = -Vector(b["forward"])
    else:
        d = _norm(b["eye"] - cf["centre"])
        if d is None:
            return None
    cw = char_words(d, cf)
    dist = None if b["eye"] is None or cf.get("centre") is None else round((b["eye"] - cf["centre"]).length, 2)
    words = cw["words"].replace("朝角色", "在角色").replace("朝上", "在头顶上方").replace("朝下", "在脚下方")
    return {"azimuth_deg": cw["azimuth_deg"], "elevation_deg": cw["elevation_deg"], "distance_m": dist,
            "words": f"{b['kind']}（{b['name']}）{words}" + (f"，距 {dist} m" if dist is not None else "")}


def describe_dir(d: Vector, *, origin: Vector | None, scene: Any, armature: Any,
                 views: Sequence[str] = ("camera", "viewer"), arrow_len: float = 0.15,
                 mmd: Any = None) -> dict:
    """一个世界方向的全部人话：相对角色、相对每个 view（镜头/视口）、画面 2D 指向、部位在画面哪里。"""
    out: dict = {"world_dir": [round(v, 4) for v in Vector(d).normalized()]}
    cf = char_frame(armature, mmd=mmd)
    if cf is not None:
        out["character"] = char_words(d, cf)
    for v in views:
        try:
            b = view_basis(scene, v)
        except RuntimeError as exc:
            out[v] = {"unavailable": str(exc)}
            continue
        label = "镜头" if v == "camera" else "视口"
        item = view_words(d, b, origin, label)
        item["view_name"] = b["name"]
        if origin is not None:
            item["part_on_screen"] = screen_point(b, origin)
            arr = screen_arrow(b, origin, d, arrow_len)
            if arr is not None:
                item["arrow_on_screen"] = arr
        if cf is not None:
            item["lr"] = lr_mapping(cf, b)
        out[v] = item
    return out


# ---------------------------------------------------------------------------
# 工具：orient_report（部位朝向的人话）/ conventions（这份文件的方向、单位、帧号约定）

_PART_CN = {"palm": "掌心", "back_of_hand": "手背", "finger_dir": "手指指向", "knuckle": "指根连线",
            "hand_axis": "腕→指根", "sole": "脚底", "instep": "脚背", "toe": "脚尖",
            "knee_front": "膝盖（膝盖骨朝向）", "elbow_front": "肘尖（鹰嘴朝向）",
            "face": "脸", "chest": "胸", "pelvis": "骨盆", "body_forward": "身体朝向"}
_SIDE_CN = {"L": "左", "R": "右"}


def _frames_for(scene, frame, frame_range, max_frames):
    if frame is not None:
        return [int(frame)]
    if frame_range is not None:
        a, b = int(frame_range[0]), int(frame_range[1])
        n = max(1, min(int(max_frames), b - a + 1))
        if n == 1:
            return [a]
        return sorted({int(round(a + i * (b - a) / (n - 1))) for i in range(n)})
    return [int(scene.frame_current)]


def orient_report(scene, armature, *, part, side=None, frame=None, frame_range=None,
                  view="both", max_frames=5, finger=None, toward=None) -> dict:
    from . import agent_anatomy as A
    from .animation import preserve_scene_frame, set_scene_frame
    part = A.canonical_part(part)
    if part == "finger":
        part = "finger_dir"
    if part not in A._KEY_MAP:
        raise RuntimeError(f"orient_report 不支持 part {part!r}，可用：{sorted(A._KEY_MAP)}（别名 {A.PART_ALIASES}）")
    side = (side or "").upper() or None
    views = ("camera", "viewer") if str(view).lower() == "both" else (str(view).lower(),)
    mmd = getattr(getattr(scene, "mocap_doctor", None), "mmd_armature", None)
    key = A._KEY_MAP[part]
    rows = []
    frames = _frames_for(scene, frame, frame_range, max_frames)
    with preserve_scene_frame(scene):
        for f in frames:
            set_scene_frame(scene, f)
            d = A._eval_part(scene, armature, part, side, finger, None)
            if not d or d.get(key) is None:
                rows.append({"frame": f, "text": "这一帧推不出方向"})
                continue
            vec = Vector(d[key])
            origin = A.part_anchor(armature, part, side)
            desc = describe_dir(vec, origin=origin, scene=scene, armature=armature, views=views, mmd=mmd)
            ev = d.get("evidence") or {}
            src = next((v for k, v in ev.items() if k.endswith("_source")), None)
            row = {"frame": f, "world_dir": desc["world_dir"], "source": src}
            for k in ("character", "camera", "viewer"):
                if k in desc:
                    row[k] = desc[k]
            name = f"{_SIDE_CN.get(side, '')}{_PART_CN.get(part, part)}"
            bits = [f"{name} @{f}"]
            for v in views:
                it = desc.get(v) or {}
                if "words" in it:
                    s = it["words"]
                    if it.get("arrow_on_screen"):
                        s += "；" + it["arrow_on_screen"]["words"]
                    bits.append(s)
            if "character" in desc:
                bits.append(desc["character"]["words"])
            if toward is not None:
                t, how = resolve_direction(toward, scene=scene, armature=armature, origin=origin,
                                           view=views[0], mmd=mmd, side=side)
                if part in ("knee_front", "elbow_front"):
                    e = A.swivel_error_deg(vec, t, d.get("chord"), pole=d.get("pole"))
                    row["err_deg"] = None if e is None else round(abs(e), 1)
                    row["err_metric"] = "swivel"
                else:
                    row["err_deg"] = round(_angle(vec, t), 1)
                row["toward_how"] = how
                bits.append(f"离目标（{how}）{row['err_deg']}°")
            row["text"] = "：".join(bits[:1]) + "：" + "；".join(bits[1:])
            rows.append(row)
    out = {"part": part, "side": side, "frames": rows, "views": list(views)}
    try:
        cf = char_frame(armature, mmd=mmd)
        for v in views:
            b = view_basis(scene, v)
            out.setdefault("view_info", {})[v] = {
                "name": b["name"], "lr": lr_mapping(cf, b)["words"] if cf else None,
                "from_character": (view_from_character(cf, b) or {}).get("words") if cf else None}
    except RuntimeError as exc:
        out["view_info_error"] = str(exc)
    out["legend"] = ("角色左右 = 角色自己的（.L/.R）；画面左右 = 镜头/视口里看到的。"
                     "arrow_on_screen.angle_deg：0 = 画面右，90 = 画面上；clock：12 点 = 画面正上方")
    return out


def _limb_modes(armature) -> dict:
    from . import agent_anatomy as A
    out = {}
    for kind, limb in (("knee", "leg"), ("elbow", "arm")):
        for s in ("L", "R"):
            if armature.pose.bones.get(A._JOINTS[kind]["switch"].format(s=s)) is None:
                continue
            ik = A.limb_is_ik(armature, kind, s)
            if kind == "knee":
                txt = ("IK：脚用 foot_ik.{s}，膝朝向用 swivel（写 thigh_ik.{s} 的 Y 旋转）；thigh_fk/shin_fk/foot_fk 写了看不见"
                       if ik else "FK：thigh_fk/shin_fk/foot_fk 有效；膝朝向用 swivel")
            else:
                txt = ("FK：shoulder/upper_arm_fk/forearm_fk/hand_fk 有效（上臂前臂是 Euler 骨）；肘朝向用 swivel"
                       if not ik else "IK：手用 hand_ik.{s}，肘朝向用 swivel（upper_arm_ik 的 Y 旋转）")
            out[f"{limb}.{s}"] = txt.format(s=s)
    return out


def conventions(scene, armature, *, frame=None) -> dict:
    from . import agent_anatomy as A
    from .animation import preserve_scene_frame, set_scene_frame
    st = getattr(scene, "mocap_doctor", None)
    mmd = getattr(st, "mmd_armature", None)
    f = int(frame) if frame is not None else int(scene.frame_current)
    fps = float(scene.render.fps) / float(scene.render.fps_base or 1.0)
    pkl0 = int(getattr(st, "source_pkl_frame_start", 1) or 1)
    out = {"frame": f,
           "frames": {"blender_range": [int(scene.frame_start), int(scene.frame_end)], "fps": round(fps, 3),
                      "pkl_first_frame": pkl0,
                      "rule": f"帧号一律用 Blender 时间轴帧号（用户在时间轴上看到的那个；frame_range 含两端）。"
                              f"视频 / pkl 的第 i 帧（从 0 数）= Blender 第 i+{pkl0} 帧。"},
           "units": {"angle": "度（四元数 w,x,y,z）", "position": "米（slide_report / ground_report 的 *_mm 是毫米；"
                     "fix_ground 的 rest_clearance 是米）", "speed": "度/帧（30 fps 时 ×30 = 度/秒）"},
           "sides": "L/R = 角色自己的左右（.L 骨 = 角色左手那一侧），与画面左右无关；画面左右用 screen_left/screen_right",
           "world": {"up": "+Z", "rest_front": "静止姿态面朝 −Y（MMD 惯例）", "mirror_plane": "X=0"},
           "direction_words": [{"word": w, "meaning": m} for w, m in WORD_TABLE],
           "limbs": _limb_modes(armature)}
    with preserve_scene_frame(scene):
        set_scene_frame(scene, f)
        cf = char_frame(armature, mmd=mmd)
        if cf is not None:
            out["character"] = {"forward": [round(v, 3) for v in cf["forward"]],
                                "forward_words": yaw_words(cf["forward"]),
                                "left": [round(v, 3) for v in cf["left"]],
                                "note": "角色会转身：'朝前' 用方向词 forward（逐帧跟躯干），别写死 [0,-1,0]",
                                "evidence": cf["evidence"]}
        for v in ("camera", "viewer"):
            try:
                b = view_basis(scene, v)
            except RuntimeError as exc:
                out[v] = {"unavailable": str(exc)}
                continue
            item = {"name": b["name"], "perspective": b["persp"]}
            if b["eye"] is not None:
                item["eye"] = [round(x, 3) for x in b["eye"]]
            item["view_dir"] = [round(x, 3) for x in b["forward"]]
            if cf is not None:
                item["from_character"] = (view_from_character(cf, b) or {}).get("words")
                item["lr"] = lr_mapping(cf, b)
                if cf.get("centre") is not None:
                    sp = screen_point(b, cf["centre"])
                    item["character_on_screen"] = sp
                    if not sp["in_frame"]:
                        item["warning"] = f"角色不在这个{('镜头' if v == 'camera' else '视口')}画面里（{sp['where']}）"
                for s in ("L", "R"):
                    hp = A.part_anchor(armature, "palm", s)
                    if hp is not None:
                        item[f"hand_{s}_on_screen"] = screen_point(b, hp)["where"]
            out[v] = item
        marks = []
        for obj in bpy.data.objects:
            if obj.type == "EMPTY" and obj.name.startswith(A.MARKER_PREFIX):
                part = obj.get("mcd_part")
                side = obj.get("mcd_side") or None
                mk, why = (A.bound_marker(part, side) if part in A.MARKER_PARTS else (None, "unknown"))
                marks.append({"name": obj.name, "kind": obj.get("mcd_marker"), "valid": mk is not None,
                              **({"why": why} if mk is None else {})})
        out["markers"] = marks
    return out


def _tool_orient_report(ctx, part, side=None, frame=None, frame_range=None, view="both",
                        max_frames=5, finger=None, toward=None, **unknown):
    from . import agent_pose as P
    P.reject_unknown_args("orient_report", _tool_orient_report, unknown)
    res = orient_report(ctx["scene"], ctx["armature"], part=part, side=side, frame=frame,
                        frame_range=frame_range, view=view, max_frames=max_frames, finger=finger,
                        toward=toward)
    first = next((r for r in res["frames"] if "text" in r), {"text": ""})
    warnings = []
    if res.get("view_info_error"):
        warnings.append(res["view_info_error"])
    for v, it in (res.get("view_info") or {}).items():
        pass
    return {"summary": first["text"] + (f"（另 {len(res['frames']) - 1} 帧见 data.frames）" if len(res["frames"]) > 1 else ""),
            "data": res, "warnings": warnings, "truncated": False,
            "hint": "这是人话翻译（只读、实时）。修复的数值验收仍用 probe_anatomy 的 err_inner_deg"}


def _tool_conventions(ctx, frame=None, **unknown):
    from . import agent_pose as P
    P.reject_unknown_args("conventions", _tool_conventions, unknown)
    res = conventions(ctx["scene"], ctx["armature"], frame=frame)
    ch = res.get("character", {})
    cam = res.get("camera", {})
    summary = (f"@{res['frame']}：角色面朝 {ch.get('forward_words', '?')}；"
               f"镜头 {cam.get('from_character') or cam.get('unavailable', '?')}；"
               f"{(cam.get('lr') or {}).get('words', '')}")
    warnings = [it["warning"] for it in (res.get("camera") or {}, res.get("viewer") or {}) if it.get("warning")]
    bad = [m["name"] for m in res.get("markers", []) if not m["valid"] and m.get("kind") == "bound"]
    if bad:
        warnings.append(f"标记 {bad} 无效（没有骨骼父级）：markers check 看原因")
    return {"summary": summary, "data": res, "warnings": warnings, "truncated": False,
            "hint": "开工先看一次：哪边是角色的前/左、镜头和视口在角色哪一侧、哪条腿/胳膊是 IK"}


# ---------------------------------------------------------------------------
# render_view：无头 Workbench 渲染一张 PNG（给能看图的 agent / 用户核对）。空物体箭头渲染不出来，
# 临时建网格箭头代替（绿 = MCD 标记，红 = 部位当前方向，蓝 = 目标方向），渲完全部删掉、场景设置还原。

_COLORS = {"marker": (0.1, 0.85, 0.2, 1.0), "part": (0.95, 0.15, 0.1, 1.0), "target": (0.15, 0.35, 1.0, 1.0)}


def _arrow_mesh(name: str, length: float):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    shaft = bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=0.04, radius2=0.04, depth=0.75)
    bmesh.ops.translate(bm, verts=shaft["verts"], vec=(0.0, 0.0, 0.375))
    head = bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.12, radius2=0.0, depth=0.25)
    bmesh.ops.translate(bm, verts=head["verts"], vec=(0.0, 0.0, 0.875))
    bm.to_mesh(me)
    bm.free()
    return me


def render_view(scene, armature, *, frame=None, view="camera", part=None, side=None, size=(640, 360),
                markers=True, toward=None, out_dir=None, distance=None) -> dict:
    import os
    import time as _time
    from mathutils import Matrix
    from . import agent_anatomy as A
    from .animation import preserve_scene_frame, set_scene_frame
    w, h = int(size[0]), int(size[1])
    if not (64 <= w <= 1920 and 64 <= h <= 1920):
        raise RuntimeError("size 要在 64..1920 像素之间")
    f = int(frame) if frame is not None else int(scene.frame_current)
    mmd = getattr(getattr(scene, "mocap_doctor", None), "mmd_armature", None)
    r, sh = scene.render, scene.display.shading
    saved = {"camera": scene.camera, "engine": r.engine, "rx": r.resolution_x, "ry": r.resolution_y,
             "rp": r.resolution_percentage, "fp": r.filepath, "ff": r.image_settings.file_format,
             "cm": r.image_settings.color_mode, "ft": r.film_transparent, "light": sh.light,
             "ct": sh.color_type, "px": r.pixel_aspect_x, "py": r.pixel_aspect_y}
    temp_objs, temp_data = [], []
    arrows = []
    out_dir = out_dir or bpy.app.tempdir
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{part or 'scene'}{('.' + side) if side else ''}"
    path = os.path.join(out_dir, f"render_{f}_{view}_{tag}_{int(_time.time()) % 100000}.png")
    try:
        with preserve_scene_frame(scene):
            set_scene_frame(scene, f)
            b = view_basis(scene, "viewer" if view in ("viewer", "part_viewer") else "camera")
            cam_data = bpy.data.cameras.new("MCD_render_cam")
            temp_data.append(cam_data)
            cam = bpy.data.objects.new("MCD_render_cam", cam_data)
            scene.collection.objects.link(cam)
            temp_objs.append(cam)
            m = Matrix((b["right"], b["up"], -b["forward"])).transposed().to_4x4()
            src_cam = scene.camera if b["kind"] == "camera" else None
            if src_cam is not None:
                cam_data.lens = src_cam.data.lens
                cam_data.sensor_width = src_cam.data.sensor_width
                cam_data.sensor_fit = src_cam.data.sensor_fit
                cam_data.type = src_cam.data.type
                cam_data.ortho_scale = src_cam.data.ortho_scale
            else:
                cam_data.lens = 50.0
                cam_data.sensor_width = 36.0
            eye = b["eye"]
            anchor = A.part_anchor(armature, part, side.upper() if side else None) if part else None
            if part and anchor is None:
                raise RuntimeError(f"找不到部位 {part} {side or ''} 的位置")
            if anchor is not None:          # 特写：同一视线方向，拉近到部位前 distance 米
                dist = float(distance) if distance else 0.9
                eye = anchor - b["forward"] * dist
                cam_data.type = "PERSP"
                cam_data.lens, cam_data.sensor_width = 50.0, 36.0
            elif eye is None:
                cam_data.type = "ORTHO"
                cam_data.ortho_scale = max(1.0, float(b.get("distance") or 3.0))
                eye = (b.get("pivot") or Vector()) - b["forward"] * 10.0
            m.translation = eye
            cam.matrix_world = m
            scene.camera = cam

            def add_arrow(kind, origin, direction, length, label):
                me = _arrow_mesh(f"MCD_render_{kind}", length)
                temp_data.append(me)
                ob = bpy.data.objects.new(f"MCD_render_{kind}_{len(temp_objs)}", me)
                scene.collection.objects.link(ob)
                temp_objs.append(ob)
                mm = Vector(direction).normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
                mm = mm @ Matrix.Diagonal((length, length, length, 1.0))
                mm.translation = origin
                ob.matrix_world = mm
                ob.color = _COLORS[kind]
                arrows.append({"label": label, "color": {"marker": "绿", "part": "红", "target": "蓝"}[kind],
                               "origin": [round(x, 3) for x in origin],
                               "dir": [round(x, 3) for x in Vector(direction).normalized()]})

            if markers:
                for ob in list(bpy.data.objects):
                    if ob.type == "EMPTY" and ob.name.startswith(A.MARKER_PREFIX) and ob not in temp_objs:
                        mw = ob.matrix_world
                        add_arrow("marker", mw.translation.copy(), A.marker_dir(ob),
                                  max(0.05, float(ob.empty_display_size)) * 1.0, ob.name)
            if part:
                pp = A.canonical_part(part)
                d = A._eval_part(scene, armature, pp, side.upper() if side else None, None, None)
                key = A._KEY_MAP.get(pp)
                if d and key and d.get(key) is not None:
                    add_arrow("part", anchor, d[key], 0.18, f"{pp} 当前方向")
                if toward is not None:
                    t, how = resolve_direction(toward, scene=scene, armature=armature, origin=anchor,
                                               view="camera", mmd=mmd)
                    add_arrow("target", anchor, t, 0.18, f"目标 {how}")
            bpy.context.view_layer.update()
            r.engine = "BLENDER_WORKBENCH"
            r.resolution_x, r.resolution_y, r.resolution_percentage = w, h, 100
            r.pixel_aspect_x = r.pixel_aspect_y = 1.0
            r.film_transparent = False
            r.image_settings.file_format = "PNG"
            r.image_settings.color_mode = "RGB"
            sh.light = "STUDIO"
            sh.color_type = "OBJECT"
            r.filepath = path
            t0 = _time.time()
            bpy.ops.render.render(write_still=True)
            secs = round(_time.time() - t0, 2)
            cam_b = camera_basis(scene, cam)
            for a in arrows:
                o = Vector(a["origin"])
                sp = screen_point(cam_b, o)
                a["on_screen"] = sp["where"]
                arr = screen_arrow(cam_b, o, Vector(a["dir"]))
                if arr is not None:
                    a["arrow_on_screen"] = arr["words"]
    finally:
        scene.camera = saved["camera"]
        r.engine = saved["engine"]
        r.resolution_x, r.resolution_y, r.resolution_percentage = saved["rx"], saved["ry"], saved["rp"]
        r.filepath = saved["fp"]
        r.image_settings.file_format = saved["ff"]
        r.image_settings.color_mode = saved["cm"]
        r.film_transparent = saved["ft"]
        r.pixel_aspect_x, r.pixel_aspect_y = saved["px"], saved["py"]
        sh.light, sh.color_type = saved["light"], saved["ct"]
        for ob in temp_objs:
            try:
                bpy.data.objects.remove(ob, do_unlink=True)
            except Exception:  # noqa: BLE001
                pass
        for dblock in temp_data:
            try:
                if isinstance(dblock, bpy.types.Mesh):
                    bpy.data.meshes.remove(dblock)
                else:
                    bpy.data.cameras.remove(dblock)
            except Exception:  # noqa: BLE001
                pass
    return {"path": path, "frame": f, "view": b["name"], "size": [w, h], "render_s": secs,
            "arrows": arrows, "legend": "绿 = MCD 标记箭头，红 = 部位当前方向，蓝 = 目标方向（箭头从部位指出）"}


def _tool_render_view(ctx, frame=None, view="camera", part=None, side=None, size=None, markers=True,
                      toward=None, distance=None, **unknown):
    from . import agent_pose as P
    P.reject_unknown_args("render_view", _tool_render_view, unknown)
    import os
    out_dir = os.path.join(str(ctx.get("data_dir") or bpy.app.tempdir), "renders")
    res = render_view(ctx["scene"], ctx["armature"], frame=frame, view=str(view), part=part, side=side,
                      size=tuple(size) if size else (640, 360), markers=bool(markers), toward=toward,
                      out_dir=out_dir, distance=distance)
    return {"summary": f"渲染 @{res['frame']}（{res['view']}）→ {res['path']}（{res['render_s']} s）",
            "data": res, "warnings": [], "truncated": False,
            "hint": "能看图就打开 data.path 核对；数值验收仍用 probe_anatomy / orient_report"}


TOOLS = {"orient_report": _tool_orient_report, "conventions": _tool_conventions,
         "render_view": _tool_render_view}
