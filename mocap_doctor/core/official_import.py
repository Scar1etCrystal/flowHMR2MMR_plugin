"""动作数据目录导入：像 ``F:\\T1`` 这样的扁平目录 → 骨架 + 贴地数据（+ 可选参考视频）。

当前格式（破坏性变更后只认这一种布局）::

    F:\\T1\\
        PREVIEW <name>_flowhmr_skin_t_pose.fbx    SMPL-H 网格 + 规范骨架，自带整段动作
        PREVIEW <name>_flowhmr.npz                poses(T,156) / trans / betas /
                                                  mocap_framerate / gender

特点与处理方式：

* 文件名带 ``PREVIEW `` 前缀，目录是**平的**：没有 take 子目录，也没有 mp4 与
  ``video_info.json``；找不到视频时参考视频平面直接跳过，不影响动作导入。
* FBX 骨架使用**规范命名**（``Hips`` / ``LeftUpLeg`` / ``LeftHandIndex1`` …），
  导入时由 :mod:`mocap_doctor.core.source_naming` 一次性改成 FlowHMR 原生名
  （``Pelvis`` / ``L_Hip`` / ``L_Index1`` …，骨骼 + 顶点组 + 曲线路径三处同步）。
  下游（bmap 源骨名、脚部几何检测、FIX A/B/C、手/脚间距）只看原生名。
* 接触分数（``foot_contacts``，(T,4)，通道 ``left_ankle / left_toe /
  right_ankle / right_toe``）有两种新格式位置，按优先级认：

  1. ``CONTACT_MANUAL``   面板手填的 npz；
  2. ``CONTACT_NPZ``      主体 npz 自带 ``foot_contacts``（``web_jobs/motion.npz`` 那种）；
  3. ``CONTACT_SIDECAR``  同目录的 ``<name>_foot_contacts.npz`` 旁路文件；
  4. ``CONTACT_GEOMETRY`` 都没有 → 回退骨架几何检测，状态行注明。

  旁路文件若带 ``frame_start``，会换算成 ``npz_frame_offset``（约定
  ``blender_frame = npz_index + npz_frame_offset``）。

视频平面（``MCD_reference_video``）仅在目录里存在 mp4 时创建；变换规格不变：
绕 Z 轴 −90° → 沿 +Y 平移 1.0 m → 放大 ×1.5 → 先缩放再把底边对齐地面。

帧范围取自导入 Action 的 ``frame_range``（新格式是 0–7010，落到场景即 1–7010）。
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

# --- 主布局（take 目录）的文件名约定 ---------------------------------------
TAKE_FBX_NAMES = ("final.fbx",)
# 正规 take 目录现在有三份 body npz（用户 2026-10-16 说明，字段完全一致、都带 foot_contacts）：
#   body_origin.npz      FlowHMR 原始输出（未后处理，手指是 FlowHMR 的）
#   body_postprocess.npz 做完脚接触/IK 等后处理（手指仍是 FlowHMR 的）= 官方第 5 步的语义输入
#   final.npz            后处理结果再替换 30 个手指关节（HaMeR 合并）= 该 take 的最终动作
# 默认选 final.npz（FBX 对应的那一份）。三份 foot_contacts 实测逐比特相同 → 选择不影响
# planted；万一不一致则回落到 body_postprocess.npz（官方第 5 步语义输入），见
# :func:`find_contacts`。
# 新格式（2026-11 起）只有一支 ``PREVIEW <name>_flowhmr.npz``，接触分数要么直接
# 存在这支 npz 里，要么单独一支 ``<name>_foot_contacts.npz``。旧的
# final / body_postprocess / body_origin 三份 npz 及其一致性校验已删除（破坏性）。
BODY_NPZ_HINT = "flowhmr"
CONTACTS_NPZ_MARK = "foot_contacts"
NPZ_SIDECAR_MARKS = (
    "foot_contacts", "bbox", "camera", "joints", "sam3d", "main_track",
    "sam_mean_components", "input_",
)
TAKE_VIDEO_NAMES = ("working_30fps.mp4",)

# --- 视频平面默认值/理由 ---------------------------------------------------
# 用户给的变换规格（顺序不能变）：RZ −90° → +Y 1.0 m → ×1.5 → **先缩放再对齐地面**。
DEFAULT_PLANE_RZ_DEG = -90.0       # 局部 +X 法向 → 世界 −Y（正视图 numpad-1 正对画面）
DEFAULT_PLANE_Y_M = 1.0            # 角色朝 −Y、镜头在 −Y 一侧 → +Y 1.0 m = 角色身后 1 m
DEFAULT_PLANE_SCALE = 1.5          # 用户要的"放大 1.5 左右"
DEFAULT_PLANE_HEIGHT_M = 1.60      # 缩放**前**的基准高度 ≈ 模型 身高；世界高度 = 1.6×1.5 = 2.4 m
PLANE_OBJECT_NAME = "MCD_reference_video"
PLANE_MATERIAL_NAME = "MCD_reference_video_mat"

CONTACT_TAKE = "TAKE"          # 保留给下游既有判断：有分数（任意来源）
CONTACT_GEOMETRY = "GEOMETRY"  # 没有分数 → 几何推断
CONTACT_NPZ = "NPZ"            # 分数在主 npz 里
CONTACT_SIDECAR = "SIDECAR"    # 分数在同目录旁路文件里
CONTACT_MANUAL = "MANUAL"      # 面板手填

TAKE_DIR_HINT = (
    r"F:\T1  （内含 PREVIEW <name>_flowhmr_skin_t_pose.fbx + PREVIEW <name>_flowhmr.npz；"
    "扁平目录，无需 take 子目录和视频）"
)
FOOT_CONTACTS_KEY = "foot_contacts"


# ---------------------------------------------------------------------------
# 纯函数（不依赖 bpy，方便单测）


def resolve_take_dir(path: str, *, fbx_names: Sequence[str] = TAKE_FBX_NAMES) -> str:
    """把一个 take 目录或目录里的某个文件解析成 take 目录。

    - 传目录 → 原样（去掉结尾分隔符）；
    - 传文件 → 父目录（``...\\<take>\\final.fbx`` → ``...\\<take>``）；
    - 传 ``...\\<take>`` 但目录不存在时也原样返回，让上层给出「找不到目录」的可见提示。
    """

    text = str(path or "").strip().strip('"')
    if not text:
        return ""
    text = os.path.normpath(text)
    if os.path.isdir(text):
        return text
    if os.path.isfile(text):
        return os.path.dirname(text)
    # 不存在的路径：带扩展名（含 final.fbx 这种）当文件，否则当目录
    if os.path.splitext(text)[1]:
        return os.path.dirname(text)
    return text


def pick_fbx(take_dir: str, names: Sequence[str] = TAKE_FBX_NAMES) -> str:
    """take 目录里的 FBX：先按约定名，再退到目录里唯一的 ``*.fbx``。"""

    for name in names:
        cand = os.path.join(take_dir, name)
        if os.path.isfile(cand):
            return cand
    try:
        found = sorted(
            os.path.join(take_dir, entry)
            for entry in os.listdir(take_dir)
            if entry.lower().endswith(".fbx")
        )
    except OSError:
        return ""
    # 新格式只有一支 ``*flowhmr*_skin_t_pose.fbx``；同目录混入别的 fbx 时优先它。
    preferred = [path for path in found if "flowhmr" in os.path.basename(path).lower()]
    return (preferred or found)[0] if found else ""


def is_body_npz(path: str) -> bool:
    """像 ``body_postprocess.npz`` 那样的 body 结果：带 ``poses``（或 ``trans``）。"""

    try:
        keys = npz_keys(path)
    except Exception:  # noqa: BLE001 - 读不了的当不是 body npz
        return False
    return "poses" in keys or "trans" in keys


def pick_npz(take_dir: str) -> str:
    """take 目录里的 body npz：优先名字带 ``flowhmr`` 的那支，跳过旁路文件。

    ``*_foot_contacts.npz`` / bbox / camera / joints 这类不是 body 结果，
    文件名先排掉，再按 `is_body_npz` 确认真的带 poses。
    """

    try:
        entries = sorted(
            entry for entry in os.listdir(take_dir)
            if entry.lower().endswith(".npz")
        )
    except OSError:
        return ""
    ordered = sorted(entries, key=lambda name: (0 if BODY_NPZ_HINT in name.lower() else 1, name))
    for entry in ordered:
        if any(mark in entry.lower() for mark in NPZ_SIDECAR_MARKS):
            continue
        cand = os.path.join(take_dir, entry)
        if is_body_npz(cand):
            return cand
    return ""


def find_video(take_dir: str) -> tuple[str, str]:
    """take 目录里的参考视频。返回 ``(path, note)``；找不到时 path 为空 + 可见提示。"""

    for name in TAKE_VIDEO_NAMES:
        cand = os.path.join(take_dir, name)
        if os.path.isfile(cand):
            return cand, ""
    try:
        found = sorted(
            os.path.join(take_dir, entry)
            for entry in os.listdir(take_dir)
            if entry.lower().endswith(".mp4")
        )
    except OSError:
        found = []
    if found:
        return found[0], f"用目录里第一支 mp4（没有 {'/'.join(TAKE_VIDEO_NAMES)}）"
    return "", "take 目录里没有 mp4（视频平面跳过，其它照常导入）"


def iter_npz_files(take_dir: str) -> list[str]:
    """take 目录（含 ``hands/`` 之类子目录）里的全部 npz，排序稳定。"""

    out: list[str] = []
    for root, dirs, files in os.walk(take_dir):
        dirs.sort()
        for entry in sorted(files):
            if entry.lower().endswith(".npz"):
                out.append(os.path.join(root, entry))
    return out


def npz_frame_count(path: str) -> int:
    """npz 的帧数：先看 ``num_frames``，再看任何首维像帧数的数组。"""

    import numpy as np

    if not path or not os.path.isfile(path):
        return 0
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.files)
        if "num_frames" in keys:
            try:
                return int(np.asarray(data["num_frames"]).reshape(-1)[0])
            except Exception:  # noqa: BLE001 - 坏字段不该让面板失效
                pass
        for key in ("poses", "trans", "local_joints_positions", "foot_contacts"):
            if key in keys:
                arr = data[key]
                if getattr(arr, "ndim", 0) >= 1:
                    return int(arr.shape[0])
        return 0


def npz_keys(path: str) -> list[str]:
    import numpy as np

    if not path or not os.path.isfile(path):
        return []
    with np.load(path, allow_pickle=True) as data:
        return list(data.files)


def npz_has_foot_contacts(path: str) -> bool:
    """npz 里有没有官方第 5 步的接触分数 ``foot_contacts``。"""

    try:
        return "foot_contacts" in npz_keys(path)
    except Exception:  # noqa: BLE001 - 读不了的 npz 当作「没有」
        return False


def load_foot_contacts(path: str) -> Any:
    """读 ``foot_contacts`` 数组（读不了/没有 → None）。"""

    import numpy as np

    if not path or not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            if FOOT_CONTACTS_KEY not in data.files:
                return None
            return np.asarray(data[FOOT_CONTACTS_KEY], dtype=np.float64)
    except Exception:  # noqa: BLE001 - 坏 npz 当没有
        return None


def pick_contacts_npz(take_dir: str) -> dict[str, Any]:
    """旁路接触文件 ``<name>_foot_contacts.npz``。"""

    try:
        entries = sorted(
            entry for entry in os.listdir(take_dir)
            if entry.lower().endswith(".npz")
        )
    except OSError:
        return {}
    for entry in entries:
        if CONTACTS_NPZ_MARK not in entry.lower():
            continue
        cand = os.path.join(take_dir, entry)
        if not npz_has_foot_contacts(cand):
            continue
        frame_start, fps, channels = 0, 0.0, []
        try:
            with np.load(cand, allow_pickle=True) as data:
                if "frame_start" in data.files:
                    frame_start = int(np.asarray(data["frame_start"]).reshape(-1)[0])
                if "fps" in data.files:
                    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                if "channels" in data.files:
                    channels = [str(item) for item in np.asarray(data["channels"]).reshape(-1)]
        except Exception:  # noqa: BLE001 - 元数据读不了不影响分数本身
            pass
        return {"path": cand, "frame_start": frame_start, "fps": fps, "channels": channels}
    return {}


def find_contacts(take_dir: str, *, explicit: str = "") -> dict[str, Any]:
    """接触分数来源：手填 > 主 npz 自带 > 旁路 ``*_foot_contacts.npz`` > 无（几何推断）。

    ``frame_start`` 是 npz 第 0 行对应的源帧号，调用方据此换算 Blender 帧偏移。
    """

    if explicit:
        if os.path.isfile(explicit) and npz_has_foot_contacts(explicit):
            return {
                "source": "MANUAL",
                "path": explicit,
                "frame_start": 0,
                "status": f"接触分数：手动指定（{os.path.basename(explicit)}）",
            }
        return {
            "source": "NONE",
            "path": "",
            "frame_start": 0,
            "status": "接触分数：手动指定的文件没有 foot_contacts → 由几何推断",
        }
    body = pick_npz(take_dir) if take_dir else ""
    if body and npz_has_foot_contacts(body):
        return {
            "source": CONTACT_NPZ,
            "path": body,
            "frame_start": 0,
            "status": f"接触分数：主 npz（{os.path.basename(body)}）",
        }
    sidecar = pick_contacts_npz(take_dir) if take_dir else {}
    if sidecar:
        return {
            "source": CONTACT_SIDECAR,
            "path": sidecar["path"],
            "frame_start": int(sidecar.get("frame_start", 0)),
            "fps": float(sidecar.get("fps", 0.0)),
            "channels": list(sidecar.get("channels") or ()),
            "status": f"接触分数：{os.path.basename(sidecar['path'])}",
        }
    return {
        "source": "NONE",
        "path": "",
        "frame_start": 0,
        "status": "接触分数：无接触分数，planted 由几何推断",
    }


def scan_take_dir(path: str, *, contacts_explicit: str = "") -> dict[str, Any]:
    """一键扫描：take 目录 → fbx / npz / 视频 / 接触分数 + 面板状态文字。"""

    take_dir = resolve_take_dir(path)
    info: dict[str, Any] = {
        "take_dir": take_dir,
        "exists": bool(take_dir) and os.path.isdir(take_dir),
        "fbx": "",
        "npz": "",
        "npz_name": "",
        "npz_frames": 0,
        "npz_keys": [],
        "video": "",
        "video_note": "",
        "contacts": {},
    }
    if not info["exists"]:
        info["status"] = f"找不到 take 目录：{take_dir or '(空)'}　例：{TAKE_DIR_HINT}"
        info["contacts"] = find_contacts("", explicit=contacts_explicit)
        return info

    info["fbx"] = pick_fbx(take_dir)
    info["npz"] = pick_npz(take_dir)
    info["npz_name"] = os.path.basename(info["npz"]) if info["npz"] else ""
    if info["npz"]:
        try:
            info["npz_frames"] = npz_frame_count(info["npz"])
            info["npz_keys"] = npz_keys(info["npz"])
        except Exception as exc:  # noqa: BLE001 - 坏 npz 也只在状态行说
            info["video_note"] = f"npz 读取失败：{exc}"
    video, video_note = find_video(take_dir)
    info["video"] = video
    if video_note:
        info["video_note"] = video_note
    info["contacts"] = find_contacts(take_dir, explicit=contacts_explicit)

    take_label = os.path.basename(take_dir.rstrip("\\/")) or take_dir
    parts = [f"take={take_label}"]
    parts.append("动作文件=" + (os.path.basename(info["fbx"]) if info["fbx"] else "缺"))
    parts.append(
        f"npz={os.path.basename(info['npz'])}（{info['npz_frames']} 帧）"
        if info["npz"] else "npz=缺"
    )
    parts.append(f"视频={os.path.basename(video)}" if video else "视频=缺")
    parts.append(str((info.get("contacts") or {}).get("status") or "接触分数：无"))
    info["status"] = "官方产物：" + " · ".join(parts)
    return info


def fbx_display_name(path: str) -> str:
    """``...\\<take>\\final.fbx`` → ``<take>/final.fbx``（面板上显示用）。"""

    if not path:
        return ""
    take = os.path.basename(os.path.dirname(path))
    return f"{take}/{os.path.basename(path)}"


def read_video_info(take_dir: str) -> dict[str, Any]:
    """take 目录的 ``video_info.json``（frames/fps/width/height）。

    正规导出里一定有它，所以面板**不用**依赖 Blender 能不能解出 mp4 的尺寸：
    视频读不出来时按这里的宽高比建平面，并在状态行说明。
    """

    import json

    path = os.path.join(str(take_dir or ""), "video_info.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:  # noqa: BLE001 - 元数据坏了不影响导入
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("frames", "fps", "width", "height"):
        if key in data:
            try:
                out[key] = float(data[key])
            except (TypeError, ValueError):
                pass
    return out


def request_source_hint(take_dir: str) -> str:
    """``request.json`` 里的原始视频路径（找不到视频文件时的线索，不保证存在）。"""

    import json

    path = os.path.join(str(take_dir or ""), "request.json")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(data, dict):
        return str(data.get("source") or "")
    return ""


# ---------------------------------------------------------------------------
# 视频平面几何（纯函数：e2e 直接验法向/底边/尺寸/顺序，不用起 Blender）
#
# 用户给死的变换规格（顺序不能变）：
#   1) 绕 Z 轴旋转 −90°；
#   2) 沿 +Y 平移 1.0 m；
#   3) 放大 ×1.5 左右；
#   4) **先缩放，再把底边对齐地面** —— 对齐用的是缩放后的包围盒最低点。
# 局部网格做成「中心在原点、位于局部 YZ 平面、法向 +X 的矩形」：
# RZ(−90°) 把局部 +X 送到世界 −Y（法向朝 −Y ✓）、局部 +Y 送到世界 +X
# （UV 的 u 随世界 +X = 正视图画面右，正立不镜像）。缩放绕原点 → 底边离开地面
# （min_z = −1.5·h/2 ≠ 0），最后按缩放后的 min_z 抬到 0。


def plane_size_for_aspect(aspect: float, height_m: float) -> tuple[float, float]:
    """按真实宽高比 + 基准高度算 ``(宽, 高)``（都是**缩放前**的尺寸）；非正输入退回 1:1。"""

    aspect = float(aspect) if aspect and aspect > 0 else 1.0
    height = max(float(height_m), 1e-3)
    return height * aspect, height


def plane_local_geometry(width_m: float, height_m: float) -> list[tuple[float, float, float]]:
    """局部网格：中心在原点、YZ 平面内、法向 +X；``宽`` 沿局部 Y、``高`` 沿局部 Z。"""

    half_w = max(float(width_m), 1e-4) * 0.5
    half_h = max(float(height_m), 1e-4) * 0.5
    return [
        (0.0, -half_w, -half_h),
        (0.0, half_w, -half_h),
        (0.0, half_w, half_h),
        (0.0, -half_w, half_h),
    ]


PLANE_UVS = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))  # u→局部+Y(世界+X)，v→局部+Z
PLANE_LOCAL_NORMAL = (1.0, 0.0, 0.0)                          # 局部法向 +X → RZ(−90°) 后 = −Y


def rotate_z(point: Sequence[float], degrees: float) -> tuple[float, float, float]:
    """绕 Z 轴旋转（度），右手系。"""

    import math

    theta = math.radians(float(degrees))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x, y, z = (float(v) for v in point)
    return (x * cos_t - y * sin_t, x * sin_t + y * cos_t, z)


def plane_transform(
    width_m: float,
    height_m: float,
    *,
    y: float = DEFAULT_PLANE_Y_M,
    rot_z_deg: float = DEFAULT_PLANE_RZ_DEG,
    scale: float = DEFAULT_PLANE_SCALE,
    x_center: float = 0.0,
) -> dict[str, Any]:
    """算平面最终变换 + 对齐前后的实际数字（纯函数，面板与 e2e 共用同一套算式）。

    顺序：局部网格 → ×scale → RZ(rot_z_deg) → 平移 (x_center, y, 0) →
    按**缩放后**包围盒的最低点把 z 抬到 0。
    """

    local = plane_local_geometry(width_m, height_m)
    s = float(scale)
    scaled = [(px * s, py * s, pz * s) for px, py, pz in local]
    rotated = [rotate_z(p, rot_z_deg) for p in scaled]
    placed = [(p[0] + float(x_center), p[1] + float(y), p[2]) for p in rotated]
    min_z_before = min(p[2] for p in placed)
    max_z_before = max(p[2] for p in placed)
    dz = -min_z_before                       # 缩放之后才对齐
    world = [(p[0], p[1], p[2] + dz) for p in placed]
    xs = [p[0] for p in world]
    ys = [p[1] for p in world]
    zs = [p[2] for p in world]
    normal = rotate_z(PLANE_LOCAL_NORMAL, rot_z_deg)
    return {
        "local": local,
        "scale": s,
        "rot_z_deg": float(rot_z_deg),
        "location": (float(x_center), float(y), dz),
        "world": world,
        "min_z_before_align": min_z_before,
        "max_z_before_align": max_z_before,
        "min_z_after_align": min(zs),
        "max_z_after_align": max(zs),
        "align_dz": dz,
        "normal": normal,
        "width": max(xs) - min(xs),
        "height": max(zs) - min(zs),
        "center_x": (max(xs) + min(xs)) * 0.5,
        "y_plane": max(ys),
        "expected_width": float(width_m) * s,
        "expected_height": float(height_m) * s,
    }


def polygon_normal(verts: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """四边形的几何法向（右手系，与 Blender 的 loop 顺序一致）。"""

    ax, ay, az = verts[0]
    bx, by, bz = verts[1]
    cx, cy, cz = verts[2]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - bx, cy - by, cz - bz
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    norm = (nx * nx + ny * ny + nz * nz) ** 0.5
    if norm < 1e-12:
        return (0.0, 0.0, 0.0)
    return (nx / norm, ny / norm, nz / norm)
