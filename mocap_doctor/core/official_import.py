"""官方 FlowHMR 产物一键导入：take 目录 → ``final.fbx`` + npz + 参考视频。

主布局（用户 2026-10-06 确认，以这条为准）::

    F:\\flowhmr_pack\\outputs\\video_to_fbx\\<take_hash>\\
        final.fbx               52 骨 ``Reference`` 骨架（source）
        body_postprocess.npz    Rh/betas/poses/trans（**没有** foot_contacts）
        final.npz               同上（后处理前的 body 结果）
        working_30fps.mp4       30 fps 参考视频
        camera.npz / main_track.npz / sam_mean_components.npz / hands\\*.pkl
                                相机、轨迹、HaMeR 手部（本模块不用）

临时布局（过渡期，仍然支持）::

    F:\\flowhmr_official_out\\<take>_OFFICIAL_post.fbx / .npz

接触分数（官方第 5 步 ``foot_contacts`` 的输入）取得顺序——**显式、可见，绝不静默**：

1. ``TAKE``            take 目录里任一 npz 带 ``foot_contacts`` → 用它；
2. ``OFFICIAL_TEMP``   没有就找**同 take** 的临时官方 npz
                       （``flowhmr_official_out\\<take>_OFFICIAL_post.npz``）→ 用它，
                       状态行注明「分数来自临时官方 npz，不是 take 目录」；
3. ``GEOMETRY``        都没有 → 回退几何检测器，状态行写
                       「无接触分数，planted 由几何推断」。

第 3 条**只让 planted 那一步降级**：FBX / 视频 / joints 的导入照常工作。

视频平面（用户原话：把参考视频以 mesh plane 导入，放到地平面上，正面朝 Y−）：
变换规格**由用户给死、顺序不能变**：绕 Z 轴 −90° → 沿 +Y 平移 1.0 m → 放大 ×1.5
→ **先缩放、再把底边对齐地面**（对齐用缩放后的包围盒最低点）。

* 竖直面，世界法向 **−Y**（局部网格法向 +X，RZ(−90°) 正好把它送到 −Y），
  Blender 正视图 numpad-1 从 −Y 看过去正对画面；
* 底边贴 **Z=0**（缩放之后再抬）、X 居中、宽高按视频真实宽高比；
* 高度是参数（默认 1.60 m ≈ 模型 身高，是**缩放前**的基准；世界高度 = 基准 × 1.5）；
* 沿 Y 的位置是参数，默认 **+1.0 m**（用户指定）：角色静止朝 −Y、正视镜头在 −Y 一侧，
  平面放在 +Y 就是「角色在画面前面、视频在后面」，1 m 也不会插进角色身体；
* 帧对齐：视频帧号是 1 基，``Blender 帧 f 显示视频第 f−offset 帧``（默认 offset=0；
  npz 第 i 帧 ↔ Blender 第 i+``npz_frame_offset`` 帧，默认 1 与
  ``foot_contacts.DEFAULT_FRAME_OFFSET`` 一致）。
"""

from __future__ import annotations

import os
from typing import Any, Sequence

# --- 主布局（take 目录）的文件名约定 ---------------------------------------
TAKE_FBX_NAMES = ("final.fbx",)
# 正规 take 目录现在有三份 body npz（用户 2026-10-16 说明，字段完全一致、都带 foot_contacts）：
#   body_origin.npz      FlowHMR 原始输出（未后处理，手指是 FlowHMR 的）
#   body_postprocess.npz 做完脚接触/IK 等后处理（手指仍是 FlowHMR 的）= 官方第 5 步的语义输入
#   final.npz            后处理结果再替换 30 个手指关节（HaMeR 合并）= 该 take 的最终动作
# 默认选 final.npz（FBX 对应的那一份）。三份 foot_contacts 实测逐比特相同 → 选择不影响
# planted；万一不一致则回落到 body_postprocess.npz（官方第 5 步语义输入），见
# :func:`contacts_consistency` / :func:`resolve_body_npz`。
BODY_NPZ_NAMES = ("final.npz", "body_postprocess.npz", "body_origin.npz")
DEFAULT_BODY_NPZ = "final.npz"
OFFICIAL_STEP5_NPZ = "body_postprocess.npz"
TAKE_NPZ_PREFERRED = BODY_NPZ_NAMES
NPZ_CHOICE_AUTO = "AUTO"
TAKE_VIDEO_NAMES = ("working_30fps.mp4",)

# --- 临时官方产物目录（第 2 条分支） ---------------------------------------
OFFICIAL_ROOT = r"F:\flowhmr_official_out"
OFFICIAL_NPZ_SUFFIX = "_OFFICIAL_post.npz"

# --- 视频平面默认值/理由 ---------------------------------------------------
# 用户给的变换规格（顺序不能变）：RZ −90° → +Y 1.0 m → ×1.5 → **先缩放再对齐地面**。
DEFAULT_PLANE_RZ_DEG = -90.0       # 局部 +X 法向 → 世界 −Y（正视图 numpad-1 正对画面）
DEFAULT_PLANE_Y_M = 1.0            # 角色朝 −Y、镜头在 −Y 一侧 → +Y 1.0 m = 角色身后 1 m
DEFAULT_PLANE_SCALE = 1.5          # 用户要的"放大 1.5 左右"
DEFAULT_PLANE_HEIGHT_M = 1.60      # 缩放**前**的基准高度 ≈ 模型 身高；世界高度 = 1.6×1.5 = 2.4 m
PLANE_OBJECT_NAME = "MCD_reference_video"
PLANE_MATERIAL_NAME = "MCD_reference_video_mat"

CONTACT_TAKE = "TAKE"
CONTACT_OFFICIAL_TEMP = "OFFICIAL_TEMP"
CONTACT_GEOMETRY = "GEOMETRY"

TAKE_DIR_HINT = (
    r"F:\flowhmr_pack\outputs\video_to_fbx\<take_hash>  （内含 final.fbx / "
    "final.npz + body_postprocess.npz + body_origin.npz / working_30fps.mp4）"
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
    return found[0] if found else ""


def is_body_npz(path: str) -> bool:
    """像 ``body_postprocess.npz`` 那样的 body 结果：带 ``poses``（或 ``trans``）。"""

    try:
        keys = npz_keys(path)
    except Exception:  # noqa: BLE001 - 读不了的当不是 body npz
        return False
    return "poses" in keys or "trans" in keys


def pick_npz(take_dir: str, preferred: Sequence[str] = TAKE_NPZ_PREFERRED) -> str:
    """take 目录里的 body npz：``body_postprocess.npz`` → ``final.npz`` → 其它带 poses 的顶层 npz。

    刻意**不**退到任意 npz：``camera.npz`` / ``main_track.npz`` /
    ``sam_mean_components.npz`` 都不是 body 结果，误当配对会让面板显示 0 帧。
    """

    for name in preferred:
        cand = os.path.join(take_dir, name)
        if os.path.isfile(cand):
            return cand
    try:
        entries = sorted(
            entry for entry in os.listdir(take_dir)
            if entry.lower().endswith(".npz")
        )
    except OSError:
        return ""
    for entry in entries:
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


def official_npz_for_take(
    take_dir: str,
    official_root: str = OFFICIAL_ROOT,
    *,
    take_token: str | None = None,
) -> str:
    """同 take 的临时官方 npz（``<official_root>\\<take>*_OFFICIAL_post.npz``）。

    先按 take token（``t999_H4_b8abd355e6f2`` → ``t999``）找，再退到「文件名以 take
    token 开头且带 foot_contacts」的任意 npz；找不到返回空串。只读，不写。
    """

    if not official_root or not os.path.isdir(official_root):
        return ""
    from . import foot_contacts as fc

    token = take_token or fc.take_identity(take_dir)
    if not token:
        return ""
    exact = os.path.join(official_root, f"{token}{OFFICIAL_NPZ_SUFFIX}")
    if os.path.isfile(exact) and npz_has_foot_contacts(exact):
        return exact
    try:
        entries = sorted(os.listdir(official_root))
    except OSError:
        return ""
    for entry in entries:
        if not entry.lower().endswith(".npz"):
            continue
        if not entry.lower().startswith(token.lower()):
            continue
        cand = os.path.join(official_root, entry)
        if npz_has_foot_contacts(cand):
            return cand
    return ""


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


def take_body_npz(take_dir: str) -> dict[str, str]:
    """take 目录里存在的三份标准 body npz：``{文件名: 路径}``（按偏好顺序）。"""

    out: dict[str, str] = {}
    if not take_dir or not os.path.isdir(take_dir):
        return out
    for name in BODY_NPZ_NAMES:
        cand = os.path.join(take_dir, name)
        if os.path.isfile(cand):
            out[name] = cand
    return out


def contacts_consistency(take_dir: str) -> dict[str, Any]:
    """三份 npz 的 ``foot_contacts`` 是否一致（运行时校验，写进状态行）。

    一致 → 选哪份都不影响 planted；不一致 → 默认改用 ``body_postprocess.npz``
    （官方第 5 步的语义输入）并说明差异。校验只读、逐比特比较（float32 精确相等）。
    """

    import numpy as np

    result: dict[str, Any] = {
        "present": take_body_npz(take_dir),
        "with_contacts": [],
        "equal": None,
        "max_diff": 0.0,
        "diff_frames": 0,
        "reference": "",
        "same_shape": True,
        "status": "",
    }
    arrays: dict[str, Any] = {}
    for name, path in result["present"].items():
        fc = load_foot_contacts(path)
        if fc is not None:
            arrays[name] = fc
            result["with_contacts"].append(name)
    if len(arrays) < 2:
        return result
    reference = OFFICIAL_STEP5_NPZ if OFFICIAL_STEP5_NPZ in arrays else result["with_contacts"][0]
    result["reference"] = reference
    ref = arrays[reference]
    for name, arr in arrays.items():
        if arr.shape != ref.shape:
            result["same_shape"] = False
            result["equal"] = False
            result["status"] = (
                f"**三份接触分数形状不一致**（{name} {arr.shape} vs {reference} {ref.shape}）"
                f" → 默认改用官方第 5 步语义输入 {reference}"
            )
            return result
    max_diff = 0.0
    diff_frames = 0
    for name, arr in arrays.items():
        diff = np.abs(arr - ref)
        max_diff = max(max_diff, float(diff.max()) if diff.size else 0.0)
        diff_frames = max(diff_frames, int((diff.max(axis=1) > 0).sum()) if diff.size else 0)
    result["max_diff"] = max_diff
    result["diff_frames"] = diff_frames
    result["equal"] = max_diff == 0.0
    names = "/".join(result["with_contacts"])
    if result["equal"]:
        result["status"] = f"三份（{names}）接触分数逐比特相同，选择不影响 planted"
    else:
        result["status"] = (
            f"**三份接触分数不一致**（{names} 最大差 {max_diff:.3g}，{diff_frames} 帧不同）"
            f" → 默认改用官方第 5 步语义输入 {reference}"
        )
    return result


def resolve_body_npz(take_dir: str, choice: str = NPZ_CHOICE_AUTO) -> dict[str, Any]:
    """按选择器/一致性校验决定用哪份 body npz。

    返回 ``{"name", "path", "frames", "reason", "consistency", "switched"}``。
    ``choice`` = ``AUTO`` 或三个标准文件名之一（选了目录里没有的 → 退回 AUTO 并说明）。
    """

    present = take_body_npz(take_dir)
    consistency = contacts_consistency(take_dir)
    with_contacts = consistency["with_contacts"]
    choice = str(choice or NPZ_CHOICE_AUTO)
    reason = ""
    switched = False
    if choice != NPZ_CHOICE_AUTO:
        if choice in present:
            name = choice
            reason = f"面板选了 {choice}"
        else:
            name = ""
            reason = f"面板选的 {choice} 不在 take 目录里 → 退回自动"
    else:
        name = ""
    if not name:
        # AUTO：先看三份带 foot_contacts 的是否一致
        if consistency["equal"] is False and OFFICIAL_STEP5_NPZ in with_contacts:
            name = OFFICIAL_STEP5_NPZ
            switched = True
            reason = reason or "三份接触分数不一致 → 用官方第 5 步语义输入"
        else:
            for candidate in (DEFAULT_BODY_NPZ, *BODY_NPZ_NAMES):
                if candidate in with_contacts:
                    name = candidate
                    break
            if not name:
                for candidate in BODY_NPZ_NAMES:
                    if candidate in present:
                        name = candidate
                        break
            reason = reason or "自动（final → body_postprocess → body_origin）"
    path = present.get(name, "")
    frames = npz_frame_count(path) if path else 0
    return {
        "name": name,
        "path": path,
        "frames": frames,
        "reason": reason,
        "consistency": consistency,
        "switched": switched,
    }


def find_contacts(
    take_dir: str,
    *,
    explicit: str = "",
    official_root: str = OFFICIAL_ROOT,
    choice: str = NPZ_CHOICE_AUTO,
) -> dict[str, Any]:
    """接触分数按 1/2/3 顺序解析，返回来源与**面板原样显示**的状态文字。

    返回 ``{"source", "path", "frames", "status", "note"}``；``source`` 取
    ``CONTACT_TAKE`` / ``CONTACT_OFFICIAL_TEMP`` / ``CONTACT_GEOMETRY``。
    第 1 档会把「三份 npz 的 foot_contacts 是否一致」写进 ``status``（运行时校验）。
    """

    if explicit:
        if os.path.isfile(explicit):
            if npz_has_foot_contacts(explicit):
                return {
                    "source": CONTACT_TAKE,
                    "path": explicit,
                    "frames": npz_frame_count(explicit),
                    "status": f"接触分数：手动指定（{explicit}）",
                    "note": "手动指定",
                }
            return {
                "source": CONTACT_GEOMETRY,
                "path": "",
                "frames": 0,
                "status": f"接触分数：手动指定的 npz 没有 foot_contacts（{explicit}）→ "
                          "无接触分数，planted 由几何推断",
                "note": "manual without foot_contacts",
            }
        return {
            "source": CONTACT_GEOMETRY,
            "path": "",
            "frames": 0,
            "status": f"接触分数：手动指定的文件不存在（{explicit}）→ "
                      "无接触分数，planted 由几何推断",
            "note": "manual missing",
        }

    # 1) take 目录里有带 foot_contacts 的 npz（正规 take：final / body_postprocess / body_origin）
    if take_dir and os.path.isdir(take_dir):
        resolved = resolve_body_npz(take_dir, choice)
        consistency = resolved["consistency"]
        if consistency["with_contacts"]:
            name = resolved["name"]
            frames = resolved["frames"]
            status = f"接触分数：take 目录内 {name}（{frames} 帧，带 foot_contacts）"
            if consistency["status"]:
                status += f"；{consistency['status']}"
            elif len(consistency["with_contacts"]) == 1:
                status += "；目录里只有这一份带 foot_contacts"
            if resolved["switched"]:
                status += f"；已自动切到 {name}"
            elif resolved["reason"]:
                status += f"；{resolved['reason']}"
            return {
                "source": CONTACT_TAKE,
                "path": resolved["path"],
                "frames": frames,
                "status": status,
                "note": f"take dir:{name}",
                "npz_name": name,
                "npz_choice": choice,
                "consistency": consistency,
            }

    # 2) 同 take 的临时官方 npz
    official = official_npz_for_take(take_dir, official_root)
    if official:
        return {
            "source": CONTACT_OFFICIAL_TEMP,
            "path": official,
            "frames": npz_frame_count(official),
            "status": "接触分数：**分数来自临时官方 npz，不是 take 目录**"
                      f"（{official}，{npz_frame_count(official)} 帧）",
            "note": "official temp",
        }

    # 3) 几何回退
    return {
        "source": CONTACT_GEOMETRY,
        "path": "",
        "frames": 0,
        "status": "接触分数：无接触分数，planted 由几何推断"
                  "（take 目录 npz 无 foot_contacts，也没有同 take 官方 npz）",
        "note": "geometry fallback",
    }


def scan_take_dir(
    path: str,
    *,
    official_root: str = OFFICIAL_ROOT,
    contacts_explicit: str = "",
    npz_choice: str = NPZ_CHOICE_AUTO,
) -> dict[str, Any]:
    """一键扫描：take 目录 → fbx / npz / 视频 / 接触分数 + 面板状态文字。

    ``npz_choice`` = ``AUTO`` 或三份标准 body npz 的文件名（面板选择器）；
    选中的那份写进 ``info["npz"]``/``npz_frames``，三份 foot_contacts 的一致性
    （``info["contacts_consistency"]``）写进状态行。
    """

    take_dir = resolve_take_dir(path)
    info: dict[str, Any] = {
        "take_dir": take_dir,
        "exists": bool(take_dir) and os.path.isdir(take_dir),
        "fbx": "",
        "npz": "",
        "npz_name": "",
        "npz_frames": 0,
        "npz_keys": [],
        "npz_candidates": {},
        "npz_choice": npz_choice,
        "contacts_consistency": {},
        "video": "",
        "video_note": "",
        "contacts": {},
    }
    if not info["exists"]:
        info["status"] = f"找不到 take 目录：{take_dir or '(空)'}　例：{TAKE_DIR_HINT}"
        info["contacts"] = find_contacts("", explicit=contacts_explicit,
                                         official_root=official_root,
                                         choice=npz_choice)
        return info

    info["fbx"] = pick_fbx(take_dir)
    info["npz_candidates"] = take_body_npz(take_dir)
    resolved = resolve_body_npz(take_dir, npz_choice)
    info["npz"] = resolved["path"] or pick_npz(take_dir)
    info["npz_name"] = os.path.basename(info["npz"]) if info["npz"] else ""
    info["contacts_consistency"] = resolved["consistency"]
    if info["npz"]:
        try:
            info["npz_frames"] = npz_frame_count(info["npz"])
            info["npz_keys"] = npz_keys(info["npz"])
        except Exception as exc:  # noqa: BLE001 - 坏 npz 也只在状态行说
            info["video_note"] = f"npz 读取失败：{exc}"
    video, video_note = find_video(take_dir)
    info["video"] = video
    info["video_note"] = video_note
    info["contacts"] = find_contacts(take_dir, explicit=contacts_explicit,
                                     official_root=official_root,
                                     choice=npz_choice)

    take_label = os.path.basename(take_dir.rstrip("\\/")) or take_dir
    parts = [f"take={take_label}"]
    parts.append(f"final.fbx={'有' if info['fbx'] else '缺'}")
    parts.append(
        f"npz={os.path.basename(info['npz'])}"
        f"（{info['npz_frames']} 帧）" if info["npz"] else "npz=缺"
    )
    if len(info["npz_candidates"]) > 1:
        parts.append(f"另可选={'/'.join(n for n in info['npz_candidates'] if n != info['npz_name'])}")
    parts.append(
        f"视频={os.path.basename(video)}" if video else "视频=缺"
    )
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
