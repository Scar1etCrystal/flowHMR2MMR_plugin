"""FlowHMR npz foot contacts → planted ranges (the default contact source).

**This is a port of FlowHMR's own post-processing step 5**, not a rule of our
own.  Official implementation:

* ``F:\\flowhmr_pack\\app\\flowhmr\\flowhmr\\utils\\postprocess_utils.py:112``
  ``find_foot_contact_interval(foot_contacts, foot_contact_positions, joint_names,
  threshold=0.5, velocity_threshold=0.15*3, shrink_frame=3,
  position_range_threshold=0.1)`` (plus the helpers ``_find_segments`` at :79 and
  ``_split_interval_by_position_range`` at :86).
* called from ``...\\utils\\runtime_v2m_postprocess_simple.py:242`` inside
  ``PostprocessPipeline.run`` (:228) as the 5th post-processing step, with
  ``threshold=self.static_threshold`` = **0.7** (:197) and
  ``position_range_threshold={'L_Ankle':0.1,'R_Ankle':0.1,'L_Foot':0.05,'R_Foot':0.05}``
  (:245-248); the 3D positions come from the FK'd keypoints of the four joints
  ``FOOT_CONTACT_NAMES = ['L_Ankle','L_Foot','R_Ankle','R_Foot']`` (:111) — so the
  4 contact scores are **left ankle, left foot, right ankle, right foot**, and the
  split thresholds are 10 cm for the ankles and 5 cm for the feet.

Channel labels (corrected): my earlier note called the second channel per side
"toe"; the official name is ``L_Foot``/``R_Foot`` (the SMPL-H foot joint, i.e. the
ball/toe end).  The measured left-then-right grouping stands: correlating each
channel against the npz's ``local_joints_positions`` gives ch0/ch1 ↔ left
(0.42/0.45 against -height, ≈-0.01 against the right foot) and ch2/ch3 ↔ right
(0.41/0.45, left side negative).  The scores are a smoothed 0..1 signal (saturate
at ~1/~0, slight overshoot -0.037..1.07, nothing between 0.25 and 0.75).

The official rule, transcribed 1:1 into :func:`find_foot_contact_interval`:

1. ``mask = score > 0.7``;
2. ``velocity[t] = |p[t+1] - p[t]| * 30`` (last frame repeats the previous value),
   ``static = velocity < 0.45`` m/s;
3. intervals = segments of ``mask & static``;
4. supplementary intervals = segments of ``~mask & (velocity < 0.15)`` that are
   longer than ``2*3+1`` frames **and** whose position range stays inside the
   10 cm / 5 cm threshold — this is the "score low but slow and stable" case;
5. both sorted/merged, then keep only ``e - s > 7`` frames;
6. split each interval wherever the per-axis position range exceeds the threshold
   (ankle 10 cm, foot 5 cm);
7. shrink both ends by 3 frames **only away from the video boundaries**
   (``s != 0`` / ``e != T`` stay put);
8. keep only ``e - s > 7`` frames again.  Intervals are half-open ``[s, e)``.

Nothing here measures a real floor height (the official step 5 does not either);
the only ground handling in the official pipeline is step 1,
``move_mesh_to_ground`` (``runtime_v2m_postprocess_simple.py:199``): the SMPL-H
mesh's per-frame lowest vertex Y, the minimum over the whole segment, subtracted
from ``trans[:,1]``/``local_joints_positions[:,0,1]``/``pelvis_world[:,1]`` as ONE
global scalar — uniform over the take, so jumps survive.  Our npz is already that
output: its lowest joint Y over the take is -0.0278 m (feet on the floor, a couple
of cm of joint-below-floor).  The plugin's own "source is floating, estimate a
floor Z" path is therefore NOT the official step-1 scalar (that one is a mesh
vertex minimum in the npz frame); it survives only as input to the geometry
fallback detector, which needs a floor plane to threshold against.

Frame mapping follows the project convention: 30 fps, npz index ``i`` is video
frame ``i``, and Blender frame ``f`` shows video frame ``f-1`` →
``blender_frame = npz_index + frame_offset`` with ``frame_offset = 1``.

--------------------------------------------------------------------------
Velocity term: route (B), the absolute joints from ``final.fbx``
--------------------------------------------------------------------------
The official step 5 feeds **FK'd absolute keypoints**.  The npz's
``local_joints_positions`` is *not* absolute: its X/Z are the take's travel
removed (the pelvis stays within a few cm of the origin) while its Y already
carries the absolute height (pelvis 0.892 m) — so a velocity built from it
misses the horizontal travel.  ``position_mode="world"`` patches that with
``pelvis_world + (local - local[pelvis])``, which is an *approximation*.

Route (B) removes the approximation: every SMPL joint is a **bone head** in
the ``Reference`` armature of ``final.fbx`` (52 bones, object scale 0.01, so
the bone heads are metres).  Measured on ``t999_H4_b8abd355e6f2``:

* Blender world is Z-up (the FBX importer converted Y-up → Z-up); the joint
  frame of the npz follows from the object matrix
  ``world = (0.01*x, -0.01*z, +0.01*y)``, so the npz-convention positions are
  ``npz = (-X_w, Z_w, +Y_w)`` — a **180° yaw** away from the raw Blender world
  (both horizontal axes negated; the vertical is Blender's world Z).
  Checking the other sign combinations: ``(+X, +Z, -Y)`` gives 39.7 cm,
  ``(+X, +Z, +Y)`` 34.0 cm, ``(-X, +Z, -Y)`` 16.6 cm, ``(-X, +Z, +Y)``
  **4.75 cm**.  The literal ``(x, -z, y)`` reading of the brief flips the
  height and is rejected by criterion 1 (pelvis −0.897 m).
* criterion 1 — pelvis height mean over the take: **0.8973 m** (npz 0.8921 m).
* criterion 2 — lowest foot joint over the take: **+0.0153 m** (npz −0.0126 m).
* frame correspondence vs the 995-frame official npz: the root-relative
  per-joint error is minimal at **offset 0** (4.75 cm mean) and grows
  monotonically with the offset (off=1 5.29, off=2 6.32, off=5 9.75,
  off=8 12.71, off=12 15.65 cm) → the official 995 frames are the first 995 of
  the fbx's 1000 (the fbx has 5 extra frames at the end, nothing was dropped
  at the start).  Bone order differs from the official ``JOINT_NAMES`` order
  (the fbx uses the SMPL kintree order), so joints are matched **by name**.

``official_planted_ranges(..., joints_path=...)`` uses those joints for the
velocity, keeps the contact scores from the same take's npz, and refuses a
pairing that is not the same take (see :func:`same_take_guard`).  Without a
joints file it falls back to the npz ``position_mode`` and says so — that
fallback is labelled "在猜" in the report, not silently trusted.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

FOOT_CONTACT_KEY = "foot_contacts"
# Official step-5 constants (see the module docstring for the file:line sources).
OFFICIAL_CONTACT_THRESHOLD = 0.7
OFFICIAL_VELOCITY_THRESHOLD = 0.15 * 3      # 0.45 m/s
OFFICIAL_EXTRA_VELOCITY_THRESHOLD = 0.15    # velocity_threshold / 3
OFFICIAL_SHRINK_FRAMES = 3
OFFICIAL_MIN_INTERVAL = 7                   # keep only e - s > 7
OFFICIAL_POSITION_RANGES = {
    "L_Ankle": 0.1, "R_Ankle": 0.1, "L_Foot": 0.05, "R_Foot": 0.05,
}
OFFICIAL_CONTACT_JOINTS = ("L_Ankle", "L_Foot", "R_Ankle", "R_Foot")
# SMPL-H 52-joint ids for those four names (same table as the official JOINT_NAMES).
SMPLH_JOINT_IDS = {"L_Ankle": 7, "L_Foot": 10, "R_Ankle": 8, "R_Foot": 11}
FPS = 30.0

# Deprecated: the previous simplified rule ("score > 0.5, union of the two
# channels").  Kept only so the old/new comparison can be reproduced.
DEPRECATED_CONTACT_THRESHOLD = 0.5
CONTACT_CHANNELS = {"L": (0, 1), "R": (2, 3)}
DEFAULT_FRAME_OFFSET = 1

# --- route (B): absolute joints dumped from final.fbx (see module docstring) ---
FBX_JOINTS_TOLERANCE_FRAMES = 8      # 1000-frame fbx vs 995-frame official npz is normal
JOINTS_POSITIONS_KEY = "positions"
JOINTS_NAMES_KEY = "names"
JOINTS_SOURCE_KEY = "source_fbx"
JOINTS_SIDECAR_SUFFIX = "_fbx_joints.npz"
AXIS_CONVENTION = "npz = (-X_w, Z_w, +Y_w)  [Blender world Z is up; 180 deg yaw]"


class TakeMismatchError(RuntimeError):
    """The joints file and the contact npz are not the same take.

    Raised (never swallowed into the geometry fallback) because pairing two
    takes is a configuration error: the scores would describe one motion and
    the velocity another, and every interval built from them is meaningless.
    """


def take_identity(path: str) -> str:
    """Take token of a FlowHMR artifact path.

    ``...\\video_to_fbx\\t999_H4_b8abd355e6f2\\final.fbx`` -> ``t999``
    ``F:\\flowhmr_official_out\\t999_OFFICIAL_post.npz``      -> ``t999``
    Falls back to the parent directory name when there is no ``t<digits>``
    token, so two files in the same folder still compare equal.
    """

    import os
    import re

    text = str(path or "")
    stem = os.path.splitext(os.path.basename(text))[0]
    for candidate in (stem, os.path.basename(os.path.dirname(text))):
        match = re.match(r"^(t\d+)", candidate, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return os.path.basename(os.path.dirname(text)).lower()


def resolve_joints_path(npz_path: str, explicit: str | None = None) -> str:
    """Explicit path, else the ``<npz stem>_fbx_joints.npz`` sidecar, else ""."""

    import os

    if explicit:
        return str(explicit)
    stem, _ext = os.path.splitext(str(npz_path or ""))
    if not stem:
        return ""
    sidecar = stem + JOINTS_SIDECAR_SUFFIX
    return sidecar if os.path.isfile(sidecar) else ""


def blender_world_to_npz(positions: Any) -> Any:
    """Blender world (Z-up, metres) -> npz joint convention.

    ``npz = (-X_w, Z_w, +Y_w)``.  The object matrix of the imported armature is
    ``world = (0.01*x, -0.01*z, +0.01*y)``, i.e. the FBX's own Y-up frame is
    reached with a 180 deg yaw from the Blender world; both horizontal axes
    therefore flip and the vertical is world Z.  Measured on t999_H4: pelvis
    height mean 0.8973 m (npz 0.8921) and lowest foot joint +0.0153 m
    (npz -0.0126), and the root-relative error is 4.75 cm vs 16.6/34.0/39.7 cm
    for the other sign combinations.
    """

    import numpy as np

    arr = np.asarray(positions, dtype=np.float64)
    return np.stack([-arr[..., 0], arr[..., 2], arr[..., 1]], axis=-1)


def load_fbx_joints(path: str) -> dict[str, Any]:
    """Load the (B) joints file written by ``tools/fbx_joints_dump.py``.

    Returns ``positions`` already in the npz convention (see
    :func:`blender_world_to_npz`), the bone ``names`` and the ``source_fbx``.
    """

    import numpy as np

    if not path:
        raise RuntimeError("未指定 fbx 绝对关节 npz 路径")
    try:
        data = np.load(str(path), allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"读不了 fbx 绝对关节 npz：{exc}") from exc
    if JOINTS_POSITIONS_KEY not in data or JOINTS_NAMES_KEY not in data:
        raise RuntimeError(
            f"{path} 不是 fbx 绝对关节 npz（需要 keys: "
            f"{JOINTS_POSITIONS_KEY}/{JOINTS_NAMES_KEY}；实际 keys: "
            f"{', '.join(sorted(str(k) for k in data.keys()))}）"
        )
    raw = np.asarray(data[JOINTS_POSITIONS_KEY], dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1] < 12 or raw.shape[2] != 3:
        raise RuntimeError(f"{JOINTS_POSITIONS_KEY} 形状应为 (T,52,3)，实际 {tuple(raw.shape)}")
    names = [str(n) for n in data[JOINTS_NAMES_KEY]]
    if len(names) != raw.shape[1]:
        raise RuntimeError(f"names 数 {len(names)} 与 positions 关节数 {raw.shape[1]} 不一致")
    missing = [n for n in OFFICIAL_CONTACT_JOINTS if n not in names]
    if missing:
        raise RuntimeError(f"fbx 关节里缺少接触关节 {missing}（names: {names[:8]}…）")
    return {
        "positions": blender_world_to_npz(raw),
        "raw_positions": raw,
        "names": names,
        "frames": int(raw.shape[0]),
        "source_fbx": str(data[JOINTS_SOURCE_KEY]) if JOINTS_SOURCE_KEY in data else "",
        "path": str(path),
        "axis": AXIS_CONVENTION,
    }


def contact_positions_from_joints(joints: Mapping[str, Any], count: int) -> Any:
    """(T,4,3) positions of the official four joints, matched BY NAME.

    The fbx bone order is the SMPL kintree order, not the official
    ``JOINT_NAMES`` order, so indexing by position would silently swap joints.
    ``count`` is the number of frames the contact scores have; frame ``i`` of
    the joints is the same video frame (offset 0, measured).

    Longer than ``count`` (the normal case: 1000-frame fbx vs 995-frame npz) is
    truncated; shorter is padded by continuing the last per-frame displacement,
    so the tail keeps its velocity instead of looking static.  The shortfall is
    already bounded by :func:`same_take_guard` (<= 8 frames).
    """

    import numpy as np

    names = list(joints["names"])
    cols = [names.index(name) for name in OFFICIAL_CONTACT_JOINTS]
    pos = np.asarray(joints["positions"], dtype=np.float64)[:, cols, :]
    count = int(count)
    if pos.shape[0] >= count:
        return pos[:count]
    missing = count - pos.shape[0]
    if pos.shape[0] < 2:
        raise RuntimeError(f"fbx 绝对关节只有 {pos.shape[0]} 帧，无法补到 {count} 帧")
    step = pos[-1] - pos[-2]
    tail = pos[-1][None, :, :] + step[None, :, :] * np.arange(1, missing + 1)[:, None, None]
    return np.concatenate([pos, tail], axis=0)


def same_take_guard(
    npz_path: str,
    joints: Mapping[str, Any],
    npz_frames: int,
    *,
    tolerance: int = FBX_JOINTS_TOLERANCE_FRAMES,
) -> dict[str, Any]:
    """Refuse to pair a joints file with a different take's contact npz.

    Two checks, both required:

    * **directory identity** — the take token of ``source_fbx`` (the fbx the
      joints were dumped from) must equal the take token of the contact npz;
    * **frame tolerance** — ``|jt - npz| <= tolerance``.  Strict equality would
      reject the normal pairing (the fbx has 1000 frames, the official
      post-processed npz 995 — the last 5 are extra, offset 0 verified), so the
      tolerance is 8 frames.

    Raises :class:`TakeMismatchError` with a message naming both sides.
    """

    jt = int(joints.get("frames") or 0)
    take_j = take_identity(joints.get("source_fbx") or joints.get("path") or "")
    take_n = take_identity(npz_path)
    if take_j != take_n:
        raise TakeMismatchError(
            f"不是同一个 take：fbx 绝对关节来自 take「{take_j}」"
            f"（{joints.get('source_fbx') or joints.get('path')}），"
            f"接触分数 npz 是 take「{take_n}」（{npz_path}）。"
            "跨 take 配对会把 A 片的接触分数和 B 片的速度项混在一起，已拒绝。"
        )
    delta = abs(jt - int(npz_frames))
    if delta > int(tolerance):
        raise TakeMismatchError(
            f"帧数超容差：fbx 绝对关节 {jt} 帧，接触分数 npz {int(npz_frames)} 帧，"
            f"相差 {delta} 帧 > {int(tolerance)} 帧（同一 take 的正常配对是 1000 vs 995，"
            "相差 5 帧）。请确认这两个文件确实来自同一段视频，或把 joints 重新 dump 一遍。"
        )
    return {
        "take": take_n,
        "joints_take": take_j,
        "joints_frames": jt,
        "npz_frames": int(npz_frames),
        "delta_frames": delta,
        "tolerance_frames": int(tolerance),
        "axis": joints.get("axis", AXIS_CONVENTION),
        "source_fbx": joints.get("source_fbx", ""),
    }



def load_contact_matrix(path: str) -> Any:
    """Load ``foot_contacts`` from a FlowHMR npz.

    Raises ``RuntimeError`` with the readable reason (missing file, not an npz,
    missing key, wrong shape) so the caller can fall back and say why.
    """

    import numpy as np

    if not path:
        raise RuntimeError("未指定 FlowHMR npz 路径")
    try:
        data = np.load(str(path), allow_pickle=True)
    except Exception as exc:  # noqa: BLE001 - 读不了就交给回退路径
        raise RuntimeError(f"读不了 npz：{exc}") from exc
    if FOOT_CONTACT_KEY not in data:
        keys = ", ".join(sorted(str(key) for key in data.keys()))
        raise RuntimeError(f"npz 里没有 {FOOT_CONTACT_KEY}（keys: {keys}）")
    contacts = np.asarray(data[FOOT_CONTACT_KEY], dtype=float)
    if contacts.ndim != 2 or contacts.shape[1] < 4:
        raise RuntimeError(f"{FOOT_CONTACT_KEY} 形状应为 (T,4)，实际 {contacts.shape}")
    return contacts


def contact_masks(contacts: Any, threshold: float = DEPRECATED_CONTACT_THRESHOLD) -> dict[str, Any]:
    """Per-foot planted masks: heel OR toe above the threshold."""

    import numpy as np

    return {
        side: (contacts[:, channels[0]] > float(threshold))
        | (contacts[:, channels[1]] > float(threshold))
        for side, channels in CONTACT_CHANNELS.items()
    }


def mask_to_ranges(
    mask: Sequence[bool],
    *,
    frame_offset: int = DEFAULT_FRAME_OFFSET,
    frame_start: int | None = None,
    frame_end: int | None = None,
    min_len: int = 1,
) -> list[list[int]]:
    """Contiguous [start, end] Blender-frame ranges (both ends inclusive)."""

    ranges: list[list[int]] = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            ranges.append([start + 1, index])
            start = None
    if start is not None:
        ranges.append([start + 1, len(mask)])

    out = []
    for begin, end in ranges:
        begin = begin + int(frame_offset) - 1
        end = end + int(frame_offset) - 1
        if frame_start is not None:
            begin = max(begin, int(frame_start))
        if frame_end is not None:
            end = min(end, int(frame_end))
        if end < begin or (end - begin + 1) < max(1, int(min_len)):
            continue
        out.append([int(begin), int(end)])
    return out


def planted_ranges_from_npz(
    path: str,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_offset: int = DEFAULT_FRAME_OFFSET,
    threshold: float = DEPRECATED_CONTACT_THRESHOLD,
    min_len: int = 1,
) -> dict[str, Any]:
    """The wizard's default planted source: ranges straight from the npz.

    ``planted`` keeps the per-side ranges, ``air`` the frames where neither foot
    is planted (the ground_feet step's exemption list), and the counters are
    what the report and the panel line quote.
    """

    contacts = load_contact_matrix(path)
    masks = contact_masks(contacts, threshold)
    planted = {
        side: mask_to_ranges(
            masks[side],
            frame_offset=frame_offset,
            frame_start=frame_start,
            frame_end=frame_end,
            min_len=min_len,
        )
        for side in ("L", "R")
    }
    lo = 0 if frame_start is None else max(0, int(frame_start) - int(frame_offset))
    hi = len(masks["L"]) if frame_end is None else min(
        len(masks["L"]), int(frame_end) - int(frame_offset) + 1
    )
    airborne = [
        not bool(masks["L"][index]) and not bool(masks["R"][index])
        for index in range(lo, max(lo, hi))
    ]
    air = mask_to_ranges(
        airborne,
        frame_offset=int(frame_offset) + lo,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    return {
        "operation": "planted_from_npz",
        "npz": str(path),
        "frames_in_npz": int(len(contacts)),
        "threshold": float(threshold),
        "frame_offset": int(frame_offset),
        "channels": {side: list(channels) for side, channels in CONTACT_CHANNELS.items()},
        "planted": planted,
        "air": air,
        "planted_frames": {side: int(sum(len(r) and (r[1] - r[0] + 1) for r in planted[side]))
                           for side in ("L", "R")},
        "segment_counts": {side: len(planted[side]) for side in ("L", "R")},
        "air_segments": len(air),
    }


# ---------------------------------------------------------------------------
# Official step 5, ported 1:1 (postprocess_utils.py:79-157)
# ---------------------------------------------------------------------------


def _find_segments(mask: Any) -> list[list[int]]:
    """``postprocess_utils.py:79`` — half-open [start, end) runs of True."""

    import numpy as np

    padded = np.concatenate([[0], np.asarray(mask).astype(np.int32), [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [[int(s), int(e)] for s, e in zip(starts, ends)]


def _split_interval_by_position_range(
    positions: Any, start: int, end: int, range_threshold: float
) -> list[tuple[int, int]]:
    """``postprocess_utils.py:86`` — cut where the per-axis range grows too big."""

    import numpy as np

    if end - start <= 1:
        return [(start, end)]
    pos = positions[start:end]
    full_range = pos.max(axis=0) - pos.min(axis=0)
    if np.all(full_range <= range_threshold):
        return [(start, end)]

    sub_intervals = []
    cur_start = start
    while cur_start < end:
        cur_min = positions[cur_start].copy()
        cur_max = positions[cur_start].copy()
        cur_end = cur_start + 1
        while cur_end < end:
            new_min = np.minimum(cur_min, positions[cur_end])
            new_max = np.maximum(cur_max, positions[cur_end])
            if np.any(new_max - new_min > range_threshold):
                break
            cur_min, cur_max = new_min, new_max
            cur_end += 1
        sub_intervals.append((cur_start, cur_end))
        cur_start = cur_end
    return sub_intervals


def find_foot_contact_interval(
    foot_contacts: Any,
    foot_contact_positions: Any,
    joint_names: Sequence[str],
    *,
    threshold: float = OFFICIAL_CONTACT_THRESHOLD,
    velocity_threshold: float = OFFICIAL_VELOCITY_THRESHOLD,
    shrink_frame: int = OFFICIAL_SHRINK_FRAMES,
    position_range_threshold: Any = 0.1,
) -> dict[str, list[tuple[int, int]]]:
    """FlowHMR 官方第 5 步（``postprocess_utils.py:112``）的逐行移植。

    逐条对齐：``mask = score > threshold``（调用方给 0.7）；``velocity =
    |p[t+1]-p[t]| * 30``，最后一帧复制前一帧的值；``static = velocity < 0.45``；
    补充区间 = ``~mask & (velocity < 0.15)`` 且长度 > ``2*3+1`` 且逐轴位移不超过
    ``position_range_threshold``；合并后只留 ``e - s > 7``；按位移拆分；非视频
    边界两端各收缩 3 帧；再过滤 ``e - s > 7``。输出 half-open ``[s, e)``。

    **唯一的实现差异**：官方用 torch，这里用 numpy（Blender 自带 Python 没有
    torch，实测 ``find_spec('torch') is None``），运算逐条等价（``torch.norm`` →
    ``np.linalg.norm``、``torch.cat`` → ``np.concatenate``），输入也兼容 torch
    张量（自动转 numpy）。
    """

    import numpy as np

    contacts = _to_numpy(foot_contacts)
    positions = _to_numpy(foot_contact_positions)
    foot_contact_mask = contacts > threshold
    contact_intervals: dict[str, list[tuple[int, int]]] = {}
    if isinstance(position_range_threshold, dict):
        _range_thresholds = position_range_threshold
    else:
        _range_thresholds = {}
    _default_range_threshold = (
        0.1 if isinstance(position_range_threshold, dict) else float(position_range_threshold)
    )

    assert len(joint_names) == foot_contact_mask.shape[-1]
    for i, jname in enumerate(joint_names):
        jnt_range_threshold = _range_thresholds.get(jname, _default_range_threshold)
        velocity = (
            np.linalg.norm(positions[1:, i, :] - positions[:-1, i, :], axis=-1) * FPS
        )
        velocity = np.concatenate([velocity, velocity[-1:]], axis=0)
        flag_static = velocity < velocity_threshold
        contact_interval = _find_segments(foot_contact_mask[..., i] & flag_static)

        positions_np = positions[:, i, :]

        # Supplementary: no contact detected but velocity low AND position range small
        non_contact_static = (~foot_contact_mask[..., i]) & (velocity < velocity_threshold / 3)
        extra_intervals = _find_segments(non_contact_static)
        extra_intervals = [
            (s, e) for s, e in extra_intervals
            if (e - s > shrink_frame * 2 + 1)
            and np.all(
                positions_np[s:e].max(axis=0) - positions_np[s:e].min(axis=0)
                <= jnt_range_threshold
            )
        ]
        contact_interval = sorted(contact_interval + extra_intervals, key=lambda x: x[0])
        contact_interval = [
            (s, e) for s, e in contact_interval if e - s > shrink_frame * 2 + 1
        ]

        split_intervals: list[tuple[int, int]] = []
        for s, e in contact_interval:
            split_intervals.extend(
                _split_interval_by_position_range(positions_np, s, e, jnt_range_threshold)
            )
        split_intervals_ = []
        for s, e in split_intervals:
            if s != 0:
                s = s + shrink_frame
            if e != positions.shape[0]:
                e = e - shrink_frame
            split_intervals_.append((s, e))
        split_intervals = split_intervals_
        split_intervals = [
            (s, e) for s, e in split_intervals if e - s > shrink_frame * 2 + 1
        ]
        contact_intervals[jname] = split_intervals
    return contact_intervals


def _to_numpy(value: Any) -> Any:
    """Accept torch tensors or numpy/py arrays (official code passes tensors)."""

    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy(), dtype=np.float64)
        except Exception:  # noqa: BLE001 - fall through to np.asarray
            pass
    return np.asarray(value, dtype=np.float64)


def _contact_positions(data: Any, *, position_mode: str) -> Any:
    """(T,4,3) positions for the official four joints.

    The official code feeds FK'd **absolute** keypoints.  We only have the npz's
    ``local_joints_positions`` (root-relative: the pelvis stays within ±6 cm while
    ``pelvis_world``/``trans`` carry the take's ~2 m of travel).  ``world`` adds
    the horizontal root motion back (``pelvis_world + (local - local[pelvis])``,
    whose Y equals the local Y exactly - measured), which is what the official
    keypoints contain; ``local`` keeps the npz's own numbers unchanged.  Both are
    reported so the difference is visible; the default is ``world``.
    """

    import numpy as np

    local = np.asarray(data["local_joints_positions"], dtype=np.float64)
    ids = [SMPLH_JOINT_IDS[name] for name in OFFICIAL_CONTACT_JOINTS]
    if str(position_mode).lower() == "local":
        return local[:, ids, :]
    pelvis_world = np.asarray(data["pelvis_world"], dtype=np.float64)
    return pelvis_world[:, None, :] + (local[:, ids, :] - local[:, [0], :])


def official_planted_ranges(
    path: str,
    *,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_offset: int = DEFAULT_FRAME_OFFSET,
    position_mode: str = "world",
    threshold: float = OFFICIAL_CONTACT_THRESHOLD,
    joints_path: str | None = None,
) -> dict[str, Any]:
    """官方第 5 步区间 → 每只脚的贴地区间（默认来源）。

    官方给的是**每关节**四个区间（L_Ankle/L_Foot/R_Ankle/R_Foot）。wizard 要的是
    每只脚的贴地帧，所以按侧取并集（该脚的踝或脚任一在接触就算这只脚贴地），
    并把相接/重叠的段合成一段；两只脚都不在接触的帧就是腾空区间。

    ``joints_path``（路线 B）指向 ``tools/fbx_joints_dump.py`` 从 ``final.fbx``
    导出的绝对关节 npz：给了就用它算速度项（官方要的就是绝对关键点），接触分数
    仍取本 npz 的 ``foot_contacts``；两个文件必须过 :func:`same_take_guard`
    （同 take + 帧数容差 ≤8）。不给就退回 npz 自身的 ``position_mode``，并在
    ``velocity_note`` 里写明**速度项在猜**。
    """

    import numpy as np

    data = np.load(str(path), allow_pickle=True)
    if FOOT_CONTACT_KEY not in data:
        keys = ", ".join(sorted(str(key) for key in data.keys()))
        raise RuntimeError(f"npz 里没有 {FOOT_CONTACT_KEY}（keys: {keys}）")
    # 路线 B 的绝对关节先解析：新的 take 目录 npz（final/body_postprocess/body_origin）
    # 有 foot_contacts 但**没有** local_joints_positions（2026-10-16 实测），
    # 这时只要给了 fbx 绝对关节（tools/fbx_joints_dump.py），速度项就不需要那份数组——
    # 帧数改用接触分数自身的帧数。两者都没有才报错（老行为不变）。
    resolved = resolve_joints_path(str(path), joints_path)
    has_local = "local_joints_positions" in data
    if not has_local and not resolved:
        keys = ", ".join(sorted(str(key) for key in data.keys()))
        raise RuntimeError(
            f"npz 里没有 local_joints_positions，无法按官方第 5 步算速度（keys: {keys}）"
        )
    contacts = np.asarray(data[FOOT_CONTACT_KEY], dtype=np.float64)
    if contacts.ndim != 2 or contacts.shape[1] < 4:
        raise RuntimeError(f"{FOOT_CONTACT_KEY} 形状应为 (T,4)，实际 {tuple(contacts.shape)}")
    if has_local:
        npz_frames = int(np.asarray(data["local_joints_positions"]).shape[0])
        if int(contacts.shape[0]) < npz_frames:
            raise RuntimeError(
                f"{FOOT_CONTACT_KEY} 只有 {int(contacts.shape[0])} 帧，少于 "
                f"local_joints_positions 的 {npz_frames} 帧"
            )
    else:
        npz_frames = int(contacts.shape[0])

    take_guard: dict[str, Any] | None = None
    joints_meta: dict[str, Any] | None = None
    if resolved:
        joints_meta = load_fbx_joints(resolved)
        take_guard = same_take_guard(str(path), joints_meta, npz_frames)
        positions = contact_positions_from_joints(joints_meta, npz_frames)
        velocity_source = "fbx_absolute"
        velocity_note = (
            f"速度项用了 fbx 绝对关节（路线 B，{take_guard['joints_frames']} 帧，"
            f"take {take_guard['take']}，容差 {take_guard['tolerance_frames']} 帧内，"
            f"轴约定 {take_guard['axis']}）"
            + ("" if has_local else
               "；接触 npz 没有 local_joints_positions，帧数取接触分数自身的"
               f" {npz_frames} 帧")
        )
    else:
        positions = _contact_positions(data, position_mode=position_mode)
        velocity_source = f"npz_{str(position_mode).lower()}_guess"
        velocity_note = (
            f"速度项在猜：没有 fbx 绝对关节，退回 npz {position_mode}"
            "（XZ 是根相对，水平位移是加的近似值；官方要的是绝对关键点）"
        )

    intervals = find_foot_contact_interval(
        contacts,
        positions,
        OFFICIAL_CONTACT_JOINTS,
        threshold=float(threshold),
        position_range_threshold=dict(OFFICIAL_POSITION_RANGES),
    )

    total = int(positions.shape[0])

    def _frames(names: Sequence[str]) -> Any:
        mask = np.zeros(total, dtype=bool)
        for name in names:
            for s, e in intervals.get(name, ()):
                mask[int(s):int(e)] = True
        return mask

    masks = {"L": _frames(("L_Ankle", "L_Foot")), "R": _frames(("R_Ankle", "R_Foot"))}
    planted = {
        side: _mask_runs_to_blender_ranges(masks[side], frame_offset, frame_start, frame_end)
        for side in ("L", "R")
    }
    airborne = ~masks["L"] & ~masks["R"]
    air = _mask_runs_to_blender_ranges(airborne, frame_offset, frame_start, frame_end)

    return {
        "operation": "planted_from_npz_official_step5",
        "npz": str(path),
        "source": "npz_official",
        "frames_in_npz": total,
        "frame_offset": int(frame_offset),
        "position_mode": str(position_mode),
        # 速度项来源：路线 B（fbx 绝对关节）还是 npz 近似（在猜）。
        "velocity_source": velocity_source,
        "velocity_note": velocity_note,
        "velocity_is_guess": bool(velocity_source.endswith("_guess")),
        "joints_path": resolved,
        "joints_frames": int(joints_meta["frames"]) if joints_meta else 0,
        "take_guard": take_guard,
        "axis": take_guard["axis"] if take_guard else "",
        "params": {
            "threshold": float(threshold),
            "velocity_threshold": float(OFFICIAL_VELOCITY_THRESHOLD),
            "extra_velocity_threshold": float(OFFICIAL_EXTRA_VELOCITY_THRESHOLD),
            "shrink_frame": int(OFFICIAL_SHRINK_FRAMES),
            "min_interval_frames": int(OFFICIAL_MIN_INTERVAL),
            "position_range_threshold": dict(OFFICIAL_POSITION_RANGES),
        },
        "contact_joints": list(OFFICIAL_CONTACT_JOINTS),
        "joint_intervals": {name: [list(pair) for pair in intervals.get(name, ())]
                            for name in OFFICIAL_CONTACT_JOINTS},
        "planted": planted,
        "air": air,
        # 窗口内的帧数（和 planted 区间同口径）与整段 take 的帧数都给，免得看错。
        "planted_frames": {
            side: int(sum(r[1] - r[0] + 1 for r in planted[side])) for side in ("L", "R")
        },
        "planted_frames_full": {side: int(masks[side].sum()) for side in ("L", "R")},
        "segment_counts": {side: len(planted[side]) for side in ("L", "R")},
        "air_segments": len(air),
    }


def _mask_runs_to_blender_ranges(
    mask: Any,
    frame_offset: int,
    frame_start: int | None,
    frame_end: int | None,
) -> list[list[int]]:
    """Boolean mask (npz index space) → inclusive Blender-frame ranges."""

    out: list[list[int]] = []
    for start, end in _find_segments(mask):
        begin = start + int(frame_offset)
        last = end - 1 + int(frame_offset)
        if frame_start is not None:
            begin = max(begin, int(frame_start))
        if frame_end is not None:
            last = min(last, int(frame_end))
        if last < begin:
            continue
        out.append([int(begin), int(last)])
    return out
