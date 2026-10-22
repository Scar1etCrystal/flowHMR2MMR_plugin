"""Pose-space plumbing shared by the motion tools.

motion_copy / anticipation / follow_through / overshoot / overlap / time_warp /
foot_lock all follow the same recipe:

    sample_visible()  → per-frame LOCAL rotations (+locations / armature-space
                        matrices) of the bones, as the viewer sees them
    <tool math>       → desired local rotations per frame (pure numpy)
    write_pose()      → rotation-mode-aware delta channels on ONE Combine strip

Facts every tool here relies on (measured on the fixture RIG, 2026-10-03):

- Rotation modes differ per bone.  upper_arm_fk / forearm_fk are Euler XYZ,
  shoulder is Euler YXZ; hand/fingers/spine/neck/head/legs are QUATERNION.
  Anything that only writes ``rotation_quaternion`` silently does nothing on
  the arms - always go through :func:`write_pose`.
- NLA COMBINE semantics: quaternion channels right-multiply
  (visible = lower ⊗ delta → delta = conj(lower) ⊗ desired); Euler and
  location channels ADD (delta = desired - lower); scale multiplies.
- "visible" = base strip + every delta strip already on the rig, i.e. the
  evaluated ``pose_bone.matrix_basis`` after ``frame_set``.  Deltas computed
  against the visible pose are exact even when stacked on other fixes (the
  older tools use the base action → the documented stacking approximation).
  ``agent_ops.reapply`` slides the op's own old strip out of the window
  before re-solving, so a reapply never measures its own previous output.
- Mirror: ``F = Rest_src⁻¹ · S · Rest_dst`` (S = armature-space X reflection,
  front = −Y) maps the source bone's rest frame onto the target's.  The local
  mirror is then ``Basis_dst = F⁻¹ · Basis_src · F`` - exact for any rest
  pose (rest maps to rest), no per-rig axis guessing.

Units: angles in DEGREES in every public metric; quaternions (w,x,y,z);
locations in metres (bone-local); frames are inclusive integer ranges.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import bpy

from . import agent_fx
from .animation import bone_path, preserve_scene_frame, set_scene_frame

# ---------------------------------------------------------------------------
# rotation channels


def rot_mode(pose_bone: Any) -> str:
    """'QUATERNION' | 'AXIS_ANGLE' | Euler order ('XYZ', 'YXZ', ...)."""
    return str(pose_bone.rotation_mode)


def rot_path(bone: str, mode: str) -> str:
    if mode == "QUATERNION":
        return bone_path(bone, "rotation_quaternion")
    if mode == "AXIS_ANGLE":
        return bone_path(bone, "rotation_axis_angle")
    return bone_path(bone, "rotation_euler")


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n[n < 1e-12] = 1.0
    return q / n


def quat_continuous(q: np.ndarray) -> np.ndarray:
    """Flip signs so neighbouring quats share a hemisphere (in place copy)."""
    out = np.array(q, dtype=np.float64, copy=True)
    for i in range(1, len(out)):
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] = -out[i]
    return out


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, vectorised over leading axes (wxyz)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def qconj(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., 1:] = -out[..., 1:]
    return out


def qangle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between two quat series, sign-agnostic."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = np.abs(np.sum(a * b, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """wxyz → rotation vector (axis * angle, radians), shortest arc."""
    q = quat_normalize(q)
    q = np.where(q[..., :1] < 0.0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    ang = 2.0 * np.arctan2(s, q[..., 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(s > 1e-12, ang / np.where(s > 1e-12, s, 1.0), 2.0)
    return v * k[..., None]


def rotvec_to_quat(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    ang = np.linalg.norm(r, axis=-1)
    half = 0.5 * ang
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(ang > 1e-12, np.sin(half) / np.where(ang > 1e-12, ang, 1.0), 0.5)
    return np.concatenate([np.cos(half)[..., None], r * k[..., None]], axis=-1)


def qpow(q: np.ndarray, t) -> np.ndarray:
    """q^t (scale the rotation angle by t, same axis)."""
    return rotvec_to_quat(quat_to_rotvec(q) * np.asarray(t, dtype=np.float64)[..., None]
                          if np.ndim(t) else quat_to_rotvec(q) * float(t))


def slerp(q0: np.ndarray, q1: np.ndarray, t) -> np.ndarray:
    """Vectorised slerp q0→q1 (shortest path); t scalar or broadcastable."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    rel = qmul(qconj(q0), q1)
    t = np.asarray(t, dtype=np.float64)
    if t.ndim == q0.ndim - 1 and t.ndim > 0:
        t = t[..., None]
    rv = quat_to_rotvec(rel) * (t if np.ndim(t) else float(t))
    return quat_normalize(qmul(q0, rotvec_to_quat(rv)))


# ---------------------------------------------------------------------------
# time remapping (float frame times → values)


def resample_quats(q: np.ndarray, frame0: int, times: Sequence[float]) -> np.ndarray:
    """q sampled at integer frames frame0.. ; return slerp at float `times`
    (clamped to the sampled span)."""
    q = quat_continuous(quat_normalize(q))
    t = np.clip(np.asarray(times, dtype=np.float64) - frame0, 0.0, len(q) - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(q) - 1)
    frac = t - i0
    out = slerp(q[i0], q[i1], frac)
    return quat_continuous(out)


def resample_vec(v: np.ndarray, frame0: int, times: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    t = np.clip(np.asarray(times, dtype=np.float64) - frame0, 0.0, len(v) - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(v) - 1)
    frac = (t - i0)
    if v.ndim > 1:
        frac = frac[:, None]
    return v[i0] * (1.0 - frac) + v[i1] * frac


# ---------------------------------------------------------------------------
# sampling


def resolve_pose_bones(armature: Any, bones: Iterable[str]) -> list[str]:
    names = []
    for b in bones:
        if armature.pose.bones.get(b) is None:
            raise RuntimeError(f"scope 越界：骨骼不存在于该骨架 {b!r}")
        if b not in names:
            names.append(b)
    return names


def sample_visible(scene: Any, armature: Any, bones: Sequence[str],
                   frames: Sequence[int], *, world: bool = False,
                   extra_fn=None) -> dict:
    """ONE frame sweep for all bones; scene frame is restored afterwards.

    Returns {"frames": (T,) int,
             "quat":  {bone: (T,4) local basis rotation, sign-continuous},
             "loc":   {bone: (T,3) local basis location (m)},
             "euler": {bone: (T,3) evaluated rotation_euler (rad)} - Euler
                      bones only (needed as `compat` reference when writing),
             "mode":  {bone: rotation mode},
             "mat":   {bone: (T,4,4) armature-space pose matrix}  (world=True),
             "extra": [extra_fn(armature, scene) per frame] (if given)}
    """
    bones = resolve_pose_bones(armature, bones)
    frames = [int(f) for f in frames]
    T = len(frames)
    out = {"frames": np.asarray(frames, dtype=np.int64),
           "quat": {b: np.zeros((T, 4)) for b in bones},
           "loc": {b: np.zeros((T, 3)) for b in bones},
           "euler": {}, "mode": {}, "mat": {}, "extra": []}
    pbs = {b: armature.pose.bones[b] for b in bones}
    for b, pb in pbs.items():
        out["mode"][b] = rot_mode(pb)
        if out["mode"][b] not in ("QUATERNION", "AXIS_ANGLE"):
            out["euler"][b] = np.zeros((T, 3))
        if world:
            out["mat"][b] = np.zeros((T, 4, 4))
    with preserve_scene_frame(scene):
        for i, f in enumerate(frames):
            set_scene_frame(scene, f)
            for b, pb in pbs.items():
                mb = pb.matrix_basis
                q = mb.to_quaternion()
                out["quat"][b][i] = (q.w, q.x, q.y, q.z)
                t = mb.translation
                out["loc"][b][i] = (t.x, t.y, t.z)
                if b in out["euler"]:
                    e = pb.rotation_euler
                    out["euler"][b][i] = (e.x, e.y, e.z)
                if world:
                    out["mat"][b][i] = np.asarray(pb.matrix)
            if extra_fn is not None:
                out["extra"].append(extra_fn(armature, scene))
    for b in bones:
        out["quat"][b] = quat_continuous(quat_normalize(out["quat"][b]))
    return out


# ---------------------------------------------------------------------------
# writing


def _euler_from_quats(quats: np.ndarray, order: str,
                      compat: np.ndarray) -> np.ndarray:
    from mathutils import Euler, Quaternion
    out = np.zeros((len(quats), 3))
    for i, q in enumerate(quats):
        ref = Euler(tuple(float(v) for v in compat[i]), order)
        e = Quaternion(tuple(float(v) for v in q)).to_euler(order, ref)
        out[i] = (e.x, e.y, e.z)
    return out


def pose_deltas(armature: Any, sample: Mapping[str, Any], *,
                desired_quat: Mapping[str, np.ndarray] | None = None,
                desired_loc: Mapping[str, np.ndarray] | None = None,
                strength: float = 1.0,
                index_slice: slice | None = None) -> tuple[dict, dict, dict]:
    """desired local poses → (scalars, quats, info) ready for _write_strip.

    ``sample`` is the visible sample the desired poses were derived from
    (its quat/euler/loc are the "lower" values the Combine strip sits on).
    ``index_slice`` picks the written window out of a wider sample.
    """
    sl = index_slice or slice(None)
    scalars: dict = {}
    quats: dict = {}
    info: dict = {}
    k = float(strength)
    for bone, des in (desired_quat or {}).items():
        mode = sample["mode"][bone]
        cur = sample["quat"][bone][sl]
        des = quat_continuous(quat_normalize(des))
        if len(des) != len(cur):
            raise RuntimeError(f"{bone}: desired 帧数 {len(des)} ≠ 采样 {len(cur)}")
        ang = qangle_deg(cur, des)
        if mode == "QUATERNION":
            dq = agent_fx.delta_quat(des, cur)
            if k != 1.0:
                dq = quat_continuous(rotvec_to_quat(quat_to_rotvec(dq) * k))
            quats[rot_path(bone, mode)] = dq
        elif mode == "AXIS_ANGLE":
            raise RuntimeError(f"{bone} 是 AXIS_ANGLE 旋转，暂不支持")
        else:
            cur_e = sample["euler"][bone][sl]
            des_e = _euler_from_quats(des, mode, cur_e)
            d = (des_e - cur_e) * k
            path = rot_path(bone, mode)
            for c in range(3):
                scalars[(path, c)] = d[:, c]
        info.setdefault(bone, {})["rot_mode"] = mode
        info[bone]["rot_change_max_deg"] = round(float(ang.max()), 2) if len(ang) else 0.0
        info[bone]["rot_change_mean_deg"] = round(float(ang.mean()), 2) if len(ang) else 0.0
    for bone, des in (desired_loc or {}).items():
        cur = sample["loc"][bone][sl]
        des = np.asarray(des, dtype=np.float64)
        d = (des - cur) * k
        path = bone_path(bone, "location")
        for c in range(3):
            scalars[(path, c)] = d[:, c]
        info.setdefault(bone, {})["loc_change_max_m"] = round(
            float(np.linalg.norm(des - cur, axis=1).max()), 4) if len(cur) else 0.0
    return scalars, quats, info


def write_pose(armature: Any, name: str, frame_start: int, scalars: dict,
               quats: dict, *, blend: int = 4, track_name: str | None = None):
    """One Combine delta strip (thin wrapper so tools never call bpy NLA)."""
    from . import agent_ops
    return agent_ops._write_strip(armature, name, int(frame_start),
                                  scalars=scalars, quats=quats,
                                  blend=int(blend), track_name=track_name)


# ---------------------------------------------------------------------------
# hierarchy / mirror


def mirror_name(name: str) -> str | None:
    """'hand_fk.L' ↔ 'hand_fk.R' (also _L/_R, .l/.r); None if unsided."""
    for a, b in ((".L", ".R"), (".R", ".L"), ("_L", "_R"), ("_R", "_L"),
                 (".l", ".r"), (".r", ".l")):
        if name.endswith(a):
            return name[: -len(a)] + b
    for a, b in ((".L.", ".R."), (".R.", ".L.")):
        if a in name:
            return name.replace(a, b)
    return None


_S = np.diag([-1.0, 1.0, 1.0])     # armature-space X reflection (front = −Y)


def rest_rot(armature: Any, bone: str) -> np.ndarray:
    m = armature.data.bones[bone].matrix_local
    return np.asarray(m.to_3x3(), dtype=np.float64)


def mirror_flip(armature: Any, src: str, dst: str) -> np.ndarray:
    """3×3 F = Rest_src⁻¹ · S · Rest_dst (det −1).  Symmetric pair → ≈diag(-1,1,1)."""
    rs = rest_rot(armature, src)
    rd = rest_rot(armature, dst)
    return rs.T @ _S @ rd


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrices (...,3,3) → wxyz (robust branch per element)."""
    m = np.asarray(m, dtype=np.float64)
    flat = m.reshape(-1, 3, 3)
    out = np.zeros((len(flat), 4))
    for i, r in enumerate(flat):
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            out[i] = (0.25 * s, (r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            out[i] = ((r[2, 1] - r[1, 2]) / s, 0.25 * s,
                      (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            out[i] = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                      0.25 * s, (r[1, 2] + r[2, 1]) / s)
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            out[i] = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                      (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    return quat_normalize(out.reshape(m.shape[:-2] + (4,)))


def mirror_local(q: np.ndarray, loc: np.ndarray | None, F: np.ndarray):
    """Local basis mirror: R' = F⁻¹ R F (F orthogonal, det −1), t' = F⁻¹ t."""
    R = quat_to_mat(q)
    Ft = F.T
    R2 = Ft @ R @ F
    q2 = quat_continuous(mat_to_quat(R2))
    t2 = None if loc is None else (np.asarray(loc) @ F)     # (F^T t)^T = t^T F
    return q2, t2


_MECH_PREFIXES = ("MCH-", "DEF-", "ORG-", "tweak_", "VIS_", "WGT-")
# Rigify FK spine controls are siblings under MCH-spine.* (each MCH follows
# the control below it), so the bone tree alone doesn't order them.
_SPINE_ORDER = ("torso_root", "torso", "spine_fk", "spine_fk.001",
                "spine_fk.002", "spine_fk.003", "neck", "head")


def semantic_parent(armature: Any, bone: str) -> str | None:
    """Nearest CONTROL-bone ancestor, seeing through Rigify plumbing.

    The RIG's FK controls are not parented to each other directly:
    f_index.01.L ← MCH-f_index.01_drv.L ← ORG-hand.L (COPY_TRANSFORMS
    hand_fk.L); upper_arm_fk.L ← MCH-upper_arm_parent.L ← ORG-shoulder.L
    (copies shoulder.L).  Walk up, skip MCH/DEF/tweak bones, and resolve an
    ORG bone to the control its first COPY_TRANSFORMS follows.
    """
    if bone in _SPINE_ORDER:
        i = _SPINE_ORDER.index(bone)
        for prev in reversed(_SPINE_ORDER[:i]):
            if prev in armature.data.bones:
                return prev
        return None
    b = armature.data.bones[bone].parent
    while b is not None:
        name = b.name
        if name.startswith("ORG-"):
            pb = armature.pose.bones.get(name)
            for c in (pb.constraints if pb is not None else ()):
                if c.type == "COPY_TRANSFORMS":
                    tgt = getattr(c, "subtarget", "")
                    if tgt and not tgt.startswith(_MECH_PREFIXES) \
                            and tgt in armature.data.bones:
                        return tgt
                    break
        elif not name.startswith(_MECH_PREFIXES):
            return name
        b = b.parent
    return None


def semantic_ancestors(armature: Any, bone: str) -> list[str]:
    out, seen = [], {bone}
    p = semantic_parent(armature, bone)
    while p is not None and p not in seen:
        out.append(p)
        seen.add(p)
        p = semantic_parent(armature, p)
    return out


def depth_in(armature: Any, bone: str, members: Sequence[str]) -> int:
    """Chain depth inside `members`: number of members that are semantic
    ancestors of `bone` (0 = chain root).  Finger segments hanging off the
    hand get hand_depth+k, siblings share a depth."""
    mem = set(members)
    return sum(1 for a in semantic_ancestors(armature, bone) if a in mem)


def ancestors(armature: Any, bone: str) -> list[str]:
    out = []
    b = armature.data.bones[bone].parent
    while b is not None:
        out.append(b.name)
        b = b.parent
    return out


def sort_parent_first(armature: Any, bones: Sequence[str]) -> list[str]:
    return sorted(bones, key=lambda b: len(ancestors(armature, b)))


# Named chains (root → tip order).  Finger chains hang off the hand: depth is
# computed from the hierarchy, so branches get the same lag as siblings.
_FINGER = ("f_index", "f_middle", "f_ring", "f_pinky", "thumb")


def chain_preset(name: str, armature: Any | None = None) -> list[str]:
    n = str(name).strip()
    side = n[-1].upper() if n[-2:-1] in (".", "_") else ""
    key = n[:-2].lower() if side else n.lower()
    if key == "spine":
        bones = ["torso_root", "spine_fk", "spine_fk.001", "spine_fk.002",
                 "spine_fk.003", "neck", "head"]
    elif key == "spine_head":
        bones = ["spine_fk", "spine_fk.001", "spine_fk.003", "neck", "head"]
    elif key == "arm" and side:
        bones = [f"shoulder.{side}", f"upper_arm_fk.{side}",
                 f"forearm_fk.{side}", f"hand_fk.{side}"]
        bones += [f"{f}.0{i}.{side}" for f in _FINGER for i in (1, 2, 3)]
    elif key == "arm_nofingers" and side:
        bones = [f"shoulder.{side}", f"upper_arm_fk.{side}",
                 f"forearm_fk.{side}", f"hand_fk.{side}"]
    elif key == "fingers" and side:
        bones = [f"{f}.0{i}.{side}" for f in _FINGER for i in (1, 2, 3)]
    elif key == "leg" and side:
        bones = [f"thigh_fk.{side}", f"shin_fk.{side}", f"foot_fk.{side}",
                 f"toe_fk.{side}"]
    else:
        raise RuntimeError(
            f"未知骨链 {name!r}；可用 spine / spine_head / arm.L / arm.R / "
            "arm_nofingers.L / fingers.L / leg.L（.R 同理），或直接给骨名列表")
    if armature is not None:
        bones = [b for b in bones if armature.pose.bones.get(b) is not None]
    return bones


def animated_bones(action: Any) -> set[str]:
    out = set()
    if action is None:
        return out
    for fc in action.fcurves:
        p = fc.data_path
        if p.startswith('pose.bones["'):
            out.add(p.split('"')[1])
    return out


# ---------------------------------------------------------------------------
# motion analysis (pure numpy) - shared by analyze_motion and the
# anticipation / follow_through / overshoot solvers


def angular_speed_deg(q: np.ndarray) -> np.ndarray:
    """(T,) deg/frame between consecutive frames; [0] copies [1]."""
    q = quat_normalize(q)
    sp = np.zeros(len(q))
    if len(q) > 1:
        sp[1:] = qangle_deg(q[1:], q[:-1])
        sp[0] = sp[1]
    return sp


def smooth(x: np.ndarray, width: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    w = max(1, int(width))
    if w <= 1 or len(x) < 3:
        return x.copy()
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, (pad, w - 1 - pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def detect_events(speed: np.ndarray, *, onset_frac: float = 0.15,
                  stop_frac: float = 0.12, peak_index: int | None = None) -> dict:
    """Main-channel motion events on a speed curve (index space).

    peak  = argmax (or `peak_index`)
    onset = last index BEFORE the peak where speed ≤ onset_frac·peak
            ("speed starts rising steeply" = leaving the still floor)
    stop  = first index AFTER the peak where speed ≤ stop_frac·peak
            ("velocity returns to ~zero")
    Returns indices (None when not found inside the window).
    """
    s = np.asarray(speed, dtype=np.float64)
    if len(s) == 0:
        return {"peak": None, "onset": None, "stop": None, "peak_speed": 0.0}
    p = int(np.argmax(s)) if peak_index is None else int(peak_index)
    pk = float(s[p])
    onset = None
    thr_on = onset_frac * pk
    for i in range(p, -1, -1):
        if s[i] <= thr_on:
            onset = i
            break
    stop = None
    thr_st = stop_frac * pk
    for i in range(p, len(s)):
        if s[i] <= thr_st:
            stop = i
            break
    return {"peak": p, "onset": onset, "stop": stop, "peak_speed": pk}


def motion_axis(q_from: np.ndarray, q_to: np.ndarray):
    """Local rotation axis (unit, in the bone's basis frame) + angle (deg)
    taking q_from → q_to, i.e. rel = conj(q_from) ⊗ q_to."""
    rel = qmul(qconj(quat_normalize(q_from)), quat_normalize(q_to))
    rv = quat_to_rotvec(rel)
    ang = float(np.linalg.norm(rv))
    if ang < 1e-9:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return rv / ang, float(np.degrees(ang))


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# 只在调用链内部用的形参：agent 不该传，也不算"可用参数"
_INTERNAL_ARGS = frozenset({"ctx", "scene", "armature", "bones", "chain", "data_dir",
                            "record", "track_name", "baseline_tracks"})


def reject_unknown_args(tool: str, fn, args: Mapping[str, Any], extra_allowed=(),
                        internal=None) -> None:
    """拼错的参数直接报错，并给出最接近的合法参数名（§10-7）。

    以前这些工具只在 warnings / metrics.ignored_args 里提一句，然后按默认值照常执行
    ——能力弱的模型常常不看 warnings，拼错 amount 就静默变成默认值。合法参数取自
    fn 的签名（去掉内部形参），再加 extra_allowed（壳层自己吃掉的参数）。"""
    import difflib
    import inspect
    sig = inspect.signature(fn)
    allowed = {n for n, p in sig.parameters.items()
               if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
               and not n.startswith("_")} - \
        (_INTERNAL_ARGS if internal is None else frozenset(internal))
    allowed |= set(extra_allowed)
    bad = sorted(k for k in args if k not in allowed)
    if not bad:
        return
    hints = []
    for k in bad:
        close = difflib.get_close_matches(k, sorted(allowed), n=1, cutoff=0.6)
        if close:
            hints.append(f"{k} → {close[0]}？")
    raise RuntimeError(f"{tool} 不认识参数 {bad}"
                       + (f"（{'；'.join(hints)}）" if hints else "")
                       + f"；可用：{', '.join(sorted(allowed))}")


def strip_window(frame_range: Sequence[int]) -> tuple[int, int, list[int]]:
    a, b = int(frame_range[0]), int(frame_range[1])
    if b - a + 1 < 2:
        raise RuntimeError("帧范围至少要 2 帧")
    return a, b, list(range(a, b + 1))
