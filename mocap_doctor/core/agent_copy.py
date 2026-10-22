"""agent_copy.py — motion_copy（把一段可见动作复制到另一时间段 / 另一侧 /
另一部位）+ compare_motion（两段动作逐帧比对，复制结果的验收工具，只读）。

motion_copy 的数学（全在"可见姿态"上：先把源窗、目标窗都采样完，再算，
最后写一条 Combine delta strip——源窗与目标窗在同骨上重叠也不会读到自己）：

  时间映射  目标帧 f' ∈ [c,d] → 源时间 τ = a + (f'−c)·(b−a)/(d−c)。等长就是
            整数平移；不等长 = 线性时间缩放（slerp / 线性插值重采样）。
  目标骨    bone_map 优先；否则 mirror=true 用镜像名（无左右的骨映射到自己）；
            否则同名。
  local     复制关节局部旋转（"同样的关节角"，随身体朝向走）：
              replace  Q(f') = src'(τ)
              add      Q(f') = dst(f') ⊗ conj(src'(a)) ⊗ src'(τ)
            mirror 时 src' = mirror_local(src, F)，F = Rest_s⁻¹·S·Rest_d——由
            "世界镜像 + 父骨同样镜像"严格推出，rest→rest 精确，不猜局部轴。
  world     复制骨架空间朝向（镜像面 X=0，前方 −Y）：
              R_des = S·R_src(τ)·F（镜像）或 R_src(τ)·Rest_s⁻¹·Rest_d（不镜像；
              同骨即 R_src）；add：R_des = [M(τ)·M(a)⁻¹]·R_cur（世界增量左乘）。
            再换算成目标骨局部：B = rel_rest⁻¹ · P_new⁻¹ · R_des，
              rel_rest = Rest_parent⁻¹·Rest_bone（真实父骨 pb.parent，可能是 MCH）；
              P_new = ΔR_anc · P_cur，anc = 复制集里最近的祖先（先查语义链——
              手指/上臂的真实父链是 MCH/ORG，经 COPY_TRANSFORMS 跟随 hand_fk /
              shoulder，真实骨树里根本没有它们——再查真实骨树），
              ΔR_anc = R_des_anc·R_cur_anc⁻¹（刚性传播）；没有则 P_new = P_cur。
            只换算旋转（3×3）。各骨 R_des 先全部算好，换算互不依赖，所以
            结果与处理顺序无关（等价于 sort_parent_first 逐骨处理）。
            不继承旋转的 hinge 骨：P_new = Rest_parent。
  world_abs 复制绝对骨架空间朝向：不镜像 R_des = R_src(τ)，镜像 R_des = S·R_src(τ)·S。
             同一骨架内，两骨实际朝向相同；各自位置仍由骨骼层级决定，不复制世界位置。
             与 world 的区别：不再带 Rest_s⁻¹·Rest_d（各骨保留自己的 rest 的转换）。
             世界镜像需左右各乘 S，保证结果 det=+1，是旋转而不是反射矩阵。
  rot+loc   位置一律按局部通道复制（镜像 t' = Fᵀt；add 加 src(τ)−src(a)）。

写入走 agent_pose.pose_deltas + write_pose（Euler 骨写 Euler 通道）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import bpy

from . import agent_ops, agent_pose as P

_S = np.diag([-1.0, 1.0, 1.0])       # 骨架空间 X 反射（前方 −Y，镜像面 X=0）
_SPACES = ("local", "world", "world_abs")
_CHANNELS = ("rot", "rot+loc")
_MODES = ("replace", "add")
# 基础设施可能塞进 args 的键（并发租约 / 归属）：不算拼写错误
_INFRA_KEYS = frozenset({"agent_id", "owner", "lease", "lease_id",
                         "request_id", "expect_version"})
_COPY_ARGS = ("bones|chain", "src_range", "dst_start|dst_range", "time_scale",
              "repeat", "mirror", "bone_map", "space", "channels", "mode",
              "strength", "blend", "dry_run")


# ---------------------------------------------------------------------------
# helpers


def _reject_unknown(tool: str, unknown: Mapping[str, Any], allowed) -> None:
    bad = sorted(k for k in unknown if k not in _INFRA_KEYS)
    if bad:
        raise RuntimeError(f"{tool} 不认识参数 {bad}；可用：{', '.join(allowed)}")


def _choice(name: str, value: Any, options: Sequence[str]) -> str:
    v = str(value).strip().lower()
    if v not in options:
        raise RuntimeError(f"{name}={value!r} 无效，可选 {list(options)}")
    return v


def _rot3(m: np.ndarray) -> np.ndarray:
    """(…,4,4)|(…,3,3) → 最近的纯旋转（SVD 极分解去掉缩放）。"""
    m = np.asarray(m, dtype=np.float64)[..., :3, :3]
    u, _s, vt = np.linalg.svd(m)
    sign = np.sign(np.linalg.det(u @ vt))
    u[..., :, -1] *= sign[..., None]
    return u @ vt


def _tr(m: np.ndarray) -> np.ndarray:
    return np.swapaxes(m, -1, -2)


def _quats_of(r: np.ndarray) -> np.ndarray:
    return P.quat_continuous(P.mat_to_quat(r))


def _resample_rot(r: np.ndarray, frame0: int, times) -> np.ndarray:
    return P.quat_to_mat(P.resample_quats(_quats_of(r), frame0, times))


def _rng(value: Any, name: str) -> list[int]:
    try:
        a, b = int(value[0]), int(value[1])
    except Exception:  # noqa: BLE001
        raise RuntimeError(f"{name} 需要 [起, 止]（含两端），收到 {value!r}") from None
    return [a, b]


def _window(src_range, dst_start=None, dst_range=None, time_scale=None):
    """→ (a, b, c, d, time_scale)。dst_range 优先（时间缩放）；否则
    dst_start + 源长度×time_scale（默认 1 = 等长平移）。"""
    if src_range is None:
        raise RuntimeError("需要 src_range=[a,b]（源帧段，含两端）")
    a, b = _rng(src_range, "src_range")
    if b - a < 1:
        raise RuntimeError(f"src_range {[a, b]} 至少 2 帧（b > a）")
    if dst_range is not None:
        c, d = _rng(dst_range, "dst_range")
        if dst_start is not None and int(dst_start) != c:
            raise RuntimeError(
                f"dst_start={dst_start} 与 dst_range[0]={c} 矛盾；"
                "等长平移只给 dst_start，时间缩放只给 dst_range")
    elif dst_start is not None:
        c = int(dst_start)
        ts = 1.0 if time_scale is None else float(time_scale)
        if not 0.05 <= ts <= 20.0:
            raise RuntimeError(f"time_scale={ts} 超出 0.05–20")
        d = c + int(round((b - a) * ts))
    else:
        raise RuntimeError("需要目标位置：dst_start（等长平移）或 "
                           "dst_range=[c,d]（线性时间缩放到该窗）")
    if d - c < 1:
        raise RuntimeError(f"目标窗 {[c, d]} 至少 2 帧")
    return a, b, c, d, (d - c) / float(b - a)


def _pairs(armature: Any, src: Sequence[str], mirror: bool,
           bone_map: Mapping[str, str] | None) -> list[tuple[str, str]]:
    """源骨 → 目标骨：bone_map 优先；mirror 用镜像名（无左右 → 自己）；否则同名。"""
    bmap = {str(k): str(v) for k, v in (bone_map or {}).items()}
    extra = [k for k in bmap if k not in src]
    if extra:
        raise RuntimeError(f"bone_map 的源骨 {extra} 不在 bones/chain 里；"
                           "把它们加进 bones，或省略 bones 只给 bone_map")
    pairs, owner = [], {}
    for s in src:
        d = bmap.get(s) or ((P.mirror_name(s) or s) if mirror else s)
        if armature.pose.bones.get(d) is None:
            how = "bone_map" if s in bmap else ("镜像名" if mirror else "同名")
            raise RuntimeError(f"目标骨 {d!r} 不存在（{s} 的{how}映射）；"
                               "用 bone_map={源骨: 目标骨} 指定")
        if d in owner:
            raise RuntimeError(f"{owner[d]} 和 {s} 都映射到 {d}；一个目标骨只能有一个源")
        owner[d] = s
        pairs.append((s, d))
    return pairs


def _member_ancestor(armature: Any, bone: str, members) -> str | None:
    """复制集里离 bone 最近的祖先：语义链优先（穿过 MCH/ORG 中转），
    再查真实骨树。"""
    for anc in P.semantic_ancestors(armature, bone):
        if anc in members:
            return anc
    for anc in P.ancestors(armature, bone):
        if anc in members:
            return anc
    return None


# Rigify 肢体：FK 控制骨 → 带 IK_FK 开关的 parent 控制骨（0 = IK，1 = FK）
_LIMB_SWITCH = {"upper_arm_fk": "upper_arm_parent", "forearm_fk": "upper_arm_parent",
                "hand_fk": "upper_arm_parent", "thigh_fk": "thigh_parent",
                "shin_fk": "thigh_parent", "foot_fk": "thigh_parent",
                "toe_fk": "thigh_parent"}


def _ik_hidden(armature: Any, bones: Sequence[str]) -> list[str]:
    """目标骨里处在 IK 模式肢体上的 FK 骨（写了也看不见）。"""
    out = []
    for bn in bones:
        stem, _dot, side = bn.rpartition(".")
        sw = armature.pose.bones.get(f"{_LIMB_SWITCH.get(stem, '')}.{side}")
        try:
            if sw is not None and "IK_FK" in sw.keys() and float(sw["IK_FK"]) < 0.5:
                out.append(bn)
        except Exception:  # noqa: BLE001 - a note, never a failure
            pass
    return out


def _frame_map(armature: Any, s: str, d: str, mirror: bool) -> np.ndarray:
    """源骨 rest 系 → 目标骨 rest 系：镜像 F = Rest_s⁻¹·S·Rest_d（det −1），
    否则 G = Rest_s⁻¹·Rest_d（同骨 = I）。"""
    if mirror:
        return P.mirror_flip(armature, s, d)
    return P.rest_rot(armature, s).T @ P.rest_rot(armature, d)


def _orientation_map(armature: Any, s: str, d: str, mirror: bool,
                     space: str) -> np.ndarray:
    """Frame mapping shared by writing and verification.

    world keeps each target's rest frame. world_abs copies absolute armature
    orientation; reflecting an orientation uses S @ R @ S (det +1), not S @ R.
    """
    if space == "world_abs":
        return _S if mirror else np.eye(3)
    return _frame_map(armature, s, d, mirror)


def _solve_world(armature, pairs, smp_s, smp_d, parents, mirror, tau, a, mode,
                 space="world"):
    """骨架空间目标朝向 → 目标骨局部旋转（见模块文档）。"""
    T = len(tau)
    members = {d for _s, d in pairs}
    r_cur = {d: _rot3(smp_d["mat"][d]) for d in members}
    r_des = {}
    for s, d in pairs:
        r_src = _rot3(smp_s["mat"][s])
        rt = _resample_rot(r_src, a, tau)
        fm = _orientation_map(armature, s, d, mirror, space)
        if mirror:
            m, m0 = _S @ rt @ fm, _S @ r_src[0] @ fm
        else:
            m, m0 = rt @ fm, r_src[0] @ fm
        if mode == "add":
            m = m @ m0.T @ r_cur[d]          # 世界增量 M(τ)·M(a)⁻¹ 左乘到当前
        r_des[d] = m
    out, anc_of = {}, {}
    for _s, d in pairs:
        rest = P.rest_rot(armature, d)
        par = parents.get(d)
        if par is None:
            rel = rest
            p_new = np.broadcast_to(np.eye(3), (T, 3, 3))
        elif not armature.data.bones[d].use_inherit_rotation:
            # hinge：Blender 用父骨的 rest 朝向代替其姿态 → R = Rest_bone·B
            rel = P.rest_rot(armature, par).T @ rest
            p_new = np.broadcast_to(P.rest_rot(armature, par), (T, 3, 3))
        else:
            rel = P.rest_rot(armature, par).T @ rest
            p_cur = _rot3(smp_d["mat"][par])
            anc = _member_ancestor(armature, d, members)
            anc_of[d] = anc
            p_new = (r_des[anc] @ _tr(r_cur[anc]) @ p_cur) if anc else p_cur
        out[d] = _quats_of(rel.T @ _tr(p_new) @ r_des[d])
    return out, anc_of


# ---------------------------------------------------------------------------
# motion_copy


def motion_copy(scene: Any, armature: Any, *, bones: Sequence[str] | None = None,
                src_range: Sequence[int] | None = None,
                dst_start: int | None = None,
                dst_range: Sequence[int] | None = None,
                time_scale: float | None = None,
                repeat: bool = False,
                frame_range: Sequence[int] | None = None,
                mirror: bool = False,
                bone_map: Mapping[str, str] | None = None,
                space: str = "local", channels: str = "rot",
                mode: str = "replace", strength: float = 1.0, blend: int = 4,
                op_mode: str = "preview", data_dir=None,
                track_name: str | None = None, dry_run: bool = False,
                record: bool = True, **unknown) -> dict:
    """把 bones 在 src_range 的可见动作复制到目标骨 × 目标窗（见模块文档）。

    bones 是骨名（桥层已把角色名 / chain 解析好）；省略 bones 时取
    bone_map 的键。frame_range 是 dst_range 的别名（通用"写窗"约定）。
    """
    _reject_unknown("motion_copy", unknown, _COPY_ARGS)
    space = _choice("space", space or "local", _SPACES)
    channels = _choice("channels", channels or "rot", _CHANNELS)
    mode = _choice("mode", mode or "replace", _MODES)
    if frame_range is not None:
        if dst_range is not None and _rng(dst_range, "dst_range") \
                != _rng(frame_range, "frame_range"):
            raise RuntimeError("frame_range 与 dst_range 矛盾；只给 dst_range")
        dst_range = frame_range
    mirror = bool(mirror)
    bone_map = {str(k): str(v) for k, v in (bone_map or {}).items()} or None
    src = list(bones or []) or list(bone_map or {})
    if not src:
        raise RuntimeError("没有要复制的骨：给 bones（角色名或骨名）/ chain，或 bone_map")
    src = P.resolve_pose_bones(armature, src)
    if bone_map:
        P.resolve_pose_bones(armature, list(bone_map.values()))
    pairs = _pairs(armature, src, mirror, bone_map)
    a, b, c, d, ts = _window(src_range, dst_start, dst_range, time_scale)
    repeat = bool(repeat)
    if repeat and time_scale is not None:
        raise RuntimeError("repeat 与 time_scale 二选一：重复平铺不做线性缩放")
    blend = max(0, int(blend))
    strength = float(strength)
    dst = [dn for _s, dn in pairs]
    if repeat:
        # 把源段 [a,b] 按**原速**循环铺满 [c,d]。最后一轮铺不满就停在 d（截断），
        # 窗口外一帧不写 → 后面仍是原动画。段内不加 taper：taper 会在每一轮接缝
        # 处淡回原姿态、把循环打断（所以 repeat 建议 blend=0）。
        seg = b - a + 1
        tau = a + np.mod(np.arange(d - c + 1, dtype=np.float64), float(seg))
        ts = 1.0
    else:
        tau = a + (np.arange(c, d + 1, dtype=np.float64) - c) * (b - a) / float(d - c)
    world = space in ("world", "world_abs")

    # ---- 先采样（源窗 + 目标窗），后面才写 ----
    smp_s = P.sample_visible(scene, armature, src, list(range(a, b + 1)),
                             world=world)
    parents: dict = {}
    if world:
        for dn in dst:
            par = armature.pose.bones[dn].parent
            parents[dn] = par.name if par is not None else None
    extra = [p for p in dict.fromkeys(parents.values()) if p and p not in dst]
    smp_d = P.sample_visible(scene, armature, dst + extra,
                             list(range(c, d + 1)), world=world)

    des_q: dict = {}
    des_l: dict = {}
    anc_of: dict = {}
    if world:
        des_q, anc_of = _solve_world(armature, pairs, smp_s, smp_d, parents,
                                     mirror, tau, a, mode, space=space)
    else:
        for s, dn in pairs:
            q = smp_s["quat"][s]
            if mirror:
                q, _ = P.mirror_local(q, None, P.mirror_flip(armature, s, dn))
            qs = P.resample_quats(q, a, tau)
            if mode == "add":
                rel = P.qmul(P.qconj(q[0])[None, :], qs)   # conj(src'(a))⊗src'(τ)
                des_q[dn] = P.qmul(smp_d["quat"][dn], rel)
            else:
                des_q[dn] = qs
    if channels == "rot+loc":
        for s, dn in pairs:
            loc = smp_s["loc"][s]
            if mirror:
                loc = loc @ P.mirror_flip(armature, s, dn)       # t' = Fᵀ t
            ls = P.resample_vec(loc, a, tau)
            des_l[dn] = (smp_d["loc"][dn] + (ls - loc[0])) if mode == "add" else ls

    scalars, quats, info = P.pose_deltas(armature, smp_d, desired_quat=des_q,
                                         desired_loc=des_l or None,
                                         strength=strength)

    # ---- metrics ----
    n_dst = d - c + 1
    inner = [c + blend, d - blend] if d - blend >= c + blend else None
    notes = []
    ovl = [dn for s, dn in pairs if s == dn and not (d < a or c > b)]
    if ovl:
        notes.append(f"{len(ovl)} 根骨源窗与目标窗重叠：已先采样后写入（结果正确），"
                     "但写入后源窗的可见姿态也变了——复测请对比写入前的数据，"
                     "或先 ab_toggle 关掉本修复再看源窗")
    if inner is None:
        notes.append(f"blend={blend} 吃掉了整个目标窗（{n_dst} 帧），没有完全生效的帧；"
                     "减小 blend 或加长目标窗")
    if channels == "rot+loc":
        notes.append("位置也复制了：髋/根骨的绝对位置会让角色瞬移，确认确实需要")
    hidden = _ik_hidden(armature, dst)
    if hidden:
        notes.append(f"{hidden} 所在肢体是 IK 模式（IK_FK<0.5），FK 骨写了也看不见；"
                     "腿请复制 foot_ik.L/R（bones=['left_foot']）")
    if space == "world_abs":
        notes.append("world_abs 复制绝对朝向，不复制世界位置；不镜像跨侧必须给 bone_map。"
                     "验收只看有效区 err_inner_deg，渐入/渐出保留原姿态，err_max 大不等于失败。"
                     "转角大小本身不是失败判据，请检查实际骨轴方向和网格姿态。")
    if repeat:
        seg = b - a + 1
        n_full, tail = divmod(d - c + 1, seg)
        notes.append(
            f"repeat：把源段 [{a},{b}]（{seg} 帧）按原速循环铺 [{c},{d}]"
            f"＝{n_full} 整轮" + (f"＋{tail} 帧（最后一轮被截断）" if tail else "")
            + "；窗口外一帧不写，后面仍是原动画。"
            + ("blend>0 会在**每一轮接缝**处淡回原姿态、把循环打断——"
               "要真正循环请用 blend=0。" if blend > 0 else ""))
    if world and strength != 1.0:
        notes.append("world 模式 strength≠1：子骨按父骨完全到位解算，结果是近似")
    metrics = {
        "bones": info,
        "mirror_map": {s: dn for s, dn in pairs},
        "time_scale": round(ts, 4),
        "src_range": [a, b], "dst_range": [c, d], "inner_range": inner,
        "space": space, "channels": channels, "mode": mode, "mirror": mirror,
        "rot_change_max_deg": max((v.get("rot_change_max_deg", 0.0)
                                   for v in info.values()), default=0.0),
    }
    if world:
        metrics["world_parent_link"] = {dn: anc_of.get(dn) for dn in dst}
    if mode == "replace":
        # a = 目标窗、b = 源窗：compare_motion 把 b 重采样到 a 的帧上，用的
        # 正是复制时的 τ 和 slerp → 时间缩放的复制也能精确验收；trim 也就
        # 按目标帧数掐掉 blend taper。（反过来放会二次插值，缩放时差 ~1°。）
        # repeat：线性重采样表达不了取模映射，所以只验**第一轮**（[c, c+seg-1]
        # 与源段等长 → 重采样退化成恒等，正是第一轮真正的映射）；后面几轮是同一轮
        # 的复制品，由 e2e 逐帧比较各轮之间是否一致来兜。
        metrics["verify"] = {
            "tool": "compare_motion",
            "args": {"a": {"bones": list(dst),
                           "frame_range": [c, (c + (b - a)) if repeat else d]},
                     "b": {"bones": list(src), "frame_range": [a, b]},
                     "mirror": mirror, "space": space, "trim": blend},
            "pass": "err_inner_deg < 0.05",
        }
        if repeat:
            metrics["verify"]["first_tile_only"] = True
    if notes:
        metrics["notes"] = notes

    # params = 全部输入（规范化后）——reapply 靠它。bones 记**目标骨**
    # （effect_check / touched / 修复列表都把 params.bones 当"写了哪些骨"），
    # 源骨在 src_bones。_applied_window 是上次写入的窗口快照，reapply 用它
    # 分辨这次改的是 dst_start / dst_range / frame_range 中的哪一个。
    params = {
        "src_bones": list(src), "bones": list(dst), "src_range": [a, b],
        "dst_start": c, "dst_range": [c, d], "time_scale": ts,
        "frame_range": [c, d], "mirror": mirror,
        "bone_map": dict(bone_map) if bone_map else None,
        "space": space, "channels": channels, "mode": mode,
        "strength": strength, "blend": blend, "repeat": repeat,
        "_applied_window": [c, d],
    }
    if dry_run:
        if "verify" in metrics:
            # 修前基线直接给出（sonnet 第六轮：写入前没有 op_id，要手抄约 1 KB 的 verify.args
            # 才能量修前，是整个流程里最容易抄错的一步）。与写入后 compare_motion(op_id)
            # 同一口径。
            try:
                va = metrics["verify"]["args"]
                # P6: the copy already sampled both windows (visible, before any
                # write) - the same per-bone samples compare_motion would take
                before = compare_motion(scene, armature,
                                        a_bones=va["a"]["bones"], a_range=va["a"]["frame_range"],
                                        b_bones=va["b"]["bones"], b_range=va["b"]["frame_range"],
                                        mirror=va["mirror"], space=va["space"], trim=va["trim"],
                                        _samples=(smp_d, smp_s))
                metrics["verify"]["err_inner_before_deg"] = before["err_inner_deg"]
            except Exception as exc:  # noqa: BLE001 - 只是附带的基线，失败不挡 dry_run
                metrics["verify"]["err_inner_before_deg"] = None
                metrics["verify"]["before_error"] = str(exc)[:160]
        return {"dry_run": True, "params": params, "metrics": metrics,
                "frames": [c, d]}
    track, strip = P.write_pose(armature, f"agent_copy_{c}_{d}", c, scalars,
                                quats, blend=blend, track_name=track_name)
    op = agent_ops._new_op("motion_copy", params, (c, d), strip.name, op_mode,
                           metrics, track=track.name)
    return agent_ops._record(data_dir, op) if (data_dir and record) else op


# ---------------------------------------------------------------------------
# compare_motion（只读）


def compare_motion(scene: Any, armature: Any, *, a_bones: Sequence[str],
                   a_range: Sequence[int],
                   b_bones: Sequence[str] | None = None,
                   b_range: Sequence[int] | None = None,
                   mirror: bool = False,
                   bone_map: Mapping[str, str] | None = None,
                   space: str = "local", trim: int = 4,
                   channels: str = "rot", _samples=None) -> dict:
    """b 段 vs a 段（mirror 时 vs a 的镜像）逐帧旋转误差（度）。

    配对：bone_map 优先；否则 b_bones 与 a_bones 等长按顺序配；否则镜像名/同名。
    b 窗长度不同 → 把 b 重采样到 a 的帧数（线性时间映射）。验收时间缩放的
    复制时 a 放目标窗、b 放源窗（b 的重采样与复制同一插值，误差≈0；反过来
    是二次插值，会有 ~1° 的假误差）——motion_copy 的 metrics.verify 就是这样给的。
    local：比较局部 basis（mirror：b vs mirror_local(a)）；
    world：比较骨架空间朝向（mirror：b vs S·R_a·F；不镜像 b vs R_a·Rest_a⁻¹·Rest_b）。
    world_abs：直接比较实际朝向（不镜像 b vs R_a；镜像 b vs S·R_a·S）。
    err_inner = 掐掉两端各 trim 帧后的最大误差（复制有 blend taper，验收看它）。
    """
    space = _choice("space", space or "local", _SPACES)
    channels = _choice("channels", channels or "rot", _CHANNELS)
    mirror = bool(mirror)
    a0, a1 = _rng(a_range, "a.frame_range")
    b0, b1 = _rng(b_range if b_range is not None else a_range, "b.frame_range")
    if a1 - a0 < 1 or b1 - b0 < 1:
        raise RuntimeError("a/b 帧段都至少 2 帧")
    A = P.resolve_pose_bones(armature, a_bones)
    if not A:
        raise RuntimeError("a 没有骨：给 bones 或 chain")
    bmap = {str(k): str(v) for k, v in (bone_map or {}).items()}
    if bmap:
        pairs = _pairs(armature, A, mirror, bmap)
    elif b_bones:
        B = P.resolve_pose_bones(armature, b_bones)
        if len(B) != len(A):
            raise RuntimeError(f"a 有 {len(A)} 根骨、b 有 {len(B)} 根，没法按顺序配对；"
                               "给 bone_map 或等长列表")
        pairs = list(zip(A, B))
    else:
        pairs = _pairs(armature, A, mirror, None)
    Bset = list(dict.fromkeys(bn for _x, bn in pairs))
    world = space in ("world", "world_abs")
    na, nb = a1 - a0 + 1, b1 - b0 + 1
    tb = b0 + np.arange(na, dtype=np.float64) * (nb - 1) / float(na - 1)
    if _samples is not None:
        # caller's visible samples of the same windows (a superset of the bones
        # is fine: sample_visible's per-bone values do not depend on the others)
        smp_a, smp_b = _samples
    else:
        smp_a = P.sample_visible(scene, armature, A, list(range(a0, a1 + 1)), world=world)
        smp_b = P.sample_visible(scene, armature, Bset, list(range(b0, b1 + 1)),
                                 world=world)
    trim = max(0, int(trim))
    sl = slice(trim, na - trim) if na > 2 * trim else slice(None)
    per_bone = {}
    for x, y in pairs:
        if world:
            ra = _rot3(smp_a["mat"][x])
            fm = _orientation_map(armature, x, y, mirror, space)
            ea = (_S @ ra @ fm) if mirror else (ra @ fm)
            rb = _resample_rot(_rot3(smp_b["mat"][y]), b0, tb)
            err = P.qangle_deg(P.mat_to_quat(ea), P.mat_to_quat(rb))
        else:
            qa = smp_a["quat"][x]
            if mirror:
                qa, _ = P.mirror_local(qa, None, P.mirror_flip(armature, x, y))
            qb = P.resample_quats(smp_b["quat"][y], b0, tb)
            err = P.qangle_deg(qa, qb)
        row = {"b_bone": y,
               "err_per_frame": [round(float(e), 4) for e in err],
               "err_max_deg": round(float(err.max()), 4),
               "err_mean_deg": round(float(err.mean()), 4),
               "err_inner_deg": round(float(err[sl].max()), 4)}
        if channels == "rot+loc":
            la = smp_a["loc"][x]
            if mirror:
                la = la @ P.mirror_flip(armature, x, y)
            lb = P.resample_vec(smp_b["loc"][y], b0, tb)
            le = np.linalg.norm(la - lb, axis=1)
            row["loc_err_max_m"] = round(float(le.max()), 5)
            row["loc_err_inner_m"] = round(float(le[sl].max()), 5)
        per_bone[x] = row
    worst = max(per_bone, key=lambda k: per_bone[k]["err_inner_deg"])
    out = {
        "err_inner_deg": per_bone[worst]["err_inner_deg"],
        "err_max_deg": max(r["err_max_deg"] for r in per_bone.values()),
        "err_mean_deg": round(float(np.mean([r["err_mean_deg"]
                                             for r in per_bone.values()])), 4),
        "worst_bone": worst,
        "pairs": {x: y for x, y in pairs},
        "a_range": [a0, a1], "b_range": [b0, b1],
        "time_scale": round((nb - 1) / float(na - 1), 4),
        "space": space, "mirror": mirror, "trim": trim,
        "frame0": a0,
        "bones": per_bone,
    }
    if channels == "rot+loc":
        out["loc_err_inner_m"] = max(r["loc_err_inner_m"] for r in per_bone.values())
    return out


# ---------------------------------------------------------------------------
# bridge shells


def _names(ctx, args) -> list:
    """bones（角色名或骨名）或 chain 预设。"""
    if args.get("chain"):
        return P.chain_preset(args["chain"], ctx["armature"])
    return ctx["resolve_bones"](args.get("bones") or [])


def _resolve_map(ctx, bone_map) -> dict | None:
    if not bone_map:
        return None
    if not isinstance(bone_map, Mapping):
        raise RuntimeError("bone_map 需要 {源骨: 目标骨} 对象（角色名或骨名）")
    return {ctx["resolve_bones"]([k])[0]: ctx["resolve_bones"]([v])[0]
            for k, v in bone_map.items()}


def _src_names(ctx, args, bmap) -> list:
    if args.get("bones") or args.get("chain"):
        return _names(ctx, args)
    return list(bmap or [])


def _tool_motion_copy(ctx, **args):
    arm = ctx["armature"]
    if arm is None:
        raise RuntimeError("没有识别到 RIG 骨架")
    bmap = _resolve_map(ctx, args.get("bone_map"))
    names = _src_names(ctx, args, bmap)
    rest = {k: v for k, v in args.items() if k not in ("bones", "chain", "bone_map")}
    src_warn = _source_claim_warnings(ctx, arm, names, rest)
    op = motion_copy(ctx["scene"], arm, bones=names, bone_map=bmap,
                     data_dir=ctx["data_dir"], **rest)
    if src_warn:
        op["_warnings"] = src_warn
    if not op.get("dry_run"):
        ctx["after_write"](op["frames"])
    return op


def _source_claim_warnings(ctx, arm, src_names, args) -> list:
    """源窗正被别的 agent 认领（= 他们正在改这段）→ 提醒：复制的是此刻的姿态。

    租约只保护"写"；复制的源是"读"——对方改完后源就变了、副本却不会跟着变。
    local 空间只看同骨；world 空间连祖先骨也算（父骨一动，世界朝向就变）。"""
    try:
        from . import agent_bridge
        a, b = _rng(args.get("src_range"), "src_range")
        me = ctx.get("agent_id") or "__anon__"
        anc = agent_bridge._ancestor_fn(arm)
        hard, soft = agent_bridge._LEASES.conflicts(me, set(src_names), (a, b), anc)
    except Exception:  # noqa: BLE001 - 提示性检查，绝不挡写入
        return []
    world = str(args.get("space", "local")).lower() == "world"
    held = list(hard) + (list(soft) if world else [])
    if not held:
        return []
    who = sorted({h["agent_id"] for h in held})
    return [f"源窗 [{a},{b}] 正被 {who} 认领（他们在改这段）：复制的是此刻的姿态，对方改完后源会变、"
            f"副本不会跟着变——最好等对方 release 后再复制；已经复制了就在对方完成后 reapply（overrides:{{}}）"]


def _side(ctx, spec, label):
    if spec is None:
        return None, None
    if not isinstance(spec, Mapping):
        raise RuntimeError(f"{label} 需要 {{bones|chain, frame_range}} 对象")
    bones = _names(ctx, spec) if (spec.get("bones") or spec.get("chain")) else None
    return bones, spec.get("frame_range")


def _verify_args_of_op(ctx, op_id):
    """op_id → 与 motion_copy 当时返回的 metrics.verify.args 完全相同的验收参数。

    sonnet 实测：手抄 19 骨 × 2 的骨名列表是整个流程里最容易出错的一步；按 op_id
    验收省掉这一步，而且永远对应 op **当前**的窗口（reapply 挪过也对）。"""
    from . import agent_ops
    op = agent_ops.get_op(ctx["data_dir"], str(op_id)) if ctx.get("data_dir") else None
    if op is None:
        raise RuntimeError(f"op {op_id} 不存在（看 list_ops）")
    if op.get("tool") != "motion_copy":
        raise RuntimeError(f"op {op_id} 是 {op.get('tool')}，compare_motion 按 op_id 只验收 motion_copy")
    p = op.get("params") or {}
    if p.get("mode", "replace") != "replace":
        raise RuntimeError("mode=add 的复制没有逐帧可比的目标（目标 = 原动作 + 源的变化量），"
                           "用 analyze_motion / effect_check 看效果")
    fr = op.get("frames") or p.get("frame_range")
    return {"a": {"bones": list(p["bones"]), "frame_range": [int(fr[0]), int(fr[1])]},
            "b": {"bones": list(p["src_bones"]), "frame_range": list(p["src_range"])},
            "mirror": bool(p.get("mirror")), "space": p.get("space", "local"),
            "trim": int(p.get("blend", 4))}


def _tool_compare_motion(ctx, a=None, b=None, mirror=False, bone_map=None,
                         space="local", trim=4, channels="rot", detail=False,
                         op_id=None, **unknown):
    _reject_unknown("compare_motion", unknown,
                    ("a", "b", "mirror", "bone_map", "space", "trim", "channels",
                     "detail", "op_id"))
    arm = ctx["armature"]
    if arm is None:
        raise RuntimeError("没有识别到 RIG 骨架")
    newer = []
    if op_id is not None:
        if a is not None or b is not None:
            raise RuntimeError("op_id 和 a/b 二选一：按 op 验收就只给 op_id")
        va = _verify_args_of_op(ctx, op_id)
        a, b, mirror, space, trim = va["a"], va["b"], va["mirror"], va["space"], va["trim"]
        # 验收的是这条复制本身：之后叠在同骨同帧上层的修复先静音（否则把别人的修复
        # 也算成这条复制的误差），读完按名字恢复
        from . import agent_ops
        op = agent_ops.get_op(ctx["data_dir"], str(op_id))
        track, _strip = agent_ops.find_op_strip(arm, op)
        if track is not None:
            newer = agent_ops.newer_same_bone_tracks(
                arm, track.name, agent_ops.op_written_bones(arm, op),
                [tuple(va["a"]["frame_range"])])
    a_bones, a_fr = _side(ctx, a, "a")
    if not a_bones or not a_fr:
        raise RuntimeError("a 需要 {bones 或 chain, frame_range}")
    b_bones, b_fr = _side(ctx, b, "b")
    from . import agent_ops as _ops
    with _ops.tracks_muted(arm, newer):
        res = compare_motion(ctx["scene"], arm, a_bones=a_bones, a_range=a_fr,
                             b_bones=b_bones, b_range=b_fr,
                             mirror=mirror, bone_map=_resolve_map(ctx, bone_map),
                             space=space, trim=trim, channels=channels)
    if newer:
        res["muted_newer_tracks"] = list(newer)
    if not detail:
        # 逐帧数组默认不回（19 骨×46 帧≈10k token，sonnet 实测被它淹没）；
        # 要看就传 detail:true
        for row in (res.get("bones") or {}).values():
            if isinstance(row, dict):
                row.pop("err_per_frame", None)
                row.pop("loc_err_per_frame_mm", None)
    summary = (f"compare_motion {res['space']}{'·镜像' if res['mirror'] else ''}："
               f"{len(res['pairs'])} 对骨 err_inner={res['err_inner_deg']}°"
               f"（最差 {res['worst_bone']}→{res['pairs'][res['worst_bone']]}，"
               f"掐两端各 {res['trim']} 帧），err_max={res['err_max_deg']}°")
    warnings = []
    if res["time_scale"] != 1.0:
        warnings.append(f"a/b 不等长（b/a={res['time_scale']}）：b 已重采样到 a 的帧上。"
                        "验收时间缩放的复制要把目标窗放 a、源窗放 b（即 motion_copy 的 "
                        "metrics.verify.args）；反过来是二次插值，会有 ~1° 的假误差")
    return {"summary": summary, "data": res, "warnings": warnings,
            "truncated": False,
            "hint": "复制验收看 err_inner_deg（trim 应 ≥ 写入时的 blend）；"
                    "时间缩放的复制 a 放目标窗、b 放源窗（直接用 motion_copy 返回的 "
                    "metrics.verify.args）；逐帧误差要 detail:true（与 a 帧段逐帧对齐）"}


def _scope_motion_copy(ctx, args):
    """并发租约：只写目标骨 × 目标窗。"""
    bmap = _resolve_map(ctx, args.get("bone_map"))
    src = _src_names(ctx, args, bmap)
    pairs = _pairs(ctx["armature"], src, bool(args.get("mirror")), bmap)
    _a, _b, c, d, _ts = _window(args.get("src_range"), args.get("dst_start"),
                                args.get("dst_range") or args.get("frame_range"),
                                args.get("time_scale"))
    return [([dn for _s, dn in pairs], (c, d))]


def _reapply_motion_copy(armature, base_action, *, params, frame_range, status,
                         scene, track_name):
    """reapply：dst_start / dst_range / frame_range / time_scale / src_range
    任一被改都能落到对的目标窗。

    agent_ops.reapply 写回 op 时把 op["frames"] 和 params["frame_range"] 设成
    传进来的 frame_range、op["params"] 就是传进来的 params 对象——所以这里
    **就地**规范化 params 并改写 frame_range 列表，op 记录才会跟上新窗口
    （否则改 dst_start 后 op.frames 还停在旧窗）。
    """
    scene = scene or bpy.context.scene
    p = params
    snap = p.get("_applied_window") or p.get("dst_range") or frame_range
    snap = _rng(snap, "_applied_window")
    a, b = _rng(p.get("src_range"), "src_range")
    fr = _rng(frame_range, "frame_range") if frame_range else snap
    dr = _rng(p["dst_range"], "dst_range") if p.get("dst_range") else snap
    if dr != snap:                       # 显式改 dst_range（时间缩放到该窗）
        win = dr
    elif fr != snap:                     # 面板拖 frame_range = 目标窗
        win = fr
    else:                                # dst_start / time_scale / src_range
        ds = int(p.get("dst_start", snap[0]))
        ts = float(p.get("time_scale") or 1.0)
        win = [ds, ds + int(round((b - a) * ts))]
    src = P.resolve_pose_bones(armature, p.get("src_bones") or [])
    bmap = p.get("bone_map") or None
    pairs = _pairs(armature, src, bool(p.get("mirror")), bmap)
    if p.get("bones") and list(p["bones"]) != [dn for _s, dn in pairs]:
        raise RuntimeError("motion_copy 的 bones 记的是目标骨（由 src_bones + "
                           "mirror/bone_map 推出）；换骨/换侧会换目标骨，"
                           "请 revert 后重新调用 motion_copy")
    res = motion_copy(scene, armature, bones=src, src_range=[a, b],
                      dst_range=win, mirror=bool(p.get("mirror")), bone_map=bmap,
                      space=p.get("space", "local"),
                      channels=p.get("channels", "rot"),
                      mode=p.get("mode", "replace"),
                      strength=float(p.get("strength", 1.0)),
                      blend=int(p.get("blend", 4)),
                      repeat=bool(p.get("repeat")), op_mode=status,
                      data_dir=None, track_name=track_name, record=False)
    params.update(res["params"])              # 就地规范化（见 docstring）
    if isinstance(frame_range, list):
        frame_range[:] = list(res["frames"])
    return res


TOOLS = {"motion_copy": _tool_motion_copy,
         "compare_motion": _tool_compare_motion}
WRITE_SCOPES = {"motion_copy": _scope_motion_copy}
TUNABLE = {"motion_copy": [
    {"key": "strength", "kind": "float", "min": 0.0, "max": 2.0},
    {"key": "blend", "kind": "int", "min": 0, "max": 40},
    {"key": "mode", "kind": "choice", "options": list(_MODES)},
    {"key": "space", "kind": "choice", "options": list(_SPACES)},
    {"key": "dst_start", "kind": "int", "min": 0, "max": 100000},
    {"key": "frame_range", "kind": "range"},
]}
REAPPLY = {"motion_copy": _reapply_motion_copy}
