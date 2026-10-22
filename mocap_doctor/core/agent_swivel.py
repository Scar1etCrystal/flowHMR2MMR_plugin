"""swivel：膝 / 肘朝向修复——绕"髋→踝 / 肩→腕"连线转整条腿 / 胳膊，脚 / 手的位置和朝向不动。

为什么不是 hold_pose（2026-10-04 §16 实测）：
- 腿是 IK（thigh_parent["IK_FK"]=0）：thigh_fk/shin_fk 写了**看不见**，旧剧本"膝朝向 → hold_pose
  thigh_fk"在这个 RIG 上是空操作，而旧 probe 读的又正是这两根 FK 骨——agent 会"修好并验收通过"，
  用户什么也看不到。能让膝盖绕连线转的是 thigh_ik 的 Y 旋转：转 θ，膝绕 髋→踝 轴转 0.85–0.99θ，
  踝漂移 < 0.06 mm、脚朝向不变（见 tests/e2e_align.py）。
- hold_pose 把一根骨的某个局部轴对准目标，会顺带改大腿 / 上臂的指向（手脚跟着挪）。
  "膝 / 肘朝哪"只有一个自由度：绕两端连线的转角，其余都该不动。
- 误差 = 投影到连线垂面上的夹角（agent_anatomy.swivel_error_deg）：膝 / 肘朝向的几种定义
  （⊥小腿、⊥大腿、凸出角平分线）投影后是同一个方向，只有它能被修、也只有它是人说"膝盖朝前"的意思。

实现：
- FK 肢（这个 RIG 的手臂）：根控制骨（upper_arm_fk / thigh_fk）在骨架空间绕连线转 φ：
  basis' = basis ⊗ (M⁻¹ R M)；keep_end 时梢控制骨（hand_fk / foot_fk）反转回去：basis' = basis ⊗ (Mₕ⁻¹ R⁻¹ Mₕ)，
  手 / 脚的世界朝向不变（手指跟着手，也不变）。一次求解精确。
- IK 肢（这个 RIG 的腿）：thigh_ik / upper_arm_ik 的 Euler Y 增量（锁了 X/Z 的那根"膝向"控制）。
  θ→φ 不是严格 1:1，先按 1:1 写一条临时 strip、量实际转了多少、按每帧增益修正一次（割线法），再写正式 strip。
"""
from __future__ import annotations

import math
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from . import agent_anatomy as A
from . import agent_fx, agent_ops, agent_pose as P, agent_view as V
from .animation import preserve_scene_frame, set_scene_frame

JOINTS = ("knee", "elbow")
_END_VISIBLE = {"knee": "ORG-foot.{s}", "elbow": "ORG-hand.{s}"}


def _joint(joint) -> str:
    j = str(joint or "").strip().lower()
    j = {"knee_front": "knee", "elbow_front": "elbow", "膝": "knee", "肘": "elbow"}.get(j, j)
    if j not in JOINTS:
        raise RuntimeError(f"joint 只能是 'knee'（膝）或 'elbow'（肘），收到 {joint!r}")
    return j


def _side(side) -> str:
    s = str(side or "").upper()
    if s not in ("L", "R"):
        raise RuntimeError(f"side 必须是 'L' 或 'R'（角色自己的左/右），收到 {side!r}")
    return s


def controls(armature, joint, side) -> tuple[str, str | None, bool]:
    """(根控制骨, 梢控制骨 | None, 是否 IK)。"""
    spec = A._JOINTS[joint]
    ik = A.limb_is_ik(armature, joint, side)
    root = (spec["ik_ctrl"] if ik else spec["fk_ctrl"]).format(s=side)
    end = None if ik else spec["fk_end"].format(s=side)
    if armature.pose.bones.get(root) is None:
        raise RuntimeError(f"骨架上没有 {root}（{'IK' if ik else 'FK'} 的{('膝' if joint == 'knee' else '肘')}控制骨）")
    if end is not None and armature.pose.bones.get(end) is None:
        end = None
    return root, end, ik


def _measure_fn(joint, side, toward, view, mmd, axis=None):
    jfn = A.frame_joint_fn(joint, side)
    end_name = _END_VISIBLE[joint].format(s=side)

    def fn(armature, scene):
        d = jfn(armature, scene)
        if d is None or d.get("front") is None:
            return None
        t, how = V.resolve_direction(toward, scene=scene, armature=armature, origin=d["joint"],
                                     view=view, mmd=mmd, side=side, axis=axis)
        phi = A.swivel_error_deg(d["front"], t, d["chord"], pole=d.get("pole"))
        eb = armature.pose.bones.get(end_name)
        endq = (armature.matrix_world @ eb.matrix).to_quaternion() if eb is not None else None
        p0, p1, p2, _c = A.limb_points(armature, joint, side)
        return {"phi": phi, "chord": d["chord"].copy(), "joint": p1.copy(), "end": p2.copy(),
                "root": p0.copy(), "endq": endq, "how": how,
                "src": d["evidence"].get(f"{joint}_source"), "bend": d["evidence"].get("bend_deg")}
    return fn


def _sweep(scene, armature, frames, fn) -> list:
    out = []
    with preserve_scene_frame(scene):
        for f in frames:
            set_scene_frame(scene, int(f))
            out.append(fn(armature, scene))
    return out


def _fill(values: list) -> tuple[np.ndarray, int]:
    """None（退化帧）用相邻有效帧线性插值补上，免得那几帧突然弹回原样。"""
    v = np.array([np.nan if x is None else float(x) for x in values])
    ok = ~np.isnan(v)
    bad = int((~ok).sum())
    if not ok.any():
        raise RuntimeError("整个帧段都量不出膝/肘朝向误差：目标几乎平行于根→梢连线（比如手臂下垂时要肘尖朝下），"
                           "或推不出朝向。换个目标方向，或报告")
    idx = np.arange(len(v))
    v[~ok] = np.interp(idx[~ok], idx[ok], v[ok])
    return v, bad


def _inner(n, blend):
    return slice(int(blend), n - int(blend)) if n > 2 * int(blend) + 1 else slice(None)


def _drop(armature, track_name, strip_name):
    anim = armature.animation_data
    if anim is None:
        return
    for tr in list(anim.nla_tracks):
        if tr.name != track_name:
            continue
        st = tr.strips.get(strip_name)
        act = st.action if st is not None else None
        chans = agent_ops.action_channels(act)
        if st is not None:
            tr.strips.remove(st)
        anim.nla_tracks.remove(anim.nla_tracks[track_name])
        if act is not None and act.users == 0:
            bpy.data.actions.remove(act)
        agent_ops.settle_unanimated(armature, chans)
        return


def swivel(scene, armature, *, joint, side, frame_range, toward="forward", view="camera",
           strength=1.0, blend=4, keep_end=True, axis=None, op_mode="preview", data_dir=None,
           track_name=None, dry_run=False, record=True):
    joint, side = _joint(joint), _side(side)
    a, b, frames = P.strip_window(frame_range)
    if toward is None:
        raise RuntimeError("swivel 需要 toward（目标方向：forward / char_left / camera / [x,y,z] …）")
    k = float(strength)
    root, end, ik = controls(armature, joint, side)
    mmd = getattr(getattr(scene, "mocap_doctor", None), "mmd_armature", None)
    cal = A.hinge_calibration(armature, side, joint, scene)
    mfn = _measure_fn(joint, side, toward, view, mmd, axis=axis)
    bones = [root] + ([end] if (end and keep_end) else [])
    smp = P.sample_visible(scene, armature, bones, frames, world=True, extra_fn=mfn)
    meas0 = smp["extra"]
    if all(m is None for m in meas0):
        raise RuntimeError(f"{joint}.{side} 在 {a}–{b} 推不出朝向（骨缺失？）")
    phi0, degen = _fill([m["phi"] if m else None for m in meas0])
    inner = _inner(len(frames), blend)
    # 渐入渐出在**转角**上做（每帧都是绕连线的精确旋转，手腕/脚踝全程不动），strip 自身 blend=0。
    # 在 Euler 增量上做 taper 的话，大角度时过渡帧会把手腕甩开（实测 175° 的肘 swivel 过渡帧手腕漂 156 mm）。
    w = agent_fx.window_weights(len(frames), int(blend))
    rot = phi0 * k * w
    srcs = sorted({m["src"] for m in meas0 if m and m.get("src")})
    metrics = {"joint": joint, "side": side, "limb": "IK" if ik else "FK", "control": root,
               "keep_end_bone": end if keep_end else None,
               "front_source": srcs[0] if len(srcs) == 1 else srcs,
               "toward_how": next((m["how"] for m in meas0 if m), None),
               "err_before_inner_deg": round(float(np.abs(phi0[inner]).max()), 1),
               "err_before_mean_deg": round(float(np.abs(phi0[inner]).mean()), 1),
               "degenerate_frames": degen, "inner_frames": [frames[inner][0], frames[inner][-1]]
               if len(frames[inner]) else [a, b],
               "bend_deg_range": [round(min(m["bend"] for m in meas0 if m), 1),
                                  round(max(m["bend"] for m in meas0 if m), 1)]}
    if isinstance(cal, dict) and cal.get("spread_p90_deg") is not None:
        metrics["hinge_spread_p90_deg"] = cal["spread_p90_deg"]
    params = {"joint": joint, "side": side, "frame_range": [a, b], "toward": toward, "view": view,
              "strength": k, "blend": int(blend), "keep_end": bool(keep_end), "bones": bones,
              "axis": axis}
    warns = []
    big = float(np.abs(phi0[inner]).max()) if len(phi0[inner]) else 0.0
    if big > 120.0:
        warns.append(f"要绕连线转 {big:.0f}°：目标几乎在{'膝' if joint == 'knee' else '肘'}现在朝向的背面。"
                     "先确认方向说法（肘窝朝前 = 肘尖朝后 = toward:\"back\"；膝盖朝前 = toward:\"forward\"）")
    if degen:
        warns.append(f"{degen} 帧目标几乎平行于根→梢连线（转不出这个朝向），用相邻帧插值补上")
    if dry_run:
        metrics["pred_swivel_max_deg"] = round(float(np.abs(rot).max()), 1)
        return {"dry_run": True, "params": params, "metrics": metrics, "frames": [a, b],
                "warnings": warns}
    arm_q = armature.matrix_world.to_quaternion()
    name = f"agent_swivel_{joint}{side}_{a}_{b}"

    if ik:
        pb = armature.pose.bones[root]
        mode = P.rot_mode(pb)
        if mode in ("QUATERNION", "AXIS_ANGLE"):
            raise RuntimeError(f"{root} 不是 Euler 骨（{mode}），IK 膝/肘向控制只支持 Euler Y")
        path = P.rot_path(root, mode)
        # thigh_ik 的 Y 轴沿大腿（髋→膝），与 髋→踝 连线同向 → 绕 Y 正转 = 绕连线正转
        sign = []
        for i in range(len(frames)):
            mq = Matrix(smp["mat"][root][i]).to_3x3().normalized().to_quaternion()
            yax = arm_q @ (mq @ Vector((0.0, 1.0, 0.0)))
            ch = meas0[i]["chord"] if meas0[i] else None
            sign.append(1.0 if (ch is None or float(yax @ ch) >= 0.0) else -1.0)
        sign = np.array(sign)
        theta1 = np.radians(rot) * sign
        tmp_track = f"{name}_tmp"
        tr, st = P.write_pose(armature, tmp_track, a, {(path, 1): theta1}, {}, blend=0,
                              track_name=None)
        tmp_tr_name, tmp_st_name = tr.name, st.name
        try:
            meas1 = _sweep(scene, armature, frames, mfn)
        finally:
            _drop(armature, tmp_tr_name, tmp_st_name)
        phi1, _d1 = _fill([m["phi"] if m else None for m in meas1])
        target_rot = rot
        achieved = phi0 - phi1                                 # 实际转了多少
        th_deg = np.degrees(theta1) * sign
        gain = np.where(np.abs(th_deg) > 0.5, achieved / np.where(np.abs(th_deg) > 0.5, th_deg, 1.0), np.nan)
        good = (gain > 0.3) & (gain < 3.0)
        g_med = float(np.median(gain[good])) if good.any() else 1.0
        gain = np.where(good, gain, g_med)
        resid = target_rot - achieved
        theta2 = theta1 + np.radians(resid / gain) * sign
        scalars = {(path, 1): theta2}
        quats = {}
        metrics["ik_gain_median"] = round(g_med, 3)
        applied = np.degrees(np.abs(theta2))
    else:
        des = {root: np.zeros((len(frames), 4))}
        if end and keep_end:
            des[end] = np.zeros((len(frames), 4))
        for i in range(len(frames)):
            ang = math.radians(float(rot[i]))
            ch = meas0[i]["chord"] if meas0[i] else None
            if ch is None:
                ch = meas0[next(j for j in range(len(frames)) if meas0[j])]["chord"]
            R = Quaternion(arm_q.inverted() @ ch, ang)
            Mu = Matrix(smp["mat"][root][i]).to_3x3().normalized().to_quaternion()
            Bu = Quaternion(smp["quat"][root][i])
            du = Mu.inverted() @ R @ Mu
            q = Bu @ du
            des[root][i] = (q.w, q.x, q.y, q.z)
            if end and keep_end:
                Mh = Matrix(smp["mat"][end][i]).to_3x3().normalized().to_quaternion()
                Bh = Quaternion(smp["quat"][end][i])
                dh = Mh.inverted() @ R.inverted() @ Mh
                qh = Bh @ dh
                des[end][i] = (qh.w, qh.x, qh.y, qh.z)
        scalars, quats, info = P.pose_deltas(armature, smp, desired_quat=des)
        metrics["bones"] = info
        applied = np.abs(rot)
    track, strip = P.write_pose(armature, name, a, scalars, quats, blend=0,
                                track_name=track_name)
    meas2 = _sweep(scene, armature, frames, mfn)
    phi2 = np.array([abs(m["phi"]) if (m and m["phi"] is not None) else np.nan for m in meas2])
    sub = phi2[inner]
    sub = sub[~np.isnan(sub)]
    metrics["err_after_inner_deg"] = round(float(sub.max()), 1) if len(sub) else None
    metrics["applied_max_deg"] = round(float(applied.max()), 1)
    metrics["applied_mean_deg"] = round(float(applied.mean()), 1)
    drift = [(m2["end"] - m0["end"]).length * 1000.0 for m0, m2 in zip(meas0, meas2) if m0 and m2]
    rootd = [(m2["root"] - m0["root"]).length * 1000.0 for m0, m2 in zip(meas0, meas2) if m0 and m2]
    jmove = [(m2["joint"] - m0["joint"]).length * 1000.0 for m0, m2 in zip(meas0, meas2) if m0 and m2]
    endrot = [math.degrees(m0["endq"].rotation_difference(m2["endq"]).angle)
              for m0, m2 in zip(meas0, meas2) if m0 and m2 and m0["endq"] is not None and m2["endq"] is not None]
    metrics["end_drift_mm"] = round(max(drift), 2) if drift else None
    metrics["root_drift_mm"] = round(max(rootd), 2) if rootd else None
    metrics["joint_moved_max_mm"] = round(max(jmove), 1) if jmove else None
    metrics["end_rot_change_max_deg"] = round(max(endrot), 2) if endrot else None
    metrics["note"] = ("复测：probe_anatomy part=" + ("knee_front" if joint == "knee" else "elbow_front")
                       + f" side={side} toward=<同一个目标> frame_range=<用户帧段> → err_inner_deg（swivel 平面）")
    op = agent_ops._new_op("swivel", params, (a, b), strip.name, op_mode, metrics, track=track.name)
    out = agent_ops._record(data_dir, op) if (data_dir and record) else op
    if warns:
        out["_warnings"] = warns
    return out


# ---------------------------------------------------------------------------
# bridge plumbing

def _tool_swivel(ctx, **args):
    P.reject_unknown_args("swivel", swivel, args)
    if args.get("frame_range") is None:
        raise RuntimeError("swivel 需要 frame_range=[A,B]（写入窗：用户帧段两端各外扩 blend 帧）")
    op = swivel(ctx["scene"], ctx["armature"], data_dir=ctx["data_dir"], **args)
    if not op.get("dry_run"):
        ctx["after_write"](op["frames"])
    return op


def _scope_swivel(ctx, args):
    if args.get("dry_run") or args.get("frame_range") is None:
        return []
    joint, side = _joint(args.get("joint")), _side(args.get("side"))
    root, end, _ik = controls(ctx["armature"], joint, side)
    bones = [root] + ([end] if (end and args.get("keep_end", True)) else [])
    fr = args["frame_range"]
    return [(bones, (int(fr[0]), int(fr[1])))]


def _reapply_swivel(armature, base_action, *, params, frame_range, status, scene, track_name):
    p = {k: v for k, v in dict(params).items() if k != "bones"}
    p["frame_range"] = frame_range
    return swivel(scene or bpy.context.scene, armature, op_mode=status, data_dir=None,
                  track_name=track_name, record=False, **p)


TOOLS = {"swivel": _tool_swivel}
WRITE_SCOPES = {"swivel": _scope_swivel}
TUNABLE = {"swivel": [
    {"key": "strength", "kind": "float", "min": 0.0, "max": 1.0},
    {"key": "blend", "kind": "int", "min": 0, "max": 40},
    {"key": "frame_range", "kind": "range"},
]}
REAPPLY = {"swivel": _reapply_swivel}
