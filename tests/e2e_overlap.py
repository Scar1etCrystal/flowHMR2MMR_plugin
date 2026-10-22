"""Headless e2e for agent_overlap: overlap / time_warp (write) + chain_lag (read).

Every check RE-MEASURES the rig (sample_visible after the write) against values
computed here from the spec - never from the solver's own intermediates:
  P   plugin registered (tools / scopes / tunables / reapply handlers)
  A   overlap arm.R 20-110 delay=1 (Euler shoulder/upper_arm/forearm + quaternion
      hand/fingers): inner frame t == baseline at t − lag_b (<0.1°),
      lag_b = min(depth·delay, 3), depths from the spec (fingers = hand+segment);
      chain root unchanged (<0.01°); outside the window unchanged; chain_lag on
      inner_frames: shoulder→upper_arm→forearm→hand each +1 (±0.5)
  D   overlap dry_run writes nothing; reapply(delay) keeps op_id / track /
      1 strip, params merged, pose follows the new delay; revert restores
  B   spine_head 65-155 (spine_fk → .001 → .003 → neck → head): same checks,
      head clamped at max_delay (+0)
  C   fractional delay=1.5 / max_delay 4.5 on arm.L 75-165: slerp-exact,
      chain_lag +1.5/level; C3 = stacking: the earlier spine op (65-155) stays
      exact under this partially-overlapping op (NLA auto-blend pinned off)
  E   time_warp: explicit linear map exact vs np.interp, window ends identity
      (blend=0); reapply(map) / revert; speed+pivot linear exact vs the spec
      knots; reapply(speed, ease=smooth) exact vs the closed-form Hermite;
      dry_run; revert; bad params raise with a fix hint
  S   another tool's strip (clean_jitter, auto-blend on) partially overlapped
      by our write keeps its result (auto-blend side effect undone)
  G   skipped_bones: missing (spine preset → spine_fk.002) + no_animation
      (torso); depths override
  H   op params mirrored into settings.agent_params (TUNABLE specs valid);
      panel-style agent_ops.reapply
"""
import os
import sys
import time

import bpy
import addon_utils
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import (  # noqa: E402
    agent_bridge, agent_ops, agent_pose as P)

RESULTS = []
T_START = time.time()


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig or next(o for o in scene.objects
                               if o.type == "ARMATURE" and o.name.startswith("RIG-"))
settings.mmr_rig = rig
data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_overlap_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)


def call(tool, **args):                       # 走桥：顺便验证注册链
    return agent_bridge._dispatch({"tool": tool, "args": args})


def sample(bones, f0, f1):
    """(quat dict, first frame) of the visible pose over [f0, f1]."""
    return P.sample_visible(scene, rig, bones, list(range(f0, f1 + 1)))["quat"], f0


def err_shift(base, after, bone, frames, times):
    """max angle between the visible pose at `frames` and baseline at `times`."""
    q0, lo0 = base
    q1, lo1 = after
    exp = P.resample_quats(q0[bone], lo0, np.asarray(times, dtype=np.float64))
    got = q1[bone][np.asarray(frames) - lo1]
    return float(P.qangle_deg(exp, got).max())


def err_same(base, after, bone, frames):
    return err_shift(base, after, bone, frames, frames)


def track_of(name):
    anim = rig.animation_data
    return next((t for t in (anim.nla_tracks if anim else ()) if t.name == name), None)


def n_strips():
    """agent strips only (the first write also pushes the base action to mcd_base)."""
    anim = rig.animation_data
    return sum(len(t.strips) for t in (anim.nla_tracks if anim else ())
               if agent_ops.is_agent_track_name(t.name))


def op_row(op_id):
    return next((o for o in agent_ops.list_ops(data_dir) if o["id"] == op_id), None)


# expected chain depth from the SPEC (not from the solver): arm =
# shoulder 0 / upper_arm 1 / forearm 2 / hand 3 / finger segment k → 3+k;
# spine_head = list order.
def arm_depth(bone):
    head = bone.split(".")[0]
    fixed = {"shoulder": 0, "upper_arm_fk": 1, "forearm_fk": 2, "hand_fk": 3}
    if head in fixed:
        return fixed[head]
    return 3 + int(bone.split(".")[1])


SPINE_HEAD = ["spine_fk", "spine_fk.001", "spine_fk.003", "neck", "head"]


def lag_of(depth, delay, max_delay):
    return min(depth * delay, max_delay)


def lag_increments(before, after, expected, label, must=(), judge=()):
    """chain_lag before/after → per-level Δ vs expected (±0.5).
    must  : has to be reliable (corr≥0.4, peak not at ±max_lag) AND within ±0.5
    judge : within ±0.5 whenever it is reliable
    rest  : reported only (noisy mocap fingers at the clamped depth)."""
    lb = {lv["child"]: lv for lv in before["data"]["levels"]}
    la = {lv["child"]: lv for lv in after["data"]["levels"]}
    rows, bad, info = [], [], [0, 0]
    for child, exp in expected.items():
        x0, x1 = lb.get(child), la.get(child)
        if not x0 or not x1 or x0["lag_frames"] is None or x1["lag_frames"] is None:
            if child in must:
                bad.append(f"{child}: no measurement")
            continue
        inc = x1["lag_frames"] - x0["lag_frames"]
        rel = x0["reliable"] and x1["reliable"]
        ok = abs(inc - exp) <= 0.5
        if child in must or child in judge:
            rows.append(f"{child}:{x0['lag_frames']:+.2f}→{x1['lag_frames']:+.2f} "
                        f"Δ{inc:+.2f}(exp {exp:+.1f},r{min(x0['corr'], x1['corr']):.2f}"
                        f"{'' if rel else ',unreliable'})")
        if child in must and not (rel and ok):
            bad.append(child)
        elif child in judge and rel and not ok:
            bad.append(child)
        elif child not in must and child not in judge and rel:
            info[0] += 1
            info[1] += int(ok)
    check(label, not bad,
          f"bad={bad} :: " + " | ".join(rows)
          + f" || other levels reliable&within±0.5: {info[1]}/{info[0]}")


# ---------- P0: registration --------------------------------------------------
check("P0 plugin registered",
      "agent_overlap" not in agent_bridge._PLUGIN_ERRORS
      and all(t in agent_bridge.TOOLS for t in ("overlap", "time_warp", "chain_lag"))
      and all(t in agent_bridge.WRITE_SCOPES for t in ("overlap", "time_warp"))
      and all(t in agent_ops.TUNABLE_PARAMS for t in ("overlap", "time_warp"))
      and all(t in agent_ops.REAPPLY_HANDLERS for t in ("overlap", "time_warp")),
      f"errors={agent_bridge._PLUGIN_ERRORS}")
ping = call("ping")
check("P1 ping lists tools", ping["ok"] and "overlap" in ping["data"]["tools"]
      and "chain_lag" in ping["data"]["tools"], f"warnings={ping.get('warnings')}")
scope = agent_bridge.WRITE_SCOPES["overlap"](agent_bridge._ctx(),
                                             {"chain": "arm.L", "frame_range": [10, 20]})
check("P2 write scope = chain bones × frame_range",
      scope[0][1] == (10, 20) and "upper_arm_fk.L" in scope[0][0]
      and "f_pinky.03.L" in scope[0][0], f"{len(scope[0][0])} bones {scope[0][1]}")

# ---------- A: overlap arm.R, delay=1 ------------------------------------------
A0, A1, BL = 20, 110, 4
ARM_R = P.chain_preset("arm.R", rig)
EULER_R = [b for b in ARM_R if rig.pose.bones[b].rotation_mode not in ("QUATERNION", "AXIS_ANGLE")]
baseA = sample(ARM_R, A0 - 10, A1 + 10)
IN_A = [A0 + BL, A1 - BL]                      # = overlap metrics.inner_frames
lagA0 = call("chain_lag", chain="arm.R", frame_range=IN_A)
n0 = n_strips()
rA = call("overlap", chain="arm.R", frame_range=[A0, A1], delay=1.0, max_delay=3.0,
          blend=4)   # 显式钉住旧默认：本节专测"末端钳在 max_delay"
okA = rA.get("ok")
dA = rA.get("data") or {}
OP_A = dA.get("op_id")
rowA = op_row(OP_A) if OP_A else None
check("A1 overlap via bridge",
      okA and rowA is not None and rowA["status"] == "preview"
      and rowA["strip"].startswith("agent_overlap_") and rowA["track"].startswith("agent_")
      and n_strips() == n0 + 1,
      f"summary={rA.get('summary')} err={rA.get('error')} op={OP_A} "
      f"strip={rowA and rowA['strip']} track={rowA and rowA['track']}")
mA = dA.get("metrics") or {}
mb = mA.get("bones", {})
check("A2 metrics depth/lag per bone",
      all(mb.get(b, {}).get("depth") == arm_depth(b)
          and abs(mb.get(b, {}).get("lag_frames", -9) - lag_of(arm_depth(b), 1.0, 3.0)) < 1e-6
          for b in ARM_R) and mA.get("skipped_bones") == {}
      and "rot_change_max_deg" in mb.get("hand_fk.R", {}),
      " ".join(f"{b}:d{mb.get(b, {}).get('depth')}/lag{mb.get(b, {}).get('lag_frames')}/"
               f"{mb.get(b, {}).get('rot_change_max_deg')}°" for b in ARM_R[:6]))
afterA = sample(ARM_R, A0 - 10, A1 + 10)
inner = np.arange(A0 + BL, A1 - BL + 1)
errs = {b: err_shift(baseA, afterA, b, inner, inner - lag_of(arm_depth(b), 1.0, 3.0))
        for b in ARM_R}
check("A3 inner t == baseline(t-lag) all bones (<0.1°)",
      max(errs.values()) < 0.1,
      f"max={max(errs.values()):.4f}° euler({','.join(EULER_R)})="
      f"{max(errs[b] for b in EULER_R):.5f}° worst={max(errs, key=errs.get)}")
check("A4 Euler bones covered + moved",
      {"upper_arm_fk.R", "forearm_fk.R", "shoulder.R"} <= set(EULER_R)
      and mb.get("upper_arm_fk.R", {}).get("rot_change_max_deg", 0) > 3
      and mb.get("forearm_fk.R", {}).get("rot_change_max_deg", 0) > 3,
      f"euler={EULER_R} upper={mb.get('upper_arm_fk.R', {}).get('rot_change_max_deg')}° "
      f"fore={mb.get('forearm_fk.R', {}).get('rot_change_max_deg')}°")
root_err = err_same(baseA, afterA, "shoulder.R", np.arange(A0 - 10, A1 + 11))
check("A5 chain root shoulder.R unchanged (<0.01°)", root_err < 0.01, f"{root_err:.6f}°")
outside = np.r_[np.arange(A0 - 10, A0), np.arange(A1 + 1, A1 + 11)]
out_err = max(err_same(baseA, afterA, b, outside) for b in ARM_R)
check("A6 outside window unchanged (<0.01°)", out_err < 0.01, f"{out_err:.6f}°")
lagA1 = call("chain_lag", chain="arm.R", frame_range=IN_A)
check("A7 chain_lag shape + inner_frames echo", lagA1["ok"]
      and lagA1["data"]["bones"]["shoulder.R"]["lag_frames"] is None
      and lagA1["data"]["bones"]["f_index.01.R"]["parent"] == "hand_fk.R"
      and lagA1["data"]["ref_range"] == [IN_A[0] + 6, IN_A[1] - 6]
      and mA.get("inner_frames") == IN_A,
      lagA1.get("summary"))
lag_increments(lagA0, lagA1,
               {b: lag_of(arm_depth(b), 1, 3) - lag_of(arm_depth(b) - 1, 1, 3)
                for b in ARM_R if b != "shoulder.R"},
               "A8 chain_lag per level +1 (shoulder→upper_arm→forearm→hand) ±0.5",
               must=("upper_arm_fk.R", "forearm_fk.R", "hand_fk.R"))

# ---------- D: dry_run / reapply / revert (overlap) ----------------------------
n_before, ops_before = n_strips(), len(agent_ops.list_ops(data_dir))
rD = call("overlap", chain="arm.R", frame_range=[A0, A1], delay=2.0, max_delay=3.0,
          blend=4, dry_run=True)
check("D1 dry_run writes nothing",
      rD["ok"] and rD["data"].get("dry_run") is True and n_strips() == n_before
      and len(agent_ops.list_ops(data_dir)) == ops_before
      and rD["data"]["metrics"]["bones"]["forearm_fk.R"]["lag_frames"] == 3,
      f"strips {n_before}->{n_strips()} hand lag="
      f"{rD['data'].get('metrics', {}).get('bones', {}).get('hand_fk.R', {}).get('lag_frames')}")
rR = call("reapply", op_id=OP_A, overrides={"delay": 0.5})
opR = rR.get("data") or {}
trA = track_of(rowA["track"]) if rowA else None
check("D2 reapply same id / track / 1 strip / params merged",
      rR["ok"] and opR.get("id") == OP_A and opR.get("track") == rowA["track"]
      and trA is not None and len(trA.strips) == 1 and trA.strips[0].name == rowA["strip"]
      and opR["params"].get("delay") == 0.5 and opR["params"].get("chain") == "arm.R"
      and opR["params"].get("max_delay") == 3.0,
      f"err={rR.get('error')} strips={[s.name for s in trA.strips] if trA else None} "
      f"params.delay={opR.get('params', {}).get('delay')}")
afterR = sample(ARM_R, A0 - 10, A1 + 10)
errsR = {b: err_shift(baseA, afterR, b, inner, inner - lag_of(arm_depth(b), 0.5, 3.0))
         for b in ARM_R}
moved = max(err_same(afterA, afterR, b, inner) for b in ("hand_fk.R", "forearm_fk.R"))
check("D3 reapply result follows delay=0.5 (<0.1°) and changed",
      max(errsR.values()) < 0.1 and moved > 1.0
      and abs(opR["metrics"]["bones"]["hand_fk.R"]["lag_frames"] - 1.5) < 1e-6,
      f"max_err={max(errsR.values()):.4f}° changed_by={moved:.1f}° "
      f"hand lag={opR.get('metrics', {}).get('bones', {}).get('hand_fk.R', {}).get('lag_frames')}")
rV = call("revert", op_id=OP_A)
afterV = sample(ARM_R, A0 - 10, A1 + 10)
rev_err = max(err_same(baseA, afterV, b, np.arange(A0 - 10, A1 + 11)) for b in ARM_R)
check("D4 revert restores (<0.01°)",
      rV["ok"] and rev_err < 0.01 and op_row(OP_A)["status"] == "reverted"
      and track_of(rowA["track"]) is None, f"{rev_err:.6f}°")

# ---------- B: spine_head ------------------------------------------------------
B0, B1 = 65, 155
baseB = sample(SPINE_HEAD, B0 - 10, B1 + 10)
IN_B = [B0 + BL, B1 - BL]
lagB0 = call("chain_lag", chain="spine_head", frame_range=IN_B)
rB = call("overlap", chain="spine_head", frame_range=[B0, B1], delay=1.0, max_delay=3.0,
          blend=4)   # 内段按 blend=4 计算
OP_B = (rB.get("data") or {}).get("op_id")
afterB = sample(SPINE_HEAD, B0 - 10, B1 + 10)
innerB = np.arange(B0 + BL, B1 - BL + 1)
errsB = {b: err_shift(baseB, afterB, b, innerB, innerB - lag_of(i, 1.0, 3.0))
         for i, b in enumerate(SPINE_HEAD)}
mB = (rB.get("data") or {}).get("metrics", {}).get("bones", {})
check("B1 spine_head inner t == baseline(t-lag) (<0.1°)",
      rB["ok"] and max(errsB.values()) < 0.1
      and [mB.get(b, {}).get("depth") for b in SPINE_HEAD] == [0, 1, 2, 3, 4],
      f"max={max(errsB.values()):.4f}° depths={[mB.get(b, {}).get('depth') for b in SPINE_HEAD]} "
      f"lags={[mB.get(b, {}).get('lag_frames') for b in SPINE_HEAD]}")
rootB = err_same(baseB, afterB, "spine_fk", np.arange(B0 - 10, B1 + 11))
outB = max(err_same(baseB, afterB, b, np.r_[np.arange(B0 - 10, B0), np.arange(B1 + 1, B1 + 11)])
           for b in SPINE_HEAD)
check("B2 root spine_fk unchanged + outside unchanged (<0.01°)",
      rootB < 0.01 and outB < 0.01, f"root={rootB:.6f}° outside={outB:.6f}°")
lagB1 = call("chain_lag", chain="spine_head", frame_range=IN_B)
lag_increments(lagB0, lagB1,
               {b: lag_of(i, 1, 3) - lag_of(i - 1, 1, 3) for i, b in enumerate(SPINE_HEAD) if i},
               "B3 chain_lag spine→neck→head +1/level (head clamped +0) ±0.5",
               must=("spine_fk.001", "spine_fk.003", "neck"), judge=("head",))

# ---------- C: fractional delay 1.5 -------------------------------------------
C0, C1, CD, CM = 75, 165, 1.5, 4.5
ARM_L = P.chain_preset("arm.L", rig)
baseC = sample(ARM_L, C0 - 10, C1 + 10)
IN_C = [C0 + BL, C1 - BL]
lagC0 = call("chain_lag", chain="arm.L", frame_range=IN_C)
rC = call("overlap", chain="arm.L", frame_range=[C0, C1], delay=CD, max_delay=CM,
          blend=4)   # 内段按 blend=4 计算
afterC = sample(ARM_L, C0 - 10, C1 + 10)
innerC = np.arange(C0 + BL, C1 - BL + 1)
errsC = {b: err_shift(baseC, afterC, b, innerC, innerC - lag_of(arm_depth(b), CD, CM))
         for b in ARM_L}
mC = (rC.get("data") or {}).get("metrics", {})
# 证明确实走了小数帧 slerp：同一帧拿整数帧近邻比，误差应明显 > 0
near = max(err_shift(baseC, afterC, b, innerC, innerC - 1.0) for b in ("upper_arm_fk.L",))
check("C1 delay=1.5 slerp-exact (<0.1°) incl. Euler upper_arm/forearm",
      rC["ok"] and max(errsC.values()) < 0.1 and near > 0.5
      and mC.get("read_range") == [C0 - 6, C1],
      f"max={max(errsC.values()):.4f}° upper={errsC['upper_arm_fk.L']:.5f}° "
      f"fore={errsC['forearm_fk.L']:.5f}° (vs integer-lag {near:.2f}°) read={mC.get('read_range')}")
afterB2 = sample(SPINE_HEAD, B0 - 10, B1 + 10)
errsB2 = {b: err_shift(baseB, afterB2, b, innerB, innerB - lag_of(i, 1.0, 3.0))
          for i, b in enumerate(SPINE_HEAD)}
anim = rig.animation_data
flags = {s.name: (s.use_auto_blend, s.blend_in, s.blend_out)
         for t in anim.nla_tracks if agent_ops.is_agent_track_name(t.name) for s in t.strips}
check("C3 stacking: spine op 65-155 still exact under the partially-overlapping "
      "arm.L op 75-165 (NLA auto-blend off)",
      max(errsB2.values()) < 0.1
      and all(not f[0] and f[1] == 0 and f[2] == 0 for f in flags.values()),
      f"spine max={max(errsB2.values()):.4f}° strips={flags}")
lagC1 = call("chain_lag", chain="arm.L", frame_range=IN_C)
lag_increments(lagC0, lagC1,
               {b: lag_of(arm_depth(b), CD, CM) - lag_of(arm_depth(b) - 1, CD, CM)
                for b in ARM_L if b != "shoulder.L"},
               "C2 chain_lag +1.5/level ±0.5",
               must=("upper_arm_fk.L", "forearm_fk.L"), judge=("hand_fk.L",))

# ---------- E: time_warp -------------------------------------------------------
E0, E1, PV = 440, 530, 478
ARM_LN = P.chain_preset("arm_nofingers.L", rig)
baseE = sample(ARM_LN, E0 - 10, E1 + 10)
fullE = np.arange(E0, E1 + 1)
outE = np.r_[np.arange(E0 - 10, E0), np.arange(E1 + 1, E1 + 11)]
MAP1 = [[460, 452], [500, 510]]
rE = call("time_warp", chain="arm_nofingers.L", frame_range=[E0, E1], map=MAP1,
          ease="linear", blend=0)
OP_E = (rE.get("data") or {}).get("op_id")
rowE = op_row(OP_E) if OP_E else None
afterE = sample(ARM_LN, E0 - 10, E1 + 10)
T1 = np.interp(fullE, [E0, 460, 500, E1], [E0, 452, 510, E1])     # 独立计算
errE = {b: err_shift(baseE, afterE, b, fullE, T1) for b in ARM_LN}
endsE = max(err_same(baseE, afterE, b, [E0, E1]) for b in ARM_LN)
outEe = max(err_same(baseE, afterE, b, outE) for b in ARM_LN)
check("E1 time_warp linear map: pose == baseline(T(t)) (<0.1°)",
      rE["ok"] and rowE is not None and rowE["strip"].startswith("agent_timewarp_")
      and rowE["status"] == "preview" and max(errE.values()) < 0.1,
      f"max={max(errE.values()):.4f}° per_bone={ {b: round(v, 4) for b, v in errE.items()} } "
      f"err={rE.get('error')}")
check("E2 window ends identity + outside unchanged (<0.01°, blend=0)",
      endsE < 0.01 and outEe < 0.01, f"ends={endsE:.6f}° outside={outEe:.6f}°")
rER = call("reapply", op_id=OP_E, overrides={"map": [[470, 460]]})
trE = track_of(rowE["track"]) if rowE else None
afterER = sample(ARM_LN, E0 - 10, E1 + 10)
T2 = np.interp(fullE, [E0, 470, E1], [E0, 460, E1])
errER = max(err_shift(baseE, afterER, b, fullE, T2) for b in ARM_LN)
chgE = max(err_same(afterE, afterER, b, fullE) for b in ARM_LN)
check("E3 reapply(map) same id/track/1 strip, follows new map, changed",
      rER["ok"] and rER["data"]["id"] == OP_E and trE is not None and len(trE.strips) == 1
      and rER["data"]["params"]["map"] == [[470, 460]] and rER["data"]["params"]["ease"] == "linear"
      and errER < 0.1 and chgE > 1.0,
      f"err={errER:.4f}° changed_by={chgE:.1f}° strips={[s.name for s in trE.strips] if trE else None}")
rEV = call("revert", op_id=OP_E)
revE = max(err_same(baseE, sample(ARM_LN, E0 - 10, E1 + 10), b, np.arange(E0 - 10, E1 + 11))
           for b in ARM_LN)
check("E4 time_warp revert restores (<0.01°)", rEV["ok"] and revE < 0.01, f"{revE:.6f}°")

# speed + pivot (linear): knots (a,a),(a+D·s/(1+s), a+D/(1+s)),(pivot,pivot),(b,b)
S1 = 1.5
D = PV - E0
xs_lin = [E0, E0 + D * S1 / (1 + S1), PV, E1]
ys_lin = [E0, E0 + D / (1 + S1), PV, E1]
rS = call("time_warp", chain="arm_nofingers.L", frame_range=[E0, E1], speed=S1, pivot=PV,
          ease="linear")
OP_S = (rS.get("data") or {}).get("op_id")
mS = (rS.get("data") or {}).get("metrics", {})
afterS = sample(ARM_LN, E0 - 10, E1 + 10)
innerE = np.arange(E0 + BL, E1 - BL + 1)
TS = np.interp(innerE, xs_lin, ys_lin)
errS = max(err_shift(baseE, afterS, b, innerE, TS) for b in ARM_LN)
pivS = max(err_same(baseE, afterS, b, [PV]) for b in ARM_LN)
check("E5 speed=1.5 pivot linear: exact vs spec knots, pivot pose kept",
      rS["ok"] and errS < 0.1 and pivS < 0.1 and abs(mS.get("speed_into_pivot", 0) - S1) < 1e-3
      and np.all(TS[innerE < PV] <= innerE[innerE < PV] + 1e-9),
      f"err={errS:.4f}° pivot={pivS:.4f}° speed_into_pivot={mS.get('speed_into_pivot')} "
      f"front_lag_max={float(np.max(innerE - TS)):.2f}f")
n_before, ops_before = n_strips(), len(agent_ops.list_ops(data_dir))
rSD = call("time_warp", chain="arm_nofingers.L", frame_range=[E0, E1], speed=2.0, pivot=PV,
           dry_run=True)
check("E6 time_warp dry_run writes nothing",
      rSD["ok"] and rSD["data"].get("dry_run") is True and n_strips() == n_before
      and len(agent_ops.list_ops(data_dir)) == ops_before, f"strips={n_strips()}")
# reapply → speed 2.0, smooth: closed-form Hermite, slopes 1 / s / 1 at a / pivot / b
S2 = 2.0
rSR = call("reapply", op_id=OP_S, overrides={"speed": S2, "ease": "smooth"})
afterSR = sample(ARM_LN, E0 - 10, E1 + 10)
u = (innerE - E0) / D
v = (innerE - PV) / (E1 - PV)
TSm = np.where(innerE <= PV, E0 + D * (u + (S2 - 1) * u * u * (u - 1)),
               PV + (E1 - PV) * (v + (S2 - 1) * v * (1 - v) ** 2))
errSR = max(err_shift(baseE, afterSR, b, innerE, TSm) for b in ARM_LN)
pivSR = max(err_same(baseE, afterSR, b, [PV]) for b in ARM_LN)
chgS = max(err_same(afterS, afterSR, b, innerE) for b in ARM_LN)
trS = track_of(op_row(OP_S)["track"])
mSR = (rSR.get("data") or {}).get("metrics", {})
check("E7 reapply speed=2 smooth: exact vs closed-form, pivot kept, 1 strip, changed",
      rSR["ok"] and rSR["data"]["id"] == OP_S and errSR < 0.1 and pivSR < 0.1
      and chgS > 1.0 and trS is not None and len(trS.strips) == 1
      and abs(mSR.get("speed_into_pivot", 0) - S2) < 0.1,
      f"err={errSR:.4f}° pivot={pivSR:.4f}° changed_by={chgS:.1f}° "
      f"speed_into_pivot={mSR.get('speed_into_pivot')} knot_speeds={mSR.get('knot_speeds')}")
outS = max(err_same(baseE, afterSR, b, outE) for b in ARM_LN)
rSV = call("revert", op_id=OP_S)
revS = max(err_same(baseE, sample(ARM_LN, E0 - 10, E1 + 10), b, np.arange(E0 - 10, E1 + 11))
           for b in ARM_LN)
check("E8 outside unchanged + revert restores (<0.01°)",
      outS < 0.01 and rSV["ok"] and revS < 0.01, f"outside={outS:.6f}° revert={revS:.6f}°")
bad_cases = [
    ({"map": [[470, 460], [460, 470]]}, "非单调"),
    ({"map": [[460, 470], [470, 465]]}, "倒退"),
    ({"map": [[600, 500]]}, "越界"),
    ({"speed": 1.5, "pivot": 400}, "越界"),
    ({"speed": 1.5}, "speed+pivot"),
    ({"map": [[460, 452]], "speed": 2.0, "pivot": 470}, "二选一"),
    ({"map": [[460, 452]], "ease": "cubic"}, "ease"),
]
bad_res = []
for args, key in bad_cases:
    rb = call("time_warp", chain="arm_nofingers.L", frame_range=[E0, E1], dry_run=True, **args)
    msg = (rb.get("error") or {}).get("message", "")
    bad_res.append((not rb["ok"]) and key in msg)
ro = call("overlap", chain="arm.L", frame_range=[E0, E1], delay=-1, dry_run=True)
bad_res.append((not ro["ok"]) and "delay" in ro["error"]["message"])
ro2 = call("overlap", chain="arm", frame_range=[E0, E1], dry_run=True)
bad_res.append((not ro2["ok"]) and "未知骨链" in ro2["error"]["message"])
check("E9 bad params raise with a fix hint", all(bad_res), f"{bad_res}")

# ---------- S: another tool's strip partially overlapped by our write ----------
# clean_jitter (shared _write_strip → use_auto_blend=True) on forearm_fk.R's
# Euler channels 700-760, then overlap spine_head 730-800 on the track above:
# Blender's auto-blend would set the jitter strip's blend_out = 30 → that earlier
# fix silently weakened.  (Euler paths on purpose: clean_jitter's bone= path
# writes quaternion components as scalar deltas, which COMBINE turns into flips.)
S0, S1 = 700, 760
JP = [[f'pose.bones["forearm_fk.R"].rotation_euler', i] for i in range(3)]
rawJ = sample(["forearm_fk.R"], S0 - 5, S1 + 5)
rJ = call("clean_jitter", frame_range=[S0, S1], paths=JP, strength=1.0, width=5)
jrow = op_row((rJ.get("data") or {}).get("op_id")) or {}
fore0 = sample(["forearm_fk.R"], S0 - 5, S1 + 5)
jit = err_same(rawJ, fore0, "forearm_fk.R", np.arange(S0, S1 + 1))
rS2 = call("overlap", chain="spine_head", frame_range=[730, 800], delay=1.0)
mS2 = (rS2.get("data") or {}).get("metrics", {})
fore1 = sample(["forearm_fk.R"], S0 - 5, S1 + 5)
keep = err_same(fore0, fore1, "forearm_fk.R", np.arange(S0 - 5, S1 + 6))
jtr = track_of(jrow.get("track"))
would = None
if jtr is not None and jtr.strips:            # how big the silent damage would have been
    js = jtr.strips[0]
    js.blend_out = 30.0
    would = err_same(fore1, sample(["forearm_fk.R"], S0 - 5, S1 + 5), "forearm_fk.R",
                     np.arange(S0 - 5, S1 + 6))
    js.blend_out = 0.0
check("S1 earlier clean_jitter fix untouched by our overlapping write (<0.01°)",
      rJ["ok"] and rS2["ok"] and jit > 0.5 and keep < 0.01
      and jtr is not None and jtr.strips and not jtr.strips[0].use_auto_blend
      and jtr.strips[0].blend_in == 0 and jtr.strips[0].blend_out == 0,
      f"jitter fix size={jit:.2f}° change after our write={keep:.6f}° "
      f"(agent_ops._write_strip keeps every strip's blend at 0; unrepaired auto-blend would have "
      f"changed it by {None if would is None else round(would, 2)}°)")

# ---------- G: skipped bones ---------------------------------------------------
rG = call("overlap", chain="spine", frame_range=[B0, B1], dry_run=True)
mG = (rG.get("data") or {}).get("metrics", {})
rG2 = call("overlap", bones=["torso", "spine_fk", "spine_fk.001"], frame_range=[B0, B1],
           dry_run=True)
mG2 = (rG2.get("data") or {}).get("metrics", {})
check("G1 skipped_bones: missing + no_animation; depth over the remaining chain",
      rG["ok"] and mG.get("skipped_bones") == {"spine_fk.002": "missing"}
      and mG["bones"]["head"]["depth"] == 5 and mG["bones"]["head"]["lag_frames"] == 5
      and rG2["ok"] and mG2.get("skipped_bones") == {"torso": "no_animation"}
      and mG2["bones"]["spine_fk"]["lag_frames"] == 0
      and mG2["bones"]["spine_fk.001"]["depth"] == 1,
      f"spine={mG.get('skipped_bones')} explicit={mG2.get('skipped_bones')} "
      f"roots={mG2.get('roots')}")
rG3 = call("overlap", chain="spine_head", frame_range=[B0, B1], dry_run=True,
           depths={"neck": 1, "head": 1.5})
mG3 = (rG3.get("data") or {}).get("metrics", {}).get("bones", {})
check("G2 depths override", rG3["ok"] and mG3.get("neck", {}).get("lag_frames") == 1
      and mG3.get("head", {}).get("lag_frames") == 1.5
      and mG3.get("spine_fk.003", {}).get("lag_frames") == 2,
      f"{ {b: c.get('lag_frames') for b, c in mG3.items()} }")

# ---------- H: params mirror ---------------------------------------------------
agent_bridge._sync_params_list(settings, data_dir)
keys_B = {p.key for p in settings.agent_params if p.op_id == (rB.get("data") or {}).get("op_id")}
OP_C = (rC.get("data") or {}).get("op_id")
keys_C = {p.key for p in settings.agent_params if p.op_id == OP_C}
check("H1 overlap TUNABLE mirrored",
      {"delay", "max_delay", "strength", "blend", "frame_range"} <= keys_B,
      f"keys={sorted(keys_B)}")
rH = call("time_warp", chain="arm_nofingers.L", frame_range=[E0, E1], speed=1.3, pivot=PV)
rH2 = call("time_warp", chain="arm_nofingers.L", frame_range=[E0, E1], map=[[470, 465]])
agent_bridge._sync_params_list(settings, data_dir)
keys_H = {p.key for p in settings.agent_params if p.op_id == rH["data"]["op_id"]}
keys_H2 = {p.key for p in settings.agent_params if p.op_id == rH2["data"]["op_id"]}
check("H2 time_warp TUNABLE (speed/pivot only in speed mode)",
      {"speed", "pivot", "ease", "frame_range"} <= keys_H and "speed" not in keys_H2
      and "ease" in keys_H2, f"speed_mode={sorted(keys_H)} map_mode={sorted(keys_H2)}")
# panel slider path: agent_ops.reapply with a single override (like _param_tick)
opC2 = agent_ops.reapply(data_dir, rig, OP_C, scene=scene, delay=1.0)
check("H3 panel-style reapply(delay) on arm.L op",
      opC2["id"] == OP_C and opC2["params"]["delay"] == 1.0
      and opC2["metrics"]["bones"]["hand_fk.L"]["lag_frames"] == 3.0,
      f"hand lag={opC2['metrics']['bones']['hand_fk.L']['lag_frames']}")

# ---------- T: typo'd parameters are rejected with the closest legal name -------------
r = call("overlap", chain="arm.R", frame_range=[A0, A1], dealy=1.0, dry_run=True)
check("T1 overlap(dealy=…) → rejected, suggests delay",
      not r["ok"] and "delay" in r["error"]["message"], r.get("error", {}).get("message", "")[:160])
r = call("time_warp", chain="arm.R", frame_range=[A0, A1], speed=1.5, pivt=A0 + 20, dry_run=True)
check("T2 time_warp(pivt=…) → rejected, suggests pivot",
      not r["ok"] and "pivot" in r["error"]["message"], r.get("error", {}).get("message", "")[:160])
r = call("chain_lag", chain="arm.R", frame_range=IN_A, max_lags=4)
check("T3 chain_lag(max_lags=…) → rejected, suggests max_lag",
      not r["ok"] and "max_lag" in r["error"]["message"], r.get("error", {}).get("message", "")[:160])

# ---------- summary -----------------------------------------------------------
print(f"elapsed {time.time() - T_START:.1f}s")
# ---------- Z: 默认值 = 用户原话"每级晚 1~3 帧、越往末端越晚"（不封顶）+ 自动 blend ----
rZ = call("overlap", chain="spine_head", frame_range=[B0, B1], delay=1.0, dry_run=True)
mZ = (rZ.get("data") or {}).get("metrics", {})
lagsZ = [mZ["bones"][bn]["lag_frames"] for bn in ("spine_fk", "spine_fk.001",
                                                  "spine_fk.003", "neck", "head")
         if bn in mZ.get("bones", {})]
check("Z1 default max_delay: every level strictly later toward the tip (no clamp)",
      rZ["ok"] and len(lagsZ) >= 4 and all(y > x for x, y in zip(lagsZ, lagsZ[1:]))
      and mZ.get("max_delay_auto") is True,
      f"lags={lagsZ} max_delay={mZ.get('max_delay')}")
check("Z2 default blend ≥ 2×max lag (no reverse play in the taper)",
      mZ.get("blend_auto") is True and mZ.get("blend", 0) >= 2 * max(lagsZ or [0]),
      f"blend={mZ.get('blend')} max_lag={max(lagsZ or [0])}")
rZ2 = call("overlap", chain="spine_head", frame_range=[B0, B1], delay=1.0, blend=2,
           dry_run=True)
check("Z3 explicit too-small blend → reverse-play warning",
      any("倒放" in w for w in (rZ2.get("data") or {}).get("metrics", {}).get("warnings", [])),
      (rZ2.get("data") or {}).get("metrics", {}).get("warnings"))

fails = [r for r in RESULTS if not r[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for n, _o, d in fails:
    print(f"FAIL {n}: {d}")
sys.exit(1 if fails else 0)
