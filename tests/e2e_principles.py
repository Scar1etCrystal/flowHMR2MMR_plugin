"""Headless e2e for agent_principles: anticipation / follow_through / overshoot
+ the analyze_motion read tool.

Fixture segments (found by a probe over the arm chains, 3-frame smoothed
local angular speed, see the report):
  ANT  R arm 550-603  upper_arm_fk.R still 545-563 → one move (onset 564,
       peak 580 ≈7.6°/帧, amplitude ≈81°) → still (stop 597)
  FT/OV R arm 684-745 forearm_fk.R onset 689, peak 716 (≈25.6°/帧),
       sharp stop 721, still 721-745
Bones = chain arm_nofingers.R: shoulder.R (Euler YXZ), upper_arm_fk.R /
forearm_fk.R (Euler XYZ), hand_fk.R (QUATERNION).

Every number is RE-MEASURED from fresh sample_visible calls (pre vs post) with
the test's own detect_events / motion_axis — never the solver's metrics.
"""
import json
import os
import sys

import bpy
import addon_utils
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import (  # noqa: E402
    agent_bridge, agent_ops, agent_pose as P)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig or next(
    o for o in scene.objects
    if o.type == "ARMATURE" and o.name.startswith("RIG-"))
settings.mmr_rig = rig
data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_principles_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)


def call(tool, **args):
    return agent_bridge._dispatch({"tool": tool, "args": args})


CHAIN = "arm_nofingers.R"
BONES = P.chain_preset(CHAIN, rig)
MODES = {b: rig.pose.bones[b].rotation_mode for b in BONES}
print("bones", BONES, MODES,
      "depth", {b: P.depth_in(rig, b, BONES) for b in BONES},
      "plugin_errors", agent_bridge._PLUGIN_ERRORS)
check("0 plugin registered",
      all(t in agent_bridge.TOOLS for t in
          ("analyze_motion", "anticipation", "follow_through", "overshoot"))
      and "agent_principles" not in agent_bridge._PLUGIN_ERRORS,
      f"errors={agent_bridge._PLUGIN_ERRORS}")
# the RIG's arm must mix Euler + quaternion bones (the silent-no-op trap)
check("0 arm mixes Euler + quat",
      MODES.get("upper_arm_fk.R") == "XYZ" and MODES.get("hand_fk.R") == "QUATERNION",
      str(MODES))


def sample(fr):
    smp = P.sample_visible(scene, rig, BONES, list(range(fr[0], fr[1] + 1)))
    return smp["quat"]


def rv_deg(q0, q1):
    return np.degrees(P.quat_to_rotvec(P.qmul(P.qconj(q0), q1)))


def own_events(q, a):
    sp = P.smooth(P.angular_speed_deg(q), 3)
    ev = P.detect_events(sp, onset_frac=0.15, stop_frac=0.12)
    return ev, sp


def lobes(x, eps=0.2):
    out = []
    for i, v in enumerate(x):
        if abs(v) <= eps:
            continue
        if not out or (out[-1][1] > 0) != (v > 0):
            out.append([i, float(v)])
        elif abs(v) > abs(out[-1][1]):
            out[-1] = [i, float(v)]
    return out


def strips_now():
    anim = rig.animation_data
    return sum(len(t.strips) for t in (anim.nla_tracks if anim else ()))


def track_of(op):
    anim = rig.animation_data
    return next((t for t in anim.nla_tracks if t.name == op["track"]), None)


def op_in_log(op_id):
    return any(o.get("id") == op_id for o in agent_ops.list_ops(data_dir))


def within(x, target, tol=0.35):
    return target > 0 and abs(x - target) <= tol * target


def json_ok(resp):
    """The socket server json.dumps every response - numpy scalars would die there."""
    try:
        json.dumps(resp, ensure_ascii=False)
        return True
    except (TypeError, ValueError):
        return False


# =============================================================================
# A  anticipation
FR_A = [550, 603]
W_A = [FR_A[0] - 10, FR_A[1] + 10]          # sample margin: outside-window check
LEAD, DELAY, BL = 6, 2, 3
q0 = sample(W_A)
iA = FR_A[0] - W_A[0]                        # index of frame FR_A[0] in W_A
qa = {b: q0[b][iA:iA + FR_A[1] - FR_A[0] + 1] for b in BONES}
ev, sp = own_events(qa["upper_arm_fk.R"], FR_A[0])
o, p, s = ev["onset"], ev["peak"], ev["stop"]
amp_m = float(P.qangle_deg(qa["upper_arm_fk.R"][o], qa["upper_arm_fk.R"][o:s + 1]).max())
print(f"ANT own events: onset {FR_A[0] + o} peak {FR_A[0] + p} ({ev['peak_speed']:.2f}°/帧) "
      f"stop {FR_A[0] + s} amp {amp_m:.1f}°")
check("A0 segment is a real move", ev["peak_speed"] > 3.0 and o is not None and s is not None
      and o - LEAD >= BL and s <= len(sp) - 1 - BL,
      f"peak {ev['peak_speed']:.2f}°/帧 onset {FR_A[0] + o} stop {FR_A[0] + s}")

r = call("analyze_motion", chain=CHAIN, frame_range=FR_A, main_bone="upper_arm_fk.R")
pre = r.get("data") or {}
pm = pre.get("main", {})
check("A1 analyze_motion pre (bridge) matches own events",
      r.get("ok") and pm.get("onset_frame") == FR_A[0] + o
      and pm.get("peak_frame") == FR_A[0] + p and pm.get("stop_frame") == FR_A[0] + s
      and abs(pm.get("amplitude_deg", 0) - amp_m) < 0.05
      and set(pre.get("suggest", {})) == {"anticipation", "follow_through", "overshoot"}
      and json_ok(r),
      f"{r.get('summary')} | counter_pre={pm.get('counter_move_deg')} "
      f"series_pts={len(pre.get('speed_series', []))} warn={r.get('warnings')}")

r = call("anticipation", chain=CHAIN, frame_range=[FR_A[0] + 10, FR_A[1]],
         main_bone="upper_arm_fk.R")
check("A2 too-tight window → actionable error",
      not r.get("ok") and "建议 frame_range" in (r.get("error") or {}).get("message", ""),
      (r.get("error") or {}).get("message", "")[:160])

n_strips = strips_now()
r = call("anticipation", chain=CHAIN, frame_range=FR_A, main_bone="upper_arm_fk.R",
         amount=0.15, lead=LEAD, delay=DELAY, blend=BL, dry_run=True)
check("A3 dry_run writes nothing",
      r.get("ok") and r["data"].get("dry_run") and strips_now() == n_strips
      and r["data"]["metrics"]["onset_frame"] == FR_A[0] + o,
      f"strips {n_strips}->{strips_now()} metrics.counter={r['data']['metrics'].get('counter_deg')}")

r = call("anticipation", chain=CHAIN, frame_range=FR_A, main_bone="upper_arm_fk.R",
         amount=0.15, lead=LEAD, delay=DELAY, blend=BL)
d = r.get("data") or {}
OP_A = d.get("op_id")
opA = agent_ops.get_op(data_dir, OP_A) if OP_A else None
want = {"bones", "frame_range", "main_bone", "onset_frame", "stop_frame", "amount", "lead",
        "delay", "onset_frac", "stop_frac", "smooth", "axis_frames", "strength", "blend"}
check("A4 write via bridge: ok / logged / preview / agent_ names / params complete / json",
      r.get("ok") and opA is not None and op_in_log(OP_A) and d.get("status") == "preview"
      and opA["strip"].startswith("agent_") and opA["track"].startswith("agent_")
      and track_of(opA) is not None and want <= set(opA["params"]) and json_ok(r),
      f"op={OP_A} strip={opA and opA['strip']} metrics(new_onset={d.get('metrics', {}).get('new_onset_frame')}, "
      f"counter={d.get('metrics', {}).get('counter_deg')}, time_scale={d.get('metrics', {}).get('time_scale')})")

q1 = sample(W_A)
r = call("analyze_motion", chain=CHAIN, frame_range=FR_A, main_bone="upper_arm_fk.R")
post = (r.get("data") or {}).get("main", {})
target = 0.15 * amp_m
check("A5 analyze post: counter_move ≈ amount×amplitude, opposite direction",
      within(post.get("counter_move_deg") or 0, target)
      and (post.get("counter_dir_cos") or 0) < -0.7,
      f"counter {post.get('counter_move_deg')}° vs 0.15×{amp_m:.1f}={target:.2f}° "
      f"dir_cos={post.get('counter_dir_cos')}")
check("A6 analyze post: onset moved by delay",
      post.get("onset_frame") is not None
      and abs(post["onset_frame"] - (FR_A[0] + o + DELAY)) <= 1,
      f"onset {FR_A[0] + o} → {post.get('onset_frame')} (expect {FR_A[0] + o + DELAY}±1)")

# independent: at the new onset every bone = q_pre(onset) ⊗ rot(−axis_b, 0.15·amp_b)
errs, counters = {}, {}
for b in BONES:
    qb = qa[b]
    ax, ang = P.motion_axis(qb[o], qb[o + 3])
    amp_b = float(P.qangle_deg(qb[o], qb[o:s + 1]).max())
    exp_q = P.qmul(qb[o], P.rotvec_to_quat(-ax * np.radians(0.15 * amp_b)))
    got = q1[b][iA + o + DELAY]
    errs[b] = float(P.qangle_deg(exp_q, got))
    counters[b] = round(float(rv_deg(qb[o], got) @ ax), 2)
check("A7 every bone (Euler YXZ/XYZ + quat) hits q(onset)⊗rot(−axis,0.15·amp) at new onset",
      all(e < 0.5 for e in errs.values()) and all(c < -0.5 for c in counters.values()),
      f"err°={ {b: round(e, 3) for b, e in errs.items()} } along+axis°={counters}")

dA = {b: P.qangle_deg(q0[b], q1[b]) for b in BONES}
E = s
after_E = max(float(dA[b][iA + E:iA + len(sp)].max()) for b in BONES)
before = max(float(dA[b][iA:iA + o - LEAD + 1].max()) for b in BONES)
outside = max(max(float(dA[b][:iA].max()), float(dA[b][iA + len(sp):].max())) for b in BONES)
check("A8 frames ≥ E unchanged (<0.1°)", after_E < 0.1,
      f"E={FR_A[0] + E} max={after_E:.4f}°; before onset−lead max={before:.4f}°")
check("A9 outside frame_range unchanged (<0.01°)", outside < 0.01, f"max={outside:.5f}°")

r = call("reapply", op_id=OP_A, overrides={"amount": 0.10})
opA2 = (r.get("data") or {})
tr = track_of(opA2) if opA2 else None
r2 = call("analyze_motion", chain=CHAIN, frame_range=FR_A, main_bone="upper_arm_fk.R")
post2 = (r2.get("data") or {}).get("main", {})
check("A10 reapply amount 0.15→0.10: same id, 1 strip, params merged, counter scales",
      r.get("ok") and opA2.get("id") == OP_A and tr is not None and len(tr.strips) == 1
      and opA2["params"].get("amount") == 0.10 and opA2["params"].get("lead") == LEAD
      and within(post2.get("counter_move_deg") or 0, 0.10 * amp_m)
      and (post2.get("counter_move_deg") or 99) < (post.get("counter_move_deg") or 0),
      f"counter {post.get('counter_move_deg')}→{post2.get('counter_move_deg')}° "
      f"(0.10×amp={0.10 * amp_m:.2f}) strips={[x.name for x in tr.strips] if tr else None}")

r = call("revert", op_id=OP_A)
q2 = sample(W_A)
back = max(float(P.qangle_deg(q0[b], q2[b]).max()) for b in BONES)
check("A11 revert restores the visible pose (<0.01°)", r.get("ok") and back < 0.01,
      f"max={back:.5f}°")

# =============================================================================
# F  follow_through   (uses analyze_motion's suggested args → pins path)
FR_S = [684, 745]
W_F = [FR_S[0] - 10, FR_S[1] + 10]
q0 = sample(W_F)
iS = FR_S[0] - W_F[0]
qs = {b: q0[b][iS:iS + FR_S[1] - FR_S[0] + 1] for b in BONES}
evf, spf = own_events(qs["forearm_fk.R"], FR_S[0])
of, pf, sf = evf["onset"], evf["peak"], evf["stop"]
qm = qs["forearm_fk.R"]
amp_f = float(P.qangle_deg(qm[of], qm[of:sf + 1]).max())
u, _ = P.motion_axis(qm[sf - 3], qm[sf])
print(f"FT own events: onset {FR_S[0] + of} peak {FR_S[0] + pf} ({evf['peak_speed']:.2f}°/帧) "
      f"stop {FR_S[0] + sf} amp {amp_f:.1f}°")
r = call("analyze_motion", chain=CHAIN, frame_range=FR_S, main_bone="forearm_fk.R")
sug = ((r.get("data") or {}).get("suggest") or {}).get("follow_through", {})
args = dict(sug.get("args") or {})
check("F0 analyze suggests a usable follow_through window",
      r.get("ok") and args.get("stop_frame") == FR_S[0] + sf
      and args.get("frame_range", [0, 0])[1] >= FR_S[0] + sf + 16 + 3
      and args.get("frame_range", [10 ** 6])[0] >= W_F[0],
      f"args={args} expect={sug.get('expect')}")
args["frame_range"] = [max(args["frame_range"][0], W_F[0] + 1),
                       min(args["frame_range"][1], W_F[1] - 1)]
FR_F = args["frame_range"]

r = call("follow_through", **args, propagate=2, dry_run=True)
check("F1 propagate past window end → actionable error",
      not r.get("ok") and "建议 frame_range" in (r.get("error") or {}).get("message", ""),
      (r.get("error") or {}).get("message", "")[:160])

n_strips = strips_now()
r = call("follow_through", **args, dry_run=True)
check("F2 dry_run writes nothing", r.get("ok") and r["data"].get("dry_run")
      and strips_now() == n_strips, f"strips {n_strips}->{strips_now()}")

r = call("follow_through", **args)
d = r.get("data") or {}
OP_F = d.get("op_id")
opF = agent_ops.get_op(data_dir, OP_F) if OP_F else None
check("F3 write via bridge: ok / logged / preview / agent_ names",
      r.get("ok") and opF is not None and op_in_log(OP_F) and d.get("status") == "preview"
      and opF["strip"].startswith("agent_") and opF["track"].startswith("agent_"),
      f"op={OP_F} strip={opF and opF['strip']} first_lobe={d.get('metrics', {}).get('first_lobe_deg')}")

q1 = sample(W_F)
k0 = iS + sf                                      # stop index in W_F samples
dd = rv_deg(q0["forearm_fk.R"], q1["forearm_fk.R"]) @ u
lob = lobes(dd[k0:])
first = lob[0][1] if lob else 0.0
second = abs(lob[1][1]) if len(lob) > 1 else 0.0
tF = 0.12 * amp_f
check("F4 ≥2 sign changes after stop, lobe2 < 0.7·lobe1, lobe1 ≈ 0.12·amp, first lobe forward",
      len(lob) >= 3 and first > 0 and second < 0.7 * first and within(first, tF),
      f"lobes={[(FR_S[0] + sf + i, round(v, 2)) for i, v in lob]} target={tF:.2f}°")
dF = {b: P.qangle_deg(q0[b], q1[b]) for b in BONES}
pre_stop = max(float(dF[b][:k0 + 1].max()) for b in BONES)
tail = max(float(dF[b][k0 + 16 + 1:].max()) for b in BONES)
check("F5 unchanged before stop (<0.1°) and after the oscillation (<0.5°)",
      pre_stop < 0.1 and tail < 0.5, f"pre_stop={pre_stop:.4f}° tail={tail:.4f}°")
moved = {b: round(float(dF[b].max()), 2) for b in BONES}
check("F6 Euler + quat bones all oscillate", all(v > 0.5 for v in moved.values()),
      f"max diff°={moved}")
iF0, iF1 = FR_F[0] - W_F[0], FR_F[1] - W_F[0]
outside = max(max(float(dF[b][:iF0].max()), float(dF[b][iF1 + 1:].max())) for b in BONES)
check("F6b outside frame_range unchanged (<0.01°)", outside < 0.01,
      f"window={FR_F} max={outside:.5f}°")

r = call("analyze_motion", chain=CHAIN, frame_range=FR_F, main_bone="forearm_fk.R",
         stop_frame=FR_S[0] + sf, onset_frame=FR_S[0] + of, baseline_op=OP_F)
vb = (((r.get("data") or {}).get("vs_baseline") or {}).get("bones") or {}).get("forearm_fk.R", {})
lb = vb.get("approach_lobes") or [[0, 0]]
check("F7 analyze baseline_op agrees with the independent measure",
      r.get("ok") and json_ok(r) and abs(lb[0][1] - first) < 0.5
      and vb.get("approach_sign_changes", 0) >= 2
      and vb.get("before_stop_max_deg", 9) < 0.1,
      f"vs_baseline lobes={lb} sc={vb.get('approach_sign_changes')} "
      f"changed={vb.get('changed_frames')} | summary: {r.get('summary')}")

r = call("reapply", op_id=OP_F, overrides={"amount": 0.06})
opF2 = r.get("data") or {}
tr = track_of(opF2) if opF2 else None
q2 = sample(W_F)
dd2 = rv_deg(q0["forearm_fk.R"], q2["forearm_fk.R"]) @ u
lob2 = lobes(dd2[k0:])
ratio = (lob2[0][1] / first) if (lob2 and first) else 0
check("F8 reapply amount 0.12→0.06: same id, 1 strip, lobe1 halves",
      r.get("ok") and opF2.get("id") == OP_F and tr is not None and len(tr.strips) == 1
      and opF2["params"].get("amount") == 0.06 and abs(ratio - 0.5) < 0.08,
      f"lobe1 {first:.2f}→{round(lob2[0][1], 2) if lob2 else None}° ratio={ratio:.3f}")

r = call("reapply", op_id=OP_F, overrides={      # +3 帧深度延迟 → 窗口末端也要 +3
    "amount": 0.12, "propagate": 1, "frame_range": [FR_F[0], FR_F[1] + 4]})
q3 = sample(W_F)
first_changed = {}
for b in BONES:
    dfb = P.qangle_deg(q0[b], q3[b])
    nz = np.nonzero(dfb > 0.05)[0]
    first_changed[b] = int(W_F[0] + nz[0]) if len(nz) else None
depth = {b: P.depth_in(rig, b, BONES) for b in BONES}
lag = (first_changed["hand_fk.R"] or 0) - (first_changed["shoulder.R"] or 0)
check("F9 propagate=1: deeper bones start later (hand − shoulder ≈ depth diff)",
      r.get("ok") and abs(lag - (depth["hand_fk.R"] - depth["shoulder.R"])) <= 1,
      f"first changed={first_changed} depth={depth}")

r = call("revert", op_id=OP_F)
q4 = sample(W_F)
back = max(float(P.qangle_deg(q0[b], q4[b]).max()) for b in BONES)
check("F10 revert restores the visible pose (<0.01°)", r.get("ok") and back < 0.01,
      f"max={back:.5f}°")

# =============================================================================
# O  overshoot   (own detection path: no pins)
r = call("overshoot", chain=CHAIN, frame_range=FR_S, main_bone="forearm_fk.R",
         amount=0.08, peak_after=2, settle=6)
d = r.get("data") or {}
OP_O = d.get("op_id")
opO = agent_ops.get_op(data_dir, OP_O) if OP_O else None
check("O1 write via bridge: ok / logged / preview / agent_ names",
      r.get("ok") and opO is not None and op_in_log(OP_O) and d.get("status") == "preview"
      and opO["strip"].startswith("agent_") and opO["track"].startswith("agent_")
      and d.get("metrics", {}).get("stop_frame") == FR_S[0] + sf,
      f"op={OP_O} metrics stop={d.get('metrics', {}).get('stop_frame')} "
      f"overshoot={d.get('metrics', {}).get('overshoot_deg')}")
q1 = sample(W_F)
dd = rv_deg(q0["forearm_fk.R"], q1["forearm_fk.R"]) @ u
lob = lobes(dd[k0:])
pk_i = int(np.argmax(dd[k0:]))
tO = 0.08 * amp_f
neg = float(-min(0.0, dd[k0:].min()))
check("O2 single forward lobe ≈ 0.08·amp peaking at stop+2",
      len(lob) == 1 and lob[0][1] > 0 and within(lob[0][1], tO) and abs(pk_i - 2) <= 1
      and neg < 0.1 * lob[0][1],
      f"lobes={[(FR_S[0] + sf + i, round(v, 2)) for i, v in lob]} peak@{FR_S[0] + sf + pk_i} "
      f"target={tO:.2f}° @ {FR_S[0] + sf + 2}")
dO = {b: P.qangle_deg(q0[b], q1[b]) for b in BONES}
tail = max(float(dO[b][k0 + 2 + 6:].max()) for b in BONES)
outside = max(max(float(dO[b][:iS].max()), float(dO[b][iS + len(spf):].max())) for b in BONES)
check("O3 back on the original pose after settle (<0.5°), outside window (<0.01°)",
      tail < 0.5 and outside < 0.01, f"tail={tail:.4f}° outside={outside:.5f}°")
moved = {b: round(float(dO[b].max()), 2) for b in BONES}
check("O4 Euler + quat bones all overshoot", all(v > 0.3 for v in moved.values()),
      f"max diff°={moved}")
r = call("analyze_motion", chain=CHAIN, frame_range=FR_S, main_bone="forearm_fk.R",
         stop_frame=FR_S[0] + sf, onset_frame=FR_S[0] + of, baseline_op=OP_O)
vb = (((r.get("data") or {}).get("vs_baseline") or {}).get("bones") or {}).get("forearm_fk.R", {})
check("O5 analyze baseline_op: overshoot peak agrees, single lobe",
      r.get("ok") and abs((vb.get("approach_peak_deg") or 0) - lob[0][1]) < 0.5
      and vb.get("approach_peak_frame") == FR_S[0] + sf + pk_i
      and vb.get("approach_sign_changes") == 0,
      f"peak={vb.get('approach_peak_deg')}° @ {vb.get('approach_peak_frame')} "
      f"sc={vb.get('approach_sign_changes')}")
r = agent_ops.reapply(data_dir, rig, OP_O, scene=scene, amount=0.04)
tr = track_of(r)
q2 = sample(W_F)
pk2 = float((rv_deg(q0["forearm_fk.R"], q2["forearm_fk.R"]) @ u)[k0:].max())
check("O6 reapply amount 0.08→0.04 (agent_ops.reapply): same id, 1 strip, peak halves",
      r.get("id") == OP_O and tr is not None and len(tr.strips) == 1
      and abs(pk2 / lob[0][1] - 0.5) < 0.08,
      f"peak {lob[0][1]:.2f}→{pk2:.2f}°")
n_strips = strips_now()
r = call("overshoot", chain=CHAIN, frame_range=FR_S, main_bone="forearm_fk.R",
         amount=0.2, dry_run=True)
check("O7 dry_run writes nothing", r.get("ok") and r["data"].get("dry_run")
      and strips_now() == n_strips, f"strips {n_strips}->{strips_now()}")

# params panel mirror sees the new tools' TUNABLE keys
agent_bridge._sync_params_list(settings, data_dir)
keys = {(p.op_id, p.key) for p in settings.agent_params}
check("O8 params panel mirrors overshoot knobs",
      (OP_O, "amount") in keys and (OP_O, "settle") in keys,
      f"{sorted(k for o_, k in keys if o_ == OP_O)}")
r = call("revert", op_id=OP_O)
q3 = sample(W_F)
back = max(float(P.qangle_deg(q0[b], q3[b]).max()) for b in BONES)
check("O9 revert restores the visible pose (<0.01°)", r.get("ok") and back < 0.01,
      f"max={back:.5f}°")

# ---------- T: typo'd parameters are rejected with the closest legal name -------------
r = call("anticipation", chain=CHAIN, frame_range=list(FR_A), amout=0.15, dry_run=True)
check("T1 anticipation(amout=…) → rejected, suggests amount",
      not r["ok"] and "amout" in r["error"]["message"] and "amount" in r["error"]["message"],
      r.get("error", {}).get("message", "")[:160])
r = call("analyze_motion", chain=CHAIN, frame_range=list(FR_A), main_bon="forearm_fk.R")
check("T2 analyze_motion(main_bon=…) → rejected, suggests main_bone",
      not r["ok"] and "main_bone" in r["error"]["message"],
      r.get("error", {}).get("message", "")[:160])
r = call("analyze_motion", chain=CHAIN, frame_range=list(FR_A), brief=True)
check("T3 shell-level params (brief) still accepted", r["ok"], r.get("summary", "")[:80])

# ---------- summary -----------------------------------------------------------
fails = [r for r in RESULTS if not r[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for name, _ok, detail in fails:
    print(f"FAIL {name}: {detail}")
sys.exit(1 if fails else 0)
