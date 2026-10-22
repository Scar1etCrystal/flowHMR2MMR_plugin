"""Headless validation of motion_copy + compare_motion (plugin agent_copy).

Every number is re-measured independently: own frame_set sweep reading
pose_bone.matrix_basis / pose_bone.matrix, own float64 quaternion math, and
agent_anatomy.probe for the palm geometry - never the solver's intermediates.

  A  bridge registration: tools / WRITE_SCOPES / TUNABLE / REAPPLY
  B  time shift (local): upper arm + forearm (Euler XYZ) + hand (quat)
  C  overlapping src/dst windows on the same bones (sample-before-write)
  D  mirror local arm.L → arm.R: compare_motion(mirror) + palms mirror in
     the chest frame (re-probed anatomy); revert restores arm.R
  E  mirror world: palms satisfy world X mirror (x → −x); armature-space
     R(arm.R) = S·R(arm.L)·F; world time shift / world add (no mirror)
  F  dst_range time scaling: target ends = source ends, interior = slerp
  G  mode=add = dst ⊗ conj(src(a)) ⊗ src(t) computed independently
  H  dry_run writes nothing
  I  reapply: dst_start / frame_range / dst_start → same op_id, one strip,
     new window live, old window restored, op.frames follows
  J  params panel sync;  M  channels rot+loc (foot_ik positions mirror) +
     IK-mode note;  K  revert all → baseline;  L  error paths
"""
import math
import os
import sys

import bpy
import addon_utils
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import (  # noqa: E402
    agent_anatomy, agent_bridge, agent_ops)

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
data_dir = os.path.join(os.path.dirname(bpy.data.filepath),
                        "e2e_motion_copy_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)


def call(tool, **args):                       # 走桥：顺便验证注册链
    return agent_bridge._dispatch({"tool": tool, "args": args})


# ---------------------------------------------------------------------------
# independent float64 helpers (no agent_pose / agent_copy math)

def qn(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qmul(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def qconj(q):
    q = np.array(q, dtype=np.float64)
    q[..., 1:] *= -1
    return q


def qang(a, b):
    """geodesic angle (deg), sign-agnostic, atan2 form (exact near 0)."""
    r = qmul(qconj(qn(a)), qn(b))
    v = np.linalg.norm(r[..., 1:], axis=-1)
    return np.degrees(2.0 * np.arctan2(v, np.abs(r[..., 0])))


def qslerp(q0, q1, t):
    q0, q1 = qn(q0), qn(q1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = min(1.0, float(np.dot(q0, q1)))
    th = math.acos(d)
    if th < 1e-9:
        return q0
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def vang(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def rortho(m):
    u, _s, vt = np.linalg.svd(np.asarray(m, dtype=np.float64))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def mang(a, b):
    """angle (deg) between two rotation matrices: orthonormalized float64,
    atan2(|axis part|, cos part) - exact near 0 (acos of the trace is not)."""
    r = rortho(a).T @ rortho(b)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return math.degrees(math.atan2(np.linalg.norm(v) / 2.0,
                                   (np.trace(r) - 1.0) / 2.0))


def sample(bones, frames):
    """own sweep: {bone: {frame: (wxyz basis, 3x3 armature rot, head)}}"""
    out = {b: {} for b in bones}
    keep = scene.frame_current
    for f in frames:
        scene.frame_set(int(f))
        for b in bones:
            pb = rig.pose.bones[b]
            q = pb.matrix_basis.to_quaternion()
            out[b][int(f)] = (np.array([q.w, q.x, q.y, q.z], dtype=np.float64),
                              np.array(pb.matrix.to_3x3(), dtype=np.float64),
                              np.array(pb.head, dtype=np.float64))
    scene.frame_set(keep)
    return out


def max_err(cur, ref, bones, frames):
    """max angle (deg) between two samples over bones × frames."""
    worst, where = 0.0, None
    for b in bones:
        for f in frames:
            e = float(qang(cur[b][f][0], ref[b][f][0]))
            if e > worst:
                worst, where = e, (b, f)
    return worst, where


def strip_count():
    anim = rig.animation_data
    return sum(len(t.strips) for t in anim.nla_tracks) if anim else 0


def is_live(op_id):
    return any(o["id"] == op_id and o["status"] in ("preview", "committed")
               for o in agent_ops.list_ops(data_dir))


ARM_L = ["shoulder.L", "upper_arm_fk.L", "forearm_fk.L", "hand_fk.L"]
FING_L = [f"{f}.0{i}.L" for f in ("f_index", "f_middle", "f_ring", "f_pinky",
                                  "thumb") for i in (1, 2, 3)]
CHAIN_L = ARM_L + FING_L
CHAIN_R = [b[:-2] + ".R" for b in CHAIN_L]
TRIO_L = ["upper_arm_fk.L", "forearm_fk.L", "hand_fk.L"]   # Euler, Euler, quat
SRC = [405, 450]                  # 探查脚本选的段：起止较慢、中段峰值 ~16°/帧
N = SRC[1] - SRC[0] + 1
BLEND = 4
S = np.diag([-1.0, 1.0, 1.0])

print("rig", rig.name, "modes",
      {b: rig.pose.bones[b].rotation_mode for b in ARM_L + ["hand_fk.R"]})

# baseline (before ANY write)
W_SRC = list(range(395, 471))
W_600 = list(range(590, 656))
W_700 = list(range(690, 861))
W_900 = list(range(890, 956))
W_1000 = list(range(990, 1079))
ALL_W = W_SRC + W_600 + W_700 + W_900 + W_1000
base = sample(CHAIN_L + CHAIN_R + ["spine_fk.003"], ALL_W)

# ---------- A: registration ---------------------------------------------------
pg = call("ping")
tools = pg.get("data", {}).get("tools", [])
check("A1 bridge lists tools, plugin loads clean",
      pg["ok"] and "motion_copy" in tools and "compare_motion" in tools
      and "agent_copy" not in (pg.get("data", {}).get("plugin_errors") or {}),
      f"plugin_errors={pg.get('data', {}).get('plugin_errors')}")
scope = agent_bridge.WRITE_SCOPES["motion_copy"](
    agent_bridge._ctx(), {"chain": "arm.L", "src_range": SRC,
                          "dst_start": SRC[0], "mirror": True})
check("A2 WRITE_SCOPES = target bones × target window",
      len(scope) == 1 and scope[0][0] == CHAIN_R
      and tuple(scope[0][1]) == tuple(SRC),
      f"bones={scope[0][0][:4]}… window={scope[0][1]}")
check("A3 TUNABLE / REAPPLY registered",
      "motion_copy" in agent_ops.TUNABLE_PARAMS
      and "motion_copy" in agent_ops.REAPPLY_HANDLERS,
      f"keys={[s['key'] for s in agent_ops.TUNABLE_PARAMS.get('motion_copy', [])]}")

# ---------- B: time shift, local ----------------------------------------------
r = call("motion_copy", bones=["left_upper_arm", "left_forearm", "left_hand"],
         src_range=SRC, dst_start=700)
dB = r.get("data") or {}
OP_B = dB.get("op_id")
opB = agent_ops.get_op(data_dir, OP_B) if OP_B else None
check("B1 write via bridge: ok, preview, agent_ strip/track, in list_ops",
      r["ok"] and opB is not None and opB["status"] == "preview"
      and opB["strip"].startswith("agent_") and opB["track"].startswith("agent_")
      and opB["frames"] == [700, 745],
      f"op={OP_B} strip={opB and opB['strip']} track={opB and opB['track']} "
      f"err={r.get('error')}")
mB = (opB or {}).get("metrics", {})
check("B2 metrics mirror_map / time_scale / rot modes",
      mB.get("mirror_map") == {b: b for b in TRIO_L}
      and mB.get("time_scale") == 1.0
      and mB["bones"]["upper_arm_fk.L"]["rot_mode"] == "XYZ"
      and mB["bones"]["hand_fk.L"]["rot_mode"] == "QUATERNION",
      f"map={mB.get('mirror_map')} ts={mB.get('time_scale')} "
      f"modes={ {b: v.get('rot_mode') for b, v in mB.get('bones', {}).items()} }")
cm = call("compare_motion", a={"bones": TRIO_L, "frame_range": SRC},
          b={"frame_range": [700, 745]}, trim=BLEND)
check("B3 compare_motion err_inner < 0.05°",
      cm["ok"] and cm["data"]["err_inner_deg"] < 0.05,
      cm.get("summary") or cm.get("error"))
cur = sample(TRIO_L, range(700, 746))
inner = range(BLEND, N - BLEND)
eB = max(float(qang(cur[b][700 + i][0], base[b][SRC[0] + i][0]))
         for b in TRIO_L for i in inner)
check("B4 independent re-sample: target inner = source (<0.05°)", eB < 0.05,
      f"max={eB:.5f}°")
cur = sample(CHAIN_L + CHAIN_R, list(range(690, 700)) + list(range(746, 756)))
eo, wo = max_err(cur, base, TRIO_L, list(range(690, 700)) + list(range(746, 756)))
cur2 = sample(CHAIN_L + CHAIN_R, range(700, 746))
others = [b for b in CHAIN_L + CHAIN_R if b not in TRIO_L]
ei, wi = max_err(cur2, base, others, range(700, 746))
check("B5 outside window / untouched bones unchanged (<0.01°)",
      eo < 0.01 and ei < 0.01, f"outside={eo:.5f}° at {wo}; others={ei:.5f}° at {wi}")

# ---------- C: overlapping windows on the same bones ---------------------------
r = call("motion_copy", bones=["forearm_fk.L", "hand_fk.L"], src_range=SRC,
         dst_start=415)
OP_C = (r.get("data") or {}).get("op_id")
cur = sample(["forearm_fk.L", "hand_fk.L"], range(415, 461))
eC = max(float(qang(cur[b][415 + i][0], base[b][SRC[0] + i][0]))
         for b in ("forearm_fk.L", "hand_fk.L") for i in inner)
check("C1 overlap src/dst: target inner = PRE-write source (<0.05°)",
      r["ok"] and eC < 0.05,
      f"max={eC:.5f}° notes={(agent_ops.get_op(data_dir, OP_C) or {}).get('metrics', {}).get('notes')}")
call("revert", op_id=OP_C)
cur = sample(["forearm_fk.L", "hand_fk.L"], W_SRC)
eCr, wCr = max_err(cur, base, ["forearm_fk.L", "hand_fk.L"], W_SRC)
check("C2 revert overlap → source window restored (<0.01°)", eCr < 0.01,
      f"max={eCr:.5f}° at {wCr}")

# ---------- D: mirror local arm.L → arm.R --------------------------------------
r = call("motion_copy", chain="arm.L", src_range=SRC, dst_start=SRC[0],
         mirror=True)
OP_D = (r.get("data") or {}).get("op_id")
mD = (agent_ops.get_op(data_dir, OP_D) or {}).get("metrics", {})
check("D1 mirror write ok, mirror_map arm.L→arm.R (19 bones)",
      r["ok"] and mD.get("mirror_map") == dict(zip(CHAIN_L, CHAIN_R)),
      f"n={len(mD.get('mirror_map', {}))} err={r.get('error')}")
cm = call("compare_motion", a={"chain": "arm.L", "frame_range": SRC},
          b={"chain": "arm.R", "frame_range": SRC}, mirror=True, trim=BLEND)
cmb = (cm.get("data") or {}).get("bones", {})
check("D2 compare_motion(mirror) err_inner < 0.05° (Euler YXZ/XYZ + quats)",
      cm["ok"] and cm["data"]["err_inner_deg"] < 0.05,
      f"{cm.get('summary')} | shoulder={cmb.get('shoulder.L', {}).get('err_inner_deg')} "
      f"upper={cmb.get('upper_arm_fk.L', {}).get('err_inner_deg')} "
      f"hand={cmb.get('hand_fk.L', {}).get('err_inner_deg')}")
# palms: world → chest (spine_fk.003) local frame; chest-frame mirror
Rc_rest = np.array(rig.data.bones["spine_fk.003"].matrix_local.to_3x3())
Mc = Rc_rest.T @ S @ Rc_rest
Rw = np.array(rig.matrix_world.to_3x3())
PROBE_F = list(range(SRC[0] + BLEND + 1, SRC[1] - BLEND, 6))
chest = sample(["spine_fk.003"], PROBE_F)


def palms(f):
    pl = agent_anatomy.probe(scene, rig, part="palm", side="L", frame_range=[f, f])
    pr = agent_anatomy.probe(scene, rig, part="palm", side="R", frame_range=[f, f])
    return (np.linalg.inv(Rw) @ np.array(pl["world_dir"]),
            np.linalg.inv(Rw) @ np.array(pr["world_dir"]),
            min(pl["confidence"], pr["confidence"]))


errs, confs, before = [], [], []
for f in PROBE_F:
    vl, vr, cf = palms(f)
    Rch = chest["spine_fk.003"][f][1]
    cl, cr = Rch.T @ vl, Rch.T @ vr
    errs.append(vang(cr, Mc @ cl))
    confs.append(cf)
check("D3 palms mirror in chest frame (re-probed anatomy) < 2°",
      max(errs) < 2.0,
      f"max={max(errs):.3f}° per_frame={[round(e, 3) for e in errs]} "
      f"min_conf={min(confs)} frames={PROBE_F}")
cur = sample(CHAIN_L, range(SRC[0], SRC[1] + 1))
eDs, _ = max_err(cur, base, CHAIN_L, range(SRC[0], SRC[1] + 1))
check("D4 source arm.L untouched (<0.01°)", eDs < 0.01, f"max={eDs:.5f}°")
call("revert", op_id=OP_D)
cur = sample(CHAIN_R, W_SRC)
eDr, wDr = max_err(cur, base, CHAIN_R, W_SRC)
check("D5 revert → arm.R back to pre-write (<0.01°)", eDr < 0.01,
      f"max={eDr:.5f}° at {wDr}")

# ---------- E: mirror world --------------------------------------------------
r = call("motion_copy", chain="arm.L", src_range=SRC, dst_start=SRC[0],
         mirror=True, space="world")
OP_E = (r.get("data") or {}).get("op_id")
werrs = []
for f in PROBE_F:
    pl = agent_anatomy.probe(scene, rig, part="palm", side="L", frame_range=[f, f])
    pr = agent_anatomy.probe(scene, rig, part="palm", side="R", frame_range=[f, f])
    vl, vr = np.array(pl["world_dir"]), np.array(pr["world_dir"])
    werrs.append(vang(vr, vl * np.array([-1.0, 1.0, 1.0])))
check("E1 world mirror: palm_R = X-mirror(palm_L) in world < 2°",
      r["ok"] and max(werrs) < 2.0,
      f"max={max(werrs):.3f}° per_frame={[round(e, 3) for e in werrs]} "
      f"err={r.get('error')}")
cm = call("compare_motion", a={"chain": "arm.L", "frame_range": SRC},
          b={"chain": "arm.R", "frame_range": SRC}, mirror=True, space="world",
          trim=BLEND)
check("E2 compare_motion(world, mirror) err_inner < 0.05°",
      cm["ok"] and cm["data"]["err_inner_deg"] < 0.05, cm.get("summary"))
# independent: armature-space rotations of arm.R = S·R_L·F
cur = sample(CHAIN_L + CHAIN_R, range(SRC[0] + BLEND, SRC[1] - BLEND + 1))
eW, wW = 0.0, None
for bl, br in zip(CHAIN_L, CHAIN_R):
    rsl = np.array(rig.data.bones[bl].matrix_local.to_3x3())
    rsr = np.array(rig.data.bones[br].matrix_local.to_3x3())
    F = rsl.T @ S @ rsr
    for f in range(SRC[0] + BLEND, SRC[1] - BLEND + 1):
        e = mang(S @ rortho(cur[bl][f][1]) @ F, cur[br][f][1])
        if e > eW:
            eW, wW = e, (br, f)
check("E3 independent: R_arm(R) = S·R_arm(L)·F for all 19 bones (<0.05°)",
      eW < 0.05, f"max={eW:.5f}° at {wW} link(upper_arm_fk.R)="
      f"{(agent_ops.get_op(data_dir, OP_E) or {}).get('metrics', {}).get('world_parent_link', {}).get('upper_arm_fk.R')}")
call("revert", op_id=OP_E)

# world, no mirror, same bones → armature orientation of the source time
r = call("motion_copy", bones=TRIO_L, src_range=SRC, dst_start=600, space="world")
OP_E4 = (r.get("data") or {}).get("op_id")
cur = sample(TRIO_L, range(600, 646))
eW4 = max(mang(base[b][SRC[0] + i][1], cur[b][600 + i][1])
          for b in TRIO_L for i in inner)
check("E4 world time shift: armature rotation = source time's (<0.05°)",
      r["ok"] and eW4 < 0.05, f"max={eW4:.5f}° err={r.get('error')}")
call("revert", op_id=OP_E4)
# world add: R_after(t') = [R_src(τ)·R_src(a)⁻¹]·R_before(t')
r = call("motion_copy", bones=TRIO_L, src_range=SRC, dst_start=600,
         space="world", mode="add")
OP_E5 = (r.get("data") or {}).get("op_id")
cur = sample(TRIO_L, range(600, 646))
eW5 = 0.0
for b in TRIO_L:
    ra = rortho(base[b][SRC[0]][1])
    for i in inner:
        want = rortho(base[b][SRC[0] + i][1]) @ ra.T @ rortho(base[b][600 + i][1])
        eW5 = max(eW5, mang(want, cur[b][600 + i][1]))
check("E5 world add: [R_src(t)·R_src(a)⁻¹]·R_before (<0.05°)",
      r["ok"] and eW5 < 0.05, f"max={eW5:.5f}° err={r.get('error')}")
call("revert", op_id=OP_E5)

# ---------- F: dst_range time scaling ------------------------------------------
TS_DST = [1000, 1068]
r = call("motion_copy", bones=["forearm_fk.L", "hand_fk.L"], src_range=SRC,
         dst_range=TS_DST, blend=0)
OP_F = (r.get("data") or {}).get("op_id")
mF = (agent_ops.get_op(data_dir, OP_F) or {}).get("metrics", {})
ts_want = (TS_DST[1] - TS_DST[0]) / (SRC[1] - SRC[0])
cur = sample(["forearm_fk.L", "hand_fk.L"], range(TS_DST[0], TS_DST[1] + 1))
eEnds = max(max(float(qang(cur[b][TS_DST[0]][0], base[b][SRC[0]][0])),
                float(qang(cur[b][TS_DST[1]][0], base[b][SRC[1]][0])))
            for b in ("forearm_fk.L", "hand_fk.L"))
check("F1 time scale: target ends = source ends (<0.05°), time_scale metric",
      r["ok"] and eEnds < 0.05 and abs(mF.get("time_scale", 0) - ts_want) < 1e-3,
      f"ends={eEnds:.5f}° time_scale={mF.get('time_scale')} (want {ts_want:.4f})")
eMid = 0.0
for b in ("forearm_fk.L", "hand_fk.L"):
    for f in range(TS_DST[0], TS_DST[1] + 1):
        tau = SRC[0] + (f - TS_DST[0]) * (SRC[1] - SRC[0]) / (TS_DST[1] - TS_DST[0])
        i0 = min(int(math.floor(tau)), SRC[1] - 1)
        want = qslerp(base[b][i0][0], base[b][i0 + 1][0], tau - i0)
        eMid = max(eMid, float(qang(cur[b][f][0], want)))
check("F2 time scale: every target frame = slerp(source, τ) (<0.05°)",
      eMid < 0.05, f"max={eMid:.5f}°")
cm = call("compare_motion", a={"bones": ["forearm_fk.L", "hand_fk.L"],
                               "frame_range": SRC},
          b={"frame_range": TS_DST}, trim=0)
print("    compare_motion a=src b=dst (double interpolation):", cm.get("summary"))
vargs = mF.get("verify", {}).get("args", {})
cm = call("compare_motion", **vargs)
check("F4 metrics.verify (a=dst, b=src) verifies the scaled copy (<0.05°)",
      cm["ok"] and cm["data"]["err_inner_deg"] < 0.05,
      f"{cm.get('summary')} args={vargs}")
cm2 = call("compare_motion", op_id=r["data"]["op_id"])
check("F5 compare_motion(op_id) == compare_motion(metrics.verify.args)",
      cm["ok"] and cm2["ok"] and cm2["data"]["err_inner_deg"] == cm["data"]["err_inner_deg"]
      and cm2["data"]["pairs"] == cm["data"]["pairs"],
      f"{cm2.get('summary') or cm2.get('error')}")
cur = sample(["forearm_fk.L", "hand_fk.L"], list(range(990, 1000)) + list(range(1069, 1079)))
eFo, _ = max_err(cur, base, ["forearm_fk.L", "hand_fk.L"],
                 list(range(990, 1000)) + list(range(1069, 1079)))
check("F3 outside scaled window unchanged (<0.01°)", eFo < 0.01, f"max={eFo:.5f}°")

# ---------- G: mode=add ---------------------------------------------------------
r = call("motion_copy", bones=TRIO_L, src_range=SRC, dst_start=900, mode="add")
OP_G = (r.get("data") or {}).get("op_id")
cur = sample(TRIO_L, range(900, 946))
eG = 0.0
for b in TRIO_L:
    for i in inner:
        want = qmul(qmul(base[b][900 + i][0], qconj(base[b][SRC[0]][0])),
                    base[b][SRC[0] + i][0])
        eG = max(eG, float(qang(cur[b][900 + i][0], want)))
check("G1 add: dst ⊗ conj(src(a)) ⊗ src(t) (<0.05°)", r["ok"] and eG < 0.05,
      f"max={eG:.5f}° err={r.get('error')}")

# ---------- H: dry_run ----------------------------------------------------------
n_strips, n_ops = strip_count(), len(agent_ops.list_ops(data_dir))
r = call("motion_copy", chain="arm.L", src_range=SRC, dst_range=[600, 660],
         mirror=True, dry_run=True)
dd = r.get("data") or {}
check("H1 dry_run: no strip, no op, metrics carry mirror_map/time_scale",
      r["ok"] and dd.get("dry_run") is True and strip_count() == n_strips
      and len(agent_ops.list_ops(data_dir)) == n_ops
      and dd["metrics"]["mirror_map"].get("hand_fk.L") == "hand_fk.R"
      and abs(dd["metrics"]["time_scale"] - 60 / 45) < 1e-3,
      f"strips {n_strips}->{strip_count()} ts={dd.get('metrics', {}).get('time_scale')}")

# ---------- I: reapply ----------------------------------------------------------
op0 = agent_ops.get_op(data_dir, OP_B)
opI = agent_ops.reapply(data_dir, rig, OP_B, scene=scene, dst_start=760)
tr = next((t for t in rig.animation_data.nla_tracks if t.name == opI["track"]), None)
check("I1 reapply dst_start: same op_id/track, one strip, op.frames follows",
      opI["id"] == OP_B and opI["track"] == op0["track"] and tr is not None
      and len(tr.strips) == 1 and tr.strips[0].name == opI["strip"]
      and opI["frames"] == [760, 805] and opI["params"]["dst_start"] == 760
      and opI["params"]["frame_range"] == [760, 805],
      f"frames={opI['frames']} strips={[s.name for s in (tr.strips if tr else [])]} "
      f"params.dst_range={opI['params'].get('dst_range')}")
cur = sample(TRIO_L, list(range(760, 806)) + list(range(700, 746)))
eI = max(float(qang(cur[b][760 + i][0], base[b][SRC[0] + i][0]))
         for b in TRIO_L for i in inner)
eIo, wIo = max_err(cur, base, TRIO_L, range(700, 746))
check("I2 new position live (<0.05°), old window back to original (<0.01°)",
      eI < 0.05 and eIo < 0.01, f"new={eI:.5f}° old={eIo:.5f}° at {wIo}")
r = call("reapply", op_id=OP_B, overrides={"frame_range": [760, 827]})
opI2 = agent_ops.get_op(data_dir, OP_B)
check("I3 reapply frame_range (panel range) → time-scaled into new window",
      r["ok"] and opI2["frames"] == [760, 827]
      and abs(opI2["metrics"]["time_scale"] - 67 / 45) < 1e-3
      and opI2["params"]["dst_start"] == 760,
      f"frames={opI2['frames']} ts={opI2['metrics'].get('time_scale')} err={r.get('error')}")
opI3 = agent_ops.reapply(data_dir, rig, OP_B, scene=scene, dst_start=780)
tr = next((t for t in rig.animation_data.nla_tracks if t.name == opI3["track"]), None)
check("I4 reapply dst_start again keeps length (scaled) → [780, 847], 1 strip",
      opI3["frames"] == [780, 847] and tr is not None and len(tr.strips) == 1
      and abs(opI3["metrics"]["time_scale"] - 67 / 45) < 1e-3,
      f"frames={opI3['frames']} ts={opI3['metrics'].get('time_scale')}")

# ---------- J: params panel sync ------------------------------------------------
agent_bridge._sync_params_list(settings, data_dir)
keys = {p.key for p in settings.agent_params if p.op_id == OP_B}
check("J1 params mirrored for motion_copy",
      {"strength", "blend", "dst_start", "frame_range"} <= keys,
      f"keys={sorted(keys)}")

# ---------- M: channels rot+loc (foot_ik carries location keys) ----------------
FEET = ["foot_ik.L", "foot_ik.R"]
FR_SRC = list(range(SRC[0], SRC[1] + 1))
FR_IN = FR_SRC[BLEND:N - BLEND]
pre = sample(FEET, FR_SRC)
r = call("motion_copy", bones=["left_foot"], src_range=SRC, dst_start=SRC[0],
         mirror=True, channels="rot+loc")
OP_M = (r.get("data") or {}).get("op_id")
cur = sample(FEET, FR_SRC)
eP = max(float(np.linalg.norm(cur["foot_ik.R"][f][2] - S @ cur["foot_ik.L"][f][2]))
         for f in FR_IN)
moved = max(float(np.linalg.norm(cur["foot_ik.R"][f][2] - pre["foot_ik.R"][f][2]))
            for f in FR_IN)
check("M1 rot+loc mirror foot_ik.L→R: armature head X-mirrored (<1 mm)",
      r["ok"] and eP < 1e-3 and moved > 0.05,
      f"mirror_err={eP * 1000:.4f} mm (foot_ik.R moved {moved:.3f} m) err={r.get('error')}")
cm = call("compare_motion", a={"bones": ["left_foot"], "frame_range": SRC},
          mirror=True, channels="rot+loc", trim=BLEND)
check("M2 compare_motion(rot+loc, mirror): rot < 0.05°, loc < 0.1 mm",
      cm["ok"] and cm["data"]["err_inner_deg"] < 0.05
      and cm["data"]["loc_err_inner_m"] < 1e-4,
      f"{cm.get('summary')} loc_err_inner={cm.get('data', {}).get('loc_err_inner_m')} m")
call("revert", op_id=OP_M)
cur = sample(FEET, FR_SRC)
eMr = max(max(float(qang(cur[b][f][0], pre[b][f][0])),
              1000 * float(np.linalg.norm(cur[b][f][2] - pre[b][f][2])))
          for b in FEET for f in FR_SRC)
check("M3 revert → feet back (<0.01° / <0.01 mm)", eMr < 0.01, f"max={eMr:.5f}")
r = call("motion_copy", chain="leg.L", src_range=SRC, dst_start=600, dry_run=True)
notes = ((r.get("data") or {}).get("metrics") or {}).get("notes") or []
check("M4 FK leg on an IK-mode limb → note says writes are invisible",
      r["ok"] and any("IK 模式" in n for n in notes), notes)

# ---------- K: revert all → baseline ---------------------------------------------
for oid in (OP_B, OP_F, OP_G):
    if oid and is_live(oid):
        call("revert", op_id=oid)
cur = sample(CHAIN_L + CHAIN_R, ALL_W)
eK, wK = max_err(cur, base, CHAIN_L + CHAIN_R, ALL_W)
live = [o["id"] for o in agent_ops.list_ops(data_dir)
        if o["status"] in ("preview", "committed")]
check("K1 revert all → every window back to pre-write (<0.01°)",
      eK < 0.01 and not live, f"max={eK:.5f}° at {wK} live={live}")

# ---------- N: dry_run hands back the 'before' baseline ---------------------------------
r = call("motion_copy", chain="arm.L", src_range=[630, 680], dst_start=1430, dry_run=True)
vb = ((r.get("data") or {}).get("metrics") or {}).get("verify") or {}
cmb = call("compare_motion", **vb.get("args", {})) if vb.get("args") else {"ok": False}
check("N1 motion_copy dry_run returns err_inner_before_deg == compare_motion(verify.args) before writing",
      r["ok"] and cmb["ok"] and vb.get("err_inner_before_deg") == cmb["data"]["err_inner_deg"]
      and vb["err_inner_before_deg"] > 1.0,
      f"before={vb.get('err_inner_before_deg')} compare={cmb.get('data', {}).get('err_inner_deg')}")

# ---------- L: error paths -------------------------------------------------------
r = call("motion_copy", bones=["left_hand"], src_range=SRC)
check("L1 missing target → actionable error",
      not r["ok"] and "dst_start" in r["error"]["message"],
      r.get("error", {}).get("message"))
r = call("motion_copy", bones=["left_hand"], src_range=SRC, dst_start=700,
         mirorr=True)
check("L2 typo'd arg rejected (no silent non-mirror copy)",
      not r["ok"] and "不认识参数" in r["error"]["message"],
      r.get("error", {}).get("message"))

r = call("compare_motion", op_id="no_such_op")
check("L3 compare_motion(op_id) on an unknown op → actionable error",
      not r["ok"] and "不存在" in r["error"]["message"], r.get("error", {}).get("message"))
add_op = next((o for o in agent_ops.list_ops(data_dir)
               if o.get("tool") == "motion_copy" and (o.get("params") or {}).get("mode") == "add"), None)
if add_op is not None:
    r = call("compare_motion", op_id=add_op["id"])
    check("L4 compare_motion(op_id) on a mode=add copy → explains why not",
          not r["ok"] and "add" in r["error"]["message"], r.get("error", {}).get("message"))

# ---------- summary --------------------------------------------------------------
fails = [r for r in RESULTS if not r[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for n, _o, d in fails:
    print(f"FAIL {n}: {d}")
sys.exit(1 if fails else 0)
