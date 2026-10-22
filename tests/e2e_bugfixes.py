"""Bug-fix regressions found during the 2026-10-03 session (measured, not assumed).

  1. clean_jitter(bone=quaternion bone) wrote per-component quaternion
     differences into a Combine strip → ~150-180° garbage rotations and
     20-120× MORE jitter.  Now: true quaternion delta, small change, less jitter.
  2. clean_jitter(bone=Euler arm bone) skipped the rotation (built
     rotation_quaternion paths that don't exist) → only location smoothed.
  3. hold_pose strength applied twice (<1 → strength², >1 → no effect).
  4. hold_pose refused Euler bones (upper_arm/forearm/shoulder) - the owner
     probe_anatomy elbow_front hands out.
  5. set_influence ignored Euler / location deltas.
"""
import os
import sys

import addon_utils
import bpy
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import (  # noqa: E402
    agent_anatomy, agent_bridge, agent_ops, agent_pose as P)
from mathutils import Matrix, Vector  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig or next(o for o in scene.objects if o.type == "ARMATURE"
                               and o.name.startswith("RIG-"))
settings.mmr_rig = rig
data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_bugfixes_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)


def call(tool, **args):
    r = agent_bridge._dispatch({"tool": tool, "args": args})
    if not r["ok"]:
        print("ERR", tool, r.get("error"), r.get("trace"))
    return r


A, B = 150, 224
FR = list(range(A, B + 1))
INNER = slice(6, -6)


def jitter(q):
    sp = P.angular_speed_deg(q)
    return float(np.abs(np.diff(sp, 2)).mean())


# ---- 1/2: clean_jitter on quaternion + Euler bones --------------------------
for role, bone in (("left_hand", "hand_fk.L"), ("spine2", "spine_fk.001"),
                   ("left_forearm", "forearm_fk.L")):
    smp_b = P.sample_visible(scene, rig, [bone], FR)
    before = smp_b["quat"][bone]
    eul_before = smp_b["euler"].get(bone)
    r = call("clean_jitter", frame_range=[A, B], bone=role, strength=1.0, width=5)
    after = P.sample_visible(scene, rig, [bone], FR)["quat"][bone]
    ch = P.qangle_deg(before, after)
    jb, ja = jitter(before[INNER]), jitter(after[INNER])
    # 独立重算：对可见四元数（符号连续）逐分量零相位平滑后归一化——内段
    # （两端 taper 外）Blender 求值结果必须等于它。旧 bug 下这里差 ~150°。
    from bl_ext.user_default.mocap_doctor.core import agent_fx
    if P.rot_mode(rig.pose.bones[bone]) == "QUATERNION":
        q = P.quat_continuous(before)
        sm = np.stack([agent_fx.clean_jitter_values(q[:, c], strength=1.0, width=5,
                                                    blend=4) for c in range(4)], axis=1)
        sm = P.quat_normalize(sm)
    else:   # Euler 骨：Combine 下 Euler 相加，平滑的是 Euler 三通道本身
        order = P.rot_mode(rig.pose.bones[bone])
        eb = eul_before
        es = np.stack([agent_fx.clean_jitter_values(eb[:, c], strength=1.0, width=5,
                                                    blend=4) for c in range(3)], axis=1)
        from mathutils import Euler
        sm = P.quat_continuous(np.array([list(Euler(tuple(e), order).to_quaternion())
                                         for e in es]))
    dev = P.qangle_deg(after, sm)[INNER]
    check(f"1 clean_jitter {bone}: = independent smoothing, less jitter",
          r["ok"] and dev.max() < 0.1 and ja < jb and ch.max() < 15.0,
          f"vs-expected max={dev.max():.4f}°  change max={ch.max():.2f}° "
          f"mean={ch.mean():.2f}°  jitter {jb:.3f}→{ja:.3f} {r['data']['metrics']}")
    if bone == "forearm_fk.L":
        op = agent_ops.get_op(data_dir, r["data"]["op_id"])
        _t, strip = agent_ops.find_op_strip(rig, op)
        paths = sorted({fc.data_path.rsplit(".", 1)[1] for fc in strip.action.fcurves})
        check("2 Euler bone smooths rotation_euler", "rotation_euler" in paths, paths)
    call("revert", op_id=r["data"]["op_id"])
    back = P.sample_visible(scene, rig, [bone], FR)["quat"][bone]
    check(f"1b revert restores {bone}", P.qangle_deg(before, back).max() < 0.01)

# ---- 3: strength applied once -----------------------------------------------
bone = "hand_fk.L"
before = P.sample_visible(scene, rig, [bone], FR)["quat"][bone]
eff = {}
for s in (1.0, 0.5, 1.5):
    r = call("hold_pose", bones=[bone], frame_range=[A, B], target="values",
             strength=s, blend=4)
    after = P.sample_visible(scene, rig, [bone], FR)["quat"][bone]
    eff[s] = float(np.median(P.qangle_deg(before, after)[INNER]))
    call("revert", op_id=r["data"]["op_id"])
check("3 strength 0.5 → half the correction (was 0.25)",
      abs(eff[0.5] / eff[1.0] - 0.5) < 0.03, f"ratio={eff[0.5] / eff[1.0]:.3f} eff={eff}")
check("3b strength 1.5 → 1.5× (was 1.0×)",
      abs(eff[1.5] / eff[1.0] - 1.5) < 0.05, f"ratio={eff[1.5] / eff[1.0]:.3f}")

# ---- 4: hold_pose on Euler bones -----------------------------------------------
fa = "forearm_fk.L"
base = agent_ops.base_action_of(rig)
ref_f = 160
smp0 = P.sample_visible(scene, rig, [fa], FR + [ref_f])
r = call("hold_pose", bones=["left_forearm"], frame_range=[A, B], target="from_frame",
         ref_frame=ref_f, blend=4)
check("4 hold_pose accepts Euler bone (forearm_fk.L)", r["ok"], r.get("error"))
smp1 = P.sample_visible(scene, rig, [fa], FR)
want = np.tile(smp0["quat"][fa][-1], (len(FR), 1))      # visible @ ref frame
err = P.qangle_deg(smp1["quat"][fa], want)[INNER]
check("4b Euler from_frame holds the reference pose (inner)", err.max() < 0.05,
      f"err_inner max={err.max():.4f}°")
op_fa = r["data"]["op_id"]
# ---- 5: set_influence scales Euler deltas -----------------------------------------
e1 = float(np.median(P.qangle_deg(smp0["quat"][fa][:-1], smp1["quat"][fa])[INNER]))
call("set_influence", op_id=op_fa, value=0.5)
smp2 = P.sample_visible(scene, rig, [fa], FR)
e05 = float(np.median(P.qangle_deg(smp0["quat"][fa][:-1], smp2["quat"][fa])[INNER]))
check("5 set_influence 0.5 halves an Euler correction (was ignored)",
      abs(e05 / e1 - 0.5) < 0.06, f"{e1:.2f}° → {e05:.2f}° ratio={e05 / e1:.3f}")
call("revert", op_id=op_fa)

# ---- 4c: world_dir on an Euler bone (static axis) -------------------------------
ua = "upper_arm_fk.L"
ax0 = agent_anatomy.probe(scene, rig, part="bone_axis", bone=ua, frame_range=[A, B])
y0 = Vector(ax0["axes"]["+Y"])
target = (Matrix.Rotation(np.radians(30.0), 3, "Z") @ y0).normalized()
r = call("hold_pose", bones=["left_upper_arm"], frame_range=[A, B], target="world_dir",
         world_dir=list(target), world_axis="Y", secondary_axis="X", blend=4)
check("4c world_dir on Euler bone writes", r["ok"], r.get("error"))
errs = []
cur = scene.frame_current
for f in FR[8:-8]:
    scene.frame_set(f)
    m = (rig.matrix_world @ rig.pose.bones[ua].matrix).to_quaternion()
    errs.append(np.degrees((m @ Vector((0, 1, 0))).angle(target)))
scene.frame_set(cur)
check("4d upper arm +Y lands on target (inner, re-measured)", max(errs) < 0.5,
      f"err max={max(errs):.3f}°")
call("revert", op_id=r["data"]["op_id"])

# ---- 4e: elbow_front flow (probe → hold_pose) on the Euler owner -----------------
pe = agent_anatomy.probe(scene, rig, part="elbow_front", side="L", frame_range=[A, B])
if (pe.get("confidence") or 0) >= 0.5 and pe.get("secondary_axis"):
    w0 = Vector(pe["world_dir"])
    tgt = (Matrix.Rotation(np.radians(25.0), 3, "Z") @ w0).normalized()
    r = call("hold_pose", bones=[pe["owner_bone"]], frame_range=[A, B], target="world_dir",
             world_dir=list(tgt), world_axis="probe:elbow_front.L",
             secondary_axis=pe["secondary_axis"], blend=4)
    pe2 = agent_anatomy.probe(scene, rig, part="elbow_front", side="L",
                              frame_range=[A, B], toward=list(tgt))
    check("4e elbow_front standard flow works on Euler owner",
          r["ok"] and pe2.get("err_inner_deg") is not None and pe2["err_inner_deg"] < 8,
          f"owner={pe['owner_bone']} conf={pe['confidence']} err_inner={pe2.get('err_inner_deg')}")
    call("revert", op_id=r["data"]["op_id"])
else:
    check("4e elbow_front standard flow works on Euler owner (skipped: low conf)", True,
          f"conf={pe.get('confidence')}")

# ---- 6: effect_check default sampling avoids the zero-weight taper ends ----------
r = call("hold_pose", bones=["left_hand"], frame_range=[A, B], target="values", blend=4)
ec = call("effect_check", op_id=r["data"]["op_id"])
check("6 effect_check(op_id) on a tapered fix: every sampled frame moved",
      ec["ok"] and ec["data"]["pass"], f"{ec.get('summary')} frames="
      f"{[x['frame'] for x in ec['data']['per_frame']]}")
call("revert", op_id=r["data"]["op_id"])
r = call("foot_lock", side="R", frame_range=[790, 853], lock="xy")
ec = call("effect_check", op_id=r["data"]["op_id"])
check("6b effect_check infers bones from the strip (foot_lock has no params.bones)",
      ec["ok"] and "foot_ik.R" in ec["data"]["per_frame"][0]["bones"],
      ec.get("summary") if ec["ok"] else ec.get("error"))
call("revert", op_id=r["data"]["op_id"])

# ---- 7: NLA auto-blend must never touch other fixes -------------------------------
fr7 = list(range(60, 170))
r1 = call("hold_pose", bones=["spine2"], frame_range=[65, 155], target="values", blend=4)
sp_mid = P.sample_visible(scene, rig, ["spine_fk.001"], fr7)["quat"]["spine_fk.001"]
r2 = call("hold_pose", bones=["left_hand"], frame_range=[75, 165], target="values", blend=4)
aft = P.sample_visible(scene, rig, ["spine_fk.001", "hand_fk.L"], fr7)
ins = slice(fr7.index(69), fr7.index(151) + 1)
inh = slice(fr7.index(79), fr7.index(161) + 1)
d_sp = P.qangle_deg(sp_mid[ins], aft["quat"]["spine_fk.001"][ins]).max()
e_h = P.qangle_deg(aft["quat"]["hand_fk.L"][inh],
                   np.tile([1.0, 0, 0, 0], (len(fr7), 1))[inh]).max()
strips = [agent_ops.find_op_strip(rig, agent_ops.get_op(data_dir, r["data"]["op_id"]))[1]
          for r in (r1, r2)]
check("7 partially overlapping fixes on adjacent tracks: no auto-blend",
      all(st.blend_in == 0 and st.blend_out == 0 and not st.use_auto_blend for st in strips)
      and d_sp < 0.01 and e_h < 0.05,
      f"blends={[(st.blend_in, st.blend_out) for st in strips]} earlier fix moved "
      f"{d_sp:.4f}° (was 16°), new fix err {e_h:.4f}° (was 53°)")
# legacy strip (old file) with auto-blend on: a new overlapping write must not change it
strips[0].use_auto_blend = True
strips[0].blend_out = 7.0
r3 = call("hold_pose", bones=["right_hand"], frame_range=[120, 200], target="values", blend=4)
check("7b legacy auto-blend strip left exactly as it was (and auto turned off)",
      strips[0].blend_out == 7.0 and not strips[0].use_auto_blend,
      f"blend_out={strips[0].blend_out} auto={strips[0].use_auto_blend}")
for r in (r1, r2, r3):
    call("revert", op_id=r["data"]["op_id"])

# ---- 8: old file with a 1-frame-lagged mcd_base: the FIRST write must be exact ------
# (the legacy correction used to run inside _write_strip - after the tool had
#  already sampled the lagged base → motion_copy first write err_inner 11.9°)
def lag_base():
    agent_ops.ensure_base_on_nla(rig)
    st = next(t for t in rig.animation_data.nla_tracks if t.name == agent_ops.BASE_TRACK).strips[0]
    st.action_frame_start = st.frame_start - 1.0
    st.action_frame_end -= 1.0
    bpy.context.view_layer.update()
    return st
st8 = lag_base()
check("8 simulated old file: base lags 1 frame", st8.frame_start - st8.action_frame_start == 1.0)
SRC8, DST8 = [405, 450], 700
r = call("motion_copy", bones=["left_upper_arm", "left_forearm", "left_hand"],
         src_range=SRC8, dst_start=DST8)
cm = call("compare_motion", **r["data"]["metrics"]["verify"]["args"]) if r["ok"] else {"ok": False}
check("8b first write on an old file is exact (base settled before sampling)",
      r["ok"] and cm["ok"] and cm["data"]["err_inner_deg"] < 0.05
      and st8.frame_start == st8.action_frame_start,
      f"err_inner={cm.get('data', {}).get('err_inner_deg')} (was 11.9° before the fix)")
check("8c compare_motion default output has no per-frame arrays",
      cm["ok"] and all("err_per_frame" not in v for v in cm["data"]["bones"].values()),
      sorted(next(iter(cm["data"]["bones"].values())).keys()) if cm.get("ok") else None)
op8 = r["data"]["op_id"]
st8 = lag_base()                                   # lag again, then the GUI path
agent_ops.reapply(data_dir, rig, op8, scene=scene)
cm = call("compare_motion", **r["data"]["metrics"]["verify"]["args"])
check("8d reapply (GUI param path) also settles before re-solving",
      cm["ok"] and cm["data"]["err_inner_deg"] < 0.05,
      f"err_inner={cm['data']['err_inner_deg']}")
call("revert", op_id=op8)

# ---- 9: dry_run must never write (legacy tools used to swallow it via **_) ----------
def n_agent_strips():
    return sum(len(t.strips) for t in rig.animation_data.nla_tracks
               if agent_ops.is_agent_track_name(t.name))
n0, ops0 = n_agent_strips(), len(agent_ops.list_ops(data_dir))
rj = call("clean_jitter", frame_range=[1000, 1060], bone="right_hand", dry_run=True)
rh = call("hold_pose", bones=["left_hand"], frame_range=[A, B], target="values", dry_run=True)
rr = call("restore_accent", frame_range=[A, B],
          data_path='pose.bones["hand_fk.L"].rotation_quaternion', dry_run=True)
check("9 dry_run on clean_jitter / hold_pose / restore_accent writes nothing",
      all(x["ok"] and x["data"].get("dry_run") for x in (rj, rh, rr))
      and n_agent_strips() == n0 and len(agent_ops.list_ops(data_dir)) == ops0,
      f"strips {n0}->{n_agent_strips()} ops {ops0}->{len(agent_ops.list_ops(data_dir))}")
check("9b clean_jitter dry_run predicts the change",
      rj["ok"] and rj["data"]["metrics"].get("pred_rot_change_max_deg", 0) > 0,
      rj.get("data", {}).get("metrics"))
rf = call("fix_ground", frame_range=[A, B], side="L",
          loc_path='pose.bones["foot_ik.L"].location', dry_run=True)
check("9c unsupported dry_run is refused (no silent write)",
      not rf["ok"] and rf["error"]["code"] == "E_SCOPE" and n_agent_strips() == n0,
      rf.get("error"))
v0 = call("ping")["version"]
rw = call("hold_pose", agent_id="other9", bones=["head"], frame_range=[700, 720], target="values")
rd = call("clean_jitter", agent_id="me9", frame_range=[1000, 1060], bone="right_hand",
          dry_run=True, expect_version=v0)
check("9d dry_run gets the scoped staleness check (unrelated write → no E_STALE)",
      rw["ok"] and rd["ok"], rd.get("error"))
lo = call("list_ops", owner="other9", compact=True)
check("9e compact list_ops rows carry the written bones",
      lo["ok"] and lo["data"]["fixes"] and lo["data"]["fixes"][0].get("bones") == ["head"],
      lo["data"]["fixes"][:1] if lo["ok"] else lo.get("error"))
call("revert", agent_id="other9", op_id=rw["data"]["op_id"])

# ---- 10: stacking warning / set_influence envelope / reapply keeps strength ------------
r1 = call("clean_jitter", agent_id="s1", frame_range=[1000, 1060], bone="right_hand")
call("release", agent_id="s1")
r2 = call("clean_jitter", agent_id="s2", frame_range=[1010, 1050], bone="right_hand")
check("10 second same-kind fix on the same bone/frames warns about stacking",
      r2["ok"] and any("同类修复" in w for w in r2["warnings"]), r2.get("warnings"))
call("revert", agent_id="s2", op_id=r2["data"]["op_id"])
call("release", agent_id="s2")
r = call("hold_pose", agent_id="s3", bones=["left_hand"], frame_range=[A, B], target="values")
oid = r["data"]["op_id"]
before = P.sample_visible(scene, rig, ["hand_fk.L"], FR)["quat"]["hand_fk.L"]
si = call("set_influence", agent_id="s3", op_id=oid, value=0.5)
check("10b set_influence answers with its own envelope (not '0-0 帧')",
      si["ok"] and si["data"].get("op_id") == oid and "力度" in si["summary"], si.get("summary"))
half = P.sample_visible(scene, rig, ["hand_fk.L"], FR)["quat"]["hand_fk.L"]
call("reapply", agent_id="s3", op_id=oid, overrides={"blend": 5})
after_re = P.sample_visible(scene, rig, ["hand_fk.L"], FR)["quat"]["hand_fk.L"]
_t, st10 = agent_ops.find_op_strip(rig, agent_ops.get_op(data_dir, oid))
check("10c reapply keeps the set_influence strength",
      abs(float(st10.action.get("applied_exp", 1.0)) - 0.5) < 1e-9
      and P.qangle_deg(half[INNER], after_re[INNER]).max() < 0.5,
      f"applied_exp={st10.action.get('applied_exp')} "
      f"inner diff vs pre-reapply={P.qangle_deg(half[INNER], after_re[INNER]).max():.3f}°")
call("revert", agent_id="s3", op_id=oid)
call("revert", agent_id="s1", op_id=r1["data"]["op_id"])
am = call("analyze_motion", bones=["right_hand"], frame_range=[1000, 1060], brief=True)
row = (am.get("data") or {}).get("bones", {}).get("hand_fk.R", {})
check("10d analyze_motion brief drops the speed series; jitter_top_frames present",
      am["ok"] and "speed_series" not in am["data"] and len(row.get("jitter_top_frames", [])) == 5,
      row.get("jitter_top_frames"))

# ---- 12: old tools no longer swallow misspelled parameters (**_) ------------------------
r = call("hold_pose", bones=["left_hand"], frame_range=[A, B], target="values", stregth=0.5)
check("12 hold_pose(stregth=…) → rejected with the closest name (was silently strength=1)",
      not r["ok"] and "strength" in r["error"]["message"], r.get("error", {}).get("message", "")[:140])
r = call("clean_jitter", bone="left_hand", frame_range=[A, B], widht=5)
check("12b clean_jitter(widht=…) → rejected, suggests width",
      not r["ok"] and "width" in r["error"]["message"], r.get("error", {}).get("message", "")[:140])
r = call("describe", target=[A, B], max_items=3)
check("12c read tools too (describe(max_items=…) → rejected, lists what it takes)",
      not r["ok"] and "不认识参数" in r["error"]["message"], r.get("error", {}).get("message", "")[:140])

# ---- 14: list_ops can filter by bone (frames alone also returns unrelated bones) --------
r = call("hold_pose", bones=["left_hand"], frame_range=[1000, 1040], target="values", blend=4)
r2 = call("hold_pose", bones=["right_hand"], frame_range=[1000, 1040], target="values", blend=4)
lo = call("list_ops", frames=[1000, 1040], bones=["hand_fk.L"], compact=True)
rows = (lo.get("data") or {}).get("fixes", [])
check("14 list_ops(bones=…) keeps only fixes that write those bones",
      lo["ok"] and rows and all("hand_fk.L" in (x.get("bones") or []) for x in rows)
      and not any("hand_fk.R" in (x.get("bones") or []) and "hand_fk.L" not in (x.get("bones") or []) for x in rows),
      [(x["op_id"], x.get("bones")) for x in rows])
for rr in (r, r2):
    if rr["ok"]:
        call("revert", op_id=rr["data"]["op_id"])

# ---- 13: probe's hold_pose_args is the recommended per-frame probe form ------------------
pr = call("probe_anatomy", part="palm", side="L", frame_range=[900, 960], toward=[0, -1, 0])
hp = (pr.get("data") or {}).get("hold_pose_args") or {}
check("13 probe hold_pose_args = probe:palm.L + probe:hand_axis.L (not a static mean axis)",
      hp.get("world_axis") == "probe:palm.L" and hp.get("secondary_axis") == "probe:hand_axis.L", hp)
if hp:
    r = call("hold_pose", frame_range=[900, 960], target="world_dir", world_dir=[0, -1, 0],
             mode="replace", blend=4, **hp)
    pr2 = call("probe_anatomy", part="palm", side="L", frame_range=[900, 960], toward=[0, -1, 0])
    e0 = pr["data"].get("err_inner_deg"); e1 = (pr2.get("data") or {}).get("err_inner_deg")
    check("13b hold_pose(**hold_pose_args) reaches the target (err_inner < 5°)",
          r["ok"] and e1 is not None and e1 < 5, f"err_inner {e0} → {e1}")
    if r["ok"]:
        call("revert", op_id=r["data"]["op_id"])

fails = [r for r in RESULTS if not r[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for n, _o, d in fails:
    print(f"FAIL {n}: {d}")
sys.exit(1 if fails else 0)
