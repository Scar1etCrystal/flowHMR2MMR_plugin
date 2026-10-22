"""Agent-layer regressions from the 2026-10-04 read-only review (fixture, never saved).

  R1  M8   the fix-list timer mirrors the scene without editing it: an empty
           list and the rebuild after a write neither bump the data version
           ("external" → E_STALE for every other agent) nor rewrite strips
  R2  M2   reapply of an OLDER op keeps a NEWER fix on the same bone/frames
           (Euler bone: was wiped exactly; quaternion bone: conjugated);
           compare_motion(op_id) / analyze_motion(baseline_op) measure the op
           itself, not the newer fix on top
  R3  M3   strength 0 = mute and comes back; a 140 deg fix survives 1.5 → 1.0
           (re-derived from the stored unit delta, not from the curves)
  R4  M4   effect_check leaves a user-muted (= rejected) fix muted
  R5  M27  effect_check samples overlap's real inner frames (auto blend = None)
  R6  M15  hold_pose world_dir refuses a parent + its child in one call
  R7  M25  an active Action on top of mcd_base never becomes a second base
  R8  M26  a write does not toggle playback off when it is already running
  R9  M9   an animated direction object does not re-run its op on every scrub
  R10 M13  the snapshot cache is re-baked when the base action changed,
           kept when only agent strips changed
  R11 M22/M23  exclusive bind on Windows; HTTP-looking connections dropped;
           optional token
  R12 M16  a corrupt op log is moved aside and reported, never overwritten
  R13 M24  fix_ground on a tilted rig moves the foot straight down in WORLD space
  R14 P9   batch-written agent keys are identical to the per-key insert path

    bash tools/mcd.sh e2e tests/e2e_review_agent.py
"""
import glob
import json
import os
import socket
import sys
import threading
import time

import addon_utils
import bpy
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import (  # noqa: E402
    agent_bake, agent_bridge as B, agent_ops, agent_pose as P)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig or next(o for o in scene.objects if o.type == "ARMATURE"
                               and o.name.startswith("RIG-"))
settings.mmr_rig = rig
data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_review_agent_data")
os.makedirs(data_dir, exist_ok=True)
for f in glob.glob(os.path.join(data_dir, "agent_ops.json*")):
    os.remove(f)
settings.data_directory = data_dir


def call(tool, **args):
    r = B._dispatch({"tool": tool, "args": args})
    if not r["ok"]:
        print("ERR", tool, r.get("error"), (r.get("trace") or "")[-300:])
    return r


def track_of(op_id):
    tr, _st = agent_ops.find_op_strip(rig, agent_ops.get_op(data_dir, op_id))
    return tr


def quats(bones, frames):
    return P.sample_visible(scene, rig, list(bones), list(frames))["quat"]


def max_diff(a, b):
    return max(float(P.qangle_deg(a[k], b[k]).max()) for k in a)


# ---- R1 / M8 (first: the op log is still empty) --------------------------------
B.start_server(6293)
try:
    B._FIXLIST_SYNCED = None
    scene.frame_set(scene.frame_current)
    bpy.context.view_layer.update()
    time.sleep(0.35)
    v0, n0 = B._DATA_VERSION, len(B._JOURNAL.rows)
    for _ in range(3):
        B._fixlist_tick()
        bpy.context.view_layer.update()
        time.sleep(0.35)
    check("R1a empty fix list: timer ticks leave the data version alone",
          B._DATA_VERSION == v0 and len(B._JOURNAL.rows) == n0,
          f"v {v0}->{B._DATA_VERSION} journal +{len(B._JOURNAL.rows) - n0}")
    calls = []
    orig_exp = agent_ops.set_strip_exponent

    def _spy(strip, e):
        calls.append(strip.name)
        return orig_exp(strip, e)

    agent_ops.set_strip_exponent = _spy
    try:
        for i in range(2):
            call("hold_pose", bones=["left_hand"], frame_range=[900 + 60 * i, 940 + 60 * i],
                 target="values", blend=4, agent_id="r1")
        scene.frame_set(scene.frame_current)
        bpy.context.view_layer.update()
        time.sleep(0.35)
        v1, n1 = B._DATA_VERSION, len(B._JOURNAL.rows)
        B._fixlist_tick()
        bpy.context.view_layer.update()
        time.sleep(0.35)
        settings.agent_ops_rev += 1          # what every bridge write does
        B._fixlist_tick()
        bpy.context.view_layer.update()
    finally:
        agent_ops.set_strip_exponent = orig_exp
    tail = [r["tool"] for r in B._JOURNAL.rows[n1:]]
    check("R1b list rebuilds after writes: no version bump, no 'external' row",
          B._DATA_VERSION == v1 and "external" not in tail,
          f"v {v1}->{B._DATA_VERSION} journal tail {tail}")
    check("R1c rebuilds never rewrite strips (set_strip_exponent not called)",
          not calls and len(settings.agent_fixes) == 2,
          f"calls={calls} rows={len(settings.agent_fixes)}")
finally:
    call("release", agent_id="r1")           # auto-claims of the R1 writes
    B.stop_server()

# ---- R2 / M2 ---------------------------------------------------------------------
W = [150, 180]
FRW = list(range(W[0], W[1] + 1))
for bone in ("upper_arm_fk.L", "hand_fk.L"):           # Euler XYZ / quaternion
    rc = call("motion_copy", bones=[bone], src_range=[300, 330], dst_start=W[0], blend=4)
    copy_id = rc["data"]["op_id"]
    rh = call("hold_pose", bones=[bone], frame_range=W, target="values", blend=4)
    hold_id = rh["data"]["op_id"]
    before = quats([bone], FRW)
    rr = call("reapply", op_id=copy_id, overrides={})
    after = quats([bone], FRW)
    hold_tr = track_of(hold_id)
    hold_tr.mute = True
    no_hold = quats([bone], FRW)
    hold_tr.mute = False
    scene.frame_set(scene.frame_current)
    mode = P.rot_mode(rig.pose.bones[bone])
    check(f"R2a {bone} ({mode}): reapply of the older copy keeps the newer fix",
          rr["ok"] and max_diff(before, after) < 0.01,
          f"Δ={max_diff(before, after):.4f}° (the newer fix moves it {max_diff(before, no_hold):.1f}°)")
    cm = call("compare_motion", op_id=copy_id)
    d = cm.get("data") or {}
    check(f"R2b {bone}: compare_motion(op_id) verifies the copy itself",
          cm["ok"] and d.get("err_inner_deg", 99) < 0.05
          and d.get("muted_newer_tracks") == [hold_tr.name],
          f"err_inner={d.get('err_inner_deg')} muted={d.get('muted_newer_tracks')}")
    am = call("analyze_motion", bones=[bone], frame_range=W, main_bone=bone,
              baseline_op=copy_id, brief=True)
    hold_tr.mute = True
    am0 = call("analyze_motion", bones=[bone], frame_range=W, main_bone=bone,
               baseline_op=copy_id, brief=True)
    hold_tr.mute = False
    scene.frame_set(scene.frame_current)
    va = ((am.get("data") or {}).get("vs_baseline") or {})
    vb0 = (((am0.get("data") or {}).get("vs_baseline") or {}).get("bones") or {}).get(bone, {})
    vb = (va.get("bones") or {}).get(bone, {})
    keys = ("max_diff_deg", "along_motion_min_deg", "approach_peak_deg")
    same = all(abs(float(vb.get(k) or 0) - float(vb0.get(k) or 0)) < 0.05 for k in keys)
    check(f"R2c {bone}: analyze_motion baseline_op = the op's own effect",
          am["ok"] and same and va.get("muted_newer_tracks") == [hold_tr.name],
          f"{[(k, vb.get(k), vb0.get(k)) for k in keys]}")
    call("revert", op_id=hold_id)
    call("revert", op_id=copy_id)

# ---- R3 / M3 ---------------------------------------------------------------------
rs = call("hold_pose", bones=["spine2"], frame_range=[400, 430], target="values", blend=4)
sid = rs["data"]["op_id"]
FRS = list(range(400, 431))
on = quats(["spine_fk.001"], FRS)
tr = track_of(sid)
tr.mute = True
off = quats(["spine_fk.001"], FRS)
tr.mute = False
r0 = call("set_influence", op_id=sid, value=0.0)
at0 = quats(["spine_fk.001"], FRS)
_tr, strip = agent_ops.find_op_strip(rig, agent_ops.get_op(data_dir, sid))
check("R3a strength 0 = the fix is off (strip muted, track untouched)",
      r0["ok"] and max_diff(at0, off) < 0.01 and strip.mute and not tr.mute,
      f"Δ_vs_off={max_diff(at0, off):.4f}° strip.mute={strip.mute}")
settings.agent_ops_rev += 1
B._fixlist_tick()
row = next((i for i in settings.agent_fixes if i.op_id == sid), None)
check("R3b the panel shows 0.00, not 1.00", row is not None and abs(row.exponent) < 1e-6,
      f"exponent={row.exponent if row else None}")
rr = call("reapply", op_id=sid, overrides={"blend": 6})
_tr, strip = agent_ops.find_op_strip(rig, agent_ops.get_op(data_dir, sid))
check("R3c reapply keeps strength 0 (new strip muted)",
      rr["ok"] and strip.mute and float(strip.action.get("applied_exp", 1.0)) == 0.0,
      f"mute={strip.mute} exp={strip.action.get('applied_exp')}")
call("reapply", op_id=sid, overrides={"blend": 4})
r1 = call("set_influence", op_id=sid, value=1.0)
back = quats(["spine_fk.001"], FRS)
check("R3d strength 0 → 1 restores the fix exactly", r1["ok"] and max_diff(back, on) < 0.01,
      f"Δ={max_diff(back, on):.4f}°")
# 140 deg delta: 1.5 crosses 180 deg; 1.0 must give the 140 deg back (old: -100 deg)
path = 'pose.bones["f_pinky.01.L"].rotation_quaternion'
ang = np.radians(140.0)
axis = np.array([0.6, 0.8, 0.0])
dq = np.tile(np.r_[np.cos(ang / 2), np.sin(ang / 2) * axis], (11, 1))
big_track, big = agent_ops._write_strip(rig, "e2e_big140", 500, quats={path: dq}, blend=0)
agent_ops.set_strip_exponent(big, 1.5)
agent_ops.set_strip_exponent(big, 1.0)
vals = np.array([[fc.keyframe_points[5].co[1] for fc in sorted(
    (fc for fc in big.action.fcurves if fc.data_path == path), key=lambda f: f.array_index)]])
got = float(np.degrees(2 * np.arccos(np.clip(abs(vals[0, 0]), 0, 1))))
same_axis = float(np.dot(vals[0, 1:] / np.linalg.norm(vals[0, 1:]), axis) * np.sign(vals[0, 0]))
check("R3e a 140° fix: strength 1.5 → 1.0 gives 140° back, same direction",
      abs(got - 140.0) < 0.01 and same_axis > 0.9999, f"angle={got:.3f}° axis·={same_axis:.5f}")
act = big.action
big_track.strips.remove(big)
rig.animation_data.nla_tracks.remove(big_track)
bpy.data.actions.remove(act)

# ---- R4 / M4 ---------------------------------------------------------------------
tr = track_of(sid)
tr.mute = True
ec = call("effect_check", op_id=sid)
check("R4a effect_check leaves a muted (rejected) fix muted", ec["ok"] and track_of(sid).mute
      and ec["data"]["pass"], f"mute={track_of(sid).mute} verdict={ec['data'].get('verdict')}")
ec = call("effect_check", bones=["spine_fk.001"], frames=[405, 415, 425])
check("R4b all-tracks A/B also restores every mute state", ec["ok"] and track_of(sid).mute,
      f"tracks={ec['data'].get('tracks')} mute={track_of(sid).mute}")
track_of(sid).mute = False
scene.frame_set(scene.frame_current)

# ---- R5 / M27 --------------------------------------------------------------------
ro = call("overlap", chain="arm.R", frame_range=[20, 110], delay=1.0, max_delay=3.0)
oid = ro["data"]["op_id"]
op = agent_ops.get_op(data_dir, oid)
inner = op["metrics"]["inner_frames"]
ec = call("effect_check", op_id=oid)
frames = [r["frame"] for r in ec["data"]["per_frame"]]
check("R5 effect_check on overlap (auto blend) samples its inner frames and passes",
      ec["ok"] and op["params"].get("blend") is None and min(frames) >= inner[0]
      and max(frames) <= inner[1] and ec["data"]["pass"],
      f"inner={inner} frames={frames} verdict={ec['data']['verdict']}")
call("revert", op_id=oid)

# ---- R6 / M15 --------------------------------------------------------------------
rw = call("hold_pose", bones=["left_forearm", "left_hand"], frame_range=[600, 640],
          target="world_dir", world_dir=[0, -1, 0], world_axis="Y")
check("R6a world_dir with a parent and its child in one call is refused",
      not rw["ok"] and "后代" in rw["error"]["message"], rw.get("error", {}).get("message", "")[:60])
rw = call("hold_pose", bones=["left_hand", "right_hand"], frame_range=[600, 640],
          target="world_dir", world_dir=[0, -1, 0], world_axis="Y", dry_run=True)
check("R6b siblings in one world_dir call still work", rw["ok"], rw.get("error"))

# ---- R7 / M25 --------------------------------------------------------------------
anim = rig.animation_data
base_act = agent_ops.base_action_of(rig)
anim.action = base_act                     # the user clicked the base action
r = call("hold_pose", bones=["right_hand"], frame_range=[700, 730], target="values")
n_base = sum(1 for t in anim.nla_tracks if t.name.startswith(agent_ops.BASE_TRACK))
check("R7a re-activated base action: unassigned, no second base strip",
      r["ok"] and anim.action is None and n_base == 1, f"ok={r['ok']} active={anim.action} bases={n_base}")
other = bpy.data.actions.new("e2e_other_active")
anim.action = other
r2 = call("hold_pose", bones=["right_hand"], frame_range=[740, 770], target="values")
check("R7b a different active Action is refused with a clear message",
      not r2["ok"] and "活动 Action" in r2["error"]["message"],
      r2.get("error", {}).get("message", "")[:60])
anim.action = None
bpy.data.actions.remove(other)
call("revert", op_id=r["data"]["op_id"])

# ---- R8 / M26 --------------------------------------------------------------------
toggles = []
saved = (B._playback_running, B._toggle_playback)
try:
    B._toggle_playback = lambda: toggles.append(1)
    B._playback_running = lambda: True
    B._focus_preview(scene, [100, 120])
    running = len(toggles)
    B._playback_running = lambda: False
    B._focus_preview(scene, [100, 120])
finally:
    B._playback_running, B._toggle_playback = saved
check("R8 playback already running → not toggled; stopped → started once",
      running == 0 and len(toggles) == 1, f"toggles running={running} stopped={len(toggles) - running}")

# ---- R9 / M9 ---------------------------------------------------------------------
empty = bpy.data.objects.new("e2e_dir_arrow", None)
scene.collection.objects.link(empty)
empty.rotation_mode = "XYZ"
empty.rotation_euler = (1.2, 0.0, 0.0)
empty.keyframe_insert("rotation_euler", frame=150)
empty.rotation_euler = (1.6, 0.3, 0.0)
empty.keyframe_insert("rotation_euler", frame=224)
rd = call("hold_pose", bones=["left_hand"], frame_range=[150, 224], target="world_dir",
          dir_object=empty.name, dir_mode="arrow", world_axis="Y")
did = rd["data"]["op_id"]
B._PARAM_PENDING.clear()
B._EMPTY_WATCH.clear()
scene.frame_set(150)
B._param_tick()
scene.frame_set(200)
B._param_tick()
scrub = (did, "__refresh__") in B._PARAM_PENDING
fc = empty.animation_data.action.fcurves.find("rotation_euler", index=1)
fc.keyframe_points[-1].co.y = 0.5            # a real edit of the arrow's animation
fc.update()
B._param_tick()
edit = (did, "__refresh__") in B._PARAM_PENDING
B._PARAM_PENDING.clear()
check("R9 animated arrow: scrubbing does not re-run the op, editing its keys does",
      rd["ok"] and not scrub and edit, f"scrub_refresh={scrub} edit_refresh={edit}")
call("revert", op_id=did)
bpy.data.objects.remove(empty)

# ---- R10 / M13 -------------------------------------------------------------------
base_act = agent_ops.base_action_of(rig)
fcz = base_act.fcurves.find('pose.bones["foot_ik.L"].location', index=2)
F = 1000
B._STORE = None
st0 = B.get_store()
i = int(np.searchsorted(st0.frames, F))
z0 = float(st0.positions["left_foot"][i][2])
npz = agent_bake.bake_cache_path(data_dir, rig.name, int(settings.mocap_frame_start)
                                 or scene.frame_start, int(settings.mocap_frame_end)
                                 or scene.frame_end, tag="rig")
# (never keep Keyframe objects across FCurve.update(): it reallocates the keys)
for k in fcz.keyframe_points:
    if 990 <= k.co.x <= 1010:
        k.co.y += 0.05
fcz.update()
try:
    B._STORE = None                           # a later session / epoch rebuild
    z1 = float(B.get_store().positions["left_foot"][i][2])
finally:
    for k in fcz.keyframe_points:
        if 990 <= k.co.x <= 1010:
            k.co.y -= 0.05
    fcz.update()
    scene.frame_set(scene.frame_current)
B._STORE = None
z2 = float(B.get_store().positions["left_foot"][i][2])
check("R10a base action edited → snapshot re-baked (not the stale npz)",
      abs((z1 - z0) - 0.05) < 0.005 and abs(z2 - z0) < 1e-4,
      f"z0={z0:.4f} edited={z1:.4f} restored={z2:.4f}")
mt = os.path.getmtime(npz)
rw = call("hold_pose", bones=["right_hand"], frame_range=[1200, 1230], target="values")
B._STORE = None
B.get_store()
check("R10b an agent write keeps the snapshot (same npz, not re-baked)",
      rw["ok"] and os.path.getmtime(npz) == mt, f"mtime same={os.path.getmtime(npz) == mt}")
call("revert", op_id=rw["data"]["op_id"])

# ---- R11 / M22 + M23 -------------------------------------------------------------
check("R11a bind flags: exclusive on Windows, SO_REUSEADDR elsewhere",
      B._bind_flags("win32") == {"reuse_address": False, "exclusive": True}
      and B._bind_flags("linux") == {"reuse_address": True, "exclusive": False},
      f"{B._bind_flags('win32')} {B._bind_flags('linux')}")
PORT = 6294
B.start_server(PORT)
out = {}


def _client():
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=10) as s:
            # (never a real "save" in tests: a harmless write stands in for it)
            s.sendall(b"POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: text/plain\r\n\r\n"
                      b'{"id": 9, "tool": "set_preview", "args": {"frame_range": [1234, 1240]}}\n')
            s.settimeout(5)
            out["http"] = s.recv(65536)
        with socket.create_connection(("127.0.0.1", PORT), timeout=10) as s:
            s.sendall(b'{"id": 1, "tool": "ping", "args": {}}\n')
            s.settimeout(10)
            out["ping"] = s.recv(65536)
        os.environ["MCD_AGENT_TOKEN"] = "t0k"
        with socket.create_connection(("127.0.0.1", PORT), timeout=10) as s:
            s.sendall(b'{"id": 2, "tool": "ping", "args": {}}\n')
            s.settimeout(10)
            out["no_token"] = s.recv(65536)
            s.sendall(b'{"id": 3, "tool": "ping", "args": {}, "token": "t0k"}\n')
            out["token"] = s.recv(65536)
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    finally:
        os.environ.pop("MCD_AGENT_TOKEN", None)
        out["done"] = True


th = threading.Thread(target=_client, daemon=True)
th.start()
deadline = time.time() + 30
B.headless_pump_loop(until=lambda: out.get("done") or time.time() > deadline)
B.stop_server()
check("R11b an HTTP request line is dropped before its body runs",
      out.get("http") == b"" and int(scene.frame_preview_start) != 1234,
      f"reply={out.get('http')} preview_start={scene.frame_preview_start}")
ping = json.loads((out.get("ping") or b"{}").decode() or "{}")
check("R11c the JSON-lines protocol is unchanged (no token configured)", ping.get("ok") is True,
      out.get("error") or ping.get("summary"))
nt = json.loads((out.get("no_token") or b"{}").decode() or "{}")
tk = json.loads((out.get("token") or b"{}").decode() or "{}")
check("R11d MCD_AGENT_TOKEN set: requests without it get E_AUTH, with it pass",
      (nt.get("error") or {}).get("code") == "E_AUTH" and tk.get("ok") is True,
      f"no_token={nt.get('error')} token_ok={tk.get('ok')}")

# ---- R13 / M24 -------------------------------------------------------------------
# fix_ground writes a WORLD Z delta into foot_ik's LOCAL location: exact only while
# local Z is world Z.  Tilt the rig like the global correction does (-4.2°/3.7°).
import math  # noqa: E402
corr = rig.parent
rot_fcs = [fc for fc in ((corr.animation_data.action.fcurves
                          if corr is not None and corr.animation_data
                          and corr.animation_data.action else ()))
           if fc.data_path.startswith("rotation")]
if corr is None:
    check("R13 tilted rig: fix_ground moves the foot straight down in world space",
          False, "fixture layout changed: RIG has no parent")
else:
    saved = (corr.rotation_mode, tuple(corr.rotation_euler), [fc.mute for fc in rot_fcs])
    for fc in rot_fcs:
        fc.mute = True
    corr.rotation_mode = "XYZ"
    corr.rotation_euler = (math.radians(-4.2), math.radians(3.7), 0.0)
    try:
        a, b = 1300, 1320
        FRG = list(range(a, b + 1))
        op = agent_ops.fix_ground(rig, agent_ops.base_action_of(rig),
                                  'pose.bones["foot_ik.L"].location',
                                  np.full(len(FRG), 0.03), floor_z=0.0, mode="float",
                                  frame_range=[a, b], blend=0)
        tr13, _st13 = agent_ops.find_op_strip(rig, op)

        def _heads():
            out = []
            for f in FRG:
                scene.frame_set(f)
                out.append(np.array(rig.matrix_world @ rig.pose.bones["foot_ik.L"].head))
            return np.array(out)

        on = _heads()
        tr13.mute = True
        off = _heads()
        tr13.mute = False
        d = on - off
        horiz = float(np.linalg.norm(d[:, :2], axis=1).max())
        check("R13 tilted rig: fix_ground moves the foot straight down in world space",
              horiz < 0.0002 and np.allclose(d[:, 2], -0.03, atol=0.0002),
              f"horizontal={horiz * 1000:.3f} mm vertical={d[:, 2].min() * 1000:.2f}.."
              f"{d[:, 2].max() * 1000:.2f} mm (old: ~2-3 mm sideways)")
        agent_ops.delete_op_strip(rig, op)
    finally:
        corr.rotation_mode = saved[0]
        corr.rotation_euler = saved[1]
        for fc, m in zip(rot_fcs, saved[2]):
            fc.mute = m
        scene.frame_set(scene.frame_current)

# ---- R14 / P9 --------------------------------------------------------------------
# agent strips are written into fresh curves in one add(n)+foreach_set; the keys
# must be what the per-key insert path produced
from bl_ext.user_default.mocap_doctor.core import animation as ANIM  # noqa: E402
act = bpy.data.actions.new("e2e_p9_keys")
try:
    vals = np.sin(np.arange(40) * 0.3) * 0.1
    act.fcurves.new("location", index=0)
    agent_bake.write_fcurve_values(act, "location", 0, 100, vals)        # batch path
    fa = act.fcurves.find("location", index=0)
    fb = act.fcurves.new("location", index=1)                          # per-key path
    cache = ANIM.keyframe_map(fb)
    for k, v in enumerate(vals):
        ANIM.set_fcurve_value(fb, 100 + k, float(v), cache=cache)
    fb.update()

    def _keys(fc):
        return [(tuple(k.co), k.interpolation, k.handle_left_type, k.handle_right_type,
                 k.easing, k.type, tuple(k.handle_left), tuple(k.handle_right))
                for k in fc.keyframe_points]

    ev = max(abs(fa.evaluate(t) - fb.evaluate(t)) for t in np.arange(99.0, 141.0, 0.25))
    check("R14 P9 batch-written keys = per-key inserted keys (co, interp, handles, eval)",
          _keys(fa) == _keys(fb) and ev == 0.0,
          f"n={len(fa.keyframe_points)} eval_diff={ev} first={_keys(fa)[0][:5]} vs {_keys(fb)[0][:5]}")
finally:
    bpy.data.actions.remove(act)

# ---- R12 / M16 (last: it resets the op log) -------------------------------------
log = os.path.join(data_dir, "agent_ops.json")
with open(log, "w", encoding="utf-8") as handle:
    handle.write('[{"id": "x", "tool": "hold_pose"')     # killed mid-write
rl = call("list_ops")
aside = glob.glob(log + ".corrupt-*")
check("R12a a corrupt op log is reported and moved aside, not read as empty",
      not rl["ok"] and "读不了" in rl["error"]["message"] and aside and not os.path.exists(log),
      f"{rl.get('error', {}).get('message', '')[:50]} aside={[os.path.basename(a) for a in aside]}")
with open(aside[0], encoding="utf-8") as handle:
    kept = handle.read()
rn = call("hold_pose", bones=["left_hand"], frame_range=[1100, 1130], target="values")
ops = agent_ops.list_ops(data_dir)
check("R12b the next write starts a fresh log; the damaged bytes survive; no temp file left",
      rn["ok"] and len(ops) == 1 and kept.startswith('[{"id": "x"')
      and not glob.glob(os.path.join(data_dir, "*.tmp")),
      f"ops={len(ops)} tmp={glob.glob(os.path.join(data_dir, '*.tmp'))}")

n_tracks = len(rig.animation_data.nla_tracks)
with open(log, "w", encoding="utf-8") as handle:
    handle.write("{garbage")
rx = call("hold_pose", bones=["right_hand"], frame_range=[1300, 1330], target="values")
check("R12c a write with a corrupt log fails before writing a strip (no orphan)",
      not rx["ok"] and len(rig.animation_data.nla_tracks) == n_tracks,
      f"ok={rx['ok']} tracks {n_tracks}->{len(rig.animation_data.nla_tracks)}")

fails = [r for r in RESULTS if not r[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for name, _ok, detail in fails:
    print("FAILED:", name, "::", detail)
if fails:
    sys.exit(1)
