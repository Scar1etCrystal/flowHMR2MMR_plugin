"""e2e: slide_report (脚滑体检) + foot_lock (踩实)。期望 ==== N/N PASS ====。"""
import os
import sys

import addon_utils
import bpy
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import (  # noqa: E402
    agent_bridge, agent_ops, agent_pose as P)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig or next(o for o in scene.objects if o.type == "ARMATURE"
                               and o.name.startswith("RIG-"))
settings.mmr_rig = rig
data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_foot_lock_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)


def call(tool, **args):
    r = agent_bridge._dispatch({"tool": tool, "args": args})
    if not r["ok"]:
        print("ERR", tool, r.get("error"), (r.get("trace") or "")[-600:])
    return r


def foot_world(side, frames):
    s = P.sample_visible(scene, rig, [f"foot_ik.{side}"], frames, world=True)
    m = s["mat"][f"foot_ik.{side}"]
    return m[:, :3, 3], P.mat_to_quat(m[:, :3, :3] / np.linalg.norm(m[:, :3, :3], axis=1, keepdims=True))


# ---- slide_report ------------------------------------------------------------
r = call("slide_report", side="R", threshold_mm=10)
rows = r["data"]["rows"] if r["ok"] else []
check("1 slide_report finds sliding contacts", r["ok"] and r["data"]["flagged"] > 0,
      r.get("summary"))
target = next((x for x in rows if x["flagged"] and 15 <= x["length"] <= 60), None)
check("1b a mid-length flagged interval exists", target is not None, target)
iv = target["interval"]
a, b = target["frames"]
BL = 4
win = list(range(a - BL, b + BL + 1))
pos0, q0 = foot_world("R", win)

# ---- foot_lock xy (default) --------------------------------------------------
r = call("foot_lock", interval=iv)
check("2 foot_lock(interval) writes", r["ok"] and r["data"]["op_id"], r.get("summary"))
op_id = r["data"]["op_id"]
m = r["data"]["metrics"]
check("2b metrics carry ref/drift_before", m.get("drift_before_mm") is not None
      and m.get("ref_frame") is not None, {k: m.get(k) for k in ("ref_frame", "drift_before_mm", "lock")})
pos1, q1 = foot_world("R", win)
inner = slice(BL, len(win) - BL)          # == the contact interval itself
ref_i = win.index(m["ref_frame"])
xy_drift = np.linalg.norm(pos1[inner, :2] - pos1[ref_i, :2], axis=1).max() * 1000
z_dev = np.abs(pos1[inner, 2] - pos0[inner, 2]).max() * 1000
rot_dev = P.qangle_deg(q0[inner], q1[inner]).max()
check("3 xy lock: horizontal drift ≈ 0 over the whole contact",
      xy_drift < 0.5, f"{target['drift_mm']} mm → {xy_drift:.3f} mm")
check("3b xy lock keeps the height curve and the rotation",
      z_dev < 0.5 and rot_dev < 0.05, f"z dev {z_dev:.3f} mm, rot dev {rot_dev:.4f}°")
r2 = call("slide_report", side="R", frame_range=[a, b], threshold_mm=10)
row2 = next((x for x in r2["data"]["rows"] if x["interval"] == iv), None)
check("3c slide_report re-measure on the same interval",
      row2 is not None and row2["drift_mm"] < 1.0, row2 and row2["drift_mm"])
outside = list(range(a - BL - 12, a - BL)) + list(range(b + BL + 1, b + BL + 13))
p_out0, _ = foot_world("R", outside)     # pre-write reference for check 4

# ---- reapply → pos+rot -------------------------------------------------------
r = call("reapply", op_id=op_id, overrides={"lock": "pos+rot"})
check("5 reapply lock=pos+rot (same op)", r["ok"] and r["data"]["id"] == op_id,
      r.get("summary"))
tracks = [t for t in rig.animation_data.nla_tracks if t.name == r["data"]["track"]]
check("5b one strip on the op track", tracks and len(tracks[0].strips) == 1)
pos2, q2 = foot_world("R", win)
ref_i = win.index(r["data"]["metrics"]["ref_frame"])
p_dev = np.linalg.norm(pos2[inner] - pos2[ref_i], axis=1).max() * 1000
r_dev = P.qangle_deg(q2[inner], np.tile(q2[ref_i], (len(win), 1))[inner]).max()
check("6 pos+rot: whole foot frozen", p_dev < 0.5 and r_dev < 0.05,
      f"pos dev {p_dev:.3f} mm rot dev {r_dev:.4f}°")

# ---- revert / dry_run / outside --------------------------------------------------
call("revert", op_id=op_id)
pos3, q3 = foot_world("R", win)
check("7 revert restores", np.abs(pos3 - pos0).max() * 1000 < 0.01
      and P.qangle_deg(q3, q0).max() < 0.01)
r = call("foot_lock", interval=iv, dry_run=True)
nt = len([t for t in rig.animation_data.nla_tracks if t.name.startswith("agent_footlock")])
check("8 dry_run writes nothing", r["ok"] and r["data"].get("dry_run") and nt == 0, nt)
r = call("foot_lock", side="R", frame_range=[a - BL, b + BL], lock="xy")
p_out1, _ = foot_world("R", outside)
check("4 frames outside the window unchanged", np.abs(p_out1 - p_out0).max() * 1000 < 0.01,
      f"{np.abs(p_out1 - p_out0).max() * 1000:.4f} mm")
call("revert", op_id=r["data"]["op_id"])
r = call("foot_lock", side="X", frame_range=[a, b])
check("9 bad side → clear error", not r["ok"] and "side" in r["error"]["message"],
      r.get("error", {}).get("message"))
ping = call("ping")
check("10 tools registered", {"foot_lock", "slide_report"} <= set(ping["data"]["tools"])
      and not ping["data"].get("plugin_errors"), ping["data"].get("plugin_errors"))

fails = [r for r in RESULTS if not r[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for n, _o, d in fails:
    print(f"FAIL {n}: {d}")
sys.exit(1 if fails else 0)
