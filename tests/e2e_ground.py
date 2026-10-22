"""e2e: ground_report（实时穿地/悬空复测）+ fix_ground（旧工具）在 IK 脚上的真实效果。

期望 ==== N/N PASS ====。fix_ground 读的是信号库快照里的脚底高度、写的是 foot_ik 的
location 通道；这里用 ground_report（与快照同一组脚底点、但读当前可见姿态）独立量修后。
"""
import os
import sys

import addon_utils
import bpy
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import agent_bridge  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig or next(o for o in scene.objects if o.type == "ARMATURE"
                               and o.name.startswith("RIG-"))
settings.mmr_rig = rig
data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_ground_data")
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


F0, F1 = int(scene.frame_start), int(scene.frame_end)
print("RIG matrix_world:", [list(map(lambda v: round(v, 4), row)) for row in rig.matrix_world])

# ---- G1: whole clip, nobody touched the feet → live == snapshot -------------------
r = call("ground_report", frame_range=[F0, F1])
check("G1 ground_report runs on the whole clip", r["ok"], r.get("summary"))
sides = r["data"]["sides"] if r["ok"] else {}
for s in ("L", "R"):
    d = sides.get(s, {}).get("snapshot_diff_max_mm")
    check(f"G1b {s}: live sole height == snapshot (untouched)", d is not None and d <= 0.01,
          f"snapshot_diff_max_mm={d}")
check("G1c no warnings when nothing was touched", r["ok"] and not r["warnings"], r.get("warnings"))

for s in ("L", "R"):
    cal = sides.get(s, {}).get("calibration", {})
    ch = sides.get(s, {}).get("contact_height_mm")
    check(f"G1d {s}: contact height calibrated from the snapshot's contact intervals",
          cal.get("contacts", 0) >= 3 and ch is not None and 20 <= ch <= 150,
          f"contact_height_mm={ch} calibration={cal}")

BL = 4


def first_suggestion(mode):
    for s in ("L", "R"):
        for a in sides.get(s, {}).get("fix_ground_args", []):
            if a["mode"] == mode:
                return s, {k: v for k, v in a.items() if k != "why"}
    return None, None


def rel_after(s, fr):
    g = call("ground_report", side=s, frame_range=fr, detail=True)
    return g, (np.asarray(g["data"]["sides"][s]["rel_mm"]) if g["ok"] else np.array([999.0]))


# ---- G2: a floating planted contact → lift to this foot's contact height -------------
s2, args = first_suggestion("lift")
check("G2 a floating contact is found and a lift suggestion is given", args is not None,
      {s: sum(1 for c in sides.get(s, {}).get("contacts", []) if c["floating"]) for s in ("L", "R")})
if args:
    lo, hi = args["frame_range"]
    _g, before = rel_after(s2, [lo + BL, hi - BL])
    r = call("fix_ground", **args)
    check("G2b fix_ground lift (suggested args, rest_clearance) writes", r["ok"],
          r.get("summary") or r.get("error"))
    g, after = rel_after(s2, [lo + BL, hi - BL])
    check("G2c after lift the contact sits at this foot's contact height (rel ≤ +1 mm)",
          float(after.max()) <= 1.0,
          f"rel {before.min():.1f}..{before.max():.1f} → {after.min():.1f}..{after.max():.1f} mm")
    check("G2d frames already at/below contact height are untouched (lift only pulls down)",
          float(np.abs(after[before <= 0] - before[before <= 0]).max(initial=0.0)) < 0.2,
          f"{int((before <= 0).sum())} frames")
    check("G2e ground_report warns that the snapshot is stale after the fix",
          g["ok"] and any("快照" in w for w in g["warnings"]), g.get("warnings"))
    if r["ok"]:
        call("revert", op_id=r["data"]["op_id"])
    _g, back = rel_after(s2, [lo + BL, hi - BL])
    check("G2f revert restores the original heights", float(np.abs(back - before).max()) < 0.05,
          f"max diff {float(np.abs(back - before).max()):.3f} mm")

# ---- G3: a sunk / penetrating stretch → push up (mode=pen) ---------------------------
s3, args = first_suggestion("pen")
if args:
    lo, hi = args["frame_range"]
    g0 = call("ground_report", side=s3, frame_range=[lo + BL, hi - BL])
    pm0 = g0["data"]["sides"][s3]["pen_max_mm"] if g0["ok"] else None
    r = call("fix_ground", **args)
    check("G3 fix_ground pen (suggested args) writes", r["ok"], r.get("summary") or r.get("error"))
    g = call("ground_report", side=s3, frame_range=[lo + BL, hi - BL])
    pm = g["data"]["sides"][s3]["pen_max_mm"] if g["ok"] else 99
    check("G3b after pen no frame is below the contact height (pen ≤ 1 mm)", pm <= 1.0,
          f"{s3} {lo + BL}–{hi - BL}: {pm0} mm → {pm} mm")
    if r["ok"]:
        call("revert", op_id=r["data"]["op_id"])
else:
    check("G3 no sunk stretch in this clip (pen path not exercised)", True,
          {s: sides.get(s, {}).get("pen_max_mm") for s in ("L", "R")})

# ---- G4/G5: mode names and backward compatibility ----------------------------------------
r = call("fix_ground", frame_range=[100, 140], side="R",
         loc_path='pose.bones["foot_ik.R"].location', mode="snap")
check("G4 the never-existing mode 'snap' fails with the real mode list",
      (not r["ok"]) and "float" in r["error"]["message"] and "pen" in r["error"]["message"],
      r.get("error", {}).get("message", "")[:120])
r = call("fix_ground", frame_range=[100, 140], side="R",
         loc_path='pose.bones["foot_ik.R"].location', mode="pen")
check("G5 without rest_clearance fix_ground behaves as before (no extra param recorded)",
      r["ok"] and "rest_clearance" not in r["data"].get("params_echo", r["data"].get("params", {})),
      r.get("summary") or r.get("error"))
if r["ok"]:
    call("revert", op_id=r["data"]["op_id"])

# ---- G8: describe / validate judge ground relative to the calibrated contact height ----
r = call("validate", frame_range=[150, 224])
ch = (r.get("data") or {}).get("contact_height_mm") or {}
viol = [(v["kind"], v["side"], len(v["frames"])) for v in (r.get("data") or {}).get("violations", [])]
check("G8 validate reports the calibration and judges against it",
      r["ok"] and ch.get("L") == sides["L"]["contact_height_mm"]
      and ch.get("R") == sides["R"]["contact_height_mm"]
      and not any(k == "floating" and s == "R" for k, s, _n in viol),
      f"contact_height_mm={ch} violations={viol}")
r = call("describe", target=[150, 224])
flags = [f for f in (r.get("data") or {}).get("flags", []) if f.startswith("foot")]
check("G8b describe's foot flags are relative ('比正常着地…'), no false R floating",
      r["ok"] and all("正常着地" in f for f in flags) and not any(f.startswith("foot.R") for f in flags),
      flags)

# ---- G9: mesh=true reports the real boot sole (target mesh) next to the joint view -------
r = call("ground_report", side="L", frame_range=[700, 770], mesh=True)
m = ((r.get("data") or {}).get("mesh") or {}).get("L") or {}
rows = [c for c in (r.get("data") or {}).get("sides", {}).get("L", {}).get("contacts", []) if "mesh_min_mm" in c]
check("G9 ground_report(mesh=true): boot-sole numbers per side and per full contact",
      r["ok"] and m.get("sole_min_mm") is not None and -100 < m["sole_min_mm"] < 100 and rows,
      {k: m.get(k) for k in ("sole_min_mm", "sole_min_frame", "frames_below_5mm", "contacts_median_min_mm")})

# ---- G7: typo'd parameters are rejected with the closest legal name -------------------
r = call("ground_report", frame_range=[700, 720], treshold_mm=5)
check("G7 ground_report(treshold_mm=…) → rejected, suggests threshold_mm",
      not r["ok"] and "threshold_mm" in r["error"]["message"], r.get("error", {}).get("message", "")[:140])
r = call("foot_lock", side="R", frame_range=[700, 720], lok="xy", dry_run=True)
check("G7b foot_lock(lok=…) → rejected, suggests lock",
      not r["ok"] and "lock" in r["error"]["message"], r.get("error", {}).get("message", "")[:140])

# ---- G6: a legacy file whose base strip lags 1 frame (and whose snapshot cache was baked
#           against the lag) - the settle must retire that snapshot, not keep using it ----------
from bl_ext.user_default.mocap_doctor.core import agent_bake, agent_io, agent_ops  # noqa: E402
anim = rig.animation_data
base = next((t.strips[0] for t in anim.nla_tracks if t.name == agent_ops.BASE_TRACK and t.strips), None)
check("G6 the base action sits on the NLA base track by now", base is not None)
if base is not None:
    base.action_frame_start = float(base.frame_start) - 1.0     # recreate the old 1-frame lag
    base.action_frame_end -= 1.0
    cache = agent_bake.bake_cache_path(data_dir, rig.name, F0, F1, tag="rig")
    lagged = cache.with_suffix(".lagged")
    for f in (cache, lagged):
        if f.exists():
            f.unlink()                                          # this test's own scratch cache
    agent_io.build_store_for_scene(scene, rig, settings, spec=agent_io.rig_bake_spec(rig),
                                   data_dir=data_dir, use_cache=True, tag="rig")
    check("G6b a snapshot was baked against the lagging base", cache.exists(), cache.name)
    agent_bridge._STORE = None                                  # a fresh session
    r = call("ground_report", frame_range=[700, 770])
    diffs = {s: r["data"]["sides"][s].get("snapshot_diff_max_mm") for s in ("L", "R")} if r["ok"] else {}
    check("G6c the first read settles the base, retires the lagged snapshot and re-bakes",
          lagged.exists() and abs(float(base.action_frame_start) - float(base.frame_start)) < 1e-4,
          f"lagged={lagged.exists()} afs={base.action_frame_start} fs={base.frame_start}")
    check("G6d the new snapshot matches the live pose again", diffs and max(diffs.values()) <= 0.01,
          diffs)

fails = [x for x in RESULTS if not x[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for name, _ok, detail in fails:
    print("FAILED:", name, "::", detail)
sys.exit(1 if fails else 0)
