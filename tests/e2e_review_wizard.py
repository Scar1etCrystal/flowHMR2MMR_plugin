"""Wizard / project-side regressions from the 2026-10-04 read-only review.

Each check reproduces the finding's scenario on the e2e fixture (never saved):

  W1  M1   sole_contact_offsets picks each ankle's sole on its OWN half in
           armature space (the fixture sits at world X~-2.56: the old world-X
           split gave both feet the same left-boot vertex, R vector 15.5 cm off)
  W2  M12  the foot_lock step hands settings.target_floor_z to
           stabilize_planted_feet (heavy calls stubbed: wiring only)
  W3  M6   MMD Bake's "foreign NLA track" check ignores EMPTY tracks (the
           user's RIG has 103 empty NlaTrack.*); a track WITH a strip still blocks
  W4  M7   set_fcurve_value / add_missing_keys on a SPARSE curve: every write
           lands, old keys keep their values (cached Keyframe pointers dangled
           after insert()/add() reallocated the key array)
  W5  M19  "载入自动腾空提示" refills a manually cleared air track
  W6  M14  the same reload finds the contacts report through a DEAD absolute
           path (project moved) under <data dir>/reports/
  W7  P4   planted-indicator refresh with nothing changed tags no object

    bash tools/mcd.sh e2e tests/e2e_review_wizard.py
"""
import json
import os
import shutil
import sys

import addon_utils
import bpy

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor import annotation, project  # noqa: E402
from bl_ext.user_default.mocap_doctor import operators as OPS  # noqa: E402
from bl_ext.user_default.mocap_doctor.core import agent_bridge  # noqa: E402
from bl_ext.user_default.mocap_doctor.core import animation as ANIM  # noqa: E402
from bl_ext.user_default.mocap_doctor.core import target as T  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


scene = bpy.context.scene
settings = scene.mocap_doctor
data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_review_wizard_data")
shutil.rmtree(data_dir, ignore_errors=True)
for sub in ("reports", "checkpoints"):
    os.makedirs(os.path.join(data_dir, sub), exist_ok=True)
settings.data_directory = data_dir
arm = settings.mmd_armature
rig = settings.mmr_rig

# ---- W1 / M1 ------------------------------------------------------------------
dirs = T.sole_contact_offsets(arm, settings.target_mesh)
dl, dr = dirs["L"], dirs["R"]
inv = arm.matrix_world.inverted()
side_ok = True
for side, d in (("L", dl), ("R", dr)):
    bone = arm.data.bones[T.DEFAULT_ANKLE_BONES[side]]
    rest = (arm.matrix_world @ bone.matrix_local).to_quaternion()
    sole_w = arm.matrix_world @ bone.head_local + rest @ d
    side_ok &= ((inv @ sole_w).x * bone.head_local.x) > 0.0
check("W1a each sole point lies on its own ankle's half (armature space)", side_ok,
      f"L={tuple(round(v, 4) for v in dl)} R={tuple(round(v, 4) for v in dr)} "
      f"rig_x={arm.matrix_world.translation.x:.3f}")
mirror_err = max(abs(dl.x + dr.x), abs(dl.y - dr.y), abs(dl.z - dr.z))
check("W1b L/R ankle→sole vectors mirror each other (< 2 mm)", mirror_err < 0.002,
      f"mirror_err={mirror_err * 1000:.2f} mm (old R vector: 155 mm off)")

# ---- W2 / M12 -----------------------------------------------------------------
captured = {}
saved = {
    "drift": T.analyze_foot_ik_drift, "lock": T.lock_foot_ik_xy,
    "stab": T.stabilize_planted_feet, "preview": project.begin_action_preview,
    "report": OPS._record_report, "rerun": OPS._require_restore_before_rerun,
}
floor_before = settings.target_floor_z
try:
    T.analyze_foot_ik_drift = lambda *a, **k: {}
    T.lock_foot_ik_xy = lambda *a, **k: {"repaired_count": 0}

    def _fake_stab(*a, **k):
        captured.update(k)
        return {"stabilized_segments": 0}

    T.stabilize_planted_feet = _fake_stab
    project.begin_action_preview = lambda scene_, owner, step_id: owner.animation_data.action
    OPS._record_report = lambda *a, **k: None
    OPS._require_restore_before_rerun = lambda *a, **k: None
    settings.target_floor_z = 0.0123
    OPS._run_foot_lock(bpy.context, settings)
finally:
    T.analyze_foot_ik_drift = saved["drift"]
    T.lock_foot_ik_xy = saved["lock"]
    T.stabilize_planted_feet = saved["stab"]
    project.begin_action_preview = saved["preview"]
    OPS._record_report = saved["report"]
    OPS._require_restore_before_rerun = saved["rerun"]
    settings.target_floor_z = floor_before
check("W2 foot_lock passes 地面 Z to the closed-loop stabilizer",
      abs(float(captured.get("floor_z", -1.0)) - 0.0123) < 1e-9,
      f"floor_z={captured.get('floor_z')}")

# ---- W3 / M6 ------------------------------------------------------------------
REACHED = "REACHED_PAST_THE_FOREIGN_TRACK_CHECK"


def _sentinel(_rig):
    raise RuntimeError(REACHED)


def _op(fn):
    """Run an operator; an ERROR report raises in background mode → message."""
    try:
        return str(fn())
    except RuntimeError as exc:
        return f"EXC {exc}"


anim = rig.animation_data
empties = []
for _ in range(3):
    tr = anim.nla_tracks.new()
    empties.append(tr.name)
orig_base = agent_bridge._base_action
agent_bridge._base_action = _sentinel
try:
    _op(bpy.ops.mocap_doctor.mmd_bake)
    msg_empty = settings.status_message
    # a foreign track that really animates something must still block
    tr = anim.nla_tracks.new()
    blocker = tr.name
    act = bpy.data.actions.new("e2e_foreign_strip")
    act.fcurves.new("location", index=2).keyframe_points.insert(1, 0.0)
    tr.strips.new("e2e_foreign", 1, act)
    _op(bpy.ops.mocap_doctor.mmd_bake)
    msg_full = settings.status_message
finally:
    agent_bridge._base_action = orig_base
    for tr in list(anim.nla_tracks):
        if tr.name in empties or tr.name == blocker:
            anim.nla_tracks.remove(tr)
    if bpy.data.actions.get("e2e_foreign_strip"):
        bpy.data.actions.remove(bpy.data.actions["e2e_foreign_strip"])
check("W3a 3 empty NlaTracks no longer block MMD Bake", REACHED in msg_empty, msg_empty[:80])
# the wizard's "no NLA" guard (ground_feet / export prep) follows the same rule
tmp_tracks = [anim.nla_tracks.new().name for _ in range(2)]
try:
    OPS._require_no_nla(rig, "MMR Rig")
    empty_ok = True
except RuntimeError:
    empty_ok = False
finally:
    for tr in list(anim.nla_tracks):
        if tr.name in tmp_tracks:
            anim.nla_tracks.remove(tr)
check("W3c _require_no_nla ignores empty tracks too", empty_ok, f"empty tracks={tmp_tracks}")
check("W3b a foreign track WITH a strip still blocks", "陌生的 NLA Track" in msg_full
      and blocker in msg_full, msg_full[:80])

# ---- W4 / M7 ------------------------------------------------------------------
act = bpy.data.actions.new("e2e_sparse_keys")
try:
    fc = act.fcurves.new("location", index=0)
    for f, v in ((1, 100.0), (10, 200.0), (20, 300.0)):
        fc.keyframe_points.insert(f, v, options={"FAST"})
    cache = ANIM.keyframe_map(fc)
    for f in range(1, 21):
        ANIM.set_fcurve_value(fc, f, f * 0.5, cache=cache)
    got = [(round(k.co.x), round(k.co.y, 4)) for k in fc.keyframe_points]
    want = [(f, f * 0.5) for f in range(1, 21)]
    check("W4a sparse curve: every set_fcurve_value lands (no stale key pointers)",
          got == want, f"got[8:11]={got[8:11]}")
    fc2 = act.fcurves.new("location", index=1)
    for f, v in ((5, 7.0), (15, 9.0)):
        fc2.keyframe_points.insert(f, v, options={"FAST"})
    cache2 = ANIM.keyframe_map(fc2)
    added = ANIM.add_missing_keys(fc2, range(1, 21), cache2)
    keep = {round(k.co.x): k.co.y for k in fc2.keyframe_points}
    check("W4b add_missing_keys: new keys are the appended ones, old keys untouched",
          added == 18 and len(fc2.keyframe_points) == 20 and keep.get(5) == 7.0
          and keep.get(15) == 9.0 and sorted(keep) == list(range(1, 21)),
          f"added={added} k5={keep.get(5)} k15={keep.get(15)} n={len(keep)}")
    for f in range(1, 21):
        key = cache2.get(f)
        key.co.y = float(f)
    vals = [k.co.y for k in fc2.keyframe_points]
    check("W4c the rebuilt cache addresses the right keys", vals == [float(f) for f in range(1, 21)],
          f"{vals[:6]}")
finally:
    bpy.data.actions.remove(act)

# ---- W5 / M19 + W6 / M14 ------------------------------------------------------
air_before = annotation.get_channel_ranges(scene, annotation.CHANNEL_AIR)
record = project.find_step_record(settings, "contacts", create=False)
artifact_before = record.artifact_path if record is not None else ""
report = {"feet": {s: {"per_frame": {
    str(f): {"state": "airborne_or_lifted" if 300 <= f <= 304 else "planted"}
    for f in range(int(settings.mocap_frame_start), int(settings.mocap_frame_end) + 1)}}
    for s in ("L", "R")}}
report_path = os.path.join(data_dir, "reports", "0000_contacts.json")
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(report, handle)
try:
    if record is None:
        record = project.find_step_record(settings, "contacts", create=True)
    record.artifact_path = report_path
    annotation.set_air_ranges(scene, [], rebuild=False)      # the user cleared the track
    r = _op(bpy.ops.mocap_doctor.reload_air_hints)
    air = annotation.get_channel_ranges(scene, annotation.CHANNEL_AIR)
    check("W5 reload refills a cleared air track (initialized flag set)",
          "FINISHED" in r and air == [(300, 304)], f"{r} air={air}")
    annotation.set_air_ranges(scene, [], rebuild=False)
    # the project moved: the stored absolute path is dead, the file sits in reports/
    record.artifact_path = "F:\\old_drive\\proj\\.mocap_doctor\\x\\reports\\0000_contacts.json"
    r = _op(bpy.ops.mocap_doctor.reload_air_hints)
    air = annotation.get_channel_ranges(scene, annotation.CHANNEL_AIR)
    check("W6 a dead absolute report path falls back to <data dir>/reports/",
          "FINISHED" in r and air == [(300, 304)], f"{r} air={air}")
finally:
    record.artifact_path = artifact_before
    annotation.set_air_ranges(scene, air_before, rebuild=False)
    shutil.rmtree(data_dir, ignore_errors=True)

# ---- W7 / P4 ------------------------------------------------------------------
# planted indicators: refresh() runs on every frame change AND inside
# depsgraph_update_post; an unconditional hide_render write re-tagged the objects
# every time (self-triggered re-evaluation).  Unchanged state = no update.
from bl_ext.user_default.mocap_doctor import planted_indicators as PI  # noqa: E402
seen = []


def _count(_scene, depsgraph):
    seen.extend(u.id.name for u in depsgraph.updates
                if u.id.name.startswith(PI.OBJECT_PREFIX))


scene.mcd_annotation_mode = True
settings.annotation_step_id = "contacts"
try:
    PI.activate(scene)
    bpy.context.view_layer.update()
    bpy.app.handlers.depsgraph_update_post.append(_count)
    bpy.context.view_layer.update()
    seen.clear()
    for _ in range(3):
        PI.refresh(scene)
        bpy.context.view_layer.update()
    n_idle = len(seen)
    made = len(PI._owned_objects(scene))
finally:
    if _count in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_count)
    PI.cleanup(scene)
    scene.mcd_annotation_mode = False
    settings.annotation_step_id = ""
check("W7 P4 indicator refresh with nothing changed tags no object",
      made == 2 and n_idle == 0, f"indicators={made} updates_while_idle={n_idle}")

fails = [r for r in RESULTS if not r[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for name, _ok, detail in fails:
    print("FAILED:", name, "::", detail)
if fails:
    sys.exit(1)
