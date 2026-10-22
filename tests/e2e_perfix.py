"""Headless e2e for the per-fix strength / per-op-track redesign.

Runs in Blender on a COPY of the fixture.  Checks, in order:
  1  two overlapping fixes both write (regression: used to fail "no space")
  2  each fix gets its own track; op log records it
  3  per-fix exponent scales that fix's delta (1.0 vs 1.6 / 0.4)
  4  muting one fix leaves the other alone
  5  legacy shared-track layout migrates, idempotently
  6  reconcile flags a vanished strip as "lost" and an orphan as "unregistered"
  7  exponent survives save + reopen (action IDProp)
"""
import json
import math
import os
import sys

import bpy
import addon_utils

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import agent_ops  # noqa: E402
from bl_ext.user_default.mocap_doctor.core import agent_bridge  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig
if rig is None or rig.type != "ARMATURE":
    rig = next((o for o in scene.objects
                if o.type == "ARMATURE" and o.name.startswith("RIG-")), None)
settings.mmr_rig = rig
print("RIG:", rig.name if rig else None)

data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_fix_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)

BONE_A = "f_index.01.L"
BONE_B = "f_middle.01.L"
FRAMES = (528, 535, 542, 549, 556)


def basis(bone, frame):
    scene.frame_set(frame)
    return rig.pose.bones[bone].matrix_basis.to_quaternion().copy()


def angle_between(q0, q1):
    d = min(1.0, abs(float(q0.normalized().dot(q1.normalized()))))
    return math.degrees(2 * math.acos(d))


def fix_angle(bone, track, frames=FRAMES):
    """Max rotation this fix applies to `bone` over `frames` (on vs muted)."""
    track.mute = False
    on = [basis(bone, f) for f in frames]
    track.mute = True
    off = [basis(bone, f) for f in frames]
    track.mute = False
    return max(angle_between(a, b) for a, b in zip(on, off))


def tracks():
    anim = rig.animation_data
    return [(t.name, t.mute, [(s.name, s.action.name if s.action else None,
                               int(s.frame_start), int(s.frame_end))
                              for s in t.strips])
            for t in (anim.nla_tracks if anim else ())]


def agent_track_names():
    anim = rig.animation_data
    return [t.name for t in (anim.nla_tracks if anim else ())
            if agent_ops.is_agent_track_name(t.name)]


def mute_only(*names):
    anim = rig.animation_data
    for t in (anim.nla_tracks if anim else ()):
        if agent_ops.is_agent_track_name(t.name):
            t.mute = t.name not in names


base = agent_bridge._base_action(rig)

# ---------------------------------------------------------------- 1 overlap
op1 = agent_ops.hold_pose(rig, base, [BONE_A], [505, 570],
                          target="values", mode="replace", blend=6,
                          data_dir=data_dir)
op2 = agent_ops.hold_pose(rig, base, [BONE_B], [520, 560],
                          target="values", mode="replace", blend=6,
                          data_dir=data_dir)
check("1 重叠写入都成功", bool(op1.get("id") and op2.get("id")),
      f"{op1['id']} / {op2['id']}")
check("1b op id 唯一", op1["id"] != op2["id"] and "op2" not in op1["id"],
      f"{op1['id']} / {op2['id']}")

# ---------------------------------------------------------------- 2 tracks
names = agent_track_names()
check("2 每条修复一条轨", len(names) == 2, f"tracks={names}")
check("2b log 记录 track",
      bool(op1.get("track")) and bool(op2.get("track"))
      and op1["track"] != op2["track"],
      f"{op1.get('track')} / {op2.get('track')}")
check("2c 轨名=strip 名",
      op1["track"] == op1["strip"] and op2["track"] == op2["strip"],
      f"{op1['strip']} == {op1['track']}")
print("TRACKS:", json.dumps(tracks(), ensure_ascii=False))

# ---------------------------------------------------------------- 3 exponent
mute_only(op1["track"])
track_a, strip_a = agent_ops.find_op_strip(rig, op1)
track_b, strip_b = agent_ops.find_op_strip(rig, op2)

a1 = fix_angle(BONE_A, track_a)
agent_ops.set_strip_exponent(strip_a, 1.6)
a2 = fix_angle(BONE_A, track_a)
ratio = a2 / a1 if a1 > 1e-6 else 0.0
check("3 力度 1.6 放大该修复", a1 > 3.0 and 1.35 <= ratio <= 1.85,
      f"1.0→{a1:.1f}° 1.6→{a2:.1f}° ratio={ratio:.2f}")
check("3b applied_exp 落在 action 上",
      abs(float(strip_a.action.get("applied_exp", 0)) - 1.6) < 1e-4,
      f"exp={strip_a.action.get('applied_exp')}")
agent_ops.set_strip_exponent(strip_a, 1.0)

mute_only(op2["track"])
b1 = fix_angle(BONE_B, track_b)
agent_ops.set_strip_exponent(strip_b, 0.4)
b2 = fix_angle(BONE_B, track_b)
ratio_b = b2 / b1 if b1 > 1e-6 else 0.0
check("3c 力度 0.4 减弱该修复", b1 > 3.0 and 0.25 <= ratio_b <= 0.55,
      f"1.0→{b1:.1f}° 0.4→{b2:.1f}° ratio={ratio_b:.2f}")
agent_ops.set_strip_exponent(strip_b, 1.0)

# ---------------------------------------------------------------- 4 mute one
mute_only()                                  # 全静音 = 基准姿态
base_a = [basis(BONE_A, f) for f in FRAMES]
base_b = [basis(BONE_B, f) for f in FRAMES]
mute_only(op2["track"])                      # B 在响，A 被静音
a_muted = max(angle_between(x, basis(BONE_A, f))
              for x, f in zip(base_a, FRAMES))
b_live = max(angle_between(x, basis(BONE_B, f))
             for x, f in zip(base_b, FRAMES))
check("4 单条静音只影响自己", a_muted < 0.5 and b_live > 3.0,
      f"A(被静音)={a_muted:.2f}° B(在响)={b_live:.2f}°")

# ---------------------------------------------------------------- 5 migrate
mute_only()
legacy = agent_ops.hold_pose(rig, base, [BONE_A], [600, 620],
                             target="values", mode="replace", blend=4,
                             data_dir=data_dir, track_name="AGENT_PREVIEW")
check("5a 旧布局写入成功", legacy.get("track") == "AGENT_PREVIEW",
      f"track={legacy.get('track')}")
_tl, strip_legacy = agent_ops.find_op_strip(rig, legacy)
legacy_action = strip_legacy.action.name
legacy_range = (int(strip_legacy.frame_start), int(strip_legacy.frame_end))
legacy_strip_name = strip_legacy.name
mig1 = agent_ops.migrate_legacy_tracks(rig, data_dir)
logged_pre = {o["id"]: o for o in agent_ops.list_ops(data_dir)}
track_after, strip_after = agent_ops.find_op_strip(rig, logged_pre[legacy["id"]])
mig2 = agent_ops.migrate_legacy_tracks(rig, data_dir)
check("5b 迁移把 strip 挪到专属轨", mig1.get("moved") == 1
      and "AGENT_PREVIEW" not in [t[0] for t in tracks()],
      f"moved={mig1.get('moved')} tracks={[t[0] for t in tracks()]}")
check("5c 迁移幂等", mig2.get("moved") == 0, f"second moved={mig2.get('moved')}")
logged = {o["id"]: o for o in agent_ops.list_ops(data_dir)}
check("5d 迁移保内容",
      strip_after is not None and strip_after.action.name == legacy_action
      and (int(strip_after.frame_start), int(strip_after.frame_end)) == legacy_range
      and logged[legacy["id"]].get("track") == track_after.name,
      f"{legacy_strip_name}→{track_after.name if track_after else None} "
      f"{legacy_action} {legacy_range} log_track="
      f"{logged[legacy['id']].get('track')}")

# ---------------------------------------------------------------- 6 reconcile
rows = agent_ops.reconcile(rig, data_dir)
check("6a 对账列出全部修复", len([r for r in rows if r["op_id"]]) == 3,
      f"rows={[(r['op_id'], r['status']) for r in rows]}")

track_x, strip_x = agent_ops.find_op_strip(rig, op1)
track_x.strips.remove(strip_x)               # simulate the user undoing it away
rows2 = agent_ops.reconcile(rig, data_dir)
row1 = next(r for r in rows2 if r["op_id"] == op1["id"])
check("6b 场景没有 → 标丢失", row1["status"] == "lost" and not row1["alive"],
      f"status={row1['status']}")

orphan = rig.animation_data.nla_tracks.new()
orphan.name = "agent_orphan"
orphan.strips.new("agent_orphan", 700, strip_b.action)
rows3 = agent_ops.reconcile(rig, data_dir)
unreg = [r for r in rows3 if r["status"] == "unregistered"]
check("6c 场景有/log 无 → 未登记", len(unreg) == 1,
      f"unregistered={[r['strip'] for r in unreg]}")
agent_ops.delete_op_strip(rig, {"strip": "agent_orphan", "track": "agent_orphan"})

# ---------------------------------------------------------------- 7 reload
agent_ops.set_strip_exponent(strip_b, 1.37)
save_path = os.path.join(data_dir, "e2e_saved.blend")
saved_strip_name = strip_b.name
bpy.ops.wm.save_as_mainfile(filepath=save_path)
bpy.ops.wm.open_mainfile(filepath=save_path)
scene2 = bpy.context.scene
rig2 = scene2.mocap_doctor.mmr_rig
strip2 = None
for t in (rig2.animation_data.nla_tracks if rig2.animation_data else ()):
    for s in t.strips:
        if s.name == saved_strip_name:
            strip2 = s
check("7 存盘重开力度仍在",
      strip2 is not None and strip2.action is not None
      and abs(float(strip2.action.get("applied_exp", 0)) - 1.37) < 1e-4,
      f"exp={strip2.action.get('applied_exp') if strip2 and strip2.action else None}")

fails = [r for r in RESULTS if not r[1]]
print(f"\n=== {len(RESULTS) - len(fails)}/{len(RESULTS)} passed ===")
for name, _ok, detail in fails:
    print("FAILED:", name, "::", detail)
if fails:
    sys.exit(1)
