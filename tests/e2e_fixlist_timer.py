"""Headless test of the fixlist timer path (the draw-safe auto-sync).

Simulates the user's exact live scenario: a legacy AGENT_PREVIEW strip with an
op log that has no `track` field, then the first tick must migrate + build the
panel rows; the second tick must be a stable no-op.
"""
import os
import sys

import bpy
import addon_utils

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import agent_ops, agent_bridge  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


scene = bpy.context.scene
settings = scene.mocap_doctor
if not settings.initialized:
    settings.initialized = True      # timer guard requires an active project
rig = settings.mmr_rig or next(
    (o for o in scene.objects if o.type == "ARMATURE" and o.name.startswith("RIG-")), None)
settings.mmr_rig = rig

data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_tick_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)

# ---- user's live layout: one strip on the legacy shared preview track
op = agent_ops.hold_pose(rig, agent_bridge._base_action(rig),
                         ["f_index.01.L"], [505, 570],
                         target="values", mode="replace", blend=6,
                         data_dir=data_dir, track_name="AGENT_PREVIEW")
anim = rig.animation_data
before = [t.name for t in anim.nla_tracks if agent_ops.is_agent_track_name(t.name)]
check("T0 旧布局就位", before == ["AGENT_PREVIEW"],
      f"tracks={before}")

# ---- first tick: migrate + build
agent_bridge._fixlist_tick()
after = [t.name for t in anim.nla_tracks if agent_ops.is_agent_track_name(t.name)]
check("T1 tick 完成迁移", after == ["agent_hold_505_570"], f"tracks={after}")
check("T2 列表建好", len(settings.agent_fixes) == 1,
      f"rows={len(settings.agent_fixes)}")
item = settings.agent_fixes[0]
logged = {o["id"]: o for o in agent_ops.list_ops(data_dir)}
check("T3 行与 log 对上",
      item.op_id == op["id"] and item.track == "agent_hold_505_570"
      and logged[op["id"]].get("track") == "agent_hold_505_570"
      and item.status == "preview",
      f"{item.op_id} track={item.track} exp={item.exponent}")

# ---- second tick: stable no-op
rows_before = [(i.op_id, i.track, i.exponent, i.muted) for i in settings.agent_fixes]
agent_bridge._fixlist_tick()
rows_after = [(i.op_id, i.track, i.exponent, i.muted) for i in settings.agent_fixes]
check("T4 二次 tick 稳定", rows_before == rows_after, f"{rows_after}")

# ---- rev bump → next tick picks up a new op
op2 = agent_ops.hold_pose(rig, agent_bridge._base_action(rig),
                          ["f_middle.01.L"], [600, 640],
                          target="values", mode="replace", blend=4,
                          data_dir=data_dir)
settings.agent_ops_rev += 1          # bridge 写工具会做这一步，这里手动模拟
agent_bridge._fixlist_tick()
check("T5 rev 变化后跟上", len(settings.agent_fixes) == 2,
      f"rows={len(settings.agent_fixes)} "
      f"tracks={[i.track for i in settings.agent_fixes]}")

# ---- timer registration helpers (register/unregister must not throw)
agent_bridge.start_fixlist_timer()
registered = bpy.app.timers.is_registered(agent_bridge._fixlist_tick)
agent_bridge.stop_fixlist_timer()
unregistered = not bpy.app.timers.is_registered(agent_bridge._fixlist_tick)
agent_bridge.start_fixlist_timer()
check("T6 定时器启停正常", registered and unregistered,
      f"registered={registered} unregistered={unregistered}")

fails = [r for r in RESULTS if not r[1]]
print(f"\n=== {len(RESULTS) - len(fails)}/{len(RESULTS)} passed ===")
for name, _ok, detail in fails:
    print("FAILED:", name, "::", detail)
if fails:
    sys.exit(1)
