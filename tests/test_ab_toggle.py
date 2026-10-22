"""ab_toggle 的正交/幂等语义（2026-10-06 新增 `muted` 显式状态）。

老语义是"只要有一个还响着 → 全部静音；全静音了 → 全部恢复"（翻转）。于是
"静音 → 写新 op → 恢复"里的"恢复"会把**刚写的新 op 一起静音**，下一轮的"静音"
反而变成取消静音——w5 跑完时 10 条全落成 muted=true。

`muted=true/false` = 显式目标状态，调几次都一样；不给 = 老的翻转行为（向后兼容）。
"""
import sys

import addon_utils
import bpy

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import agent_bridge  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


def call(tool, **args):
    return agent_bridge._dispatch({"tool": tool, "args": dict(args, agent_id="ab")})


def agent_mutes():
    arm = next(o for o in bpy.context.scene.objects
               if o.type == "ARMATURE" and o.name.startswith("RIG-"))
    anim = arm.animation_data
    return {t.name: bool(t.mute) for t in anim.nla_tracks if t.name.startswith("agent_")}


# ── 两条互不相干的修复 ─────────────────────────────────────────────────
o1 = (call("hold_pose", bones=["hand_fk.L"], frame_range=[120, 132],
           target="world_dir", world_dir="up", world_axis="probe:palm.L",
           secondary_axis="probe:hand_axis.L", mode="replace", blend=3)
      .get("data") or {}).get("op_id")
o2 = (call("hold_pose", bones=["hand_fk.R"], frame_range=[220, 232],
           target="world_dir", world_dir="up", world_axis="probe:palm.R",
           secondary_axis="probe:hand_axis.R", mode="replace", blend=3)
      .get("data") or {}).get("op_id")
check("两条修复写入成功", bool(o1 and o2), f"{o1} / {o2}")

if o1 and o2:
    check("前提：两条都不静音", not any(agent_mutes().values()), str(agent_mutes()))

    r1 = call("ab_toggle", muted=True)
    m1 = agent_mutes()
    check("muted=true → 全部静音", all(m1.values()) and m1, f"{r1.get('data')}")

    r2 = call("ab_toggle", muted=True)          # 幂等：再来一次还是全静音
    m2 = agent_mutes()
    check("muted=true 幂等（重复调用不翻转）", all(m2.values()), str(m2))

    # 关键场景：静音状态下写一条新 op，再"恢复"——老语义会把新 op 一起静音
    o3 = (call("hold_pose", bones=["hand_fk.L"], frame_range=[150, 162],
               target="world_dir", world_dir="up", world_axis="probe:palm.L",
               secondary_axis="probe:hand_axis.L", mode="replace", blend=3)
          .get("data") or {}).get("op_id")
    call("ab_toggle", muted=False)              # 显式恢复
    m3 = agent_mutes()
    check("静音→写新 op→muted=false：新旧全部恢复（老语义在这里会把新 op 静音）",
          not any(m3.values()), f"新 op={o3} 状态={m3}")

    r4 = call("ab_toggle", muted=False)
    check("muted=false 幂等", not any(agent_mutes().values()), str(r4.get("data")))

    # 向后兼容：不给 muted 时仍是翻转
    call("ab_toggle")
    a = all(agent_mutes().values())
    call("ab_toggle")
    b = not any(agent_mutes().values())
    check("不给 muted 时保留老的翻转行为（向后兼容）", a and b,
          f"第一次后全静音={a} 第二次后全恢复={b}")

    for op in (o1, o2, o3):
        if op:
            call("revert", op_id=op)

bad = [n for n, ok in RESULTS if not ok]
print(f"\n==== {len(RESULTS) - len(bad)}/{len(RESULTS)} PASS ====")
for n in bad:
    print("FAILED:", n)
if bad:
    sys.exit(1)
