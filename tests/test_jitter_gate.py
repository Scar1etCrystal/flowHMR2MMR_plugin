"""抖动闸门：analyze_motion 自己报警（用户 2026-10-06 要求）。

两条检查：
  ① 孤峰：某帧速度 / 前后两帧均值 > 2 → 这一帧突然抽了一下（不需要修前数据）
  ② 修前修后比值：有 baseline_op 时，jitter 或 peak_speed 涨过 2 倍 → 不通过

这里在 fixture 上真写一条"明显把动作变差"的修复（hold_pose 把手强行拧到别处），
验证闸门会报警；再验证一条无害窗口不会误报。
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
    return agent_bridge._dispatch({"tool": tool, "args": dict(args, agent_id="g")})


# ── 静默前提：这条链的窗口本身没有孤峰 ─────────────────────────────────
r0 = call("analyze_motion", chain="arm.L", frame_range=[120, 150], brief=True)
g0 = (r0.get("data") or {}).get("gate") or {}
check("gate 字段结构存在",
      "pass" in g0 and "thresholds" in g0 and "failed_bones" in g0,
      f"keys={sorted(g0.keys())} pass={g0.get('pass')}")
check("干净窗口不该报孤峰",
      not any("孤峰" in str(w) for w in (r0.get("warnings") or [])),
      str(r0.get("warnings"))[:120])

# ── 写一条故意把动作拧坏/拧快的修复 ────────────────────────────────────
w = call("hold_pose", bones=["hand_fk.L"], frame_range=[128, 142],
         target="world_dir", world_dir="up",
         world_axis="probe:palm.L", secondary_axis="probe:hand_axis.L",
         mode="replace", blend=4)
op = (w.get("data") or {}).get("op_id")
check("测试用修复写入成功", bool(op), f"op={op} keys={sorted((w.get('data') or {}).keys())}")

if op:
    r1 = call("analyze_motion", chain="arm.L", frame_range=[120, 150], brief=True,
              baseline_op=op)
    d1 = r1.get("data") or {}
    g1 = d1.get("gate") or {}
    bones = g1.get("bones") or {}
    hit = [b for b, v in bones.items() if v.get("jitter_ratio") or v.get("peak_ratio")]
    worst = max((v.get("jitter_ratio") or 0) for v in bones.values()) if bones else 0
    peak_ratio = max((v.get("peak_ratio") or 0) for v in bones.values()) if bones else 0
    check("闸门给出修前/修后比值",
          bool(bones) and any(("jitter_before" in v) for v in bones.values()),
          f"有比值的骨 {len([b for b,v in bones.items() if 'jitter_before' in v])} 根；"
          f"最大 jitter×{worst} peak×{peak_ratio}")
    warns = " ".join(str(x) for x in (r1.get("warnings") or []))
    check("闸门在变差时报警（warnings 里点名）",
          ("抖动闸门" in warns) or (not g1.get("pass")),
          f"pass={g1.get('pass')} failed={g1.get('failed_bones')} warn={warns[:150]}")
    print("\n  gate.bones 摘要：")
    for b, v in sorted(bones.items()):
        if "jitter_before" in v:
            print(f"    {b:<16} jitter {v['jitter_before']}→{v['jitter_after']}"
                  f"（×{v.get('jitter_ratio')}）  peak {v['peak_before']}→{v['peak_after']}"
                  f"（×{v.get('peak_ratio')}）")
    call("revert", op_id=op)

bad = [n for n, ok in RESULTS if not ok]
print(f"\n==== {len(RESULTS) - len(bad)}/{len(RESULTS)} PASS ====")
for n in bad:
    print("FAILED:", n)
if bad:
    sys.exit(1)
