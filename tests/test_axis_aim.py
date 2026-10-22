"""膝盖/肘朝空物体 + 按轴瞄准（axis）。

用户 2026-10-06："膝盖朝空物体，注意是 Z 方向的旋转，就是膝盖和空物体可以有高低差"，
并要求"也要考虑会不会有只转 X 或 Y"。

axis="z" = 把瞄准方向投影到 ⊥Z 的平面（只用水平方向，忽略高低差）；x / y 同理。
axis=None（默认）= 原样三维瞄准，和以前逐位一致。
"""
import math
import sys

import addon_utils
import bpy

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import agent_bridge, agent_view  # noqa: E402
from mathutils import Vector  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


def call(tool, **args):
    return agent_bridge._dispatch({"tool": tool, "args": dict(args, agent_id="ax")})


scene = bpy.context.scene
arm = next(o for o in scene.objects if o.type == "ARMATURE" and o.name.startswith("RIG-"))

# 摆一个在右前方、且比膝盖高的空物体（高低差正是要忽略的那一维）
emp = bpy.data.objects.get("AXIS_TEST_EMPTY")
if emp is None:
    emp = bpy.data.objects.new("AXIS_TEST_EMPTY", None)
    scene.collection.objects.link(emp)
emp.location = (1.2, -0.8, 1.6)          # x=右, y=前, z=比膝盖高一大截
bpy.context.view_layer.update()

origin = Vector((0.0, 0.0, 0.5))         # 膝盖附近

d3, why3 = agent_view.resolve_direction("AXIS_TEST_EMPTY", scene=scene, armature=arm, origin=origin)
dz, whyz = agent_view.resolve_direction("AXIS_TEST_EMPTY", scene=scene, armature=arm, origin=origin, axis="z")
dx, _ = agent_view.resolve_direction("AXIS_TEST_EMPTY", scene=scene, armature=arm, origin=origin, axis="x")
dy, _ = agent_view.resolve_direction("AXIS_TEST_EMPTY", scene=scene, armature=arm, origin=origin, axis="y")

check("axis=None 保持三维（默认行为不变）",
      abs(d3.z) > 0.1, f"dir=({d3.x:.3f},{d3.y:.3f},{d3.z:.3f}) why={why3}")
check('axis="z" → z 分量为 0（忽略高低差）',
      abs(dz.z) < 1e-9 and abs(dz.x) > 0.1 and abs(dz.y) > 0.1,
      f"dir=({dz.x:.3f},{dz.y:.3f},{dz.z:.3f}) why={whyz}")
check('axis="x" → x 分量为 0', abs(dx.x) < 1e-9 and abs(dx.z) > 0.1,
      f"({dx.x:.3f},{dx.y:.3f},{dx.z:.3f})")
check('axis="y" → y 分量为 0', abs(dy.y) < 1e-9 and abs(dy.z) > 0.1,
      f"({dy.x:.3f},{dy.y:.3f},{dy.z:.3f})")
check("水平投影仍是单位向量", abs(dz.length - 1.0) < 1e-6, f"|dir|={dz.length:.6f}")

# 方向词也能按轴投影
fz, _ = agent_view.resolve_direction("forward", scene=scene, armature=arm, origin=origin, axis="z")
check("方向词也支持按轴投影", abs(fz.z) < 1e-6, f"forward⊥Z=({fz.x:.3f},{fz.y:.3f},{fz.z:.3f})")

# 非法 axis
try:
    agent_view.resolve_direction("forward", scene=scene, armature=arm, origin=origin, axis="w")
    check("非法 axis 报错", False, "没有报错")
except RuntimeError as exc:
    check("非法 axis 给出可读错误", "只能是" in str(exc) and "axis" in str(exc), str(exc)[:90])

# 目标正好在轴上 → 投影长度为零 → 清楚报错
emp.location = (0.0, 0.0, 3.0)
bpy.context.view_layer.update()
try:
    agent_view.resolve_direction("AXIS_TEST_EMPTY", scene=scene, armature=arm, origin=origin, axis="z")
    check("目标在正上方时 axis=z 报错", False, "没有报错")
except RuntimeError as exc:
    check("目标在正上方时 axis=z 给出可读错误",
          "长度为零" in str(exc), str(exc)[:90])
emp.location = (1.2, -0.8, 1.6)
bpy.context.view_layer.update()

# ── swivel 真的接受 axis，而且 dry_run 不写场景 ────────────────────────
r = call("swivel", joint="knee", side="R", frame_range=[940, 960], toward="AXIS_TEST_EMPTY",
         axis="z", strength=0.3, blend=2, dry_run=True)
d = r.get("data") or {}
check("swivel 接受 axis 并 dry_run 通过",
      bool(r.get("ok")) and bool(d.get("dry_run")),
      f"ok={r.get('ok')} pred={((d.get('metrics') or {}).get('pred_swivel_max_deg'))} "
      f"err={str(r.get('error'))[:80]}")

w = call("swivel", joint="knee", side="R", frame_range=[940, 960], toward="AXIS_TEST_EMPTY",
         axis="z", strength=0.3, blend=2)
wd = w.get("data") or {}
op = wd.get("op_id")
check("真写时 axis 进了 params（reapply 才能还原）",
      bool(op) and (wd.get("params_echo") or {}).get("axis") == "z",
      f"op={op} axis={(wd.get('params_echo') or {}).get('axis')} err={str(w.get('error'))[:80]}")
if op:
    call("revert", op_id=op)

bpy.data.objects.remove(emp, do_unlink=True)

bad = [n for n, ok in RESULTS if not ok]
print(f"\n==== {len(RESULTS) - len(bad)}/{len(RESULTS)} PASS ====")
for n in bad:
    print("FAILED:", n)
if bad:
    sys.exit(1)
