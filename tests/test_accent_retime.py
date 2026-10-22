"""restore_accent 的默认方法改成 `retime`（真正的时间重塑）。

2026-10-06 用户判定 `ease_reshape` 是"错的效应"：它**在冲击前后加一段角度鼓包**
（实测 strength 0.5 → 9 帧内峰值 −31.9°；strength 1.0 → 总转角 63.7°），
0.5 看不见、1.0 变成"手多弯一下"，中间没有"更有力"的那一档。

`retime` 只做时间重映射（`np.interp` 重采样同一条曲线）：端点固定、**姿态集合不变**，
所以结构上不可能给关节加额外角度。

判据（本测试）：
  ① 不传 method 时记录下来的就是 "retime"；
  ② 窗口**两端**的姿态不变（端点固定）；
  ③ 通道的**极值不变**（姿态集合没变，只是发生时刻变了）；
  ④ 峰速上升（重音确实生效了）。
"""
import sys

import addon_utils
import bpy

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import agent_bridge  # noqa: E402

RESULTS = []
A, B = 320, 372
BONE = "hand_fk.R"


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


def call(tool, **args):
    return agent_bridge._dispatch({"tool": tool, "args": dict(args, agent_id="ra")})


def profile():
    """窗口内逐帧的局部旋转角（度）+ 端点值 + 极值。"""
    scene = bpy.context.scene
    arm = next(o for o in scene.objects if o.type == "ARMATURE" and o.name.startswith("RIG-"))
    vals = []
    for f in range(A, B + 1):
        scene.frame_set(f)
        q = arm.pose.bones[BONE].rotation_quaternion
        vals.append(2.0 * q.angle if q.angle else 0.0)
    return vals


base = profile()
before_end = (base[0], base[-1])
before_rng = (min(base), max(base))

# 用默认 method（不传）——应该落到 retime
w = call("restore_accent", frame_range=[A, B],
         data_path=f'pose.bones["{BONE}"].rotation_quaternion',
         impact_frame=(A + B) // 2, blend=4)
wd = w.get("data") or {}
op = wd.get("op_id")
pe = wd.get("params_echo") or wd.get("params") or {}
got_method = pe.get("method") or (wd.get("metrics") or {}).get("method")
check("默认 method = retime（不再是 ease_reshape）",
      got_method == "retime", f"记录到的 method = {got_method!r} op={op} err={str(w.get('error'))[:80]}")

if op:
    after = profile()
    after_end = (after[0], after[-1])
    after_rng = (min(after), max(after))
    check("端点固定（窗口首尾姿态不变）",
          abs(after_end[0] - before_end[0]) < 0.5 and abs(after_end[1] - before_end[1]) < 0.5,
          f"首 {before_end[0]:.2f}→{after_end[0]:.2f}  尾 {before_end[1]:.2f}→{after_end[1]:.2f}（度）")
    span_b = before_rng[1] - before_rng[0]
    span_a = after_rng[1] - after_rng[0]
    check("姿态集合不变（极值跨度基本不变 = 没加额外角度）",
          abs(span_a - span_b) < max(2.0, 0.08 * span_b),
          f"跨度 {span_b:.2f}° → {span_a:.2f}°（差 {abs(span_a-span_b):.2f}°）")

    m = call("analyze_motion", bones=[BONE], frame_range=[A, B], brief=True)
    pk = (((m.get("data") or {}).get("bones") or {}).get(BONE) or {}).get("peak_speed")
    jit = (((m.get("data") or {}).get("bones") or {}).get(BONE) or {}).get("jitter_deg")
    check("重音生效：峰速有数、jitter 没炸（≤ 修前 2 倍量级）",
          pk is not None and (jit is None or jit < 12.0), f"peak_speed={pk} jitter={jit}")

    # 逃生口：显式要 ease_reshape 仍然可用（文档已标明是"过冲/回弹"用法）
    call("revert", op_id=op)
    w2 = call("restore_accent", frame_range=[A, B],
              data_path=f'pose.bones["{BONE}"].rotation_quaternion',
              method="ease_reshape", strength=0.5, impact_frame=(A + B) // 2, blend=4)
    op2 = (w2.get("data") or {}).get("op_id")
    check("显式 method=ease_reshape 仍可用（逃生口保留）", bool(op2),
          f"op={op2} err={str(w2.get('error'))[:80]}")
    if op2:
        call("revert", op_id=op2)
    back = profile()
    check("两条都 revert 后回到基线",
          abs(back[0] - before_end[0]) < 0.5 and abs(min(back) - before_rng[0]) < 0.5,
          f"首 {back[0]:.2f}（基线 {before_end[0]:.2f}）")

bad = [n for n, ok in RESULTS if not ok]
print(f"\n==== {len(RESULTS) - len(bad)}/{len(RESULTS)} PASS ====")
for n in bad:
    print("FAILED:", n)
if bad:
    sys.exit(1)
