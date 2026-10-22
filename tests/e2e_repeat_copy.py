"""repeat：把一段动作循环平铺到一个窗口，最后一轮铺不满就截断，窗口外走原动画。

用户 2026-10-09 的场景：235–248 手键了一段动作（action `manual`），要"重复到 308 帧，
最后那段动作做不完就截断，走原动画"。

判据（回读 Blender 实际姿态矩阵，不看工具自评）：
  ① [c,d] 内每一帧 == 源段第 ((f−c) mod seg) 帧（用写入前的源姿态做基准）
  ② 最后一轮被截断：[c+5*seg, d] == 源段前几帧，且 d 之后一帧都没写
  ③ 窗口外（d 之后、c 之前）与写入前逐帧相同 = 走原动画
  ④ 轮与轮之间自相似（第 2 轮 == 第 1 轮）
  ⑤ compare_motion(op_id) 验第一轮通过；reapply 保留 repeat；revert 还原
"""
import os
import sys

import addon_utils
import bpy
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import agent_bridge, agent_pose as P  # noqa: E402

RESULTS = []
SEG = [235, 248]
WIN = [235, 308]
BONES = ["shoulder.L", "upper_arm_fk.L", "forearm_fk.L", "hand_fk.L"]
LEN = SEG[1] - SEG[0] + 1


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig or next(
    o for o in scene.objects if o.type == "ARMATURE" and o.name.startswith("RIG-"))
settings.mmr_rig = rig
data_dir = os.path.join(os.environ.get("MCD_TEST_OUTPUT", os.path.dirname(bpy.data.filepath)),
                        "e2e_repeat_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
log = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(log):
    os.remove(log)


def call(tool, **args):
    return agent_bridge._dispatch({"tool": tool, "args": dict(args, agent_id="rp")})


def mats(frames):
    """{frame: {bone: 3x3 骨架空间旋转}}（只读评估结果）。"""
    out = {}
    keep = scene.frame_current
    try:
        for f in frames:
            scene.frame_set(int(f))
            out[int(f)] = {b: np.array(rig.pose.bones[b].matrix.to_3x3(), float) for b in BONES}
    finally:
        scene.frame_set(keep)
    return out


def angle(a, b):
    def rot(m):
        u, _s, vt = np.linalg.svd(m)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vt
        return r
    r = rot(a).T @ rot(b)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    import math
    return math.degrees(math.atan2(np.linalg.norm(v) / 2.0, (np.trace(r) - 1.0) / 2.0))


def worst(fa, fb, obj_a, obj_b):
    return max(angle(obj_a[fa][b], obj_b[fb][b]) for b in BONES)


ALL = list(range(SEG[0] - 6, WIN[1] + 8))
before = mats(ALL)
src = {f: before[f] for f in range(SEG[0], SEG[1] + 1)}

r = call("motion_copy", bones=BONES, src_range=SEG, dst_range=WIN,
         repeat=True, blend=0)
d = r.get("data") or {}
op = d.get("op_id")
check("repeat 写入成功", bool(op), f"op={op} err={str(r.get('error'))[:140]}")
if not op:
    print(f"\n==== {sum(1 for _n, ok in RESULTS if ok)}/{len(RESULTS)} PASS ====")
    sys.exit(1)

notes = " ".join(d.get("metrics", {}).get("notes", []) or [])
check("notes 说清了轮数与截断", ("整轮" in notes) and ("截断" in notes), notes[:140])

after = mats(ALL)

# ① 窗口内逐帧 == 源段取模
worst_in, worst_at = 0.0, None
for f in range(WIN[0], WIN[1] + 1):
    tau = SEG[0] + ((f - SEG[0]) % LEN)
    e = worst(f, tau, after, src)
    if e > worst_in:
        worst_in, worst_at = e, (f, tau)
check("① 窗口内逐帧 == 源段第 ((f−c) mod seg) 帧（< 0.05°）",
      worst_in < 0.05, f"最差 {worst_in:.4f}° @ dst {worst_at[0]} ← src {worst_at[1]}" if worst_at else "")

# ② 最后一轮截断：305–308 == 235–238
tail_worst = max(angle(after[f][b], src[SEG[0] + (f - 305)][b])
                 for f in range(305, WIN[1] + 1) for b in BONES)
check("② 最后一轮铺不满 → 截断到 308（305–308 == 235–238）",
      tail_worst < 0.05, f"最差 {tail_worst:.4f}°")

# ③ 窗口外走原动画
out_worst = max(worst(f, f, after, before) for f in list(range(SEG[0] - 6, SEG[0])) + list(range(WIN[1] + 1, WIN[1] + 8)))
check("③ 窗口外（234 前 / 309 后）逐帧不变 = 走原动画", out_worst < 0.01, f"最差 {out_worst:.5f}°")

# ④ 轮与轮自相似
self_worst = max(angle(after[SEG[0] + k][b], after[SEG[0] + LEN + k][b])
                 for k in range(LEN) for b in BONES)
check("④ 第 2 轮 == 第 1 轮（循环自相似）", self_worst < 0.05, f"最差 {self_worst:.4f}°")

# ⑤ 验收 / reapply / revert
cm = call("compare_motion", op_id=op)
cd = cm.get("data") or {}
check("⑤ compare_motion(op_id) 验第一轮通过", cm.get("ok") and cd.get("err_inner_deg", 9) < 0.05,
      f"err_inner={cd.get('err_inner_deg')} a_range={cd.get('a_range')} "
      f"first_tile_only={(d.get('metrics') or {}).get('verify', {}).get('first_tile_only')}")
check("verify.args 只覆盖第一轮（不是整个平铺窗）",
      (d.get("metrics") or {}).get("verify", {}).get("args", {}).get("a", {}).get("frame_range")
      == [WIN[0], WIN[0] + LEN - 1],
      str((d.get("metrics") or {}).get("verify", {}).get("args", {}).get("a")))

rr = call("reapply", op_id=op, overrides={"blend": 0})
after2 = mats(ALL)
re_worst = max(worst(f, SEG[0] + ((f - SEG[0]) % LEN), after2, src)
               for f in range(WIN[0], WIN[1] + 1))
check("⑤ reapply 保留 repeat 语义", rr.get("ok") and re_worst < 0.05,
      f"ok={rr.get('ok')} 最差 {re_worst:.4f}°")

call("revert", op_id=op)
back = mats(ALL)
restore = max(worst(f, f, back, before) for f in ALL)
check("⑤ revert 完全还原", restore < 0.01, f"最差 {restore:.5f}°")

bad = [n for n, ok in RESULTS if not ok]
print(f"\n==== {len(RESULTS) - len(bad)}/{len(RESULTS)} PASS ====")
for n in bad:
    print("FAILED:", n)
if bad:
    sys.exit(1)
