"""骨盆旋转支点修正（重定向里程碑时静默执行，2026-10-04）。

fixture = 用户的 0999_fsb_showretargetproblem.blend（干净的 ARP 重定向结果，未建项目）的副本：
    bash <套件>/tools/mcd.sh e2e <套件>/tests/e2e_root_pivot.py <套件>/sandbox/work/fixture_root_pivot.blend
  P1  现场：torso_root 带四元数动画、hips 在其下且无动画；修前 151–277 帧胯摆 < 0.10 m
  P2  fix_root_pivot 写了全部帧
  P3  几何：修后 hips 头 = 修前 torso_root 头 + 静止偏移（逐帧，< 0.1 mm）
  P4  torso_root 旋转关键帧逐位不变；P5 脚 IK 目标位置不变
  P6  胯摆恢复：151–277 帧两髋中点横移 ≥ 0.18 m，且与源骨架时序相关 > 0.98
  P7  幂等：第二次调用 skipped=already applied，关键帧不再变
  P8  静默挂钩 operators._silent_root_pivot_fix：返回修正说明，结果与直接调用逐位一致（未建项目时不写报告）
  P9  骨骼不对（源骨架没有 torso_root）→ skipped，不报错
"""
import sys

import addon_utils
import bpy
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor import operators as OPS  # noqa: E402
from bl_ext.user_default.mocap_doctor.core import target as T  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


scene = bpy.context.scene
rig = bpy.data.objects["RIG-arue式重音テトver 2.01_arm"]
src = bpy.data.objects.get("Armature")
act = rig.animation_data.action
F0, F1 = 1, 999
A, B = 151, 277


def w(obj, bone):
    return np.array(obj.matrix_world @ obj.pose.bones[bone].head)


def sample(frames):
    out = {k: [] for k in ("root", "hips", "hl", "hr", "fl", "fr", "shl", "shr")}
    for f in frames:
        scene.frame_set(f)
        out["root"].append(w(rig, "torso_root")); out["hips"].append(w(rig, "hips"))
        out["hl"].append(w(rig, "ORG-thigh.L")); out["hr"].append(w(rig, "ORG-thigh.R"))
        out["fl"].append(w(rig, "foot_ik.L")); out["fr"].append(w(rig, "foot_ik.R"))
        out["shl"].append(w(src, "f_avg_L_Hip")); out["shr"].append(w(src, "f_avg_R_Hip"))
    return {k: np.array(v) for k, v in out.items()}


def sway(S):
    seg = slice(A - F0, B - F0 + 1)
    lat = (S["hr"] - S["hl"])[seg].mean(0); lat[2] = 0; lat /= np.linalg.norm(lat)
    hip = ((S["hl"] + S["hr"]) / 2)[seg] @ lat
    slat = (S["shr"] - S["shl"])[seg].mean(0); slat[2] = 0; slat /= np.linalg.norm(slat)
    shp = ((S["shl"] + S["shr"]) / 2)[seg] @ slat
    pp = float(np.percentile(hip, 97.5) - np.percentile(hip, 2.5))
    return pp, float(np.corrcoef(hip, shp)[0, 1])


def rot_keys():
    return [tuple(k.co) for i in range(4) for k in T.get_fcurve(act, T.bone_path("torso_root", "rotation_quaternion"), i).keyframe_points]


frames = list(range(F0, F1 + 1))
quat_fc = [T.get_fcurve(act, T.bone_path("torso_root", "rotation_quaternion"), i) for i in range(4)]
hips_anim = [fc.data_path for fc in T._fcurves(act) if fc.data_path.startswith('pose.bones["hips"].')]
before = sample(frames)
pp0, c0 = sway(before)
check("P1 现场：torso_root 有四元数动画、hips 无动画、修前胯摆偏小", all(fc is not None for fc in quat_fc) and not hips_anim and pp0 < 0.10,
      f"修前胯摆 {pp0:.3f} m，与源相关 {c0:+.2f}")
rk0 = rot_keys()
orig = act.copy()                                            # 修前副本：P8 用它走静默挂钩
rep = T.fix_root_pivot(rig, act)
check("P2 写了全部帧", rep.get("frames") == 1000 and not rep.get("skipped"), {k: rep.get(k) for k in ("frames", "max_shift_m", "pivot_offset_m", "skipped")})
after = sample(frames)
d = np.array(rig.matrix_world.to_3x3() @ (rig.data.bones["hips"].head_local - rig.data.bones["torso_root"].head_local))
err = np.linalg.norm(after["hips"] - (before["root"] + d), axis=1)
check("P3 修后 hips 头 = 修前 torso_root 头 + 静止偏移", err.max() < 1e-4, f"最大误差 {err.max() * 1000:.4f} mm")
check("P4 torso_root 旋转关键帧逐位不变", rot_keys() == rk0, f"{len(rk0)} 个关键帧")
fe = max(np.abs(after["fl"] - before["fl"]).max(), np.abs(after["fr"] - before["fr"]).max())
check("P5 脚 IK 目标不变", fe < 1e-9, f"最大变化 {fe:.2e} m")
pp1, c1 = sway(after)
check("P6 胯摆恢复（≥0.18 m，与源相关 >0.98）", pp1 >= 0.18 and c1 > 0.98, f"{pp0:.3f} → {pp1:.3f} m，相关 {c0:+.2f} → {c1:+.3f}")
keys1 = [tuple(k.co) for i in range(3) for k in T.get_fcurve(act, T.bone_path("torso_root", "location"), i).keyframe_points]
rep2 = T.fix_root_pivot(rig, act)
keys2 = [tuple(k.co) for i in range(3) for k in T.get_fcurve(act, T.bone_path("torso_root", "location"), i).keyframe_points]
check("P7 幂等", rep2.get("skipped") == "already applied" and keys1 == keys2, rep2.get("skipped"))
settings = scene.mocap_doctor
settings.mmr_rig = rig
rig.animation_data.action = orig                             # 换回修前副本，走重定向里程碑的静默挂钩
note = OPS._silent_root_pivot_fix(bpy.context, settings)
keys_hook = [tuple(k.co) for i in range(3) for k in T.get_fcurve(orig, T.bone_path("torso_root", "location"), i).keyframe_points]
check("P8 静默挂钩：返回修正说明，结果与直接调用逐位一致", "已修正骨盆旋转支点" in note and keys_hook == keys1
      and bool(orig.get(T.ROOT_PIVOT_MARK)), note)
rep3 = T.fix_root_pivot(src, src.animation_data.action)
check("P9 骨骼不对 → skipped", rep3.get("skipped") == "bones missing", rep3.get("skipped"))

fails = [x for x in RESULTS if not x[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for name, _ok, detail in fails:
    print("FAILED:", name, "::", detail)
if fails:
    sys.exit(1)
