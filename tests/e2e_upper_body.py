"""上半身位置对齐源（重定向里程碑时静默执行，紧接骨盆支点修正之后，2026-10-04）。

fixture = 用户的 0999_fsb_showretargetproblem.blend（干净的 ARP 重定向结果，未建项目）的副本：
    bash <套件>/tools/mcd.sh e2e <套件>/tests/e2e_upper_body.py <套件>/sandbox/work/fixture_root_pivot.blend
先按里程碑的顺序跑 fix_root_pivot，再测 fix_upper_body_follow：
  U1  现场：spine_fk.001 有四元数关键帧、源是 SMPL 骨架；支点修后 151–277 帧肩仍跟着胯晃（肩 > 0.12 m，髋相对肩 < 0.08 m）
  U2  写了全部帧，没有限幅/够不着的帧
  U3  151–277：肩中横移在源 ±25%（0.058–0.096 m）、髋相对肩 ≥ 0.10 m、头横移比修前更接近源
  U4  151–277：髋中横移不变（±0.01 m）且与源相关 ≥ 0.99（支点修正的效果还在：≥ 0.18 m、> 0.98）
  U5  全片：肩中/颈 相对髋中（按躯干长度比缩放）与源的距离，中位数和 P95 都比修前小
  U6  脚和髋不动：foot_ik / DEF-foot / ORG-thigh 头最大变化 < 1 mm
  U7  只改了 spine_fk.001 的四个旋转曲线：其余曲线（含 torso_root 的支点修正平移）逐位不变
  U8  刚体摆正：spine_fk.001 头不动，肩中到它的距离逐帧不变（< 0.1 mm）
  U9  脊柱骨局部角速度 P95 不比修前差 10% 以上、局部转角不超过 60°
  U10 幂等：第二次调用 skipped=already applied，关键帧不再变
  U11 静默挂钩：在修前副本上依次走 _silent_root_pivot_fix + _silent_upper_body_fix，返回说明、结果与直接调用逐位一致
  U12 跳过：动作不是活动动作 / 源不是 SMPL / 源没有动画 / 目标骨骼缺失 → skipped，不报错
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
SEG = slice(A - F0, B - F0 + 1)
FPS = float(scene.render.fps)
SPINE = ("spine_fk", "spine_fk.001", "spine_fk.003")


def w(obj, bone):
    return np.array(obj.matrix_world @ obj.pose.bones[bone].head)


def sample():
    keys = {"hl": (rig, "ORG-thigh.L"), "hr": (rig, "ORG-thigh.R"), "sl": (rig, "ORG-upper_arm.L"), "sr": (rig, "ORG-upper_arm.R"),
            "neck": (rig, "ORG-spine.004"), "head": (rig, "ORG-spine.006"), "piv": (rig, "spine_fk.001"),
            "fl": (rig, "foot_ik.L"), "fr": (rig, "foot_ik.R"), "dl": (rig, "DEF-foot.L"), "dr": (rig, "DEF-foot.R"),
            "s_hl": (src, "f_avg_L_Hip"), "s_hr": (src, "f_avg_R_Hip"), "s_sl": (src, "f_avg_L_Shoulder"),
            "s_sr": (src, "f_avg_R_Shoulder"), "s_neck": (src, "f_avg_Neck"), "s_head": (src, "f_avg_Head")}
    out = {k: [] for k in keys}
    for f in range(F0, F1 + 1):
        scene.frame_set(f)
        for k, (obj, bone) in keys.items():
            out[k].append(w(obj, bone))
    S = {k: np.array(v) for k, v in out.items()}
    for p in ("", "s_"):
        S[p + "hip"] = (S[p + "hl"] + S[p + "hr"]) / 2
        S[p + "sho"] = (S[p + "sl"] + S[p + "sr"]) / 2
    return S


def pp(x):
    x = x[SEG]
    return float(np.percentile(x, 97.5) - np.percentile(x, 2.5))


def lat_axis(left, right):
    lat = (right - left)[SEG].mean(0)
    lat[2] = 0
    return lat / np.linalg.norm(lat)


def sway(S):
    lat, slat = lat_axis(S["hl"], S["hr"]), lat_axis(S["s_hl"], S["s_hr"])
    hip, sho = S["hip"] @ lat, S["sho"] @ lat
    return {"hip": pp(hip), "sho": pp(sho), "rel": pp(hip - sho), "head": pp(S["head"] @ lat),
            "corr": float(np.corrcoef(hip[SEG], (S["s_hip"] @ slat)[SEG])[0, 1])}


def scaled_err(S, key, k):
    e = np.linalg.norm((S[key] - S["hip"]) - (S["s_" + key] - S["s_hip"]) * k, axis=1)
    return float(np.median(e)), float(np.percentile(e, 95))


def all_keys(action, skip=()):
    return {(fc.data_path, fc.array_index): [tuple(k.co) for k in fc.keyframe_points]
            for fc in T._fcurves(action) if (fc.data_path, fc.array_index) not in skip}


def quats(action, bone):
    fcs = [T.get_fcurve(action, T.bone_path(bone, "rotation_quaternion"), i) for i in range(4)]
    q = np.array([[k.co.y for k in fc.keyframe_points] for fc in fcs]).T
    return q / np.linalg.norm(q, axis=1)[:, None]


def spine_stats(action):
    out = {}
    for bone in SPINE:
        q = quats(action, bone)
        angle = np.degrees(2 * np.arccos(np.clip(np.abs(q[:, 0]), 0, 1)))
        step = np.degrees(2 * np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", q[1:], q[:-1])), 0, 1))) * FPS
        out[bone] = (float(angle.max()), float(np.percentile(step, 95)))
    return out


SWING_QUAT = {(T.bone_path("spine_fk.001", "rotation_quaternion"), i) for i in range(4)}
orig = act.copy()                                            # 修前（支点也没修）副本：U11 走里程碑挂钩
quat_fc = [T.get_fcurve(act, T.bone_path("spine_fk.001", "rotation_quaternion"), i) for i in range(4)]
pivot_rep = T.fix_root_pivot(rig, act)                       # 里程碑顺序：先支点
before = sample()
sw0 = sway(before)
k_torso = float(np.linalg.norm(before["sho"] - before["hip"], axis=1).mean()
                / np.linalg.norm(before["s_sho"] - before["s_hip"], axis=1).mean())
check("U1 现场：spine_fk.001 四元数关键帧、SMPL 源；支点修后肩仍跟着胯晃",
      all(fc is not None for fc in quat_fc) and src is not None and not pivot_rep.get("skipped") and sw0["sho"] > 0.12 and sw0["rel"] < 0.08,
      f"支点修后 151–277：髋 {sw0['hip']:.3f} 肩 {sw0['sho']:.3f} 髋相对肩 {sw0['rel']:.3f} 头 {sw0['head']:.3f} m")
keys0 = all_keys(act)
spine0 = spine_stats(act)
rep = T.fix_upper_body_follow(rig, src, act)
check("U2 写了全部帧、没有限幅", rep.get("frames") == 1000 and not rep.get("skipped") and rep.get("source") == src.name
      and rep.get("clamped_frames") == 0 and rep.get("unreachable_frames") == 0,
      {k: rep.get(k) for k in ("frames", "skipped", "source", "torso_ratio", "swing_deg", "clamped_frames", "shoulder_lateral_err_cm")})
after = sample()
sw1 = sway(after)
check("U3 151–277：肩横移在源 ±25%、髋相对肩 ≥ 0.10、头更接近源",
      0.058 <= sw1["sho"] <= 0.096 and sw1["rel"] >= 0.10 and abs(sw1["head"] - 0.091) < abs(sw0["head"] - 0.091),
      f"肩 {sw0['sho']:.3f}→{sw1['sho']:.3f}（源 0.077）髋相对肩 {sw0['rel']:.3f}→{sw1['rel']:.3f}（源 0.131）头 {sw0['head']:.3f}→{sw1['head']:.3f}（源 0.091）")
check("U4 151–277：髋中横移不变、与源相关 ≥ 0.99（支点修正效果还在）",
      abs(sw1["hip"] - sw0["hip"]) <= 0.01 and sw1["corr"] >= 0.99 and sw1["hip"] >= 0.18,
      f"髋 {sw0['hip']:.3f}→{sw1['hip']:.3f} m，相关 {sw0['corr']:+.4f}→{sw1['corr']:+.4f}")
errs = {key: (scaled_err(before, key, k_torso), scaled_err(after, key, k_torso)) for key in ("sho", "neck")}
check("U5 全片：肩中/颈相对髋中（缩放）与源的距离，中位数和 P95 都变小",
      all(e1[0] < e0[0] and e1[1] < e0[1] for e0, e1 in errs.values()),
      "; ".join(f"{key} P50 {e0[0] * 100:.2f}→{e1[0] * 100:.2f} P95 {e0[1] * 100:.2f}→{e1[1] * 100:.2f} cm" for key, (e0, e1) in errs.items())
      + f"（躯干长度比 {k_torso:.3f}）")
moved = {key: float(np.abs(after[key] - before[key]).max()) for key in ("fl", "fr", "dl", "dr", "hl", "hr")}
check("U6 脚和髋不动（< 1 mm）", max(moved.values()) < 1e-3, {k: f"{v * 1000:.4f} mm" for k, v in moved.items()})
keys1 = all_keys(act)
others_same = all(keys1[key] == value for key, value in keys0.items() if key not in SWING_QUAT)
swing_changed = all(keys1[key] != keys0[key] for key in SWING_QUAT)
check("U7 只改 spine_fk.001 的四个旋转曲线（torso_root 等其余曲线逐位不变）", others_same and swing_changed and len(keys1) == len(keys0),
      f"{len(keys0) - 4} 条其余曲线不变，spine_fk.001 旋转 4 条已改")
piv_move = float(np.abs(after["piv"] - before["piv"]).max())
radius = np.abs(np.linalg.norm(after["sho"] - after["piv"], axis=1) - np.linalg.norm(before["sho"] - before["piv"], axis=1)).max()
check("U8 刚体摆正：枢轴不动、肩中到枢轴距离逐帧不变", piv_move < 1e-6 and radius < 1e-4,
      f"枢轴 {piv_move * 1000:.4f} mm，距离变化 {radius * 1000:.4f} mm")
spine1 = spine_stats(act)
check("U9 脊柱骨局部角速度 P95 不差 10% 以上、转角 ≤ 60°",
      all(spine1[b][1] <= spine0[b][1] * 1.10 and spine1[b][0] <= 60.0 for b in SPINE),
      "; ".join(f"{b} 最大 {spine0[b][0]:.1f}→{spine1[b][0]:.1f}° 角速度P95 {spine0[b][1]:.1f}→{spine1[b][1]:.1f}°/s" for b in SPINE))
rep2 = T.fix_upper_body_follow(rig, src, act)
check("U10 幂等", rep2.get("skipped") == "already applied" and all_keys(act) == keys1, rep2.get("skipped"))
settings = scene.mocap_doctor
settings.mmr_rig = rig
settings.source_armature = src
rig.animation_data.action = orig                             # 换回修前副本，按里程碑的顺序走两个静默挂钩
note = OPS._silent_root_pivot_fix(bpy.context, settings) + OPS._silent_upper_body_fix(bpy.context, settings)
check("U11 静默挂钩：返回说明、结果与直接调用逐位一致", "已修正骨盆旋转支点" in note and "上半身位置已对齐源" in note
      and all_keys(orig) == keys1 and bool(orig.get(T.UPPER_BODY_MARK)) and bool(orig.get(T.ROOT_PIVOT_MARK)), note)
rig.animation_data.action = act
spare = act.copy()
for key in (T.UPPER_BODY_MARK,):
    if key in spare:
        del spare[key]
r_src = T.fix_upper_body_follow(rig, rig, spare)             # spare 不是活动动作，rig 也不是 SMPL 源
rig.animation_data.action = spare
r_src2 = T.fix_upper_body_follow(rig, rig, spare)            # 活动动作，但源不是 SMPL
still = src.copy()                                           # 同一骨架数据、没有动画的源（不进场景）
still.animation_data_clear()
r_still = T.fix_upper_body_follow(rig, still, spare)
r_bones = T.fix_upper_body_follow(src, src, src.animation_data.action)
rig.animation_data.action = act
check("U12 跳过：动作不活动 / 源不是 SMPL / 源没有动画 / 骨骼缺失", "not the armature's active action" in str(r_src.get("skipped"))
      and "not an SMPL" in str(r_src2.get("skipped")) and "no animation" in str(r_still.get("skipped"))
      and str(r_bones.get("skipped", "")).startswith("bones missing") and not spare.get(T.UPPER_BODY_MARK),
      [r_src.get("skipped"), r_src2.get("skipped"), r_still.get("skipped"), r_bones.get("skipped")])

fails = [x for x in RESULTS if not x[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for name, _ok, detail in fails:
    print("FAILED:", name, "::", detail)
if fails:
    sys.exit(1)
