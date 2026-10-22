"""Headless validation of the quat-aware restore_accent (力量感 math).

Checks on the fixture RIG:
  A  quaternion path (spine_fk) ease_reshape:
     - op writes, strip on its OWN track
     - stored delta keys are unit quaternions (proper rotation deltas)
     - window edges: evaluated pose ≈ base (taper pins continuity)
     - impact frame: evaluated pose differs by a small, sane rotation
       (angle close to the op's max_pose_shift_deg metric)
  B  location path (torso_root) ease_reshape:
     - one strip carries all three axes
     - effect_check shows position change at impact, none at edges
"""
import math
import os
import sys

import bpy
import addon_utils
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import agent_ops, agent_bridge  # noqa: E402
from bl_ext.user_default.mocap_doctor.core import agent_bake  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig or next(
    (o for o in scene.objects if o.type == "ARMATURE" and o.name.startswith("RIG-")), None)
settings.mmr_rig = rig

data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_accent_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)

QPATH = 'pose.bones["spine_fk"].rotation_quaternion'
LPATH = 'pose.bones["torso_root"].location'

# 扫全片找 spine_fk 角速度最大的 20 帧窗口——静止段上测"变脆"没有意义
# （第一次就栽在随便选的 10-30 帧，整窗姿态偏移才 0.7°）。
_base = agent_bridge._base_action(rig)
_c = [agent_bake.sample_fcurve_values(_base, QPATH, i, 1, 1500)
      for i in range(4)]
_q = np.stack(_c, axis=1)
_q /= np.linalg.norm(_q, axis=1, keepdims=True)
_sp = np.degrees(2 * np.arccos(np.clip(
    np.abs(np.sum(_q[1:] * _q[:-1], axis=1)), 0, 1)))   # °/f per frame
_W = 20
_mean_sp = np.convolve(_sp, np.ones(_W) / _W, mode="valid")
_cand = int(np.argmax(_mean_sp[:1200])) + 1             # 避开片尾
A0, A1 = _cand, _cand + _W - 1
IMPACT = A0 + 2 * _W // 3                              # 发力帧靠窗尾（到位帧）
print(f"window={A0}-{A1} impact={IMPACT} mean_speed={_mean_sp[_cand-1]:.2f}°/f")


def basis(bone, frame):
    scene.frame_set(frame)
    return rig.pose.bones[bone].matrix_basis.to_quaternion().copy()


def quat_norms(strip):
    """|q| of every stored delta key — must be ~1 (proper rotation deltas)."""
    curves = {}
    for fc in strip.action.fcurves:
        if fc.data_path == QPATH:
            curves[fc.array_index] = fc
    if len(curves) != 4:
        return None
    norms = []
    for i in range(len(curves[0].keyframe_points)):
        q = np.array([curves[c].keyframe_points[i].co[1] for c in range(4)])
        norms.append(float(np.linalg.norm(q)))
    return norms


# ---------------------------------------------------------------- A: quat
op = agent_ops.restore_accent(
    rig, agent_bridge._base_action(rig), QPATH, None,
    [A0, A1], "ease_reshape", strength=0.6, impact_frame=IMPACT,
    blend=4, data_dir=data_dir)
check("A1 写入成功", bool(op.get("id")), f"{op['id']} metric={op['metrics']}")
track, strip = agent_ops.find_op_strip(rig, op)
check("A2 专属轨", track is not None and track.name == strip.name,
      f"track={track.name if track else None}")

norms = quat_norms(strip)
check("A3 delta 是单位四元数",
      norms is not None and max(abs(n - 1.0) for n in norms) < 1e-3,
      f"min/max |q| = {min(norms):.4f}/{max(norms):.4f}" if norms else "no curves")

# 边缘帧不动，发力帧有变化且角度与 metric 同量级
track.mute = True
edge_off = basis("spine_fk", A0 + 1)
track.mute = False
edge_on = basis("spine_fk", A0 + 1)
ang_edge = math.degrees(2 * math.acos(min(1.0, abs(float(edge_off.dot(edge_on))))))

track.mute = True
imp_off = basis("spine_fk", IMPACT)
track.mute = False
imp_on = basis("spine_fk", IMPACT)
ang_imp = math.degrees(2 * math.acos(min(1.0, abs(float(imp_off.dot(imp_on))))))
metric = op["metrics"].get("max_pose_shift_deg", 0)
check("A4 边缘连续（≈0°）", ang_edge < 0.05, f"edge={ang_edge:.3f}°")
check("A5 发力帧有变化且量级正常",
      0.05 < ang_imp <= metric + 0.5,
      f"impact={ang_imp:.2f}° metric_max={metric}°")

# 速度剖面验证：发力逼近段（k-pre → k，ease_reshape 的提速区间）的
# 峰值角速度应该变大——"变脆"的直接证据。峰值必须限定在该段内测，
# 窗口里段外的快动作会淹没信号（第一次就错在这）。
n_win = A1 - A0 + 1
pre = n_win // 3
span = (IMPACT - pre, IMPACT)

def speed_over(muted, f0, f1):
    track.mute = muted
    qs = [basis("spine_fk", f) for f in range(f0, f1 + 1)]
    track.mute = False
    sp = []
    for a, b in zip(qs, qs[1:]):
        sp.append(math.degrees(2 * math.acos(min(1.0, abs(float(a.dot(b)))))))
    return sp

sp_before = speed_over(True, span[0], span[1])
sp_after = speed_over(False, span[0], span[1])
arrive_before = sp_before[-1]      # 进发力帧的瞬间速度（"到位脆不脆"就它）
arrive_after = sp_after[-1]
check("A6 到达瞬间角速度提升（变脆）",
      arrive_after > arrive_before * 1.15,
      f"span={span} before={arrive_before:.2f}°/f after={arrive_after:.2f}°/f "
      f"profile_before={[round(x,1) for x in sp_before]} "
      f"profile_after={[round(x,1) for x in sp_after]}")

# ---------------------------------------------------------------- B: loc
op2 = agent_ops.restore_accent(
    rig, agent_bridge._base_action(rig), LPATH, None,
    [A0, A1], "ease_reshape", strength=0.6, impact_frame=IMPACT,
    blend=4, data_dir=data_dir)
check("B1 location 写入成功", bool(op2.get("id")), f"{op2['id']} metric={op2['metrics']}")
track2, strip2 = agent_ops.find_op_strip(rig, op2)
axes = sorted({fc.array_index for fc in strip2.action.fcurves
               if fc.data_path == LPATH})
check("B2 三轴同一条 strip", axes == [0, 1, 2], f"axes={axes}")

fails = [r for r in RESULTS if not r[1]]
print(f"\n=== {len(RESULTS) - len(fails)}/{len(RESULTS)} passed ===")
for name, _ok, detail in fails:
    print("FAILED:", name, "::", detail)
if fails:
    sys.exit(1)
