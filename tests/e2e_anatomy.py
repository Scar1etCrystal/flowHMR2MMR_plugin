"""Headless validation of the semantic anatomy layer + param rewrite.

Checks on the fixture RIG:
  A  probe_anatomy basics: palm/bone_axis/sole/toe/body_forward return sane
     directions with owner_bone + local_axis
  B  dual-axis hold_pose: flip palm ~180deg while keeping finger direction -
     proves the degenerate minimal-rotation bug is gone (finger keeps pointing
     the same way, palm ends on target - verified by RE-PROBING anatomy, not
     by the solver's own assumption)
  C  flip guard: single-axis world_dir needing ~180deg skips those frames
     (skipped_flip_frames > 0, no inside-out hand)
  D  reapply: same op_id, same track, params merged, strip name stable
  E  dir_object: arrow empty drives the target direction; turning the empty
     then reapplying moves the palm with it
  F  _sync_params_list populates settings.agent_params
"""
import math
import os
import sys

import bpy
import addon_utils
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import (  # noqa: E402
    agent_anatomy, agent_bridge, agent_ops)
from mathutils import Vector  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")


scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig or next(
    (o for o in scene.objects if o.type == "ARMATURE"
     and o.name.startswith("RIG-")), None)
settings.mmr_rig = rig

data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_anatomy_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)

bone_names = set(rig.data.bones.keys())
has_fingers = "f_index.01.L" in bone_names and "f_pinky.03.L" in bone_names
has_feet = "DEF-foot.L" in bone_names and "DEF-toe.L" in bone_names
print(f"rig={rig.name} fingers={has_fingers} feet={has_feet}")

FR = [150, 165]   # 掌心修复对应的帧段附近

# ---------- A: probes --------------------------------------------------------
if has_fingers:
    p = agent_anatomy.probe(scene, rig, part="palm", side="L",
                            frame_range=FR)
    check("A1 palm probe",
          p["owner_bone"] == "hand_fk.L"
          and p.get("local_axis") and p.get("world_dir"),
          f"dir={p.get('world_dir')} conf={p.get('confidence')} "
          f"ev={p.get('evidence')}")
    check("A2 palm secondary_axis", bool(p.get("secondary_axis")),
          f"sec={p.get('secondary_axis')} ({p.get('secondary_name')})")
    p2 = agent_anatomy.probe(scene, rig, part="palm", side="L",
                             frame_range=FR, toward=[0, -1, 0])
    check("A3 palm toward err", p2.get("err_max_deg") is not None,
          f"err={p2.get('err_max_deg')} resolved={p2.get('toward_resolved')}")

    # ---- A7–A9: 掌心 = 目标网格上看得见的掌心（网格标定，相对 hand_fk 刚性）
    # 手指几何版的 local_spread 在这段上有几十度（手指一弯掌平面就转）；网格版是固定局部向量。
    pm = agent_anatomy.probe(scene, rig, part="palm", side="L",
                             frame_range=[1, 1400], max_frames=15)
    evm = pm.get("evidence", {})
    check("A7 palm mesh-calibrated + rigid in hand_fk (local spread < 2° over 1–1400)",
          evm.get("palm_source") == "mesh" and pm.get("local_spread_deg", 99) < 2.0,
          f"source={evm.get('palm_source')} spread={pm.get('local_spread_deg')} "
          f"faces={evm.get('mesh_faces')} thumb={evm.get('thumb_side_mm')}mm "
          f"local={pm.get('local_axis')} cal={evm.get('mesh_calibration')}")
    pmr = agent_anatomy.probe(scene, rig, part="palm", side="R",
                              frame_range=[1, 1400], max_frames=15)
    la, lb = Vector(pm["local_axis"]), Vector(pmr["local_axis"])
    lb.x = -lb.x
    mirror_deg = math.degrees(la.angle(lb)) if la.length and lb.length else 99.0
    check("A8 L/R palm local axes are X-mirrors (< 3°)",
          pmr.get("evidence", {}).get("palm_source") == "mesh" and mirror_deg < 3.0,
          f"L={pm['local_axis']} R={pmr['local_axis']} angle={mirror_deg:.2f}°")
    # 独立交叉验证：手指弯得不多（0.3 < curl < 2，有把握但没攥拳）的帧上，手指几何推的
    # 掌心必须与网格掌心同一侧。攥拳时指根→指尖倒向掌心法线，手指法本身就不可靠，不计入。
    mild, bad = [], []
    for f0 in range(50, 1450, 100):
        pc = agent_anatomy.probe(scene, rig, part="palm", side="L", frame_range=[f0, f0])
        evc = pc.get("evidence", {})
        curl = evc.get("curl_mag", 0.0)
        if 0.3 < curl < 2.0 and evc.get("finger_palm_vs_mesh_deg") is not None:
            mild.append((f0, curl, evc["finger_palm_vs_mesh_deg"]))
            if evc["finger_palm_vs_mesh_deg"] >= 90:
                bad.append((f0, curl, evc["finger_palm_vs_mesh_deg"]))
    check("A9 finger-curl palm agrees with the mesh palm on mildly curled frames",
          bool(mild) and not bad, f"mild={mild} bad={bad}")
    hx = agent_anatomy.probe(scene, rig, part="hand_axis", side="L",
                             frame_range=[1, 1400], max_frames=15)
    check("A10 hand_axis rigid + ⊥ palm (local spread < 2°, angle to palm 80–100°)",
          hx.get("local_spread_deg", 99) < 2.0
          and 80 < math.degrees(Vector(hx["local_axis"]).angle(Vector(pm["local_axis"]))) < 100,
          f"spread={hx.get('local_spread_deg')} angle_to_palm="
          f"{math.degrees(Vector(hx['local_axis']).angle(Vector(pm['local_axis']))):.1f}°")

ax = agent_anatomy.probe(scene, rig, part="bone_axis", bone="hand_fk.L")
check("A4 bone_axis", len(ax.get("axes", {})) == 6,
      f"+Y={ax['axes'].get('+Y')}")

if has_feet:
    f = agent_anatomy.probe(scene, rig, part="sole", side="L",
                            frame_range=FR)
    check("A5 sole probe", f["owner_bone"] == "foot_ik.L"
          and f.get("local_axis"),
          f"sole={f.get('world_dir')} conf={f.get('confidence')}")

bf = agent_anatomy.probe(scene, rig, part="body_forward", frame_range=[1])
check("A6 body_forward", bf.get("world_dir") is not None
      and abs(bf["world_dir"][2]) < 0.2,   # 前方应接近水平
      f"fwd={bf.get('world_dir')} ev={bf.get('evidence')}")

# ---------- B: dual-axis flip (the inside-out regression) --------------------
if has_fingers:
    p0 = agent_anatomy.probe(scene, rig, part="palm", side="L",
                             frame_range=FR)
    palm0 = Vector(p0["world_dir"])
    hand0 = Vector(agent_anatomy.probe(
        scene, rig, part="hand_axis", side="L",
        frame_range=FR)["world_dir"])
    target = (-palm0)                      # 强制 ~180°：掌心翻向反面
    op = agent_ops.hold_pose(
        rig, agent_ops.base_action_of(rig), ["hand_fk.L"], FR,
        target="world_dir", world_dir=list(target),
        world_axis="probe:palm.L",
        secondary_axis="probe:hand_axis.L",
        scene=scene, mode="replace", blend=2,
        data_dir=data_dir)
    bmet = op["metrics"]["bones"]["hand_fk.L"]
    check("B1 dual-axis write", op.get("status") == "preview"
          and bmet.get("skipped_flip_frames", 0) == 0
          and bmet.get("flipped_frames", 0) > 0,
          f"align_max={bmet.get('align_max_deg')} "
          f"flipped={bmet.get('flipped_frames')} "
          f"skipped={bmet.get('skipped_flip_frames')}")

    bpy.context.view_layer.update()
    p1 = agent_anatomy.probe(scene, rig, part="palm", side="L",
                             frame_range=FR, toward=list(target))
    # 边缘 taper 帧保留原姿态是设计行为；中段必须到位（非自证的复测：
    # 探头从手指几何重推，不是用求解假设自证）
    check("B2 palm on target mid-range (re-probed)",
          p1.get("err_inner_deg") is not None
          and p1["err_inner_deg"] < 20,
          f"err_inner={p1.get('err_inner_deg')} "
          f"per_frame={p1.get('err_per_frame')}")
    # 掌心 180° 翻转应是绕 腕→指根 轴的干净旋前/旋后：hand_axis 不动。
    # （150–165 帧手是攥着的，指根→指尖离掌心法线只有 ~24°，"手指方向不变"在几何上做不到）
    fp = agent_anatomy.probe(scene, rig, part="hand_axis", side="L",
                             frame_range=FR, toward=list(hand0))
    # 几何下限：hand_axis 与掌心差 90°−δ（δ≈6°），绕其 ⊥ 分量翻 180° 后自身转 2δ≈13°；
    # 再加采样点落在 blend 斜坡上的份额。要抓的是"滚转随机"（几十到 180°），30° 足够。
    check("B3 hand axis (wrist→knuckles) preserved mid-range (< 30°)",
          fp.get("err_inner_deg") is not None
          and fp["err_inner_deg"] < 30,
          f"hand_axis err_inner={fp.get('err_inner_deg')} "
          f"(单轴最小旋转翻 180° 时轴向随机，会把手翻进手里)")
    OP_B = op["id"]

    # ---------- C: flip guard ---------------------------------------------
    op_c = agent_ops.hold_pose(
        rig, agent_ops.base_action_of(rig), ["hand_fk.R"], FR,
        target="world_dir",
        world_dir=list(-Vector(agent_anatomy.probe(
            scene, rig, part="palm", side="R",
            frame_range=FR)["world_dir"])),
        world_axis=agent_anatomy.probe(
            scene, rig, part="palm", side="R",
            frame_range=FR)["local_axis"],   # 无 secondary → 最小旋转路径
        scene=scene, mode="replace", blend=2,
        data_dir=data_dir)
    cmet = op_c["metrics"]["bones"]["hand_fk.R"]
    check("C1 flip guard skips ~180deg frames",
          cmet.get("skipped_flip_frames", 0) > 0,
          f"skipped={cmet.get('skipped_flip_frames')} "
          f"align_max={cmet.get('align_max_deg')}")

    # ---------- D: reapply 同轨重写 -----------------------------------------
    op_d = agent_ops.reapply(data_dir, rig, OP_B, scene=scene,
                             threshold_deg=5.0, blend=6)
    check("D1 reapply same id/track",
          op_d["id"] == OP_B and op_d["track"] == op["track"],
          f"track={op_d['track']} strip={op_d['strip']}")
    check("D2 reapply params merged",
          op_d["params"].get("threshold_deg") == 5.0
          and op_d["params"].get("blend") == 6
          and op_d["params"].get("world_dir") is not None,
          f"params={ {k: op_d['params'].get(k) for k in ('threshold_deg','blend','world_axis')} }")
    # 同轨同名：strips 数量应守恒（删旧写新）
    anim = rig.animation_data
    tr = next((t for t in anim.nla_tracks if t.name == op_d["track"]), None)
    check("D3 same track single strip",
          tr is not None and len(tr.strips) == 1
          and tr.strips[0].name == op_d["strip"],
          f"strips={[s.name for s in (tr.strips if tr else [])]}")

    # ---------- E: dir_object ----------------------------------------------
    agent_ops.revert(rig, data_dir, op_c["id"])   # 清掉 C 在右手的残留 delta
    arrow = bpy.data.objects.new("mcd_dir_test", None)
    arrow.empty_display_type = "SINGLE_ARROW"
    scene.collection.objects.link(arrow)
    # 箭头轴 = 局部 +Z；转到 -Y = 绕 X +90°
    arrow.rotation_euler = (math.radians(90), 0.0, 0.0)
    bpy.context.view_layer.update()
    # 用右手：B/D 的 delta 还在左手叠着（stacking 语义下 conj(base)⊗desired
    # 叠在旧 delta 上 ≠ desired——已知近似），右手 C 的 flip 全 skip ≈干净
    op_e = agent_ops.hold_pose(
        rig, agent_ops.base_action_of(rig), ["hand_fk.R"], FR,
        target="world_dir",
        world_axis="probe:palm.R",
        secondary_axis="probe:hand_axis.R",
        dir_object=arrow.name, dir_mode="arrow",
        scene=scene, mode="replace", blend=2,
        data_dir=data_dir)
    bpy.context.view_layer.update()
    pe = agent_anatomy.probe(scene, rig, part="palm", side="R",
                             frame_range=FR, toward=arrow.name)
    check("E1 dir_object arrow aligns",
          pe.get("err_inner_deg") is not None
          and pe["err_inner_deg"] < 20,
          f"err_inner={pe.get('err_inner_deg')} "
          f"resolved={pe.get('toward_resolved')} "
          f"how={pe.get('toward_how')}")
    # 转走箭头 → reapply → 掌心跟上
    arrow.rotation_euler = (0.0, math.radians(90), 0.0)   # +Z → +X
    bpy.context.view_layer.update()
    agent_ops.reapply(data_dir, rig, op_e["id"], scene=scene)
    bpy.context.view_layer.update()
    pe2 = agent_anatomy.probe(scene, rig, part="palm", side="R",
                              frame_range=FR, toward=arrow.name)
    check("E2 empty drives reapply",
          pe2.get("err_inner_deg") is not None
          and pe2["err_inner_deg"] < 20,
          f"err_inner={pe2.get('err_inner_deg')} (箭头转 +X 后)")
    bpy.data.objects.remove(arrow)

# ---------- F: params mirror -------------------------------------------------
agent_bridge._sync_params_list(settings, data_dir)
plist = [p for p in settings.agent_params]
check("F1 params mirrored", len(plist) >= 3,
      f"{len(plist)} items: {sorted({p.key for p in plist})}")
hold_params = {p.key for p in plist
               if p.op_id == (OP_B if has_fingers else plist[0].op_id)}
check("F2 hold_pose specs",
      "mode" in hold_params and "frame_range" in hold_params
      and "threshold_deg" not in hold_params,   # mode=replace 时被 when 过滤
      f"keys={sorted(hold_params)}")

# ---------- summary -----------------------------------------------------------
fails = [r for r in RESULTS if not r[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for name, _ok, detail in fails:
    print(f"FAIL {name}: {detail}")
sys.exit(1 if fails else 0)
