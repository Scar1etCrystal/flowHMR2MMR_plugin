"""markers 工具 + probe 读标记：用户先绑箭头，agent 以箭头为准（2026-10-04）。

  M1–M4  create：MCD_palm.L/R、MCD_sole.L/R 建成、骨骼父级在 MMD 骨架上、初值 = 网格标定方向
  M5–M7  刚性：标记方向在 hand_fk 局部坐标里跨 150 帧不变；probe 读到 palm_source=marker
  M8–M9  用户转动标记 → probe 跟着变；ignore_markers 下回到网格标定
  M10    hold_pose(probe:palm.R) 以标记为准，复测 err_inner < 5°
  M11    没父级的标记被忽略 + warning
  M12–13 bake knee_front：逐帧 K 帧箭头 = 形变链当帧凸出角平分线，与 probe knee_front 在 髋→踝 垂面上一致
  M14    list / remove
  M16–18 审查 M20：bake 不再写绑定标记同名物体（MCD_bake_*）；带关键帧的"绑定标记"不读回 + check 报错；
         create overwrite 清掉旧关键帧后读回正确；adopt 拒绝带关键帧的箭头
"""
import math
import os
import sys

import addon_utils
import bpy
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import (  # noqa: E402
    agent_anatomy, agent_bridge)
from mathutils import Matrix, Vector  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig or next(o for o in scene.objects if o.type == "ARMATURE"
                               and o.name.startswith("RIG-"))
settings.mmr_rig = rig
data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_markers_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)


def call(tool, **args):
    r = agent_bridge._dispatch({"tool": tool, "args": args})
    if not r["ok"]:
        print("ERR", tool, r.get("error"), r.get("trace"))
    return r


def ang(a, b):
    a, b = Vector(a).normalized(), Vector(b).normalized()
    return math.degrees(math.acos(max(-1.0, min(1.0, a.dot(b)))))


def zdir(obj):
    return obj.matrix_world.to_quaternion() @ Vector((0, 0, 1))


# clean slate
call("markers", action="remove", all=True)
scene.frame_set(179)

# ---- M1–M4 create
r = call("markers", action="create", parts=["palm", "sole"], sides=["L", "R"])
rows = (r.get("data") or {}).get("markers", [])
names = sorted(x["name"] for x in rows)
check("M1 create 4 bound markers", r["ok"] and names == ["MCD_palm.L", "MCD_palm.R", "MCD_sole.L", "MCD_sole.R"]
      and all(x["status"] == "created" for x in rows), f"{names} warnings={r.get('warnings')}")
mk = bpy.data.objects.get("MCD_palm.R")
mmd = settings.mmd_armature
check("M2 palm marker bone-parented to the MMD hand bone",
      mk is not None and mk.parent is not None and mk.parent_type == "BONE"
      and mk.parent_bone == "手首.R" and (mmd is None or mk.parent == mmd) and mk.get("mcd_marker") == "bound",
      f"parent={getattr(mk.parent, 'name', None)}/{getattr(mk, 'parent_bone', None)}")
with agent_anatomy.ignore_markers():
    pm = agent_anatomy.probe(scene, rig, part="palm", side="R", frame_range=[179, 179])
check("M3 marker initial direction = mesh-calibrated palm (< 0.5°)",
      pm["evidence"].get("palm_source") == "mesh" and ang(zdir(mk), pm["world_dir"]) < 0.5,
      f"angle={ang(zdir(mk), pm['world_dir']):.3f}° src={pm['evidence'].get('palm_source')} "
      f"place_err={[x.get('place_err_deg') for x in rows]}")
sk = bpy.data.objects.get("MCD_sole.L")
with agent_anatomy.ignore_markers():
    ps = agent_anatomy.probe(scene, rig, part="sole", side="L", frame_range=[179, 179])
check("M4 sole marker on 足首.L, initial = three-point sole normal (< 0.5°)",
      sk is not None and sk.parent_bone == "足首.L" and ang(zdir(sk), ps["world_dir"]) < 0.5,
      f"bone={getattr(sk, 'parent_bone', None)} angle={ang(zdir(sk), ps['world_dir']):.3f}°")

# ---- M5–M7 rigid follow + probe reads marker
locs = []
for f in range(1, 1500, 10):
    scene.frame_set(f)
    bpy.context.view_layer.update()
    q = (rig.matrix_world @ rig.pose.bones["hand_fk.R"].matrix).to_quaternion()
    locs.append(q.inverted() @ zdir(mk))
mean = sum(locs, Vector()).normalized()
spread = max(ang(mean, v) for v in locs)
check("M5 marker rigid in hand_fk space over 150 frames (spread < 1°)", spread < 1.0,
      f"spread={spread:.3f}° mean_local={[round(v, 4) for v in mean]}")
scene.frame_set(179)
p = agent_anatomy.probe(scene, rig, part="palm", side="R", frame_range=[150, 230])
ev = p.get("evidence", {})
check("M6 probe palm uses the marker", ev.get("palm_source") == "marker" and ev.get("marker") == "MCD_palm.R"
      and ev.get("marker_vs_mesh_deg", 99) < 0.5 and p.get("confidence") == 1.0,
      f"src={ev.get('palm_source')} marker_vs_mesh={ev.get('marker_vs_mesh_deg')} bone={ev.get('marker_bone')}")
hp = p.get("hold_pose_args", {})
check("M7 hold_pose_args still per-frame probe axes", hp.get("world_axis") == "probe:palm.R"
      and hp.get("secondary_axis") == "probe:hand_axis.R", hp)

# ---- M8–M9 user rotates the marker 20° about the hand axis → probe follows; ignore → mesh
hx = Vector(agent_anatomy.probe(scene, rig, part="hand_axis", side="R", frame_range=[179, 179])["world_dir"])
before = zdir(mk).copy()
rot = Matrix.Rotation(math.radians(20.0), 4, hx)
mk.matrix_world = rot @ mk.matrix_world
bpy.context.view_layer.update()
p2 = agent_anatomy.probe(scene, rig, part="palm", side="R", frame_range=[179, 179])
d_marker = ang(p2["world_dir"], before)
check("M8 rotated marker → probe palm follows (≈20°)", 18.0 < d_marker < 22.0 and p2["evidence"].get("marker_vs_mesh_deg", 0) > 18,
      f"delta={d_marker:.2f}° marker_vs_mesh={p2['evidence'].get('marker_vs_mesh_deg')}")
# still rigid after the user's adjustment (other frame)
scene.frame_set(600)
bpy.context.view_layer.update()
p3 = agent_anatomy.probe(scene, rig, part="palm", side="R", frame_range=[600, 600])
check("M9 adjusted marker stays 20° off the mesh palm at another frame",
      18.0 < p3["evidence"].get("marker_vs_mesh_deg", 0) < 22.0, f"@600 marker_vs_mesh={p3['evidence'].get('marker_vs_mesh_deg')}")
with agent_anatomy.ignore_markers():
    p4 = agent_anatomy.probe(scene, rig, part="palm", side="R", frame_range=[600, 600])
check("M9b ignore_markers → mesh calibration", p4["evidence"].get("palm_source") == "mesh"
      and 18.0 < ang(p4["world_dir"], p3["world_dir"]) < 22.0, f"src={p4['evidence'].get('palm_source')}")

# ---- M10 hold_pose follows the (adjusted) marker definition
scene.frame_set(179)
tgt = [0.0, -1.0, 0.0]
r = call("hold_pose", bones=["hand_fk.R"], frame_range=[300, 360], target="world_dir", world_dir=tgt,
         world_axis="probe:palm.R", secondary_axis="probe:hand_axis.R", mode="replace", blend=4)
pr = call("probe_anatomy", part="palm", side="R", frame_range=[304, 356], toward=tgt)
e1 = (pr.get("data") or {}).get("err_inner_deg")
src = ((pr.get("data") or {}).get("evidence") or {}).get("palm_source")
check("M10 hold_pose(probe:palm.R) aligns the MARKER to the target (err_inner < 5°, source=marker)",
      r["ok"] and e1 is not None and e1 < 5 and src == "marker", f"err_inner={e1} src={src} err={r.get('error')}")
if r["ok"]:
    call("revert", op_id=r["data"]["op_id"])

# ---- M11 unbound marker is ignored with a warning
mk.parent = None
bpy.context.view_layer.update()
pr = call("probe_anatomy", part="palm", side="R", frame_range=[179, 179])
evu = ((pr.get("data") or {}).get("evidence") or {})
check("M11 unparented marker ignored (mesh used) + warning",
      evu.get("palm_source") == "mesh" and evu.get("marker_ignored") == "MCD_palm.R"
      and any("MCD_palm.R" in w for w in pr.get("warnings", [])),
      f"src={evu.get('palm_source')} ignored={evu.get('marker_ignored')} warnings={pr.get('warnings')}")
r = call("markers", action="create", parts=["palm"], sides=["R"], overwrite=True)
mk = bpy.data.objects.get("MCD_palm.R")
check("M11b overwrite=true rebinds it", r["ok"] and mk.parent is not None and mk.parent_bone == "手首.R",
      f"status={[x['status'] for x in r['data']['markers']]}")

# ---- M12–M13 bake knee_front (per-frame keyed arrow == probe direction)
r = call("markers", action="bake", part="knee_front", side="L", frame_range=[200, 260])
kb = bpy.data.objects.get("MCD_bake_knee_front.L")
check("M12 bake creates a keyed arrow", r["ok"] and kb is not None and kb.get("mcd_marker") == "baked"
      and kb.animation_data is not None and kb.animation_data.action is not None,
      f"err={r.get('error')} data={r.get('data')}")
worst, worst_sw, bent = 0.0, 0.0, 0
for f in (205, 215, 225, 235, 245, 255):
    scene.frame_set(f)
    bpy.context.view_layer.update()
    # bake 显示的是"当帧凸出角平分线"（形变链 ORG-*，= 视口里的膝）；probe knee_front 是膝盖骨朝向
    # （⊥小腿，弯着时在当帧弯曲平面里）——两者在 髋→踝 垂面上的投影是同一个方向（§16）
    p0, p1, p2, _c = agent_anatomy.limb_points(rig, "knee", "L")
    bis = ((p1 - p0).normalized() - (p2 - p1).normalized()).normalized()
    worst = max(worst, ang(zdir(kb), bis))
    jd = agent_anatomy._joint_frame(rig, "L", "knee")
    if jd["evidence"]["bend_deg"] > 35.0:       # 直腿时凸出方向没有定义，只比明显弯着的帧
        bent += 1
        e = agent_anatomy.swivel_error_deg(jd["front"], zdir(kb), jd["chord"], pole=jd.get("pole"))
        if e is not None:
            worst_sw = max(worst_sw, abs(e))
check("M13 baked arrow = per-frame convex bisector of the deform chain (< 0.5°), and on bent frames agrees with "
      "probe knee_front in the swivel plane (< 0.5°)", worst < 0.5 and worst_sw < 0.5,
      f"vs bisector {worst:.3f}° swivel-plane vs probe {worst_sw:.3f}° ({bent} bent frames)")
r = call("markers", action="bake", part="palm", side="R", frame_range=[700, 710])
bp = bpy.data.objects.get("MCD_bake_palm.R")
mk = bpy.data.objects.get("MCD_palm.R")
check("M13b bake palm writes MCD_bake_palm.R and never touches the bound MCD_palm.R (审查 M20)",
      r["ok"] and bp is not None and bp.get("mcd_marker") == "baked" and mk.get("mcd_marker") == "bound"
      and mk.animation_data is None and mk.parent_bone == "手首.R",
      f"err={r.get('error')} baked={getattr(bp, 'name', None)} bound_anim={mk.animation_data}")

# ---- M15 adopt: the user's own bone-parented arrow becomes the definition
scene.frame_set(179)
bpy.context.view_layer.update()
mine = bpy.data.objects.new("UserArrow.L", None)
mine.empty_display_type = "SINGLE_ARROW"
scene.collection.objects.link(mine)
mmd_arm = settings.mmd_armature
mine.parent = mmd_arm
mine.parent_type = "BONE"
mine.parent_bone = "手首.L"
mine.matrix_parent_inverse = Matrix.Identity(4)
bpy.context.view_layer.update()
with agent_anatomy.ignore_markers():
    ref = agent_anatomy.probe(scene, rig, part="palm", side="L", frame_range=[179, 179])
hxl = Vector(agent_anatomy.probe(scene, rig, part="hand_axis", side="L", frame_range=[179, 179])["world_dir"])
# the user aimed it 10° off the mesh normal (about the hand axis)
d10 = Matrix.Rotation(math.radians(10.0), 4, hxl) @ Vector(ref["world_dir"])
mine.matrix_world = Matrix.Translation(Vector((0, 0, 1))) @ d10.to_track_quat("Z", "Y").to_matrix().to_4x4()
bpy.context.view_layer.update()
r = call("markers", action="adopt", name="UserArrow.L", part="palm", side="L", frame_range=[150, 230], overwrite=True)
dd = r.get("data") or {}
check("M15 adopt renames + tags the user's bone-parented arrow and measures it",
      r["ok"] and bpy.data.objects.get("MCD_palm.L") is not None and bpy.data.objects.get("UserArrow.L") is None
      and dd.get("bone") == "手首.L" and 9.0 < dd.get("vs_geometry_max_deg", 0) < 11.0,
      f"err={r.get('error')} data={dd}")
pa = agent_anatomy.probe(scene, rig, part="palm", side="L", frame_range=[600, 600])
check("M15b probe palm.L now uses the adopted arrow (10° from mesh at another frame)",
      pa["evidence"].get("palm_source") == "marker" and 9.0 < pa["evidence"].get("marker_vs_mesh_deg", 0) < 11.0,
      f"src={pa['evidence'].get('palm_source')} vs_mesh={pa['evidence'].get('marker_vs_mesh_deg')}")
r = call("markers", action="adopt", name="MCD_bake_knee_front.L", part="palm", side="L")
check("M15c adopt refuses a baked (keyed, unparented) arrow", not r["ok"] and ("关键帧" in str(r.get("error"))
                                                                              or "骨骼父级" in str(r.get("error"))),
      str(r.get("error"))[:100])
st0 = bpy.data.objects.new("UserArrow.static", None)
st0.empty_display_type = "SINGLE_ARROW"
scene.collection.objects.link(st0)
r = call("markers", action="adopt", name="UserArrow.static", part="palm", side="L")
check("M15d adopt refuses an arrow without a bone parent", not r["ok"] and "骨骼父级" in str(r.get("error")),
      str(r.get("error"))[:100])
bpy.data.objects.remove(st0, do_unlink=True)

# ---- M16–M18 审查 M20：带关键帧的"绑定标记"（旧文件里 bake 写到了同名物体上）
mk = bpy.data.objects["MCD_palm.R"]
scene.frame_set(100)
mk.keyframe_insert("rotation_euler", frame=100)
mk.rotation_euler.x += math.radians(60.0)
mk.keyframe_insert("rotation_euler", frame=800)
scene.frame_set(600)
bpy.context.view_layer.update()
pr = call("probe_anatomy", part="palm", side="R", frame_range=[600, 600])
evk = (pr.get("data") or {}).get("evidence") or {}
ck = call("markers", action="check", frame_range=[100, 900], name="MCD_palm.R")
rowk = ((ck.get("data") or {}).get("arrows") or [{}])[0]
check("M16 a keyed 'bound' marker is NOT read back (source=mesh, why=keyed, warning) and check reports it",
      evk.get("palm_source") == "mesh" and evk.get("marker_ignored_why") == "keyed"
      and any("关键帧" in w for w in pr.get("warnings", [])) and rowk.get("status") == "error"
      and any("关键帧" in p for p in rowk.get("problems", [])),
      f"src={evk.get('palm_source')} why={evk.get('marker_ignored_why')} check={rowk.get('status')} {rowk.get('problems')}")
r = call("markers", action="create", parts=["palm"], sides=["R"], overwrite=True)
mk = bpy.data.objects["MCD_palm.R"]
devs = []
for f in (300, 900):
    scene.frame_set(f)
    bpy.context.view_layer.update()
    pm = agent_anatomy.probe(scene, rig, part="palm", side="R", frame_range=[f, f])
    with agent_anatomy.ignore_markers():
        pg = agent_anatomy.probe(scene, rig, part="palm", side="R", frame_range=[f, f])
    devs.append((pm["evidence"].get("palm_source"), round(ang(pm["world_dir"], pg["world_dir"]), 3)))
check("M17 create overwrite:true clears the old keyframes; the rebuilt marker = mesh palm on other frames (< 0.5°)",
      r["ok"] and mk.animation_data is None and all(s_ == "marker" and d_ < 0.5 for s_, d_ in devs),
      f"anim={mk.animation_data} per-frame={devs}")
ua = bpy.data.objects.new("UserArrow.keyed", None)
ua.empty_display_type = "SINGLE_ARROW"
scene.collection.objects.link(ua)
ua.parent, ua.parent_type, ua.parent_bone = settings.mmd_armature, "BONE", "手首.L"
ua.keyframe_insert("location", frame=10)
r = call("markers", action="adopt", name="UserArrow.keyed", part="palm", side="L", overwrite=True)
check("M18 adopt refuses a bone-parented arrow that carries its own keyframes", not r["ok"] and "关键帧" in str(r.get("error")),
      str(r.get("error"))[:100])
bpy.data.objects.remove(ua, do_unlink=True)

# ---- M14 list / remove
scene.frame_set(179)
r = call("markers", action="list", frame_range=[150, 230])
rows = (r.get("data") or {}).get("markers", [])
by = {x["name"]: x for x in rows}
check("M14 list reports kinds + agreement with geometry", r["ok"] and by.get("MCD_palm.R", {}).get("kind") == "bound"
      and by["MCD_palm.R"].get("valid") is True and by["MCD_palm.R"].get("vs_geometry_max_deg", 99) < 0.5
      and by.get("MCD_bake_knee_front.L", {}).get("kind") == "baked",
      f"{[(x['name'], x.get('kind'), x.get('valid'), x.get('vs_geometry_max_deg')) for x in rows]}")
r = call("markers", action="remove", all=True)
check("M14b remove all", r["ok"] and not [o for o in bpy.data.objects if o.name.startswith("MCD_")],
      f"removed={r.get('data', {}).get('removed')}")

fails = [x for x in RESULTS if not x[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for name, _ok, detail in fails:
    print("FAILED:", name, "::", detail)
if fails:
    sys.exit(1)
