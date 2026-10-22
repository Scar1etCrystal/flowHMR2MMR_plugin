"""人看到的 ≠ agent 算的：方向词 / 角色与镜头参照系 / 膝肘朝向 / 标记体检 / swivel（2026-10-04 §16）。

  V1–V4   方向词：camera = 从部位指向镜头（旧版是视线方向 = 背对镜头）；away / screen_* / char_*；
          裸 left/right 报歧义；forward = 躯干朝向（逐帧，不是世界 −Y）
  V5–V7   镜像 / 左右：正面镜头角色左在画面右（mirrored），背面同向（same）；orient_report 的画面坐标
          与手算投影一致；screen_left 在正面镜头下 = 角色的右
  K1–K4   膝/肘读形变链（= MMD 骨 = 网格），旧 FK 读法的偏差；全片铰链标定（刚性 / 符号 / 半角关系 /
          MMD 膝铰链轴）
  M1–M5   markers create 膝肘脸胸骨盆：绑 ひざD/ひじ/頭/上半身2/下半身、初值 = 标定、刚性、probe 读回
  C1–C5   markers check：好的 ok；绑错侧 / 顶点父级在另一只手（掌心事件）/ 没父级 / Child Of 都报 error
  S1–S6   swivel：IK 膝（thigh_ik Y）、FK 肘（上臂 + 手反转），独立复测（MMD 骨）、脚踝/手腕不动、
          窗外不变、reapply 半力度、revert 还原
  H1–H3   hold_pose world_dir 方向词（camera / forward）、IK 腿 FK 骨写入被拒、world 向量提醒
  R1–R2   conventions / render_view（PNG + 场景状态还原）
  Q1–Q4   审查 M21：目标网格在视口里被禁用时仍按当帧姿态标定 / ground_report 网格模式量当帧；失败不缓存；
          读文件清缓存
"""
import math
import os
import sys

import addon_utils
import bpy
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import (  # noqa: E402
    agent_anatomy as A, agent_bridge, agent_view as V)
from mathutils import Matrix, Vector  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


scene = bpy.context.scene
st = scene.mocap_doctor
rig = st.mmr_rig or next(o for o in scene.objects if o.type == "ARMATURE" and o.name.startswith("RIG-"))
st.mmr_rig = rig
mmd = st.mmd_armature
data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_align_data")
os.makedirs(data_dir, exist_ok=True)
st.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)


def call(tool, **args):
    r = agent_bridge._dispatch({"tool": tool, "args": args})
    if not r["ok"]:
        print("ERR", tool, r.get("error"), (r.get("trace") or "")[-600:])
    return r


def ang(a, b):
    a, b = Vector(a).normalized(), Vector(b).normalized()
    return math.degrees(math.acos(max(-1.0, min(1.0, a.dot(b)))))


def W(arm, bn, end="head"):
    pb = arm.pose.bones[bn]
    return arm.matrix_world @ (pb.head if end == "head" else pb.tail)


def zdir(obj):
    return (obj.matrix_world.to_quaternion() @ Vector((0, 0, 1))).normalized()


def make_cam(name, eye, look_at):
    cd = bpy.data.cameras.new(name)
    ob = bpy.data.objects.new(name, cd)
    scene.collection.objects.link(ob)
    d = (look_at - eye).normalized()
    ob.matrix_world = Matrix.Translation(eye) @ d.to_track_quat("-Z", "Y").to_matrix().to_4x4()
    return ob


call("markers", action="remove", all=True)
cam0 = scene.camera

# ---------------------------------------------------------------- V: direction words
F = 230
scene.frame_set(F)
cf = V.char_frame(rig, mmd=mmd)
pel = cf["centre"]
cam_front = make_cam("e2e_cam_front", pel + cf["forward"] * 3.0 + Vector((0, 0, 0.4)), pel + Vector((0, 0, 0.4)))
cam_back = make_cam("e2e_cam_back", pel - cf["forward"] * 3.0 + Vector((0, 0, 0.4)), pel + Vector((0, 0, 0.4)))
scene.camera = cam_front
bpy.context.view_layer.update()
palmR = A.part_anchor(rig, "palm", "R")
t_cam, how = V.resolve_direction("camera", scene=scene, armature=rig, origin=palmR)
truth = (cam_front.matrix_world.translation - palmR).normalized()
viewdir = cam_front.matrix_world.to_quaternion() @ Vector((0, 0, -1))
check("V1 'camera' = from the part to the camera eye (not the camera's view direction)",
      ang(t_cam, truth) < 0.1 and t_cam.dot(viewdir) < -0.9,
      f"vs truth {ang(t_cam, truth):.4f}° dot(view_dir)={t_cam.dot(viewdir):.3f} how={how}")
t_away, _ = V.resolve_direction("away", scene=scene, armature=rig, origin=palmR)
t_sl, _ = V.resolve_direction("screen_left", scene=scene, armature=rig, origin=palmR)
cam_right = cam_front.matrix_world.to_quaternion() @ Vector((1, 0, 0))
check("V2 away = −camera, screen_left = camera −X",
      ang(t_away, -truth) < 0.1 and ang(t_sl, -cam_right) < 0.1,
      f"away {ang(t_away, -truth):.4f}° screen_left {ang(t_sl, -cam_right):.4f}°")
errs = []
for bad in ("left", "right"):
    try:
        V.resolve_direction(bad, scene=scene, armature=rig, origin=palmR)
        errs.append(f"{bad}: no error")
    except RuntimeError as exc:
        if "char_" not in str(exc) or "screen_" not in str(exc):
            errs.append(f"{bad}: {exc}")
check("V3 bare 'left'/'right' rejected with char_/screen_ hint", not errs, errs or "ok")
# forward = 躯干（MMD 下半身 + 上半身2 的静止前方随骨转，与 RIG 独立），逐帧
fw_err, minus_y = [], []
for f in (1, 151, 451, 901, 1351):
    scene.frame_set(f)
    t_fw, _ = V.resolve_direction("forward", scene=scene, armature=rig)
    mm = []
    for bn in ("下半身", "上半身2"):
        pb = mmd.pose.bones[bn]
        q = (mmd.matrix_world @ pb.matrix).to_quaternion()
        v = q @ (pb.bone.matrix_local.to_quaternion().inverted() @ Vector((0, -1, 0)))
        mm.append(Vector((v.x, v.y, 0)).normalized())
    truth = (mm[0] + mm[1]).normalized()
    fw_err.append(round(ang(t_fw, truth), 2))
    minus_y.append(round(ang(t_fw, Vector((0, -1, 0))), 1))
check("V4 'forward' = character torso facing per frame (MMD torso truth < 3°), not world −Y",
      max(fw_err) < 3.0 and max(minus_y) > 20.0, f"vs MMD torso {fw_err}° ; vs world −Y {minus_y}°")

# V8：跟着身体走的目标——toes（同侧脚尖，"膝盖对着脚尖"）/ part:<部位>.<侧>
scene.frame_set(F)
toeL = A.probe(scene, rig, part="toe", side="L", frame_range=[F, F])["world_dir"]
t_toes, how_t = V.resolve_direction("toes", scene=scene, armature=rig, side="L")
t_part, _ = V.resolve_direction("part:toe.R", scene=scene, armature=rig)
toeR = A.probe(scene, rig, part="toe", side="R", frame_range=[F, F])["world_dir"]
need_side = False
try:
    V.resolve_direction("toes", scene=scene, armature=rig)
except RuntimeError as exc:
    need_side = "part:toe" in str(exc)
check("V8 'toes' = same-side toe direction; 'part:toe.R' works; 'toes' without a side asks for one",
      ang(t_toes, toeL) < 0.1 and ang(t_part, toeR) < 0.1 and need_side,
      f"toes.L {ang(t_toes, toeL):.4f}° part:toe.R {ang(t_part, toeR):.4f}° how={how_t} need_side={need_side}")

# ---------------------------------------------------------------- V5–V7: 镜像 / 左右
scene.frame_set(F)
bpy.context.view_layer.update()
res = {}
for label, cam in (("front", cam_front), ("back", cam_back)):
    scene.camera = cam
    bpy.context.view_layer.update()
    b = V.view_basis(scene, "camera")
    lr = V.lr_mapping(cf, b)
    xL = V.screen_point(b, W(rig, "ORG-upper_arm.L"))["x"]
    xR = V.screen_point(b, W(rig, "ORG-upper_arm.R"))["x"]
    sl, _ = V.resolve_direction("screen_left", scene=scene, armature=rig, origin=palmR)
    res[label] = (lr["kind"], xL, xR, round(sl.dot(cf["left"]), 3))
check("V5 front camera: mirrored, the character's LEFT shoulder is on screen RIGHT, screen_left = char right",
      res["front"][0] == "mirrored" and res["front"][1] > res["front"][2] and res["front"][3] < -0.9,
      f"front={res['front']}")
check("V6 back camera: same, LEFT shoulder on screen LEFT, screen_left = char left",
      res["back"][0] == "same" and res["back"][1] < res["back"][2] and res["back"][3] > 0.9,
      f"back={res['back']}")
scene.camera = cam_front
bpy.context.view_layer.update()
r = call("orient_report", part="palm", side="L", frame=F, view="camera")
fr0 = (r.get("data") or {}).get("frames", [{}])[0]
b = V.view_basis(scene, "camera")
scene.frame_set(F)
pL = A.part_anchor(rig, "palm", "L")
dL = Vector(fr0.get("world_dir") or (0, 0, 1))
x0, y0, _z0 = b["project"](pL)
x1, y1, _z1 = b["project"](pL + dL * 0.15)
manual = math.degrees(math.atan2(y1 - y0, (x1 - x0) * b["aspect"]))
rep = ((fr0.get("camera") or {}).get("arrow_on_screen") or {}).get("angle_deg")
pos_rep = ((fr0.get("camera") or {}).get("part_on_screen") or {})
check("V7 orient_report: screen position + 2D arrow angle match a manual projection; text mentions 画面",
      r["ok"] and rep is not None and abs(((rep - manual + 180) % 360) - 180) < 0.5
      and abs(pos_rep.get("x", -1) - round(x0, 3)) < 0.002 and "画面" in fr0.get("text", ""),
      f"angle rep={rep} manual={manual:.2f} pos={pos_rep.get('where')} text={fr0.get('text')}")

# ---------------------------------------------------------------- K: 膝肘读形变链 + 铰链标定
fk_off, conv_off, mmd_off = [], [], []
for f in range(100, 1400, 50):
    scene.frame_set(f)
    p0, p1, p2, chain = A.limb_points(rig, "knee", "L")
    mk = W(mmd, "ひざ.L")
    mmd_off.append((p1 - mk).length * 1000)
    fk1 = W(rig, "thigh_fk.L", "tail")
    fk_off.append((p1 - fk1).length * 1000)
    s1, s2 = (p1 - p0).normalized(), (p2 - p1).normalized()
    f0, f2 = W(rig, "thigh_fk.L"), W(rig, "shin_fk.L", "tail")
    g1, g2 = (fk1 - f0).normalized(), (f2 - fk1).normalized()
    if s1.angle(s2) > math.radians(15) and g1.angle(g2) > math.radians(15):
        conv_off.append(ang(s1 - s2, g1 - g2))
check("K1 knee reads the deform chain: == MMD ひざ (mesh) to 0.1 mm; the old FK reading was off",
      chain == "ORG" and max(mmd_off) < 0.1 and np.median(fk_off) > 10.0,
      f"vs MMD max {max(mmd_off):.3f} mm; old FK knee off median {np.median(fk_off):.1f} mm max {max(fk_off):.1f} mm, "
      f"convex dir off median {np.median(conv_off):.1f}° max {max(conv_off):.1f}°")
cal = {}
for kind in ("knee", "elbow"):
    for s in ("L", "R"):
        cal[(kind, s)] = A.hinge_calibration(rig, s, kind, scene)
ck = [cal[("knee", s)] for s in "LR"]
ce = [cal[("elbow", s)] for s in "LR"]
check("K2 knee hinge rigid on the shin (p90 < 10°), sign = convex side 100%, front = bisector + half bend",
      all(c.get("spread_p90_deg", 99) < 10 and c.get("sign_ok_frac") == 1.0
          and c.get("vs_bisector_minus_half_med_deg", 99) < 2.0 for c in ck),
      [(c.get("anchor"), c.get("spread_med_deg"), c.get("spread_p90_deg"), c.get("spread_max_deg"),
        c.get("sign_ok_frac"), c.get("vs_bisector_minus_half_med_deg"), c.get("bend_frames")) for c in ck])
check("K3 elbow hinge on the forearm (p90 < 25°), sign 100%",
      all(c.get("spread_p90_deg", 99) < 25 and c.get("sign_ok_frac") == 1.0 for c in ce),
      [(c.get("anchor"), c.get("spread_med_deg"), c.get("spread_p90_deg"), c.get("spread_max_deg"),
        c.get("sign_ok_frac"), c.get("vs_bisector_minus_half_med_deg")) for c in ce])
check("K4 knee hinge axis vs the model author's MMD ひざ hinge (IK limit X axis) < 15°",
      all((c.get("vs_mmd_hinge_axis_med_deg") or 99) < 15 for c in ck),
      [c.get("vs_mmd_hinge_axis_med_deg") for c in ck])

# ---------------------------------------------------------------- M: 新部位标记
scene.frame_set(F)
r = call("markers", action="create", parts=["knee", "elbow", "face", "chest", "pelvis", "palm"], frame=F)
rows = {x["name"]: x for x in (r.get("data") or {}).get("markers", [])}
want = {"MCD_knee.L": "ひざD.L", "MCD_knee.R": "ひざD.R", "MCD_elbow.L": "ひじ.L", "MCD_elbow.R": "ひじ.R",
        "MCD_face": "頭", "MCD_chest": "上半身2", "MCD_pelvis": "下半身", "MCD_palm.L": "手首.L"}
bad = {n: (rows.get(n) or {}).get("bone") for n, b0 in want.items() if (rows.get(n) or {}).get("bone") != b0}
check("M1 create binds knee→ひざD, elbow→ひじ, face→頭, chest→上半身2, pelvis→下半身 (MMD = mesh bones)",
      r["ok"] and not bad and all(bpy.data.objects[n].parent == mmd for n in want),
      bad or f"{len(rows)} created, skin offsets {[(n, x.get('skin_offset_mm')) for n, x in rows.items() if x.get('skin_offset_mm')]}")
scene.frame_set(F)
with A.ignore_markers():
    pk = A.probe(scene, rig, part="knee_front", side="L", frame_range=[F, F])
    jk = A._joint_frame(rig, "L", "knee")
    je = A._joint_frame(rig, "R", "elbow")
    pf = A.probe(scene, rig, part="face", frame_range=[F, F])
check("M2 initial direction = rigid hinge calibration / rest-front definition (< 0.5°), sources hinge/rest",
      ang(zdir(bpy.data.objects["MCD_knee.L"]), jk["front_rigid"]) < 0.5
      and ang(zdir(bpy.data.objects["MCD_elbow.R"]), je["front_rigid"]) < 0.5
      and ang(zdir(bpy.data.objects["MCD_face"]), pf["world_dir"]) < 0.5
      and pk["evidence"].get("knee_source") == "hinge" and pf["evidence"].get("face_source") == "rest",
      f"knee {ang(zdir(bpy.data.objects['MCD_knee.L']), jk['front_rigid']):.3f}° elbow "
      f"{ang(zdir(bpy.data.objects['MCD_elbow.R']), je['front_rigid']):.3f}° face "
      f"{ang(zdir(bpy.data.objects['MCD_face']), pf['world_dir']):.3f}°")
spreads = {}
for name, ref in (("MCD_knee.L", "ORG-shin.L"), ("MCD_knee.R", "ORG-shin.R"), ("MCD_elbow.L", "ORG-forearm.L"),
                  ("MCD_elbow.R", "ORG-forearm.R"), ("MCD_face", "ORG-spine.006"), ("MCD_chest", "ORG-spine.003"),
                  ("MCD_pelvis", "ORG-spine")):
    locs = []
    for f in range(1, 1500, 10):
        scene.frame_set(f)
        q = (rig.matrix_world @ rig.pose.bones[ref].matrix).to_quaternion()
        locs.append(q.inverted() @ zdir(bpy.data.objects[name]))
    m = sum(locs, Vector()).normalized()
    spreads[name] = round(max(ang(m, v) for v in locs), 2)
check("M3 bound arrows are rigid in the visible bone frame over 150 frames (spread < 2.5°; knee on the ひざD "
      "mesh bone vs ORG-shin ≈ 1.7°)", all(v < 2.5 for v in spreads.values()), spreads)
# 标记 vs 当帧凸出方向：明显弯曲的帧上同侧（外凸侧），角度差 = 半个弯角 ± 铰链散布
agree, sides = [], []
for f in range(1, 1500, 10):
    scene.frame_set(f)
    p0, p1, p2, _c = A.limb_points(rig, "knee", "L")
    s1, s2 = (p1 - p0).normalized(), (p2 - p1).normalized()
    bend = math.degrees(s1.angle(s2))
    if bend > 30:
        bis = (s1 - s2).normalized()
        mk = zdir(bpy.data.objects["MCD_knee.L"])
        sides.append(mk.dot(bis) > 0)
        agree.append(abs(ang(mk, bis) - bend / 2))
check("M4 knee marker on the kneecap side on every bent frame; |angle − bend/2| median < 4°",
      all(sides) and np.median(agree) < 4.0,
      f"{len(sides)} frames bend>30°, same side {sum(sides)}/{len(sides)}, |Δ−half| median {np.median(agree):.2f}° p90 "
      f"{np.percentile(agree, 90):.2f}°")
scene.frame_set(F)
p2 = A.probe(scene, rig, part="knee_front", side="L", frame_range=[200, 260])
pc = A.probe(scene, rig, part="chest", frame_range=[200, 260])
check("M5 probe prefers the markers (knee_source / chest_source = marker)",
      p2["evidence"].get("knee_source") == "marker" and pc["evidence"].get("chest_source") == "marker",
      f"knee={p2['evidence'].get('knee_source')} chest={pc['evidence'].get('chest_source')}")

# ---------------------------------------------------------------- C: markers check
r = call("markers", action="check", frame_range=[100, 1400])
arr = {x["name"]: x for x in (r.get("data") or {}).get("arrows", [])}
okn = [n for n in want if (arr.get(n) or {}).get("status") == "ok"]
check("C1 check: freshly created markers are all ok", len(okn) == len(want),
      {n: (arr.get(n) or {}).get("problems") for n in want if n not in okn})
# 绑错侧：MCD_palm.R 改绑到 手首.L
mkR = bpy.data.objects["MCD_palm.R"]
mkR.parent_bone = "手首.L"
bpy.context.view_layer.update()
# 掌心事件：顶点父级的三个顶点在右手（手首.R 权重 1.0），箭头放在左掌心
mesh = st.target_mesh
gi = mesh.vertex_groups["手首.R"].index
vids = [v.index for v in mesh.data.vertices if any(g.group == gi and g.weight > 0.99 for g in v.groups)][:3]
inc = bpy.data.objects.new("UserArrow.vtx", None)
inc.empty_display_type = "SINGLE_ARROW"
scene.collection.objects.link(inc)
inc.parent = mesh
inc.parent_type = "VERTEX_3"
inc.parent_vertices = vids
inc.matrix_parent_inverse = Matrix.Identity(4)
scene.frame_set(F)
bpy.context.view_layer.update()
W0 = inc.matrix_world.copy()          # 顶点三角形的当前坐标系（basis = 单位阵时）
inc.matrix_basis = W0.inverted() @ Matrix.Translation(A.part_anchor(rig, "palm", "L") + Vector((0, 0, 0.01)))
bpy.context.view_layer.update()
stat = bpy.data.objects.new("UserArrow.static", None)
stat.empty_display_type = "SINGLE_ARROW"
scene.collection.objects.link(stat)
stat.matrix_world = Matrix.Translation(A.part_anchor(rig, "knee_front", "R"))
cho = bpy.data.objects.new("UserArrow.childof", None)
cho.empty_display_type = "SINGLE_ARROW"
scene.collection.objects.link(cho)
c = cho.constraints.new("CHILD_OF")
c.target = mmd
c.subtarget = "ひじ.L"
good = bpy.data.objects.new("UserArrow.good", None)
good.empty_display_type = "SINGLE_ARROW"
scene.collection.objects.link(good)
good.parent, good.parent_type, good.parent_bone = mmd, "BONE", "足首.R"
good.matrix_parent_inverse = Matrix.Identity(4)
bpy.context.view_layer.update()
with A.ignore_markers():
    ps = A.probe(scene, rig, part="sole", side="R", frame_range=[F, F])
good.matrix_world = Matrix.Translation(A.part_anchor(rig, "sole", "R")) @ \
    Vector(ps["world_dir"]).to_track_quat("Z", "Y").to_matrix().to_4x4()
bpy.context.view_layer.update()
r = call("markers", action="check", frame_range=[100, 1400])
arr = {x["name"]: x for x in (r.get("data") or {}).get("arrows", [])}


def has(n, *words):
    x = arr.get(n) or {}
    txt = " ".join(x.get("problems") or [])
    return x.get("status") == "error" and all(w in txt for w in words), (x.get("status"), txt[:160])


ok_side, d_side = has("MCD_palm.R", "手首.L", "L 侧")
ok_vtx, d_vtx = has("UserArrow.vtx", "顶点父级", "手首.R", "左")
check("C2 wrong-side bone parent → error (MCD_palm.R on 手首.L)", ok_side, d_side)
check("C3 palm incident: vertex parent on the RIGHT hand while sitting on the left palm → error + not rigid",
      ok_vtx and (arr.get("UserArrow.vtx") or {}).get("rigid_spread_deg", 0) > 5, d_vtx)
ok_st, d_st = has("UserArrow.static", "没有父级")
ok_co, d_co = has("UserArrow.childof", "Child Of")
check("C4 unparented arrow and Child Of binding → error", ok_st and ok_co, (d_st, d_co))
gx = arr.get("UserArrow.good") or {}
check("C5 a correct user bone-parented arrow → ok + adopt hint",
      gx.get("status") == "ok" and "adopt" in str(gx.get("fix")) and gx.get("part") == "sole" and gx.get("side") == "R",
      (gx.get("status"), gx.get("part"), gx.get("side"), gx.get("fix"), gx.get("problems")))
for o in (inc, stat, cho, good):
    bpy.data.objects.remove(o, do_unlink=True)
call("markers", action="remove", all=True)


# ---------------------------------------------------------------- S: swivel
def mmd_swivel_err(side, joint, target_word, frames, min_bend=35.0):
    """独立复测：用 MMD 骨（网格跟的骨）的凸出方向（当帧弯曲角的平分线 = 用户看到的膝盖/肘尖），
    投影到 根→梢 垂面上与目标比。只在明显弯着（> min_bend）的帧上量：直腿时凸出方向没有定义。"""
    names = {"knee": ("足.{s}", "ひざ.{s}", "足首.{s}"), "elbow": ("腕.{s}", "ひじ.{s}", "手首.{s}")}[joint]
    out = []
    for f in frames:
        scene.frame_set(f)
        P0, P1, P2 = [W(mmd, n.format(s=side)) for n in names]
        s1, s2 = (P1 - P0).normalized(), (P2 - P1).normalized()
        if math.degrees(s1.angle(s2)) < min_bend:
            continue
        t, _ = V.resolve_direction(target_word, scene=scene, armature=rig, origin=P1)
        e = A.swivel_error_deg((s1 - s2).normalized(), t, (P2 - P0).normalized())
        if e is not None:
            out.append(abs(e))
    return out


A_, B_ = 200, 260
inner = list(range(A_ + 1, B_))
pre = mmd_swivel_err("L", "knee", "forward", inner)
outside = {}
for f in (A_ - 8, B_ + 8):
    scene.frame_set(f)
    outside[f] = W(mmd, "ひざ.L").copy()
r = call("swivel", joint="knee", side="L", frame_range=[A_ - 4, B_ + 4], toward="forward")
m = (r.get("data") or {}).get("metrics") or {}
post = mmd_swivel_err("L", "knee", "forward", inner)
moved_out = max((W(mmd, "ひざ.L") - outside[f]).length * 1000 for f in outside if scene.frame_set(f) is None)
pr = call("probe_anatomy", part="knee_front", side="L", frame_range=[A_, B_], toward="forward", max_frames=31)
check("S1 swivel IK knee (thigh_ik Y): solver err 0, ankle/hip fixed, foot rotation unchanged",
      r["ok"] and m.get("limb") == "IK" and m.get("control") == "thigh_ik.L" and m.get("err_after_inner_deg", 99) < 1.0
      and m.get("end_drift_mm", 99) < 0.5 and m.get("root_drift_mm", 99) < 0.5 and m.get("end_rot_change_max_deg", 99) < 0.2,
      {k: m.get(k) for k in ("err_before_inner_deg", "err_after_inner_deg", "ik_gain_median", "end_drift_mm",
                             "root_drift_mm", "end_rot_change_max_deg", "joint_moved_max_mm", "applied_max_deg")})
check("S2 independent re-test on the MMD (mesh) bones, frames bent > 35°: knee faces character-forward (< 1.5°); "
      "probe err_inner < 2°; frames outside the window untouched",
      len(post) >= 3 and max(post) < 1.5 and (pr.get("data") or {}).get("err_inner_deg", 99) < 2.0 and moved_out < 0.01,
      f"MMD swivel err ({len(post)} bent frames) before max {max(pre):.1f}° → after max {max(post):.2f}°; probe "
      f"{(pr.get('data') or {}).get('err_inner_deg')}°; outside moved {moved_out:.4f} mm")
op_knee = (r.get("data") or {}).get("op_id")
# "膝盖别内扣" = 膝盖对着脚尖：同一段改朝 toes（reapply 换目标），复测按 toes 量
r = call("reapply", op_id=op_knee, overrides={"toward": "toes"})
pt = call("probe_anatomy", part="knee_front", side="L", frame_range=[A_, B_], toward="toes", max_frames=31)
check("S1b swivel toward 'toes' (knee tracks over the toes): reapply with the new target, probe err_inner < 2°",
      r["ok"] and (pt.get("data") or {}).get("err_inner_deg", 99) < 2.0
      and ((r.get("data") or {}).get("metrics") or {}).get("toward_how", "").startswith("toes.L"),
      f"err_inner={(pt.get('data') or {}).get('err_inner_deg')} how={((r.get('data') or {}).get('metrics') or {}).get('toward_how')}")
r = call("reapply", op_id=op_knee, overrides={"toward": "forward"})
# FK 肘：朝角色外侧（右肘 → char_right）
EA, EB = 600, 650
pre_e = mmd_swivel_err("R", "elbow", "char_right", list(range(EA + 1, EB)))
hand_q0, wrist0 = {}, {}
for f in range(EA - 4, EB + 5):
    scene.frame_set(f)
    hand_q0[f] = (mmd.matrix_world @ mmd.pose.bones["手首.R"].matrix).to_quaternion()
    wrist0[f] = W(rig, "ORG-hand.R").copy()
r = call("swivel", joint="elbow", side="R", frame_range=[EA - 4, EB + 4], toward="char_right")
m = (r.get("data") or {}).get("metrics") or {}
post_e = mmd_swivel_err("R", "elbow", "char_right", list(range(EA + 1, EB)))
dq, dw = [], []
for f in range(EA - 4, EB + 5):
    scene.frame_set(f)
    dq.append(math.degrees(hand_q0[f].rotation_difference(
        (mmd.matrix_world @ mmd.pose.bones["手首.R"].matrix).to_quaternion()).angle))
    dw.append((W(rig, "ORG-hand.R") - wrist0[f]).length * 1000)
check("S3 swivel FK elbow: upper_arm_fk + hand_fk counter-rotation; wrist position & hand (手首, mesh) orientation unchanged "
      "on EVERY frame incl. the blend ramps",
      r["ok"] and m.get("limb") == "FK" and m.get("err_after_inner_deg", 99) < 1.0 and max(dw) < 0.5 and max(dq) < 0.2,
      f"err {m.get('err_before_inner_deg')}→{m.get('err_after_inner_deg')}° wrist drift max {max(dw):.3f} mm "
      f"hand rot change max {max(dq):.3f}° elbow moved {m.get('joint_moved_max_mm')} mm")
check("S4 independent MMD re-test (bent > 35°): right elbow points to the character's right (< 1.5°)",
      len(post_e) >= 3 and max(post_e) < 1.5, f"{len(post_e)} frames: before max {max(pre_e):.1f}° → after max {max(post_e):.2f}°")
op_el = (r.get("data") or {}).get("op_id")
r = call("reapply", op_id=op_knee, overrides={"strength": 0.5})
ph = call("probe_anatomy", part="knee_front", side="L", frame_range=[A_, B_], toward="forward", max_frames=31)
e_before = (r.get("data") or {}).get("metrics", {}).get("err_before_inner_deg")
e_half = (ph.get("data") or {}).get("err_inner_deg")
check("S5 reapply strength 0.5 → half the swivel error remains (probe err_inner ≈ before/2 ± 25%)",
      r["ok"] and e_before and e_half is not None and 0.375 * e_before < e_half < 0.625 * e_before,
      f"before {e_before}° → half-strength {e_half}°")
call("revert", op_id=op_knee)
call("revert", op_id=op_el)
back = mmd_swivel_err("L", "knee", "forward", inner)
stuck = rig.pose.bones["thigh_ik.L"].rotation_euler[1]
check("S6 revert restores the original knee direction exactly (no 'stuck' thigh_ik value)",
      len(back) == len(pre) and max(abs(a - b) for a, b in zip(back, pre)) < 0.05 and abs(stuck) < 1e-9,
      f"max |Δ| {max(abs(a - b) for a, b in zip(back, pre)):.4f}° thigh_ik.L Y={stuck}")

# ---------------------------------------------------------------- H: hold_pose 方向词 / 守卫
scene.camera = cam_front
bpy.context.view_layer.update()
pb0 = call("probe_anatomy", part="palm", side="R", frame_range=[120, 150], toward="camera")
r = call("hold_pose", bones=["right_hand"], frame_range=[116, 154], target="world_dir", world_dir="camera",
         world_axis="probe:palm.R", secondary_axis="probe:hand_axis.R", blend=4)
pa = call("probe_anatomy", part="palm", side="R", frame_range=[120, 150], toward="camera")
orr = call("orient_report", part="palm", side="R", frame=135, view="camera")
fr1 = ((orr.get("data") or {}).get("frames") or [{}])[0]
check("H1 hold_pose world_dir='camera' → palm faces the camera (probe err_inner < 3°; orient_report '正对镜头', "
      "arrow foreshortened)",
      r["ok"] and (pa.get("data") or {}).get("err_inner_deg", 99) < 3.0 and "正对镜头" in fr1.get("text", "")
      and ((fr1.get("camera") or {}).get("arrow_on_screen") or {}).get("foreshorten", 1) < 0.35,
      f"{(pb0.get('data') or {}).get('err_inner_deg')}° → {(pa.get('data') or {}).get('err_inner_deg')}°; "
      f"{fr1.get('text')}")
call("revert", op_id=(r.get("data") or {}).get("op_id"))
r = call("hold_pose", bones=["left_hand"], frame_range=[146, 194], target="world_dir", world_dir="forward",
         world_axis="probe:palm.L", secondary_axis="probe:hand_axis.L", blend=4)
pa = call("probe_anatomy", part="palm", side="L", frame_range=[150, 190], toward="forward")
check("H2 world_dir='forward' follows the character per frame (probe toward forward err_inner < 3°)",
      r["ok"] and (pa.get("data") or {}).get("err_inner_deg", 99) < 3.0 and
      (r.get("data") or {}).get("params_echo", {}).get("world_dir") == "forward",
      f"err_inner {(pa.get('data') or {}).get('err_inner_deg')}°")
call("revert", op_id=(r.get("data") or {}).get("op_id"))
r1 = call("hold_pose", bones=["thigh_fk.L"], frame_range=[200, 230], target="values")
r2 = call("probe_anatomy", part="palm", side="L", frame_range=[150, 190], toward=[0, -1, 0])
r3 = call("get_joint_angles", bones=["left_knee"], frame_range=[200, 230])
check("H3 invisible FK-leg write rejected; world-vector toward warns when the character faces elsewhere; "
      "snapshot leg roles warn",
      (not r1["ok"]) and "swivel" in str(r1.get("error")) and any("forward" in w for w in r2.get("warnings", []))
      and any("IK" in w for w in r3.get("warnings", [])),
      f"{str(r1.get('error'))[:90]} | {r2.get('warnings')} | {r3.get('warnings')}")

# ---------------------------------------------------------------- R: conventions / render_view
r = call("conventions", frame=F)
d = r.get("data") or {}
check("R1 conventions: frame rule, limb modes (legs IK / arms FK), character facing, camera L/R mapping",
      r["ok"] and "Blender" in d.get("frames", {}).get("rule", "") and "IK" in d.get("limbs", {}).get("leg.L", "")
      and "FK" in d.get("limbs", {}).get("arm.R", "") and d.get("camera", {}).get("lr", {}).get("kind") == "mirrored",
      f"{r.get('summary')} | limbs={d.get('limbs', {}).get('leg.L', '')[:30]}")
n_obj, cam_before, eng = len(bpy.data.objects), scene.camera, scene.render.engine
call("markers", action="create", parts=["palm"], sides=["R"], frame=135)
n_obj += 1
r = call("render_view", frame=135, view="camera", part="palm", side="R", toward="camera", size=[320, 240])
d = r.get("data") or {}
ok_png = bool(d.get("path")) and os.path.exists(d["path"]) and os.path.getsize(d["path"]) > 5000
check("R2 render_view writes a PNG with marker/part/target arrows and restores the scene",
      r["ok"] and ok_png and len(bpy.data.objects) == n_obj and scene.camera == cam_before
      and scene.render.engine == eng and not [o for o in bpy.data.objects if o.name.startswith("MCD_render")]
      and {a["color"] for a in d.get("arrows", [])} >= {"绿", "红", "蓝"},
      f"{d.get('path')} {d.get('render_s')}s arrows={[(a['label'], a.get('arrow_on_screen')) for a in d.get('arrows', [])]}")
call("markers", action="remove", all=True)

# ---------------------------------------------------------------- Q: 审查 M21（网格在视口里被禁用 / 失败缓存）
mesh = st.target_mesh
A.reset_caches()
scene.frame_set(300)
p_vis = A.probe(scene, rig, part="palm", side="R", frame_range=[300, 300])
g_vis = call("ground_report", side="L", frame_range=[400, 450], mesh=True)
mesh.hide_viewport = True
bpy.context.view_layer.update()
not_eval = not mesh.visible_get()     # 注意：is_evaluated 此时仍是 True，几何却不求值
A.reset_caches()
p_hid = A.probe(scene, rig, part="palm", side="R", frame_range=[300, 300])
g_hid = call("ground_report", side="L", frame_range=[400, 450], mesh=True)
still_disabled = mesh.hide_viewport
check("Q1 target mesh disabled in viewport: palm calibration still from the POSED mesh (same local axis), "
      "mesh stays disabled afterwards",
      not_eval and p_hid["evidence"].get("palm_source") == "mesh"
      and ang(p_hid["local_axis"], p_vis["local_axis"]) < 0.1 and still_disabled,
      f"visible(hidden)={not not_eval} src={p_hid['evidence'].get('palm_source')} "
      f"Δaxis={ang(p_hid['local_axis'], p_vis['local_axis']):.4f}° still_disabled={still_disabled}")
mv = ((g_vis.get("data") or {}).get("mesh") or {}).get("L") or {}
mh = ((g_hid.get("data") or {}).get("mesh") or {}).get("L") or {}
check("Q2 ground_report mesh mode with the mesh disabled = the posed heights (Δ < 0.5 mm), not rest heights",
      g_hid["ok"] and mv.get("sole_min_mm") is not None and abs(mv["sole_min_mm"] - mh.get("sole_min_mm", 1e9)) < 0.5
      and mesh.hide_viewport,
      f"visible {mv.get('sole_min_mm')} mm vs disabled {mh.get('sole_min_mm')} mm")
mesh.hide_viewport = False
bpy.context.view_layer.update()
# 失败不缓存：顶点组临时改名 → 标定失败退回手指；改回来 → 同一帧立刻重新标定成功
vg = mesh.vertex_groups["手首.R"]
A.reset_caches()
vg.name = "手首.R_tmp"
p_bad = A.probe(scene, rig, part="palm", side="R", frame_range=[300, 300])
vg.name = "手首.R"
p_ok = A.probe(scene, rig, part="palm", side="R", frame_range=[300, 300])
check("Q3 a failed calibration is not cached: after fixing the cause the same frame recalibrates from the mesh",
      p_bad["evidence"].get("palm_source") == "fingers" and p_ok["evidence"].get("palm_source") == "mesh",
      f"broken={p_bad['evidence'].get('palm_source')}/{p_bad['evidence'].get('mesh_calibration')} "
      f"fixed={p_ok['evidence'].get('palm_source')}")
n_cal = len(A._PALM_CAL) + len(A._HINGE_CAL)
agent_bridge._on_file_loaded()
check("Q4 caches are cleared on file load (load_post handler)",
      n_cal > 0 and len(A._PALM_CAL) == 0 and len(A._HINGE_CAL) == 0 and len(A._PALM_FAIL) == 0,
      f"entries before={n_cal} after={len(A._PALM_CAL) + len(A._HINGE_CAL)}")

scene.camera = cam0
for o in (cam_front, cam_back):
    bpy.data.objects.remove(o, do_unlink=True)

fails = [x for x in RESULTS if not x[1]]
print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS ====")
for name, _ok, detail in fails:
    print("FAILED:", name, "::", detail)
if fails:
    sys.exit(1)
