"""弱模型快速上手（docs/prompts/05_quickstart_weak_model.md）里的每一条命令都在 fixture 上跑一遍。

  Q0   文档里每条 `tools/agent <tool> '<json>'` 都是合法 JSON、工具名存在、参数名都被接受
  Q1   按文档顺序执行：除了文档自己说"没有模板就别用"的 apply_exemplar，全部 ok
  Q2   覆盖全部写工具（hold_pose reapply clean_jitter restore_accent fix_ground solve_pelvis apply_exemplar
       motion_copy anticipation follow_through overshoot overlap time_warp foot_lock markers set_influence
       revert ab_toggle swivel）
  Q3   例子真的做到了文档说的事：掌心朝镜头、左膝朝前 err_inner < 5°；contact.L:13 drift < 1 mm；
       motion_copy compare_motion < 0.05°；ab_toggle 两次后回到"修复生效"
  Q4   fixture 文件没被写（save 只校验参数、绝不执行）
  Q5   list_timeline_markers 读场景里的 M 键标记、接上覆盖帧的标注区间、能过滤、反向帧段报 E_RANGE
       （标记在内存里建、跑完删掉，不动共享 fixture）
"""
import json
import os
import re
import sys

import addon_utils
import bpy

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import agent_bridge  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)


KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(KIT, "docs", "prompts", "05_quickstart_weak_model.md")
scene = bpy.context.scene
st = scene.mocap_doctor
rig = st.mmr_rig or next(o for o in scene.objects if o.type == "ARMATURE" and o.name.startswith("RIG-"))
st.mmr_rig = rig
data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "e2e_quickstart_data")
os.makedirs(data_dir, exist_ok=True)
st.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)

text = open(DOC, encoding="utf-8").read()
PAT = re.compile(r"^\s*<套件>/tools/agent (\S+) '(.*)'\s*$")
cmds, bad_json = [], []
for ln, line in enumerate(text.splitlines(), 1):
    m = PAT.match(line)
    if not m:
        continue
    tool, raw = m.group(1), m.group(2)
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        bad_json.append((ln, tool, str(exc)))
        continue
    cmds.append((ln, tool, args))
tools = set(agent_bridge.TOOLS)
unknown = [(ln, t) for ln, t, _a in cmds if t not in tools]
check("Q0 every documented command is valid JSON with a real tool name",
      cmds and not bad_json and not unknown,
      f"{len(cmds)} commands; bad_json={bad_json} unknown={unknown}")

EXPECT_FAIL = {"apply_exemplar"}
# 绝不执行 save：它会把**共享 fixture** 存盘（2026-10-04 本测试第一版就这样把 fixture_1499.blend 覆盖了一次，
# 原件留在 fixture_1499.blend1）。save 只检查参数合法，不调用。
NEVER_RUN = {"save"}
blend_mtime0 = os.path.getmtime(bpy.data.filepath)
WRITES = {"hold_pose", "reapply", "clean_jitter", "restore_accent", "fix_ground", "solve_pelvis",
          "apply_exemplar", "motion_copy", "anticipation", "follow_through", "overshoot", "overlap",
          "time_warp", "foot_lock", "markers", "set_influence", "revert", "ab_toggle", "swivel"}
last_op = None
fails, ran, toggles = [], set(), []
for ln, tool, args in cmds:
    a = json.loads(json.dumps(args).replace("<ME>", "qs").replace("<模板名>", "nonexistent_template"))
    if "<op_id>" in json.dumps(a):
        if last_op is None:
            fails.append((ln, tool, "no op yet for <op_id>"))
            continue
        a = json.loads(json.dumps(a).replace("<op_id>", last_op))
    if tool in NEVER_RUN:
        ran.add(tool)
        if set(a) - {"agent_id"}:
            fails.append((ln, tool, f"save takes only agent_id, got {sorted(a)}"))
        continue
    r = agent_bridge._dispatch({"tool": tool, "args": a})
    ran.add(tool)
    if tool in EXPECT_FAIL:
        if r["ok"] or "模板" not in str((r.get("error") or {}).get("message")):
            fails.append((ln, tool, f"expected a clear 'no template' error, got ok={r['ok']} {r.get('error')}"))
        continue
    if not r["ok"]:
        fails.append((ln, tool, (r.get("error") or {}).get("message", "")[:200]))
        continue
    d = r.get("data") or {}
    if tool == "ab_toggle":
        toggles.append(d.get("muted"))
    if isinstance(d, dict) and d.get("op_id") and not d.get("dry_run") and tool not in ("revert",):
        last_op = d["op_id"]
check("Q1 all documented commands run on the fixture in document order (apply_exemplar fails with a clear message)",
      not fails, fails or f"{len(cmds)} ran")
check("Q2 the quick start covers every write tool", WRITES <= ran, f"missing={sorted(WRITES - ran)}")


def call(tool, **args):
    return agent_bridge._dispatch({"tool": tool, "args": dict(args, agent_id="qs")})


r1 = call("probe_anatomy", part="palm", side="R", frame_range=[120, 150], toward="camera")
r2 = call("probe_anatomy", part="knee_front", side="L", frame_range=[200, 260], toward="forward")
r3 = call("slide_report", side="L", frame_range=[400, 450])
row = next((x for x in (r3.get("data") or {}).get("rows", []) if x["interval"] == "contact.L:13"), {})
mc = [o for o in (call("list_ops", owner="qs", live=True).get("data") or {}).get("ops", [])
      if o.get("tool") == "motion_copy" and o.get("status") == "preview"]
r4 = call("compare_motion", op_id=mc[-1]["id"]) if mc else {"ok": False}
check("Q3 the examples do what the doc says: palm→camera & knee→forward err_inner < 5°, contact.L:13 drift < 1 mm, "
      "mirror copy < 0.05°, ab_toggle ends un-muted",
      (r1.get("data") or {}).get("err_inner_deg", 99) < 5 and (r2.get("data") or {}).get("err_inner_deg", 99) < 5
      and row.get("drift_mm", 99) < 1.0 and (r4.get("data") or {}).get("err_inner_deg", 99) < 0.05
      and toggles == [True, False],
      f"palm {(r1.get('data') or {}).get('err_inner_deg')}° knee {(r2.get('data') or {}).get('err_inner_deg')}° "
      f"drift {row.get('drift_mm')} mm copy {(r4.get('data') or {}).get('err_inner_deg')}° toggles={toggles}")

check("Q4 the fixture file was never written (no save)", os.path.getmtime(bpy.data.filepath) == blend_mtime0,
      f"mtime {blend_mtime0} → {os.path.getmtime(bpy.data.filepath)}")

# Q5: the bridge end of list_timeline_markers - reads the real scene and joins
# each marker to the annotation intervals covering that frame.  Markers are
# created in memory and removed again, so the shared fixture stays untouched.
sc = bpy.context.scene
_keep = [(m.name, m.frame) for m in sc.timeline_markers]
for m in list(sc.timeline_markers):
    sc.timeline_markers.remove(m)
try:
    sc.timeline_markers.new("出拳", frame=505)
    sc.timeline_markers.new("落地", frame=620)
    q5 = call("list_timeline_markers")
    items = (q5.get("data") or {}).get("items") or []
    frames = [i["frame"] for i in items]
    kinds = {c["kind"] for i in items for c in (i.get("covered_by") or [])}
    empty = call("list_timeline_markers", frame_range=[900, 950])
    bad = call("list_timeline_markers", frame_range=[560, 505])
    check("Q5 list_timeline_markers reads the scene, joins covering intervals, "
          "filters, and rejects a reversed range with E_RANGE",
          q5.get("ok") and frames == [505, 620] and kinds
          and empty.get("ok") and (empty.get("data") or {}).get("total") == 0
          and not bad.get("ok") and (bad.get("error") or {}).get("code") == "E_RANGE",
          f"frames={frames} kinds={sorted(kinds)} empty={(empty.get('data') or {}).get('total')} "
          f"bad={(bad.get('error') or {}).get('code')}")
finally:
    for m in list(sc.timeline_markers):
        sc.timeline_markers.remove(m)
    for _name, _frame in _keep:
        sc.timeline_markers.new(_name, frame=_frame)

fl = [x for x in RESULTS if not x[1]]
print(f"\n==== {len(RESULTS) - len(fl)}/{len(RESULTS)} PASS ====")
for name, _ok, detail in fl:
    print("FAILED:", name, "::", detail)
if fl:
    sys.exit(1)
