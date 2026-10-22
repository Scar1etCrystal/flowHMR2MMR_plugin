"""Benchmark + golden-output capture for MoCap Doctor's agent layer (任务2 的尺子).

    bash tools/mcd.sh run tests/bench_baseline.py -- --label before
    bash tools/mcd.sh run tests/bench_baseline.py -- --label after
    python3 tests/bench_compare.py logs/bench_before.json logs/bench_after.json

Two jobs in one pass, both through ``agent_bridge._dispatch`` (so bridge
overhead - _ctx / store cache / envelope - is part of every number):

1. TIMING: each tool is timed ``--reps`` times (default 3) → median.  Write
   tools are measured as write→(record)→revert cycles so every repetition
   starts from the same scene state.
2. GOLDEN: every numeric output is captured - read payloads, op params and
   metrics, the written delta action's keyframes (co.x/co.y per fcurve) and
   the evaluated world matrices of the touched bones at sample frames.  The
   optimisation rule is "zero numeric change": bench_compare.py diffs two
   golden files exactly (floats via repr → JSON round-trip is lossless).

Volatile fields (op ids, timestamps, data versions, timings) are stripped
before the golden snapshot so two runs of the same code compare equal.
"""
from __future__ import annotations

import json
import os
import socket
import statistics
import sys
import threading
import time

import addon_utils
import bpy
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor.core import (  # noqa: E402
    agent_bake, agent_bridge, agent_ops)

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def _arg(name, default=None):
    if name in ARGV:
        i = ARGV.index(name)
        return ARGV[i + 1] if i + 1 < len(ARGV) else default
    return default


LABEL = _arg("--label", "run")
REPS = int(_arg("--reps", "3"))
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = _arg("--out", os.path.join(KIT, "logs"))
os.makedirs(OUT_DIR, exist_ok=True)

A, B = 150, 224                 # 75 帧（含两端）——掌心修复所在区段
MID = (A + B) // 2

scene = bpy.context.scene
settings = scene.mocap_doctor
rig = settings.mmr_rig or next(
    (o for o in scene.objects if o.type == "ARMATURE"
     and o.name.startswith("RIG-")), None)
settings.mmr_rig = rig
data_dir = os.path.join(os.path.dirname(bpy.data.filepath), "bench_data")
os.makedirs(data_dir, exist_ok=True)
settings.data_directory = data_dir
oplog = os.path.join(data_dir, "agent_ops.json")
if os.path.exists(oplog):
    os.remove(oplog)

TIMINGS: dict[str, dict] = {}
GOLDEN: dict[str, object] = {}
VOLATILE = {"id", "op_id", "ts", "version", "trace", "elapsed_s",
            "expires_at", "claimed_at", "renewed_at", "age_s", "ttl_left_s"}


def _clean(obj):
    """Strip volatile keys recursively; numpy → python; keep float repr."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
                if str(k) not in VOLATILE}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and (obj != obj):
        return "nan"
    return obj


def call(tool, **args):
    resp = agent_bridge._dispatch({"tool": tool, "args": args})
    if not resp.get("ok"):
        raise RuntimeError(f"{tool} failed: {resp.get('error')} {resp.get('trace', '')}")
    return resp


def timed(name, fn, reps=REPS, setup=None, teardown=None):
    ts = []
    out = None
    for i in range(reps):
        if setup:
            setup()
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
        if teardown:
            teardown(out)
    TIMINGS[name] = {"median_s": statistics.median(ts), "min_s": min(ts),
                     "max_s": max(ts), "reps": reps}
    print(f"BENCH {name:<34s} median={statistics.median(ts) * 1000:9.1f} ms  "
          f"min={min(ts) * 1000:9.1f}  max={max(ts) * 1000:9.1f}", flush=True)
    return out


def action_keys(action):
    """Every keyframe of a (delta) action: {data_path[index]: [[x, y], ...]}."""
    out = {}
    if action is None:
        return out
    for fc in action.fcurves:
        co = [0.0] * (2 * len(fc.keyframe_points))
        fc.keyframe_points.foreach_get("co", co)
        interp = sorted({kp.interpolation for kp in fc.keyframe_points})
        out[f"{fc.data_path}[{fc.array_index}]"] = {
            "co": co, "interp": interp, "extrap": fc.extrapolation}
    return out


def pose_snapshot(bones, frames):
    """Evaluated world matrices of bones at frames (scene frame restored)."""
    out = {}
    cur = scene.frame_current
    for f in frames:
        scene.frame_set(int(f))
        for b in bones:
            pb = rig.pose.bones.get(b)
            if pb is None:
                continue
            m = rig.matrix_world @ pb.matrix
            out[f"{b}@{f}"] = [list(row) for row in m]
    scene.frame_set(cur)
    return out


def op_golden(resp, bones):
    data = resp.get("data") or {}
    op_id = data.get("op_id") or data.get("id")
    op = agent_ops.get_op(data_dir, op_id) if op_id else None
    strip_keys = {}
    if op is not None:
        _tr, strip = agent_ops.find_op_strip(rig, op)
        if strip is not None:
            strip_keys = action_keys(strip.action)
            strip_keys["__strip__"] = {
                "frame_start": strip.frame_start, "frame_end": strip.frame_end,
                "influence": strip.influence, "blend_type": strip.blend_type,
                "extrapolation": strip.extrapolation}
    return {"envelope": _clean({k: v for k, v in resp.items() if k != "data"}),
            "data": _clean(data),
            "op": _clean(op) if op else None,
            "keys": _clean(strip_keys),
            "pose": _clean(pose_snapshot(bones, [A, A + 4, MID, B - 4, B]))}


def revert_resp(resp):
    data = resp.get("data") or {}
    op_id = data.get("op_id")
    if op_id:
        agent_ops.revert(rig, data_dir, op_id)


# ---------------------------------------------------------------- scene stats
meshes = [o for o in scene.objects if o.type == "MESH"]
stats = {
    "objects": len(scene.objects),
    "meshes": len(meshes),
    "mesh_verts": sum(len(o.data.vertices) for o in meshes),
    "armatures": [o.name for o in scene.objects if o.type == "ARMATURE"],
    "rig": rig.name,
    "rig_bones": len(rig.pose.bones),
    "rig_constraints": sum(len(pb.constraints) for pb in rig.pose.bones),
    "frames": [scene.frame_start, scene.frame_end],
    "blender": bpy.app.version_string,
}
print("BENCH scene", json.dumps(stats, ensure_ascii=False), flush=True)

# ---------------------------------------------------------------- primitives
fr_cur = scene.frame_current


def _fs30():
    for f in range(A, A + 30):
        scene.frame_set(f)


timed("prim.frame_set x30", _fs30)
scene.frame_set(fr_cur)
base_act = agent_ops.base_action_of(rig)
qpath = 'pose.bones["hand_fk.L"].rotation_quaternion'
timed("prim.sample_fcurve 4ch x75f",
      lambda: [agent_bake.sample_fcurve_values(base_act, qpath, i, A, B)
               for i in range(4)])
GOLDEN["prim.sample_fcurve"] = _clean(
    [agent_bake.sample_fcurve_values(base_act, qpath, i, A, B) for i in range(4)])

# ---------------------------------------------------------------- read tools
GOLDEN["ping"] = _clean({k: v for k, v in call("ping")["data"].items()
                         if k != "version"})
timed("tool.ping", lambda: call("ping"), reps=20)

# store: cold (no npz) → warm (npz, no memory) → hot (memory)
cache_dir = os.path.join(data_dir, "agent_cache")
if os.path.isdir(cache_dir):
    for fn in os.listdir(cache_dir):
        os.remove(os.path.join(cache_dir, fn))
agent_bridge._STORE = None
t0 = time.perf_counter()
ov = call("get_overview")
TIMINGS["store.cold_bake(get_overview)"] = {"median_s": time.perf_counter() - t0, "reps": 1}
print(f"BENCH store.cold_bake(get_overview)   {TIMINGS['store.cold_bake(get_overview)']['median_s'] * 1000:9.1f} ms", flush=True)
GOLDEN["get_overview"] = _clean(ov.get("data"))


def _drop_mem():
    agent_bridge._STORE = None


timed("store.warm_npz(get_overview)", lambda: call("get_overview"), setup=_drop_mem)
timed("tool.get_overview(hot)", lambda: call("get_overview"))

GOLDEN["describe"] = _clean(call("describe", target=[A, B]).get("data"))
timed("tool.describe 75f", lambda: call("describe", target=[A, B]))


def _bump():   # simulates "another agent wrote something": version moved
    agent_bridge._DATA_VERSION += 1


timed("tool.describe 75f after-version-bump",
      lambda: call("describe", target=[A, B]), setup=_bump)
GOLDEN["bake_range"] = _clean(call("bake_range", frame_range=[A, B]).get("data"))
timed("tool.bake_range 75f", lambda: call("bake_range", frame_range=[A, B]))
GOLDEN["get_series"] = _clean(call(
    "get_series", channels=["pelvis.speed"], frame_range=[A, B]).get("data"))
timed("tool.get_series", lambda: call("get_series", channels=["pelvis.speed"],
                                      frame_range=[A, B]))
GOLDEN["get_joint_angles"] = _clean(call(
    "get_joint_angles", bones=["left_hand", "left_forearm"],
    frame_range=[A, B]).get("data"))
timed("tool.get_joint_angles", lambda: call(
    "get_joint_angles", bones=["left_hand", "left_forearm"], frame_range=[A, B]))

probe_args = dict(part="palm", side="L", frame_range=[A, B], toward=[0, -1, 0])
GOLDEN["probe_palm"] = _clean(call("probe_anatomy", **probe_args).get("data"))
timed("tool.probe_anatomy palm 75f(9 smp)",
      lambda: call("probe_anatomy", **probe_args))
probe_full = dict(probe_args, max_frames=75)
GOLDEN["probe_palm_75"] = _clean(call("probe_anatomy", **probe_full).get("data"))
timed("tool.probe_anatomy palm 75f(75 smp)",
      lambda: call("probe_anatomy", **probe_full), reps=1)
GOLDEN["probe_sole"] = _clean(call("probe_anatomy", part="sole", side="R",
                                   frame_range=[A, B]).get("data"))
GOLDEN["probe_body"] = _clean(call("probe_anatomy", part="body_forward",
                                   frame_range=[A, B]).get("data"))
GOLDEN["validate"] = _clean(call("validate", frame_range=[A, B]).get("data"))

# ---------------------------------------------------------------- write tools
HP_WORLD = dict(bones=["left_hand"], frame_range=[A, B], target="world_dir",
                world_dir=[0, -1, 0], world_axis="probe:palm.L",
                secondary_axis="probe:finger_dir.L", blend=4)
r = call("hold_pose", **HP_WORLD)
GOLDEN["hold_pose.world_dir"] = op_golden(r, ["hand_fk.L", "f_index.03.L"])
revert_resp(r)
timed("tool.hold_pose world_dir 75f", lambda: call("hold_pose", **HP_WORLD),
      teardown=revert_resp)

HP_MULTI = dict(bones=["left_hand", "right_hand", "spine2"], frame_range=[A, B],
                target="values", blend=6, mode="clamp", threshold_deg=10)
r = call("hold_pose", **HP_MULTI)
GOLDEN["hold_pose.values3"] = op_golden(r, ["hand_fk.L", "hand_fk.R",
                                            "spine_fk.001"])
revert_resp(r)
timed("tool.hold_pose values 3bones 75f", lambda: call("hold_pose", **HP_MULTI),
      teardown=revert_resp)

HP_WORLD2 = dict(HP_WORLD, bones=["left_hand", "right_hand"],
                 world_axis="Y", secondary_axis=None)
r = call("hold_pose", **HP_WORLD2)
GOLDEN["hold_pose.world_dir2"] = op_golden(r, ["hand_fk.L", "hand_fk.R"])
revert_resp(r)
timed("tool.hold_pose world_dir 2bones 75f",
      lambda: call("hold_pose", **HP_WORLD2), teardown=revert_resp)

# reapply: same op, changed blend - measured as a pure rewrite
r = call("hold_pose", **HP_WORLD)
rid = r["data"]["op_id"]
rr = call("reapply", op_id=rid, overrides={"blend": 8})
GOLDEN["reapply"] = op_golden(
    {"data": {"op_id": rid, **(rr.get("data") or {})}},
    ["hand_fk.L"])
timed("tool.reapply hold_pose 75f",
      lambda: call("reapply", op_id=rid, overrides={"blend": 8}))
GOLDEN["effect_check"] = _clean(call("effect_check", op_id=rid).get("data"))
timed("tool.effect_check", lambda: call("effect_check", op_id=rid))
GOLDEN["list_ops"] = _clean(call("list_ops").get("data"))
timed("tool.list_ops (1 op)", lambda: call("list_ops"))
agent_ops.revert(rig, data_dir, rid)

CJ = dict(frame_range=[A, B], bone="left_hand", strength=1.0, width=5)
r = call("clean_jitter", **CJ)
GOLDEN["clean_jitter"] = op_golden(r, ["hand_fk.L", "f_index.03.L"])
revert_resp(r)
timed("tool.clean_jitter 75f", lambda: call("clean_jitter", **CJ),
      teardown=revert_resp)

RA = dict(frame_range=[A, B], data_path='pose.bones["spine_fk.001"].rotation_quaternion',
          method="ease_reshape", strength=0.5)
r = call("restore_accent", **RA)
GOLDEN["restore_accent"] = op_golden(r, ["spine_fk.001", "head"])
revert_resp(r)
timed("tool.restore_accent quat 75f", lambda: call("restore_accent", **RA),
      teardown=revert_resp)

RL = dict(frame_range=[A, B], data_path='pose.bones["torso_root"].location',
          method="retime", strength=0.5)
r = call("restore_accent", **RL)
GOLDEN["restore_accent.loc"] = op_golden(r, ["torso_root"])
revert_resp(r)

# set_influence on a live op
r = call("hold_pose", **HP_WORLD)
iid = r["data"]["op_id"]
call("set_influence", op_id=iid, value=1.5)
GOLDEN["set_influence"] = op_golden({"data": {"op_id": iid}}, ["hand_fk.L"])
agent_ops.revert(rig, data_dir, iid)

# list_ops with a realistic log size (20 live ops)
ids = []
for k in range(20):
    a = 300 + k * 40
    ids.append(call("hold_pose", bones=["left_hand"], frame_range=[a, a + 30],
                    target="values", blend=4)["data"]["op_id"])
timed("tool.list_ops (20 ops)", lambda: call("list_ops"))
timed("tool.ping (20 ops in log)", lambda: call("ping"), reps=20)
GOLDEN["list_ops20"] = _clean([{k: v for k, v in row.items()}
                               for row in call("list_ops")["data"]["fixes"]])
t0 = time.perf_counter()
for i in ids:
    call("revert", op_id=i)
TIMINGS["tool.revert"] = {"median_s": (time.perf_counter() - t0) / len(ids),
                          "reps": len(ids)}
print(f"BENCH tool.revert (avg of 20)          {TIMINGS['tool.revert']['median_s'] * 1000:9.1f} ms", flush=True)

# ---------------------------------------------------------------- socket pump
# headless 模式的真实循环：主线程 _pump + sleep，客户端走真 socket。
PORT = 6299
agent_bridge.start_server(PORT)
rtts = []
done = threading.Event()


def _client():
    try:
        for _ in range(25):
            t0 = time.perf_counter()
            with socket.create_connection(("127.0.0.1", PORT), timeout=30) as s:
                s.sendall(b'{"id":1,"tool":"ping","args":{}}\n')
                buf = b""
                while b"\n" not in buf:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
            rtts.append(time.perf_counter() - t0)
    finally:
        done.set()


th = threading.Thread(target=_client, daemon=True)
th.start()
pump_loop = getattr(agent_bridge, "headless_pump_loop", None)
deadline = time.time() + 60
if pump_loop is not None:                      # 优化后的阻塞排水（若存在）
    pump_loop(until=lambda: done.is_set() or time.time() > deadline)
else:
    while not done.is_set() and time.time() < deadline:
        agent_bridge._pump()
        time.sleep(agent_bridge.TIMER_INTERVAL)
agent_bridge.stop_server()
if rtts:
    TIMINGS["socket.ping_rtt"] = {"median_s": statistics.median(rtts),
                                  "min_s": min(rtts), "max_s": max(rtts),
                                  "reps": len(rtts)}
    print(f"BENCH socket.ping_rtt                   median={statistics.median(rtts) * 1000:9.1f} ms  "
          f"min={min(rtts) * 1000:.1f} max={max(rtts) * 1000:.1f}", flush=True)

# ---------------------------------------------------------------- write out
res = {"label": LABEL, "stats": stats, "timings": TIMINGS,
       "golden": GOLDEN, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
path = os.path.join(OUT_DIR, f"bench_{LABEL}.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(res, fh, ensure_ascii=False, indent=0)
print(f"GOLDEN written {path}  ({len(GOLDEN)} entries)", flush=True)
print("==== BENCH DONE ====", flush=True)
