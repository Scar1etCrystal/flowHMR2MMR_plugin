"""Timing + output digest of the export-side wizard steps (MMD Bake → VMD 导出准备 → 导出 VMD).

    bash tools/mcd.sh run tests/bench_export.py -- --label before [--reps 1]

Same idea as bench_wizard.py: a throwaway copy of the fixture (step 9), the
"already committed" guard relaxed in THIS process only, project data in a
scratch folder.  The chain is run the way the GUI runs it (bake → accept →
prep → accept → export).  Oracles: sha1 over every action's keyframes after
each step, and the exported .vmd file's bytes - an optimisation must keep
all of them identical.
"""
import hashlib
import json
import os
import shutil
import statistics
import sys
import time

import addon_utils
import bpy
import numpy as np

addon_utils.enable("bl_ext.user_default.mocap_doctor")
from bl_ext.user_default.mocap_doctor import operators as OPS  # noqa: E402

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
LABEL = ARGV[ARGV.index("--label") + 1] if "--label" in ARGV else "run"
REPS = int(ARGV[ARGV.index("--reps") + 1]) if "--reps" in ARGV else 1
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(KIT, "sandbox", "tmp", "bench_export_data")
SRC = bpy.data.filepath
FIX = os.path.join(KIT, "sandbox", "tmp", "bench_export_fixture.blend")

OPS._require_restore_before_rerun = lambda scene, step_id: None   # throwaway only


def digest():
    h = hashlib.sha1()
    for act in sorted(bpy.data.actions, key=lambda a: a.name):
        h.update(act.name.encode())
        for fc in act.fcurves:
            n = len(fc.keyframe_points)
            co = np.zeros(2 * n, dtype=np.float32)
            fc.keyframe_points.foreach_get("co", co)
            h.update(f"{fc.data_path}[{fc.array_index}]:{n}".encode())
            h.update(co.tobytes())
    return h.hexdigest()[:16]


PROFILE = ARGV[ARGV.index("--profile") + 1] if "--profile" in ARGV else ""


def timed(fn, name=""):
    prof = None
    if name and name == PROFILE:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    t0 = time.perf_counter()
    try:
        r = str(fn())
    except Exception as exc:  # noqa: BLE001
        r = f"EXC {exc!r}"[:200]
    dt = time.perf_counter() - t0
    if prof is not None:
        import io
        import pstats
        prof.disable()
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(30)
        print("PROFILE " + name + "\n" + buf.getvalue(), flush=True)
    return dt, r


runs = []
for rep in range(REPS):
    shutil.copyfile(SRC, FIX)
    bpy.ops.wm.open_mainfile(filepath=FIX)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)
    scene = bpy.context.scene
    st = scene.mocap_doctor
    st.data_directory = SCRATCH
    row = {"state": {"current_step": st.current_step, "preview": st.preview_step_id,
                     "bake_mode": st.mmd_bake_mode}}
    t, r = timed(lambda: bpy.ops.mocap_doctor.mmd_bake(), "mmd_bake")
    row["mmd_bake"] = {"s": t, "r": r, "msg": st.status_message[:160], "digest": digest()}
    row["accept_bake"] = timed(lambda: bpy.ops.mocap_doctor.accept_preview())[1]
    scene["mcd_hand_ik_export_confirmed"] = True
    t, r = timed(lambda: bpy.ops.mocap_doctor.run_step(step_id="export_prep"), "export_prep")
    row["export_prep"] = {"s": t, "r": r, "msg": st.status_message[:160], "digest": digest()}
    import glob
    for rp in sorted(glob.glob(os.path.join(SCRATCH, "**", "*export_prep*.json"), recursive=True)):
        try:
            rep_json = json.load(open(rp, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        hc = (rep_json.get("report") or rep_json).get("mmr_hand_export_cleanup")
        if hc:
            row["export_prep"]["hand_cleanup"] = {k: v for k, v in hc.items() if k != "arm_fk_bake"}
            gb = (rep_json.get("report") or rep_json).get("global_correction_bake") or {}
            row["export_prep"]["root_bake"] = {k: v for k, v in gb.items() if not isinstance(v, (list, dict))}
    row["accept_prep"] = timed(lambda: bpy.ops.mocap_doctor.accept_preview())[1]
    vmd = os.path.join(SCRATCH, "bench_export.vmd")
    st.vmd_export_path = vmd
    t, r = timed(lambda: bpy.ops.mocap_doctor.export_vmd(), "export")
    vmd_sha = (hashlib.sha1(open(vmd, "rb").read()).hexdigest()[:16]
               if os.path.isfile(vmd) else None)
    row["export"] = {"s": t, "r": r, "msg": st.status_message[:160], "vmd_sha1": vmd_sha,
                     "vmd_bytes": os.path.getsize(vmd) if vmd_sha else 0}
    runs.append(row)
    print("EXPBENCH", json.dumps(row, ensure_ascii=False), flush=True)

res = {"runs": runs}
for step in ("mmd_bake", "export_prep", "export"):
    res[step] = {"median_s": statistics.median(r[step]["s"] for r in runs),
                 "result": runs[0][step]["r"],
                 "digest": runs[0][step].get("digest") or runs[0][step].get("vmd_sha1"),
                 "deterministic": len({(r[step].get("digest") or r[step].get("vmd_sha1"))
                                       for r in runs}) == 1}
shutil.rmtree(SCRATCH, ignore_errors=True)
for leftover in (FIX, FIX + "1"):
    if os.path.exists(leftover):
        os.remove(leftover)
out = os.path.join(KIT, "logs", f"expbench_{LABEL}.json")
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"EXPBENCH written {out}")
