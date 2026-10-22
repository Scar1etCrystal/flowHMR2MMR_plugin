"""Timing + output digest of the wizard's auto-fix steps (向导自动修复步骤).

    bash tools/mcd.sh run tests/bench_wizard.py -- --label before [--reps 1]

The shipped fixture is at step 9 with every step committed, and the
pre-step checkpoints live on the user's machine - so the steps refuse to
re-run.  In THIS throwaway process only, the "already committed" guard is
relaxed and each step runs on a freshly reloaded fixture (identical input).
Nothing is saved; the project data dir points at a scratch folder (steps
write 181 MB checkpoints) that is wiped afterwards.

Output digest per step = sha1 over every action's keyframes (foreach_get
co / interpolation) + every object's matrix_world - two runs of the same code
must match exactly; an optimisation must keep them identical.
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
SCRATCH = os.path.join(KIT, "sandbox", "tmp", "bench_wizard_data")
# Steps may save the work file (contacts → save_workfile): run on a throwaway
# copy so the shared e2e fixture is never overwritten.
SRC = bpy.data.filepath
FIX = os.path.join(KIT, "sandbox", "tmp", "bench_wizard_fixture.blend")
STEPS = ["source_check", "source_floor", "contacts", "global_correction", "tilt",
         "ground_feet", "target_floor", "foot_lock"]
if "--steps" in ARGV:
    STEPS = ARGV[ARGV.index("--steps") + 1].split(",")

OPS._require_restore_before_rerun = lambda scene, step_id: None   # throwaway only


def digest():
    """Actions only - the step's real product.  Object matrices are NOT part of
    the oracle: the MMD rig's hair/skirt are rigid-body physics objects whose
    transforms depend on simulation-cache state (frame history), not on the
    step's math (they made two runs of identical code look different)."""
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


def obj_digest():
    h = hashlib.sha1()
    for ob in sorted(bpy.context.scene.objects, key=lambda o: o.name):
        if ob.rigid_body is not None:
            continue
        h.update(ob.name.encode())
        h.update(np.asarray(ob.matrix_world, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


res = {}
for step in STEPS:
    times, outs = [], []
    for _ in range(REPS):
        shutil.copyfile(SRC, FIX)          # pristine input every time (contacts saves)
        bpy.ops.wm.open_mainfile(filepath=FIX)
        shutil.rmtree(SCRATCH, ignore_errors=True)
        os.makedirs(SCRATCH, exist_ok=True)
        st = bpy.context.scene.mocap_doctor
        st.data_directory = SCRATCH
        rbw = bpy.context.scene.rigidbody_world
        no_rb = "--no-rb" in ARGV and rbw is not None and rbw.enabled
        if no_rb:                       # experiment: physics suspended during the step
            rbw.enabled = False
        prof = None
        if "--profile" in ARGV and ARGV[ARGV.index("--profile") + 1] == step:
            import cProfile
            prof = cProfile.Profile()
            prof.enable()
        t0 = time.perf_counter()
        try:
            r = bpy.ops.mocap_doctor.run_step(step_id=step)
            msg = st.status_message
        except Exception as exc:  # noqa: BLE001
            r, msg = f"EXC {exc!r}"[:160], ""
        times.append(time.perf_counter() - t0)
        if prof is not None:
            import io
            import pstats
            prof.disable()
            buf = io.StringIO()
            pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(35)
            print("PROFILE " + step + "\n" + buf.getvalue(), flush=True)
        if no_rb:
            rbw.enabled = True
        outs.append((str(r), msg[:100], digest(), obj_digest()))
    res[step] = {"median_s": statistics.median(times), "runs": len(times),
                 "result": outs[0][0], "msg": outs[0][1], "digest": outs[0][2],
                 "obj_digest": outs[0][3],
                 "deterministic": len({o[2] for o in outs}) == 1,
                 "obj_deterministic": len({o[3] for o in outs}) == 1}
    print(f"WIZBENCH {step:18s} {res[step]['median_s'] * 1000:9.1f} ms  {outs[0][0]}  "
          f"digest={outs[0][2]}  {outs[0][1]!r}", flush=True)
shutil.rmtree(SCRATCH, ignore_errors=True)
for leftover in (FIX, FIX + "1"):
    if os.path.exists(leftover):
        os.remove(leftover)
out = os.path.join(KIT, "logs", f"wizbench_{LABEL}.json")
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"WIZBENCH written {out}")
