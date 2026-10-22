"""Pure-numpy checks of agent_pose's math (no Blender): python3 tests/test_agent_pose_math.py"""
import os
import sys
import types

import numpy as np

sys.modules.setdefault("bpy", types.ModuleType("bpy"))          # agent_pose imports bpy at top
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mocap_doctor.core import agent_pose as P  # noqa: E402

rng = np.random.default_rng(7)
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


q1 = P.quat_normalize(rng.normal(size=(50, 4)))
q2 = P.quat_normalize(rng.normal(size=(50, 4)))
m = P.quat_to_mat(P.qmul(q1, q2))
check("qmul matches matrix product", np.allclose(m, P.quat_to_mat(q1) @ P.quat_to_mat(q2), atol=1e-12))
check("mat→quat→mat round trip", np.allclose(P.quat_to_mat(P.mat_to_quat(P.quat_to_mat(q1))),
                                              P.quat_to_mat(q1), atol=1e-12))
rv = rng.normal(size=(50, 3)) * 0.9
check("rotvec round trip", np.allclose(P.quat_to_rotvec(P.rotvec_to_quat(rv)), rv, atol=1e-12))
check("qangle of identical = 0", P.qangle_deg(q1, -q1).max() < 1e-6)
mid = P.slerp(q1, q2, 0.5)
a1, a2 = P.qangle_deg(q1, mid), P.qangle_deg(mid, q2)
check("slerp midpoint halves the angle", np.allclose(a1, a2, atol=1e-6)
      and np.allclose(a1 + a2, P.qangle_deg(q1, q2), atol=1e-6))
check("slerp endpoints", P.qangle_deg(P.slerp(q1, q2, 0.0), q1).max() < 1e-6
      and P.qangle_deg(P.slerp(q1, q2, 1.0), q2).max() < 1e-6)
F = np.diag([-1.0, 1.0, 1.0])
th = np.radians(30.0)
qx = P.rotvec_to_quat(np.array([[th, 0, 0]]))
qy = P.rotvec_to_quat(np.array([[0, th, 0]]))
mx, _ = P.mirror_local(qx, None, F)
my, loc = P.mirror_local(qy, np.array([[0.1, 0.2, 0.3]]), F)
check("mirror: rotation about the mirror normal (X) is kept", P.qangle_deg(mx, qx).max() < 1e-9)
check("mirror: rotation about Y flips sign",
      P.qangle_deg(my, P.rotvec_to_quat(np.array([[0, -th, 0]]))).max() < 1e-9)
check("mirror: location X flips", np.allclose(loc, [[-0.1, 0.2, 0.3]]))
qs = P.quat_continuous(P.slerp(np.tile(q1[0], (10, 1)), np.tile(q2[0], (10, 1)), np.linspace(0, 1, 10)))
r_int = P.resample_quats(qs, 100, np.arange(100, 110, dtype=float))
r_half = P.resample_quats(qs, 100, [100.5])
check("resample at integer frames = identity", P.qangle_deg(r_int, qs).max() < 1e-6)
check("resample at half frame = slerp midpoint",
      P.qangle_deg(r_half, P.slerp(qs[:1], qs[1:2], 0.5)).max() < 1e-6)
t = np.arange(60, dtype=float)
speed = np.where((t > 20) & (t < 40), 10 * np.sin((t - 20) / 20 * np.pi), 0.0)
ev = P.detect_events(speed, onset_frac=0.15, stop_frac=0.12)
check("detect_events on a synthetic bump", ev["peak"] == 30 and 20 <= ev["onset"] <= 22
      and 38 <= ev["stop"] <= 40, ev)
ax, ang = P.motion_axis(qx[0], P.qmul(qx, qy)[0])
check("motion_axis recovers local axis/angle", np.allclose(ax, [0, 1, 0], atol=1e-9)
      and abs(ang - 30.0) < 1e-9, (ax, ang))
check("mirror_name", P.mirror_name("hand_fk.L") == "hand_fk.R"
      and P.mirror_name("f_index.01.R") == "f_index.01.L" and P.mirror_name("head") is None)
print(f"\n==== {len(fails)} FAIL ====")
sys.exit(1 if fails else 0)
