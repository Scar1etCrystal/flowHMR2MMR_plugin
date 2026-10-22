"""Dump the absolute joint positions (route B) out of a FlowHMR ``final.fbx``.

    blender -b --factory-startup --python tools/fbx_joints_dump.py -- \
        --fbx "F:\\flowhmr_pack\\outputs\\video_to_fbx\\t999_H4_...\\final.fbx" \
        --out "F:\\flowhmr_official_out\\t999_OFFICIAL_post_fbx_joints.npz"

Every SMPL joint is a bone head in the imported ``Reference`` armature (52 bones,
object scale 0.01 -> metres), so the absolute joints are just the posed bone
heads in world space, one sample per frame.

The file is written in the **Blender world** frame (Z-up); the plugin converts it
to the npz convention with ``core/foot_contacts.blender_world_to_npz``
(``npz = (-X_w, Z_w, +Y_w)``, a 180 deg yaw — see that module's docstring for the
measurements behind it).  Keeping the dump raw means the conversion lives in one
tested place instead of being baked into an artifact.

Put the result next to the official contact npz, named
``<npz 名>_fbx_joints.npz`` (e.g. ``t999_OFFICIAL_post_fbx_joints.npz``), and the
plugin picks it up automatically; or point 「fbx 绝对关节」 at it in the wizard.
"""

from __future__ import annotations

import argparse
import os
import sys

import bpy
import numpy as np


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(prog="fbx_joints_dump")
    ap.add_argument("--fbx", required=True, help="final.fbx 路径")
    ap.add_argument("--out", default="", help="输出 npz（默认 <fbx 同目录>/<fbx 名>_joints.npz）")
    return ap.parse_args(argv)


def main() -> int:
    args = _parse_args()
    fbx = os.path.abspath(args.fbx)
    if not os.path.isfile(fbx):
        print(f"[dump] FATAL: 找不到 {fbx}")
        return 2
    out = args.out or os.path.join(os.path.dirname(fbx),
                                   os.path.splitext(os.path.basename(fbx))[0] + "_joints.npz")
    out = os.path.abspath(out)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx)
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        print("[dump] FATAL: fbx 里没有骨架")
        return 2
    arm = arms[0]
    scene = bpy.context.scene
    pbs = list(arm.pose.bones)
    names = [pb.name for pb in pbs]
    parents = [pb.parent.name if pb.parent else "" for pb in pbs]
    act = arm.animation_data.action if arm.animation_data else None
    if act is None:
        print("[dump] FATAL: 骨架没有 action（fbx 里没有烘焙动画）")
        return 2
    f0, f1 = int(round(act.frame_range[0])), int(round(act.frame_range[1]))
    print(f"[dump] {arm.name}: {len(pbs)} bones, object scale "
          f"{tuple(round(v, 6) for v in arm.scale)}, action {act.name} frames {f0}-{f1}")

    frames = list(range(f0, f1 + 1))
    pos = np.zeros((len(frames), len(names), 3), dtype=np.float64)
    for i, f in enumerate(frames):
        scene.frame_set(f)
        for j, pb in enumerate(pbs):
            pos[i, j] = np.array((arm.matrix_world @ pb.matrix).translation, dtype=np.float64)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, positions=pos, names=np.array(names, dtype=object),
             parents=np.array(parents, dtype=object),
             frames=np.array(frames, dtype=np.int64),
             arm_matrix=np.array(arm.matrix_world, dtype=np.float64),
             source_fbx=fbx)
    print(f"[dump] wrote {out}  positions{pos.shape}")

    # The two sanity criteria, in the plugin's own convention.
    npz_pos = np.stack([-pos[..., 0], pos[..., 2], pos[..., 1]], axis=-1)
    idx = {n: k for k, n in enumerate(names)}
    if "Pelvis" in idx:
        print(f"[dump] criterion 1  Pelvis height mean = {npz_pos[:, idx['Pelvis'], 1].mean():.4f} m"
              "   (target ~0.892)")
    feet = [idx[n] for n in ("L_Ankle", "L_Foot", "R_Ankle", "R_Foot") if n in idx]
    if feet:
        print(f"[dump] criterion 2  lowest foot joint  = {npz_pos[:, feet, 1].min():+.4f} m"
              "   (target ~0)")
    print("[dump] 放到官方接触 npz 旁边并命名 <npz 名>_fbx_joints.npz 即被自动发现；"
          "或在向导「贴地区间」里用「fbx 绝对关节」直接选它。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
