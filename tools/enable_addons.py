# -*- coding: utf-8 -*-
"""一次性启用本套件带的全部插件，并把启用状态写进 userpref。

用法（在套件根目录）：
    source sandbox/env.sh
    blender -b --python tools/enable_addons.py

名字匹配做三级：精确 -> 后缀（bl_ext.user_default.xxx）-> 子串，
所以对扩展（bl_ext.*）和老式 addon 都生效。
"""
import sys

import addon_utils
import bpy

WANT = [
    "mocap_doctor",
    "mikumikurig",
    "mmd_tools",
    "ajc27_freemocap_blender_addon",
    "gvhmr_pose_capture",
]

mods = {}
for m in addon_utils.modules(refresh=True):
    mods.setdefault(m.__name__.lower(), m.__name__)

enabled, missing = [], []
for want in WANT:
    w = want.lower()
    hit = mods.get(w)
    if hit is None:
        hit = next((n for n in mods.values() if n.lower().endswith("." + w)), None)
    if hit is None:
        hit = next((n for n in mods.values() if w in n.lower()), None)
    if hit is None:
        missing.append(want)
        continue
    try:
        bpy.ops.preferences.addon_enable(module=hit)
        enabled.append(hit)
    except Exception as exc:  # noqa: BLE001
        print(f"[enable_addons] {hit} 启用失败: {exc!r}")
        missing.append(want)

try:
    bpy.ops.wm.save_userpref()
except Exception as exc:  # noqa: BLE001
    print(f"[enable_addons] save_userpref 失败: {exc!r}（GUI 里手动勾选即可）")

print(f"[enable_addons] enabled = {enabled}")
if missing:
    print(f"[enable_addons] MISSING = {missing}")
    sys.exit(1)
print("[enable_addons] all ok")
