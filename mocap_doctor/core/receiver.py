"""Receiver-template prep planning.

The plugin exports a VMD that plays correctly in MMD, but re-importing it into
a Blender template that went through MikuMikuRig (MMR) breaks the arm for three
independent reasons:

1. MMR slaves every bone of the native MMD armature to the control rig with
   COPY_TRANSFORMS / COPY_ROTATION / TRANSFORM constraints, so the imported VMD
   FK curves are overridden and the arm stays in the template's rest pose.
2. MMR adds 手IK.L/R helper bones that duplicate the native 手首.L/R
   mmd_bone.name_j (左手首/右手首).  mmd_tools' PMX bone map is built with a
   dict comprehension in which the later bone wins, so the VMD wrist tracks are
   assigned to the helpers and the real wrists never move.
3. MMR re-parents ひじ.L/R from 腕捩.L/R to 腕.L/R, so the elbow FK rotation is
   applied in the wrong parent frame (off by the upper-arm twist).

The functions here only *decide* what to change and return plain data, so they
can be unit-tested without Blender; the operator in operators.py applies the
plan to the live scene.
"""

# PMX-native IK targets use full-width ＩＫ (足ＩＫ, つま先ＩＫ, 手首ＩＫ); MMR
# helpers use half-width IK (手IK).  Native leg IK must survive so that VMDs
# with IK property keys still animate the legs via mmd_ik_toggle.
FULL_WIDTH_IK = "ＩＫ"
HALF_WIDTH_IK = "IK"

ARM_IK_TOGGLE_BONES = ("ひじ.L", "ひじ.R", "手首.L", "手首.R")
# (bone, MMR parent, PMX parent) per side
ELBOW_PARENT_RESTORE = {
    "L": ("ひじ.L", "腕.L", "腕捩.L"),
    "R": ("ひじ.R", "腕.R", "腕捩.R"),
}


# mmd_tools rebuilds MMD's 付与 (additional transform) rig with its own
# constraints: mmd_additional_rotation drives 足D / ひざD / 足首D off the
# _shadow_* bones and mmd_tools_at_dummy feeds those from the _dummy_* ones.
# A PMX leg is skinned to the D bones, NOT to 足 / ひざ / 足首, so those
# constraints are what actually carries the animation into the mesh.  Deleting
# them leaves the bones moving with the leg mesh frozen at rest.
MMD_TOOLS_CONSTRAINT_PREFIX = "mmd_"


def _keeps_native_ik(constraint):
    if constraint.get("type") != "IK":
        return False
    return FULL_WIDTH_IK in constraint.get("subtarget", "")


def _keeps_mmd_tools_rig(constraint):
    return str(constraint.get("name", "")).startswith(MMD_TOOLS_CONSTRAINT_PREFIX)


def plan_constraints(bones):
    """Split each bone's constraints into keep vs remove.

    ``bones`` is an iterable of dicts with ``name`` and ``constraints``
    (a list of dicts with ``name``, ``type`` and ``subtarget``).

    Keep mmd_tools' own rig and the native full-width leg IK.  Everything else
    is what MMR added to slave the armature to its control rig, and that is what
    has to go before an imported VMD can drive the model.
    """
    keep, remove = [], []
    for bone in bones:
        for constraint in bone.get("constraints", ()):
            item = {
                "bone": bone["name"],
                "name": constraint.get("name", ""),
                "type": constraint.get("type", ""),
                "subtarget": constraint.get("subtarget", ""),
            }
            keeps = _keeps_native_ik(constraint) or _keeps_mmd_tools_rig(constraint)
            (keep if keeps else remove).append(item)
    return {"keep": keep, "remove": remove}


def plan_arm_ik_toggle_off(bones):
    """Bone names among ARM_IK_TOGGLE_BONES that currently have IK enabled.

    ``bones`` maps name -> dict with ``has_ik_toggle`` and ``ik_toggle``.
    """
    return [
        name
        for name in ARM_IK_TOGGLE_BONES
        if name in bones
        and bones[name].get("has_ik_toggle")
        and bones[name].get("ik_toggle")
    ]


def plan_helper_bone_deletion(bones):
    """Names of MMR helper bones to delete.

    A bone is a helper when it shares its mmd name_j with another bone and its
    own name contains the half-width "IK" marker (手IK.L/R) — the helper wins
    the mmd_tools PMX bone map, so it must go.  Bones that have children are
    left alone and reported instead.
    """
    delete, skipped = [], []
    by_name_j = {}
    for bone in bones:
        name_j = bone.get("name_j") or ""
        if name_j:
            by_name_j.setdefault(name_j, []).append(bone)
    for name_j, group in by_name_j.items():
        if len(group) < 2:
            continue
        helpers = [bone for bone in group if HALF_WIDTH_IK in bone.get("name", "")]
        others = [bone for bone in group if bone not in helpers]
        if len(helpers) != 1 or not others:
            continue
        for bone in helpers:
            if bone.get("has_children"):
                skipped.append(bone["name"])
            else:
                delete.append(bone["name"])
    return {"delete": delete, "skipped": skipped}


def plan_elbow_reparent(bones):
    """(bone, new_parent) pairs where ひじ currently sits under the MMR
    parent (腕) while the PMX parent (腕捩) exists.

    ``bones`` maps name -> dict with ``parent``.
    """
    reparent = []
    for bone_name, mmr_parent, pmx_parent in ELBOW_PARENT_RESTORE.values():
        bone = bones.get(bone_name)
        if bone is None or pmx_parent not in bones:
            continue
        if bone.get("parent") == mmr_parent:
            reparent.append((bone_name, pmx_parent))
    return reparent


def is_rig_text(name):
    """True for an MMR/rigify UI script text block (``RIG-*`` / ``*arm_ui*``)."""

    text = str(name or "")
    return text.startswith("RIG-") or "arm_ui" in text


def plan_rig_deletion(objects, texts):
    """Control-rig armature object names and UI text names to delete.

    ``objects`` is an iterable of (name, type) pairs for ARMATURE objects;
    the native MMD armature is passed separately so it is never matched.
    ``texts`` is an iterable of text names.
    """
    rig_objects = [
        name
        for name, kind in objects
        if kind == "ARMATURE" and (name.startswith("RIG-") or "MMR" in name)
    ]
    rig_texts = [name for name in texts if is_rig_text(name)]
    return {"rig_objects": rig_objects, "rig_texts": rig_texts}


# ---------------------------------------------------------------------------
# MMR UI 复活的全局 load_post（Q8-2）
# ---------------------------------------------------------------------------
# 打开文件时 Blender 会跳过带 RIG-* / *arm_ui* 文本块里的自动脚本
# （控制台原话："scripts disabled for …skipping 'RIG-MMR-…_arm_ui.py'"），而 MMR
# 那个悬浮面板正是那段脚本 register() 出来的——于是面板就没了。这里在**每次打开
# 任何文件**之后把它们重新执行一遍；project.restore_checkpoint 的
# open_mainfile(..., load_ui=False) 也走 load_post，所以那条路径同样覆盖。
#
# 同一个 (文件, 文本块) 只执行一次：再执行会在 register_class 上撞
# "already registered"，而面板本来就还在。命名空间留着，注册的类才不会被回收。

_REVIVED_MMR_TEXTS = {}
_LAST_MMR_REVIVE = None
_revive_mmr_ui_on_load = None


def revive_mmr_ui_texts(force=False):
    """Re-execute the MMR UI text blocks of the current file.

    Returns ``{"file", "ran", "already", "skipped", "failed"}``; ``failed`` maps
    the text name to the error, so a silent no-op is impossible.
    """

    import bpy

    result = {
        "file": bpy.data.filepath or "",
        "ran": [],
        "already": [],
        "skipped": [],
        "failed": {},
    }
    for text in list(getattr(bpy.data, "texts", ()) or ()):
        name = str(getattr(text, "name", ""))
        if not is_rig_text(name):
            continue
        key = (str(bpy.data.filepath or ""), name)
        if not force and key in _REVIVED_MMR_TEXTS:
            result["skipped"].append(name)
            continue
        try:
            source = text.as_string()
        except Exception as exc:  # noqa: BLE001 - 读不了就记下来，别抛
            result["failed"][name] = f"read: {exc}"
            continue
        if not source.strip():
            result["skipped"].append(name)
            continue
        namespace = {
            "__name__": f"mcd_mmr_ui_{len(_REVIVED_MMR_TEXTS)}",
            "__file__": f"<text:{name}>",
        }
        try:
            exec(compile(source, f"<text:{name}>", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - 面板脚本再跑一次本来就会撞注册
            message = str(exc)
            if "already registered" in message:
                _REVIVED_MMR_TEXTS[key] = namespace
                result["already"].append(name)
                continue
            result["failed"][name] = f"{type(exc).__name__}: {message}"
            continue
        _REVIVED_MMR_TEXTS[key] = namespace
        result["ran"].append(name)
    return result


def _mmr_revive_handler(_dummy):
    global _LAST_MMR_REVIVE
    try:
        _LAST_MMR_REVIVE = revive_mmr_ui_texts()
        if _LAST_MMR_REVIVE["ran"] or _LAST_MMR_REVIVE["failed"]:
            print(
                "[MoCap Doctor] MMR UI 文本块：执行 "
                f"{_LAST_MMR_REVIVE['ran']}，失败 {_LAST_MMR_REVIVE['failed']}"
            )
    except Exception as exc:  # noqa: BLE001 - 打开文件绝不能被它带崩
        print(f"[MoCap Doctor] MMR UI revival failed: {exc}")


def register_handlers():
    global _revive_mmr_ui_on_load
    import bpy
    from bpy.app.handlers import persistent

    if _revive_mmr_ui_on_load is None:
        _revive_mmr_ui_on_load = persistent(_mmr_revive_handler)
    if _revive_mmr_ui_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_revive_mmr_ui_on_load)


def unregister_handlers():
    global _revive_mmr_ui_on_load
    import bpy

    if _revive_mmr_ui_on_load is not None and _revive_mmr_ui_on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_revive_mmr_ui_on_load)


def last_revive_report():
    """The most recent ``revive_mmr_ui_texts`` result (tests / diagnostics)."""

    return _LAST_MMR_REVIVE


def plan_stale_actions(actions):
    """Action names to delete: previous VMD imports leave pose.bone fcurves
    that mmd_tools would merge the next import into."""
    return [
        action["name"]
        for action in actions
        if action.get("has_pose_bone_fcurves")
    ]
