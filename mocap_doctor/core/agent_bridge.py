"""In-process tool server for LLM co-editing inside the GUI Blender session.

Architecture (per the plan):
- a background thread runs a tiny JSON-lines socket server on 127.0.0.1
- bpy may ONLY be touched on the main thread: requests go into a queue, and a
  ``bpy.app.timers`` pump drains it every TIMER_INTERVAL seconds
- preview writes land on the dedicated ``AGENT_PREVIEW`` Combine NLA track,
  viewport is redrawn immediately, playback range jumps to the op's frames
- a data version counter bumps on every depsgraph change, so a tool call can
  carry ``expect_version`` and get "scene changed" instead of clobbering the
  user's fresh hand edits
- commit does ``undo_push`` so Ctrl+Z covers agent work too

Client protocol: one JSON object per line, both directions.
    {"id": 1, "tool": "describe", "args": {"target": "contact.L:0"}}
    {"id": 1, "ok": true, "version": 12, "result": {...}}
"""

from __future__ import annotations

import json
import os
import queue
import re
import socket
import socketserver
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

import bpy
import numpy as np

from . import (agent_anatomy, agent_bake, agent_claims, agent_fx, agent_io,
             agent_ops, agent_query)

HOST = "127.0.0.1"
PORT = int(os.environ.get("MCD_AGENT_PORT", "6207"))
TIMER_INTERVAL = 0.07
BUSY_INTERVAL = 0.005     # GUI timer: re-poll quickly right after serving
_PUMP_BUSY = False
WATCHDOG_INTERVAL = 2.0   # how often we check the pump is still ticking
PREVIEW_TRACK = agent_ops.PREVIEW_TRACK   # legacy shared preview track
COMMIT_TRACK = agent_ops.AGENT_TRACK_PREFIX


def agent_track_prefixes():
    """Names that mark a track as ours.  Per-op tracks are named after their
    strip ("agent_hold_505_570"); the legacy shared names stay recognised so an
    old .blend keeps working.  mcd_base is excluded - it is the baseline."""
    return agent_ops.AGENT_TRACK_PREFIXES


_running = False
_requests: "queue.Queue" = queue.Queue()
_DATA_VERSION = 0
_LAST_BUMP = 0.0
_STORE = None           # cached DataStore
_STORE_KEY = None
_LAST_TOOL = ""
_LAST_PUMP = 0.0          # wall-clock of last successful _pump tick
_STATUS = {"clients": 0, "last_tool": "", "last_error": ""}

# ---- 并发（任务1）：租约 + 写入日志 + 工具执行期不让 depsgraph 抖版本 ----
_LEASES = agent_claims.LeaseTable()
_JOURNAL = agent_claims.WriteJournal()
_IN_TOOL = False            # True while a tool runs: its own frame_set must not bump
_IN_SYNC = False            # True while the fix-list timer mirrors the scene (same rule)
_LAST_FRAME_SEEN = None     # frame-only depsgraph updates (scrub/playback) ≠ edits
_ANC_CACHE: dict = {}       # armature name → {bone: ancestors(actual ∪ semantic)}
# 任务2：信号库的输入是 npz 缓存 + 标注区间 + 设置，agent 写 delta 动不了其中
# 任何一样——所以只有外部改动（GUI 里用户编辑）才让它失效，agent 写入不再
# 触发 220ms 的重建（重建结果本来就逐位相同）。
_STORE_EPOCH = 0
_WAKE = threading.Event()   # socket 线程收到请求就叫醒 headless 主循环


# ---------------------------------------------------------------------------
# context / store

def _settings(scene=None):
    scene = scene or bpy.context.scene
    return scene, scene.mocap_doctor


def _source_armature(settings):
    arm = getattr(settings, "source_armature", None)
    if arm is not None and getattr(arm, "type", "") == "ARMATURE":
        return arm
    return None


def _rig_armature(settings, scene=None):
    """The MMR control rig is what the agent layer operates on: Teto 自身没有
    action（全靠约束跟 RIG），源骨架 f_avg 与 Teto 间是烘焙链路——修源骨架
    下游看不到，修 RIG 立刻反映到最终模型。"""
    rig = getattr(settings, "mmr_rig", None)
    if rig is not None and getattr(rig, "type", "") == "ARMATURE":
        return rig
    scene = scene or bpy.context.scene
    for ob in scene.objects:
        if ob.type == "ARMATURE" and ob.name.startswith("RIG-"):
            return ob
    return None


def _resolve_bones(armature, names):
    """bones 参数既可以是字面骨名也可以是角色名（finger_l_index1 等）。"""
    if armature is None:
        raise RuntimeError("没有识别到 RIG 骨架")
    spec = agent_io.rig_bake_spec(armature)
    role_map = spec["bones"]
    out = []
    unknown = []
    for n in names:
        if n in role_map:
            out.append(role_map[n])
        elif n in armature.data.bones:
            out.append(n)
        else:
            unknown.append(n)
    if unknown:
        raise agent_query.AgentQueryError(
            f"RIG 骨架上没有这些骨：{unknown}",
            code="E_UNKNOWN",
            fix=f"可用角色名：{sorted(role_map)}")
    return out


def _data_dir(settings):
    """op 日志目录解析：blend 文件挪机器后 settings 里的绝对路径会失效，
    resolve_data_dir 按当前文件位置重推并回写。延迟 import 避开包循环。"""
    try:
        from .. import project
        return project.resolve_data_dir(settings)
    except Exception:
        return settings.data_directory or "."


def get_store(force: bool = False):
    """Build (or reuse) the query store over the current RIG armature."""
    global _STORE, _STORE_KEY
    scene, settings = _settings()
    armature = _rig_armature(settings, scene)
    if armature is None:
        raise RuntimeError("没有识别到 RIG 骨架（settings.mmr_rig 为空，"
                           "且场景里没有 RIG-* 骨架）")
    # 老文件基底 1 帧修正要先于任何快照烘焙：否则快照按滞后的基底烘（或者在第一次
    # 写入之后才重烘、把那次写入也烘进去）。服务里启动时已做过，这里是 no-op。
    _settle_base_strip()
    key = (armature.name, scene.frame_start, scene.frame_end, _STORE_EPOCH,
           str(_data_dir(settings)))
    if force or _STORE is None or _STORE_KEY != key:
        _STORE = agent_io.build_store_for_scene(
            scene, armature, settings,
            spec=agent_io.rig_bake_spec(armature),
            data_dir=_data_dir(settings),
            use_cache=True, tag="rig",
        )
        _STORE_KEY = key
    return _STORE


def data_version() -> int:
    return _DATA_VERSION


def _bump_version(scene=None, depsgraph=None):
    """depsgraph_update_post fires on EVERY depsgraph event - frame scrub,
    influence drag, cursor blink.  Throttle so the version still means
    'something changed' instead of 'wall clock moved'.

    Not counted (任务1): updates caused by a tool's own frame_set while it
    runs (reads must not invalidate other agents' versions - writes bump
    explicitly via _note_write), and pure frame changes (scrub / playback)
    that carry no Action edit.  What is left is an EXTERNAL change (the user
    edited something) → journal row with unscoped bones = stale for all."""
    global _DATA_VERSION, _LAST_BUMP, _LAST_FRAME_SEEN
    if _IN_TOOL or _IN_SYNC:
        return
    try:
        frame = int(scene.frame_current) if scene is not None else None
    except Exception:
        frame = None
    frame_moved = frame is not None and frame != _LAST_FRAME_SEEN
    _LAST_FRAME_SEEN = frame
    if frame_moved and not _has_action_update(depsgraph):
        return
    now = time.monotonic()
    if now - _LAST_BUMP < 0.3:
        return
    global _STORE_EPOCH
    _LAST_BUMP = now
    _DATA_VERSION += 1
    _STORE_EPOCH += 1
    _JOURNAL.note(_DATA_VERSION, None, agent_claims.ALL, None, "external")


def _has_action_update(depsgraph) -> bool:
    try:
        return any(isinstance(u.id, bpy.types.Action)
                   for u in depsgraph.updates)
    except Exception:
        return False


def _redraw():
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    except Exception:
        pass


def _playback_running() -> bool:
    try:
        return any(getattr(w.screen, "is_animation_playing", False)
                   for w in bpy.context.window_manager.windows)
    except Exception:
        return False


def _toggle_playback():
    with bpy.context.temp_override():
        bpy.ops.screen.animation_play()


def _focus_preview(scene, frame_range):
    try:
        agent_ops.set_preview(scene, frame_range)
        scene.frame_set(int(frame_range[0]))
    except Exception:
        pass
    # animation_play() is a TOGGLE: with several agents writing, playback went
    # on / off / on.  Only start it when it is not running already.
    if _playback_running():
        return
    try:
        _toggle_playback()
    except Exception:
        pass  # playback start is a nicety, never a failure


def _base_action(armature):
    """The baseline action - active action, or the mcd_base strip's action once
    it has been pushed onto NLA (active action sits ABOVE tracks and REPLACEd
    our deltas, so the first write pushes it down).  Same rule as
    agent_ops.base_action_of (a foreign active Action over mcd_base raises)."""
    return agent_ops.base_action_of(armature)


def _track_by_name(armature, name):
    anim = getattr(armature, "animation_data", None)
    if anim is None:
        return None
    for track in anim.nla_tracks:
        if track.name.startswith(name):
            return track
    return None


# ---------------------------------------------------------------------------
# tool registry

def _check_version(args, scope=None, agent_id=None, ctx=None, notes=None):
    """expect_version 校验。读工具（scope=None）保持严格：版本不等即过期。
    写工具按 scope 判定：期间只有"别人改了与我范围相交的骨（含祖先骨）/帧"
    或外部改动才算过期——别的 agent 在别处写不会误伤你。"""
    expect = args.pop("expect_version", None)
    if expect is None or int(expect) == _DATA_VERSION:
        return
    if scope is None:
        raise agent_query.AgentQueryError(
            f"场景已变（期望 v{expect}，当前 v{_DATA_VERSION}）",
            code="E_STALE", fix="重新调用读工具拿新数据")
    anc = _ancestor_fn(ctx["armature"]) if ctx else (lambda b: set())
    why = _JOURNAL.stale_against(int(expect), _DATA_VERSION, agent_id,
                                 scope, anc)
    if why is not None:
        raise agent_query.AgentQueryError(
            f"场景已变（期望 v{expect}，当前 v{_DATA_VERSION}）：{why['why']}"
            + (f" {why.get('changes')}" if why.get("changes") else ""),
            code="E_STALE",
            fix="重读你的骨/帧段（probe/describe/sample）拿新版本号后再写")
    if notes is not None:
        notes.append(f"版本 v{expect}→v{_DATA_VERSION}：期间改动与你的范围无关，已放行")


# ---------------------------------------------------------------------------
# concurrency scope helpers (任务1)

_OP_TOOLS = ("reapply", "revert", "set_influence")
_LEGACY_DRY_RUN = ("hold_pose", "clean_jitter", "restore_accent")


def _dry_run_tools() -> set:
    """Write tools that really honour dry_run: the three legacy tools that
    implement it + every plugin write tool (plugin contract requires it)."""
    return set(_LEGACY_DRY_RUN) | set(WRITE_SCOPES)


def _ancestor_fn(armature):
    """bone → ancestors (real bone tree ∪ Rigify semantic parents), cached."""
    if armature is None:
        return lambda b: set()
    cache = _ANC_CACHE.setdefault(armature.name, {})

    def fn(bone):
        hit = cache.get(bone)
        if hit is not None:
            return hit
        out = set()
        db = armature.data.bones.get(bone)
        p = db.parent if db is not None else None
        while p is not None:
            out.add(p.name)
            p = p.parent
        if db is not None:
            try:
                from . import agent_pose
                out |= set(agent_pose.semantic_ancestors(armature, bone))
            except Exception:
                pass
        cache[bone] = out
        return out
    return fn


def _bone_of_path(path) -> str | None:
    path = str(path or "")
    if path.startswith('pose.bones["'):
        return path.split('"')[1]
    return None


def _frames_of(args, key="frame_range"):
    fr = args.get(key)
    if fr is None:
        return None
    return (int(fr[0]), int(fr[1]))


def _legacy_scope(name, ctx, args):
    arm = ctx["armature"]
    frames = _frames_of(args)
    bones: set = set()
    if name == "hold_pose":
        bones = set(_resolve_bones(arm, list(args.get("bones") or [])))
    elif name == "clean_jitter":
        if args.get("bone"):
            bones = set(_resolve_bones(arm, [args["bone"]]))
        for p in args.get("paths") or []:
            b = _bone_of_path(p[0] if isinstance(p, (list, tuple)) else p)
            if b:
                bones.add(b)
    elif name == "fix_ground":
        bones = {_bone_of_path(args.get("loc_path"))}
    elif name == "restore_accent":
        bones = {_bone_of_path(args.get("data_path"))}
    elif name == "solve_pelvis":
        bones = {_bone_of_path(args.get("pelvis_path"))}
    elif name == "apply_exemplar":
        bones = {_bone_of_path(args.get("loc_path")),
                 _bone_of_path(args.get("quat_path"))}
    bones.discard(None)
    return [(bones or agent_claims.ALL, frames)]


_LEGACY_WRITES = ("hold_pose", "clean_jitter", "fix_ground", "restore_accent",
                  "solve_pelvis", "apply_exemplar")


def _strip_bones(armature, op) -> set:
    """Bones an op's strip really writes (its delta action's fcurves)."""
    try:
        _tr, strip = agent_ops.find_op_strip(armature, op)
        act = strip.action if strip is not None else None
        return {b for b in (_bone_of_path(fc.data_path)
                            for fc in (act.fcurves if act else ())) if b}
    except Exception:
        return set()


def _op_scope(ctx, op, overrides=None) -> list:
    """Scope of an op-addressed write (reapply / revert / set_influence /
    ab_toggle): the bones the op REALLY writes × its old window (+ the new
    window when a reapply moves it).

    Recorded params are not call args: motion_copy records `bones` = its
    TARGET bones, while at call time `bones` means sources - feeding recorded
    params back through WRITE_SCOPES mirrored arm.R back to arm.L and
    auto-claimed the untouched source arm.  So plugin ops use the strip's
    fcurve bones + recorded `bones`, never the call-time scope function."""
    arm = ctx["armature"]
    overrides = dict(overrides or {})
    params = dict(op.get("params") or {})
    params.update(overrides)
    old = tuple(int(x) for x in op["frames"]) if op.get("frames") else None
    new = None
    if overrides.get("frame_range"):
        fr = overrides["frame_range"]
        new = (int(fr[0]), int(fr[1]))
    elif overrides.get("dst_start") is not None and old is not None:
        s0 = int(overrides["dst_start"])
        new = (s0, s0 + (old[1] - old[0]))
    bones = _strip_bones(arm, op)
    tool = op.get("tool")
    if tool in _LEGACY_WRITES:
        p2 = dict(params)
        if tool == "restore_accent":
            p2["data_path"] = params.get("path")
        try:
            for bs, _fr in _legacy_scope(tool, ctx, p2):
                if bs is not agent_claims.ALL:
                    bones |= set(bs)
        except Exception:
            pass
    else:
        pb = params.get("bones")
        if isinstance(pb, (list, tuple)):
            bones |= {b for b in pb if isinstance(b, str)
                      and arm.pose.bones.get(b) is not None}
    windows = [w for w in (old, new) if w is not None] or [None]
    return [(bones or agent_claims.ALL, w) for w in dict.fromkeys(windows)]


def _write_scope(name, ctx, args, agent_id, force):
    """None = read-only call.  Otherwise [(bones|ALL, (a,b)|None), ...].
    Raises E_OWNER for op-addressed writes on someone else's op."""
    if name in _OP_TOOLS:
        op_id = args.get("op_id")
        if not op_id:                      # set_influence(track_name=...) 旧布局
            return [(agent_claims.ALL, None)]
        op = _op_by_id(ctx, op_id)
        owner = op.get("owner")
        if agent_id and not force and owner != agent_id:
            who = owner or "用户/旧版（无 owner）"
            raise agent_query.AgentQueryError(
                f"{op_id} 属于 {who}，不是你（{agent_id}）",
                code="E_OWNER",
                fix="只改你自己写的 op（list_ops 看 owner）；别人的修复找协调者")
        return _op_scope(ctx, op, args.get("overrides"))
    if name == "ab_toggle":
        if not agent_id:
            return [(agent_claims.ALL, None)]
        mine = [o for o in agent_ops.list_ops(ctx["data_dir"])
                if o.get("owner") == agent_id
                and o.get("status") in ("preview", "committed")]
        return [sc for o in mine for sc in _op_scope(ctx, o)] or []
    if name in _LEGACY_WRITES:
        return _legacy_scope(name, ctx, args)
    if name in WRITE_SCOPES:
        return [(set(b) if b is not agent_claims.ALL else b,
                 None if f is None else (int(f[0]), int(f[1])))
                for b, f in WRITE_SCOPES[name](ctx, args)]
    return None


def _enforce_claims(name, ctx, scope, agent_id, force, notes):
    anc = _ancestor_fn(ctx["armature"])
    who = agent_id or "__anonymous__"
    for bones, frames in scope:
        hard, soft = _LEASES.conflicts(who, bones, frames, anc)
        if hard and not force:
            h = hard[0]
            raise agent_query.AgentQueryError(
                f"scope 被占用：{h['agent_id']} 持有 {h['bones']} @ {h['frames']}"
                f"（{h['claim_id']}，还剩 {h['ttl_left_s']}s）",
                code="E_CLAIMED",
                fix="换骨骼或帧段（list_claims 看谁占了哪里），或等对方 release；"
                    "不要 force")
        for r in soft[:2]:
            notes.append(f"层级相关：{r['agent_id']} 占着 {r['pairs']} @ {r['frames']}"
                         "——父骨改动会带动子骨世界朝向，写完提醒对方复测")
    if agent_id:
        for bones, frames in scope:
            if bones != agent_claims.ALL and not bones:
                continue
            if not _LEASES.covering(agent_id, bones, frames):
                r = _LEASES.claim(agent_id, bones, frames, ancestors=anc,
                                  auto=True, note=f"auto:{name}")
                if r.get("claim_id"):
                    notes.append(f"已自动认领 {r['claim_id']}（{name} 的写入范围）"
                                 "——建议写前先 claim")


def _stack_warnings(name, ctx, scope, agent_id) -> list:
    """Same tool already alive on the same bone × overlapping frames → warn.
    Stacking two identical fixes over-corrects (verification: 3 stacked
    clean_jitter strips pushed jitter to 152% of the untouched baseline)."""
    out = []
    try:
        ops = [o for o in agent_ops.list_ops(ctx["data_dir"])
               if o.get("tool") == name and o.get("status") in ("preview", "committed")]
        if not ops:
            return out
        arm = ctx["armature"]
        for o in ops:
            ob = _strip_bones(arm, o)
            if not ob:
                continue
            for bones, frames in scope:
                if frames is not None and o.get("frames") and not \
                        agent_claims.frames_overlap(frames, o["frames"]):
                    continue
                shared = ob if bones is agent_claims.ALL else (ob & set(bones))
                if shared:
                    who = o.get("owner") or "无主/旧版"
                    mine = agent_id and o.get("owner") == agent_id
                    out.append(f"同骨同帧已有同类修复 {o['id']}（owner={who}，骨 "
                               f"{sorted(shared)[:3]}，帧 {o.get('frames')}）——叠加会过度修正；"
                               + ("改参数请 reapply 那条" if mine else "不该叠就 revert 你这条并报告"))
                    break
    except Exception:
        pass
    return out[:3]


def _track_read_dependencies(ctx, scope, written_id, agent_id, name):
    """复制类 op 读过的源（params.src_bones × src_range）被这次写入改了 → 在 op 日志里给它标
    `stale`（list_ops 行里看得到）；被 reapply 的 op 重新按当前源算过 → 清掉标记。

    租约只保护"写"：别人在你复制之后改了你的源，副本不会跟着变，以前没人知道（§10-12）。
    local 空间只看同骨；world 空间连祖先骨也算。只是标记，不自动重算。"""
    data_dir = ctx.get("data_dir")
    if not data_dir or not scope:
        return
    try:
        ops = agent_ops._load_oplog(data_dir)
        anc = _ancestor_fn(ctx["armature"])
    except Exception:  # noqa: BLE001 - 记账失败绝不影响写入本身
        return
    changed = False
    for o in ops:
        if o.get("tool") != "motion_copy":
            continue
        if o.get("id") == written_id:
            if name == "reapply" and o.pop("stale", None) is not None:
                changed = True
            continue
        if o.get("status") != "preview":
            continue
        p = o.get("params") or {}
        src, sr = set(p.get("src_bones") or ()), p.get("src_range")
        if not src or not sr:
            continue
        sr = (int(sr[0]), int(sr[1]))
        world = str(p.get("space", "local")) == "world"
        for bones, frames in scope:
            if frames is None or not agent_claims.frames_overlap(sr, frames):
                continue
            if bones is agent_claims.ALL or src & set(bones) or (
                    world and agent_claims._bones_related(set(bones), src, anc)):
                o["stale"] = {"since_version": _DATA_VERSION, "by": agent_id or "?",
                              "tool": name, "op": written_id,
                              "frames": list(agent_claims.frames_intersection(sr, frames)),
                              "fix": "源被改过：reapply {op_id, overrides:{}} 按当前源重新复制"}
                changed = True
                break
    if changed:
        agent_ops._save_oplog(data_dir, ops)


def _note_write(agent_id, scope, name):
    global _DATA_VERSION
    _DATA_VERSION += 1
    for bones, frames in scope or [(agent_claims.ALL, None)]:
        _JOURNAL.note(_DATA_VERSION, agent_id, bones, frames, name)
    if agent_id:
        _LEASES.renew(agent_id)


def _op_envelope(op: dict, tool: str) -> dict:
    """Write-tool result → unified envelope fields."""
    frames = op.get("frames", [0, 0])
    return {
        "summary": f"{tool} {op.get('status','preview')}："
                   f"{frames[0]}-{frames[1]} 帧",
        "data": {
            "op_id": op.get("id"),
            "status": op.get("status"),
            "params_echo": op.get("params"),
            "frames": frames,
            "touched": op.get("params", {}).get("bones")
                       or op.get("params", {}).get("paths")
                       or op.get("params", {}).get("loc_path"),
            "metrics": op.get("metrics"),
            "preview": {"track": op.get("strip"),
                        "playback_range": frames},
        },
        "warnings": list(op.get("_warnings") or []),   # 插件写工具的附加提示（不进 op 日志）
        "truncated": False,
        "hint": "看视口 A/B 后 commit(op_id) 或 revert(op_id)",
    }


def _tool_ping(_ctx, **_):
    data = {"ok": True, "version": _DATA_VERSION, "running": _running,
            "tools": sorted(TOOLS)}
    warnings = []
    if _PLUGIN_ERRORS:
        data["plugin_errors"] = dict(_PLUGIN_ERRORS)
        warnings.append(f"插件载入失败：{sorted(_PLUGIN_ERRORS)}")
    return {"summary": f"服务运行中 · 数据 v{_DATA_VERSION}",
            "data": data, "warnings": warnings, "truncated": False, "hint": ""}


def _tool_overview(_ctx, force_refresh=False, **_):
    store = get_store(force=bool(force_refresh))
    env = agent_query.get_overview(store)
    # expose the bake spec so "missing finger signals" is one glance away
    env["data"]["bones"] = dict(getattr(store, "spec", {}).get("bones", {}))
    env["data"]["bake_missing"] = list(getattr(store, "bake_missing", []))
    env["data"]["data_dir"] = str(_ctx.get("data_dir") or "")
    return env


def _tool_list_intervals(_ctx, kind=None, frame_range=None, tag=None, **_):
    return agent_query.list_intervals(
        get_store(), kind=kind, frame_range=frame_range, tag=tag)


def _tool_list_timeline_markers(_ctx, frame_range=None, name=None,
                                with_intervals=True, **_):
    """The user's M-key timeline markers, joined to the covering intervals.

    Reads the live scene (markers are scene state, not action data), so it
    works before any bake and stays correct while the user keeps adding them.
    """
    scene = _ctx.get("scene")
    markers = [
        {"name": str(getattr(marker, "name", "") or ""),
         "frame": int(getattr(marker, "frame", 0))}
        for marker in (getattr(scene, "timeline_markers", None) or ())
    ]
    intervals = None
    if with_intervals and scene is not None:
        intervals = agent_io.scene_intervals(scene)
    return agent_query.list_timeline_markers(
        markers, intervals, frame_range=frame_range, name=name,
        with_intervals=bool(with_intervals))


def _tool_describe(_ctx, target=None, channels=None, context=15,
                   frame_range=None, **_):
    if target is None and frame_range is not None:
        target = list(frame_range)
    return agent_query.describe(get_store(), target,
                                channels=channels, context=int(context))


def _tool_get_series(_ctx, channels, frame_range, max_points=60,
                     agg="mean", **_):
    return agent_query.get_series(get_store(), channels, frame_range,
                                  max_points=max_points, agg=agg)


def _tool_find_events(_ctx, cond, frame_range=None, **_):
    return agent_query.find_events(get_store(), cond, frame_range)


def _tool_compare(_ctx, channel, a, b, **_):
    return agent_query.compare(get_store(), channel, a, b)


def _tool_snapshot(_ctx, frame, roles=None, **_):
    res = agent_query.snapshot(get_store(), frame, roles)
    warn = _leg_role_warning(_ctx, roles if roles else _LEG_ROLES)
    if warn and isinstance(res, dict):
        res = dict(res)
        res["warnings"] = list(res.get("warnings") or []) + warn[:1]
    return res


def _tool_eval(_ctx, expr, **_):
    """Evaluate a Python expression in the live scene; returns repr() ≤ 4000 chars.

    Debugging/inspection escape hatch - single expressions only, result must be
    repr-able. This socket is localhost-only by design.
    """
    # env must be globals so list-comprehension scopes can see bpy/scene/pb
    env = {
        "__builtins__": None,   # replaced below
        "bpy": bpy,
        "scene": _ctx["scene"],
        "armature": _ctx["armature"],      # RIG 控制架
        "source": _ctx["source_armature"], # f_avg 源骨架
        "data": bpy.data,
    }
    try:
        armature = _ctx["armature"]
        if armature is not None:
            env["pb"] = armature.pose.bones
    except Exception:
        pass
    import builtins as _bi
    safe_builtins = {k: getattr(_bi, k) for k in (
        "len", "list", "dict", "tuple", "set", "sorted", "str", "repr",
        "round", "float", "int", "bool", "min", "max", "sum", "any",
        "all", "enumerate", "zip", "range", "getattr", "hasattr", "isinstance",
        "type", "abs", "iter",
    ) if hasattr(_bi, k)}
    env["__builtins__"] = safe_builtins
    result = eval(str(expr), env, env)  # noqa: S307
    text = repr(result)
    truncated = len(text) > 4000
    return {"summary": text[:120],
            "data": {"expr": str(expr), "result": text[:4000]},
            "warnings": [], "truncated": truncated,
            "hint": "结果被截断到 4000 字符" if truncated else ""}


_LEG_ROLES = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")


def _leg_role_warning(ctx, names) -> list:
    """快照里的腿部角色（left_hip/left_knee/left_ankle → thigh_fk/shin_fk）是 FK 骨：腿是 IK 时
    它们看不见——膝位置与视口里的膝差中位数 40–46 mm、最大 236 mm（2026-10-04 §16）。"""
    arm = ctx.get("armature")
    if arm is None:
        return []
    out = []
    for n in names or ():
        s = "L" if str(n).startswith("left") or str(n).endswith(".L") else (
            "R" if str(n).startswith("right") or str(n).endswith(".R") else None)
        if s is None:
            continue
        if (n in _LEG_ROLES or str(n).startswith(("thigh_fk", "shin_fk"))) and \
                agent_anatomy.limb_is_ik(arm, "knee", s):
            out.append(f"{n}：{s} 腿是 IK，这个快照角色映射到 FK 骨（看不见；膝位置与视口里的膝差中位数 4 cm、最大 24 cm）。"
                       "看膝的朝向/位置用实时工具 probe_anatomy part=knee_front 或 orient_report（读形变骨 = 视口里的膝）")
    return out


def _tool_get_joint_angles(_ctx, bones, frame_range, max_points=60, **_):
    res = agent_query.get_joint_angles(
        get_store(), bones, frame_range, max_points=max_points)
    warn = _leg_role_warning(_ctx, bones)
    if warn and isinstance(res, dict):
        res = dict(res)
        res["warnings"] = list(res.get("warnings") or []) + warn
    return res


def _tool_effect_check(ctx, track_name=None, op_id=None, bones=None,
                       frames=None, **_):
    armature = ctx["armature"]
    if armature is None:
        raise RuntimeError("没有识别到 RIG 骨架")
    op = None
    if op_id:
        op = _op_by_id(ctx, op_id)
        track, strip = agent_ops.find_op_strip(armature, op)
        if track is None:
            raise RuntimeError(f"{op_id} 的轨找不到了（可能已被撤销/删除）")
        track_name = track.name
    if bones is None or frames is None:
        if op is None:
            ops = agent_ops.list_ops(ctx["data_dir"])
            op = next((o for o in reversed(ops)
                       if o.get("status") in ("preview", "committed")), None)
        if op is None:
            raise RuntimeError("op log 里没有修复记录，传 bones/frames")
        if not bones:
            pbones = [b for b in (op.get("params", {}).get("bones")
                                  or [op.get("params", {}).get("bone")])
                      if b and armature.pose.bones.get(b) is not None]
            # 参数里没写骨（foot_lock 记的是 side、restore_accent 记的是 path）
            # → 用 strip 实际写了的骨
            bones = pbones or sorted(_strip_bones(armature, op))
        if not frames:
            # 默认采样落在 taper 之外：strip 两端各 blend 帧权重从 0 渐升，
            # 端点权重恰为 0——旧版采 [起, 中, 止] 必然报"1/3 帧有变化"。
            # 工具自己报了生效段（overlap 的 metrics.inner_frames、motion_copy 的
            # inner_range）就用它：overlap 的自动 blend 在 params 里记成 None，
            # 按 0 算会采到 taper 端点。
            fr = op.get("frames")
            mets = op.get("metrics") or {}
            inner = mets.get("inner_frames") or mets.get("inner_range")
            if inner and int(inner[1]) >= int(inner[0]):
                lo, hi = int(inner[0]), int(inner[1])
            else:
                bl = (op.get("params") or {}).get("blend", 4)
                if bl is None:
                    bl = mets.get("blend", 0)
                bl = int(bl or 0)
                lo, hi = fr[0] + bl, fr[1] - bl
                if hi < lo:
                    lo, hi = fr[0], fr[1]
            frames = sorted({lo, (lo + hi) // 2, hi})
    if track_name is None:
        # 没指定就查所有 agent 轨（A/B 语义：全部修复一起 mute——一次 A/B，
        # 不再逐轨测再取最大：既与文档一致，求值次数也从 6×轨数 降到 6）
        armature_anim = getattr(armature, "animation_data", None)
        names = [t.name for t in (armature_anim.nla_tracks if armature_anim else ())
                 if agent_ops.is_agent_track_name(t.name)]
        if not names:
            raise RuntimeError("RIG 上没有 agent 轨")
        res = agent_ops.effect_check(ctx["scene"], armature, track_names=names,
                                     bones=list(bones), frames=list(frames))
        res["tracks"] = names
        label = f"全部 agent 轨({len(names)})"
    else:
        res = agent_ops.effect_check(ctx["scene"], armature,
                                     track_name=track_name,
                                     bones=list(bones), frames=list(frames))
        label = track_name
    return {"summary": f"{label}: {res['verdict']}",
            "data": res, "warnings": [], "truncated": False, "hint": ""}


_FK_LEG = ("thigh_fk.{s}", "shin_fk.{s}", "foot_fk.{s}", "toe_fk.{s}")
_IK_LEG = ("thigh_ik.{s}", "foot_ik.{s}", "foot_heel_ik.{s}", "foot_spin_ik.{s}")
_FK_ARM = ("upper_arm_fk.{s}", "forearm_fk.{s}", "hand_fk.{s}")
_IK_ARM = ("upper_arm_ik.{s}", "hand_ik.{s}")


def _invisible_bones(armature, bones) -> list:
    """写了看不见的控制骨（2026-10-04 §16）：腿是 IK 时的 FK 腿骨、肢体是 FK 时的 IK 控制骨。
    旧剧本"膝朝向 → hold_pose thigh_fk"在这个 RIG 上就是这样一次"修好了、验收也过了、用户什么都没看到"。"""
    out = []
    for b in bones:
        for s in ("L", "R"):
            leg_ik = agent_anatomy.limb_is_ik(armature, "knee", s)
            arm_ik = agent_anatomy.limb_is_ik(armature, "elbow", s)
            if leg_ik and b in [p.format(s=s) for p in _FK_LEG]:
                out.append((b, f"{s} 腿是 IK：{b} 写了看不见——膝朝向用 swivel，脚的位置/朝向用 foot_ik.{s}"))
            elif (not leg_ik) and b in [p.format(s=s) for p in _IK_LEG] and \
                    armature.pose.bones.get(f"thigh_parent.{s}") is not None:
                out.append((b, f"{s} 腿是 FK：{b} 写了看不见——改 thigh_fk/shin_fk/foot_fk.{s}"))
            elif arm_ik and b in [p.format(s=s) for p in _FK_ARM]:
                out.append((b, f"{s} 臂是 IK：{b} 写了看不见——手用 hand_ik.{s}，肘朝向用 swivel"))
            elif (not arm_ik) and b in [p.format(s=s) for p in _IK_ARM] and \
                    armature.pose.bones.get(f"upper_arm_parent.{s}") is not None:
                out.append((b, f"{s} 臂是 FK：{b} 写了看不见——改 upper_arm_fk/forearm_fk/hand_fk.{s}"))
    return out


def _reject_invisible(armature, bones, tool):
    bad = _invisible_bones(armature, bones)
    if bad:
        raise RuntimeError(f"{tool}：" + "；".join(w for _b, w in bad))


def _world_axis_warning(ctx, toward, frame_range) -> list:
    """给的是世界向量（比如 [0,-1,0]），而这段角色躯干朝别处 → 提醒"朝前"该用 forward。"""
    if toward is None or isinstance(toward, str):
        return []
    try:
        from mathutils import Vector
        from . import agent_view
        from .animation import preserve_scene_frame, set_scene_frame
        v = Vector([float(x) for x in toward]).normalized()
        if abs(v.z) > 0.5:
            return []
        scene, arm = ctx["scene"], ctx["armature"]
        mmd = getattr(ctx.get("settings"), "mmd_armature", None)
        if frame_range:
            f = (int(frame_range[0]) + int(frame_range[-1])) // 2
        else:
            f = int(scene.frame_current)
        with preserve_scene_frame(scene):
            set_scene_frame(scene, f)
            cf = agent_view.char_frame(arm, mmd=mmd)
        if cf is None:
            return []
        h = Vector((v.x, v.y, 0.0)).normalized()
        diff = agent_view._angle(h, cf["forward"])
        if diff > 25.0:
            return [f"你给的是世界向量 {[round(x, 3) for x in v]}；第 {f} 帧角色躯干朝 "
                    f"{agent_view.yaw_words(cf['forward'])}（与你的向量水平差 {diff:.0f}°）。"
                    "用户说的'朝前/朝左'指角色自己的前/左：用方向词 forward / char_left（逐帧跟着角色转）"]
    except Exception:  # noqa: BLE001 - 只是提醒
        return []
    return []


def _tool_hold_pose(ctx, bones, frame_range, target="values",
                    values=None, ref_frame=None, world_dir=None,
                    world_axis="Y", secondary_axis=None,
                    dir_object=None, dir_mode="arrow",
                    flip_guard_deg=150.0,
                    mode="replace", threshold_deg=8.0, strength=1.0,
                    blend=4, op_mode="preview", track_name=None,
                    dry_run=False, view="camera", **_):
    armature = ctx["armature"]
    if armature is None:
        raise RuntimeError("没有识别到 RIG 骨架")
    names = _resolve_bones(armature, list(bones))
    _reject_invisible(armature, names, "hold_pose")
    op = agent_ops.hold_pose(
        armature, _base_action(armature),
        names, frame_range,
        target=target, values=values, ref_frame=ref_frame,
        world_dir=world_dir, world_axis=world_axis,
        secondary_axis=secondary_axis,
        dir_object=dir_object, dir_mode=dir_mode,
        flip_guard_deg=float(flip_guard_deg),
        scene=ctx["scene"],
        mode=mode, threshold_deg=float(threshold_deg),
        strength=float(strength), blend=int(blend), op_mode=op_mode,
        data_dir=ctx["data_dir"], track_name=track_name,
        dry_run=bool(dry_run), view=str(view))
    if not op.get("dry_run"):
        _write_common(ctx, armature, frame_range)
    if target == "world_dir" and world_dir is not None and not dir_object:
        warn = _world_axis_warning(ctx, world_dir, frame_range)
        if warn:
            op["_warnings"] = list(op.get("_warnings") or []) + warn
    return op


def _tool_probe_anatomy(ctx, part, side=None, bone=None, finger=None,
                        frame_range=None, toward=None, max_frames=9, view="camera", **_):
    """语义解剖探头：从几何推世界方向 + owner 骨局部向量 + 置信度。"""
    armature = ctx["armature"]
    if armature is None:
        raise RuntimeError("没有识别到 RIG 骨架")
    res = agent_anatomy.probe(
        ctx["scene"], armature, part=part, side=side, bone=bone,
        finger=finger, frame_range=frame_range, toward=toward,
        max_frames=int(max_frames), view=str(view))
    conf = res.get("confidence")
    summary = (f"{part}{'.' + side if side else ''}: "
               f"world={res.get('world_dir')} conf={conf}"
               + (f" err={res['err_max_deg']}°"
                  if res.get("err_max_deg") is not None else ""))
    warnings = _world_axis_warning(ctx, toward, frame_range)
    ev0 = res.get("evidence") or {}
    for kind in ("knee", "elbow"):
        src = ev0.get(f"{kind}_source")
        if src == "bend":
            warnings.append(f"{'膝' if kind == 'knee' else '肘'}朝向没有全片标定（{ev0.get('hinge_calibration')}），"
                            "用的是当帧弯曲方向：直腿/直臂的帧没有定义。建议 markers create 建 MCD_"
                            f"{kind}.{(side or 'L').upper()} 让用户确认")
        elif src == "hinge":
            cal = res.get("hinge_calibration") or {}
            if (cal.get("spread_p90_deg") or 0) > 20:
                warnings.append(f"铰链轴不太刚性（全片 p90 散布 {cal.get('spread_p90_deg')}°）：朝向定义有 ±"
                                f"{cal.get('spread_p90_deg')}° 的不确定，拿不准就让用户用 MCD_{kind} 箭头确认")
        if src is not None and res.get("owner_bone") and "_ik." in str(res.get("owner_bone")):
            warnings.append(f"这条{'腿' if kind == 'knee' else '胳膊'}是 IK：朝向修复用 swivel（swivel_args 可直接用），"
                            f"别 hold_pose {res.get('owner_bone')}")
    if conf is not None and conf < 0.5:
        warnings.append(f"低置信度({conf})：{res.get('evidence')}")
        if res.get("alternatives"):
            warnings.append("符号歧义：alternatives 给出两个候选方向")
    ev = res.get("evidence") or {}
    if str(part).lower() in ("palm", "back_of_hand") and ev.get("palm_source") == "fingers":
        warnings.append("掌心没做网格标定（" + str(ev.get("mesh_calibration", "没有 settings.target_mesh"))
                        + "）：按手指几何推断，手指弯曲时与视口里看到的掌心可差 45°+（中位数）；"
                          "拿不准就报告，别修")
    if ev.get("sign_conflict"):
        warnings.append("掌心符号：网格标定（拇指所在侧）与手指弯曲方向不一致——让用户在视口确认一帧再修")
    if ev.get("marker_ignored"):
        why = agent_anatomy.MARKER_IGNORE_REASONS.get(ev.get("marker_ignored_why"), "不合格")
        warnings.append(f"标记箭头 {ev['marker_ignored']} {why}，已忽略（按几何/网格定义算）："
                        "用 markers 工具重建（action=create, overwrite=true），或在 Blender 里清掉它的关键帧/约束、"
                        "Ctrl+P→骨骼 绑到对应的骨；markers check 看详情")
    return {"summary": summary, "data": res, "warnings": warnings,
            "truncated": False, "hint": ""}


def _tool_bake_range(_ctx, frame_range, roles=None, point_ids=None, **_):
    """Return world pos/quat arrays for roles/points over a range - the LLM
    feeds these into apply_exemplar / pelvis math."""
    store = get_store()
    a, b = int(frame_range[0]), int(frame_range[1])
    mask = (store.frames >= a) & (store.frames <= b)
    out = {"frames": [int(f) for f in store.frames[mask]]}
    for role in (roles or list(store.positions)):
        if role in store.positions:
            out[f"pos::{role}"] = np.asarray(store.positions[role])[mask].tolist()
    return out


def _write_common(ctx, armature, frame_range):
    _bump_ops_rev(ctx)
    _redraw()
    _focus_preview(ctx["scene"], frame_range)


def _tool_clean_jitter(ctx, frame_range, bone=None, paths=None,
                       strength=1.0, width=5, blend=4, mode="preview",
                       dry_run=False, **_):
    armature = ctx["armature"]
    if paths is None and bone:
        bone = _resolve_bones(armature, [bone])[0]
        mode_r = armature.pose.bones[bone].rotation_mode
        rot_prop, n_rot = (("rotation_quaternion", 4) if mode_r == "QUATERNION"
                           else ("rotation_axis_angle", 4) if mode_r == "AXIS_ANGLE"
                           else ("rotation_euler", 3))   # 手臂是 Euler 骨
        paths = (
            [(agent_ops.bone_path(bone, "location"), i) for i in range(3)]
            + [(agent_ops.bone_path(bone, rot_prop), i) for i in range(n_rot)]
        )
    if not paths:
        raise RuntimeError("clean_jitter 需要 bone 或 paths")
    _reject_invisible(armature, sorted({b for b in (_bone_of_path(p[0] if isinstance(p, (list, tuple)) else p)
                                                    for p in paths) if b}), "clean_jitter")
    op = agent_ops.clean_jitter(
        armature, _base_action(armature),
        [tuple(p) for p in paths], frame_range,
        strength=float(strength), width=int(width), blend=int(blend),
        mode=mode, data_dir=ctx["data_dir"], dry_run=bool(dry_run))
    if not op.get("dry_run"):
        _write_common(ctx, armature, frame_range)
    return op


_GROUND_MODES = {"lift": "高于 地面+rest_clearance 的帧往下拉到这个高度（治悬空）",
                 "pen": "低于 地面+rest_clearance 的帧往上推到这个高度（治穿地）",
                 "float": "整段钉在 地面+rest_clearance（脚跟/脚尖滚动也会被抹平）"}


def _tool_fix_ground(ctx, frame_range, side, loc_path,
                     mode="lift", pin_xy=False, blend=4, rest_clearance=0.0,
                     op_mode="preview", **_):
    """rest_clearance（米）：脚底点（关节中心）正常着地时离地面的高度。穿厚底鞋的
    模型这里不是 0——用 ground_report 的 fix_ground_args（已按这只脚标定好）。"""
    if mode not in _GROUND_MODES:
        raise RuntimeError(
            f"fix_ground mode={mode!r} 不存在；只有 "
            + "；".join(f"{k}：{v}" for k, v in _GROUND_MODES.items())
            + "。（旧手册写的 snap 从来不存在——整段贴地是 float）")
    armature = ctx["armature"]
    store = get_store()
    sole = store.signals.get(f"foot.{side}.sole_h")
    if sole is None:
        raise RuntimeError(f"信号库没有 foot.{side}.sole_h")
    a, b = int(frame_range[0]), int(frame_range[1])
    mask = (store.frames >= a) & (store.frames <= b)
    op = agent_ops.fix_ground(
        armature, _base_action(armature), loc_path, sole[mask],
        floor_z=store.floor_z, rest_clearance=float(rest_clearance),
        mode=mode, pin_xy=bool(pin_xy),
        frame_range=frame_range, blend=int(blend), op_mode=op_mode,
        data_dir=ctx["data_dir"])
    _write_common(ctx, armature, frame_range)
    return op


def _tool_restore_accent(ctx, frame_range, data_path, index=None, method="retime",
                         strength=0.5, impact_frame=None,
                         retime_speed=1.5, retime_split=0.35,
                         raw_action=None, blend=4, op_mode="preview",
                         dry_run=False, **_):
    """data_path 指到 .rotation_quaternion / .location 时四分量/三轴整体
    重塑（index 忽略）；其他通道才需要 index 选分量。

    默认 `method="retime"`（**真正的「时间重塑」**：只重映射时间、端点固定、姿态集合
    不变 → 结构上不可能产生角度偏移）。2026-10-06 前默认是 `ease_reshape`——它在冲击
    前后**加一段角度鼓包**（实测：strength 0.5 时 9 帧内峰值 −31.9°、strength 1.0 时
    总转角 63.7°），用户判定为"错的效应"：0.5 看不见、1.0 变成"手多弯一下"。要过冲
    效果请显式写 `method="ease_reshape"`，并知道自己在要什么。
    """
    armature = ctx["armature"]
    if _bone_of_path(data_path):
        _reject_invisible(armature, [_bone_of_path(data_path)], "restore_accent")
    raw_values = None
    if raw_action:
        raw_act = bpy.data.actions.get(raw_action)
        if raw_act is None:
            raise RuntimeError(f"参考动作 {raw_action} 不存在")
        raw_values = agent_bake.sample_fcurve_values(
            raw_act, data_path, int(index) if index is not None else 0,
            int(frame_range[0]), int(frame_range[1]))
    op = agent_ops.restore_accent(
        armature, _base_action(armature), data_path,
        int(index) if index is not None else None,
        frame_range, method, strength=float(strength),
        raw_values=raw_values, impact_frame=impact_frame,
        retime_speed=retime_speed, retime_split=retime_split,
        blend=int(blend), op_mode=op_mode,
        data_dir=ctx["data_dir"], dry_run=bool(dry_run))
    if not op.get("dry_run"):
        _write_common(ctx, armature, frame_range)
    return op


def _tool_solve_pelvis(ctx, frame_range, pelvis_path, pelvis_dz,
                       blend=4, op_mode="preview", **_):
    armature = ctx["armature"]
    op = agent_ops.solve_pelvis(
        armature, _base_action(armature), pelvis_path,
        list(pelvis_dz), frame_range=frame_range, blend=int(blend),
        op_mode=op_mode, data_dir=ctx["data_dir"])
    _write_common(ctx, armature, frame_range)
    return op


def _tool_apply_exemplar(ctx, frame_range, ex_id, loc_path, quat_path,
                         target_pos, target_quat, anchor_yaw_deg=0.0,
                         yaw_scale=1.0, mirror=False, blend=4,
                         op_mode="preview", **_):
    armature = ctx["armature"]
    folder = agent_ops.exemplar_dir(ctx["data_dir"])
    if not (folder / f"{ex_id}.npz").is_file():
        have = sorted(p.stem for p in folder.glob("*.npz")) if folder.is_dir() else []
        raise RuntimeError(f"模板 {ex_id!r} 不存在；这个文件登记过的模板：{have or '无'}。"
                           "没有模板就别用 apply_exemplar（报告给协调者）")
    ex = agent_ops.load_exemplar(ctx["data_dir"], ex_id)
    op = agent_ops.apply_exemplar(
        armature, ex, target_pos, target_quat, float(anchor_yaw_deg),
        loc_path, quat_path, frame_range=frame_range,
        yaw_scale=float(yaw_scale), mirror=bool(mirror), blend=int(blend),
        op_mode=op_mode, data_dir=ctx["data_dir"])
    _write_common(ctx, armature, frame_range)
    return op


def _tool_validate(ctx, frame_range=None, **_):
    store = get_store()
    heights = agent_query.contact_heights(store)
    res = agent_ops.validate(store.signals, store.frames, frame_range,
                             floor_z=store.floor_z,
                             contact_height={s: h["height_m"] for s, h in heights.items()})
    if heights:
        res["contact_height_mm"] = {s: round(h["height_m"] * 1000.0, 1) for s, h in heights.items()}
    return res


def _tool_list_ops(ctx, owner=None, op_id=None, live=None, frames=None,
                   compact=None, bones=None, **_):
    """Op log enriched with live state (track / exponent / mute) so an agent can
    see exactly what is on the rig without guessing.

    过滤（都可选；不传 = 旧版全量输出）：owner="<agent_id>"（"none"=无 owner 的
    历史修复）、op_id、live=true（去掉 reverted 历史）、frames=[a,b]（与之相交）、
    bones=[...]（只要写了这些骨的修复；可用角色名/链名会被解析成骨名）、
    compact=true（只回 fixes 对账行，不带完整 params/metrics——自查用这个）。"""
    armature = ctx["armature"]
    rows = agent_ops.reconcile(armature, ctx["data_dir"])
    ops = agent_ops.list_ops(ctx["data_dir"])
    owners = {o.get("id"): o.get("owner") for o in ops if o.get("owner")}
    for row in rows:                   # owner 列（只在有 owner 时出现）
        if row.get("op_id") in owners:
            row["owner"] = owners[row["op_id"]]
    stale = {o.get("id"): o.get("stale") for o in ops if o.get("stale")}
    for row in rows:                   # 复制的源被改过（只在被标记时出现）
        if row.get("op_id") in stale:
            row["stale"] = stale[row["op_id"]]
    if all(v is None for v in (owner, op_id, live, frames, compact, bones)):
        return {"ops": ops, "fixes": rows}

    def keep(item, oid, fr, own):
        if owner is not None and (own or "none") != str(owner):
            return False
        if op_id and oid != op_id:
            return False
        if frames is not None and fr and not agent_claims.frames_overlap(
                (int(frames[0]), int(frames[1])), fr):
            return False
        return True
    ops = [o for o in ops
           if keep(o, o.get("id"), o.get("frames"), o.get("owner"))
           and not (live and o.get("status") == "reverted")]
    rows = [r for r in rows
            if keep(r, r.get("op_id"), r.get("frames"), r.get("owner"))]
    if compact or bones:
        for r in rows:
            if r.get("alive") and r.get("strip"):
                r["bones"] = sorted(_strip_bones(
                    armature, {"track": r.get("track"), "strip": r.get("strip")}))
    if bones:
        # 写前查"这根骨在这段上有没有别人的同类修复"：按帧过滤还会带出无关骨，
        # 弱模型得自己读 bones 判断（sonnet 第六轮反馈）——这里直接按骨过滤。
        want = _claim_bones(ctx, [bones] if isinstance(bones, str) else list(bones), None)
        if want is not agent_claims.ALL:
            rows = [r for r in rows if set(want) & set(r.get("bones") or ())]
            ids = {r.get("op_id") for r in rows}
            ops = [o for o in ops if o.get("id") in ids]
    out = {"fixes": rows,
           "filters": {k: v for k, v in (("owner", owner), ("op_id", op_id),
                                          ("live", live), ("frames", frames),
                                          ("bones", bones),
                                          ("compact", compact)) if v is not None}}
    if not compact:
        out["ops"] = ops
    return {"summary": f"{len(rows)} 条修复在场景里"
                       + ("" if compact else f"，{len(ops)} 条日志记录"),
            "data": out, "warnings": [], "truncated": False,
            "hint": "fixes 行：status=preview、alive=true、owner=你 = 写上了"}


def _tool_commit(ctx, op_id, **_):
    """Mark committed.  With per-op tracks the strip no longer moves: track is
    identity, status is intent, and a committed fix stays adjustable."""
    op = agent_ops.commit(ctx["data_dir"], op_id)
    _bump_ops_rev(ctx)
    try:
        bpy.ops.ed.undo_push()
    except Exception:
        pass
    _redraw()
    return op


def _op_by_id(ctx, op_id):
    for op in agent_ops.list_ops(ctx["data_dir"]):
        if op["id"] == op_id:
            return op
    raise RuntimeError(f"op {op_id} 不在记录里")


def _op_strip(ctx, op_id):
    return _op_by_id(ctx, op_id).get("strip", "")


def _bump_ops_rev(ctx):
    """Tell the panel its fix list is stale (write/commit/revert happened)."""
    try:
        ctx["settings"].agent_ops_rev = int(ctx["settings"].agent_ops_rev) + 1
    except Exception:
        pass


def _tool_revert(ctx, op_id, **_):
    op = agent_ops.revert(ctx["armature"], ctx["data_dir"], op_id)
    _bump_ops_rev(ctx)
    _redraw()
    return op


def _tool_reapply(ctx, op_id, overrides=None, **_):
    """用新参数重写某条 op 的 strip（同轨删旧写新，op_id 不变）。

    overrides 是 params 的局部覆盖，如 {"threshold_deg": 6, "blend": 8}、
    {"dir_object": "mcd_dir", "dir_mode": "arrow"}、{"frame_range": [a,b]}。"""
    armature = ctx["armature"]
    if armature is None:
        raise RuntimeError("没有识别到 RIG 骨架")
    op = agent_ops.reapply(ctx["data_dir"], armature, str(op_id),
                           scene=ctx["scene"], **dict(overrides or {}))
    _bump_ops_rev(ctx)
    _sync_params_list(ctx["settings"], ctx["data_dir"])
    _redraw()
    return {"summary": f"已重写 {op['id']}（{op.get('tool')}）",
            "data": op, "warnings": [], "truncated": False, "hint": ""}


def _tool_ab_toggle(ctx, muted=None, **_):
    """A/B 对比：mute/unmute 所有 agent 轨（每条修复各自一轨），
    mcd_base 保持原样。

    `muted=true/false` = **显式目标状态**（幂等：调几次都一样）；不给 = 老的"翻转"。
    2026-10-06 为什么要加：老语义是"**只要有一个还响着 → 全部静音；全静音了 → 全部恢复**"，
    于是"静音 → 写新 op → 恢复"这套流程里的"恢复"会把**刚写的新 op 一起静音**，
    下一轮的"静音"反而变成取消静音（w5 跑完时 10 条全落成 muted=true）。
    要做 A/B 请用显式状态，别靠翻转。"""
    armature = ctx["armature"]
    anim = getattr(armature, "animation_data", None)
    agent_tracks = [t for t in (anim.nla_tracks if anim else ())
                    if agent_ops.is_agent_track_name(t.name)]
    if ctx.get("agent_id"):            # 多 agent：只拨自己的轨，别动别人的
        mine = {o.get("track") for o in agent_ops.list_ops(ctx["data_dir"])
                if o.get("owner") == ctx["agent_id"]}
        agent_tracks = [t for t in agent_tracks if t.name in mine]
    if not agent_tracks:
        return {"muted": None, "note": "没有 agent 轨"}
    # 显式给了就照办（幂等）；没给才回退到老的方向判定
    if muted is None:
        new_state = any(not t.mute for t in agent_tracks)
    else:
        new_state = bool(muted)
    chans = set()
    for t in agent_tracks:
        t.mute = new_state
        for st in t.strips:
            chans |= agent_ops.action_channels(st.action)
    if new_state:
        # 基底没 key 的通道（thigh_ik 的 Y 旋转等）静音后会停在最后一次求值的值上，A 看到的就不是原样
        agent_ops.settle_unanimated(armature, chans)
    _redraw()
    return {"muted": new_state,
            "tracks": [t.name for t in agent_tracks]}


def _tool_set_preview(ctx, frame_range, **_):
    agent_ops.set_preview(ctx["scene"], frame_range)
    _redraw()
    return {"preview_range": [int(frame_range[0]), int(frame_range[1])]}


def _tool_set_influence(ctx, value, op_id=None, track_name=None, **_):
    """力度旋钮：按 op 把该修复的 delta 曲线重写成 delta^value。

    op_id 是首选寻址（每条修复一条轨）；track_name 是回退（找该轨上全部
    strip），后者只留给迁移前的旧布局。"""
    armature = ctx["armature"]
    anim = getattr(armature, "animation_data", None)
    if anim is None:
        raise RuntimeError("RIG 没有动画数据")
    if op_id:
        op = _op_by_id(ctx, op_id)
        track, strip = agent_ops.find_op_strip(armature, op)
        if strip is None:
            raise RuntimeError(f"{op_id} 的 strip 找不到了（可能已被撤销/删除）")
        res = agent_ops.set_strip_exponent(strip, float(value))
        _redraw()
        return {"summary": f"{op_id} 力度 → {float(value):g}（delta^{float(value):g}）",
                "data": {"op_id": op_id, "strip": strip.name,
                         "track": track.name if track else None,
                         "exponent": float(value),
                         "touched": res.get("touched", 0),
                         "scalar_touched": res.get("scalar_touched", 0)},
                "warnings": [], "truncated": False,
                "hint": "reapply 会保留这个力度；复测用实时类读工具"}
    tracks = [t for t in anim.nla_tracks
              if t.name.startswith(track_name or PREVIEW_TRACK)]
    if not tracks:
        raise RuntimeError(f"没有轨 {track_name or PREVIEW_TRACK}")
    touched = 0
    for t in tracks:
        for s in t.strips:
            agent_ops.set_strip_exponent(s, float(value))
            touched += 1
    _redraw()
    return {"influence": float(value), "strips": touched,
            "tracks": [t.name for t in tracks]}


def _tool_save(ctx, filepath=None, **_):
    """存盘。headless 盲写模式（tools/headless_server.py）下 op 全在内存，
    不写盘进程一关就没了——远程 agent 每完成一段工作必须调一次。"""
    if filepath:
        bpy.ops.wm.save_as_mainfile(filepath=str(filepath))
    else:
        bpy.ops.wm.save_mainfile()
    return {"saved": bpy.data.filepath}


def _claim_bones(ctx, bones, chain):
    if chain:
        from . import agent_pose
        return set(agent_pose.chain_preset(chain, ctx["armature"]))
    names = list(bones or [])
    if not names or names in (["*"], ["ALL"], ["all"]):
        return agent_claims.ALL
    return set(_resolve_bones(ctx["armature"], names))


def _need_agent(ctx, tool):
    if not ctx.get("agent_id"):
        raise agent_query.AgentQueryError(
            f"{tool} 需要 agent_id", code="E_SCOPE",
            fix='每个调用都带 "agent_id":"<你的名字>"（全程同一个）')
    return ctx["agent_id"]


def _tool_claim(ctx, bones=None, chain=None, frames=None, frame_range=None,
                ttl_s=agent_claims.DEFAULT_TTL_S, note="", strict=False,
                check_only=False, **_):
    """咨询性租约：声明"我要改这些骨的这段帧"。同骨帧段重叠的别人 → 不批；
    父子骨重叠 → 批但给 related 警告（strict=true 则不批）。TTL 到期自动失效，
    你的每次 claim/写入都会续期。check_only=true 只查不占。"""
    agent_id = _need_agent(ctx, "claim")
    fr = frames if frames is not None else frame_range
    bset = _claim_bones(ctx, bones, chain)
    res = _LEASES.claim(agent_id, bset,
                        None if fr is None else (int(fr[0]), int(fr[1])),
                        ancestors=_ancestor_fn(ctx["armature"]),
                        ttl_s=float(ttl_s), note=str(note or ""),
                        strict=bool(strict), check_only=bool(check_only))
    warnings = [f"层级相关：{r['agent_id']} 占着 {r['pairs']} @ {r['frames']}"
                for r in res["related"][:3]]
    scope_txt = ("ALL" if bset is agent_claims.ALL else
                 ",".join(sorted(bset)[:4]) + ("…" if len(bset) > 4 else ""))
    if res["granted"]:
        summary = (f"{'可认领（未占用）' if check_only else '已认领 ' + str(res['claim_id'])}"
                   f"：{scope_txt} @ {list(fr) if fr is not None else 'ALL'}")
    else:
        c = res["conflicts"][0] if res["conflicts"] else res["related"][0]
        summary = (f"未获批：{c['agent_id']} 占着 "
                   f"{c.get('bones') or c.get('pairs')} @ {c['frames']}——换 scope 或等")
    res["your_claims"] = _LEASES.of(agent_id)
    res["version"] = _DATA_VERSION
    return {"summary": summary, "data": res, "warnings": warnings,
            "truncated": False,
            "hint": "" if res["granted"] else "list_claims 看全表；别 force"}


def _tool_release(ctx, claim_id=None, **_):
    agent_id = _need_agent(ctx, "release")
    try:
        ids = _LEASES.release(agent_id, claim_id)
    except PermissionError as exc:
        raise agent_query.AgentQueryError(str(exc), code="E_OWNER",
                                          fix="只能 release 自己的 claim")
    return {"summary": f"已释放 {ids or '（无）'}",
            "data": {"released": ids, "your_claims": _LEASES.of(agent_id)},
            "warnings": [], "truncated": False, "hint": ""}


def _tool_list_claims(ctx, **_):
    rows = _LEASES.table()
    by_agent: dict = {}
    for o in agent_ops.list_ops(ctx["data_dir"]):
        if o.get("owner") and o.get("status") in ("preview", "committed"):
            by_agent.setdefault(o["owner"], []).append(o["id"])
    return {"summary": f"{len(rows)} 条租约，{len(by_agent)} 个 agent 有在册修复",
            "data": {"claims": rows, "ops_by_owner": by_agent,
                     "version": _DATA_VERSION},
            "warnings": [], "truncated": False, "hint": ""}


def _tool_plan_scopes(ctx, tasks=None, **_):
    """派单前体检（协调者用，只读）：把打算并行派出去的任务 scope 先过一遍。

    tasks=[{"name":"左臂去抖","bones":[...]|"chain":"arm.L","frames":[a,b],
            "reads":{"chain":"arm.L","frames":[c,d]}（可选，也可以是列表）}, ...]
    返回：两两冲突（同骨+帧重叠 = hard，必须分批；父子骨+帧重叠 = related，
    父骨的任务应先做；A 写的骨×帧落在 B 声明要**读**的范围里 = read，A 先做——
    比如动作复制的源窗被别人改，复制的就是旧姿态）、与现有租约的冲突、建议的
    并行批次 waves（同一批内两两无 hard 冲突，子骨任务/读者排在父骨任务/写者之后）。"""
    tasks = list(tasks or [])
    if not tasks:
        raise agent_query.AgentQueryError(
            "plan_scopes 需要 tasks 列表", code="E_SCOPE",
            fix='tasks=[{"name":"…","chain":"arm.L","frames":[a,b]}, …]')
    anc = _ancestor_fn(ctx["armature"])
    rows = []
    for i, t in enumerate(tasks):
        bset = _claim_bones(ctx, t.get("bones"), t.get("chain"))
        fr = t.get("frames") or t.get("frame_range")
        reads = t.get("reads") or []
        if isinstance(reads, Mapping):
            reads = [reads]
        rd = []
        for r_ in reads:
            rfr = r_.get("frames") or r_.get("frame_range")
            rd.append((_claim_bones(ctx, r_.get("bones"), r_.get("chain")),
                       None if rfr is None else (int(rfr[0]), int(rfr[1]))))
        rows.append({"i": i, "name": str(t.get("name") or f"task{i}"),
                     "bones": bset, "reads": rd,
                     "frames": None if fr is None else (int(fr[0]), int(fr[1]))})
    pairs = []
    parent_of = {r["i"]: set() for r in rows}       # j ∈ parent_of[i]: j 改的是 i 的祖先骨
    hard_with = {r["i"]: set() for r in rows}
    for x in range(len(rows)):
        for y in range(x + 1, len(rows)):
            a, b = rows[x], rows[y]
            if not agent_claims.frames_overlap(a["frames"], b["frames"]):
                continue
            fr = agent_claims.frames_intersection(a["frames"], b["frames"])
            shared = agent_claims._bones_hard(a["bones"], b["bones"])
            if shared is agent_claims.ALL or shared:
                pairs.append({"a": a["name"], "b": b["name"], "kind": "hard",
                              "bones": "ALL" if shared is agent_claims.ALL
                              else sorted(shared)[:8], "frames": fr})
                hard_with[x].add(y)
                hard_with[y].add(x)
                continue
            rel = agent_claims._bones_related(a["bones"], b["bones"], anc)
            if rel:
                pairs.append({"a": a["name"], "b": b["name"], "kind": "related",
                              "pairs": [f"{p}→{c}" for p, c in rel[:6]],
                              "frames": fr})
                for p_bone, c_bone in rel:
                    if p_bone in (a["bones"] or ()):
                        parent_of[y].add(x)
                    else:
                        parent_of[x].add(y)
    # 读依赖：x 写的骨（或其祖先）× 帧落在 y 要读的范围里 → x 先做
    for x in range(len(rows)):
        for y in range(len(rows)):
            if x == y:
                continue
            for rbones, rfr in rows[y]["reads"]:
                if not agent_claims.frames_overlap(rows[x]["frames"], rfr):
                    continue
                shared = agent_claims._bones_hard(rows[x]["bones"], rbones)
                rel = agent_claims._bones_related(rows[x]["bones"], rbones, anc)
                if not (shared is agent_claims.ALL or shared or rel):
                    continue
                pairs.append({"a": rows[x]["name"], "b": rows[y]["name"], "kind": "read",
                              "bones": ("ALL" if shared is agent_claims.ALL else
                                        sorted(shared)[:8] if shared else
                                        [f"{p}→{c}" for p, c in rel[:6]]),
                              "frames": agent_claims.frames_intersection(rows[x]["frames"], rfr),
                              "note": f"{rows[y]['name']} 读的范围被 {rows[x]['name']} 改：后者先做"})
                parent_of[y].add(x)
                break
    # 批次：父骨任务/写者先；同批不许 hard 冲突
    wave_of: dict = {}
    order = sorted(range(len(rows)), key=lambda i: len(parent_of[i]))
    for _round in range(len(rows) + 1):
        changed = False
        for i in order:
            need = max([wave_of.get(j, 0) + 1 for j in parent_of[i]
                        if j in wave_of] + [0])
            w = need
            while any(wave_of.get(k) == w for k in hard_with[i] if k != i):
                w += 1
            if wave_of.get(i) != w:
                wave_of[i] = w
                changed = True
        if not changed:
            break
    waves = []
    for w in sorted(set(wave_of.values())):
        waves.append([rows[i]["name"] for i in sorted(wave_of) if wave_of[i] == w])
    vs_claims = []
    for r in rows:
        hard, soft = _LEASES.conflicts("__plan__", r["bones"], r["frames"], anc)
        if hard or soft:
            vs_claims.append({"task": r["name"],
                              "held_by": sorted({h["agent_id"] for h in hard}),
                              "related_to": sorted({h["agent_id"] for h in soft})})
    for r in rows:
        for rbones, rfr in r["reads"]:
            hard, _soft = _LEASES.conflicts("__plan__", rbones, rfr, anc)
            if hard:
                vs_claims.append({"task": r["name"], "reads_held_by": sorted({h["agent_id"] for h in hard})})
    warnings = []
    for y, deps in parent_of.items():
        for x in deps:
            if wave_of.get(y, 0) <= wave_of.get(x, 0):
                warnings.append(f"{rows[x]['name']} 与 {rows[y]['name']} 互相依赖（循环），排不出先后：拆开或手动定顺序")
    n_hard = sum(1 for p in pairs if p["kind"] == "hard")
    n_read = sum(1 for p in pairs if p["kind"] == "read")
    n_rel = len(pairs) - n_hard - n_read
    summary = (f"{len(rows)} 个任务 → {len(waves)} 批"
               f"（hard 冲突 {n_hard}，父子相关 {n_rel}，读依赖 {n_read}，与现有租约冲突 {len(vs_claims)}）")
    return {"summary": summary,
            "data": {"waves": waves, "conflicts": pairs, "vs_claims": vs_claims,
                     "parallel_ok": len(waves) == 1 and not vs_claims},
            "warnings": sorted(set(warnings)), "truncated": False,
            "hint": "同一批内的任务可以同时派；下一批等上一批 release 后再派"}


TOOLS = {
    "plan_scopes": _tool_plan_scopes,
    "claim": _tool_claim,
    "release": _tool_release,
    "list_claims": _tool_list_claims,
    "ping": _tool_ping,
    "get_overview": _tool_overview,
    "list_intervals": _tool_list_intervals,
    "list_timeline_markers": _tool_list_timeline_markers,
    "describe": _tool_describe,
    "get_series": _tool_get_series,
    "find_events": _tool_find_events,
    "compare": _tool_compare,
    "snapshot": _tool_snapshot,
    "probe_anatomy": _tool_probe_anatomy,
    "get_joint_angles": _tool_get_joint_angles,
    "eval_bpy": _tool_eval,
    "bake_range": _tool_bake_range,
    "clean_jitter": _tool_clean_jitter,
    "hold_pose": _tool_hold_pose,
    "effect_check": _tool_effect_check,
    "fix_ground": _tool_fix_ground,
    "restore_accent": _tool_restore_accent,
    "solve_pelvis": _tool_solve_pelvis,
    "apply_exemplar": _tool_apply_exemplar,
    "validate": _tool_validate,
    "list_ops": _tool_list_ops,
    "commit": _tool_commit,
    "revert": _tool_revert,
    "reapply": _tool_reapply,
    "save": _tool_save,
    "ab_toggle": _tool_ab_toggle,
    "set_preview": _tool_set_preview,
    "set_influence": _tool_set_influence,
}


# ---------------------------------------------------------------------------
# plugin tools: each module may export
#   TOOLS        {name: fn(ctx, **args)}            bridge-level tool functions
#   WRITE_SCOPES {name: fn(ctx, args) -> [(bones, (a, b)), ...]}  marks a write
#                 tool + the bones/frames it may touch (concurrency leases)
#   TUNABLE      {name: [param specs]}              → agent_ops.TUNABLE_PARAMS
#   REAPPLY      {name: fn(...)}                    → agent_ops.REAPPLY_HANDLERS
# A missing module is fine (not shipped yet); a broken one is reported by ping
# instead of taking every other tool down with it.
_PLUGIN_MODULES = ("agent_motion", "agent_copy", "agent_principles",
                   "agent_overlap", "agent_contact", "agent_markers",
                   "agent_swivel", "agent_view")
_PLUGIN_ERRORS: dict = {}
WRITE_SCOPES: dict = {}


def _load_plugins():
    import importlib
    for name in _PLUGIN_MODULES:
        full = f"{__package__}.{name}"
        try:
            mod = importlib.import_module(full)
        except ModuleNotFoundError as exc:
            if exc.name != full:
                _PLUGIN_ERRORS[name] = repr(exc)
            continue
        except Exception as exc:  # noqa: BLE001 - report, keep serving
            _PLUGIN_ERRORS[name] = repr(exc)
            continue
        TOOLS.update(getattr(mod, "TOOLS", {}) or {})
        WRITE_SCOPES.update(getattr(mod, "WRITE_SCOPES", {}) or {})
        agent_ops.TUNABLE_PARAMS.update(getattr(mod, "TUNABLE", {}) or {})
        agent_ops.REAPPLY_HANDLERS.update(getattr(mod, "REAPPLY", {}) or {})


_load_plugins()


def _ctx():
    scene, settings = _settings()
    ctx = {
        "scene": scene,
        "settings": settings,
        "armature": _rig_armature(settings, scene),
        "source_armature": _source_armature(settings),
        "data_dir": _data_dir(settings),
    }
    # helpers for plugin tools (so they never import agent_bridge back)
    arm = ctx["armature"]
    ctx["resolve_bones"] = lambda names: _resolve_bones(arm, list(names))
    ctx["base_action"] = lambda: _base_action(arm)
    ctx["after_write"] = lambda frame_range: _write_common(ctx, arm, frame_range)
    return ctx


def _reject_swallowed(name, tool, args):
    """旧工具的包装用 `**_` 吞掉未知参数——拼错的参数被静默忽略、按默认值执行
    （§7-11 的 dry_run 漏写就是这么来的）。按它们的签名统一拒绝，并提示最接近的
    合法参数名；对合法调用零影响。插件工具（`**args` / `**unknown`）自己校验。"""
    import inspect
    from . import agent_pose
    try:
        params = inspect.signature(tool).parameters
    except (TypeError, ValueError):
        return
    var_kw = [p for p in params.values() if p.kind is p.VAR_KEYWORD]
    if not var_kw or var_kw[0].name != "_":
        return
    # expect_version / dry_run 由桥自己处理（版本校验、dry_run 支持表）
    agent_pose.reject_unknown_args(name, tool, args, internal=("ctx",),
                                   extra_allowed=("expect_version", "dry_run"))


def _dispatch(request: Mapping[str, Any]) -> dict:
    global _LAST_TOOL, _IN_TOOL
    name = str(request.get("tool", ""))
    args = dict(request.get("args") or {})
    agent_id = args.pop("agent_id", None)
    agent_id = str(agent_id) if agent_id not in (None, "") else None
    force = bool(args.pop("force", False))
    notes: list = []
    try:
        tool = TOOLS.get(name)
        if tool is None:
            args.pop("expect_version", None)
            raise agent_query.AgentQueryError(
                f"未知工具 {name!r}", code="E_UNKNOWN",
                fix=f"可用：{', '.join(sorted(TOOLS))}")
        dry = bool(args.get("dry_run"))
        if dry and name not in _dry_run_tools():
            # 旧版工具用 **_ 吞掉未知参数：dry_run 会被忽略、照样真写——而且绕过
            # 租约/owner/版本号（sonnet 验证里留下了 3 条无主 strip）。不支持就直说。
            raise agent_query.AgentQueryError(
                f"{name} 不支持 dry_run", code="E_SCOPE",
                fix=f"支持 dry_run 的：{', '.join(sorted(_dry_run_tools()))}；"
                    "其它写工具直接写（preview，可 revert/reapply）")
        _reject_swallowed(name, tool, args)
        ctx = _ctx()
        ctx["agent_id"] = agent_id
        scope = _write_scope(name, ctx, args, agent_id, force)
        _check_version(args, scope, agent_id, ctx, notes)
        if scope is not None:
            if not dry:
                # 坏掉的 op 日志在写 strip 之前就报出来（M16）：否则 strip 写上了、
                # 记账失败，留下一条"未登记"的孤儿
                agent_ops._load_oplog(ctx["data_dir"])
                _enforce_claims(name, ctx, scope, agent_id, force, notes)
            if name not in _OP_TOOLS and name != "ab_toggle":
                notes.extend(_stack_warnings(name, ctx, scope, agent_id))
            _settle_base_strip()          # 老文件基底 1 帧修正：先于该工具的任何采样
        _LAST_TOOL = name
        _STATUS["last_tool"] = name
        _IN_TOOL = True
        agent_ops.CURRENT_OWNER = (agent_id if scope is not None and not dry
                                   else None)
        try:
            result = tool(ctx, **args)
        finally:
            _IN_TOOL = False
            agent_ops.CURRENT_OWNER = None
        if scope is not None and not (isinstance(result, dict)
                                      and result.get("dry_run")):
            _note_write(agent_id, scope, name)
            written = (result.get("id") if isinstance(result, dict) else None) or args.get("op_id")
            _track_read_dependencies(ctx, scope, written, agent_id, name)
        # op dicts (from agent_ops) get the op envelope; _env-shaped dicts
        # pass through; anything else lands under data verbatim
        if isinstance(result, dict) and "summary" in result:
            payload = result
        elif isinstance(result, dict) and ("strip" in result
                                           or "op" in result
                                           or "id" in result):
            payload = _op_envelope(result, name)
        else:
            payload = {"summary": f"{name} 完成", "data": result,
                       "warnings": [], "truncated": False, "hint": ""}
        if notes:
            payload = dict(payload)
            payload["warnings"] = list(payload.get("warnings") or []) + notes
        if agent_id and payload.get("hint", "").startswith("看视口"):
            payload = dict(payload)
            payload["hint"] = ("用实时类读工具复测（probe_anatomy/analyze_motion/"
                               "compare_motion/chain_lag/slide_report）→ list_ops "
                               "owner=你 compact=true 自查 → save。commit 只有用户能做")
        return {"ok": True, "tool": name, "version": _DATA_VERSION,
                **payload}
    except agent_query.AgentQueryError as exc:
        _STATUS["last_error"] = f"{name}: {exc}"
        return {"ok": False, "tool": name, "version": _DATA_VERSION,
                "summary": "", "data": None, "warnings": [],
                "truncated": False, "hint": "",
                "error": {"code": exc.code, "message": str(exc),
                          "fix": exc.fix}}
    except Exception as exc:  # noqa: BLE001 - relay everything
        _STATUS["last_error"] = f"{name}: {exc}"
        msg = repr(exc)
        code = "E_SCOPE" if "越界" in msg else "E_TOOL"
        return {"ok": False, "tool": name, "version": _DATA_VERSION,
                "summary": "", "data": None, "warnings": [],
                "truncated": False, "hint": "",
                "error": {"code": code, "message": msg,
                          "fix": "message 看不懂就翻 trace 定位行号"},
                "trace": traceback.format_exc(limit=4)}


# ---------------------------------------------------------------------------
# server + main-thread pump

def _pump():
    """Runs on the main thread via bpy.app.timers; drains the request queue.

    Bare ``except Exception`` everywhere: a raised exception in a timer
    callback silently unregisters it (that's the "server died" bug) - every
    failure path must return the interval so the pump keeps ticking.
    """
    global _LAST_PUMP, _PUMP_BUSY
    _LAST_PUMP = time.time()
    _PUMP_BUSY = False
    try:
        while True:
            try:
                sock_file, request = _requests.get_nowait()
            except queue.Empty:
                break
            _PUMP_BUSY = True
            try:
                response = _dispatch(request)
                response["id"] = request.get("id")
                sock_file.write(
                    (json.dumps(response, ensure_ascii=False)
                     + "\n").encode("utf-8"))
                sock_file.flush()
            except Exception as exc:  # noqa: BLE001
                _STATUS["last_error"] = f"pump: {exc!r}"
    except Exception as exc:  # noqa: BLE001 - never propagate out of a timer
        _STATUS["last_error"] = f"pump-loop: {exc!r}"
    if not _running:
        return None
    # 刚处理过请求 → 很快再看一眼（agent 常连发），空闲才回到常规间隔
    return BUSY_INTERVAL if _PUMP_BUSY else TIMER_INTERVAL


def headless_pump_loop(until=None):
    """Background-mode main loop (tools/headless_server.py): sleep until the
    socket thread enqueues a request, then drain immediately.  Replaces the
    fixed 70 ms poll, so a tool call no longer waits up to a full tick before
    it even starts.  ``until()`` lets tests stop the loop."""
    while True:
        _WAKE.wait(TIMER_INTERVAL)
        _WAKE.clear()
        _pump()
        if until is not None and until():
            return


def _watchdog():
    """If the pump timer vanished while the server is running, re-register it.
    Also stamps _LAST_PUMP so stale-pump diagnostics are visible in status()."""
    if _running and not bpy.app.timers.is_registered(_pump):
        _STATUS["last_error"] = "pump 掉线，watchdog 重启"
        try:
            bpy.app.timers.register(_pump, first_interval=TIMER_INTERVAL)
        except Exception as exc:  # noqa: BLE001
            _STATUS["last_error"] = f"watchdog 失败: {exc!r}"
    return WATCHDOG_INTERVAL if _running else None


# A web page can POST to 127.0.0.1:6207 (text/plain needs no CORS preflight):
# the body line would be executed - save to any path, eval_bpy.  The JSON-lines
# protocol never starts with an HTTP request line, so such a connection is
# dropped before its body is read.  An optional shared token (off by default,
# protocol unchanged): set MCD_AGENT_TOKEN for Blender and the clients; every
# request line must then carry "token".
_HTTP_REQUEST_LINE = re.compile(rb"^[A-Z]{3,10} \S+ HTTP/\d")
TOKEN_ENV = "MCD_AGENT_TOKEN"


def _looks_like_http(line: bytes) -> bool:
    return bool(_HTTP_REQUEST_LINE.match(line.lstrip(b"\xef\xbb\xbf")))


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        _STATUS["clients"] += 1
        token = os.environ.get(TOKEN_ENV) or None
        try:
            first = True
            for line in self.rfile:
                if first:
                    first = False
                    if _looks_like_http(line):
                        _STATUS["last_error"] = "拒绝了一个 HTTP 请求（不是 JSON-lines 客户端）"
                        return
                try:
                    request = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(request, dict):
                    continue
                if token is not None and request.pop("token", None) != token:
                    self.wfile.write((json.dumps({
                        "id": request.get("id"), "ok": False, "tool": request.get("tool"),
                        "error": {"code": "E_AUTH", "message": "token 不对或缺失",
                                  "fix": f"请求里带 \"token\"（与 Blender 的 {TOKEN_ENV} 相同）"}},
                        ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()
                    continue
                _requests.put((self.wfile, request))
                _WAKE.set()
        finally:
            _STATUS["clients"] -= 1


def _bind_flags(platform=None) -> dict:
    """Windows: SO_REUSEADDR (socketserver sets it when allow_reuse_address)
    lets a SECOND process bind the same port - two Blenders both "start" the
    server and requests land on either one, i.e. in the wrong file.  Windows'
    exclusive bind is SO_EXCLUSIVEADDRUSE.  POSIX SO_REUSEADDR only skips
    TIME_WAIT and never allows two listeners, so it stays."""
    win = str(platform if platform is not None else sys.platform).startswith("win")
    return {"reuse_address": not win, "exclusive": win}


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = _bind_flags()["reuse_address"]
    daemon_threads = True

    def server_bind(self):
        if _bind_flags()["exclusive"] and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


_server = None


def start_server(port: int = PORT) -> dict:
    _STATUS["last_error"] = ""
    global _running, _server
    if _running:
        return {"running": True, "port": _server.server_address[1]}
    _server = _Server((HOST, int(port)), _Handler)
    threading.Thread(target=_server.serve_forever, daemon=True).start()
    _running = True
    _settle_base_strip()
    if not bpy.app.timers.is_registered(_pump):
        bpy.app.timers.register(_pump, first_interval=TIMER_INTERVAL)
    if not bpy.app.timers.is_registered(_watchdog):
        bpy.app.timers.register(_watchdog, first_interval=WATCHDOG_INTERVAL)
    if _bump_version not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_bump_version)
    return {"running": True, "port": _server.server_address[1]}


def _settle_base_strip():
    """旧文件的 mcd_base 有"基底滞后 1 帧"修正（agent_ops.ensure_base_on_nla 里）。
    以前它在**第一次写入**时才发生——多 agent 时，谁先写谁就把整条基底挪了 1 帧，
    其它 agent 写前量的"修前"数字跟着悄悄变（实测：右脚某段漂移 158.7→70.8 mm，
    没人碰过脚）。服务启动时就做掉，并记一次外部改动，让所有读都从同一基底开始。"""
    global _DATA_VERSION
    try:
        scene, settings = _settings()
        rig = _rig_armature(settings, scene)
        anim = getattr(rig, "animation_data", None) if rig is not None else None
        if anim is None or anim.action is not None:
            return
        for tr in anim.nla_tracks:
            if tr.name == agent_ops.BASE_TRACK and tr.strips:
                st = tr.strips[0]
                if agent_ops.settle_base_strip(rig):
                    _DATA_VERSION += 1
                    _JOURNAL.note(_DATA_VERSION, None, agent_claims.ALL, None,
                                  "base_frame_fix")
                    _STATUS["base_frame_fixed"] = True
                    _retire_lagged_snapshot(rig, scene, settings)
                break
    except Exception as exc:  # noqa: BLE001 - never block the server start
        _STATUS["last_error"] = f"base settle: {exc!r}"


def _retire_lagged_snapshot(rig, scene, settings):
    """基底刚被挪了 1 帧：之前按滞后基底烘的快照（磁盘缓存 + 内存 store）整体错一帧
    （实测验证副本：没人碰过的脚，快照与当前姿态也差 21–76 mm；fix_ground/describe/
    validate 都按它算）。缓存改名 *.lagged 作废（不删），内存 store 重建。
    修正只会在一个文件上发生一次（存盘后 gap=0），所以这里也只走一次。"""
    global _STORE_EPOCH
    _STORE_EPOCH += 1
    data_dir = _data_dir(settings)
    if not data_dir:
        return
    start = int(getattr(settings, "mocap_frame_start", 0) or 0) or scene.frame_start
    end = int(getattr(settings, "mocap_frame_end", 0) or 0) or scene.frame_end
    path = agent_bake.bake_cache_path(data_dir, rig.name, start, end, tag="rig")
    if path.exists():
        path.rename(path.with_suffix(".lagged"))
        _STATUS["lagged_snapshot_retired"] = str(path)


def stop_server() -> dict:
    global _running, _server
    _running = False
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        _server = None
    if bpy.app.timers.is_registered(_pump):
        bpy.app.timers.unregister(_pump)
    if bpy.app.timers.is_registered(_watchdog):
        bpy.app.timers.unregister(_watchdog)
    if _bump_version in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_bump_version)
    return {"running": False}


def is_running() -> bool:
    return _running


def ensure_layout(settings=None, scene=None) -> dict:
    """One-shot housekeeping: migrate the legacy shared tracks to per-op
    tracks (idempotent) and keep the fix list in sync.

    MUST NOT be called from panel draw - renaming NLA tracks writes ID data,
    which is forbidden in the draw context ("Writing to ID classes in this
    context is not allowed").  The fixlist timer calls this instead."""
    scene = scene or bpy.context.scene
    settings = settings or scene.mocap_doctor
    rig = _rig_armature(settings, scene)
    if rig is None:
        return {"migrated": 0, "rows": 0, "rig": None}
    mig = agent_ops.migrate_legacy_tracks(rig, _data_dir(settings) or None)
    if mig.get("moved"):
        _bump_ops_rev_from(settings)
    rows = sync_fixes_list(settings, scene)
    _sync_params_list(settings)
    return {"migrated": mig.get("moved", 0), "rows": rows, "rig": rig.name}


FIXLIST_INTERVAL = 1.0
_LAST_FIXLIST = 0.0          # wall-clock of last _fixlist_tick
_FIXLIST_SYNCED = None       # (scene, rig, data dir) the empty list was last synced for

# ---- 参数控件防抖重写 + 方向空物体监视 ------------------------------------
# 修复条目的参数控件（properties.MD_PG_AgentParam）不直接调 reapply：
# 滑块拖动一帧一个 update，逐个写会堆 strip/卡死。这里统一排队，静默
# PARAM_DEBOUNCE 秒后一次性同轨重写（删旧写新同名，op_id 不变）。
# dir_object 绑定的空物体被拖动/k动画时同理排队刷新（key="__refresh__"）。
PARAM_TICK = 0.2
PARAM_DEBOUNCE = 0.25
_PARAM_PENDING: dict = {}      # (op_id, key) -> value
_PARAM_LAST_EDIT = 0.0
_EMPTY_WATCH: dict = {}        # op_id -> 方向物体 matrix 签名


def _dir_object_signature(obj) -> tuple:
    """What the user can EDIT on a direction object - not where it happens to be
    at the current frame.  The old signature was the current-frame matrix_world:
    an animated or bone-parented arrow (both documented) changed it on every
    scrub / playback stop and re-ran the whole op (75 f ≈ 0.3 s, 300 f ≈ 1 s)
    for an identical result.  Parent + parent inverse + constraints + (keyed:
    the action's keys | static: matrix_basis)."""
    import hashlib
    parts = [obj.parent.name if obj.parent is not None else "",
             str(obj.parent_type), str(getattr(obj, "parent_bone", "") or "")]
    parts += [round(float(v), 5) for row in obj.matrix_parent_inverse for v in row]
    for con in getattr(obj, "constraints", ()):
        parts += [con.type, con.name, bool(con.mute), round(float(con.influence), 5),
                  getattr(getattr(con, "target", None), "name", ""),
                  str(getattr(con, "subtarget", "") or "")]
    anim = getattr(obj, "animation_data", None)
    act = getattr(anim, "action", None) if anim is not None else None
    if act is not None:
        h = hashlib.sha1()
        for fc in act.fcurves:
            n = len(fc.keyframe_points)
            buf = np.empty(2 * n, dtype=np.float32)
            h.update(f"{fc.data_path}[{fc.array_index}]:{n}:{fc.mute}".encode())
            for prop in ("co", "handle_left", "handle_right"):
                fc.keyframe_points.foreach_get(prop, buf)
                h.update(buf.tobytes())
            ip = np.empty(n, dtype=np.int32)
            fc.keyframe_points.foreach_get("interpolation", ip)
            h.update(ip.tobytes())
        parts.append(h.hexdigest())
        # channels the action does not drive are static edits: keep them
        keyed = {(fc.data_path, fc.array_index) for fc in act.fcurves}
        parts.append(str(obj.rotation_mode))
        for prop in ("location", "rotation_euler", "rotation_quaternion",
                     "rotation_axis_angle", "scale"):
            for i, v in enumerate(getattr(obj, prop)):
                if (prop, i) not in keyed:
                    parts.append(round(float(v), 5))
    else:
        parts += [round(float(v), 5) for row in obj.matrix_basis for v in row]
    return tuple(parts)


def schedule_param_apply(op_id, key, value):
    """参数控件/监视器的唯一入口：排队 + 记时间戳 + 拉起计时器。"""
    global _PARAM_LAST_EDIT
    _PARAM_PENDING[(str(op_id), str(key))] = value
    _PARAM_LAST_EDIT = time.time()
    try:
        if not bpy.app.timers.is_registered(_param_tick):
            bpy.app.timers.register(_param_tick, first_interval=PARAM_TICK)
    except Exception:
        pass


def _param_tick():
    """防抖排水 + dir_object 空物体变换监视。"""
    global _PARAM_LAST_EDIT
    try:
        scene = bpy.context.scene
        settings = getattr(scene, "mocap_doctor", None)
        if settings is None or not getattr(settings, "initialized", False):
            return PARAM_TICK
        data_dir = _data_dir(settings)
        rig = _rig_armature(settings, scene)
        if rig is None:
            return PARAM_TICK
        # 方向物体监视：绑定的空物体矩阵变了 → 排队重写（无 overrides，
        # reapply 用 op params 里的 dir_object 重新逐帧取方向）
        try:
            ops = agent_ops.list_ops(data_dir)
        except Exception:
            ops = []
        live = set()
        for op in ops:
            if op.get("status") not in ("preview", "committed"):
                continue
            dob = (op.get("params") or {}).get("dir_object")
            if not dob:
                continue
            live.add(op.get("id"))
            obj = bpy.data.objects.get(str(dob))
            if obj is None:
                continue
            sig = _dir_object_signature(obj)
            prev = _EMPTY_WATCH.get(op.get("id"))
            _EMPTY_WATCH[op["id"]] = sig
            if prev is not None and prev != sig:
                schedule_param_apply(op["id"], "__refresh__", True)
        for dead in set(_EMPTY_WATCH) - live:
            _EMPTY_WATCH.pop(dead, None)
        # 静默期满 → 排水重写
        if _PARAM_PENDING \
                and time.time() - _PARAM_LAST_EDIT >= PARAM_DEBOUNCE:
            pending = dict(_PARAM_PENDING)
            _PARAM_PENDING.clear()
            applied = False
            for (op_id, key), value in pending.items():
                try:
                    overrides = {} if key == "__refresh__" else {key: value}
                    agent_ops.reapply(data_dir, rig, op_id,
                                      scene=scene, **overrides)
                    applied = True
                except Exception as exc:
                    _STATUS["last_error"] = f"参数 {key}: {exc!r}"
            if applied:
                _bump_ops_rev_from(settings)
                _sync_params_list(settings, data_dir)
                _redraw()
    except Exception as exc:  # noqa: BLE001 - a timer must never die
        _STATUS["last_error"] = f"paramtick: {exc!r}"
    return PARAM_TICK


def _sync_params_list(settings, data_dir=None):
    """把每个 op 的可调参数镜像进 settings.agent_params（面板只读这里）。

    key = "op_id::param"；pending 中的项跳过（不覆盖用户正在拖的值）。"""
    from .. import properties as _props

    data_dir = data_dir or _data_dir(settings)
    try:
        ops = agent_ops.list_ops(data_dir)
    except Exception:
        return
    want = {}
    for op in ops:
        specs = agent_ops.TUNABLE_PARAMS.get(op.get("tool"))
        if not specs or not op.get("id"):
            continue
        params = op.get("params") or {}
        for spec in specs:
            when = spec.get("when")      # 条件暴露：{"param": 允许值|True}
            if when and not all(
                    (params.get(wk) in wv
                     if isinstance(wv, list)
                     else (bool(params.get(wk)) if wv is True
                          else params.get(wk) == wv))
                    for wk, wv in when.items()):
                continue
            k = spec["key"]
            v = params.get(k)
            if k == "frame_range" and v is None:
                v = op.get("frames")
            want[f"{op['id']}::{k}"] = (op["id"], spec, v)
    coll = settings.agent_params
    existing = {it.pkey: it for it in coll}
    for i in range(len(coll) - 1, -1, -1):
        if coll[i].pkey not in want:
            coll.remove(i)
    for pkey, (op_id, spec, v) in want.items():
        it = existing.get(pkey)
        if it is None:
            it = coll.add()
            it.pkey = pkey
        if (op_id, spec["key"]) in _PARAM_PENDING:
            continue                     # 用户正在拖，别覆盖
        _props.sync_param_item(it, op_id, spec, v)


def _fixlist_tick():
    """Auto-sync the panel's fix list outside the draw context.

    Draw may only read; every mutation (NLA migration, collection rebuild)
    happens here.  Stale = op log revision moved ahead of the list, or the
    list is empty while the rig exists (fresh file)."""
    global _LAST_FIXLIST, _IN_SYNC, _FIXLIST_SYNCED
    _LAST_FIXLIST = time.time()
    try:
        scene = bpy.context.scene
        if scene is None:
            return FIXLIST_INTERVAL
        settings = getattr(scene, "mocap_doctor", None)
        if settings is None or not settings.initialized:
            return FIXLIST_INTERVAL
        rig = _rig_armature(settings, scene)
        key = (scene.as_pointer(), rig.name if rig is not None else None,
               str(_data_dir(settings)))
        # 空列表只在"这个文件/骨架/日志目录还没同步过"时算过期——以前列表为空时
        # 条件恒真，每秒重建一次并写场景属性，每秒把数据版本号 +1（"外部改动"）
        stale = (int(settings.agent_fixes_rev) != int(settings.agent_ops_rev)
                 or (len(settings.agent_fixes) == 0 and rig is not None
                     and key != _FIXLIST_SYNCED))
        if stale:
            # 先把还没求值的改动（用户刚做的编辑）在标志外求值掉，照常记版本
            bpy.context.view_layer.update()
            # 镜像场景状态不是编辑：期间（含把这些写入求值掉的 update）不记版本
            _IN_SYNC = True
            try:
                ensure_layout(settings, scene)
                _FIXLIST_SYNCED = key
                bpy.context.view_layer.update()
            finally:
                _IN_SYNC = False
    except Exception as exc:  # noqa: BLE001 - a timer must never die
        _STATUS["last_error"] = f"fixlist: {exc!r}"
    return FIXLIST_INTERVAL


def _on_file_loaded(*_args):
    """A file load can outlive the startup registration - re-arm the timer."""
    global _STORE, _STORE_EPOCH, _FIXLIST_SYNCED
    _PARAM_PENDING.clear()
    _EMPTY_WATCH.clear()       # 新文件里旧签名无意义
    _FIXLIST_SYNCED = None
    # 内存里的信号库属于上一个文件（恢复检查点 = 重新打开文件）：作废，下次读
    # 按新文件的基底指纹决定用缓存还是重烘
    _STORE = None
    _STORE_EPOCH += 1
    try:
        agent_anatomy.reset_caches()     # 掌心/铰链标定是"这个文件、这套骨架"的（审查 M21）
        from . import agent_view
        agent_view._REST_SIGN.clear()
    except Exception:
        pass
    try:
        _data_dir(_settings()[1])   # 文件挪机器后先自愈日志路径
    except Exception:
        pass
    try:
        start_fixlist_timer()
    except Exception:
        pass


def start_fixlist_timer():
    # timer first: if the handler append below fails (odd context), the timer
    # itself is already armed
    if not bpy.app.timers.is_registered(_fixlist_tick):
        bpy.app.timers.register(_fixlist_tick, first_interval=FIXLIST_INTERVAL)
    if not bpy.app.timers.is_registered(_param_tick):
        bpy.app.timers.register(_param_tick, first_interval=PARAM_TICK)
    if _on_file_loaded not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_file_loaded)


def stop_fixlist_timer():
    if bpy.app.timers.is_registered(_fixlist_tick):
        bpy.app.timers.unregister(_fixlist_tick)
    if bpy.app.timers.is_registered(_param_tick):
        bpy.app.timers.unregister(_param_tick)
    if _on_file_loaded in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_file_loaded)


def _bump_ops_rev_from(settings):
    try:
        settings.agent_ops_rev = int(settings.agent_ops_rev) + 1
    except Exception:
        pass


def _same_float(a, b) -> bool:
    return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))


def sync_fixes_list(settings, scene=None) -> int:
    """Rebuild the panel's fix list from op log + live NLA state.

    Values already shown are preserved by op_id, so a rebuild never fights the
    slider the user is dragging (drags do not touch agent_ops_rev).  It mirrors
    the scene - it must not edit it: an unchanged list is not rewritten, and
    力度/静音 are written quietly (their update callbacks re-write the strip
    and frame_set, which the server counted as an external edit → E_STALE for
    every other agent, once per second while the list was empty)."""
    scene = scene or bpy.context.scene
    rig = _rig_armature(settings, scene)
    if rig is None:
        return 0
    from .. import properties as _props
    rows = agent_ops.reconcile(rig, _data_dir(settings))
    # owner / 源已变（复制类）只在 op 日志里——面板标签里也带上，GUI 里一眼看出是谁写的、
    # 哪条复制需要 reapply。没有 owner 的（用户自己的修复）标签不变。
    try:
        log = {o.get("id"): o for o in agent_ops.list_ops(_data_dir(settings))}
    except Exception:  # noqa: BLE001 - 面板刷新绝不因日志读不了而失败
        log = {}
    coll = settings.agent_fixes
    keep = {item.op_id: (item.exponent, item.muted, item.selected)
            for item in coll}
    want = []
    for row in rows:
        op_id = row["op_id"] or ""
        label = (row.get("label")
                 or (f"{row['frames'][0]}-{row['frames'][1]} {row['tool']}"
                     if row.get("frames")
                     else f"{row['tool']} {row['strip']}"))
        o = log.get(op_id) or {}
        # 名字不是一列：owner 以前拼进这一行的 label 尾巴（"… · w1-jump"），
        # 用户看到的只是一串陌生 ID。只留真正需要动作的 tag（源已变/reapply）。
        tags = (["源已变,需reapply"] if o.get("stale") else [])
        if tags:
            label = f"{label} · {' · '.join(tags)}"
        prev = keep.get(op_id)
        if prev is not None and not row.get("alive"):
            exponent, muted = prev[0], prev[1]   # 丢失行：保住调过的值
        else:
            exponent = float(row.get("exponent", 1.0))   # 0 = 力度静音，别显示成 1
            muted = bool(row.get("muted"))
        want.append({
            "op_id": op_id, "label": label,
            "strip": row.get("strip") or "", "track": row.get("track") or "",
            "status": row.get("status") or "", "alive": bool(row.get("alive")),
            "frames": (f"{row['frames'][0]}-{row['frames'][1]}"
                       if row.get("frames") else ""),
            "exponent": exponent, "muted": muted,
            "selected": bool(prev[2]) if prev is not None else False,
        })

    def _same_row(item, w):
        return (item.op_id == w["op_id"] and item.label == w["label"]
                and item.strip == w["strip"] and item.track == w["track"]
                and item.status == w["status"] and bool(item.alive) == w["alive"]
                and item.frames == w["frames"] and bool(item.muted) == w["muted"]
                and bool(item.selected) == w["selected"]
                and _same_float(item.exponent, w["exponent"]))

    if len(coll) != len(want) or not all(_same_row(i, w) for i, w in zip(coll, want)):
        active_id = (coll[settings.agent_fix_index].op_id
                     if 0 <= settings.agent_fix_index < len(coll) else "")
        coll.clear()
        for w in want:
            item = coll.add()
            for name in ("op_id", "label", "strip", "track", "status", "alive",
                         "frames"):
                setattr(item, name, w[name])
            _props.set_quietly(item, "exponent", w["exponent"])
            _props.set_quietly(item, "muted", w["muted"])
            _props.set_quietly(item, "selected", w["selected"])
        if active_id:
            for i, item in enumerate(coll):
                if item.op_id == active_id:
                    _props.set_quietly(settings, "agent_fix_index", i)
                    break
    if int(settings.agent_fixes_rev) != int(settings.agent_ops_rev):
        settings.agent_fixes_rev = int(settings.agent_ops_rev)
    if settings.agent_fix_index >= len(coll):
        _props.set_quietly(settings, "agent_fix_index", max(0, len(coll) - 1))
    return len(coll)


def fixes_snapshot(settings, scene=None) -> list:
    """Read-only rows for the panel (no rebuild)."""
    scene = scene or bpy.context.scene
    rig = _rig_armature(settings, scene)
    if rig is None:
        return []
    return agent_ops.reconcile(rig, _data_dir(settings))


def status() -> dict:
    pump_age = round(time.time() - _LAST_PUMP, 1) if _LAST_PUMP else None
    fixlist_age = round(time.time() - _LAST_FIXLIST, 1) if _LAST_FIXLIST else None
    return {"running": _running, "version": _DATA_VERSION,
            "clients": _STATUS["clients"], "last_tool": _STATUS["last_tool"],
            "last_error": _STATUS["last_error"],
            "pump_age_s": pump_age,
            "pump_registered": bpy.app.timers.is_registered(_pump),
            "fixlist_registered": bpy.app.timers.is_registered(_fixlist_tick),
            "fixlist_age_s": fixlist_age,
            "port": _server.server_address[1] if _server else PORT}
