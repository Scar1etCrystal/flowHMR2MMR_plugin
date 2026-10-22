import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import bpy
from bpy.app.handlers import persistent

from .core import animation as core_animation
from .presets import DEFAULTS, EXPECTED_FPS, OBJECT_NAMES, fixed_object_name_matches
from .workflow import STEP_INDEX, STEPS


_ACTION_PREVIEWS = {}
_RESTORE_WORK_PATH = None
_RESTORE_RESUME_STEP_ID = ""
_RESTORE_RESET_STEP_ID = ""
_RESTORE_MESSAGE = ""

# Bump when saved projects need their workflow state adjusted on load; add a
# migration branch in _migrate_project_schema for each previous version.
SCHEMA_VERSION = 10

# 工作副本文件名标记：导入时如果没有项目，就按这个后缀自动建一份工作副本。
WORK_COPY_MARK = "_mocap_doctor_work"

# v6 wizard layout (indices), kept only so a saved current_step can be remapped:
# 0 project / 1 source_check / 2 source_floor / 3 contacts / 4 retarget /
# 5 global_correction / 6 tilt / 7 ground_feet / 8 target_floor / 9 foot_lock /
# 10 mmd_bake / 11 export_prep / 12 export.
# Each old index maps onto the retained step that now covers that point of the
# pipeline (a step that left the UI hands over to the retained step after it).
RETIRED_V6_STEP_MAP = (0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7)

# v6 index of "source_floor", used by the pre-v6 clamp below (its step id left
# the wizard, so the name can no longer be looked up in STEP_INDEX).
RETIRED_V6_SOURCE_FLOOR_INDEX = 2

# Index of the first surviving step after the retired hand chain: the
# source_analyze / hand_ranges / hand_repair block (old indices 2, 3, 4) moved
# to a pre-import pkl tool, so every step after it shifted down by three.
RETIRED_HAND_CHAIN_INDEX = 5
RETIRED_HAND_STEP_IDS = ("source_analyze", "hand_ranges", "hand_repair")

# Where the retired "fingers" step used to sit: between mmd_bake and
# export_prep.  Everything after it shifted down by one when it was removed.
RETIRED_FINGERS_INDEX = 14

# Where the retired "smooth" step used to sit: after the old hand chain,
# before source_floor.  GVHMR pre-filters its own output, so the extra Gaussian
# pass only removed the last of the snap; everything after it shifted down.
RETIRED_SMOOTH_INDEX = 5

# The "ground_feet" step (pin the planted foot back onto the floor) was inserted
# right after foot_lock, so mmd_bake and everything after it shifted up by one.
GROUND_FEET_INDEX = 12


@persistent
def _migrate_project_schema(_dummy):
    try:
        scene = getattr(bpy.context, 'scene', None)
        if scene is None or not hasattr(scene, "mocap_doctor"):
            return
        settings = scene.mocap_doctor
        if not settings.initialized or settings.schema_version >= SCHEMA_VERSION:
            return
        # 0 -> 1: the "fingers" step was inserted between mmd_bake and
        # export_prep, so files saved past that point need current_step + 1.
        fingers_index = STEP_INDEX.get("fingers", RETIRED_FINGERS_INDEX)
        if settings.schema_version < 1 and settings.current_step >= fingers_index:
            settings.current_step += 1
        # 1 -> 2: the FreeMoCap "source_bake" step became the frontend-agnostic
        # "source_check". The step keeps its index, so only the orphaned record
        # id needs renaming; a fresh record would otherwise be created next to it.
        if settings.schema_version < 2:
            for record in getattr(settings, "steps", ()):
                if record.step_id == "source_bake":
                    record.step_id = "source_check"
        # 2 -> 3: the "fingers" step (pose one hand shape, copy it to every
        # frame) is gone.  HaMeR's hands are good on their own, and flattening
        # them to a single pose measured worse than leaving them alone.  Every
        # step after it shifted down by one, and its record is now an orphan.
        if settings.schema_version < 3:
            if settings.current_step > RETIRED_FINGERS_INDEX:
                settings.current_step -= 1
            for index in range(len(settings.steps) - 1, -1, -1):
                if settings.steps[index].step_id == "fingers":
                    settings.steps.remove(index)
        # 3 -> 4: the "smooth" step is gone.  GVHMR already filters its own
        # output, and a take that never ran step 6 still came out visibly
        # smooth - the extra pass only ate the remaining sharpness.  Every
        # step after it shifted down by one, and its record is now an orphan.
        # The same version adds "ground_feet" right after foot_lock, which
        # shifts mmd_bake and everything after it back up by one.
        if settings.schema_version < 4:
            if settings.current_step > RETIRED_SMOOTH_INDEX:
                settings.current_step -= 1
            for index in range(len(settings.steps) - 1, -1, -1):
                if settings.steps[index].step_id == "smooth":
                    settings.steps.remove(index)
            if settings.current_step >= GROUND_FEET_INDEX:
                settings.current_step += 1
        # 4 -> 5: "ground_feet" moved ahead of "target_floor", because the mesh
        # floor repair is only meaningful once the model has been grounded.
        # Old 10 target_floor -> 11, old 11 foot_lock -> 12, old 12 ground_feet
        # -> 10; step records are keyed by id and need no remapping.
        if settings.schema_version < 5:
            if settings.current_step == 10:
                settings.current_step = 11
            elif settings.current_step == 11:
                settings.current_step = 12
            elif settings.current_step == 12:
                settings.current_step = 10
            # The same version zeroes the ground heights (0.0257/0.02 were
            # hand fitted to old FreeMoCap output) and re-tunes the mesh floor
            # repair for its new role, which runs after the pin and must track
            # it closely.  Blender keeps property values inside the .blend, so
            # a saved project would otherwise keep the stale numbers.
            settings.source_floor_z = DEFAULTS["source_floor_z"]
            settings.target_floor_z = DEFAULTS["target_floor_z"]
            settings.vmd_floor_offset = DEFAULTS["vmd_floor_offset"]
            settings.target_floor_strength = DEFAULTS["target_floor_strength"]
            settings.target_floor_smooth_radius = DEFAULTS["target_floor_smooth_radius"]
            settings.target_floor_max_delta = DEFAULTS["target_floor_max_delta"]
        # 5 -> 6: the hand bad-range chain left the wizard.  It now runs on the
        # pkl before PoseCapture import (the annotator is reused there), so the
        # three retired indices collapse and their step records are orphans.
        if settings.schema_version < 6:
            if settings.current_step >= RETIRED_HAND_CHAIN_INDEX:
                settings.current_step -= 3
            else:
                # "source_floor" 在 v6 的索引空间里是 2；它已经从 STEPS 里退休，
                # 这里只能用当时的字面量（迁移链接下来还会把索引重映射到 v7）。
                settings.current_step = min(settings.current_step, RETIRED_V6_SOURCE_FLOOR_INDEX)
            for index in range(len(settings.steps) - 1, -1, -1):
                if settings.steps[index].step_id in RETIRED_HAND_STEP_IDS:
                    settings.steps.remove(index)
        # 6 -> 7: the wizard was trimmed to the retained steps (the user's
        # 2026-10 大改).  Retired steps keep their code but leave STEPS, so a
        # saved current_step has to be remapped by the old index, not by id:
        # the record for the open step stays valid, only the pointer moves.
        if settings.schema_version < 7:
            old = int(settings.current_step)
            if 0 <= old < len(RETIRED_V6_STEP_MAP):
                settings.current_step = RETIRED_V6_STEP_MAP[old]
            settings.current_step = max(0, min(int(settings.current_step), len(STEPS) - 1))
        if settings.schema_version < 8:
            old_ids = (
                ('project', 'import_official', 'retarget', 'ground_feet', 'target_floor', 'mmd_bake', 'export_prep', 'export')
                if settings.schema_version < 7 else
                ('project', 'import_official', 'retarget', 'target_skeleton_fix', 'foot_spacing',
                 'ground_feet', 'target_floor', 'mmd_bake', 'export_prep', 'export')
            )
            old_index = max(0, min(int(settings.current_step), len(old_ids) - 1))
            step_id = old_ids[old_index]
            if step_id == 'target_skeleton_fix':
                step_id = 'contacts'
            pending = settings.preview_step_id
            if pending in STEP_INDEX:
                step_id = pending
            settings.current_step = STEP_INDEX.get(step_id, 0)
        # 8 -> 9: the "创建工作副本" page is gone; the working copy is created
        # automatically on the first import.  Every page after it shifts down by
        # one, and the old project page collapses onto the new first page.
        if settings.schema_version < 9:
            pending = settings.preview_step_id
            if pending in STEP_INDEX:
                settings.current_step = STEP_INDEX[pending]
            else:
                settings.current_step = max(0, int(settings.current_step) - 1)
            settings.current_step = max(0, min(int(settings.current_step), len(STEPS) - 1))
        # 9 -> 10: the foot XY lock ("脚滑修复") is back, right after 贴地锁定.
        # Every page from 模型穿地修复 onwards shifts up by one.
        if settings.schema_version < 10:
            pending = settings.preview_step_id
            if pending in STEP_INDEX:
                settings.current_step = STEP_INDEX[pending]
            elif int(settings.current_step) >= STEP_INDEX["target_floor"] - 1:
                # -1: the threshold is compared in the previous (pre-insert) index space.
                settings.current_step = int(settings.current_step) + 1
            settings.current_step = max(0, min(int(settings.current_step), len(STEPS) - 1))
        settings.schema_version = SCHEMA_VERSION
    except Exception as exc:
        print(f"[MoCap Doctor] schema migration failed: {exc}")


def _migrate_open_project_once():
    _migrate_project_schema(None)
    _recover_on_load(None)
    return None


def recover_stuck_ui(scene):
    """Unlock a file that was saved mid-step.

    ``busy`` and ``preview_step_id`` are saved inside the .blend, so a crash or
    an error mid-step leaves them set and the whole panel greyed out on the next
    load.  A preview can never survive a file load (the action-preview state is
    in memory only), so both are stale by definition here.
    """

    settings = scene.mocap_doctor
    changed: list[str] = []
    if settings.busy:
        settings.busy = False
        changed.append('busy')
    if settings.preview_step_id:
        stale = settings.preview_step_id
        settings.preview_step_id = ""
        settings.preview_owner_name = ""
        settings.preview_base_action = ""
        settings.preview_action = ""
        settings.preview_restore_checkpoint = ""
        record = find_step_record(settings, stale, create=False)
        if record is not None and record.status in {"PREVIEW", "FAILED"}:
            record.status = "PENDING"
            record.message = ""
        changed.append('preview')
    if changed:
        save_manifest(scene)
    return changed


def register_handlers():
    if _migrate_project_schema not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_migrate_project_schema)
    if _recover_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_recover_on_load)
    if getattr(bpy.context, 'scene', None) is not None:
        _migrate_project_schema(None)
        _recover_on_load(None)
    elif not bpy.app.timers.is_registered(_migrate_open_project_once):
        bpy.app.timers.register(_migrate_open_project_once, first_interval=0.1)


@persistent
def _recover_on_load(_dummy):
    try:
        scene = getattr(bpy.context, 'scene', None)
        if scene is None or not hasattr(scene, 'mocap_doctor'):
            return
        if not scene.mocap_doctor.initialized:
            return
        recover_stuck_ui(scene)
    except Exception as exc:  # noqa: BLE001 - 解锁失败不能挡住加载
        print(f'[MoCap Doctor] unlock failed: {exc}')



def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value):
    return re.sub(r"[^\w\-.]+", "_", value, flags=re.UNICODE).strip("._") or "mocap_project"


def ensure_fps(scene):
    changed = scene.render.fps != EXPECTED_FPS or scene.render.fps_base != 1.0
    scene.render.fps = EXPECTED_FPS
    scene.render.fps_base = 1.0
    return changed


def sync_scene_range(scene, settings):
    settings.mocap_frame_start = int(scene.frame_start)
    settings.mocap_frame_end = int(scene.frame_end)


def apply_project_range(scene, settings):
    scene.frame_start = int(settings.mocap_frame_start)
    scene.frame_end = int(settings.mocap_frame_end)


def project_data_dir(work_filepath):
    work = Path(work_filepath)
    return work.parent / ".mocap_doctor" / _safe_name(work.stem)


def _usable_data_dir(path_text) -> bool:
    """存的目录只有"在本系统上是绝对路径且存在"才算数。

    Windows 写进 .blend 的 ``F:/...`` 在 Linux 上是**相对**路径：is_dir() 会按
    当前工作目录解析——只要哪段代码曾按原样 mkdir 过（向导步骤写报告/
    检查点就会），它就"存在"了，自愈从此失效：op 日志去了 CWD 下的怪名
    目录，blend 里的 agent strip 全变成"未登记"孤儿（2026-10-03 在远程
    套件上实测复现）。"""
    if not path_text:
        return False
    p = Path(str(path_text))
    return p.is_absolute() and p.is_dir()


def resolve_work_filepath(settings):
    """settings.work_filepath 和 data_directory 一样是别的机器写进去的绝对路径
    （远程套件里是 ``F:/mocap_ai_doctor/...``）：Linux 上它是相对路径，向导的
    "Planted 检测"步骤存盘直接失败（或者写进 CWD 下的怪名目录）。只有"本系统
    绝对路径且所在目录存在"才用它；否则用当前打开的文件并回写自愈。"""
    stored = getattr(settings, "work_filepath", "") or ""
    if stored:
        p = Path(str(stored))
        if p.is_absolute() and p.parent.is_dir():
            return stored
    current = bpy.data.filepath or ""
    if current:
        if stored != current:
            try:
                settings.work_filepath = current
            except Exception:
                pass
        return current
    return stored


def resolve_data_dir(settings):
    """settings.data_directory 存的是绝对路径——工作文件挪到别的机器/盘符
    后它就是死路径（Linux 上 ``F:\\...`` 只会变成怪名目录）。还在就用；
    不在就按当前 blend 位置重推并回写自愈。"""
    stored = settings.data_directory or ""
    if _usable_data_dir(stored):
        return stored
    fp = bpy.data.filepath or resolve_work_filepath(settings) or ""
    if fp:
        derived = str(project_data_dir(fp))
        if stored != derived:
            settings.data_directory = derived
        return derived
    return stored or "."


def resolve_project_file(settings, stored, subdir):
    """Step records keep ABSOLUTE checkpoint / report paths.  Once the project
    moved (other drive letter, other machine, the Linux kit) the stored path is
    dead while the file still sits in ``<data dir>/<subdir>/`` under the same
    name - restore said "找不到检查点" and the air-hint reload "还没有检测结果".
    Return the stored path if it is a real file on this system, else that name
    under the current data dir if it exists there, else the stored text (the
    caller reports it missing)."""
    text = str(stored or "")
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute() and path.is_file():
        return text
    name = PureWindowsPath(text).name        # splits on "\\" and "/" alike
    if name:
        candidate = Path(resolve_data_dir(settings)) / subdir / name
        if candidate.is_file():
            return str(candidate)
    return text


def ensure_project_directories(settings):
    root = Path(resolve_data_dir(settings))
    for name in ("checkpoints", "reports", "recovery", "logs", "tmp"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def atomic_write_json(path, data, *, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_report(scene, step_id, data):
    """Write an immutable, sequential JSON artifact inside the project."""

    settings = scene.mocap_doctor
    root = ensure_project_directories(settings) / "reports"
    safe_step = _safe_name(step_id)
    sequence = 0
    while True:
        path = root / f"{sequence:04d}_{safe_step}.json"
        if not path.exists():
            break
        sequence += 1
    atomic_write_json(path, data)
    return path


def save_manifest(scene):
    settings = scene.mocap_doctor
    if not settings.initialized or not settings.data_directory:
        return
    atomic_write_json(
        Path(resolve_data_dir(settings)) / "project.json",
        {
            "schema_version": "mocap_doctor_project_v1",
            "state_authority": "blend_scene",
            "updated_at": _utc_now(),
            "project_uuid": settings.project_uuid,
            "project_name": settings.project_name,
            "original_filepath": settings.original_filepath,
            "work_filepath": settings.work_filepath,
            "expected_fps": EXPECTED_FPS,
            "frame_start": settings.mocap_frame_start,
            "frame_end": settings.mocap_frame_end,
            "current_step": settings.current_step,
            "objects": {
                name: getattr(getattr(settings, name), "name", None)
                for name in ("source_armature", "model_root", "mmr_rig", "mmd_armature", "target_mesh", "correction_empty")
            },
            "annotations": [
                {
                    "uid": item.uid,
                    "channel": item.channel,
                    "frame_start": int(item.frame_start),
                    "frame_end": int(item.frame_end),
                    "source": item.source,
                }
                for item in getattr(scene, "mcd_annotation_ranges", ())
            ],
            "annotation_channels": {
                "FOOT_L_EFFECTIVE": {
                    "initialized": bool(scene.get("mcd_foot_l_effective_initialized", False))
                },
                "FOOT_R_EFFECTIVE": {
                    "initialized": bool(scene.get("mcd_foot_r_effective_initialized", False))
                },
            },
            "steps": [
                {
                    "step_id": item.step_id,
                    "status": item.status,
                    "checkpoint": item.checkpoint,
                    "before_checkpoint": item.before_checkpoint,
                    "artifact_path": item.artifact_path,
                    "parameter_hash": item.parameter_hash,
                    "message": item.message,
                }
                for item in settings.steps
            ],
        },
    )


def find_step_record(settings, step_id, create=True):
    for item in settings.steps:
        if item.step_id == step_id:
            return item
    if not create:
        return None
    item = settings.steps.add()
    item.step_id = step_id
    return item


def record_step(scene, step_id, status, message="", checkpoint="", artifact_path="", parameter_hash=""):
    settings = scene.mocap_doctor
    item = find_step_record(settings, step_id)
    item.status = status
    item.message = message
    if checkpoint:
        item.checkpoint = checkpoint
        settings.last_checkpoint = checkpoint
    if artifact_path:
        item.artifact_path = artifact_path
    if parameter_hash:
        item.parameter_hash = parameter_hash
    save_manifest(scene)
    return item


def mark_downstream_stale(scene, step_id):
    settings = scene.mocap_doctor
    index = STEP_INDEX.get(step_id, -1)
    for item in settings.steps:
        if STEP_INDEX.get(item.step_id, -1) > index and item.status in {"ACCEPTED", "PREVIEW"}:
            item.status = "STALE"
    save_manifest(scene)


def parameter_hash(data):
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def discover_known_objects(scene):
    settings = scene.mocap_doctor
    for attr, name in OBJECT_NAMES.items():
        if hasattr(settings, attr):
            obj = bpy.data.objects.get(name)
            if obj is None:
                candidates = [
                    item
                    for item in bpy.data.objects
                    if fixed_object_name_matches(item.name, name)
                ]
                # Auto-discovery is conservative when multiple imported Teto
                # copies exist; the user can choose the intended one manually.
                obj = candidates[0] if len(candidates) == 1 else None
            if obj is not None:
                setattr(settings, attr, obj)
    return {
        attr: getattr(settings, attr)
        for attr in ("source_armature", "model_root", "mmr_rig", "mmd_armature", "target_mesh", "correction_empty")
    }


def delete_checkpoints(scene):
    """Remove every checkpoint file and forget the recorded paths."""

    settings = scene.mocap_doctor
    root = ensure_project_directories(settings) / "checkpoints"
    removed = 0
    for item in sorted(root.glob("*.blend")):
        try:
            item.unlink()
            removed += 1
        except OSError:
            continue
    for record in settings.steps:
        record.checkpoint = ""
        record.before_checkpoint = ""
    settings.last_checkpoint = ""
    save_manifest(scene)
    return removed


def checkpoint_path(settings, label):
    root = ensure_project_directories(settings)
    sequences = []
    for item in (root / "checkpoints").glob("*.blend"):
        match = re.match(r"^(\d+)_", item.name)
        if match:
            sequences.append(int(match.group(1)))
    sequence = max(sequences, default=-1) + 1
    return root / "checkpoints" / f"{sequence:04d}_{_safe_name(label)}.blend"


def _save_snapshot(scene, path):
    """save_as(copy=True) with busy forced off, so snapshots never lock the UI.

    ``busy`` is true while a step runs; a checkpoint saved then would come back
    with ``busy=True`` after a restore and disable the whole panel.
    """

    settings = scene.mocap_doctor
    previous = bool(settings.busy)
    settings.busy = False
    try:
        return bpy.ops.wm.save_as_mainfile(
            filepath=str(path), copy=True, relative_remap=True)
    finally:
        settings.busy = previous


def create_checkpoint(scene, label):
    settings = scene.mocap_doctor
    if not settings.initialized:
        raise RuntimeError("Project is not initialized")
    path = checkpoint_path(settings, label)
    result = _save_snapshot(scene, path)
    if "FINISHED" not in result:
        raise RuntimeError(f"Unable to create checkpoint: {path}")
    settings.last_checkpoint = str(path)
    return path


def create_accepted_checkpoint(scene, step_id, message="预览已接受", label=None):
    """Persist a checkpoint that already contains its accepted StepRecord."""

    settings = scene.mocap_doctor
    path = checkpoint_path(settings, label or step_id)
    previous_steps = [
        {
            "step_id": item.step_id,
            "status": item.status,
            "checkpoint": item.checkpoint,
            "before_checkpoint": item.before_checkpoint,
            "artifact_path": item.artifact_path,
            "parameter_hash": item.parameter_hash,
            "message": item.message,
        }
        for item in settings.steps
    ]
    previous_last = settings.last_checkpoint
    try:
        record_step(scene, step_id, "ACCEPTED", message, str(path))
        mark_downstream_stale(scene, step_id)
        result = _save_snapshot(scene, path)
        if "FINISHED" not in result:
            raise RuntimeError(f"Unable to create checkpoint: {path}")
    except Exception:
        settings.steps.clear()
        for values in previous_steps:
            item = settings.steps.add()
            for name, value in values.items():
                setattr(item, name, value)
        settings.last_checkpoint = previous_last
        save_manifest(scene)
        raise
    settings.last_checkpoint = str(path)
    return path


def _same_file(a, b):
    try:
        return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(
            os.path.realpath(str(b)))
    except Exception:
        return str(a) == str(b)


def _is_inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def current_file_is_workfile(settings):
    """True only when the OPEN file is the project's recorded working file.

    保存前的最后一道闸。``workfile_save_target`` 会在"记录的工作文件路径为空/
    对不上"时退回"当前打开的文件"——这对用户手动点保存是合理的，但对
    "某一步顺手存一下"是危险的：只要当前打开的是别的 .blend（原始文件、别人的
    文件、从没建过项目的场景），那次自动保存就会原地覆盖它。历史上真的这么毁过
    未保存工作，所以自动保存一律先问这一句，答不上来就不存。
    """

    current = bpy.data.filepath or ""
    stored = getattr(settings, "work_filepath", "") or ""
    if not current or not stored:
        return False
    return _same_file(current, stored)


def workfile_save_target(settings):
    """Where save_workfile may write: the file that is actually open.

    The recorded work_filepath is not "the file being edited": with a
    checkpoint or a copy open, 上一步/下一步 or accepting a step used to save
    THAT content over the real work file (Blender keeps a single .blend1).
    Only restore_checkpoint (via _finish_restore) writes back to the work file.
    Another project file that is open is saved in place; a checkpoint or
    recovery copy is never written (it is a restore point) - RuntimeError."""
    target = resolve_work_filepath(settings)
    current = bpy.data.filepath or ""
    if current and target and not _same_file(current, target):
        data_dir = resolve_data_dir(settings)
        for sub, label in (("checkpoints", "检查点"), ("recovery", "恢复副本")):
            if _is_inside(current, Path(data_dir) / sub):
                raise RuntimeError(
                    f"当前打开的是项目的{label}（{Path(current).name}），不是工作文件"
                    f"（{Path(target).name}）——为了不覆盖工作文件，没有保存。要回到这个"
                    "状态请在工作文件里用「恢复检查点」；要继续工作请打开工作文件")
        return current
    return target


def save_workfile(scene):
    settings = scene.mocap_doctor
    result = bpy.ops.wm.save_as_mainfile(filepath=workfile_save_target(settings),
                                         relative_remap=True)
    if "FINISHED" not in result:
        raise RuntimeError("Unable to save working file")
    save_manifest(scene)


def auto_work_filepath():
    """The file the plugin writes: the open file itself (in-place operation).

    The old design saved a separate ``*_mocap_doctor_work.blend`` copy.  The
    user's decision (2026-10-20) is to work directly on their own file, so this
    now returns the open file and the work-copy step no longer exists.
    """

    return bpy.data.filepath or ""


def prepare_source_file():
    """Make sure the open file exists on disk before we start writing in place.

    Unsaved-but-has-a-path -> save silently in place; never saved -> hand over to
    Blender's own save dialog (no new dialog of our own).
    """

    if not bpy.data.filepath:
        bpy.ops.wm.save_as_mainfile("INVOKE_DEFAULT")
        if not bpy.data.filepath:
            raise RuntimeError("请先保存工程文件")
        return "saved_as"
    if bpy.data.is_dirty:
        bpy.ops.wm.save_mainfile("EXEC_DEFAULT")
        return "saved"
    return "clean"


def backup_original_once(scene):
    """Keep one copy of the file as it was before the plugin ever touched it."""

    settings = scene.mocap_doctor
    current = bpy.data.filepath or ""
    if not current:
        return None
    root = ensure_project_directories(settings) / "recovery"
    target = root / "original.blend"
    if target.is_file():
        return target
    try:
        import shutil
        shutil.copy2(current, target)
        return target
    except OSError:
        return None


def ensure_project(scene, target_template_path=""):
    """Register the open file as the project, on demand.

    The wizard has no "创建工作副本" page: the first import registers the current
    file, creates the project directories and the baseline checkpoint.  Previews,
    checkpoints and restore-before-step all need that project.
    """

    settings = scene.mocap_doctor
    if settings.initialized:
        return None
    work = auto_work_filepath()
    if not work:
        raise RuntimeError("请先保存工程文件")
    return initialize_project(scene, work, target_template_path)


def initialize_project(scene, work_filepath, target_template_path=""):
    """Register the project's working file (in place by default).

    Q8-1（2026-10 大改）：工作副本不出现在 Blender 的 Recent Files 里 —— 放弃，不修。
    2026-10-20：不再创建 ``*_mocap_doctor_work.blend`` 副本；就地写用户自己的文件，
    首次登记时把未被动过的原件备份到 recovery/original.blend。
    """
    settings = scene.mocap_doctor
    original = bpy.data.filepath
    work = str(Path(work_filepath).resolve())
    same_file = bool(original) and Path(original).resolve() == Path(work).resolve()
    if not same_file:
        result = bpy.ops.wm.save_as_mainfile(filepath=work, relative_remap=True)
        if "FINISHED" not in result:
            raise RuntimeError("Unable to create working file")
    settings.initialized = True
    settings.project_uuid = uuid.uuid4().hex
    settings.project_name = Path(work).stem
    settings.original_filepath = original
    settings.work_filepath = work
    settings.data_directory = str(project_data_dir(work))
    settings.target_template_path = target_template_path
    settings.current_step = STEP_INDEX.get("import_official", 0)
    settings.schema_version = SCHEMA_VERSION
    sync_scene_range(scene, settings)
    ensure_fps(scene)
    if settings.tilt_reference_frame == 0:
        settings.tilt_reference_frame = settings.mocap_frame_start
    ensure_project_directories(settings)
    backup_original_once(scene)
    discover_known_objects(scene)
    baseline = create_accepted_checkpoint(
        scene,
        "project",
        "工作副本和基线已创建",
        label="baseline",
    )
    settings.status_message = "项目已创建"
    save_workfile(scene)
    return baseline


def begin_action_preview(scene, owner, step_id):
    if owner is None:
        raise RuntimeError("Animation owner is missing")
    settings = scene.mocap_doctor
    if settings.preview_step_id:
        if not settings.preview_action:
            raise RuntimeError("A structural preview is waiting for review")
        if not rollback_action_preview(scene):
            raise RuntimeError("Unable to restore the existing Action preview")
    recovery = create_checkpoint(scene, f"{step_id}_before")
    if not owner.animation_data:
        owner.animation_data_create()
    base = owner.animation_data.action
    preview = base.copy() if base else bpy.data.actions.new(f"MD_PREVIEW_{step_id}_{owner.name}")
    if base:
        preview.name = f"MD_PREVIEW_{step_id}_{base.name}"
    owner.animation_data.action = preview
    # A brand new preview Action is assigned unbound; on Blender 4.4+ that
    # leaves every later fcurve write animating nothing (see
    # core.animation.ensure_action_slot).  A copy of an existing Action already
    # carries a bound slot, so this only matters for owners that had none.
    core_animation.ensure_action_slot(owner)
    _ACTION_PREVIEWS[scene.as_pointer()] = {"owner": owner, "base": base, "preview": preview, "step_id": step_id}
    settings.preview_step_id = step_id
    settings.preview_owner_name = owner.name
    settings.preview_base_action = base.name if base else ""
    settings.preview_action = preview.name
    settings.preview_restore_checkpoint = str(recovery)
    record = record_step(scene, step_id, "PREVIEW", "等待人工检查")
    record.before_checkpoint = str(recovery)
    save_manifest(scene)
    return preview


def recover_action_preview(scene):
    """Rebuild an Action preview transaction after a saved-file reload."""

    settings = scene.mocap_doctor
    if not settings.preview_action or not settings.preview_owner_name:
        return False
    owner = bpy.data.objects.get(settings.preview_owner_name)
    preview = bpy.data.actions.get(settings.preview_action)
    base = bpy.data.actions.get(settings.preview_base_action) if settings.preview_base_action else None
    if settings.preview_base_action and base is None:
        return False
    if owner is None or preview is None or not owner.animation_data:
        return False
    if owner.animation_data.action != preview:
        return False
    _ACTION_PREVIEWS[scene.as_pointer()] = {
        "owner": owner,
        "base": base,
        "preview": preview,
        "step_id": settings.preview_step_id,
    }
    return True


def rollback_action_preview(scene):
    settings = scene.mocap_doctor
    data = _ACTION_PREVIEWS.get(scene.as_pointer())
    if data is None and settings.preview_action:
        recover_action_preview(scene)
    data = _ACTION_PREVIEWS.pop(scene.as_pointer(), None)
    if data is None and settings.preview_action:
        return False
    if data:
        owner, preview = data["owner"], data["preview"]
        if owner and owner.animation_data:
            owner.animation_data.action = data["base"]
        if preview and preview.users == 0:
            bpy.data.actions.remove(preview)
    settings.preview_step_id = ""
    settings.preview_owner_name = ""
    settings.preview_base_action = ""
    settings.preview_action = ""
    settings.preview_restore_checkpoint = ""
    return data is not None


def accept_action_preview(scene, message="预览已接受"):
    data = _ACTION_PREVIEWS.get(scene.as_pointer())
    if not data and recover_action_preview(scene):
        data = _ACTION_PREVIEWS.get(scene.as_pointer())
    if not data:
        raise RuntimeError("There is no active Action preview")
    settings = scene.mocap_doctor
    preview_fields = {
        name: getattr(settings, name)
        for name in (
            "preview_step_id",
            "preview_owner_name",
            "preview_base_action",
            "preview_action",
            "preview_restore_checkpoint",
        )
    }
    old_name = data["preview"].name
    data["preview"].name = f"MD_{data['step_id']}_{old_name}"
    settings.preview_step_id = ""
    settings.preview_owner_name = ""
    settings.preview_base_action = ""
    settings.preview_action = ""
    settings.preview_restore_checkpoint = ""
    try:
        checkpoint = create_accepted_checkpoint(scene, data["step_id"], message)
    except Exception:
        data["preview"].name = old_name
        for name, value in preview_fields.items():
            setattr(settings, name, value)
        raise
    _ACTION_PREVIEWS.pop(scene.as_pointer(), None)
    save_workfile(scene)
    return checkpoint


def apply_restored_step_state(
    scene,
    *,
    resume_step_id="",
    reset_step_id="",
    message="",
):
    settings = scene.mocap_doctor
    if resume_step_id in STEP_INDEX:
        settings.current_step = STEP_INDEX[resume_step_id]
    if reset_step_id:
        reset_index = STEP_INDEX.get(reset_step_id, -1)
        reset_record = find_step_record(settings, reset_step_id, create=False)
        if reset_record is not None:
            reset_record.status = "PENDING"
            reset_record.message = "等待重新执行"
        for record in settings.steps:
            if (
                STEP_INDEX.get(record.step_id, -1) > reset_index
                and record.status in {"ACCEPTED", "PREVIEW"}
            ):
                record.status = "STALE"
    settings.status_message = message or "已恢复检查点"
    settings.status_level = 'INFO'
    # A restored file must never keep a pending preview/busy flag: it makes the
    # whole panel grey and unusable after "discard preview".
    settings.busy = False
    if not settings.preview_step_id:
        settings.preview_owner_name = ""
        settings.preview_base_action = ""
        settings.preview_action = ""
        settings.preview_restore_checkpoint = ""


@persistent
def _finish_restore(_dummy):
    global _RESTORE_WORK_PATH, _RESTORE_RESUME_STEP_ID, _RESTORE_RESET_STEP_ID, _RESTORE_MESSAGE
    if not _RESTORE_WORK_PATH:
        return
    target = _RESTORE_WORK_PATH
    if _finish_restore in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_finish_restore)
    attempts = {"count": 0}

    def save_back():
        global _RESTORE_WORK_PATH, _RESTORE_RESUME_STEP_ID, _RESTORE_RESET_STEP_ID, _RESTORE_MESSAGE
        attempts["count"] += 1
        try:
            scene = bpy.context.scene
            if hasattr(scene, "mocap_doctor"):
                settings = scene.mocap_doctor
                settings.work_filepath = target
                apply_restored_step_state(
                    scene,
                    resume_step_id=_RESTORE_RESUME_STEP_ID,
                    reset_step_id=_RESTORE_RESET_STEP_ID,
                    message=_RESTORE_MESSAGE,
                )
            result = bpy.ops.wm.save_as_mainfile(filepath=target, relative_remap=True)
            if "FINISHED" not in result:
                raise RuntimeError("Blender 未能把检查点另存回工作文件")
            if hasattr(scene, "mocap_doctor"):
                save_manifest(scene)
            _RESTORE_WORK_PATH = None
            _RESTORE_RESUME_STEP_ID = ""
            _RESTORE_RESET_STEP_ID = ""
            _RESTORE_MESSAGE = ""
            return None
        except Exception as exc:
            if attempts["count"] < 3:
                return 0.5
            if hasattr(bpy.context.scene, "mocap_doctor"):
                bpy.context.scene.mocap_doctor.status_message = f"恢复检查点后保存失败：{exc}"
            _RESTORE_WORK_PATH = None
            _RESTORE_RESUME_STEP_ID = ""
            _RESTORE_RESET_STEP_ID = ""
            _RESTORE_MESSAGE = ""
            print(f"[MoCap Doctor] restore save failed: {exc}")
            return None

    bpy.app.timers.register(save_back, first_interval=0.1)


def restore_checkpoint(
    checkpoint,
    work_filepath,
    *,
    resume_step_id="",
    reset_step_id="",
    message="",
):
    global _RESTORE_WORK_PATH, _RESTORE_RESUME_STEP_ID, _RESTORE_RESET_STEP_ID, _RESTORE_MESSAGE
    settings = bpy.context.scene.mocap_doctor
    # moved project: same file name under the current data dir (see resolve_project_file)
    checkpoint = resolve_project_file(settings, checkpoint, "checkpoints")
    checkpoint = str(Path(checkpoint).resolve())
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    recovery_root = ensure_project_directories(settings) / "recovery"
    sequences = []
    for item in recovery_root.glob("*.blend"):
        match = re.match(r"^(\d+)_", item.name)
        if match:
            sequences.append(int(match.group(1)))
    recovery = recovery_root / f"{max(sequences, default=-1) + 1:04d}_before_restore.blend"
    recovery_result = bpy.ops.wm.save_as_mainfile(
        filepath=str(recovery), copy=True, relative_remap=True
    )
    if "FINISHED" not in recovery_result:
        raise RuntimeError("无法创建恢复前安全副本，已取消检查点恢复")
    _RESTORE_WORK_PATH = str(Path(work_filepath).resolve())
    _RESTORE_RESUME_STEP_ID = str(resume_step_id)
    _RESTORE_RESET_STEP_ID = str(reset_step_id)
    _RESTORE_MESSAGE = str(message)
    if _finish_restore not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_finish_restore)
    try:
        result = bpy.ops.wm.open_mainfile(filepath=checkpoint, load_ui=False)
        if "FINISHED" not in result:
            raise RuntimeError("Blender 未能打开检查点")
    except Exception:
        _RESTORE_WORK_PATH = None
        _RESTORE_RESUME_STEP_ID = ""
        _RESTORE_RESET_STEP_ID = ""
        _RESTORE_MESSAGE = ""
        if _finish_restore in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(_finish_restore)
        raise


def unregister_handlers():
    global _RESTORE_WORK_PATH, _RESTORE_RESUME_STEP_ID, _RESTORE_RESET_STEP_ID, _RESTORE_MESSAGE
    if bpy.app.timers.is_registered(_migrate_open_project_once):
        bpy.app.timers.unregister(_migrate_open_project_once)
    if _finish_restore in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_finish_restore)
    if _migrate_project_schema in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_migrate_project_schema)
    _RESTORE_WORK_PATH = None
    _RESTORE_RESUME_STEP_ID = ""
    _RESTORE_RESET_STEP_ID = ""
    _RESTORE_MESSAGE = ""
