"""Rest-pose presets, built on top of ARP's own saved rest pose.

ARP stores the rest snapshot in the scene ID property ``rest_transf_offset``
(``_save_pose_rest`` in auto_rig_remap.py): a dict of
``{bone: [location, rotation_mode, rotation_euler, rotation_quaternion]}``, and
its Redefine Rest Pose dialog reads the very same key back when
``rest_pose='SAVED'``.  That property only lives inside one .blend, so this
module exports it to JSON (one file per target model) and writes it back before
the retarget, which is exactly "borrow ARP's saved option, but make it a preset".

The values are bone-local basis transforms, so a preset is valid for the same
source rest skeleton and the same target rig - FlowHMR's source is fixed, so one
preset per MMD model is reusable across takes.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import bpy

SCENE_KEY = 'rest_transf_offset'
SCHEMA = 1


def preset_dir() -> Path:
    root = Path(bpy.utils.user_resource('CONFIG', path='mocap_doctor/rest_poses', create=True))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe(name: str) -> str:
    return re.sub(r'[^\w\-.]+', '_', str(name or ''), flags=re.UNICODE).strip('._') or 'preset'


def _plain(value: Any) -> Any:
    """Convert Blender ID-property containers into plain JSON data."""

    if hasattr(value, 'to_list'):
        return [round(float(item), 8) for item in value.to_list()]
    if isinstance(value, (list, tuple)):
        return [round(float(item), 8) for item in value]
    return value


def export_snapshot(scene: Any, target_armature_name: str, name: str) -> Path:
    """Write the scene's ARP rest snapshot to ``<name>.json``."""

    raw = scene.get(SCENE_KEY)
    if not raw:
        raise RuntimeError('还没有 rest 快照；请先在 ARP 里 Redefine Rest Pose 并 Apply')
    bones: dict[str, list] = {}
    for bone_name in raw.keys():
        entry = raw[bone_name]
        bones[str(bone_name)] = [
            _plain(entry[0]),
            str(entry[1]),
            _plain(entry[2]),
            _plain(entry[3]),
        ]
    stem = _safe(name)
    path = preset_dir() / f'{stem}.json'
    sequence = 2
    while path.exists():
        path = preset_dir() / f'{stem}_{sequence}.json'
        sequence += 1
    path.write_text(json.dumps({
        'schema': SCHEMA,
        'target_armature': str(target_armature_name),
        'bones': bones,
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    return path


def load_preset(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if int(data.get('schema', 0)) != SCHEMA or 'bones' not in data:
        raise RuntimeError('这不是有效的 rest 预设文件')
    return data


def items() -> list[tuple[str, str]]:
    """(path, display name) for every preset, sorted by name."""

    found = []
    for file in sorted(preset_dir().glob('*.json')):
        try:
            data = json.loads(file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        target = str(data.get('target_armature') or '')
        label = file.stem if not target else f'{file.stem}（{target}）'
        found.append((str(file), label))
    return found


def find_matching(target_armature_name: str) -> str:
    """Path of the preset saved for this target model, or ""."""

    wanted = str(target_armature_name or '')
    if not wanted:
        return ''
    for file in sorted(preset_dir().glob('*.json')):
        try:
            data = json.loads(file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if str(data.get('target_armature') or '') == wanted:
            return str(file)
    return ''


def apply_to_scene(scene: Any, path: str, source_armature: Any = None) -> dict[str, Any]:
    """Load a preset into ``scene['rest_transf_offset']`` for ARP's SAVED mode.

    A preset must describe the *whole* source skeleton: ARP assigns every bone
    from the snapshot and resets anything missing, so a partial preset silently
    destroys the pose (measured: a one-bone preset lifted the model into the air).
    Anything below 90% coverage is rejected and the caller falls back to the
    geometric alignment.
    """

    data = load_preset(path)
    bones = data['bones']
    if source_armature is not None:
        source_names = [bone.name for bone in source_armature.pose.bones]
        unknown = [name for name in bones if name not in source_names]
        if unknown:
            raise RuntimeError(
                f'rest 预设与当前源骨架不匹配（{len(unknown)} 根骨不存在）')
        uncovered = [name for name in source_names if name not in bones]
        allowed = max(3, int(len(source_names) * 0.1))
        if len(uncovered) > allowed:
            raise RuntimeError(
                f'rest 预设不完整（{len(source_names)} 根骨里缺 {len(uncovered)} 根）')
    scene[SCENE_KEY] = bones
    return {'path': str(path), 'bones': len(bones),
            'target_armature': data.get('target_armature', '')}


def autosave_unused(*_args, **_kwargs):  # pragma: no cover - kept for older imports
    """Deprecated: presets are saved on demand now (the list grew too long)."""

    return {'created': False}


def snapshot_exists(scene: Any) -> bool:
    return bool(scene.get(SCENE_KEY))
