"""Install the delivered ZIP into an isolated local extension repository."""
import addon_utils
import bpy
import importlib
import json
from pathlib import Path

root = Path(r'F:\mocap_ai_doctor\sandbox\review_package_198_verified')
root.mkdir(parents=True, exist_ok=True)
repo = bpy.context.preferences.extensions.repos.new(
    name='MoCap Doctor QA', module='mcd_review_qa', custom_directory=str(root / 'extensions'))
result = bpy.ops.extensions.package_install_files(
    filepath=r'F:\mocap_ai_doctor\dist\mocap_doctor-1.9.8-20261016.zip',
    repo=repo.module, enable_on_install=True)
assert 'FINISHED' in result, result
module_id = 'bl_ext.' + repo.module + '.mocap_doctor'
enabled = addon_utils.check(module_id)
assert enabled == (True, True), enabled
module = importlib.import_module(module_id)
rt = importlib.import_module(module_id + '.core.flowhmr_retarget')
ops = importlib.import_module(module_id + '.operators')
assert Path(rt.BMAP).is_file(), rt.BMAP
assert 'retarget' in ops.RUN_STEP_HANDLERS
assert 'target_skeleton_fix' in ops.RUN_STEP_HANDLERS  # deprecated script compatibility
assert 'hand_spacing' in ops.RUN_STEP_HANDLERS
hands = importlib.import_module(module_id + '.core.hand_spacing')
assert callable(hands.compensate_hand_spacing_fk)
steps = importlib.import_module(module_id + '.workflow').STEPS
assert steps[3].id == 'contacts' and all(step.id != 'target_skeleton_fix' for step in steps)
assert 'target_fix_all' in dir(bpy.ops.mocap_doctor)
assert hasattr(bpy.context.scene.mocap_doctor, 'status_level')
report = {'package_install': sorted(result), 'enabled': enabled,
          'module': module_id, 'mapping_inside_installed_package': rt.BMAP,
          'preview_handlers': sorted(k for k in ops.RUN_STEP_HANDLERS
                                     if k in ('retarget', 'target_skeleton_fix', 'foot_spacing', 'import_official'))}
(root / 'install_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('PACKAGE_INSTALL_PASS', report, flush=True)
module._stop_agent_server()
