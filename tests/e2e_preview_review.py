"""Real Blender regression for preview / accept / discard / exact checkpoint restore.

Run with an isolated copy of the native FlowHMR scene loaded on the command line.
All writes go under --out; no user workfile or official output is saved over.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mocap_doctor
mocap_doctor.register()
from mocap_doctor import annotation, operators as O, project, ui, workflow

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
MODE = args[args.index('--mode') + 1] if '--mode' in args else 'action'
OUT = Path(args[args.index('--out') + 1]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = []


def check(label, condition, detail=None):
    row = {'label': label, 'pass': bool(condition), 'detail': detail}
    RESULTS.append(row)
    print(('PASS ' if condition else 'FAIL ') + label + ' ' + str(detail or ''), flush=True)
    (OUT / 'results.json').write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    if not condition:
        raise AssertionError(label + ': ' + str(detail))


def action_digest(action):
    if action is None:
        return None
    payload = [(fc.data_path, fc.array_index,
                [(tuple(p.co), tuple(p.handle_left), tuple(p.handle_right), p.interpolation)
                 for p in fc.keyframe_points]) for fc in action.fcurves]
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def scene_digest():
    return {o.name: {'rest': [tuple(v for row in b.matrix_local for v in row) for b in o.data.bones],
                     'action': action_digest(o.animation_data.action if o.animation_data else None),
                     'ikfk': {b.name: dict(b.items()) for b in o.pose.bones
                              if b.name in ('thigh_parent.L', 'thigh_parent.R', 'upper_arm_parent.L', 'upper_arm_parent.R')}}
            for o in bpy.data.objects if o.type == 'ARMATURE'}


class Layout:
    def __init__(self, log):
        self.log = log
    def row(self, **kw): return Layout(self.log)
    def column(self, **kw): return Layout(self.log)
    def box(self): return Layout(self.log)
    def separator(self, **kw): pass
    def label(self, **kw): self.log.append(('label', kw))
    def prop(self, owner, name, **kw):
        assert hasattr(owner, name), name
        self.log.append(('prop', {'name': name, **kw}))
    def operator(self, operator, **kw):
        category, name = operator.split('.')
        assert name in dir(getattr(bpy.ops, category)), operator
        self.log.append(('operator', {'id': operator, **kw}))
        return SimpleNamespace()


def draw():
    log = []
    fake = SimpleNamespace(layout=Layout(log))
    fake.draw_step = lambda ctx, layout, sid: ui.MD_PT_Main.draw_step(fake, ctx, layout, sid)
    fake.draw_navigation = lambda layout, settings: ui.MD_PT_Main.draw_navigation(fake, layout, settings)
    ui.MD_PT_Main.draw(fake, bpy.context)
    return log


def run(step):
    return bpy.ops.mocap_doctor.run_step('EXEC_DEFAULT', step_id=step)


def restored_call(call):
    # Headless scripts do not pump GUI timers. Run the real save-back callback
    # synchronously after the actual checkpoint open; never stub checkpoint IO.
    pending = []
    register = bpy.app.timers.register
    def capture(callback, **kwargs):
        if callback.__name__ == 'save_back':
            pending.append(callback)
        else:
            register(callback, **kwargs)
    bpy.app.timers.register = capture
    try:
        result = call()
    finally:
        bpy.app.timers.register = register
    for callback in pending:
        check('restore save-back callback', callback() is None)
    return result


scene = bpy.context.scene
s = scene.mocap_doctor
s.initialized = False
s.steps.clear()
for attr in ('preview_step_id', 'preview_owner_name', 'preview_base_action', 'preview_action', 'preview_restore_checkpoint'):
    setattr(s, attr, '')
project._ACTION_PREVIEWS.clear()
source = next(o for o in bpy.data.objects if o.type == 'ARMATURE' and 'Pelvis' in o.data.bones and 'L_Wrist' in o.data.bones)
rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE' and 'torso_root' in o.data.bones and 'foot_ik.L' in o.data.bones)
mmd = next(o for o in bpy.data.objects if o.type == 'ARMATURE' and o != rig and '足.L' in o.data.bones)
s.source_armature = source
s.official_reference = source
s.mmr_rig = rig
s.mmd_armature = mmd
s.target_mesh = bpy.data.objects.get('arue式重音テトver 2.01_mesh')
project.initialize_project(scene, str(OUT / 'work.blend'))
s = bpy.context.scene.mocap_doctor
s.source_armature = source
s.official_reference = source
s.mmr_rig = rig
s.mmd_armature = mmd
s.mocap_frame_start, s.mocap_frame_end = 1, 60
scene.frame_set(1)
for side in ('L', 'R'):
    annotation.set_planted_auto_ranges(scene, side, [(1, 60)], initialize_effective=True, rebuild=False)
    annotation.set_planted_effective_ranges(scene, side, [(1, 60)], rebuild=False)
project.save_workfile(scene)

try:
    if MODE == 'action':
        s.current_step = workflow.STEP_INDEX['hand_spacing']
        rig_name = rig.name
        base = rig.animation_data.action
        original = action_digest(base)
        check('repair operator FINISHED', 'FINISHED' in run('hand_spacing'))
        check('repair creates pending preview', s.preview_step_id == 'hand_spacing')
        check('base Action remains unchanged', action_digest(base) == original)
        check('repair uses a separate Action', rig.animation_data.action != base)
        log = draw()
        ids = [kw['id'] for kind, kw in log if kind == 'operator']
        check('accept and discard are visible', 'mocap_doctor.accept_preview' in ids and 'mocap_doctor.discard_preview' in ids)
        check('no ABC applied labels', not any('A 已应用' in kw.get('text', '') for kind, kw in log if kind == 'label'))
        check('no success warning icon', not any(kw.get('icon') == 'ERROR' for kind, kw in log if kind == 'label'))
        check('discard FINISHED', 'FINISHED' in bpy.ops.mocap_doctor.discard_preview())
        check('discard restores exact original Action', rig.animation_data.action == base and action_digest(base) == original)
        check('discard clears pending preview', not s.preview_step_id)

        check('rerun repair FINISHED', 'FINISHED' in run('hand_spacing'))
        check('accept FINISHED', 'FINISHED' in bpy.ops.mocap_doctor.accept_preview())
        check('accept clears pending and records checkpoint', not s.preview_step_id and project.find_step_record(s, 'hand_spacing').status == 'ACCEPTED')
        check('exact before-checkpoint retained', Path(project.find_step_record(s, 'hand_spacing').before_checkpoint).is_file())
        s.current_step = workflow.STEP_INDEX['hand_spacing']
        check('restore-before button visible', any(kind == 'operator' and kw['id'] == 'mocap_doctor.restore_before_step' for kind, kw in draw()))
        restored_call(lambda: bpy.ops.mocap_doctor.restore_before_step('EXEC_DEFAULT', step_id='hand_spacing'))
        s = bpy.context.scene.mocap_doctor
        scene = bpy.context.scene
        rig = bpy.data.objects[rig_name]
        check('accepted repair can restore original keys', action_digest(rig.animation_data.action) == original)
        check('restored step is pending', project.find_step_record(s, 'hand_spacing').status == 'PENDING')
        check('restore stays on repaired page', s.current_step == workflow.STEP_INDEX['hand_spacing'])

        # Spacing uses the same Action transaction.
        s.current_step = workflow.STEP_INDEX['foot_spacing']
        base = rig.animation_data.action
        original_spacing = action_digest(base)
        check('spacing FINISHED', 'FINISHED' in run('foot_spacing'))
        check('spacing pending preview', s.preview_step_id == 'foot_spacing')
        check('spacing base keys unchanged', action_digest(base) == original_spacing)
        bpy.ops.mocap_doctor.discard_preview()
        check('spacing discard restores original keys', action_digest(rig.animation_data.action) == original_spacing)

        # A partial repair must not claim success or lose its recovery controls.
        from mocap_doctor.core import hand_spacing as H
        real = H.compensate_hand_spacing_fk
        def broken(*args, **kwargs):
            raise RuntimeError('测试：手间距修复中断')
        H.compensate_hand_spacing_fk = broken
        try:
            op = O.MD_OT_RunStep
            reporter = SimpleNamespace(step_id='hand_spacing', report=lambda *a: None)
            result = op.execute(reporter, bpy.context)
        finally:
            H.compensate_hand_spacing_fk = real
        check('partial failure is CANCELLED', 'CANCELLED' in result)
        check('failure is visible and recoverable', s.status_level == 'ERROR' and s.preview_step_id == 'hand_spacing')
        ids = [kw['id'] for kind, kw in draw() if kind == 'operator']
        check('failed preview has discard but no accept', 'mocap_doctor.discard_preview' in ids and 'mocap_doctor.accept_preview' not in ids)
        bpy.ops.mocap_doctor.discard_preview()
        check('failed repair discard restores original keys', action_digest(rig.animation_data.action) == original_spacing)

        # Planted editing must work from in-scene ranges, even without a disk report.
        result = bpy.ops.mocap_doctor.enter_annotation_mode(channel_group='FOOT')
        check('planted editor opens', 'FINISHED' in result and scene.mcd_annotation_mode)
        result = bpy.ops.mocap_doctor.exit_annotation_mode()
        check('planted editor saves and returns to contacts page', 'FINISHED' in result and s.current_step == workflow.STEP_INDEX['contacts'])
        project.record_step(scene, 'contacts', 'ACCEPTED', 'legacy contact record')
        check('planted reset not blocked by legacy accepted record', 'FINISHED' in bpy.ops.mocap_doctor.reset_effective_contacts('EXEC_DEFAULT'))
        check('planted reset gives concise success feedback', s.status_level == 'INFO' and s.status_message == '贴地区间已重置')

    elif MODE == 'import':
        s.current_step = workflow.STEP_INDEX['import_official']
        s.official_take_dir = r'F:\flowhmr_pack\outputs\video_to_fbx\t1499_H4_77b6a19f8351'
        names_before = set(bpy.data.objects.keys())
        original_range = (s.mocap_frame_start, s.mocap_frame_end)
        work_hash = hashlib.sha256((OUT / 'work.blend').read_bytes()).hexdigest()
        check('import FINISHED', 'FINISHED' in run('import_official'))
        check('import is structural preview', s.preview_step_id == 'import_official' and not s.preview_action)
        check('import does not save workfile', hashlib.sha256((OUT / 'work.blend').read_bytes()).hexdigest() == work_hash)
        check('import selects its own source', s.source_armature == s.official_reference and len(s.source_armature.data.bones) == 52)
        check('import re-detects scene objects automatically',
              bool(s.mmr_rig and s.model_root and s.target_mesh),
              [getattr(s, n).name if getattr(s, n) else None
               for n in ('mmr_rig', 'model_root', 'target_mesh')])
        check('import sets complete processing range', (s.mocap_frame_start, s.mocap_frame_end) == (1, 1499), (s.mocap_frame_start, s.mocap_frame_end))
        check('absolute foot joints generated automatically', Path(s.foot_contacts_joints_npz).is_file())
        check('automatic cache belongs to project', Path(s.foot_contacts_joints_npz).is_relative_to(Path(s.data_directory)))
        plane = bpy.data.objects['MCD_reference_video']
        min_z = min((plane.matrix_world @ vertex.co).z for vertex in plane.data.vertices)
        normal = (plane.matrix_world.to_3x3() @ plane.data.polygons[0].normal).normalized()
        check('reference plane grounded and faces minus Y', abs(min_z) < 1e-5 and normal.y < -0.999)
        restored_call(lambda: bpy.ops.mocap_doctor.discard_preview())
        s = bpy.context.scene.mocap_doctor
        check('import discard restores object set', set(bpy.data.objects.keys()) == names_before)
        check('import discard restores old processing range', (s.mocap_frame_start, s.mocap_frame_end) == original_range)
        check('import can rerun', 'FINISHED' in run('import_official'))
        check('import accept FINISHED', 'FINISHED' in bpy.ops.mocap_doctor.accept_preview())
        check('import accept advances to retarget', s.current_step == workflow.STEP_INDEX['retarget'] and not s.preview_step_id)
        restored_call(lambda: bpy.ops.mocap_doctor.restore_before_step(step_id='import_official'))
        check('accepted import restores complete original scene', set(bpy.data.objects.keys()) == names_before)
    elif MODE == 'structure':
        import addon_utils
        key = 'bl_ext.user_default.auto_rig_pro'
        if key not in bpy.context.preferences.addons:
            pref = bpy.context.preferences.addons.new()
            pref.module = key
        addon_utils.enable(key, default_set=False)
        s.current_step = workflow.STEP_INDEX['retarget']
        before = scene_digest()
        work_hash = hashlib.sha256((OUT / 'work.blend').read_bytes()).hexdigest()
        check('retarget FINISHED', 'FINISHED' in run('retarget'))
        check('retarget pending structural preview', s.preview_step_id == 'retarget' and not s.preview_action)
        check('retarget does not save workfile', hashlib.sha256((OUT / 'work.blend').read_bytes()).hexdigest() == work_hash)
        report_path = project.find_step_record(s, 'retarget').artifact_path
        report = json.loads(Path(report_path).read_text(encoding='utf-8'))
        check('retarget uses displayed source and target', report['source'] == source.name and report['target'] == rig.name, report)
        check('retarget includes all skeleton fixes', set(report.get('skeleton_fixes', {})) == {'A','B','C'})
        check('automatic skeleton fixes actually applied', all(not r.get('skipped') and not str(r.get('status','')).startswith('skipped') for r in report['skeleton_fixes'].values()))
        props = [kw['name'] for kind, kw in draw() if kind == 'prop']
        check('source and target selectors visible', 'source_armature' in props and 'mmr_rig' in props)
        restored_call(lambda: bpy.ops.mocap_doctor.discard_preview())
        s = bpy.context.scene.mocap_doctor
        check('structural discard restores rigs and actions', scene_digest() == before)
        check('structural discard stays at retarget', not s.preview_step_id and s.current_step == workflow.STEP_INDEX['retarget'])
        check('retarget can rerun', 'FINISHED' in run('retarget'))
        check('retarget acceptance FINISHED', 'FINISHED' in bpy.ops.mocap_doctor.accept_preview())
        check('retarget accepted checkpoint exists', Path(project.find_step_record(s, 'retarget').checkpoint).is_file())
        s.current_step = workflow.STEP_INDEX['retarget']
        restored_call(lambda: bpy.ops.mocap_doctor.restore_before_step(step_id='retarget'))
        check('accepted retarget can restore original scene', scene_digest() == before)
    else:
        raise ValueError(MODE)
    print('REVIEW_TEST_COMPLETE', MODE, len(RESULTS), flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    mocap_doctor._stop_agent_server()
