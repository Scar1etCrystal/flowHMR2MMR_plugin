"""Workflow-state migration: old wizard indices -> current 10-step workflow.

v7 (8 steps) and v8 (10 steps, with 创建工作副本) both have to land on the
current layout, where step 0 is 导入 and the work copy is created implicitly.
"""
import sys, json
from pathlib import Path
import bpy
sys.path.insert(0, r'F:\mocap_ai_doctor')
import mocap_doctor
mocap_doctor.register()
from mocap_doctor import project, workflow

s = bpy.context.scene.mocap_doctor
s.initialized = True
results = []

# v8 index -> current index: page 0 (project) collapses onto 导入, the rest shift by one.
V8_EXPECTED = (0, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10)
for old, expected in enumerate(V8_EXPECTED):
    s.schema_version = 8
    s.current_step = old
    s.preview_step_id = ''
    project._migrate_project_schema(None)
    ok = s.current_step == expected and s.schema_version == project.SCHEMA_VERSION
    results.append({'from': 'v8', 'old': old, 'new': s.current_step, 'expected': expected, 'pass': ok})
    assert ok, results[-1]

# v7 (8 steps, before 贴地区间/手间距 were added) is a best-effort remap: the
# wizard has since been restructured twice, so only require a valid landing page.
for old in range(8):
    s.schema_version = 7
    s.current_step = old
    s.preview_step_id = ''
    project._migrate_project_schema(None)
    ok = (0 <= s.current_step < len(workflow.STEPS)
          and s.schema_version == project.SCHEMA_VERSION)
    results.append({'from': 'v7', 'old': old, 'new': s.current_step, 'pass': ok})
    assert ok, results[-1]

# A waiting preview keeps its own page.
s.schema_version = 7
s.current_step = 3
s.preview_step_id = 'foot_spacing'
project._migrate_project_schema(None)
assert s.current_step == workflow.STEP_INDEX['foot_spacing'] and s.preview_step_id == 'foot_spacing'
results.append({'pending_preview_preserved': True, 'pass': True})

Path(r'F:\mocap_ai_doctor\agent_reports\hand_workflow_migration.json').write_text(
    json.dumps(results, indent=2), encoding='utf-8')
print('MIGRATION_PASS', len(results), flush=True)
mocap_doctor._stop_agent_server()
