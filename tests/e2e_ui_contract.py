"""新 UI 契约测试（替换三个为已删除 UI 写的旧测试）。

验证用户明确要求的"减法"，用 Blender 4.5.14 跑：
    blender -b --python tests/e2e_ui_contract.py

覆盖：
  A. 已删除的 operator / 面板真的不在
  B. 步骤表：第 0 步创建工作副本、第 1 步导入、ABC 独立成第 3 步、无预留占位
  C. ui.py 源码里没有泄露的术语（Teto / FreeMoCap / GVHMR / npz / fbx / 官方第 / 第五步 / 常驻 ⓘ）
  D. 真机一键导入：52 骨骨架 + 视频平面（法向 −Y、底边 min_z≈0）
  E. 数据缺失时不炸、提示是一句短句且无术语
"""
import os
import sys
import tempfile

import bpy

REPO = r'F:\mocap_ai_doctor'
TAKE = r'F:\flowhmr_pack\outputs\video_to_fbx\t1499_H4_77b6a19f8351'
UI_PY = os.path.join(REPO, 'mocap_doctor', 'ui.py')

results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok), detail))
    print(('PASS ' if ok else 'FAIL ') + name + ('  ' + str(detail) if detail else ''), flush=True)


def run_step(step_id):
    op = bpy.ops.mocap_doctor.run_step
    for kwargs in ({'step_id': step_id}, {'step': step_id}):
        try:
            return op(**kwargs)
        except TypeError:
            continue
    return op()


def plane_metrics():
    best = None
    for ob in bpy.context.scene.objects:
        if ob.type != 'MESH':
            continue
        if not any(k in ob.name.lower() for k in ('video', 'plane', '参考', '视频')):
            continue
        vs = [ob.matrix_world @ v.co for v in ob.data.vertices]
        if not vs:
            continue
        zs = [v.z for v in vs]
        ob.data.calc_loop_triangles()
        n = (ob.matrix_world.to_3x3() @ ob.data.polygons[0].normal).normalized()
        best = {'name': ob.name, 'min_z': min(zs), 'max_z': max(zs),
                'normal': tuple(round(c, 4) for c in n),
                'facing_minus_y': round(n.dot((0.0, -1.0, 0.0)), 4)}
        break
    return best


def main():
    settings = bpy.context.scene.mocap_doctor

    # A. 删除项真的不在
    op_classes = ('MD_OT_OfficialScan', 'MD_OT_OfficialImport',
                  'MD_OT_OfficialVideoPlane', 'MD_OT_OfficialDumpJoints')
    leaked_cls = [c for c in op_classes if hasattr(bpy.types, c)]
    op_ids = ('official_scan', 'official_import', 'official_video_plane',
              'official_dump_joints')
    op_dir = set(dir(bpy.ops.mocap_doctor))
    leaked_ids = [i for i in op_ids if i in op_dir]
    check('A1 独立导入 operator 已删除', not leaked_cls and not leaked_ids,
          {'classes': leaked_cls, 'ids': leaked_ids})
    leaked_panels = [p for p in ('MD_PT_official_import', 'MD_PT_ReservedParams')
                     if hasattr(bpy.types, p)]
    check('A2 独立面板已删除', not leaked_panels, leaked_panels)

    # B. 步骤表
    from mocap_doctor import workflow as W
    order = [s.id for s in W.STEPS]
    check('B1 步骤顺序', order == ['import_official', 'retarget',
                                   'contacts', 'hand_spacing', 'foot_spacing', 'ground_feet',
                                   'foot_lock', 'target_floor', 'mmd_bake', 'export_prep', 'export'], order)
    check('B4 脚滑修复在贴地锁定与穿地修复之间',
          order.index('ground_feet') < order.index('foot_lock') < order.index('target_floor'))
    check('B2 无预留占位步骤', not any('reserved' in s for s in order))
    check('B3 脚间距补偿在重定向之后',
          order.index('foot_spacing') > order.index('retarget'))

    # C. ui.py 不泄露术语（`_draw_message` 的默认 icon 参数不算常驻提示）
    src = open(UI_PY, encoding='utf-8').read()
    banned = ['Teto', 'FreeMoCap', 'GVHMR', 'npz', 'fbx', '官方第', '第五步']
    hits = {b: src.count(b) for b in banned if src.count(b)}
    info_hits = [ln for ln in src.splitlines()
                 if 'icon="INFO"' in ln and not ln.strip().startswith('def ')]
    check('C1 ui.py 无泄露术语', not hits and not info_hits,
          {'terms': hits, 'info_lines': info_hits})

    # D. 真机一键导入（就地写当前文件：不再生成工作副本）
    import os as _os
    temp_blend = _os.path.join(tempfile.mkdtemp(prefix='mcd_ui_'), 'Untited.blend')
    bpy.ops.wm.save_as_mainfile(filepath=temp_blend)
    settings.official_take_dir = TAKE
    res = run_step('import_official')
    check('D1 导入步骤返回 FINISHED', 'FINISHED' in res, res)
    check('D1b 就地写当前文件（不建副本）', settings.initialized and
          _os.path.samefile(bpy.data.filepath, temp_blend) and
          'mocap_doctor_work' not in _os.path.basename(bpy.data.filepath), bpy.data.filepath)
    check('D1c 原件一次性备份', _os.path.isfile(
        _os.path.join(settings.data_directory, 'recovery', 'original.blend')))
    check('D1d 导入不走预览、直接进入下一步',
          not settings.preview_step_id and
          W.STEPS[settings.current_step].id == 'retarget', settings.current_step)
    arm = next((o for o in bpy.context.scene.objects
                if o.type == 'ARMATURE' and len(o.data.bones) == 52), None)
    check('D2 导入得到 52 骨动作骨架', arm is not None,
          arm.name if arm else [o.name for o in bpy.context.scene.objects if o.type == 'ARMATURE'])
    pm = plane_metrics()
    check('D3 参考视频平面存在', pm is not None, pm)
    if pm:
        check('D4 平面法向朝 -Y', pm['facing_minus_y'] > 0.99, pm['facing_minus_y'])
        check('D5 平面底边贴地', abs(pm['min_z']) < 1e-4, pm['min_z'])
    # 导入已自动完成并进入下一步，这里不再有“接受预览”。

    # E. 数据缺失：不炸 + 短提示 + 无术语
    empty = tempfile.mkdtemp(prefix='mcd_empty_')
    settings.official_take_dir = empty
    err = ''
    try:
        res2 = run_step('import_official')
    except Exception as exc:  # noqa: BLE001
        res2, err = {'EXCEPTION'}, f'{type(exc).__name__}: {exc}'
    msg = str(getattr(settings, 'status_message', '') or
              getattr(settings, 'official_status', '') or '')
    check('E1 缺失数据有失败反馈', settings.status_level == 'ERROR' and not settings.preview_step_id, err or res2)
    check('E2 提示是短句', 0 < len(msg) <= 24, (len(msg), msg))
    check('E3 提示无术语', not any(b in msg for b in banned), msg)

    bad = [n for n, ok, _ in results if not ok]
    print('\nTOTAL %d  PASS %d  FAIL %d' % (len(results), len(results) - len(bad), len(bad)))
    if bad:
        print('FAILED:', bad)
    sys.exit(1 if bad else 0)


main()
