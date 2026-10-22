"""Real FK hand spacing and transaction regression on the user's 577-frame case."""
import sys,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import bpy
sys.path.insert(0,r'F:\mocap_ai_doctor')
import mocap_doctor;mocap_doctor.register()
from mocap_doctor import operators as O,project,workflow,ui
root=Path(r'F:\mocap_ai_doctor\sandbox\hand_spacing_577_review')
root.mkdir(parents=True,exist_ok=True)
scene=bpy.context.scene;s=scene.mocap_doctor;rig=s.mmr_rig;src=s.source_armature
base=rig.animation_data.action
checks=[]
def check(name,ok,detail=None):
 checks.append({'name':name,'pass':bool(ok),'detail':detail})
 print(('PASS ' if ok else 'FAIL ')+name+' '+str(detail or ''),flush=True)
 (root/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 if not ok:raise AssertionError(name)
def curves(action):
 return {(fc.data_path,fc.array_index):[(tuple(k.co),k.interpolation) for k in fc.keyframe_points] for fc in action.fcurves}
def wrists(frame):
 scene.frame_set(frame);bpy.context.view_layer.update()
 return [rig.matrix_world@rig.pose.bones['hand_fk.'+side].head for side in ('L','R')]
def restore_call(call):
 pending=[];register=bpy.app.timers.register
 def capture(cb,**kwargs):
  if cb.__name__=='save_back':pending.append(cb)
  else:register(cb,**kwargs)
 bpy.app.timers.register=capture
 try:r=call()
 finally:bpy.app.timers.register=register
 for cb in pending:assert cb() is None
 return r
s.steps.clear();s.preview_step_id='';s.preview_action='';s.preview_restore_checkpoint=''
project._ACTION_PREVIEWS.clear()
project.initialize_project(scene,str(root/'work.blend'))
s.mocap_frame_start,s.mocap_frame_end=1,1000;s.current_step=workflow.STEP_INDEX['hand_spacing']
old=curves(base);before=wrists(577);before_mid=(before[0]+before[1])*.5
project.save_workfile(scene)
assert 'FINISHED' in bpy.ops.mocap_doctor.run_step(step_id='hand_spacing')
check('hand compensation creates preview',s.preview_step_id=='hand_spacing')
check('base keys remain untouched',curves(base)==old)
act=rig.animation_data.action;new=curves(act)
allowed=('upper_arm_fk.','forearm_fk.','hand_fk.')
changed=[p for p in old.keys()|new.keys() if old.get(p)!=new.get(p)]
check('only FK arm rotations changed',all(any(b in p[0] for b in allowed) and '.rotation_' in p[0] for p in changed),len(changed))
check('finger curves untouched',all(old.get(p)==new.get(p) for p in old.keys()|new.keys() if any(n in p[0] for n in ('f_index','f_middle','f_ring','f_pinky','thumb'))))
after=wrists(577);distance=(after[0]-after[1]).length
check('577 wrist gap restored',.07 < distance < .085,{'before_m':(before[0]-before[1]).length,'after_m':distance})
check('577 midpoint fixed',(((after[0]+after[1])*.5)-before_mid).length<1e-5)
report=json.loads(Path(project.find_step_record(s,'hand_spacing').artifact_path).read_text(encoding='utf-8'))
check('wrist orientation preserved',report['max_wrist_rotation_error_deg']<.1,report['max_wrist_rotation_error_deg'])
check('evaluated FK endpoints correct',report['max_wrist_position_error_m']<.001,report['max_wrist_position_error_m'])
check('normal arm motion retained',0<report['compensated_frames']<1000,report['compensated_frames'])
(root/'hand_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
assert 'FINISHED' in bpy.ops.mocap_doctor.discard_preview()
check('discard restores exact base',rig.animation_data.action==base and curves(base)==old)
assert 'FINISHED' in bpy.ops.mocap_doctor.run_step(step_id='hand_spacing')
assert 'FINISHED' in bpy.ops.mocap_doctor.accept_preview()
check('accepted hand correction recorded',project.find_step_record(s,'hand_spacing').status=='ACCEPTED')
rig_name=rig.name
restore_call(lambda:bpy.ops.mocap_doctor.restore_before_step(step_id='hand_spacing'))
scene=bpy.context.scene;s=scene.mocap_doctor;rig=bpy.data.objects[rig_name]
check('accepted correction can restore exact original keys',curves(rig.animation_data.action)==old)
check('fourth step is contact detection',workflow.STEPS[3].id=='contacts')
check('separate skeleton repair page removed','target_skeleton_fix' not in workflow.STEP_INDEX)
class Layout:
 def __init__(self,log):self.log=log
 def row(self,**kw):return Layout(self.log)
 def column(self,**kw):return Layout(self.log)
 def separator(self,**kw):pass
 def label(self,**kw):self.log.append(('label',kw))
 def prop(self,obj,name,**kw):self.log.append(('prop',{'name':name,**kw}))
 def operator(self,op,**kw):self.log.append(('operator',{'id':op,**kw}));return SimpleNamespace()
log=[];ui.MD_PT_Main.draw_step(SimpleNamespace(),bpy.context,Layout(log),'contacts')
check('contact page has no repair button',not any(k=='operator' and v['id']=='mocap_doctor.target_fix_all' for k,v in log))
check('contact page has detection and editor',all(op in [v['id'] for k,v in log if k=='operator'] for op in ('mocap_doctor.run_step','mocap_doctor.enter_annotation_mode')))
print('HAND_REVIEW_COMPLETE',len(checks),flush=True)
mocap_doctor._stop_agent_server()
