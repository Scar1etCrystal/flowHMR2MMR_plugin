"""Regression: absolute cross-bone orientations, independently evaluated.

Only inner frames are expected to match. Blended endpoints must retain the
old target pose, so an all-window maximum is NOT an acceptance metric.
"""
import os, sys
import addon_utils, bpy, numpy as np
addon_utils.enable('bl_ext.user_default.mocap_doctor')
from bl_ext.user_default.mocap_doctor.core import agent_bridge as AB, agent_pose as P
scene=bpy.context.scene; keep=scene.frame_current; st=scene.mocap_doctor
rig=st.mmr_rig or next(o for o in scene.objects if o.type=='ARMATURE' and o.name.startswith('RIG-'))
st.mmr_rig=rig
st.data_directory=os.path.join(os.environ.get('MCD_TEST_OUTPUT', os.path.dirname(bpy.data.filepath)),
                               'test_world_abs_data')
os.makedirs(st.data_directory,exist_ok=True)
# Isolate op records too; never save the input blend.
log=os.path.join(st.data_directory,'agent_ops.json')
if os.path.isfile(log): os.remove(log)
checks=[]
def check(name,ok,detail=''):
    checks.append(bool(ok)); print(('PASS ' if ok else 'FAIL ')+name+' :: '+str(detail),flush=True)
def call(t,**a):
    r=AB._dispatch({'tool':t,'args':dict(a,agent_id='world-abs-test')})
    if not r.get('ok'): print('TOOL_ERROR',t,r.get('error'),flush=True)
    return r
left=P.chain_preset('arm.L',rig); right=[b[:-2]+'.R' for b in left]
bm=dict(zip(left,right)); main=left[:4]; dstmain=right[:4]
W=[235,283]; frames=list(range(W[0],W[1]+1)); inner=slice(4,-4)

def sample(names,fr):
    out={b:[] for b in names}
    old=scene.frame_current
    try:
        for f in fr:
            scene.frame_set(f)
            for b in names:out[b].append(np.array(rig.pose.bones[b].matrix.to_3x3(),float))
    finally:scene.frame_set(old)
    return {b:np.asarray(v) for b,v in out.items()}

def angle(a,b):
    # independent SVD + atan2 geodesic; stable close to zero
    def rot(m):
        u,s,v=np.linalg.svd(m); r=u@v
        if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@v
        return r
    r=rot(a).T@rot(b)
    v=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])
    return float(np.degrees(np.arctan2(np.linalg.norm(v)/2,(np.trace(r)-1)/2)))
def errors(src,cur):return {s:np.asarray([angle(a,b) for a,b in zip(src[s],cur[d])]) for s,d in bm.items()}
base=sample(right,frames); source=sample(left,frames)
args=dict(chain='arm.L',bone_map=bm,src_range=W,dst_start=W[0],space='world_abs',mirror=False,blend=4)
before=call('compare_motion',a={'bones':right,'frame_range':W},b={'bones':left,'frame_range':W},space='world_abs',trim=4)['data']['err_inner_deg']
count=lambda:sum(len(t.strips) for t in rig.animation_data.nla_tracks)
n=count();dry=call('motion_copy',dry_run=True,**args)
check('dry_run leaves tracks untouched',dry.get('ok') and count()==n)
check('dry_run baseline matches actual world comparison',abs(dry['data']['metrics']['verify']['err_inner_before_deg']-before)<.001,(dry['data']['metrics']['verify']['err_inner_before_deg'],before))
r=call('motion_copy',**args); op=r.get('data',{}).get('op_id')
check('whole 19-bone chain written',bool(op) and len(left)==19)
if not op:sys.exit(1)
cur=sample(right,frames); es=errors(source,cur)
mx=max(float(e[inner].max()) for e in es.values()); allmx=max(float(e.max()) for e in es.values())
check('inner world matrices match source (<.05 deg)',mx<.05,mx)
check('endpoint mismatch is intentionally retained (not failure)',allmx>20,allmx)
check('endpoints retain original target',max(angle(base[b][i],cur[b][i]) for b in right for i in (0,-1))<.05)
cm=call('compare_motion',op_id=op)['data']
check('compare_motion agrees with independent matrices',cm['err_inner_deg']<.05 and abs(cm['err_inner_deg']-mx)<.001,cm['err_inner_deg'])
# User-visible hand long axis: not just the tool comparing itself.
check('hand axes point the same way',max(angle(source['hand_fk.L'][i],cur['hand_fk.R'][i]) for i in range(4,len(frames)-4))<.05)
# Source window must not change while copying to the other side.
src_now=sample(left,frames)
check('source left arm unchanged',max(angle(source[b][i],src_now[b][i]) for b in left for i in range(len(frames)))<.05)
# Reapply uses persisted space; changing blend alone must retain the semantics.
rr=call('reapply',op_id=op,overrides={'blend':3})
check('reapply preserves world_abs',rr.get('ok') and call('compare_motion',op_id=op)['data']['err_inner_deg']<.05)
call('revert',op_id=op)
back=sample(right,frames)
check('revert restores target',max(angle(base[b][i],back[b][i]) for b in right for i in range(len(frames)))<.05)
# Shifted repeat: source 235-275 -> target 275-315, effective 279-311.
W2=[275,315];fs2=list(range(275,316));src2=sample(left,list(range(235,276)))
r=call('motion_copy',chain='arm.L',bone_map=bm,src_range=[235,275],dst_start=275,space='world_abs',mirror=False,blend=4)
op=r.get('data',{}).get('op_id');check('repeat segment written',bool(op))
if op:
    cur2=sample(right,fs2);e2=errors(src2,cur2)
    check('repeat 279-311 matches original template',max(float(e[inner].max()) for e in e2.values())<.05,max(float(e[inner].max()) for e in e2.values()))
    check('repeat op verifier agrees',call('compare_motion',op_id=op)['data']['err_inner_deg']<.05)
    call('revert',op_id=op)
# Mirror+absolute must yield a rotation (det +1), not a reflection matrix.
S=np.diag([-1.,1.,1.])
r=call('motion_copy',chain='arm.L',bone_map=bm,src_range=W,dst_start=235,space='world_abs',mirror=True,blend=4)
op=r.get('data',{}).get('op_id')
if op:
    cur=sample(right,frames)
    me=max(angle(S@source[s][i]@S,cur[d][i]) for s,d in bm.items() for i in range(4,len(frames)-4))
    check('absolute mirrored frames are proper rotations SRS',me<.05,me)
    check('absolute mirror verifier agrees',call('compare_motion',op_id=op)['data']['err_inner_deg']<.05)
    call('revert',op_id=op)
else:check('absolute mirror write',False)
call('release');scene.frame_set(keep)
print('==== %d/%d PASS ===='%(sum(checks),len(checks)))
if not all(checks):sys.exit(1)
