"""Compare pose reach and static load for one explicitly specified recovery stroke.
Not a collision certificate or a hardware torque rating. Run after CAD placement
changes too: the target poses are expressed relative to the proposed bench.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
import mujoco
ROOT=Path(__file__).parent
MODELS={
 'so101':('robotstudio_so101/so101.xml','base','gripperframe',5,[0,0,0,0,0],.080),
 'lite6':('ufactory_lite6/lite6.xml','link_base','attachment_site',6,[0,0,1.57,0,1.57,0],.100),
 'panda':('franka_emika_panda/panda.xml','link0',None,7,[0,0,0,-1.57,0,1.57,-.785],.100)}

def prepare(name,base):
 path,body,site,n,home,length=MODELS[name]
 p=ROOT/'vendor'/path; root=ET.parse(p).getroot()
 root.find('compiler').set('meshdir',str(p.parent/'assets'))
 key=root.find('keyframe')
 if key is not None:root.remove(key)
 robot=root.find('worldbody').find(f"body[@name='{body}']")
 robot.set('pos',' '.join(map(str,base)))
 if name=='panda':
  parent=robot.find(".//body[@name='hand']")
  pos=[0,0,.10];quat=[.70710678,0,-.70710678,0]
 else:
  target=robot.find(f".//site[@name='{site}']")
  parent=next(e for e in robot.iter('body') if target in list(e))
  pos=list(map(float,target.get('pos','0 0 0').split()))
  quat=list(map(float,target.get('quat','1 0 0 0').split()))
  if name=='lite6':quat=[.70710678,0,-.70710678,0]
 tool=ET.SubElement(parent,'body',name='tool_frame',pos=' '.join(map(str,pos)),quat=' '.join(map(str,quat)))
 ET.SubElement(tool,'geom',type='box',pos=f'{length/2} 0 0',size=f'{length/2} .01 .003',mass='.020',contype='0',conaffinity='0',rgba='.1 .4 .8 1')
 ET.SubElement(tool,'site',name='blade_tip',pos=f'{length} 0 0',size='.002')
 m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
 d=mujoco.MjData(m)
 return m,d,n,np.array(home)

def rotation_error(current,target):
 a=np.empty(4);e=np.empty(3)
 mujoco.mju_mat2Quat(a,(target@current.T).ravel());mujoco.mju_quat2Vel(e,a,1.)
 return e

def ik(m,d,n,q,target,rot,tries=10):
 sid=m.site('blade_tip').id; jp=np.zeros((3,m.nv));jr=jp.copy()
 limits=m.jnt_range[:n]; rng=np.random.default_rng(812)
 best=None
 for attempt in range(tries):
  if attempt:q=rng.uniform(limits[:,0]+.02,limits[:,1]-.02)
  for _ in range(180):
   d.qpos[:n]=q;mujoco.mj_forward(m,d)
   ep=target-d.site_xpos[sid];er=rotation_error(d.site_xmat[sid].reshape(3,3),rot)
   score=np.linalg.norm(ep)+.1*np.linalg.norm(er)
   if best is None or score<best[0]:best=(score,q.copy(),float(np.linalg.norm(ep)),float(np.linalg.norm(er)))
   if np.linalg.norm(ep)<.001 and np.linalg.norm(er)<np.deg2rad(2):return best
   mujoco.mj_jacSite(m,d,jp,jr,sid)
   j=np.vstack((jp[:,:n],.1*jr[:,:n]));e=np.r_[ep,.1*er]
   dq=j.T@np.linalg.solve(j@j.T+.003**2*np.eye(6),e)
   dq*=min(1,.18/max(abs(dq).max(),1e-10));q=np.clip(q+dq,limits[:,0]+.01,limits[:,1]-.01)
 return best

def run():
 # Bench top z=.70, proposed parked plate top .80. CAD must replace these
 # layout assumptions before reporting printer collision clearance.
 results={}
 angle=np.deg2rad(15)
 rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
 targets=[np.array([x,y,.802]) for y in [-.06,0,.06] for x in [-.10,-.05,0,.05,.10,.16]]
 for name in MODELS:
  m,d,n,q=prepare(name,[-.30,0,.70]); rows=[]
  for target in targets:
   score,q,pe,re=ik(m,d,n,q,target,rot)
   d.qpos[:n]=q;mujoco.mj_forward(m,d)
   jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,m.site('blade_tip').id)
   torque=d.qfrc_bias[:n]+jp[:,:n].T@np.array([2.,0,0])
   rows.append({'target_m':target.tolist(),'position_error_mm':pe*1000,'orientation_error_deg':np.rad2deg(re),'pose_pass':bool(pe<.002 and re<np.deg2rad(3)),'static_gravity_plus_2N_joint_torque_Nm':torque.tolist(),'q_rad':q.tolist()})
  results[name]={'passed':sum(r['pose_pass'] for r in rows),'count':len(rows),'samples':rows}
  print(name,results[name]['passed'],'/',len(rows),flush=True)
 out=ROOT/'outputs/arm_audit.json';out.write_text(json.dumps({'scope':'kinematic pose and static modeled load only; no printer collision clearance','base_m':[-.30,0,.70],'blade_pitch_deg':15,'results':results},indent=2))
if __name__=='__main__':run()
