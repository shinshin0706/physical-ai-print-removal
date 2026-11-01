"""Search finite placements/strokes against downloaded A1 CAD.

This is a kinematic preview, NOT a dynamic recovery or hardware test. Load
sweep is quasi-static gravity + J.T @ opposing force, 0/1/2/5/10 N selected
by AI as sensitivity cases, not measured PLA peel forces. SO's 2.94 Nm model
limit is not a continuous rating. OpenArm rated limits use its official 2.0
motor table. Exact source revisions are recorded in vendor manifests.
Run: nice -n 15 .venv/bin/python audit_workcell.py
"""
import json
import itertools
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from arm_audit import ik
from a1_workcell import build,collision_pairs,ROOT,PLATE_Z,FRONT,vec


def ik_flat(m,d,n,q,target,tries=12,pitch=0):
    """Five constraints: XYZ, blade pitch and horizontal width; yaw is free."""
    sid=m.site('blade_tip').id;jp=np.zeros((3,m.nv));jr=jp.copy()
    limits=m.jnt_range[:n];rng=np.random.default_rng(731);best=None
    for attempt in range(tries):
        if attempt:q=rng.uniform(limits[:,0]+.03,limits[:,1]-.03)
        for _ in range(220):
            d.qpos[:n]=q;mujoco.mj_forward(m,d)
            rotation=d.site_xmat[sid].reshape(3,3);normal=rotation[:,2]
            axes=rotation[:,:2].T
            ep=target-d.site_xpos[sid];en=np.array([-np.sin(np.deg2rad(pitch)),0.])-rotation[2,:2]
            score=np.linalg.norm(ep)+.10*np.linalg.norm(en)
            angle=max(abs(np.arcsin(np.clip(rotation[2,0],-1,1))+np.deg2rad(pitch)),abs(np.arcsin(np.clip(rotation[2,1],-1,1))))
            if normal[2]<0:angle=np.pi
            if best is None or score<best[0]:best=(score,q.copy(),float(np.linalg.norm(ep)),float(angle))
            if np.linalg.norm(ep)<.0008 and angle<np.deg2rad(1):return best
            mujoco.mj_jacSite(m,d,jp,jr,sid)
            J=np.vstack([jp[:,:n],.1*np.cross(axes,[0,0,1])@jr[:,:n]])
            dq=J.T@np.linalg.solve(J@J.T+.005**2*np.eye(5),np.r_[ep,.1*en])
            dq*=min(1,.12/max(abs(dq).max(),1e-10));q=np.clip(q+dq,limits[:,0]+.005,limits[:,1]-.005)
    return best


def rotation(yaw,pitch=10):
    c,s=np.cos(yaw),np.sin(yaw);p=np.deg2rad(pitch)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])@np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])


def evaluate(m,d,n,q,targets,R,tries=3,check_collision=True,free_yaw_pitch=None):
    samples=[]
    for t in targets:
        solver=(ik(m,d,n,q,np.array(t),R,tries=tries) if free_yaw_pitch is None else
                ik_flat(m,d,n,q,np.array(t),tries=tries,pitch=free_yaw_pitch))
        _,q,pe,re=solver
        d.qpos[:n]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
        passed=bool(pe<.002 and re<np.deg2rad(3))
        contacts=collision_pairs(m,d) if passed and check_collision else []
        axis=d.site_xmat[m.site('blade_tip').id].reshape(3,3)[:,0]
        samples.append({'target_m':list(t),'q_rad':q.tolist(),'blade_yaw_deg':float(np.rad2deg(np.arctan2(axis[1],axis[0]))),'position_error_mm':pe*1000,'orientation_error_deg':float(np.rad2deg(re)),
                        'pose_pass':passed,'clearance_pass':passed and not contacts,'near_pairs':contacts[:8]})
    return samples


def main():
    results={};selected={}
    for name in ('so101','openarm'):
        candidates=[]
        # SO: radial diagonal sweeps, plus frontward reference. OpenArm: frontward.
        bases=([(-x,.02,.74) for x in (.30,.34,.38)] if name=='so101'
               else [(-x,y,1.08) for x in (.34,.42) for y in (-.1,.02)])
        angles=(0,) if name=='so101' else (90,180)
        root,m,d,n,home,parts=build(name,bases[0])
        body=m.body('base' if name=='so101' else 'openarm_right_base_link').id
        for base,base_yaw,pitch in itertools.product(bases,angles,(25,35)):
            # Relocate the base and its support for collision purposes.
            m.body_pos[body]=base
            by=np.deg2rad(base_yaw);m.body_quat[body]=[np.cos(by/2),0,0,np.sin(by/2)]
            # The post is excluded from this printer/table screen. Hardware
            # support clearances must be checked separately before adoption.
            end=np.array([0.,FRONT-.018,PLATE_Z+.001])
            if name=='so101':
                direction=end-np.array([base[0]+.0388353,base[1],end[2]])
                direction/=np.linalg.norm(direction)
            else:direction=np.array([0.,-1.,0.])
            yaw=np.arctan2(direction[1],direction[0]);angle=float(np.rad2deg(yaw))
            start=end-.110*direction
            targets=[start+u*(end-start) for u in np.linspace(0,1,9)]
            rows=evaluate(m,d,n,home.copy(),targets,rotation(yaw,pitch),tries=8,free_yaw_pitch=pitch if name=='so101' else None)
            candidate={'base_m':list(base),'base_yaw_deg':base_yaw,'pitch_deg':pitch,'yaw_deg':angle,'positioning_joints':n,
                       'orientation_constraint':'pitch and horizontal blade width; free yaw' if name=='so101' else 'full pose',
                       'pose_pass':sum(x['pose_pass'] for x in rows),'clearance_pass':sum(x['clearance_pass'] for x in rows),'samples':rows}
            candidates.append(candidate)
            print(name,base,base_yaw,pitch,candidate['pose_pass'],candidate['clearance_pass'],flush=True)
        best=max(candidates,key=lambda c:(c['clearance_pass'],c['pose_pass'],-sum(s['position_error_mm'] for s in c['samples'])))
        # Rebuild selected base to make the pictured support match the robot.
        root,m,d,n,home,parts=build(name,best['base_m'],np.deg2rad(best['base_yaw_deg']))
        start=np.array(best['samples'][0]['target_m']);end=np.array(best['samples'][-1]['target_m'])
        R=rotation(np.deg2rad(best['yaw_deg']),best['pitch_deg'])
        # Dense continuation, no random branch switching along the selected stroke.
        q=np.array(best['samples'][0]['q_rad'])
        dense=evaluate(m,d,n,q,[start+u*(end-start) for u in np.linspace(0,1,61)],R,tries=1,free_yaw_pitch=best['pitch_deg'] if name=='so101' else None)
        jp=np.zeros((3,m.nv));jr=jp.copy();direction=(end-start)/np.linalg.norm(end-start)
        loads={str(f):np.zeros(n) for f in (0,1,2,5,10)}
        for row in dense:
            if not row['pose_pass']:continue
            d.qpos[:n]=row['q_rad'];d.qvel[:]=0;mujoco.mj_forward(m,d)
            mujoco.mj_jacSite(m,d,jp,jr,m.site('blade_tip').id)
            for f in loads:
                # External force is -F*direction; equilibrium actuator = bias - J.T*external.
                loads[f]=np.maximum(loads[f],abs(d.qfrc_bias[:n]+jp[:,:n].T@(float(f)*direction)))
        jumps=np.max(abs(np.diff(np.array([r['q_rad'] for r in dense]),axis=0)),axis=1)
        best.update(dense_samples=dense,dense_pose_pass=sum(r['pose_pass'] for r in dense),
                    dense_clearance_pass=sum(r['clearance_pass'] for r in dense),max_joint_step_deg=float(np.rad2deg(jumps.max())),
                    max_abs_static_torque_Nm={f:t.tolist() for f,t in loads.items()},
                    model_peak_limits_Nm=m.actuator_forcerange[:n,1].tolist(),
                    continuous_rated_limits_Nm=[.980665]*n if name=='so101' else [20,20,9,9,3,3,3],
                    rating_scope='Waveshare ST3215 12V variant rated 10kgf.cm; 7.4V variant rated 5kgf.cm, not interchangeable' if name=='so101' else 'OpenArm 2.0 official motor table',
                    max_blade_motion_yaw_difference_deg=max(abs(row['blade_yaw_deg']-best['yaw_deg']) for row in dense))
        results[name]={'candidates':candidates,'selected':best}
        (ROOT/f'outputs/{name}_cad_scene.xml').write_text(ET.tostring(root,encoding='unicode'))
    output={'scope':'finite kinematic stroke search; no dynamics, no approach/retreat, no self-collision certificate; per-solid convex printer collision screen',
            'clearance_threshold_m':.003,'pose_tolerances':[.002,3],'force_cases_N':[0,1,2,5,10],
            'force_cases_measured':False,'tool_mass_kg':.0241,'tool_retention':'ideal rigid grasp; not verified',
            'rating_sources':['https://docs.waveshare.com/ST3215_Servo','https://docs.openarm.dev/hardware/openarm-2.0/motor/'],
            'results':results}
    (ROOT/'outputs/cad_arm_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Saved outputs/cad_arm_audit.json',flush=True)


if __name__=='__main__':main()
