"""Frozen 81-condition contact-dynamics evaluation. Simulator truth scores outcomes.

The trajectory is never replanned to each part pose. Part position is initialized
once. Explicit friction pairs fix the previous geom-mixing ambiguity. No force
or object state is given to the controller. No actual printer commands exist.
Run: nice -n 15 .venv/bin/python evaluate_motion.py [--videos]
"""
import argparse,hashlib,itertools,json,time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np,mujoco
from a1_workcell import ROOT,PLATE_Z,vec
from evaluation_protocol import sha
OUT=ROOT/'outputs/evaluation';INPUT=ROOT/'data/evaluation'
CORNERS=np.array(list(itertools.product([-1,1],repeat=3)),float)*[.015,.015,.010]


def cases(protocol):
    p=protocol['motion']
    return [{'id':f'motion_{i:03d}','mass_kg':mass,'friction':friction,'along_offset_m':along,'cross_offset_m':cross}
            for i,(mass,friction,along,cross) in enumerate(itertools.product(p['masses_kg'],p['surface_friction'],p['along_offsets_m'],p['cross_offsets_m']))]


def setup(case):
    rows=json.loads((INPUT/'trajectory.json').read_text())['samples']
    push=[r for r in rows if r['phase']=='押す経路'];start=np.array(push[0]['target_m']);end=np.array(push[-1]['target_m'])
    direction=(end-start)/np.linalg.norm(end-start);cross=np.array([-direction[1],direction[0],0])
    center=start+direction*(.025+case['along_offset_m'])+cross*case['cross_offset_m'];center[2]=PLATE_Z+.0103
    root=ET.parse(INPUT/'scene.xml').getroot();root.find('option').set('timestep','.002');root.find('option').set('iterations','50')
    for actuator in root.findall('actuator/*'):actuator.set('forcerange','-.980665 .980665')
    body=ET.SubElement(root.find('worldbody'),'body',name='released_part',pos=vec(center))
    ET.SubElement(body,'freejoint',name='part_free')
    ET.SubElement(body,'geom',name='released_part_geom',type='box',size='.015 .015 .010',mass=str(case['mass_kg']),friction='.2 .005 .0001',rgba='1 .25 .02 1')
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    surfaces=['a1_05']+[g.get('name') for g in root.findall('worldbody/geom') if g.get('name','').startswith('chute_')]
    for name in surfaces:
        ET.SubElement(contact,'pair',name=f'part_{name}',geom1='released_part_geom',geom2=name,condim='3',friction=vec([case['friction'],case['friction'],.005,.0001,.0001]))
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
    d.qpos[:5]=rows[0]['q_rad'];d.qpos[5]=.20;d.ctrl[:5]=d.qpos[:5];d.ctrl[5]=.20;mujoco.mj_forward(m,d)
    t=np.array([r['time_s'] for r in rows]);q=np.array([r['q_rad'] for r in rows]);keep=np.r_[True,np.diff(t)>0]
    return m,d,t[keep],q[keep],center


def inside_bin(points,tol=.001):
    lower=np.array([-.204,-.779,.557])-tol;upper=np.array([.204,-.491,.649])+tol
    return bool(np.all(points>=lower) and np.all(points<=upper))


def run(case,video=None):
    m,d,t,q,center=setup(case);pid=m.geom('released_part_geom').id;bid=m.body('released_part').id
    adr=m.joint('part_free').dofadr[0];sid=m.site('blade_tip').id;end=t[-1]+2.;dt=m.opt.timestep
    force=np.zeros(6);unintended={};frictions={};peak_tool_force=0.;contact_steps=0;maxerror=0.;saturated=np.zeros(5,int);tail=[];records=[]
    names=[m.geom(i).name or f'geom_{i}' for i in range(m.ngeom)];weld=m.body_weldid[m.geom_bodyid]
    renderer=writer=None;frame_at=0.
    if video:
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw,ImageFont
        renderer=mujoco.Renderer(m,720,1280);writer=imageio.get_writer(str(video),fps=15,codec='libx264',quality=8)
        font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf',21)
    try:
        for step in range(round(end/dt)):
            d.ctrl[:5]=[np.interp(d.time,t,q[:,j]) for j in range(5)];mujoco.mj_step(m,d)
            if not np.isfinite(d.qpos).all():raise RuntimeError(f'Nonfinite simulation: {case}')
            error=float(np.max(abs(d.qpos[:5]-d.ctrl[:5])));maxerror=max(maxerror,error)
            saturated+=abs(d.actuator_force[:5])>=.980664
            for ci,c in enumerate(d.contact):
                a,b=int(c.geom1),int(c.geom2)
                if pid in (a,b):
                    other=b if a==pid else a;name=names[other]
                    frictions.setdefault(name,set()).add(round(float(c.friction[0]),6))
                    if name=='spatula':
                        mujoco.mj_contactForce(m,d,ci,force);peak_tool_force=max(peak_tool_force,float(np.linalg.norm(force[:3])));contact_steps+=1
                if weld[a]==0:a,b=b,a
                if weld[a]>0 and a!=pid and weld[b]==0:
                    mujoco.mj_contactForce(m,d,ci,force)
                    if force[0]>.01:
                        key=f'{names[a]} / {names[b]}';unintended[key]=max(unintended.get(key,0.),float(force[0]))
            if d.time>end-.5:
                corners=CORNERS@d.xmat[bid].reshape(3,3).T+d.xpos[bid]
                tail.append(inside_bin(corners) and np.linalg.norm(d.qvel[adr:adr+3])<.01 and np.linalg.norm(d.qvel[adr+3:adr+6])<.1)
            if step%50==0:records.append({'t':float(d.time),'part_xyz':d.xpos[bid].tolist(),'tip_xyz':d.site_xpos[sid].tolist(),'joint_error_deg':float(np.rad2deg(error))})
            if renderer and d.time>=frame_at:
                frame_at+=1/15;renderer.update_scene(d,camera='layout');im=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,1280,65),fill='white')
                draw.text((16,5),f'{case["id"]} / {case["mass_kg"]*1000:g}g / 接触面の摩擦 {case["friction"]:g} / 経路は全条件で固定',font=font,fill='black')
                draw.text((16,35),f'{d.time:.1f}s / 初期位置ずれ：押す方向{case["along_offset_m"]*1000:+g}mm、横{case["cross_offset_m"]*1000:+g}mm',font=font,fill='black')
                writer.append_data(np.array(im))
    finally:
        if writer:writer.close()
        if renderer:renderer.close()
    # Assert real contact coefficients, not just configured geom attributes.
    for name,values in frictions.items():
        if name=='a1_05' or name.startswith('chute_'):
            if values!={round(case['friction'],6)}:raise AssertionError((name,values,case['friction']))
    collected=bool(tail and all(tail));tracking=bool(np.rad2deg(maxerror)<3);clearance=not unintended
    reasons=[]
    if not collected:reasons.append('not_stably_in_bin')
    if not tracking:reasons.append('joint_tracking_error')
    if not clearance:reasons.append('robot_surface_contact')
    return {**case,'initial_part_xyz':center.tolist(),'final_part_xyz':d.xpos[bid].tolist(),'final_part_quat':d.xquat[bid].tolist(),
            'final_robot_q_rad':d.qpos[:6].tolist(),'collected':collected,'tracking_pass':tracking,'contact_pass':clearance,
            'success':collected and tracking and clearance,'failure_reasons':reasons,'max_joint_error_deg':float(np.rad2deg(maxerror)),
            'torque_saturation_seconds':(saturated*dt).tolist(),'tool_part_contact_points_over_steps':contact_steps,
            'peak_tool_part_force_N':peak_tool_force,'robot_static_surface_contact_peak_N':unintended,
            'observed_part_contact_friction':{k:sorted(v) for k,v in frictions.items()},'records':records}


def main(videos=False):
    protocol=json.loads((INPUT/'protocol.json').read_text());assert sha(INPUT/'scene.xml')==protocol['scene_sha256'];assert sha(INPUT/'trajectory.json')==protocol['trajectory_sha256']
    if videos:
        result=json.loads((OUT/'motion_results.json').read_text());selected=result['video_case_ids']
        for name in selected:
            original=next(r for r in result['results'] if r['id']==name);case={k:original[k] for k in ['id','mass_kg','friction','along_offset_m','cross_offset_m']}
            reproduced=run(case,OUT/f'{name}.mp4')
            assert reproduced['success']==original['success'];np.testing.assert_allclose(reproduced['final_part_xyz'],original['final_part_xyz'],atol=1e-9)
            print('video',name,reproduced['success'],flush=True)
        return
    rows=[];start=time.monotonic()
    for case in cases(protocol):
        row=run(case);rows.append(row)
        print(case['id'],row['success'],row['failure_reasons'],f'{time.monotonic()-start:.1f}s',flush=True)
        with (OUT/'motion_trials.jsonl').open('w' if len(rows)==1 else 'a') as f:f.write(json.dumps(row)+'\n')
    summaries={}
    for key in ['mass_kg','friction','along_offset_m','cross_offset_m']:
        summaries[key]=[{key:v,'count':len(group),'success':sum(r['success'] for r in group),'collected':sum(r['collected'] for r in group)} for v in sorted({r[key] for r in rows}) for group in [[r for r in rows if r[key]==v]]]
    chosen=[next(r['id'] for r in rows if r['mass_kg']==.008 and r['friction']==.3 and r['along_offset_m']==0 and r['cross_offset_m']==0)]
    first_failed=next((r['id'] for r in rows if not r['success']),None)
    if first_failed and first_failed not in chosen:chosen.append(first_failed)
    result={'scope':'fixed, predeclared synthetic parameter grid; no probability-of-real-world-success claim','protocol_sha256':sha(INPUT/'protocol.json'),
            'script_sha256':sha(__file__),'count':len(rows),'success':sum(r['success'] for r in rows),'collected':sum(r['collected'] for r in rows),
            'contact_failures':sum(not r['contact_pass'] for r in rows),'tracking_failures':sum(not r['tracking_pass'] for r in rows),
            'by_factor':summaries,'video_case_ids':chosen,'results':rows}
    (OUT/'motion_results.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('results','by_factor')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--videos',action='store_true');args=parser.parse_args();main(args.videos)
