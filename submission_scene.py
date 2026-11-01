"""Redistributable primitive workcell; does not read or convert the private A1 CAD.
Published build area: 256mm square. Every internal box/parked height is an assumed
engineering envelope, not a dimensional replica. Exact A1 collision claims are
excluded. Own furniture and printed tool, Apache-2.0 SO-101. Units metres.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from a1_workcell import robot_xml,vec,ROOT
from station_geometry import components
from camera_geometry import add_markers
from removal import add_camera
from audit_workcell import ik_flat
P=ROOT/'data/submission';O=ROOT/'outputs/submission';PLATE_Z=.824;BED_Y=-.150;FRONT=BED_Y-.128

def build():
    root,_,home=robot_xml('so101',(-.30,.02,.740));world=root.find('worldbody')
    def box(name,pos,size,rgba='.76 .77 .78 1',**extra):
        return ET.SubElement(world,'geom',name=name,type='box',pos=vec(pos),size=vec(np.array(size)/2),rgba=rgba,**extra)
    for c in components():
        if c['name'].startswith('a1_') or c['name']=='build_plate':continue
        box(c['name'],c['center_m'],c['size_m'],vec(c['rgba']))
    # Independent assumed envelopes, not meshes or dimensions extracted from CAD.
    for name,pos,size,color in [
      ('printer_base',[0,.035,.772],[.32,.34,.064],'.82 .83 .84 1'),
      ('printer_rail',[0,0,.803],[.075,.40,.018],'.25 .26 .28 1'),
      ('printer_carriage',[0,BED_Y,.813],[.24,.24,.018],'.30 .31 .33 1'),
      ('a1_05',[0,BED_Y,PLATE_Z-.001],[.256,.256,.002],'.55 .43 .19 1'),
      ('printer_left',[-.205,.055,.98],[.035,.055,.48],'.76 .77 .78 1'),
      ('printer_right',[.205,.055,.98],[.035,.055,.48],'.76 .77 .78 1'),
      ('printer_top',[0,.055,1.205],[.445,.055,.03],'.76 .77 .78 1'),
      ('printer_beam',[0,.025,1.08],[.41,.04,.04],'.76 .77 .78 1'),
      ('printer_head',[.15,-.01,1.04],[.06,.08,.08],'.20 .21 .23 1')]:box(name,pos,size,color)
    points=[np.array([0,FRONT-.001,PLATE_Z-.007]),np.array([0,-.48,.770]),np.array([0,-.56,.665])]
    for j,(a,b) in enumerate(zip(points,points[1:])):
        length=np.linalg.norm(b-a);pitch=np.arctan2(a[2]-b[2],a[1]-b[1])
        for name,x,width,height in [('floor',0,.20,.002),('left',-.101,.002,.025),('right',.101,.002,.025)]:
            box(f'chute_{j}_{name}',(a+b)/2+[x,0,0 if name=='floor' else .012],[width,length,height],'.84 .85 .86 1',euler=vec([pitch,0,0]),friction='.2 .005 .0001')
    for name,pos,size in [('camera_post',[-.65,-.40,1.12],[.03,.03,.76]),('camera_boom',[-.325,-.40,1.50],[.68,.03,.03]),('camera_bracket',[0,-.475,1.50],[.03,.15,.03]),('camera_body',[0,-.55,1.47],[.095,.03,.03])]:box(name,pos,size,'.17 .18 .20 1')
    add_markers(root,bed_y=BED_Y,plate_z=PLATE_Z)
    for x in [-.143,.143]:box(f'marker_rail_{x}',[x,BED_Y,PLATE_Z-.006],[.036,.25,.006],'.83 .83 .83 1',contype='0',conaffinity='0')
    ET.SubElement(world,'geom',name='ground',type='plane',size='3 3 .1',rgba='.94 .94 .94 1')
    ET.SubElement(world,'light',pos='0 -1 2.8',dir='0 0 -1',diffuse='.8 .8 .8')
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='1280',offheight='720');ET.SubElement(visual,'headlight',ambient='.5 .5 .5',diffuse='.6 .6 .6',specular='.1 .1 .1')
    add_camera(world,'layout',[1.05,-1.65,1.65],[-.10,-.04,.88],43)
    fov=np.rad2deg(2*np.arctan(np.tan(np.deg2rad(78)/2)/np.sqrt(1+(16/9)**2)))
    add_camera(world,'webcam',[0,-.55,1.47],[0,BED_Y,PLATE_Z],fov)
    return root,home

def main():
    P.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
    root,home=build();(P/'scene.xml').write_text(ET.tostring(root,encoding='unicode'))
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m);d.qpos[5]=.2
    end=np.array([0.,FRONT-.018,PLATE_Z+.001]);direction=end-np.array([-.30+.0388353,.02,end[2]]);direction/=np.linalg.norm(direction);start=end-.110*direction
    def solve(target,q,tries=1):
        _,q,pe,ae=ik_flat(m,d,5,q,target,tries=tries,pitch=35)
        if pe>.002 or ae>np.deg2rad(3):raise RuntimeError(('IK failed',target,pe,ae))
        return q
    approach=solve(start+[0,0,.08],home,12);park=approach.copy();park[0]=1.6;rows=[];clock=1.
    def append(q,phase,t):
        d.qpos[:5]=q;mujoco.mj_forward(m,d);rows.append({'q_rad':q.tolist(),'target_m':d.site_xpos[m.site('blade_tip').id].tolist(),'phase':phase,'time_s':float(t)})
    def joint(a,b,phase,duration):
        nonlocal clock
        for u in np.linspace(0,1,31):append(a+(b-a)*u,phase,clock+duration*u)
        clock+=duration
    joint(park,approach,'待機位置から移動',2.5);q=approach.copy()
    for phase,targets,duration in [('接近',[start+[0,0,.08*(1-u)] for u in np.linspace(0,1,31)],4.),('押す経路',[start+(end-start)*u for u in np.linspace(0,1,61)],5.5),('退避',[end+[0,0,.08*u] for u in np.linspace(0,1,31)],4.)]:
        for target,t in zip(targets,np.linspace(clock,clock+duration,len(targets))):q=solve(target,q);append(q,phase,t)
        clock+=duration
    joint(q,approach,'戻る',6.);joint(approach,park,'検査位置へ退避',2.5)
    (P/'trajectory.json').write_text(json.dumps({'scope':'independent primitive scene, no A1 CAD import','samples':rows},indent=2)+'\n')
    d.qpos[:5]=rows[50]['q_rad'];mujoco.mj_forward(m,d)
    with mujoco.Renderer(m,720,1280) as renderer:
        renderer.update_scene(d,camera='layout');Image.fromarray(renderer.render().copy()).save(O/'layout.png')
    print('public scene and path',len(rows))
if __name__=='__main__':main()
