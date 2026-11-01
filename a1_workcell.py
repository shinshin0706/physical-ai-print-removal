"""A1 CAD workcell for offline geometry and camera experiments (metres).

External sources and model revisions live in local_assets/bambu_a1 and vendor.
CAD solids are visual meshes and per-solid convex collision hulls. Holes and
concavities may be conservatively filled; this is not a real-printer certificate.
AI design values: 740-mm bench, parked bed Y=-150.5 mm from CAD default,
raised gantry +183.1 mm, right-parked head +156.9 mm, fixed webcam position,
and rigid 24.1-g spatula grasp. The grasp is an idealization, not a mount test.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
import mujoco
from removal import add_camera
from station_geometry import components

ROOT=Path(__file__).parent
CAD=ROOT/'local_assets/bambu_a1'
TABLE=.740
PLATE_Z=TABLE+.0838
BED_Y=-.1505
FRONT=BED_Y-.1395


def vec(a):
    return ' '.join(str(float(x)) for x in a)


def robot_xml(name, base, yaw=0):
    if name=='so101':
        path=ROOT/'vendor/robotstudio_so101/so101.xml'
        root=ET.parse(path).getroot()
        # User-selected white printed parts; preserve original vendor assets
        # and the dark motor housings. Appearance only, no physics change.
        for material in root.findall('asset/material'):
            if material.get('rgba')=='1 0.82 0.12 1':
                material.set('rgba','.90 .91 .92 1')
        robot=root.find("worldbody/body[@name='base']")
        parent=robot.find(".//body[@name='gripper']")
        site=parent.find("site[@name='gripperframe']")
        pos=site.get('pos');quat=site.get('quat')
        n=5; home=np.array([0.,-.5,1.,.5,0.])
    else:
        path=ROOT/'vendor/enactic_openarm/v2/openarm_bimanual.xml'
        root=ET.parse(path).getroot()
        world=root.find('worldbody')
        world.remove(world.find("body[@name='openarm_left_base_link']"))
        for section in ('actuator','equality'):
            for e in list(root.find(section)):
                if 'left' in e.get('name',''):root.find(section).remove(e)
        robot=world.find("body[@name='openarm_right_base_link']")
        parent=robot.find(".//body[@name='openarm_right_ee_base_link']")
        pos='-.014 0 -.114';quat='.70710678 0 .70710678 0'
        n=7;home=np.array([.5,.5,0.,1.2,0.,0.,0.])
    root.find('compiler').set('meshdir',str(path.parent/'assets'))
    # Absolute paths permit saving the scene outside the vendor directories.
    for mesh in root.findall('asset/mesh'):
        mesh.set('file',str(path.parent/'assets'/mesh.get('file')))
    robot.set('pos',vec(base));robot.set('quat',vec([np.cos(yaw/2),0,0,np.sin(yaw/2)]))
    if root.find('keyframe') is not None:root.remove(root.find('keyframe'))
    tool=ET.SubElement(parent,'body',name='tool_frame',pos=pos,quat=quat)
    ET.SubElement(root.find('asset'),'mesh',name='printable_spatula',file=str(ROOT/'cad/spatula.stl'),scale='.001 .001 .001')
    ET.SubElement(tool,'geom',name='spatula',type='mesh',mesh='printable_spatula',pos='-.03 0 -.003',mass='.0241',rgba='.12 .32 .65 1',group='2')
    ET.SubElement(tool,'site',name='blade_tip',pos='.09 0 -.003',size='.002',rgba='1 .3 0 1',group='4')
    return root,n,home


def add_printer(root):
    parts=json.loads((CAD/'geometry.json').read_text())['states']['Default']['solids']
    world=root.find('worldbody');asset=root.find('asset')
    for i,p in enumerate(parts):
        group=p['group'].split('/')[-1]
        offset=np.array([0.,0.,TABLE])
        if group.endswith(('heatbed','buildplate')):offset[1]+=BED_Y
        if group.endswith(('X axis','Toolhead')):offset[2]+=.1831
        if group.endswith('Toolhead'):offset[0]+=.1569
        color='.76 .77 .78 1'
        if group.endswith('buildplate'):color='.55 .43 .19 1'
        if group.endswith(('display','Toolhead')):color='.18 .19 .21 1'
        name=f'a1_{i:02d}'
        ET.SubElement(asset,'mesh',name=name,file=str(CAD/'meshes'/p['mesh']))
        ET.SubElement(world,'geom',name=name,type='mesh',mesh=name,pos=vec(offset),rgba=color,group='2')
    return parts


def build(name='so101',base=(-.30,-.15,.74),yaw=0, furniture=True):
    root,n,q=robot_xml(name,base,yaw)
    parts=add_printer(root);world=root.find('worldbody')
    for c in components():
        if c['name'].startswith('a1_') or c['name']=='build_plate':continue
        if not furniture and not c['name']=='bench_top':continue
        ET.SubElement(world,'geom',name=c['name'],type='box',pos=vec(c['center_m']),size=vec(np.array(c['size_m'])/2),rgba=vec(c['rgba']))
    # SO bolted to a rigid riser, OpenArm bolted to a 40-mm extrusion post.
    h=base[2]-TABLE
    if h>.003:
        width=.10 if name=='so101' else .04
        ET.SubElement(world,'geom',name='robot_mount_post',type='box',pos=vec([base[0],base[1],TABLE+h/2]),size=vec([width/2,width/2,h/2]),rgba='.25 .26 .27 1')
    ET.SubElement(world,'geom',name='ground',type='plane',size='3 3 .1',rgba='.94 .94 .94 1')
    ET.SubElement(world,'light',pos='0 -1 2.8',dir='0 0 -1',diffuse='.8 .8 .8')
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    for e in list(visual.findall('global')):visual.remove(e)
    ET.SubElement(visual,'global',offwidth='1920',offheight='1080')
    ET.SubElement(visual,'headlight',ambient='.5 .5 .5',diffuse='.6 .6 .6',specular='.1 .1 .1')
    ET.SubElement(root.find('asset'),'texture',name='background',type='skybox',builtin='flat',rgb1='.94 .94 .94',rgb2='.94 .94 .94',width='512',height='3072')
    add_camera(world,'layout',[1.05,-1.65,1.65],[-.10,-.04,.88],43)
    add_camera(world,'side',[-1.05,-1.4,1.4],[-.05,-.04,.89],43)
    # 78deg diagonal at 16:9 -> 43.2deg vertical (pinhole approximation).
    vfov=np.rad2deg(2*np.arctan(np.tan(np.deg2rad(78)/2)/np.sqrt(1+(16/9)**2)))
    add_camera(world,'webcam',[0,-.55,1.47],[0,-.15,PLATE_Z],vfov)
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    d=mujoco.MjData(m);d.qpos[:n]=q
    if name=='so101':d.qpos[n]=.20
    else:d.qpos[n:n+2]=-.15
    mujoco.mj_forward(m,d)
    return root,m,d,n,q,parts


def collision_pairs(m,d,clearance=.003,include_fixtures=False):
    """Explicit robot-vs-printer/table checks, including disabled visual meshes.

    The tool may contact the plate, but no other printer intersection is ignored.
    Grasped tool-vs-finger contact and adjacent-link self-contact are not audited.
    """
    robot=[i for i in range(m.ngeom) if m.body_weldid[m.geom_bodyid[i]]>0 and m.geom_group[i] in (2,3,4)]
    obstacles=[i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('a1_') or m.geom(i).name=='bench_top']
    if include_fixtures:
        obstacles += [i for i in range(m.ngeom) if m.body_weldid[m.geom_bodyid[i]]==0 and (m.geom(i).name or '').startswith(('chute_', 'camera_', 'marker_rail_', 'marker_carrier_', 'bin_', 'shelf_'))]
    contacts=[];ray=np.zeros(6)
    for a in robot:
        for b in obstacles:
            # Near-contact is expected only for the blade and plate/carriage.
            # Still reject penetration; don't hide an intersecting tool.
            threshold=0. if m.geom(a).name=='spatula' and m.geom(b).name in ('a1_04','a1_05') else clearance
            dist=mujoco.mj_geomDistance(m,d,a,b,clearance,ray)
            if dist<threshold:
                contacts.append({'robot':m.geom(a).name or f'geom_{a}','obstacle':m.geom(b).name,'distance_mm':float(dist*1000)})
    return contacts
