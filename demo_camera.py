"""Render A1 CAD with physical marker geometry; evaluate detection from pixels.

Run: nice -n 15 .venv/bin/python demo_camera.py
Calibration tests are synthetic. Camera pose is used only to render and score,
never by the ArUco estimator. Actual hardware requires intrinsic calibration
and measured marker-to-bed / marker-to-robot transforms, not these numbers.
"""
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import mujoco
from PIL import Image,ImageDraw,ImageFont
from a1_workcell import build,ROOT,PLATE_Z,BED_Y,vec
from camera_geometry import add_markers,estimate_plane,rectify,write_markers
from removal import add_camera


def intrinsics(m,width,height):
    f=height/(2*np.tan(np.deg2rad(m.cam_fovy[m.camera('webcam').id])/2))
    return np.array([[f,0,width/2],[0,f,height/2],[0,0,1.]])


def project_points(m,d,points,K):
    cid=m.camera('webcam').id
    local=(points-d.cam_xpos[cid])@d.cam_xmat[cid].reshape(3,3)
    cvlocal=local*np.array([1,-1,-1])
    pix=cvlocal@K.T
    return pix[:,:2]/pix[:,2,None]


def main():
    out=ROOT/'outputs';rows=[]
    settings=[('nominal',[0,-.55,1.47],()),('camera_left',[-.10,-.55,1.47],()),
              ('camera_lower',[0,-.55,1.37],()),('one_marker_hidden',[0,-.55,1.47],(10,)),
              ('two_markers_hidden',[0,-.55,1.47],(10,11))]
    for name,pos,hidden in settings:
        root,_,_,n,q,_=build('so101',[-.30,-.15,.74])
        world=root.find('worldbody');world.remove(world.find("camera[@name='webcam']"))
        vfov=np.rad2deg(2*np.arctan(np.tan(np.deg2rad(78)/2)/np.sqrt(1+(16/9)**2)))
        add_camera(world,'webcam',pos,[0,-.15,PLATE_Z],vfov)
        add_markers(root,hidden)
        # Demonstration object, not fed to marker localization.
        ET.SubElement(world,'geom',name='sample_print',type='box',pos=vec([0,BED_Y,PLATE_Z+.015]),size='.018 .018 .015',rgba='.9 .3 .05 1')
        m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
        d.qpos[:n]=q;d.qpos[0]=-1.4;mujoco.mj_forward(m,d)
        K=intrinsics(m,1920,1080)
        with mujoco.Renderer(m,1080,1920) as renderer:
            renderer.update_scene(d,camera='webcam');rgb=renderer.render().copy()
        result=estimate_plane(rgb,K,np.zeros(5))
        row={k:v for k,v in result.items() if k!='undistorted_rgb'}
        row.update(case=name,expected_accept=len(hidden)<=1,simulated_camera_position_m=pos)
        if result['ok']:
            points=np.array([[x,y,PLATE_Z] for x in [-.10,0,.10] for y in [BED_Y-.10,BED_Y,BED_Y+.10]])
            pixels=project_points(m,d,points,K)
            recovered=cv2.perspectiveTransform(pixels.astype(np.float32)[None],np.array(result['H_pixel_to_plane']))[0]
            truth=points[:,:2]-[0,BED_Y]
            row['max_plane_error_mm']=float(np.linalg.norm(recovered-truth,axis=1).max()*1000)
            warped=rectify(rgb,K,np.zeros(5),result)
            if name=='nominal':Image.fromarray(warped).save(out/'webcam_rectified.png')
        if name=='nominal':
            im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
            font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf',26)
            draw.rectangle((0,0,1920,80),fill='white')
            draw.text((25,12),'固定Webカメラの模擬画像：A1 CAD＋30mm ArUcoマーカー',font=font,fill='black')
            draw.text((25,43),'1080p / 実機のカメラ校正とマーカー固定具は未検証',font=font,fill='black')
            im.save(out/'webcam_view.png')
            (out/'webcam_calibration_sim.json').write_text(json.dumps({'synthetic_only':True,'image_size':[1920,1080],'K':K.tolist(),'distortion':[0]*5,'source':'78deg diagonal pinhole approximation, not measured C920 intrinsics'},indent=2))
        rows.append(row);print(name,row['ok'],row.get('max_plane_error_mm'),flush=True)
    write_markers(ROOT/'cad/aruco_markers.pdf')
    (out/'webcam_validation.json').write_text(json.dumps({'scope':'synthetic pixel detection and planar mapping only; no real camera accuracy claim','opencv':cv2.__version__,'results':rows},indent=2)+'\n')


if __name__=='__main__':main()
