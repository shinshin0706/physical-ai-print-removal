"""Webcam-only planar localization using a rigid four-marker bed fixture.

Input: RGB frame, calibrated K/distortion, measured marker corners (metres).
Output: image-to-bed-plane homography or explicit rejection. No simulator
object position is used by estimate_plane(). Markers are not learned AI.
Design assumptions: 30-mm DICT_4X4_50 markers on four 40-mm carriers, at
X=+/-155 and Y=+/-105 mm relative to bed origin, coplanar with the plate.
The carrier attachment/thermal compatibility remains a mechanical design task.
Intrinsics generated from simulated FOV are NOT a real C920 calibration.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from a1_workcell import PLATE_Z, BED_Y, vec

DICTIONARY=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_SIZE=.030
CENTRES={10:(-.155,-.105),11:(.155,-.105),12:(.155,.105),13:(-.155,.105)}


def marker_corners():
    half=MARKER_SIZE/2
    # Marker image top corresponds to positive bed Y.
    return {i:np.array([[x-half,y+half],[x+half,y+half],[x+half,y-half],[x-half,y-half]],np.float32) for i,(x,y) in CENTRES.items()}


def add_markers(root, hidden_ids=(), *, bed_y=BED_Y, plate_z=PLATE_Z):
    world=root.find('worldbody')
    for i,(x,y) in CENTRES.items():
        # Small carrier with white quiet zone; mounting arms are design proxies.
        ET.SubElement(world,'geom',name=f'marker_carrier_{i}',type='box',pos=vec([x,y+bed_y,plate_z-.001]),size='.020 .020 .001',rgba='1 1 1 1',group='1',contype='0',conaffinity='0')
        if i in hidden_ids:continue
        code=cv2.aruco.generateImageMarker(DICTIONARY,i,6)
        for row,col in zip(*np.where(code==0)):
            ET.SubElement(world,'geom',name=f'marker_{i}_{row}_{col}',type='box',pos=vec([x+(col-2.5)*.005,y+bed_y+(2.5-row)*.005,plate_z+.00003]),size='.0025 .0025 .00003',rgba='.005 .005 .005 1',group='1',contype='0',conaffinity='0')


def estimate_plane(rgb,K,distortion):
    """Requires >=3 known markers, <=1.5px RMS and >=5000px board hull area.

    Inputs must already share the capture resolution. Rejects missing markers
    and inconsistent corners; callers must never reuse a stale transform.
    """
    undistorted=cv2.undistort(rgb,K,distortion)
    detector=cv2.aruco.ArucoDetector(DICTIONARY,cv2.aruco.DetectorParameters())
    corners,ids,_=detector.detectMarkers(cv2.cvtColor(undistorted,cv2.COLOR_RGB2GRAY))
    known=marker_corners();pairs=[]
    if ids is not None:
        pairs=[(int(i),c.reshape(4,2)) for i,c in zip(ids.ravel(),corners) if int(i) in known]
    if len(pairs)<3:return {'ok':False,'reason':'fewer_than_three_markers','ids':[i for i,_ in pairs]}
    obj=np.concatenate([known[i] for i,_ in pairs]);img=np.concatenate([c for _,c in pairs])
    H,_=cv2.findHomography(obj,img,method=0)
    if H is None or not np.isfinite(H).all():return {'ok':False,'reason':'homography_failed'}
    projected=cv2.perspectiveTransform(obj[None],H)[0]
    rms=float(np.sqrt(np.mean(np.sum((projected-img)**2,axis=1))))
    area=float(abs(cv2.contourArea(cv2.convexHull(img))))
    if rms>1.5 or area<5000:return {'ok':False,'reason':'inconsistent_or_small_board','rms_px':rms}
    return {'ok':True,'ids':[i for i,_ in pairs],'rms_px':rms,'H_plane_to_pixel':H.tolist(),
            'H_pixel_to_plane':np.linalg.inv(H).tolist(),'undistorted_rgb':undistorted}


def rectify(rgb,K,distortion,result,size=512):
    if not result['ok']:raise ValueError('Marker localization rejected')
    # Only 256x256-mm printable region; image origin top left.
    plane_to_rect=np.array([[size/.256,0,size/2],[0,-size/.256,size/2],[0,0,1.]])
    return cv2.warpPerspective(result['undistorted_rgb'],plane_to_rect@np.array(result['H_pixel_to_plane']),(size,size))


def write_markers(path):
    """A4 PDF: print at 100%, check 30-mm edges with a ruler."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    c=canvas.Canvas(str(path),pagesize=(210*mm,297*mm))
    c.setFont('Helvetica',13);c.drawString(20*mm,277*mm,'A1 bed fixture / ArUco DICT_4X4_50 / 30 mm')
    c.setFont('Helvetica',10);c.drawString(20*mm,266*mm,'Print at 100%. Measure black square edges before use.')
    for k,i in enumerate(CENTRES):
        x=(30+(k%2)*100)*mm;y=(180-(k//2)*80)*mm
        c.drawImage(ImageReader(Image.fromarray(cv2.aruco.generateImageMarker(DICTIONARY,i,600))),x,y,30*mm,30*mm)
        c.drawString(x,y-7*mm,f'ID {i} / 30 x 30 mm')
    c.showPage();c.save()
